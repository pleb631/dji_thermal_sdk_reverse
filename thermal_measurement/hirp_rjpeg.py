"""Decode the HIRP input blocks used by the MicroTA measurement path."""

from array import array
from dataclasses import dataclass
from struct import unpack_from
import sys

if __package__:
    from .hirp_fixed_point import (
        FrameTemperatureScale, GrayCalibration, ObjectTemperatureCorrection, _truncate_division,
        atmospheric_temperature_trim, atmospheric_transmission_q14, blackbody_gray,
        housing_compensation, humidity_transmission_q14, prepare_inverse_response,
        response_gray_offset,
    )
    from .iirp_rjpeg import _jpeg_segments
else:
    from hirp_fixed_point import (
        FrameTemperatureScale, GrayCalibration, ObjectTemperatureCorrection, _truncate_division,
        atmospheric_temperature_trim, atmospheric_transmission_q14, blackbody_gray,
        housing_compensation, humidity_transmission_q14, prepare_inverse_response,
        response_gray_offset,
    )
    from iirp_rjpeg import _jpeg_segments


@dataclass(frozen=True)
class HirpFrame:
    width: int
    height: int
    raw_values: array
    frame_header: bytes
    calibration: bytes

    def gray_response(self, sensor_mode):
        """Prepare the gray response from the frame's recorded sensor state."""
        sensor_temperature, calibration_group = unpack_from("<hB", self.frame_header, 0x78)
        temperature_range = self.frame_header[0x7C]
        inputs = self.response_inputs(sensor_mode, temperature_range, calibration_group)
        return inputs.gray_response(sensor_temperature)

    def blackbody_response(self, sensor_mode):
        """Read and adjust the response coefficients used for blackbody gray."""
        group = self.frame_header[0x7A]
        row = self.frame_header[0x7C]
        inputs = self.response_inputs(sensor_mode, row, group)
        table = memoryview(self.calibration)[0x1F0:]
        base_offset = 0x654 + 68 * (5 * row + group)
        adjustment_offset = 0xCF8 + 36 * (25 * sensor_mode + 5 * row + group)
        amplitude, exponent, floor, slope = unpack_from("<4i", table, base_offset)
        floor_offset, gain = unpack_from("<2i", table, adjustment_offset)
        table_version, = unpack_from("<I", table)
        return (
            (amplitude * gain >> 20, exponent, (floor * gain + (floor_offset << 12)) >> 20, slope),
            273150 + 20 * inputs.reference_temperature,
            table_version > 0x402,
        )

    def inverse_response(self, sensor_mode, gain):
        """Return the blackbody reference gray and centered inverse coefficients."""
        coefficients, temperature, use_calibrated_slope = self.blackbody_response(sensor_mode)
        reference_gray = blackbody_gray(coefficients, temperature, use_calibrated_slope)
        reference_gain, = unpack_from("<I", self.calibration, 0x1F0 + 0x2B18)
        inverse = prepare_inverse_response(coefficients, reference_gray, gain, reference_gain)
        return reference_gray, inverse

    def housing_adjusted_gray_offset(self, sensor_mode, reference_gray, inverse_coefficients, dj20=True):
        """Calculate the gray offset through the housing compensation stage."""
        sensor_temperature, group = unpack_from("<hB", self.frame_header, 0x78)
        row = self.frame_header[0x7C]
        inputs = self.response_inputs(sensor_mode, row, group)
        table = memoryview(self.calibration)[0x1F0:]
        index = 5 * row + group
        reference_housing, = unpack_from("<i", table, 0x460 + 4 * (25 * sensor_mode + index))
        gain, breakpoint, low_slope, high_slope, gain_slope = unpack_from(
            "<5i", table, 0x1E8C + 32 * index
        )
        if not dj20 and table[0x2F8A] == 1 and group == 2:
            override = unpack_from("<5i", table, 0x2FB4 + 20 * row)
            if any(override):
                gain, breakpoint, low_slope, high_slope, gain_slope = override
            elif row == 0:
                gain, breakpoint, low_slope, high_slope, gain_slope = (16384, 81920, 196608, 196608, 3)
            elif row == 1:
                gain, breakpoint, low_slope, high_slope, gain_slope = (17203, 81920, 24576, 24576, -49)
        table_version, = unpack_from("<I", table)
        if table_version > 0x404:
            raise ValueError("HIRP housing compensation for this calibration version is unsupported")
        housing_temperature, = unpack_from("<i", self.frame_header, 0x70)
        housing_delta = sensor_temperature - _truncate_division(housing_temperature, 20)
        adjustment = housing_compensation(
            housing_delta, inputs.reference_temperature - reference_housing,
            (breakpoint, low_slope, high_slope),
        )
        sensor_gray = blackbody_gray(
            inverse_coefficients, 273150 + 20 * sensor_temperature, table_version > 0x402
        )
        base_offset = response_gray_offset(
            reference_gray, sensor_gray, inputs.gray_response(sensor_temperature),
            sensor_temperature - inputs.reference_temperature, gain, gain_slope,
        )
        return base_offset - adjustment

    def environment_correction(self, sensor_mode, distance_cm, emissivity, dj20):
        """Prepare the object correction and optional atmospheric trim."""
        reflected, humidity_milli = unpack_from("<ii", self.frame_header, 0x70)
        background = _truncate_division(reflected, 20)
        humidity = _truncate_division(humidity_milli, 1000)
        index = 5 * sensor_mode + self.frame_header[0x7C]
        table = memoryview(self.calibration)[0x1F0:]
        atmosphere = unpack_from("<3i", table, 0x21AC + 32 * index)
        trim = unpack_from("<4i", table, 0x2DDC + 16 * index)
        if not dj20 and table[0x2F8A] == 1:
            atmosphere, trim = self._process_atmosphere_row(table, sensor_mode, index, atmosphere, trim)
        trim_gain, trim_offset = None, 0
        if trim[1]:
            _, _, trim_gain, trim_offset = atmospheric_temperature_trim(
                background, distance_cm - atmosphere[2], trim, dj20
            )
            transmission = 1 << 14
        else:
            transmission = atmospheric_transmission_q14(
                background, humidity, distance_cm, atmosphere
            )
        if not dj20:
            transmission = (transmission * humidity_transmission_q14(background, humidity)) >> 14
            reflected = _truncate_division(reflected, 100) * 100
        correction = ObjectTemperatureCorrection.prepare(
            background, emissivity, reflected, transmission
        )
        return correction, trim_gain, trim_offset

    def _process_atmosphere_row(self, table, mode, index, atmosphere, trim):
        """Apply the measurement-time atmosphere table overrides."""
        selector = 2 * mode + self.frame_header[0x7C]
        alternate_linear, = unpack_from("<i", table, 0x2F8C + 16 * selector)
        alternate_distance_gain, = unpack_from("<i", table, 0x2FAC + 4 * selector)
        if alternate_linear or alternate_distance_gain:
            if index < 2:
                distance_gain, = unpack_from("<i", table, 0x2FAC + 4 * index)
                atmosphere = (atmosphere[0], distance_gain, atmosphere[2])
                trim = unpack_from("<4i", table, 0x2F8C + 16 * index)
        elif index == 0:
            atmosphere = (atmosphere[0], 260, atmosphere[2])
        elif index == 1:
            trim = (410, -10307, 0, 0)
        return atmosphere, trim

    def response_inputs(self, sensor_mode, temperature_range, calibration_group):
        """Read the MicroTA calibration row selected for a frame."""
        if sensor_mode < 0 or not 0 <= temperature_range < 5 or not 0 <= calibration_group < 5:
            raise ValueError("Invalid HIRP calibration row")
        table = self.calibration[0x1F0:]
        row = temperature_range
        group = calibration_group
        reference_offset = 0x26C + 4 * (25 * sensor_mode + 5 * row + group)
        reference, = unpack_from("<i", table, reference_offset)
        sensor_family, = unpack_from("<I", table, 8)
        polynomial = unpack_from("<4q", table, 0x24D0 + 64 * (5 * row + group))
        return HirpResponseInputs(reference, sensor_family, polynomial)


@dataclass(frozen=True)
class HirpResponseInputs:
    reference_temperature: int
    sensor_family: int
    polynomial: tuple

    def gray_response(self, sensor_temperature):
        """Prepare the linear and quadratic gray-response coefficients."""
        if not 0x50 <= self.sensor_family <= 0x57:
            return 1 << 32, 0
        delta = sensor_temperature - self.reference_temperature
        if abs(delta) <= 750:
            scale = delta * 2
            linear_divisor, quadratic_divisor = 1000000, 1000000000000
        else:
            scale = delta // 10 if delta >= 0 else -(-delta // 10)
            linear_divisor, quadratic_divisor = 50000, 2500000000

        def term(numerator, denominator):
            return numerator // denominator if numerator >= 0 else -(-numerator // denominator)

        linear_slope, linear_curve, quadratic_slope, quadratic_curve = self.polynomial
        linear = (1 << 32) + term(linear_slope * scale, linear_divisor)
        linear += term(linear_curve * scale * scale, quadratic_divisor)
        quadratic = term(quadratic_slope * scale, linear_divisor)
        quadratic += term(quadratic_curve * scale * scale, quadratic_divisor)
        if not 0x28F5C29 <= linear <= 0x500000001:
            raise ValueError("HIRP linear response is outside the calibrated range")
        return linear, 0 if abs(quadratic) <= 511 else quadratic


def decode_hirp(data):
    """Read the thermal plane and calibration blocks from a HIRP R-JPEG."""
    headers = []
    samples = bytearray()
    calibration = bytearray()
    for marker, payload in _jpeg_segments(data):
        if marker == 0xE3:
            samples.extend(payload)
        elif marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffhirp"):
            headers.append(payload)
        elif marker == 0xE5:
            calibration.extend(payload)
    if len(headers) != 1 or len(headers[0]) < 0x7D:
        raise ValueError("Missing or truncated HIRP frame header")
    header = headers[0]
    width, height = unpack_from("<II", header, 40)
    if not width or not height or len(samples) != width * height * 2:
        raise ValueError("HIRP dimensions do not match the thermal payload")
    if len(calibration) < 12 or calibration[:4] != b"HIDR":
        raise ValueError("Missing HIRP calibration header")
    size, = unpack_from("<I", calibration, 8)
    if not 12 <= size <= len(calibration):
        raise ValueError("Truncated HIRP calibration payload")
    words = array("H")
    words.frombytes(samples)
    if sys.byteorder != "little":
        words.byteswap()
    return HirpFrame(width, height, words, header, bytes(calibration))


def measure_hirp(data):
    """Return float32 temperatures for the supported HIRP calibration families."""
    frame = decode_hirp(data)
    table = memoryview(frame.calibration)[0x1F0:]
    if len(table) < 0x2FDC:
        raise ValueError("Truncated HIRP measurement calibration")
    model = bytes(table[0x10:0x30]).split(b"\0", 1)[0]
    if model == b"HM-TM56-9RGUT(D)":
        dj20, distance_cm = True, 100
    elif model == b"HM-TM640-17":
        dj20, distance_cm = False, 500
    else:
        raise ValueError(f"Unsupported HIRP calibration model {model!r}")
    row = frame.frame_header[0x7C]
    if row not in (0, 1):
        raise ValueError(f"Unsupported HIRP temperature range {row}")

    sensor_mode = 0
    reference_gray, inverse = frame.inverse_response(sensor_mode, gain=1000)
    gray_offset = frame.housing_adjusted_gray_offset(sensor_mode, reference_gray, inverse, dj20)
    linear, quadratic = frame.gray_response(sensor_mode)
    minimum, = unpack_from("<i", table, 0x50 + 4 * row)
    maximum, = unpack_from("<i", table, 0x64 + 4 * row)
    correction, trim_gain, trim_offset = frame.environment_correction(
        sensor_mode, distance_cm, emissivity=950, dj20=dj20
    )
    high_temperature_trim = 0 if dj20 else unpack_from("<h", table, 0x2F88)[0]
    calibration = GrayCalibration(
        gray_offset, linear, quadratic, inverse, minimum, maximum,
        correction, trim_gain, trim_offset, high_temperature_trim,
    )
    version, = unpack_from("<I", table)
    scale = FrameTemperatureScale.prepare(calibration, row, version > 0x402)
    temperatures = {gray: scale.temperature(gray, calibration) for gray in set(frame.raw_values)}
    return array("f", (temperatures[gray] for gray in frame.raw_values)), (frame.height, frame.width)
