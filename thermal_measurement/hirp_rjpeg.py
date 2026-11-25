"""Decode the HIRP input blocks used by the MicroTA measurement path."""

from dataclasses import dataclass

import numpy as np

from .hirp_fixed_point import (
    FrameTemperatureScale,
    GrayCalibration,
    ObjectTemperatureCorrection,
    _truncate_division,
    _VectorFallback,
    atmospheric_temperature_trim,
    atmospheric_transmission_q14,
    blackbody_gray,
    housing_compensation,
    humidity_transmission_q14,
    prepare_inverse_response,
    response_gray_offset,
)
from .xts_radiance import ThermalArray


@dataclass(frozen=True)
class HirpResponseInputs:
    reference_temperature: int
    sensor_family: int
    polynomial: tuple

    def gray_response(self, sensor_temperature: int) -> tuple[int, int]:
        if not 0x50 <= self.sensor_family <= 0x57:
            return 1 << 32, 0
        delta = sensor_temperature - self.reference_temperature
        if abs(delta) <= 750:
            scale = delta * 2
            linear_divisor, quadratic_divisor = 1000000, 1000000000000
        else:
            scale = delta // 10 if delta >= 0 else -(-delta // 10)
            linear_divisor, quadratic_divisor = 50000, 2500000000

        def term(num, den):
            return num // den if num >= 0 else -(-num // den)

        linear_slope, linear_curve, quadratic_slope, quadratic_curve = self.polynomial
        linear = (1 << 32) + term(linear_slope * scale, linear_divisor)
        linear += term(linear_curve * scale * scale, quadratic_divisor)
        quadratic = term(quadratic_slope * scale, linear_divisor)
        quadratic += term(quadratic_curve * scale * scale, quadratic_divisor)
        if not 0x28F5C29 <= linear <= 0x500000001:
            raise ValueError("HIRP linear response is outside the calibrated range")
        return linear, 0 if abs(quadratic) <= 511 else quadratic


class HirpHandler:
    """Object-oriented measurement handler for Matrice 3TD / Matrice 4T HIRP frames."""

    _instance = None

    def __init__(self):
        self._lut = np.empty(1 << 16, dtype=np.float32)

    @classmethod
    def get_instance(cls) -> "HirpHandler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def can_handle(segments: list[tuple[int, bytes | memoryview]]) -> bool:
        return any(
            marker == 0xE4 and payload[:8] == b"\xff\xd2\xd1\xffhirp"
            for marker, payload in segments
        )

    def measure(
        self, data: bytes, segments: list[tuple[int, bytes | memoryview]] | None = None
    ) -> tuple[ThermalArray, tuple[int, int]]:
        if segments is None:
            from .iirp_rjpeg import _jpeg_segments

            segments = list(_jpeg_segments(data))

        headers = []
        samples = bytearray()
        calibration = bytearray()

        for marker, payload in segments:
            if marker == 0xE3:
                samples.extend(payload)
            elif marker == 0xE4 and payload[:8] == b"\xff\xd2\xd1\xffhirp":
                headers.append(payload)
            elif marker == 0xE5:
                calibration.extend(payload)

        if len(headers) != 1 or len(headers[0]) < 0x7D:
            raise ValueError("Missing or truncated HIRP frame header")
        header = headers[0]

        width = int(np.frombuffer(header[40:44], dtype="<u4")[0])
        height = int(np.frombuffer(header[44:48], dtype="<u4")[0])
        if not width or not height or len(samples) != width * height * 2:
            raise ValueError("HIRP dimensions do not match the thermal payload")

        if len(calibration) < 12 or calibration[:4] != b"HIDR":
            raise ValueError("Missing HIRP calibration header")
        size = int(np.frombuffer(calibration[8:12], dtype="<u4")[0])
        if not 12 <= size <= len(calibration):
            raise ValueError("Truncated HIRP calibration payload")

        table = memoryview(calibration)[0x1F0:]
        if len(table) < 0x2FDC:
            raise ValueError("Truncated HIRP measurement calibration")

        model = bytes(table[0x10:0x30]).split(b"\0", 1)[0]
        if model == b"HM-TM56-9RGUT(D)":
            dj20, distance_cm = True, 100
        elif model == b"HM-TM640-17":
            dj20, distance_cm = False, 500
        else:
            raise ValueError(f"Unsupported HIRP calibration model {model!r}")

        row = header[0x7C]
        if row not in (0, 1):
            raise ValueError(f"Unsupported HIRP temperature range {row}")

        calib_obj, scale = self._prepare_scale(
            header, calibration, table, row, dj20, distance_cm
        )

        # Map raw codes over the observed span instead of looping per unique value
        raw_pixels = np.frombuffer(samples, dtype="<u2")
        low = int(raw_pixels.min())
        high = int(raw_pixels.max())
        codes = np.arange(low, high + 1, dtype=np.int64)
        lut = self._lut
        try:
            lut[low : high + 1] = scale.temperature_array(codes, calib_obj)
        except _VectorFallback:
            unique_grays = np.flatnonzero(np.bincount(raw_pixels)).tolist()
            for gray in unique_grays:
                lut[gray] = scale.temperature(gray, calib_obj)

        temperatures = lut[raw_pixels]
        return temperatures.astype("<f4", copy=False).view(ThermalArray), (height, width)

    def _prepare_scale(
        self,
        header: bytes,
        calibration: bytes,
        table: memoryview,
        row: int,
        dj20: bool,
        distance_cm: int,
    ):
        sensor_mode = 0
        group = header[0x7A]
        inputs = self._response_inputs(calibration, sensor_mode, row, group)

        base_offset = 0x654 + 68 * (5 * row + group)
        adj_offset = 0xCF8 + 36 * (25 * sensor_mode + 5 * row + group)
        amplitude, exponent, floor, slope = (
            int(x)
            for x in np.frombuffer(table, dtype="<i4", count=4, offset=base_offset)
        )
        floor_offset, gain = (
            int(x)
            for x in np.frombuffer(table, dtype="<i4", count=2, offset=adj_offset)
        )
        table_version = int.from_bytes(table[:4], "little")
        blackbody_coeffs = (
            (
                amplitude * gain >> 20,
                exponent,
                (floor * gain + (floor_offset << 12)) >> 20,
                slope,
            ),
            273150 + 20 * inputs.reference_temperature,
            table_version > 0x402,
        )

        ref_gray = blackbody_gray(
            blackbody_coeffs[0], blackbody_coeffs[1], blackbody_coeffs[2]
        )
        ref_gain = int.from_bytes(
            calibration[0x1F0 + 0x2B18 : 0x1F0 + 0x2B1C], "little"
        )
        inverse = prepare_inverse_response(
            blackbody_coeffs[0], ref_gray, 1000, ref_gain
        )

        sensor_temp = int.from_bytes(header[0x78:0x7A], "little", signed=True)
        calib_grp = header[0x7A]
        idx = 5 * row + calib_grp
        ref_housing = int(
            np.frombuffer(
                table, dtype="<i4", count=1, offset=0x460 + 4 * (25 * sensor_mode + idx)
            )[0]
        )
        gain_val, breakpoint, low_slope, high_slope, gain_slope = (
            int(x)
            for x in np.frombuffer(
                table, dtype="<i4", count=5, offset=0x1E8C + 32 * idx
            )
        )

        if not dj20 and table[0x2F8A] == 1 and calib_grp == 2:
            override = tuple(
                int(x)
                for x in np.frombuffer(
                    table, dtype="<i4", count=5, offset=0x2FB4 + 20 * row
                )
            )
            if any(override):
                gain_val, breakpoint, low_slope, high_slope, gain_slope = override
            elif row == 0:
                gain_val, breakpoint, low_slope, high_slope, gain_slope = (
                    16384,
                    81920,
                    196608,
                    196608,
                    3,
                )
            elif row == 1:
                gain_val, breakpoint, low_slope, high_slope, gain_slope = (
                    17203,
                    81920,
                    24576,
                    24576,
                    -49,
                )

        if table_version > 0x404:
            raise ValueError("HIRP housing compensation unsupported")

        housing_temp = int.from_bytes(header[0x70:0x74], "little", signed=True)
        housing_delta = sensor_temp - _truncate_division(housing_temp, 20)
        adjustment = housing_compensation(
            housing_delta,
            inputs.reference_temperature - ref_housing,
            (breakpoint, low_slope, high_slope),
        )
        sensor_gray = blackbody_gray(
            inverse, 273150 + 20 * sensor_temp, table_version > 0x402
        )
        base_offset_val = response_gray_offset(
            ref_gray,
            sensor_gray,
            inputs.gray_response(sensor_temp),
            sensor_temp - inputs.reference_temperature,
            gain_val,
            gain_slope,
        )
        gray_offset = base_offset_val - adjustment

        linear, quadratic = inputs.gray_response(sensor_temp)
        minimum = int(
            np.frombuffer(table, dtype="<i4", count=1, offset=0x50 + 4 * row)[0]
        )
        maximum = int(
            np.frombuffer(table, dtype="<i4", count=1, offset=0x64 + 4 * row)[0]
        )

        reflected, humidity_milli = (
            int(x) for x in np.frombuffer(header, dtype="<i4", count=2, offset=0x70)
        )
        background = _truncate_division(reflected, 20)
        humidity = _truncate_division(humidity_milli, 1000)
        atm_idx = 5 * sensor_mode + row
        atmosphere = tuple(
            int(x)
            for x in np.frombuffer(
                table, dtype="<i4", count=3, offset=0x21AC + 32 * atm_idx
            )
        )
        trim = tuple(
            int(x)
            for x in np.frombuffer(
                table, dtype="<i4", count=4, offset=0x2DDC + 16 * atm_idx
            )
        )

        if not dj20 and table[0x2F8A] == 1:
            sel = 2 * sensor_mode + row
            alt_lin = int(
                np.frombuffer(table, dtype="<i4", count=1, offset=0x2F8C + 16 * sel)[0]
            )
            alt_dg = int(
                np.frombuffer(table, dtype="<i4", count=1, offset=0x2FAC + 4 * sel)[0]
            )
            if alt_lin or alt_dg:
                if atm_idx < 2:
                    dg = int(
                        np.frombuffer(
                            table, dtype="<i4", count=1, offset=0x2FAC + 4 * atm_idx
                        )[0]
                    )
                    atmosphere = (atmosphere[0], dg, atmosphere[2])
                    trim = tuple(
                        int(x)
                        for x in np.frombuffer(
                            table, dtype="<i4", count=4, offset=0x2F8C + 16 * atm_idx
                        )
                    )
            elif atm_idx == 0:
                atmosphere = (atmosphere[0], 260, atmosphere[2])
            elif atm_idx == 1:
                trim = (410, -10307, 0, 0)

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
            transmission = (
                transmission * humidity_transmission_q14(background, humidity)
            ) >> 14
            reflected = _truncate_division(reflected, 100) * 100

        correction = ObjectTemperatureCorrection.prepare(
            background, 950, reflected, transmission
        )
        high_temperature_trim = (
            0 if dj20 else int.from_bytes(table[0x2F88:0x2F8A], "little", signed=True)
        )

        calib_obj = GrayCalibration(
            gray_offset,
            linear,
            quadratic,
            inverse,
            minimum,
            maximum,
            correction,
            trim_gain,
            trim_offset,
            high_temperature_trim,
        )
        scale = FrameTemperatureScale.prepare(calib_obj, row, table_version > 0x402)
        return calib_obj, scale

    def _response_inputs(
        self, calibration: bytes, sensor_mode: int, row: int, group: int
    ) -> HirpResponseInputs:
        table = calibration[0x1F0:]
        ref_offset = 0x26C + 4 * (25 * sensor_mode + 5 * row + group)
        reference = int(
            np.frombuffer(table, dtype="<i4", count=1, offset=ref_offset)[0]
        )
        sensor_family = int.from_bytes(table[8:12], "little")
        polynomial = tuple(
            int(x)
            for x in np.frombuffer(
                table, dtype="<i8", count=4, offset=0x24D0 + 64 * (5 * row + group)
            )
        )
        return HirpResponseInputs(reference, sensor_family, polynomial)


def measure_hirp(
    data: bytes, segments: list[tuple[int, bytes | memoryview]] | None = None
) -> tuple[ThermalArray, tuple[int, int]]:
    """Measure Matrice 3TD / Matrice 4T HIRP R-JPEG using HirpHandler."""
    return HirpHandler.get_instance().measure(data, segments)
