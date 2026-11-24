"""DIRP radiance correction once the frame calibration has been prepared."""

from array import array
from dataclasses import dataclass
from math import ceil, floor, pow
from struct import Struct, unpack, unpack_from
import sys


_FLOAT = Struct("<f")


def _float32(value):
    return _FLOAT.unpack(_FLOAT.pack(value))[0]


def _round_away(value):
    return floor(value + 0.5) if value >= 0 else ceil(value - 0.5)


@dataclass(frozen=True)
class DirpRadiance:
    reference_code: int
    zero_code: int
    gain: float
    offset: float
    first_subtraction: float
    second_subtraction: float
    divisor: float
    reflected_tenths: int
    throughput_percent: int
    output_gain_percent: int
    output_offset_tenths: int
    temperature_lut: tuple

    @classmethod
    def from_header(cls, header, temperature_lut):
        """Prepare the DIRP calibration from a version 0x0612 APP4 header."""
        params = header[4:100] + b"".join(
            header[offset:offset + 4] for offset in (0x64, 0xA4, 0x88, 0xA8, 0x84)
        )
        zero_code, = unpack_from("<H", header, 0x146)
        flux_tenths, = unpack_from("<h", params, 0x54)
        baseline = (
            _float32(flux_tenths / 10)
            if flux_tenths >= -2735 else unpack_from("<f", params, 0x60)[0]
        )
        _, target, sensor_now, sensor_ref = unpack_from("<4f", params, 0x60)
        a, b, c, d, gain, intercept, offset = unpack_from("<7f", params, 0x20)

        def polynomial(value):
            value = _float32(value - baseline)
            square_term = _float32(_float32(b * value) * value)
            return _float32(_float32(_float32(c * value) + square_term) + d)

        first = _float32(_float32(_float32(target - baseline) * a) + intercept)
        second = _float32(polynomial(sensor_now) - polynomial(sensor_ref))
        reference_temperature = _round_away(_float32(target * 10))
        if reference_temperature >= temperature_lut[-1]:
            reference_code = 65532
        else:
            reference_code = next((
                index * 4
                for index in range(1, len(temperature_lut))
                if temperature_lut[index - 1] < reference_temperature <= temperature_lut[index]
            ), 0)
        if not reference_code:
            raise ValueError("DIRP reference temperature is below the lookup range")

        output_gain, output_offset = unpack_from("<Hh", params, 0x50)
        # Handle creation replaces the stored environment with defaults.
        # Humidity code 50 and ambient code 70 make the divisor exactly one.
        return cls(
            reference_code, zero_code, _float32(gain / 79), offset, first, second,
            1.0, 230, 100, output_gain, output_offset, temperature_lut
        )

    def temperature(self, raw):
        """Apply the calibrated DIRP lookup and radiance transform to one code."""
        if self.reference_code <= 0 or len(self.temperature_lut) != 16384:
            raise ValueError("DIRP requires a reference code and a 16,384-entry lookup")
        if self.divisor == 0 or self.throughput_percent == 0:
            raise ValueError("Invalid DIRP radiance divisor")

        code = _float32((raw - self.zero_code) * self.gain)
        code = _float32(code + self.reference_code)
        code = _float32(code + self.offset)
        code = _float32(code - self.first_subtraction)
        code = _float32(code - self.second_subtraction)
        code = max(_float32(code / self.divisor), 0.0)
        rounded = _round_away(code)
        index = 16383 if rounded > 65533 else (rounded + 2) // 4
        apparent = _float32(self.temperature_lut[index] / 10)
        reflected = _float32(self.reflected_tenths / 10)
        apparent_kelvin = _float32(apparent + 273)
        reflected_kelvin = _float32(reflected + 273)
        apparent_radiance = _float32(pow(apparent_kelvin, 4))
        reflected_radiance = _float32(pow(reflected_kelvin, 4))
        difference = _float32(apparent_radiance - reflected_radiance)
        correction = _float32(_float32(100 * difference) / self.throughput_percent)
        corrected = _float32(reflected_radiance + correction)
        kelvin = _float32(pow(max(corrected, 0.0), 0.25))
        celsius = _float32(kelvin - 273)
        scaled = _float32(_float32(celsius * self.output_gain_percent) / 100)
        return _float32(scaled + _float32(self.output_offset_tenths / 10))


def measure_dirp(segments):
    """Measure a 640×512 DIRP frame with the 0x0612 calibration header."""
    headers = [p for m, p in segments if m == 0xE4 and p.startswith(b"\xaa\x55\x12\x06")]
    raw = b"".join(p for m, p in segments if m == 0xE3)
    lut = b"".join(p for m, p in segments if m == 0xE5)
    width, height = 640, 512
    if (
        len(headers) != 1 or len(headers[0]) != 2560
        or len(raw) != width * height * 2 or len(lut) != 32768
    ):
        raise ValueError("Unsupported or truncated DIRP frame")
    calibration = DirpRadiance.from_header(headers[0], unpack("<16384h", lut))
    pixels = array("H")
    pixels.frombytes(raw)
    if sys.byteorder != "little":
        pixels.byteswap()
    temperatures = {code: calibration.temperature(code) for code in set(pixels)}
    return array("f", (temperatures[code] for code in pixels)), (height, width)
