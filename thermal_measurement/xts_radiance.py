"""Pure Python measurement for the Zenmuse XT S Plug R-JPEG layout."""

from array import array
from struct import Struct, unpack_from
import sys


_FLOAT = Struct("<f")
_WIDTH, _HEIGHT = 640, 512


def _float32(value):
    return _FLOAT.unpack(_FLOAT.pack(value))[0]


def _signed16(value):
    return ((value + 32768) & 0xFFFF) - 32768


def measure_xts(segments):
    plugs = [data for marker, data in segments if marker == 0xE2 and data.startswith(b"Plug")]
    raw = b"".join(data for marker, data in segments if marker == 0xE4)
    table = b"".join(data for marker, data in segments if marker == 0xE3)
    if (
        len(plugs) != 1 or len(plugs[0]) != 1260
        or len(raw) != _WIDTH * _HEIGHT * 2 or len(table) != 32768
    ):
        raise ValueError("Unsupported or truncated XT S R-JPEG")

    plug = plugs[0]
    # The packed camera block has an extra two-byte field before its second half.
    calibration = plug[1178:1198] + plug[1200:1260]
    lut = array("h")
    lut.frombytes(table)
    pixels = array("H")
    pixels.frombytes(raw)
    if sys.byteorder != "little":
        lut.byteswap()
        pixels.byteswap()
    convert = _pixel_converter(calibration, lut)
    values = {code: convert(code) for code in set(pixels)}
    return array("f", (values[code] for code in pixels)), (_HEIGHT, _WIDTH)


def _pixel_converter(calibration, lut):
    words = Struct("<40H").unpack(calibration)

    def unsigned(offset):
        return words[offset // 2]

    def signed(offset):
        return _signed16(unsigned(offset))

    initial_sensor = _float32(unsigned(0x30) / 100)
    current_sensor = _float32(unsigned(0x2C) / 100)
    target = _float32(unsigned(0x40) / 100)
    baseline = _float32(unsigned(0x4E) / 100)

    drift = int((signed(0x0E) / 1000 * target) * target + signed(0x10) / 1000 * target)
    drift = (drift + unsigned(0x22)) & 0xFFFF
    sensor_delta = _float32(current_sensor - initial_sensor)
    sensor_offset = int(_float32(signed(0x0A) + _float32(signed(0x0C) * sensor_delta))) & 0xFFFF

    delta = _float32(target - baseline)
    quadratic, = unpack_from("<i", calibration, 0x3C)
    gain = signed(0x38) / 1000
    gain += _float32(signed(0x3A) * delta) / 10000
    gain += _float32(_float32(delta * delta) * _float32(quadratic)) / 10000
    gain = _signed16(int(gain * abs(signed(0x4A))))
    if target > current_sensor:
        gain -= 350
    thermal_offset = int((gain / 10) * _float32(current_sensor - target))

    reference = _float32(target * 10)
    reference_index = next((index for index, value in enumerate(lut) if value >= reference), 0)
    raw_scale = signed(0x08) / 100
    raw_offset = drift - sensor_offset + thermal_offset - unsigned(0x36)
    lut_offset = signed(0x4C)
    temperature_offset = int(signed(0x12) / 10) + signed(0x1E)
    # Default XT S measurement uses 70% humidity, distance code 5, emissivity 100%.
    humidity_scale = _float32(1 - (70 - 60) / 5000)

    def convert(raw):
        adjusted = _signed16(raw + raw_offset)
        index = _signed16(int(raw_scale * adjusted + reference_index + lut_offset))
        index = _signed16(int(_float32(index * humidity_scale)))
        index = min(max(index, 0), len(lut) - 1)
        return _signed16(lut[index] + temperature_offset) / 10

    return convert
