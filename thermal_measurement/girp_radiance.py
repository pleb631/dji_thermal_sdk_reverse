"""Pure Python measurement for the GIRP R-JPEG format."""

from array import array
from base64 import b64decode
from functools import lru_cache
from math import pow
from struct import Struct, unpack_from
import sys
from zlib import decompress


_FLOAT = Struct("<f")
# Signed temperature tables selected by libv_girp.so. Each byte stores the
# change from the previous entry. 0x80 escapes to an absolute signed word.
_PACKED_TABLES = {
    "HIGH_MODE_3": (
        b"eNrtUotOwzAMTLd9OB/J9yDE2NY0iWOffemQ8NHa90qQoB+fXyWRSCQSiUQikUg4cbvernc81oFJuLTs0o/Hu6/q"
        b"qcdrNT/1eo7DU637247XOjAJW8O2ftTvk2wd3deBHZ8BHY92VUQ0tpkxWP2o9kj9PCNRLfHdx646+pjNUrds7MTM"
        b"bLxVrez0yJFdLQq3Dyi0ejGaZa6LSRSFDpiBKHssR0sT02mjZZpIPUEdyFxYAps9pcpSWStkLRl60cWcYqDHlujq"
        b"mfo0uoRu84SoISq1M+8C0XDbQ0UIji2LDSRCQ8jyL6MNBBWxhhj1iU5ajKElmNPAFAfr7h9uHQuAvwBXxKTtW7H/"
        b"V2mfGeujcn5LZ32yp3/04DF/jAZW02LBxkyaspYsDJyaaZtS1pHQotVJiWIBsbPacU7Ra4Rq0ciUe5rsGxf8cmbV"
        b"kZntJQ6ny6xPzXgYKgObewSLYMuceooiZdUJNUihMckIeiw7GhpypIV311zwF58Tz5NK9GyJ67MclakmhUSBdZiF"
        b"RcxXCnVZRSgOlFctUg0IfFVK6DzivBdKvMWBdDVCFm4rkSE2VhzVN1z3et974cICFgRMukGpGoS7orjWGO7hhFaM"
        b"2MvUGdbUD4aB7tpLfsfZ1yyIyWZAhyO4NaOEE7GyYBBtQ+gqAjORSCQSiX+EbxX2GcM="
    ),
    "LOW_MODE_3": (
        b"eNrtjUsWwjAMA+HmHJLz8FjzayxZTlnMAG0sjcPt/rgAAAAAwPXl8zYagTA2h+BRzpyDXVhx8CU1i+jwrJXfg1JV"
        b"6v1Hy5ITcdDDwDsRicdi8TGt6/XKKj5sVmVbb33TC0rhRb/ng5NSlYNw3pcDr5AtFO2kWk4eA80pv33LWXF3qRVb"
        b"QtMSg6geUXNDxh9P572/PMREsy/FhuReF/wHR4hHAXVoSTE811vYcNPg47xrUqJdx5pwOLIkzCdVo7vFY3Yj1g0G"
        b"Tc+JC2VSaejD17SeZ1yUkNxaKXZGUu/LqSFsB7Kcvq5M0fTd12u27W/TarqMp440MHoWAAAAAAAAAAAAAADANp4X"
        b"5Am/"
    ),
    "HIGH_MODE_1": (
        b"eNrtVYtywyAMI9v//8E+ttuuTQjgt4FeTyrYsiyTNrm2PwUAAAAAAAAAgG34/sVXH577StWqw5maV51e4baq9L/b"
        b"cKYbo3A07OhDvV/k6OiVbuy+BnQc2lQRUjg4YZD6UOVR9bdGRZXIfYWr6ugzNqnNrImYYQSJ1ZTkbdWWXU1pfCPo"
        b"PVESrEWjFaakCoEWRiq6rE6DgohjYVjLgahuhC8Um5dMiWBtMajtixMGlM19QYbE5bdY78dIsNw89d3V3nDXg6Gf"
        b"lvAchec5MyhVVjBwbceR3LWXCJqC+gqp1Ag2UWwpDemDzWveoLWlEyXBUNp0gbIa05SniJqXQ0mUtB83bA/PvSlZ"
        b"NRxw7Gi4jZvqJbbI9z9zaJZpz1XT31owhX6dF/0JZNgDhqTOQsnQdloTxjq+7gDbZLZ7wpxl2HzunJaqrzcZnVmD"
        b"W461hrc6N90aMiT31HLA6BvNHrC0c09Ju+g0BysmT6iaIXNoaGFeM7qiFXbz0pKux7icTrBHGy6b98C0K6xtJwmn"
        b"BgAAAAAAAAAAAAAAAHwoHhjwFO0="
    ),
    "LOW_SHARED": (
        b"eNrtl1EOwjAMQ9nNOSTnQfCL0DbHTtLAbGBr7Jd0H1CJ++N5syzLsizL6tb2fn1fYPvEicQiIUGBuxzBacryqPoM"
        b"8D7a2PHCNhACeUZH0zubpjLYBSyp1grWpleRCDZCMd5DDYvnhw4aRpdwcl5K1oENxkGs7aODWBAtaIRy0TjMFS9U"
        b"sMBuqhhAsyhfDEFCbfi1Sz5Nx8GENykoUFdGK5YspsE7hsipPTAi4OOu2Q1s3GmF8vqCAhK88wgmNL79loUSoeAy"
        b"0E5ZQeXES79l47+4i38MfUPXoSsOhezjpOoEEs6lvCNNOfe6Ji4f2UDt3s+NCrJik8oRs4ORnTCwYFIdSDesaywb"
        b"1Db4Mg+4XfnxZozpaavBW7G/i1O75uwxbqvMLQqfa8qfh5mlZVmWZVmWZVmWZVnD9ALjXAm/"
    ),
    "HIGH_SHARED": (
        b"eNrtUld2wzAMk+Pe/wQ9bF+buNEgJXAoyQdgm8RSm/VdCIIgCIIgCIL4DNy+OpzPed6f6j6f6+9qxnmtajTkjltD"
        b"rnG71mNUlzKEu1rN04/HvFZFOq7ikPhRr2oe1fM7Kn207OhoPRoyu2QKHhYvTfZEWAPXhLC7+5/KhYGpC3oUIo6B"
        b"NlMS+n6SitW0E4OcumYUxCwyLyMbSNHXdK5HQwbRPSPVVtF00V6g+ArRd2F5o5o1+VwQYhDIR40Vxu8FjGGpfuXA"
        b"jwD7nWjGkk0F9lNUXX+vxFoLQ1UwdW2LWNr4AN0F1y1owXpKFp5GZ2KQgiFaigmFJeNopjX9t/6Xmv2WNxwOXq84"
        b"HCnYAtRErEgOZ5oA/GXFEU+9EGn1wGUjVJu4gQjKPc380xvGS06GCpvsLY4lfoFwNqZmPAyVN+zcqi3KsJLSjTTU"
        b"AYM3qLVh7UuGu6R48ao/CYVAbmnZuxnn3v23Dffn/ekN7VgjPdviplmOSkjjobNpFoHO0jYUXFUjy6tbokTD6afE"
        b"aWfy1pZ2RuCr+uuO3otoQuItrqWrEbJszVlEEARBEARBEARBEARBEHvwAwUMFO0="
    ),
    "LOW_MODE_11": (
        b"eNrtVFsOgzAMg5vvkDvPtP1OCJrEdlqBvUEbP9J+tLzen80wDMMwDKMb++93fIXpAZORQQdkSoywFFYp06vqX4jn"
        b"ysQJl6YDYkBnJJr+bHdJC7MBCqqxokqXZxkpTKTkeKbULK9fMlExOw0r4xKiLuignLS1PbhxLJxVQwMUrhgSFMFa"
        b"4scizVHzoqkbvuQ5gqdBAe06U6LmWGGJFICmskRKKxAHrdxUT6Ni2ALYxW0I7xmNGKZJAoei6jML3C3ianaiA/ML"
        b"Wyr2UugNe5CDyDrQuSjtfqxxWTql7Jlvupj75oH6cen8gJQ/MoDYxFZMqcxhHBMKp2IRZYu1hSWTYcOETjpjOTAv"
        b"KGvU1vgxG9yfvL012vTENPZW2+1kamqdNZZbirmEcF+MUHvLttIwDMMwDMMwDMMwjMXwBTGpCb8="
    ),
    "HIGH_MODE_11": (
        b"eNrtUouOwyAMY+v//8E+djptXYGEPCDAtLPbJo7tcDeJRwIAAAAAAAAAYBGO+3H8fVcrX1LO59Vy5Z63z/vql0K+"
        b"rLzr2Qom4Vaz21WZInxXK1jjrSkpdauo8lCat4sXGqHFS1jW2K8kWWHq2UjPGcvZpnaGKKz5f5RDNdGZUyTZGXEg"
        b"9SaSKklj4vUkMQNJ9jE1NUMVaGKclmslSROrrz1SqrSCiYPFSEaHn8ROB620hHZr/zrhl4meL+R8R4NW2W5bDJl6"
        b"r0XX9WnfKXptLErnBXVeX82y3H3m5iu2M/BpPrc5D5MIah600SKwkiALhjkQsmJ6Zqx4DYukCdIYevKoP2sM8fqC"
        b"/mEoNKD7Ip1nBP6JeFvU+hecqd64TozzWGzK9ryl6W54aMfKwuN+pn3LiZPiWwMbndHwhP3Vhyyzg9eCj5jG9xwW"
        b"Eww0B9PR60Zh9UHLlt1q5MJya4lvD01L9sQDFgO29539D8s3H75kaVN0cyDKAwAAAAAAAAAAAAAAAIAfwBPNFRTt"
    ),
}
_LOW_TABLES = {1: "LOW_SHARED", 2: "LOW_SHARED", 3: "LOW_MODE_3",
               10: "LOW_SHARED", 11: "LOW_MODE_11"}
_HIGH_TABLES = {1: "HIGH_MODE_1", 2: "HIGH_SHARED", 3: "HIGH_MODE_3",
                10: "HIGH_SHARED", 11: "HIGH_MODE_11"}


def _float32(value):
    return _FLOAT.unpack(_FLOAT.pack(value))[0]


def _signed16(value):
    return ((value + 32768) & 0xFFFF) - 32768


def _signed32(value):
    return ((value + 2147483648) & 0xFFFFFFFF) - 2147483648


def _truncate_division(value, divisor):
    return value // divisor if value >= 0 else -(-value // divisor)


@lru_cache(maxsize=None)
def _temperature_table(name):
    packed = decompress(b64decode(_PACKED_TABLES[name]))
    values = []
    value = offset = 0
    while offset < len(packed):
        delta = packed[offset]
        offset += 1
        if delta == 128:
            value, = unpack_from("<h", packed, offset)
            offset += 2
        else:
            value += delta if delta < 128 else delta - 256
        values.append(value)
    return tuple(values)


def measure_girp(segments):
    headers = [p for m, p in segments if m == 0xE4 and p.startswith(b"\xff\xd2\xd1\xffgirp")]
    raw = b"".join(p for m, p in segments if m == 0xE3)
    width, height = 640, 512
    if len(headers) != 1 or len(headers[0]) < 302 or len(raw) != width * height * 2:
        raise ValueError("Unsupported or truncated GIRP frame")
    calibration = headers[0][32:]
    if calibration[:2] != b"\xaa\x55":
        raise ValueError("Invalid GIRP calibration header")
    convert = _pixel_converter(calibration)
    pixels = array("h")
    pixels.frombytes(raw)
    if sys.byteorder != "little":
        pixels.byteswap()
    values = {raw_code: convert(raw_code) for raw_code in set(pixels)}
    return array("f", (values[raw_code] for raw_code in pixels)), (height, width)


def _pixel_converter(calibration):
    humidity, distance, emissivity, ambient = unpack_from("<3Hh", calibration, 4)
    if not (0 < emissivity <= 100 and humidity <= 100 and 5 <= distance <= 5000):
        raise ValueError("Unsupported GIRP measurement parameters")
    emissivity = max(50, emissivity)
    high_range, = unpack_from("<H", calibration, 0x24)
    temperature_bias, = unpack_from("<h", calibration, 0x16)
    mode = calibration[0xE8]
    tables = _HIGH_TABLES if high_range else _LOW_TABLES
    lut = _temperature_table(tables.get(mode, tables[1]))
    a, b, gain, c, d, offset, distance_gain, distance_offset, fixed_offset = unpack_from(
        "<iiHiiHHii", calibration, 0xF0
    )
    distance_delta = distance - 50
    gain_poly = _signed32((a * distance_delta + b) * distance_delta)
    offset_poly = _signed32((c * distance_delta + d) * distance_delta)
    gain = (gain + _truncate_division(gain_poly, 10000)) & 0xFFFF
    offset = (offset + _truncate_division(offset_poly, 10000)) & 0xFFFF
    humidity_gain = _truncate_division((5060 - humidity) * gain, 5000) & 0xFFFF
    index_gain = _truncate_division(humidity_gain * distance_gain, 1000) & 0xFFFF
    index_offset = _truncate_division(_signed32(humidity_gain * distance_offset), 1000)
    index_offset = (index_offset + fixed_offset + offset) & 0xFFFF
    reflected = ambient if abs(ambient - 230) > 20 else 230
    reflected_radiance = pow((reflected + 2735) / 10, 4)

    def convert(raw):
        index = _signed16(_truncate_division(raw * index_gain, 8192) + index_offset)
        temperature = _signed16(lut[min(max(index, 0), 16383)] + temperature_bias)
        if emissivity < 100:
            apparent = pow((temperature + 2735) / 10, 4)
            corrected = (apparent - reflected_radiance) * 100 / (emissivity + 1)
            kelvin_tenths = pow(corrected + reflected_radiance, 0.25) * 10
            temperature = _signed16(int(kelvin_tenths - 2735))
        correction = 0.0
        if emissivity == 95 and temperature > 1550:
            polynomial = _float32(_float32(temperature * temperature) + _float32(21148 * temperature))
            correction = _float32(_float32(polynomial / 1000000) - 12)
        # The native path applies its default 100% output gain in float32.
        scaled = _float32(_float32(temperature + correction) * 100)
        return _float32(_float32(scaled / 100) / 10)

    return convert
