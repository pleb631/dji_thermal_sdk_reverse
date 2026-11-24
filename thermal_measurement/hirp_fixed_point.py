"""MicroTA fixed-point inverse response used by HIRP measurement."""

from base64 import b64decode
from dataclasses import dataclass
from itertools import accumulate
from math import exp
from struct import Struct, unpack_from
from zlib import decompress


# Reciprocal bases, reciprocal slopes and logarithm values from MicroTA.
_PACKED_TABLES = (
    b"eNo11nt8zfUDx/Fzv9/vt++5fM/3/j3L2DQzl2yG2BCmZi6LYZnbXGIZeriuFMvcKpcWUyGEkiYUMSKmCblkQu6R"
    b"e2l+v8fv+/7tn+f+2ONsO+fzeb++e6bL/vfVAe6dJtkJHpwq2R02lEsWwItTJEvg/Tckp0MDXFImGYdfTJZsD49O"
    b"khwE77wuOQO64acTJdvA+gmSw2HTeMnFMAnuH4fXh09KJRfBZHh4rORrUAtrxkh2hL+PlpwFabh/lGQxNMKNIyV7"
    b"wUclksthJrw2QrISpsHfXpOsgM3hmWLJ2TAZnh2On4epsHGY5ALYFt4cKvkh7Ar/LpJcD/tDC9wzRHIcpOHpwZLv"
    b"wA7w4at4PVgIvfBIId5X2AbeHyS5ARbBMDw5UPI92A1q4HcDJKfCdPigv+QWOAYm4PUCnDs4HDLwcj+cD1gEaXgl"
    b"X/ITWAwFePMVnA9YClPho5cla+F0mAV18EhfyYXwFRiGl/LwvsNS2ArKYF0fnD/4CozCa73xfsFymA2t8Ndekmvg"
    b"GNgaqmH9S5Ir4GuwJVTAoz1xX+BrMA2q4c89JKvhWNgOmuG57jhPsBzmwBC8lSv5LZwPB8HmUAEbciTXwsmwGyTg"
    b"nW7YW7gEjoDtoB1e7ir5DZwPh8BW0Awvvij5NXwXDoHp0AqvdJHcCavgCNgB+uCdzpIH4Co4CfaAHJTB051wfuA8"
    b"WATbQS+8my35I6yB02E+TIUWeK2j5D64Ck6BfWELaIbXsiR/gNVwGuwH06AL3s2U/AmuhxVwGMyCMfisA84jrIUf"
    b"wMnwZdgSuuC9FySPwy/ge3As7AmToQ2+ie7PRO/fQucXou8r0fUN6Pi36Hc9un0VvZZDAp3OQJ/7o8vT0eEa9PcI"
    b"evsYfaVhH/R1Nnq6A/38E91k4avo5Ur08Sx6SMBB6OAadO86OpcCp6JvB9ExLxyOfu1Ar0xwMPr0NXpkhyPRnzp0"
    b"hoaz0Zcr6EgXuAHdcMAydOIiepADt2P/KViFnVfACdj1K9jtfvAodroT3IU9Tvv/bmN3k+A67CsHP8WOcnAd9jIJ"
    b"foFdfB7uxO51hIexb33hBezXSPgE+zQXumENduh5WIed6QdvYz9mQh/chJ3oBM/j/k+GDvg57ncXeBn3diaMwT24"
    b"l4VQDlfjvmXDq7g/78Dm8Jf2OF8wDn9sh88NEnB/W/QYhmBdG8mJkITHMrBH8Dl4rjW6ANvDu+n4+2FfaIC7WuH5"
    b"DXLwfBqej2E3qIS1z+P/gUnwj5bYTdgfeuHPqXj+hDnQAOtSJOfAbKiC+1rg+RBmQw2say75NsyFNvhzsuRSWACj"
    b"8HIz7DcshWmw6Tl8bvBdmAfD8I8kyc2wDGZBCzyVwE7BMTADamGDKPkRHA0zoB6eFPD8ASfCjtAFL/GS2+AsmAdZ"
    b"+JiTPASXw9GwA3TBP1icD7gADoFp0AQbGcmv4DtwMGwFLfAyjd2BVXAEzIQB+BeF+wdXw3KYB5tBHbwYx++BS2Ep"
    b"zIEsVMDzJJ7T4BI4DnaHItTCSzHJ7+EqOBUWwHTohQ+iOB9wK1wIS+FLMBna4H+/k0VlybIXZD1lhbKxsjdlC2Sr"
    b"ZJtku2VHZedlt2VNMouckCfJ28pz5f3lI+Xl8nnyD+Xr5N/ID8lPy6/JH8t1Cq+CVTyv6KTIUxQpJihmKqoUHyu+"
    b"UHynOKb4TfGnoklhVoaUCWWGsqsyX1msnKycq1yirFFuU+5VHlc2Ku8onyktKkKVUGWouqryVcWqSao5qsWqNaqt"
    b"qu9Vx1S/qW6r/lWZ1AE1r26l7qzOUxepx6tnqCvVq9Qb1TvVP6pPq6+qH6nVGpeG1DTXtNd01/TXlGjKNBWaJZoa"
    b"zVbNd5pjmvOaW5p/NHqtT8toU7WZ2pe0g7SjtOXat7XLtGu127R7tfXa37S3tP9o9Tqvjtal6jJ1PXUDdSN1b+gq"
    b"dEt0a3RbdHt0P+nO6q7rHunUeqc+pm+mb6vvpn9FP0w/QT9DX6lfoV+v36E/oG/QX9Tf0TfpjQa/gTWkGjINPQwD"
    b"DCWGyYY5hipDtWGTYafhkOGk4bLhL4PMaDYGjZyxpTHL2NM40FhinGycY6wyfmTcaKw11hl/Mf5uvGNsMhpNPhNt"
    b"SjG9YMo19TMNN000zTAtMC03fWb6yrTXVG86b7phemxSmx3miDlhTjd3Mvc2F5pHmd8wzzUvMlebN5przQfMDeZG"
    b"823zP2adxW0hLc9ZMixdLHmWwZbRlimWCstiS7Vlo6XWcsDSYGm03Lb8bdFaXdaoNcmabu1k7W0dZB1pnWydbX3P"
    b"utK63rrdus96zHrOes360KqwWWxBG2tLtb1gy7Xl24baxtmm2d62LbWttm2y7bTV2RpsjbZbtic2jd1hD9sFe5o9"
    b"y97DXmAfbp9gf9P+jn2ZfY19s32n/aD9hL3Rfsv+xK522B2Eg3e0dHRw5DryHUMdpY6pjrccix3Vjs8dOxw/OOod"
    b"5xzXHA8ccqfZ6XNSzmRnG2cXZx9nobPEOck507nA+aFzrXOLc5fzkPOEs9F5y/nYqXLZXCEX60pxtXd1dfV1DXaN"
    b"cpW5ZrkqXctdn7i2una5DrlOuBpdt1yPXSq31R10M+4W7rbuF9193IXuEvfr7hnud93vu9e4N7tr3fvdx93n3Ffd"
    b"993P3EaPxxPzJHlaebI83T39PEM9Yz3lnrmehZ4Vns882zy7PYc8JzyNnpueRx6F1+z1eSlvM29rb7a3p7fAO8xb"
    b"6i33VnirvCu9n3m3eXd7D3lPeC94b3gfeuU+k8/rI31Jvla+LF93X76vyDfGV+ab5Vvg+8BX49vsq/Xt99X7zviu"
    b"+O76nvq0foc/5Gf9Lfxt/V38vfwD/MX+8f6p/gp/lX+l/1P/Vv8uf53/Z/95/1X/PX+TXxdwBsIBLpASaBfoEugd"
    b"GBAoDowPTA1UBBYGVgQ+CWwJfBs4EDgeOBv4I3A38DSgCdqDwSATTA5mBLODPYL9gkXBMcGy4Kzg/OCy4Org58Gv"
    b"g98HDwdPBhuDN4IPg7KQIeQORUJ8KDXULtQl1Cs0IDQ8VBqaEpoTqgx9EFoT2hTaEdobOhI6GboYuhl6GJIRBsJN"
    b"RAieSCHaEp2Jl4gCYhgxlniDmEXMJ5YRHxMbiK+IPcQhooE4T1wl7hH/EtqwPRwM0+Fm4VbhzHBOuG+4MDwiPCE8"
    b"LVwRXhheHl4b3hz+JrwvfCR8MtwYvhF+EH4W1keckVCEiSRHWkc6RrpHXo68GimJTIxMj1REFkaWR9ZGNkV2RPZG"
    b"jkR+iVyIXI/ci/wb0Ubt0UCUij4XTYt2iHaL9okOjBZHS6NTorOj86PLotXR9dEvo7uiddH66Jnopejt6OOoPGaM"
    b"uWLhGBtrHmsd6xjLjb0cK4yNiI2PTY3NiVXG3o+tjm2IfRXbHTsYOx47E7scux17HJOTRtJFEiRLJpPpZBaZQ/Yl"
    b"B5HF5DhyCjmLnE8uJT8i15FbyZ3kD+RR8hTZSN4g75P/kpq4Le6Pk3Ex3jLeLt453jOeHx8SHxmfGJ8Wnxt/L/5B"
    b"fHV8Q/zL+K54Xbw+/mv89/jN+IN4U1xH2akAFacSVEuqHdWZ6knlU4OpEmoCNY2aS1VS71MfU+upbdS31H7qKHWK"
    b"aqSuU/eop5SattBeOkrzdHO6NZ1F59B59EB6OD2WLqNn0PPoRfQKuobeSG+n99AH6eP0Gfp3+ib9gG6itYyN8TMx"
    b"RmBSmDZMNpPL9GUGMcXMWKaMmcHMYxYxy5kaZiOzndnN1DH1zGnmInODuc88ZdSshfWwEZZjk9l0NpPtyvZm+7NF"
    b"7Cj2dXYaO5etZJex1exn7Ba2lt3LHmYb2HPsFfY2+4h9xuo4G+fnSE7gUrgMriOXw+VxA7hh3GhuEjedq+Aqufe5"
    b"am4dt4Wr5fZyh7kG7ix3mbvFPeSaOC1v5X18lOf45nw6n8l35XvxBfwQvoSfwJfzs/l3+cX8Sn4tv5Hfzu/mD/DH"
    b"+FP8Bf4af5d/wisEg+AQAkJcEIUUoY3QUcgR+ggDhKHCKGGiME2YI8wXlgirhE+ETcJ2YbdQJxwTTgkXhKvCXeGJ"
    b"IBf1ol30i6QoiC3E1mKm2FXsJRaIQ8QScbw4RZwpzhOrxA/F1eIGcZu4U9wnHhYbxLPiJfGmeF98KqoSpoQrEUpQ"
    b"iUQiNdEmkZ3ITeQl/gNMt/FM"
)
_TABLES = decompress(b64decode(_PACKED_TABLES))
_RECIPROCAL_BASES = unpack_from("<256q", _TABLES)
_RECIPROCAL_SLOPES = unpack_from("<256q", _TABLES, 2048)
_LOGARITHMS = unpack_from("<1024H", _TABLES, 4096)
_SQRT_DELTAS = (
    b"eNqNigkOwCAMw6Ty/z9PSGjrEWdwNHYg1l7xjg9Dm7kTO3VRx+CUPHwkKCj0YrsCQQQWVhBNrb+BYDDa"
    b"+yPCeDjEWeOIfOMUICkjcrfpquH2/r2sB9AhCV4="
)
_SQRT_VALUES = tuple(accumulate(decompress(b64decode(_SQRT_DELTAS)), initial=5793))
_SQRT64_PACKED = (
    b"eNot1Xks12EcwHGSIs0RSUSOyLpJQ5mSkiaMlKSMmg6TDhkzGTpI0oGpVZIsIZQuM5JMl5mlY3KlmJRGhysdWnt/"
    b"fv+8tt++z/N8Pp/n8zzPodfjFP7/PJvQsAU/tWFpB0Z1okM3/unBil6M7EOrb9jzAy8OocdPGfcLC/7iBkUl/lfC"
    b"qxNwjSr2qGGiOs7Swipt9NXFPj1MMEAdI8w1RmszrDSX+S2xYQ76zMemhehnjW9tZB1bfGmP7g5Y64jLnLDEGU1d"
    b"8KwrKrphmLvM64nO3ljog1q+GOEn8fijXQCeC8TBbegVLON34vgQ9A/FW2GovF/yCMfrETgciaui8UwMtsRKveMx"
    b"9DCWHpVxibg0GWNSsCIVf51G+zTJJwNvZuKn82hyETdl4clsrM7BgVw0z8MN+XikUPIrwrYSVCmV/b6Dm+9hfBnm"
    b"lWNdBfY/kLpXy7ga9K7FfU8krmdYUCf7Xo/tDVKPF6jxSuJ9I/VpQo9mDGrF8HbJowPTP2BOl9SpW+rZg08/Y+MX"
    b"2Z8+7PyKvd/x2wAODeHoiJyvURz7jQpj0jeNnMN3DTijHgOfY9ETVKzFrY+wugrnVWJWOU4rw8y7aHQbb9zEFcXY"
    b"Uogx+WiWhy9y8WgOOmbj30tYcwFTz2NAJtpkoGYafj2Nb1Il3hS8lYzXkjD7GF4+gjkJcj/F4f1YfBqD7dE4EiX5"
    b"RqJ9hNQtHE/sx/K92LcHzUPluxBZdxe+34GWwXhgOz4MQq1A3BGAVVtQ3x+j/bDVF1dulLr7oN56TPLCUU/c5yH3"
    b"/ToMdsOutbKeK/a6SFyr5R53xuSVEoeT9MtyWd8Rmx0wYhlqL5V3xU7uf1uJZwlesUG3xThsJfu1CH0XotoCqc88"
    b"eZ/movUc7LfE4tkYZiHvk7nMa4YPTDHRBL2N0XimzGMo/TMD0w1wtz46TZf89XBQFxunSr/p4BltPDgFN2lJf2ui"
    b"hQZqqEs9JuNHNenjSfhYFctUsHCi9PEEOXfKeGo8HleS8zQOExQxTgH/AeHDAhA="
)
_SQRT64 = decompress(b64decode(_SQRT64_PACKED))
_SQRT64_BASES = unpack_from("<128q", _SQRT64)
_SQRT64_SLOPES = unpack_from("<128q", _SQRT64, 1024)
_FLOAT32 = Struct("<f")


def _signed32(value):
    return ((value + 2147483648) & 0xFFFFFFFF) - 2147483648


def _signed64(value):
    return ((value + (1 << 63)) & 0xFFFFFFFFFFFFFFFF) - (1 << 63)


def _float32(value):
    return _FLOAT32.unpack(_FLOAT32.pack(value))[0]


def _square_root_64(value):
    if value <= 0:
        raise ValueError("HIRP quadratic response has a nonpositive discriminant")
    highest_bit = value.bit_length() - 1
    normalized = int(float(_signed64(value << 20)) / (1 << (highest_bit + 1)))
    bin_number = ((normalized - 0x7F000) >> 12) & 0xFF
    if not 1 <= bin_number <= 128:
        raise ValueError("HIRP quadratic response is outside the square-root grid")
    midpoint = (bin_number << 12) + 0x7F800
    root = ((normalized - midpoint) * _SQRT64_SLOPES[bin_number - 1]) >> 17
    root += _SQRT64_BASES[bin_number - 1] * 8
    scaled = normalized << 22
    threshold = root * root + root + 1
    if scaled > threshold + root * 2 + 2:
        root += 2
    elif scaled > threshold:
        root += 1
    if (highest_bit + 1) & 1:
        return ((1 << (highest_bit >> 1)) * 46341 * root) >> 36
    return ((1 << ((highest_bit + 1) >> 1)) * root) >> 21


def _log_mantissa(value, highest_bit):
    normalized = (value << 14) >> highest_bit
    return _LOGARITHMS[(normalized - 16384) >> 4]


def _reciprocal_product(numerator, denominator):
    magnitude = abs(denominator)
    exponent = magnitude.bit_length() - 1
    normalized = (magnitude << 15) >> exponent
    index = (normalized - 32768) >> 7
    midpoint = 32832 + (index << 7)
    reciprocal = _RECIPROCAL_BASES[index]
    reciprocal += (_RECIPROCAL_SLOPES[index] * (midpoint - normalized)) >> 15
    refined = ((2147483648 - normalized * reciprocal) * reciprocal) >> 15
    result = (refined * abs(numerator)) >> (exponent + 30)
    return -result if (numerator < 0) != (denominator < 0) else result


def inverse_sensor_response(gray, coefficients):
    """Convert a MicroTA response code to its fixed-point temperature."""
    base, temperature_scale, response_floor, response_slope = coefficients
    scaled_gray = _signed32(gray << 8)
    if scaled_gray <= response_floor:
        raise ValueError("HIRP response is below the calibrated floor")
    response = _signed32(scaled_gray - response_floor)
    adjusted_base = _signed32(base - (_signed32(response_floor * response_slope) >> 8))
    adjusted_base += _signed32(gray * response_slope)
    if adjusted_base <= 0 or adjusted_base < response:
        raise ValueError("HIRP response is outside the calibrated range")

    base_bit = adjusted_base.bit_length() - 1
    response_bit = response.bit_length() - 1
    logarithm = (base_bit - response_bit) * 11356
    logarithm += _log_mantissa(adjusted_base, base_bit)
    logarithm -= _log_mantissa(response, response_bit)
    denominator = logarithm >> 2
    if denominator == 0:
        raise ValueError("HIRP response denominator is zero")
    return _signed32(_reciprocal_product(_signed32(temperature_scale << 6), denominator))


def _truncate_division(value, divisor):
    return value // divisor if value >= 0 else -(-value // divisor)


def blackbody_gray(coefficients, temperature_millikelvin, use_calibrated_slope):
    """Evaluate MicroTA's forward blackbody response at a reference temperature."""
    if temperature_millikelvin <= 0:
        raise ValueError("HIRP blackbody temperature must be positive")
    amplitude, exponent, floor, slope = coefficients
    exponent_q14 = _truncate_division(exponent * 1000, temperature_millikelvin)
    whole, fractional = divmod(exponent_q14, 1 << 14)
    fractional_exp = round((1 << 14) * exp(fractional / (1 << 14)))
    scaled_exp = _float32(_float32(exp(whole)) * fractional_exp)
    denominator = int(_float32(scaled_exp / 64)) - (slope if use_calibrated_slope else 256)
    if not denominator:
        raise ValueError("HIRP blackbody response denominator is zero")
    response = abs(amplitude) // abs(denominator)
    if (amplitude < 0) != (denominator < 0):
        response = -response
    return _truncate_division(floor, 256) + response


def prepare_inverse_response(coefficients, reference_gray, gain, reference_gain):
    """Center and scale the radiometric coefficients for inverse conversion."""
    if not 0 <= reference_gray <= 0x3FFF:
        raise ValueError("HIRP reference gray is outside its valid range")
    amplitude, exponent, floor, slope = coefficients
    scale = (1000 * gain) // max(reference_gain, 10)
    return (
        _truncate_division(amplitude * scale, 1000),
        exponent,
        _truncate_division((floor - reference_gray * 256) * scale, 1000),
        slope,
    )


def housing_compensation(temperature_delta, reference_delta, coefficients):
    """Apply the two-slope housing temperature compensation."""
    breakpoint, low_slope, high_slope = coefficients
    threshold = breakpoint * 50
    current_high = (temperature_delta << 14) > threshold
    reference_high = (reference_delta << 14) > threshold
    current_slope = high_slope if current_high else low_slope
    reference_slope = high_slope if reference_high else low_slope
    numerator = temperature_delta * current_slope - reference_delta * reference_slope
    adjustment = (breakpoint * (current_slope - reference_slope)) >> 28
    result = (_truncate_division(numerator, 50) >> 14) + adjustment
    if not -1000 <= result <= 1000:
        raise ValueError("HIRP housing compensation is outside its valid range")
    return result


def response_gray_offset(reference_gray, sensor_gray, gray_response, temperature_delta, gain, slope):
    """Calculate the base gray before additive frame compensation."""
    linear, quadratic = gray_response
    corrected = _signed64(sensor_gray * (linear + sensor_gray * quadratic)) >> 32
    adjusted_gain = gain + _truncate_division(_signed32(temperature_delta * slope), 50)
    return reference_gray - (_signed32(corrected * adjusted_gain) >> 14)


def humidity_transmission_q14(background, humidity):
    """Interpolate the native humidity correction by temperature and humidity."""
    humidity_points = (0, 20, 50, 80, 100)
    cold = (0x41AF, 0x4102, 0x4000, 0x3F54, 0x3EE1)
    warm = (0x423F, 0x4159, 0x4000, 0x3EFD, 0x3E51)
    hot = (0x42CF, 0x41AF, 0x4000, 0x3E2E, 0x3CF8)

    def interpolate(value, start, end, first, last):
        return first + _truncate_division((last - first) * (value - start), end - start)

    if background <= 1250:
        curve = cold
    elif background <= 2000:
        curve = tuple(interpolate(background, 1250, 2000, a, b) for a, b in zip(cold, warm))
    elif background <= 2500:
        curve = tuple(interpolate(background, 2000, 2500, a, b) for a, b in zip(warm, hot))
    else:
        curve = hot
    for index, (start, end) in enumerate(zip(humidity_points, humidity_points[1:])):
        if start <= humidity <= end:
            return interpolate(humidity, start, end, curve[index], curve[index + 1])
    raise ValueError("HIRP humidity must be between 0 and 100 percent")


def _atmospheric_exponential(value, fractional_bits):
    """Evaluate the native fifth-order exponential in fixed point."""
    square = value * value
    cube = _signed64(square * value)
    power_shift = 32 if fractional_bits == 16 else 0
    fourth = _signed64((cube >> power_shift) * value)
    fifth = _signed64(fourth * value)
    return (
        (1 << fractional_bits) + value + (square >> (fractional_bits + 1))
        + _truncate_division(cube >> (2 * fractional_bits + 1), 3)
        + _truncate_division(fourth >> (3 * fractional_bits + 3 - power_shift), 3)
        + _truncate_division(fifth >> (4 * fractional_bits + 3 - power_shift), 15)
    )


def atmospheric_transmission_q14(background, humidity, distance, coefficients):
    """Calculate the fixed-point atmospheric factor from the calibration row."""
    temperature_gain, distance_gain, reference_distance = coefficients
    delta = distance - reference_distance
    temperature_term = temperature_gain * delta
    distance_term = distance_gain * delta
    reverse = _signed64(temperature_term << 23) <= 100000
    if reverse:
        temperature_term = -temperature_term
        distance_term = -distance_term
    distance_term = min(distance_term, 20000000)

    squared = background * background * 128 // 625
    temperature_curve = 2445 + (_truncate_division(22358 * background, 50) >> 7)
    temperature_curve += (80123 * squared) >> 23
    temperature_curve += (11027 * squared * _truncate_division(background, 50)) >> 26
    temperature_curve += (10632 * squared * _truncate_division(background, 50) ** 2) >> 32
    humidity_term = _truncate_division(
        temperature_curve * humidity * ((temperature_term * 1022) >> 10), 78125
    )
    humidity_factor = _atmospheric_exponential((-2086 * humidity_term) >> 17, 16)

    thresholds = (3, 8, 16, 32, 80, 160, 320, 800, 1600, 3200)
    bases = (16384, 16357, 16204, 16073, 15439, 14615, 13926, 12900, 11436, 10207)
    slopes = (-78643, -69905, -50332, -41943, -21671, -11115, -6711, -3426, -1552, -765)
    index = next((i for i, end in enumerate(thresholds) if distance_term * 16 <= end * 100000), 9)
    dry_base = bases[index] + (_truncate_division(((distance_term * 8145) >> 13) * slopes[index], 100000) >> 6)
    dry_exponent = _truncate_division(-154 * temperature_term, 100000)
    dry_factor = _atmospheric_exponential(dry_exponent, 14)
    result = (dry_factor * ((humidity_factor * dry_base) >> 16)) >> 14
    if not 0 < result <= 16384:
        raise ValueError("HIRP atmospheric transmission is outside its valid range")
    return 16384 + ((393216 - 24 * result) >> 4) if reverse else result


def atmospheric_temperature_trim(background, distance_delta, coefficients, dj20):
    """Prepare the Q23 gains and Q8 offsets for calibrated atmospheric trim."""
    offset, linear, quadratic, cubic = coefficients
    delta = distance_delta
    scale = 1 << 23
    scale += _truncate_division((linear * delta) << 13, 100000)
    squared = _truncate_division(quadratic * delta * delta, 1000)
    scale += _truncate_division(squared << 13, 10000000)
    cubed = _truncate_division(cubic * delta * delta, 1000)
    cubed = _truncate_division(cubed * delta, 10000)
    scale += _truncate_division(cubed << 13, 100000000)
    minimum, maximum = (0xCCCCD, 0x851EB8) if dj20 else (0x59999A, 0x8CCCCC)
    scale = min(max(scale, minimum), maximum)
    reciprocal = (1 << 46) // scale
    anchor = background * offset
    anchor_i32 = _signed32(anchor)
    base = ((anchor_i32 >> 2) // 50) + (anchor_i32 < 0)

    def correction(gain):
        return base - _truncate_division(_signed64(anchor * gain) >> 25, 50)

    return scale, correction(scale), reciprocal, correction(reciprocal)


def _fourth_power_q12(value):
    squared = (value * value) >> 12
    return (squared * squared) >> 12


def _lookup_square_root(value):
    if value <= 0:
        raise ValueError("HIRP square root requires a positive input")
    highest_bit = value.bit_length() - 1
    scaled = (value << 12) & 0xFFFFFFFFFFFFFFFF
    if scaled >> 63:
        numerator = float(scaled >> 1) * 2
    else:
        numerator = float(scaled)
    normalized = int(numerator / (1 << (highest_bit + 1)))
    index = (((normalized + 0x3FF800) << 15) & 0xFFFFFFFF) >> 16
    entry = _SQRT_VALUES[index]
    if (highest_bit + 1) & 1:
        exponent = highest_bit >> 1
        return ((1 << exponent) * 11585 * entry) >> 26
    exponent = (highest_bit + 1) >> 1
    return ((1 << exponent) * entry) >> 13


@dataclass(frozen=True)
class ObjectTemperatureCorrection:
    """Prepared coefficients for MicroTA's blackbody-to-object transform."""

    throughput_q14: int
    radiance_offset_q12: int
    radiance_gain_q12: int = 3973

    @classmethod
    def prepare(cls, background, emissivity, reflected, atmosphere_q14):
        """Build the correction using native fixed-point environment fields."""
        emissivity_q12 = (emissivity << 12) // 1000
        throughput = (emissivity_q12 * atmosphere_q14) >> 12
        if throughput <= 0:
            raise ValueError("HIRP environment has zero throughput")
        background_kelvin = _truncate_division(_signed32(background << 12), 50) + 0x111266
        reflected_kelvin = _truncate_division(reflected << 12, 1000) + 0x111266
        background_radiance = _fourth_power_q12(background_kelvin)
        reflected_radiance = _fourth_power_q12(reflected_kelvin)
        environmental_radiance = (
            (atmosphere_q14 - throughput) * reflected_radiance
            + (16384 - atmosphere_q14) * background_radiance
        ) >> 14
        offset = ((123 * background_radiance) >> 12) - environmental_radiance
        return cls(throughput, offset)

    def apply(self, blackbody_q12):
        """Convert a blackbody temperature to the native Q8 object temperature."""
        shifted = _signed32(blackbody_q12 + 0x111266)
        radiance = (_fourth_power_q12(shifted) * self.radiance_gain_q12) >> 12
        radiance += self.radiance_offset_q12
        if (radiance << 11) & 0xFFFFFFFFFFFFFFFF == 0:
            raise ValueError("HIRP environment correction produced zero radiance")
        gain = (1 << 26) // self.throughput_q14
        adjusted = ((radiance * gain) & 0xFFFFFFFFFFFFFFFF) >> 12
        root = _lookup_square_root(adjusted)
        kelvin_q8 = _lookup_square_root(root << 6) << 2
        return _signed32(kelvin_q8 - 0x11126)


@dataclass(frozen=True)
class GrayCalibration:
    """Per-frame coefficients for the integer MicroTA gray conversion."""

    gray_offset: int
    linear_response: int
    quadratic_response: int
    inverse_coefficients: tuple
    minimum_q8: int
    maximum_q8: int
    object_correction: ObjectTemperatureCorrection
    trim_gain_q23: int | None = None
    trim_offset_q8: int = 0
    high_temperature_trim: int = 0

    def gray_from_blackbody(self, temperature_q8, use_calibrated_slope):
        """Convert a blackbody temperature to a bounded sensor gray code."""
        temperature = 273150 + _truncate_division(_signed32(temperature_q8 * 1000), 256)
        gray = blackbody_gray(self.inverse_coefficients, temperature, use_calibrated_slope)
        corrected = _signed64(gray * (self.linear_response + gray * self.quadratic_response)) >> 32
        return min(max(corrected + self.gray_offset, 0), 0x3FFF)

    def response(self, raw):
        """Correct one raw code before applying the inverse response."""
        gray = _signed32(raw - self.gray_offset)
        if self.quadratic_response >> 9 == 0:
            multiplier = int(float(1 << 46) / self.linear_response)
            return _signed32(gray * multiplier) >> 14

        linear = self.linear_response >> 5
        discriminant = _signed64((self.quadratic_response << 2) * gray)
        discriminant += _signed64(linear * linear) >> 22
        root = _square_root_64(_signed64(discriminant << 8))
        multiplier = int(float(1 << 32) / self.quadratic_response)
        corrected = _signed64(((root << 12) - self.linear_response) * multiplier)
        return _signed32(corrected >> 33)

    def temperature_q6(self, raw):
        """Convert a raw gray code to the native per-pixel Q6 temperature."""
        kelvin_q8 = inverse_sensor_response(self.response(raw), self.inverse_coefficients)
        blackbody_q8 = min(max(kelvin_q8 - 0x11126, self.minimum_q8), self.maximum_q8)
        object_q8 = self.object_correction.apply(blackbody_q8 << 4)
        if self.trim_gain_q23 is not None:
            object_q8 = _signed32(
                ((_signed64(object_q8 * self.trim_gain_q23)) >> 23) + self.trim_offset_q8
            )
        temperature = object_q8 >> 2
        if (object_q8 >> 8) > 371:
            trim = self.high_temperature_trim
            if (object_q8 >> 8) <= 500:
                trim = _signed32((temperature - 0x5D00) * trim) >> 13
            temperature = _signed32(temperature + trim)
        return temperature


@dataclass(frozen=True)
class FrameTemperatureScale:
    """Apply the sensor's valid gray range and the CLI output quantization."""

    minimum_gray: int
    maximum_gray: int
    minimum_temperature_q13: int
    maximum_temperature_q13: int

    @classmethod
    def prepare(cls, calibration, temperature_range, use_calibrated_slope):
        """Derive sensor gray limits using the object correction's radiance bounds."""
        correction = calibration.object_correction
        radiance_offset = correction.radiance_offset_q12
        gain = correction.radiance_gain_q12
        throughput = correction.throughput_q14
        lower = calibration.minimum_q8
        if radiance_offset < 0:
            minimum_radiance = _truncate_division(-radiance_offset << 12, gain)
            root = _lookup_square_root(minimum_radiance)
            kelvin_q12 = _lookup_square_root(root << 6) << 6
            lower = max(lower, _truncate_division(kelvin_q12 - 0x111266, 16))

        saturation = (1 << 50) - 4
        offset = _truncate_division(radiance_offset << 12, throughput)
        maximum_radiance = _truncate_division(saturation - offset, gain) * throughput
        root = _lookup_square_root((maximum_radiance & 0xFFFFFFFFFFFFFFFF) >> 4)
        kelvin_q12 = _lookup_square_root(root << 8) << 8
        upper = min(calibration.maximum_q8, 0x2EED9, (kelvin_q12 - 0x111266) >> 4)

        minimum_gray = calibration.gray_from_blackbody(lower, use_calibrated_slope) + 10
        maximum_gray = calibration.gray_from_blackbody(upper, use_calibrated_slope) - 20
        maximum_temperature = (165, 600)[temperature_range]
        return cls(minimum_gray, maximum_gray, -30 * 8192, maximum_temperature * 8192)

    def temperature(self, raw, calibration):
        if raw < self.minimum_gray:
            q13 = self.minimum_temperature_q13
        elif raw > self.maximum_gray:
            q13 = self.maximum_temperature_q13
        else:
            q6 = calibration.temperature_q6(raw)
            tenths = int(_float32(_float32(q6 / 64) * 10))
            q13 = int(_float32(_float32(tenths / 10) * 8192))
            q13 = min(max(q13, self.minimum_temperature_q13), self.maximum_temperature_q13)
        return _float32(q13 / 8192)
