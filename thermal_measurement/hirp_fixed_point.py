"""MicroTA fixed-point inverse response used by HIRP measurement with JSON tables."""

import json
from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np

# Load tables from JSON
_DATA_PATH = Path(__file__).resolve().parent / "data" / "hirp_tables.json"
_JSON_TABLES = json.loads(_DATA_PATH.read_text(encoding="utf-8"))

_RECIPROCAL_BASES = tuple(_JSON_TABLES["reciprocal_bases"])
_RECIPROCAL_SLOPES = tuple(_JSON_TABLES["reciprocal_slopes"])
_LOGARITHMS = tuple(_JSON_TABLES["logarithms"])
_SQRT_VALUES = tuple(_JSON_TABLES["sqrt_values"])
_SQRT64_BASES = tuple(_JSON_TABLES["sqrt64_bases"])
_SQRT64_SLOPES = tuple(_JSON_TABLES["sqrt64_slopes"])


def _signed32(value):
    return ((value + 2147483648) & 0xFFFFFFFF) - 2147483648


def _signed64(value):
    return ((value + (1 << 63)) & 0xFFFFFFFFFFFFFFFF) - (1 << 63)


def _float32(value):
    return float(np.float32(value))


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
    return _signed32(
        _reciprocal_product(_signed32(temperature_scale << 6), denominator)
    )


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
    denominator = int(_float32(scaled_exp / 64)) - (
        slope if use_calibrated_slope else 256
    )
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


def response_gray_offset(
    reference_gray, sensor_gray, gray_response, temperature_delta, gain, slope
):
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
        curve = tuple(
            interpolate(background, 1250, 2000, a, b) for a, b in zip(cold, warm)
        )
    elif background <= 2500:
        curve = tuple(
            interpolate(background, 2000, 2500, a, b) for a, b in zip(warm, hot)
        )
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
        (1 << fractional_bits)
        + value
        + (square >> (fractional_bits + 1))
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
    temperature_curve += (
        10632 * squared * _truncate_division(background, 50) ** 2
    ) >> 32
    humidity_term = _truncate_division(
        temperature_curve * humidity * ((temperature_term * 1022) >> 10), 78125
    )
    humidity_factor = _atmospheric_exponential((-2086 * humidity_term) >> 17, 16)

    thresholds = (3, 8, 16, 32, 80, 160, 320, 800, 1600, 3200)
    bases = (16384, 16357, 16204, 16073, 15439, 14615, 13926, 12900, 11436, 10207)
    slopes = (-78643, -69905, -50332, -41943, -21671, -11115, -6711, -3426, -1552, -765)
    index = next(
        (i for i, end in enumerate(thresholds) if distance_term * 16 <= end * 100000), 9
    )
    dry_base = bases[index] + (
        _truncate_division(((distance_term * 8145) >> 13) * slopes[index], 100000) >> 6
    )
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


# --- Vectorized fixed-point helpers -----------------------------------------
#
# The functions below mirror the scalar routines above elementwise. They are
# used to fill the per-frame LUT over the observed raw code span, which avoids a
# Python loop per distinct code. Overflow semantics intentionally match Python's
# explicit 32/64-bit wrapping masks.

_RECIPROCAL_BASES_NP = np.array(_RECIPROCAL_BASES, dtype=np.int64)
_RECIPROCAL_SLOPES_NP = np.array(_RECIPROCAL_SLOPES, dtype=np.int64)
_LOGARITHMS_NP = np.array(_LOGARITHMS, dtype=np.int64)
_SQRT_VALUES_NP = np.array(_SQRT_VALUES, dtype=np.int64)
_SQRT64_BASES_NP = np.array(_SQRT64_BASES, dtype=np.int64)
_SQRT64_SLOPES_NP = np.array(_SQRT64_SLOPES, dtype=np.int64)
_MASK32 = np.uint64(0xFFFFFFFF)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


class _VectorFallback(Exception):
    """Raised when the vectorized path cannot reproduce the scalar semantics."""


def _f32_array(values):
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def _signed32_array(values):
    low = values.astype(np.int64) & 0xFFFFFFFF
    return (low ^ 0x80000000) - 0x80000000


def _bit_length_array(values):
    """floor(log2(value)) for an array of positive int64 values."""
    work = values.astype(np.int64, copy=True)
    highest = np.zeros(work.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        big = work >= (1 << shift)
        highest += np.where(big, shift, 0)
        work = np.where(big, np.right_shift(work, shift), work)
    return highest


def _shift_pair(values, left, right):
    """``(values << left) >> right`` for nonnegative integer arrays."""
    delta = left - right
    positive = delta >= 0
    return np.where(
        positive,
        np.left_shift(values, np.where(positive, delta, 0)),
        np.right_shift(values, np.where(positive, 0, -delta)),
    )


def _square_root_64_array(values):
    values = np.asarray(values, dtype=np.int64)
    if np.any(values <= 0):
        raise _VectorFallback
    highest_bit = _bit_length_array(values)
    wrapped = (values.astype(np.uint64) << np.uint64(20)).view(np.int64)
    divisor = np.exp2((highest_bit + 1).astype(np.float64))
    normalized = np.trunc(wrapped.astype(np.float64) / divisor).astype(np.int64)
    bin_number = ((normalized - 0x7F000) >> 12) & 0xFF
    if np.any((bin_number < 1) | (bin_number > 128)):
        raise _VectorFallback
    midpoint = (bin_number << 12) + 0x7F800
    slope = _SQRT64_SLOPES_NP[bin_number - 1]
    root = np.right_shift((normalized - midpoint) * slope, 17)
    root = root + _SQRT64_BASES_NP[bin_number - 1] * 8
    scaled = normalized << 22
    threshold = root * root + root + 1
    root = root + np.where(
        scaled > threshold + root * 2 + 2, 2, np.where(scaled > threshold, 1, 0)
    )
    odd = ((highest_bit + 1) & 1).astype(bool)
    odd_result = np.right_shift(root * 46341, 36 - (highest_bit >> 1))
    even_exponent = (highest_bit + 1) >> 1
    even_result = _shift_pair(root, even_exponent, 21)
    return np.where(odd, odd_result, even_result)


def _log_mantissa_array(values, highest_bit):
    shift = highest_bit - 14
    normalized = np.where(
        shift >= 0,
        np.right_shift(values, np.where(shift >= 0, shift, 0)),
        np.left_shift(values, np.where(shift < 0, -shift, 0)),
    )
    return _LOGARITHMS_NP[(normalized - 16384) >> 4]


def _reciprocal_product_array(numerator, denominator):
    magnitude = np.abs(denominator)
    exponent = _bit_length_array(magnitude)
    normalized = np.right_shift(magnitude << 15, exponent)
    index = (normalized - 32768) >> 7
    midpoint = 32832 + (index << 7)
    reciprocal = _RECIPROCAL_BASES_NP[index] + np.right_shift(
        _RECIPROCAL_SLOPES_NP[index] * (midpoint - normalized), 15
    )
    refined = np.right_shift((2147483648 - normalized * reciprocal) * reciprocal, 15)
    result = np.right_shift(refined * np.abs(numerator), exponent + 30)
    negative = (numerator < 0) != (denominator < 0)
    return np.where(negative, -result, result)


def inverse_sensor_response_array(grays, coefficients):
    """Array form of :func:`inverse_sensor_response`."""
    base, temperature_scale, response_floor, response_slope = coefficients
    grays = np.asarray(grays, dtype=np.int64)
    scaled_gray = _signed32_array(grays << 8)
    if np.any(scaled_gray <= response_floor):
        raise _VectorFallback
    response = _signed32_array(scaled_gray - response_floor)
    floor_product = (
        np.uint64(response_floor & 0xFFFFFFFF) * np.uint64(response_slope & 0xFFFFFFFF)
    )
    floor_term = _signed32_array((floor_product & _MASK32).astype(np.int64))
    adjusted_base = _signed32_array(base - (floor_term >> 8))
    gray_product = (grays.astype(np.uint64) & _MASK32) * np.uint64(
        response_slope & 0xFFFFFFFF
    )
    adjusted_base = adjusted_base + _signed32_array(
        (gray_product & _MASK32).astype(np.int64)
    )
    if np.any(adjusted_base <= 0) or np.any(adjusted_base < response):
        raise _VectorFallback

    base_bit = _bit_length_array(adjusted_base)
    response_bit = _bit_length_array(response)
    logarithm = (base_bit - response_bit) * 11356
    logarithm = logarithm + _log_mantissa_array(adjusted_base, base_bit)
    logarithm = logarithm - _log_mantissa_array(response, response_bit)
    denominator = logarithm >> 2
    if np.any(denominator == 0):
        raise _VectorFallback
    return _signed32_array(
        _reciprocal_product_array(_signed32(temperature_scale << 6), denominator)
    )


def _fourth_power_q12_array(value):
    if np.any(np.abs(value) >= (1 << 21)):
        raise _VectorFallback
    squared = np.right_shift(value * value, 12)
    return np.right_shift(squared * squared, 12)


def _lookup_square_root_array(values):
    values = np.asarray(values, dtype=np.int64)
    if np.any(values <= 0):
        raise _VectorFallback
    highest_bit = _bit_length_array(values)
    scaled = values.astype(np.uint64) << np.uint64(12)
    high = (scaled >> np.uint64(63)) != 0
    numerator = np.where(
        high,
        (scaled >> np.uint64(1)).astype(np.float64) * 2.0,
        scaled.astype(np.float64),
    )
    divisor = np.exp2((highest_bit + 1).astype(np.float64))
    normalized = np.trunc(numerator / divisor).astype(np.int64)
    index = (((normalized + 0x3FF800) << 15) & 0xFFFFFFFF) >> 16
    entry = _SQRT_VALUES_NP[index]
    odd = ((highest_bit + 1) & 1).astype(bool)
    odd_result = _shift_pair(entry * 11585, highest_bit >> 1, 26)
    even_result = _shift_pair(entry, (highest_bit + 1) >> 1, 13)
    return np.where(odd, odd_result, even_result)


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
        background_kelvin = (
            _truncate_division(_signed32(background << 12), 50) + 0x111266
        )
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

    def apply_array(self, blackbody_q12):
        """Array form of :meth:`apply`."""
        shifted = _signed32_array(np.asarray(blackbody_q12, dtype=np.int64) + 0x111266)
        radiance = np.right_shift(
            _fourth_power_q12_array(shifted) * self.radiance_gain_q12, 12
        )
        radiance = radiance + self.radiance_offset_q12
        if np.any(
            ((radiance.astype(np.uint64) << np.uint64(11)) & _MASK64) == 0
        ):
            raise _VectorFallback
        gain = (1 << 26) // self.throughput_q14
        adjusted = np.right_shift(
            radiance.astype(np.uint64) * np.uint64(gain), np.uint64(12)
        ).astype(np.int64)
        root = _lookup_square_root_array(adjusted)
        kelvin_q8 = np.left_shift(_lookup_square_root_array(np.left_shift(root, 6)), 2)
        return _signed32_array(kelvin_q8 - 0x11126)


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
        gray = blackbody_gray(
            self.inverse_coefficients, temperature, use_calibrated_slope
        )
        corrected = (
            _signed64(gray * (self.linear_response + gray * self.quadratic_response))
            >> 32
        )
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

    def response_array(self, raws):
        """Array form of :meth:`response`."""
        raws = np.asarray(raws, dtype=np.int64)
        gray = _signed32_array(raws - self.gray_offset)
        if self.quadratic_response >> 9 == 0:
            multiplier = int(float(1 << 46) / self.linear_response)
            product = (gray.astype(np.uint64) & _MASK32) * np.uint64(
                multiplier & 0xFFFFFFFF
            )
            return (
                _signed32_array((product & _MASK32).astype(np.int64)) >> 14
            )

        linear = self.linear_response >> 5
        discriminant = gray * (self.quadratic_response << 2)
        deterministic = linear * linear
        deterministic = ((deterministic + (1 << 63)) & 0xFFFFFFFFFFFFFFFF) - (1 << 63)
        discriminant = discriminant + (np.int64(deterministic) >> 22)
        root = _square_root_64_array(discriminant << 8)
        multiplier = int(float(1 << 32) / self.quadratic_response)
        corrected = ((root << 12) - self.linear_response) * multiplier
        return _signed32_array(corrected >> 33)

    def temperature_q6(self, raw):
        """Convert a raw gray code to the native per-pixel Q6 temperature."""
        kelvin_q8 = inverse_sensor_response(
            self.response(raw), self.inverse_coefficients
        )
        blackbody_q8 = min(max(kelvin_q8 - 0x11126, self.minimum_q8), self.maximum_q8)
        object_q8 = self.object_correction.apply(blackbody_q8 << 4)
        if self.trim_gain_q23 is not None:
            object_q8 = _signed32(
                ((_signed64(object_q8 * self.trim_gain_q23)) >> 23)
                + self.trim_offset_q8
            )
        temperature = object_q8 >> 2
        if (object_q8 >> 8) > 371:
            trim = self.high_temperature_trim
            if (object_q8 >> 8) <= 500:
                trim = _signed32((temperature - 0x5D00) * trim) >> 13
            temperature = _signed32(temperature + trim)
        return temperature

    def temperature_q6_array(self, raws):
        """Array form of :meth:`temperature_q6`."""
        kelvin_q8 = inverse_sensor_response_array(
            self.response_array(raws), self.inverse_coefficients
        )
        blackbody_q8 = np.clip(
            kelvin_q8 - 0x11126, self.minimum_q8, self.maximum_q8
        )
        object_q8 = self.object_correction.apply_array(blackbody_q8 << 4)
        if self.trim_gain_q23 is not None:
            object_q8 = _signed32_array(
                np.right_shift(object_q8 * self.trim_gain_q23, 23)
                + self.trim_offset_q8
            )
        temperature = object_q8 >> 2
        high = (object_q8 >> 8) > 371
        middle = (object_q8 >> 8) <= 500
        trimmed = _signed32_array(
            (temperature - 0x5D00) * self.high_temperature_trim
        ) >> 13
        offset = np.where(middle, trimmed, self.high_temperature_trim)
        temperature = np.where(
            high, _signed32_array(temperature + offset), temperature
        )
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
            q13 = min(
                max(q13, self.minimum_temperature_q13), self.maximum_temperature_q13
            )
        return _float32(q13 / 8192)

    def temperature_array(self, raws, calibration):
        """Array form of :meth:`temperature`."""
        raws = np.asarray(raws, dtype=np.int64)
        below = raws < self.minimum_gray
        above = raws > self.maximum_gray
        inside = ~(below | above)
        values = np.empty(raws.shape, dtype=np.float32)
        if np.any(inside):
            q6 = calibration.temperature_q6_array(raws[inside])
            tenths = np.trunc(
                _f32_array(_f32_array(q6.astype(np.float64) / 64.0) * 10.0)
            ).astype(np.int64)
            q13 = np.trunc(
                _f32_array(_f32_array(tenths.astype(np.float64) / 10.0) * 8192.0)
            ).astype(np.int64)
            q13 = np.clip(
                q13, self.minimum_temperature_q13, self.maximum_temperature_q13
            )
            values[inside] = (q13.astype(np.float64) / 8192.0).astype(np.float32)
        values[below] = np.float32(self.minimum_temperature_q13 / 8192)
        values[above] = np.float32(self.maximum_temperature_q13 / 8192)
        return values
