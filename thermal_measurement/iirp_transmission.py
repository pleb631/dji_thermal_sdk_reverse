"""Atmospheric transmission lookup for IIRP calibration with JSON tables."""

import json
from bisect import bisect_right
from math import exp, log, pow
from pathlib import Path

import numpy as np

_DATA_PATH = Path(__file__).resolve().parent / "data" / "iirp_tables.json"
_JSON_TABLES = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
_M2EA_DEFAULT_GRID = bytes.fromhex(_JSON_TABLES["m2ea_default_grid_hex"])

_TEMPERATURES = tuple(
    value / 100 for value in (*range(24815, 82316, 2500), *range(87315, 187316, 5000))
)
_DISTANCES = (
    *(value / 100 for value in range(25, 121, 5)),
    *(value / 10 for value in range(13, 21)),
    *(value / 10 for value in range(22, 41, 2)),
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    35,
    40,
    45,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
)
_LEGACY_TEMPERATURES = tuple(value / 100 for value in range(24815, 162316, 2500))


def restore_legacy_m2ea_calibration(calibration):
    """Restore the SDK's fallback grid for an erased legacy M2EA APP5 block."""
    if calibration.startswith(b"\xff" * 7424):
        return calibration[:256] + _M2EA_DEFAULT_GRID
    return calibration


def _float32(value):
    return float(np.float32(value))


def correct_humidity(
    transmission, apparent_celsius, distance, humidity, ambient, version
):
    """Adjust Q14 transmission for relative humidity as a fraction of one."""
    if version not in (1, 2, 3):
        raise ValueError("Unknown IIRP humidity calibration version")
    humidity, ambient, distance = map(_float32, (humidity, ambient, distance))
    if not 0 <= humidity <= 1:
        raise ValueError("Relative humidity must be between zero and one")

    pressure = _float32(611.2 * exp(17.67 * (ambient / (243.5 + ambient))) * humidity)
    pressure = min(max(pressure, 800), 3000 if version == 1 else 4000)
    pressure_ratio = _float32(int(pressure) / 1425)
    kelvin = _float32(_float32(apparent_celsius) + 273.15)
    if kelvin < 273.15 or kelvin >= 1273.1501:
        coefficient = 0.0
    elif kelvin < 410.15:
        coefficient = _float32(0.01 * (pow(410.15 / kelvin, 4) - 1) - 0.04)
    else:
        coefficient = _float32(-0.003 * (pow(kelvin / 410.15, 4) - 1) - 0.04)
    if version != 1:
        coefficient = _float32(coefficient * (0.6 if version == 2 else 0.3))

    distance = max(distance, 1)
    if version != 1:
        distance = min(distance, 10)
    exponent = (1.4427 * log(distance)) * coefficient
    multiplier = _float32(pow(pressure_ratio, exponent))
    return int(_float32(transmission * multiplier)) & 0xFFFF


class TransmissionTable:
    def __init__(self, calibration):
        has_header = calibration[4:8] == b"\xff" * 4
        version = int.from_bytes(calibration[2:4], "little") if has_header else 0
        offset = 256 if has_header else 0
        if version <= 0x3F:
            self._temperatures = _LEGACY_TEMPERATURES
            self._distances = _DISTANCES[:64]
        elif version <= 0x100:
            self._temperatures = _LEGACY_TEMPERATURES[:42]
            self._distances = _DISTANCES
        else:
            self._temperatures = _TEMPERATURES
            self._distances = _DISTANCES

        self._columns = len(self._distances)
        self._temperature_index = tuple(map(_float32, self._temperatures))
        self._distance_index = tuple(map(_float32, self._distances))
        total_elements = len(self._temperatures) * self._columns
        if len(calibration) < offset + total_elements * 2:
            raise ValueError("Truncated IIRP transmission table")
        self._values = np.frombuffer(
            calibration, dtype="<u2", count=total_elements, offset=offset
        )

    def lookup(self, apparent_celsius, distance):
        """Interpolate the transmission represented as an unsigned Q14 value."""
        kelvin = _float32(_float32(apparent_celsius) + 273.15)
        distance = _float32(distance)
        temperatures, distances = self._temperatures, self._distances
        row = max(0, bisect_right(self._temperature_index, kelvin) - 1)
        column = max(0, bisect_right(self._distance_index, distance) - 1)
        index = row * self._columns + column
        end_row = row == len(temperatures) - 1
        end_column = column == self._columns - 1
        a = self._values[index]
        if end_row and end_column:
            return a
        if end_row:
            d0, d1 = distances[column : column + 2]
            # The SDK reads the same terminal-row cell for both weights.
            return int(
                a * ((d1 - distance) / (d1 - d0)) + a * ((distance - d0) / (d1 - d0))
            )
        if end_column:
            t0, t1 = temperatures[row : row + 2]
            c = self._values[index + self._columns]
            return int(
                a * ((t1 - kelvin) / (t1 - t0)) + c * ((kelvin - t0) / (t1 - t0))
            )

        t0, t1 = temperatures[row : row + 2]
        d0, d1 = distances[column : column + 2]
        b = self._values[index + 1]
        c, d = self._values[index + self._columns : index + self._columns + 2]
        area = (t1 - t0) * (d1 - d0)
        return int(
            a * ((t1 - kelvin) * (d1 - distance) / area)
            + b * ((t1 - kelvin) * (distance - d0) / area)
            + c * ((kelvin - t0) * (d1 - distance) / area)
            + d * ((kelvin - t0) * (distance - d0) / area)
            + 0.5
        )
