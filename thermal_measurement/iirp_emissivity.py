"""IIRP emissivity calibration with JSON pre-loaded tables."""

import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

_DATA_PATH = Path(__file__).resolve().parent / "data" / "iirp_tables.json"
_JSON_TABLES = json.loads(_DATA_PATH.read_text(encoding="utf-8"))

_EMISSIVITIES = tuple(value / 1000 for value in range(800, 1000, 2))
_TEMPERATURES = tuple(
    value / 100 for value in (*range(22315, 82316, 1000), *range(87315, 182316, 5000))
)
_COLUMNS = len(_EMISSIVITIES)
_LEGACY_EMISSIVITIES = tuple(value / 100 for value in range(80, 100))
_LEGACY_TEMPERATURES = tuple(value / 100 for value in range(22315, 107316, 1000))


def _float32(value):
    return float(np.float32(value))


_CELLS = tuple(_JSON_TABLES["cells"])
_LEGACY_CELLS = tuple(_JSON_TABLES["legacy_cells"])
_LEGACY_TEMPERATURE_INDEX = tuple(map(_float32, _LEGACY_TEMPERATURES))
_LEGACY_EMISSIVITY_INDEX = tuple(map(_float32, _LEGACY_EMISSIVITIES))
# Both SDK bin searches advance past a cell within 0.0001 of its axis value.
_TEMPERATURE_INDEX = tuple(value - 0.0001 for value in _TEMPERATURES)
_EMISSIVITY_INDEX = tuple(value - 0.0001 for value in _EMISSIVITIES)


def _nearest(axis, value):
    upper = bisect_right(axis, value)
    if upper == 0:
        return 0
    if upper == len(axis) or value - axis[upper - 1] < axis[upper] - value:
        return upper - 1
    return upper


def lookup_emissivity_v1(apparent_celsius, emissivity):
    """Look up the nearest legacy temperature and emissivity bins."""
    emissivity = _float32(emissivity)
    if not _LEGACY_EMISSIVITIES[0] <= emissivity <= _LEGACY_EMISSIVITIES[-1]:
        return emissivity
    kelvin = _float32(_float32(apparent_celsius) + 273.15)
    row = _nearest(_LEGACY_TEMPERATURE_INDEX, kelvin)
    column = _nearest(_LEGACY_EMISSIVITY_INDEX, emissivity)
    return _LEGACY_CELLS[row * len(_LEGACY_EMISSIVITIES) + column]


def lookup_emissivity(apparent_celsius, emissivity, version):
    """Apply the selected IIRP emissivity calibration."""
    emissivity = _float32(emissivity)
    if not 0 < emissivity <= 1:
        raise ValueError("Emissivity must be greater than zero and at most one")
    if version == 0:
        return emissivity
    if version == 1:
        return lookup_emissivity_v1(apparent_celsius, emissivity)
    if version not in (2, 3):
        raise ValueError("Unsupported IIRP emissivity calibration")

    kelvin = _float32(_float32(apparent_celsius) + 273.15)
    row = max(0, bisect_left(_TEMPERATURE_INDEX, kelvin) - 1)
    column = max(0, bisect_left(_EMISSIVITY_INDEX, emissivity) - 1)
    first, last = _EMISSIVITIES[0], _EMISSIVITIES[-1]
    if version == 2 and not first <= emissivity <= last:
        return emissivity
    if emissivity < first:
        edge = _interpolate_column(kelvin, row, 0)
        return _float32(emissivity * (edge / first))
    if emissivity > last:
        edge = _interpolate_column(kelvin, row, _COLUMNS - 1)
        slope = _float32(1 - edge) / (1 - last)
        return _float32(edge + (emissivity - last) * slope)
    if column == _COLUMNS - 1:
        return _interpolate_column(kelvin, row, column)
    if row == len(_TEMPERATURES) - 1:
        e0, e1 = _EMISSIVITIES[column : column + 2]
        a, b = _CELLS[row * _COLUMNS + column : row * _COLUMNS + column + 2]
        return _float32(
            a * ((e1 - emissivity) / (e1 - e0)) + b * ((emissivity - e0) / (e1 - e0))
        )
    t0, t1 = _TEMPERATURES[row : row + 2]
    e0, e1 = _EMISSIVITIES[column : column + 2]
    area = (e1 - e0) * (t1 - t0)
    index = row * _COLUMNS + column
    a, b = _CELLS[index : index + 2]
    c, d = _CELLS[index + _COLUMNS : index + _COLUMNS + 2]
    return _float32(
        a * ((e1 - emissivity) * (t1 - kelvin) / area)
        + b * ((emissivity - e0) * (t1 - kelvin) / area)
        + c * ((e1 - emissivity) * (kelvin - t0) / area)
        + d * ((emissivity - e0) * (kelvin - t0) / area)
    )


def _interpolate_column(kelvin, row, column):
    index = row * _COLUMNS + column
    a = _CELLS[index]
    if row == len(_TEMPERATURES) - 1:
        return a
    t0, t1 = _TEMPERATURES[row : row + 2]
    b = _CELLS[index + _COLUMNS]
    return _float32(a * ((t1 - kelvin) / (t1 - t0)) + b * ((kelvin - t0) / (t1 - t0)))
