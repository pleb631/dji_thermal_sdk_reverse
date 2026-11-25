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

_CELLS_NP = np.array(_CELLS, dtype=np.float64)
_LEGACY_CELLS_NP = np.array(_LEGACY_CELLS, dtype=np.float64)
_LEGACY_TEMPERATURE_INDEX_NP = np.array(_LEGACY_TEMPERATURE_INDEX, dtype=np.float64)
_TEMPERATURE_INDEX_NP = np.array(_TEMPERATURE_INDEX, dtype=np.float64)
_TEMPERATURES_NP = np.array(_TEMPERATURES, dtype=np.float64)


def _f32_array(values):
    """Round an array to float32 precision, keeping float64 storage."""
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def _interpolate_column_array(kelvin, row, column):
    index = row * _COLUMNS + column
    a = _CELLS_NP[index]
    last_row = len(_TEMPERATURES) - 1
    next_row = np.minimum(row + 1, last_row)
    b = _CELLS_NP[next_row * _COLUMNS + column]
    t0 = _TEMPERATURES_NP[row]
    t1 = _TEMPERATURES_NP[next_row]
    with np.errstate(divide="ignore", invalid="ignore"):
        interp = a * ((t1 - kelvin) / (t1 - t0)) + b * ((kelvin - t0) / (t1 - t0))
    interp = interp.astype(np.float32).astype(np.float64)
    return np.where(row == last_row, a, interp)


def lookup_emissivity_array(apparent_celsius, emissivity, version):
    """Vectorized form of :func:`lookup_emissivity` for a batch of temperatures."""
    emissivity = _float32(emissivity)
    if not 0 < emissivity <= 1:
        raise ValueError("Emissivity must be greater than zero and at most one")
    apparent = np.asarray(apparent_celsius, dtype=np.float64)
    if version == 0:
        return np.full(apparent.shape, emissivity, dtype=np.float64)
    if version == 1:
        if not _LEGACY_EMISSIVITIES[0] <= emissivity <= _LEGACY_EMISSIVITIES[-1]:
            return np.full(apparent.shape, emissivity, dtype=np.float64)
        kelvin = _f32_array(_f32_array(apparent) + 273.15)
        axis = _LEGACY_TEMPERATURE_INDEX_NP
        upper = np.searchsorted(axis, kelvin, side="right")
        previous = axis[np.maximum(upper - 1, 0)]
        following = axis[np.minimum(upper, len(axis) - 1)]
        take_previous = (upper == len(axis)) | (
            (kelvin - previous) < (following - kelvin)
        )
        row = np.where(
            upper == 0, 0, np.where(take_previous, upper - 1, upper)
        )
        column = _nearest(_LEGACY_EMISSIVITY_INDEX, emissivity)
        return _LEGACY_CELLS_NP[row * len(_LEGACY_EMISSIVITIES) + column]
    if version not in (2, 3):
        raise ValueError("Unsupported IIRP emissivity calibration")

    kelvin = _f32_array(_f32_array(apparent) + 273.15)
    row = np.searchsorted(_TEMPERATURE_INDEX_NP, kelvin, side="left") - 1
    np.maximum(row, 0, out=row)
    column = max(0, bisect_left(_EMISSIVITY_INDEX, emissivity) - 1)
    first, last = _EMISSIVITIES[0], _EMISSIVITIES[-1]
    if version == 2 and not first <= emissivity <= last:
        return np.full(apparent.shape, emissivity, dtype=np.float64)
    if emissivity < first:
        edge = _interpolate_column_array(kelvin, row, 0)
        return emissivity * (edge / first)
    if emissivity > last:
        edge = _interpolate_column_array(kelvin, row, _COLUMNS - 1)
        slope = _f32_array(1 - edge) / (1 - last)
        return edge + (emissivity - last) * slope
    if column == _COLUMNS - 1:
        return _interpolate_column_array(kelvin, row, column)

    last_row = len(_TEMPERATURES) - 1
    next_row = np.minimum(row + 1, last_row)
    t0 = _TEMPERATURES_NP[row]
    t1 = _TEMPERATURES_NP[next_row]
    e0, e1 = _EMISSIVITIES[column : column + 2]
    area = (e1 - e0) * (t1 - t0)
    index = row * _COLUMNS + column
    next_index = next_row * _COLUMNS + column
    a = _CELLS_NP[index]
    b = _CELLS_NP[index + 1]
    c = _CELLS_NP[next_index]
    d = _CELLS_NP[next_index + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        bilinear = (
            a * ((e1 - emissivity) * (t1 - kelvin) / area)
            + b * ((emissivity - e0) * (t1 - kelvin) / area)
            + c * ((e1 - emissivity) * (kelvin - t0) / area)
            + d * ((emissivity - e0) * (kelvin - t0) / area)
        )
        terminal = a * ((e1 - emissivity) / (e1 - e0)) + b * (
            (emissivity - e0) / (e1 - e0)
        )
    return np.where(row == last_row, terminal, bilinear)


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
