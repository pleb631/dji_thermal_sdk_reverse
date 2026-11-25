"""Pure Python and NumPy measurement for the GIRP R-JPEG format with JSON LUT."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .xts_radiance import ThermalArray

_TABLES_PATH = Path(__file__).resolve().parent / "data" / "girp_tables.json"
_LOW_TABLES = {
    1: "LOW_SHARED",
    2: "LOW_SHARED",
    3: "LOW_MODE_3",
    10: "LOW_SHARED",
    11: "LOW_MODE_11",
}
_HIGH_TABLES = {
    1: "HIGH_MODE_1",
    2: "HIGH_SHARED",
    3: "HIGH_MODE_3",
    10: "HIGH_SHARED",
    11: "HIGH_MODE_11",
}


def _trunc_div(v: int, d: int) -> int:
    return int(v // d if v >= 0 else -(-v // d))


@dataclass(frozen=True)
class GirpCalibration:
    index_gain: int
    index_offset: int
    temperature_bias: int
    emissivity: int
    reflected_radiance: float
    lut: np.ndarray

    @classmethod
    def from_calibration_payload(
        cls, calibration: bytes, tables: dict[str, np.ndarray]
    ) -> "GirpCalibration":
        humidity, distance, emissivity = (
            int(x) for x in np.frombuffer(calibration, dtype="<u2", count=3, offset=4)
        )
        ambient = int(np.frombuffer(calibration, dtype="<i2", count=1, offset=10)[0])
        if not (0 < emissivity <= 100 and humidity <= 100 and 5 <= distance <= 5000):
            raise ValueError("Unsupported GIRP measurement parameters")
        emissivity = max(50, emissivity)

        high_range = int.from_bytes(calibration[0x24:0x26], "little")
        temperature_bias = int.from_bytes(calibration[0x16:0x18], "little", signed=True)
        mode = calibration[0xE8]

        table_map = _HIGH_TABLES if high_range else _LOW_TABLES
        lut_name = table_map.get(mode, table_map[1])
        lut = tables[lut_name]

        a, b = (
            int(x)
            for x in np.frombuffer(calibration, dtype="<i4", count=2, offset=0xF0)
        )
        gain = int.from_bytes(calibration[0xF8:0xFA], "little")
        c, d = (
            int(x)
            for x in np.frombuffer(calibration, dtype="<i4", count=2, offset=0xFA)
        )
        offset = int.from_bytes(calibration[0x102:0x104], "little")
        distance_gain = int.from_bytes(calibration[0x104:0x106], "little")
        distance_offset, fixed_offset = (
            int(x)
            for x in np.frombuffer(calibration, dtype="<i4", count=2, offset=0x106)
        )

        distance_delta = distance - 50
        gain_poly = int(np.int32((a * distance_delta + b) * distance_delta))
        offset_poly = int(np.int32((c * distance_delta + d) * distance_delta))
        gain = (gain + _trunc_div(gain_poly, 10000)) & 0xFFFF
        offset = (offset + _trunc_div(offset_poly, 10000)) & 0xFFFF
        humidity_gain = _trunc_div((5060 - humidity) * gain, 5000) & 0xFFFF
        index_gain = _trunc_div(humidity_gain * distance_gain, 1000) & 0xFFFF
        index_offset = _trunc_div(int(np.int32(humidity_gain * distance_offset)), 1000)
        index_offset = (index_offset + fixed_offset + offset) & 0xFFFF

        reflected = ambient if abs(ambient - 230) > 20 else 230
        reflected_radiance = ((reflected + 2735) / 10.0) ** 4

        return cls(
            index_gain=index_gain,
            index_offset=index_offset,
            temperature_bias=temperature_bias,
            emissivity=emissivity,
            reflected_radiance=reflected_radiance,
            lut=lut,
        )


class GirpHandler:
    """Object-oriented measurement handler for Matrice 30T GIRP R-JPEG images."""

    WIDTH = 640
    HEIGHT = 512
    _instance = None
    _tables: dict[str, np.ndarray] | None = None

    @classmethod
    def get_instance(cls) -> "GirpHandler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._lut = np.empty(1 << 16, dtype=np.float32)
        if GirpHandler._tables is None:
            GirpHandler._load_tables()

    @classmethod
    def _load_tables(cls):
        """Explicitly load pre-calculated lookup tables from JSON."""
        data = json.loads(_TABLES_PATH.read_text(encoding="utf-8"))
        cls._tables = {k: np.array(v, dtype=np.int16) for k, v in data.items()}

    @staticmethod
    def can_handle(segments: list[tuple[int, bytes | memoryview]]) -> bool:
        return any(
            marker == 0xE4 and payload[:8] == b"\xff\xd2\xd1\xffgirp"
            for marker, payload in segments
        )

    def measure(
        self, segments: list[tuple[int, bytes | memoryview]]
    ) -> tuple[ThermalArray, tuple[int, int]]:
        headers = [
            p
            for m, p in segments
            if m == 0xE4 and p[:8] == b"\xff\xd2\xd1\xffgirp"
        ]
        raw_bytes = b"".join(p for m, p in segments if m == 0xE3)

        if (
            len(headers) != 1
            or len(headers[0]) < 302
            or len(raw_bytes) != self.WIDTH * self.HEIGHT * 2
        ):
            raise ValueError("Unsupported or truncated GIRP frame")

        calibration_payload = headers[0][32:]
        if calibration_payload[:2] != b"\xaa\x55":
            raise ValueError("Invalid GIRP calibration header")

        calib = GirpCalibration.from_calibration_payload(
            calibration_payload, self._tables
        )
        raw_pixels = np.frombuffer(raw_bytes, dtype="<i2")

        # Vectorized GIRP temperature calculation over the observed raw code span
        raw_words = raw_pixels.view(np.uint16)
        low = int(raw_words.min())
        high = int(raw_words.max())
        codes = np.arange(low, high + 1, dtype=np.uint16)
        unique_raws = codes.view("<i2").astype(np.int64)
        prod = unique_raws * calib.index_gain
        div = np.where(prod >= 0, prod // 8192, -(-prod // 8192))
        index_raw = div + calib.index_offset
        index = ((index_raw + 32768) & 0xFFFF) - 32768
        clipped_index = np.clip(index, 0, 16383)
        temperature = (
            (calib.lut[clipped_index].astype(np.int64) + calib.temperature_bias + 32768)
            & 0xFFFF
        ) - 32768

        if calib.emissivity < 100:
            apparent = ((temperature.astype(np.float64) + 2735.0) / 10.0) ** 4
            corrected = (
                (apparent - calib.reflected_radiance) * 100.0 / (calib.emissivity + 1)
            )
            kelvin_tenths = (
                np.maximum(corrected + calib.reflected_radiance, 0.0) ** 0.25
            ) * 10.0
            temperature = (
                np.trunc(kelvin_tenths - 2735.0).astype(np.int64) + 32768 & 0xFFFF
            ) - 32768

        correction = np.zeros_like(temperature, dtype=np.float32)
        if calib.emissivity == 95:
            mask = temperature > 1550
            t_m = temperature[mask].astype(np.float32)
            poly = (t_m * t_m).astype(np.float32) + (t_m * 21148.0).astype(np.float32)
            correction[mask] = (poly / 1000000.0).astype(np.float32) - 12.0

        scaled = ((temperature.astype(np.float32) + correction) * 100.0).astype(
            np.float32
        )
        t_unique = ((scaled / 100.0).astype(np.float32) / 10.0).astype(np.float32)

        lut_full = self._lut
        lut_full[codes] = t_unique
        temperatures = lut_full[raw_words]

        return temperatures.astype("<f4", copy=False).view(ThermalArray), (self.HEIGHT, self.WIDTH)


def measure_girp(
    segments: list[tuple[int, bytes | memoryview]],
) -> tuple[ThermalArray, tuple[int, int]]:
    """Measure Matrice 30T GIRP R-JPEG segments using GirpHandler."""
    return GirpHandler.get_instance().measure(segments)
