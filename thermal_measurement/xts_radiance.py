"""Pure Python and NumPy measurement for the Zenmuse XT S Plug R-JPEG layout."""

from dataclasses import dataclass

import numpy as np


class ThermalArray(np.ndarray):
    """NumPy ndarray subclass providing array.array backward-compatibility attributes."""

    @property
    def typecode(self) -> str:
        return "f"


@dataclass(frozen=True)
class XtsCalibration:
    raw_scale: float
    raw_offset: int
    reference_index: int
    lut_offset: int
    humidity_scale: float
    temperature_offset: int

    @classmethod
    def from_plug_and_lut(
        cls, plug_payload: bytes, lut: np.ndarray
    ) -> "XtsCalibration":
        plug_payload = bytes(plug_payload)
        calibration = plug_payload[1178:1198] + plug_payload[1200:1260]
        words = np.frombuffer(calibration, dtype="<u2")
        signed_words = np.frombuffer(calibration, dtype="<i2")

        def unsigned(offset: int) -> int:
            return int(words[offset // 2])

        def signed(offset: int) -> int:
            return int(signed_words[offset // 2])

        initial_sensor = np.float32(unsigned(0x30) / 100.0)
        current_sensor = np.float32(unsigned(0x2C) / 100.0)
        target = np.float32(unsigned(0x40) / 100.0)
        baseline = np.float32(unsigned(0x4E) / 100.0)

        drift = int(
            (signed(0x0E) / 1000.0 * target) * target + signed(0x10) / 1000.0 * target
        )
        drift = (drift + unsigned(0x22)) & 0xFFFF
        sensor_delta = np.float32(current_sensor - initial_sensor)
        sensor_offset = (
            int(np.float32(signed(0x0A) + np.float32(signed(0x0C) * sensor_delta)))
            & 0xFFFF
        )

        delta = np.float32(target - baseline)
        quadratic = int(np.frombuffer(calibration[0x3C:0x40], dtype="<i4")[0])
        gain = signed(0x38) / 1000.0
        gain += np.float32(signed(0x3A) * delta) / 10000.0
        gain += np.float32(np.float32(delta * delta) * np.float32(quadratic)) / 10000.0
        gain = int(np.int16(int(gain * abs(signed(0x4A)))))
        if target > current_sensor:
            gain -= 350
        thermal_offset = int((gain / 10.0) * np.float32(current_sensor - target))

        reference = np.float32(target * 10.0)
        reference_index = int(np.searchsorted(lut, reference))
        raw_scale = signed(0x08) / 100.0
        raw_offset = drift - sensor_offset + thermal_offset - unsigned(0x36)
        lut_offset = signed(0x4C)
        temperature_offset = int(signed(0x12) / 10.0) + signed(0x1E)
        humidity_scale = np.float32(1.0 - (70.0 - 60.0) / 5000.0)

        return cls(
            raw_scale=raw_scale,
            raw_offset=raw_offset,
            reference_index=reference_index,
            lut_offset=lut_offset,
            humidity_scale=humidity_scale,
            temperature_offset=temperature_offset,
        )


class XtsHandler:
    """Object-oriented measurement handler for Zenmuse XT S R-JPEG frames."""

    WIDTH = 640
    HEIGHT = 512
    _instance = None

    def __init__(self):
        self._lut = np.empty(1 << 16, dtype=np.float32)

    @classmethod
    def get_instance(cls) -> "XtsHandler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def can_handle(segments: list[tuple[int, bytes | memoryview]]) -> bool:
        return any(
            marker == 0xE2 and payload[:4] == b"Plug"
            for marker, payload in segments
        )

    def measure(
        self, segments: list[tuple[int, bytes | memoryview]]
    ) -> tuple[ThermalArray, tuple[int, int]]:
        plugs = [
            data
            for marker, data in segments
            if marker == 0xE2 and data[:4] == b"Plug"
        ]
        raw_bytes = b"".join(data for marker, data in segments if marker == 0xE4)
        table_bytes = b"".join(data for marker, data in segments if marker == 0xE3)

        if (
            len(plugs) != 1
            or len(plugs[0]) != 1260
            or len(raw_bytes) != self.WIDTH * self.HEIGHT * 2
            or len(table_bytes) != 32768
        ):
            raise ValueError("Unsupported or truncated XT S R-JPEG")

        lut = np.frombuffer(table_bytes, dtype="<i2")
        raw_pixels = np.frombuffer(raw_bytes, dtype="<u2")
        calibration = XtsCalibration.from_plug_and_lut(plugs[0], lut)

        # Vectorized temperature mapping over the observed raw code span
        low = int(raw_pixels.min())
        high = int(raw_pixels.max())
        codes = np.arange(low, high + 1, dtype=np.uint16)
        adjusted = (codes.astype(np.int64) + calibration.raw_offset).astype(
            np.int16
        )
        index_f = calibration.raw_scale * adjusted.astype(np.float64) + (
            calibration.reference_index + calibration.lut_offset
        )
        index = index_f.astype(np.int32).astype(np.int16)
        index = (
            (index.astype(np.float32) * calibration.humidity_scale)
            .astype(np.int32)
            .astype(np.int16)
        )
        index = np.clip(index, 0, len(lut) - 1)
        t_span = (
            lut[index].astype(np.int32) + calibration.temperature_offset
        ).astype(np.int16).astype(np.float32) / 10.0

        lut_full = self._lut
        lut_full[low : high + 1] = t_span
        temperatures = lut_full[raw_pixels]

        return temperatures.astype("<f4", copy=False).view(ThermalArray), (self.HEIGHT, self.WIDTH)


def measure_xts(
    segments: list[tuple[int, bytes | memoryview]],
) -> tuple[ThermalArray, tuple[int, int]]:
    """Measure Zenmuse XT S R-JPEG segments using XtsHandler."""
    return XtsHandler.get_instance().measure(segments)
