"""Zenmuse H20T (DIRP R-JPEG) radiometric measurement handler."""

from dataclasses import dataclass

import numpy as np

from .xts_radiance import ThermalArray


@dataclass(frozen=True)
class DirpCalibration:
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

    @classmethod
    def from_header_and_lut(cls, header: bytes, lut: np.ndarray) -> "DirpCalibration":
        params = header[4:100] + b"".join(
            header[offset : offset + 4] for offset in (0x64, 0xA4, 0x88, 0xA8, 0x84)
        )
        zero_code = int(np.frombuffer(header[0x146:0x148], dtype="<u2")[0])
        flux_tenths = int(np.frombuffer(params[0x54:0x56], dtype="<i2")[0])

        baseline = (
            np.float32(flux_tenths / 10.0)
            if flux_tenths >= -2735
            else np.frombuffer(params[0x60:0x64], dtype="<f4")[0]
        )
        _, target, sensor_now, sensor_ref = np.frombuffer(
            params[0x60:0x70], dtype="<f4"
        )
        a, b, c, d, gain, intercept, offset = np.frombuffer(
            params[0x20:0x3C], dtype="<f4"
        )

        def polynomial(val: np.float32) -> np.float32:
            v = np.float32(val - baseline)
            square_term = np.float32(np.float32(b * v) * v)
            return np.float32(np.float32(np.float32(c * v) + square_term) + d)

        first = np.float32(np.float32(np.float32(target - baseline) * a) + intercept)
        second = np.float32(polynomial(sensor_now) - polynomial(sensor_ref))
        reference_temperature = int(
            np.floor(np.float32(target * 10.0) + 0.5)
            if target >= 0
            else np.ceil(np.float32(target * 10.0) - 0.5)
        )

        if reference_temperature >= lut[-1]:
            reference_code = 65532
        else:
            reference_code = next(
                (
                    index * 4
                    for index in range(1, len(lut))
                    if lut[index - 1] < reference_temperature <= lut[index]
                ),
                0,
            )
        if not reference_code:
            raise ValueError("DIRP reference temperature is below the lookup range")

        output_gain = int(np.frombuffer(params[0x50:0x52], dtype="<u2")[0])
        output_offset = int(np.frombuffer(params[0x52:0x54], dtype="<i2")[0])

        return cls(
            reference_code=reference_code,
            zero_code=zero_code,
            gain=np.float32(gain / 79.0),
            offset=offset,
            first_subtraction=first,
            second_subtraction=second,
            divisor=1.0,
            reflected_tenths=230,
            throughput_percent=100,
            output_gain_percent=output_gain,
            output_offset_tenths=output_offset,
        )


class DirpHandler:
    """Object-oriented measurement handler for Zenmuse H20T DIRP thermal images."""

    WIDTH = 640
    HEIGHT = 512
    _instance = None

    @classmethod
    def get_instance(cls) -> "DirpHandler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def can_handle(segments: list[tuple[int, bytes]]) -> bool:
        return any(
            marker == 0xE4 and payload.startswith(b"\xaa\x55\x12\x06")
            for marker, payload in segments
        )

    def measure(
        self, segments: list[tuple[int, bytes]]
    ) -> tuple[ThermalArray, tuple[int, int]]:
        headers = [
            p for m, p in segments if m == 0xE4 and p.startswith(b"\xaa\x55\x12\x06")
        ]
        raw_bytes = b"".join(p for m, p in segments if m == 0xE3)
        lut_bytes = b"".join(p for m, p in segments if m == 0xE5)

        if (
            len(headers) != 1
            or len(headers[0]) != 2560
            or len(raw_bytes) != self.WIDTH * self.HEIGHT * 2
            or len(lut_bytes) != 32768
        ):
            raise ValueError("Unsupported or truncated DIRP frame")

        lut = np.frombuffer(lut_bytes, dtype="<i2")
        raw_pixels = np.frombuffer(raw_bytes, dtype="<u2")
        calibration = DirpCalibration.from_header_and_lut(headers[0], lut)

        # Vectorized DIRP radiance calculation via unique raw code LUT
        unique_raws = np.flatnonzero(np.bincount(raw_pixels))
        code = (
            (unique_raws.astype(np.float32) - calibration.zero_code) * calibration.gain
        ).astype(np.float32)
        code = (code + calibration.reference_code).astype(np.float32)
        code = (code + calibration.offset).astype(np.float32)
        code = (code - calibration.first_subtraction).astype(np.float32)
        code = (code - calibration.second_subtraction).astype(np.float32)
        code = np.maximum((code / calibration.divisor).astype(np.float32), 0.0)

        rounded = np.where(code >= 0, np.floor(code + 0.5), np.ceil(code - 0.5)).astype(
            np.int64
        )
        index = np.where(rounded > 65533, 16383, (rounded + 2) // 4)
        apparent = (lut[index].astype(np.float32) / 10.0).astype(np.float32)
        reflected = np.float32(calibration.reflected_tenths / 10.0)

        apparent_kelvin = (apparent + 273.0).astype(np.float32)
        reflected_kelvin = (reflected + 273.0).astype(np.float32)
        apparent_radiance = (apparent_kelvin**4).astype(np.float32)
        reflected_radiance = (reflected_kelvin**4).astype(np.float32)

        diff = (apparent_radiance - reflected_radiance).astype(np.float32)
        correction = (
            (diff * 100.0).astype(np.float32) / calibration.throughput_percent
        ).astype(np.float32)
        corrected = (reflected_radiance + correction).astype(np.float32)
        kelvin = (np.maximum(corrected, 0.0) ** 0.25).astype(np.float32)
        celsius = (kelvin - 273.0).astype(np.float32)
        scaled = (
            (celsius * calibration.output_gain_percent).astype(np.float32) / 100.0
        ).astype(np.float32)
        t_unique = (
            scaled + np.float32(calibration.output_offset_tenths / 10.0)
        ).astype(np.float32)

        lut_full = np.empty(65536, dtype=np.float32)
        lut_full[unique_raws] = t_unique
        temperatures = lut_full[raw_pixels]

        return temperatures.astype("<f4", copy=False).view(ThermalArray), (self.HEIGHT, self.WIDTH)


def measure_dirp(
    segments: list[tuple[int, bytes]],
) -> tuple[ThermalArray, tuple[int, int]]:
    """Measure Zenmuse H20T R-JPEG segments using DirpHandler."""
    return DirpHandler.get_instance().measure(segments)
