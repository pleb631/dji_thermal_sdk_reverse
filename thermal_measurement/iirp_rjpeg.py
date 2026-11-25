"""SDK-free decoding and NumPy-vectorized measurement of the DJI IIRP R-JPEG layout."""

from dataclasses import dataclass
from math import pow, sqrt
from collections.abc import Generator

import numpy as np

from .iirp_emissivity import lookup_emissivity
from .iirp_transmission import (
    TransmissionTable,
    prepare_humidity_correction,
    restore_legacy_m2ea_calibration,
)
from .xts_radiance import ThermalArray

_KELVIN_OFFSET = 273.15
_TRANSMISSION_SCALE = 16384


def _float32(value: float) -> float:
    return float(np.float32(value))


def correct_temperature(
    apparent: float,
    emissivity: float,
    transmission: int,
    ambient: float,
    reflected: float | None = None,
) -> float:
    """Apply the IIRP radiance correction using a Q14 transmission value."""
    apparent = _float32(apparent)
    emissivity = _float32(emissivity)
    ambient = _float32(ambient)
    received = pow(_float32(apparent + _KELVIN_OFFSET), 4)
    air = pow(ambient + _KELVIN_OFFSET, 4)
    reflected_radiance = (
        None if reflected is None else pow(_float32(reflected) + _KELVIN_OFFSET, 4)
    )
    return _correct_radiance(received, emissivity, transmission, air, reflected_radiance)


def _correct_radiance(
    received: float,
    emissivity: float,
    transmission: int,
    air: float,
    reflected_radiance: float | None,
) -> float:
    if not 0 < transmission <= 0xFFFF or not 0 < emissivity <= 1:
        raise ValueError("Invalid IIRP transmission or emissivity")

    throughput = transmission * emissivity
    if reflected_radiance is None:
        received -= air * (1.0 - throughput / _TRANSMISSION_SCALE)
    else:
        reflected_weight = (1.0 - emissivity) * transmission
        received -= reflected_radiance * reflected_weight / _TRANSMISSION_SCALE
        received -= air * (1.0 - transmission / _TRANSMISSION_SCALE)

    divisor = _float32(_float32(throughput) / _TRANSMISSION_SCALE)
    kelvin = _float32(sqrt(sqrt(max(received, 0.0) / divisor)))
    return _float32(kelvin - _KELVIN_OFFSET)


@dataclass
class IirpFrame:
    width: int
    height: int
    raw_values: np.ndarray
    measurement_fields: tuple
    calibration_payload: bytes


def _jpeg_segments(data: bytes) -> Generator[tuple[int, bytes], None, None]:
    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("Expected a JPEG start marker")
    offset = 2
    length_data = len(data)
    while offset < length_data:
        if data[offset] != 0xFF:
            raise ValueError("Invalid JPEG marker")
        while offset < length_data and data[offset] == 0xFF:
            offset += 1
        if offset == length_data:
            break
        marker = data[offset]
        offset += 1
        if marker == 0xDA:
            return
        if marker == 0xD9 or offset + 2 > length_data:
            break
        length = (data[offset] << 8) | data[offset + 1]
        end = offset + length
        if length < 2 or end > length_data:
            raise ValueError("Invalid JPEG segment length")
        yield marker, data[offset + 2 : end]
        offset = end
    raise ValueError("Missing JPEG start of scan")


class IirpHandler:
    """Object-oriented measurement handler for DJI IIRP R-JPEG frames."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "IirpHandler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def can_handle(segments: list[tuple[int, bytes]]) -> bool:
        is_xts = any(
            marker == 0xE2 and payload.startswith(b"Plug")
            for marker, payload in segments
        )
        is_girp = any(
            marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffgirp")
            for marker, payload in segments
        )
        is_hirp = any(
            marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffhirp")
            for marker, payload in segments
        )
        is_dirp = any(
            marker == 0xE4 and payload.startswith(b"\xaa\x55\x12\x06")
            for marker, payload in segments
        )
        return not (is_xts or is_girp or is_hirp or is_dirp)

    def decode_frame(
        self, data: bytes, segments: list[tuple[int, bytes]] | None = None
    ) -> IirpFrame:
        if segments is None:
            segments = list(_jpeg_segments(data))

        sample_chunks = []
        header = None
        legacy_header = None
        calibration = bytearray()

        for marker, payload in segments:
            if marker == 0xE3:
                sample_chunks.append(payload)
            elif marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffiirp"):
                if header is not None:
                    raise ValueError("Multiple IIRP headers")
                header = payload
            elif (
                marker == 0xE4
                and len(payload) == 224
                and payload[101:109] == b"Mini_640"
            ):
                if legacy_header is not None:
                    raise ValueError("Multiple legacy IIRP headers")
                legacy_header = payload
            elif marker == 0xE5:
                calibration.extend(payload)

        if header is not None and len(header) >= 161:
            width, height = (
                int(x) for x in np.frombuffer(header, dtype="<u2", count=2, offset=157)
            )
            measurement_fields = tuple(
                float(x) for x in np.frombuffer(header, dtype="<f4", count=5, offset=32)
            )
        elif header is None and legacy_header is not None:
            width, height = 640, 512
            measurement_fields = tuple(
                float(x)
                for x in np.frombuffer(legacy_header, dtype="<f4", count=5, offset=0)
            )
            calibration = restore_legacy_m2ea_calibration(calibration)
        else:
            raise ValueError("Missing or truncated IIRP header")

        samples = b"".join(sample_chunks)
        if width == 0 or height == 0 or len(samples) != 2 * width * height:
            raise ValueError("IIRP dimensions do not match the thermal payload")

        raw_words = np.frombuffer(samples, dtype="<u2")
        raw_values = raw_words >> 2

        return IirpFrame(
            width, height, raw_values, measurement_fields, bytes(calibration)
        )

    def measure(
        self, data: bytes, segments: list[tuple[int, bytes]] | None = None
    ) -> tuple[ThermalArray, tuple[int, int]]:
        frame = self.decode_frame(data, segments)
        calibration = frame.calibration_payload
        table = TransmissionTable(calibration)

        calibration_version = 0
        emissivity_enabled = emissivity_version = humidity_enabled = (
            humidity_version
        ) = 0
        reflection_enabled = reflection_version = 0

        if calibration[4:8] == b"\xff" * 4:
            calibration_version = int.from_bytes(calibration[2:4], "little")
            (
                emissivity_enabled,
                emissivity_version,
                humidity_enabled,
                humidity_version,
            ) = (
                int(x)
                for x in np.frombuffer(calibration, dtype="<u2", count=4, offset=8)
            )
            reflection_enabled, reflection_version = (
                int(x)
                for x in np.frombuffer(calibration, dtype="<u2", count=2, offset=20)
            )

        if calibration_version == 0x101:
            emissivity_enabled, emissivity_version = 1, 1
        if calibration_version <= 0x101:
            humidity_enabled, humidity_version = 1, 1
        elif calibration_version == 0x110:
            humidity_enabled, humidity_version = 1, 2
        elif calibration_version in (0x104, 0x113, 0x116, 0x119):
            humidity_enabled, humidity_version = 1, 3

        if reflection_enabled == 1 and reflection_version != 1:
            raise ValueError("Unsupported IIRP reflection calibration")

        ambient, distance, emissivity, humidity, reflection = frame.measurement_fields
        air = pow(ambient + _KELVIN_OFFSET, 4)
        reflected_radiance = (
            pow(reflection + _KELVIN_OFFSET, 4) if reflection_enabled == 1 else None
        )

        # Pre-bind distance-interpolated transmission lookup and humidity correction
        table_lookup = table.lookup_at_distance(distance)
        if humidity_enabled == 1:
            correct_hum = prepare_humidity_correction(
                distance, humidity, ambient, humidity_version
            )

        # The decoder shifts away two bits, leaving a 14-bit raw code.
        # Both bincount and LUT indexing use native-sized indices.
        raw_indices = frame.raw_values.astype(np.intp, copy=False)
        unique_raws = np.flatnonzero(np.bincount(raw_indices)).tolist()
        lut = np.empty(1 << 14, dtype=np.float32)
        active_emissivity_version = emissivity_version if emissivity_enabled == 1 else 0

        for raw in unique_raws:
            apparent = _float32(raw / 16.0 - _KELVIN_OFFSET)
            effective = _float32(
                lookup_emissivity(apparent, emissivity, active_emissivity_version)
            )
            initial_transmission = table_lookup(apparent)
            received = pow(_float32(apparent + _KELVIN_OFFSET), 4)
            estimate = _correct_radiance(
                received, effective, initial_transmission, air, reflected_radiance
            )
            transmission = table_lookup(estimate)
            if humidity_enabled == 1:
                transmission = correct_hum(transmission, estimate)
            if transmission != initial_transmission:
                estimate = _correct_radiance(
                    received, effective, transmission, air, reflected_radiance
                )
            lut[raw] = estimate

        temperatures = lut[raw_indices]
        return temperatures.astype("<f4", copy=False).view(ThermalArray), (
            frame.height,
            frame.width,
        )


def measure_float32(
    data: bytes, segments: list[tuple[int, bytes]] | None = None
) -> tuple[ThermalArray, tuple[int, int]]:
    """Measure IIRP R-JPEG bytes using IirpHandler."""
    return IirpHandler.get_instance().measure(data, segments)
