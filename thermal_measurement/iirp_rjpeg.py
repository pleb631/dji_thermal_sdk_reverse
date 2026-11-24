"""SDK-free decoding of the observed DJI IIRP R-JPEG layout."""

from array import array
from dataclasses import dataclass
from math import pow, sqrt
from struct import Struct, unpack_from
import sys

if __package__:
    from .iirp_emissivity import lookup_emissivity
    from .iirp_transmission import (
        TransmissionTable, correct_humidity, restore_legacy_m2ea_calibration,
    )
else:
    from iirp_emissivity import lookup_emissivity
    from iirp_transmission import (
        TransmissionTable, correct_humidity, restore_legacy_m2ea_calibration,
    )


_FLOAT32 = Struct("<f")
_KELVIN_OFFSET = 273.15
_TRANSMISSION_SCALE = 16384


def _float32(value):
    return _FLOAT32.unpack(_FLOAT32.pack(value))[0]


def correct_temperature(apparent, emissivity, transmission, ambient, reflected=None):
    """Apply the IIRP radiance correction using a Q14 transmission value."""
    apparent = _float32(apparent)
    emissivity = _float32(emissivity)
    ambient = _float32(ambient)
    if not 0 < transmission <= 0xFFFF or not 0 < emissivity <= 1:
        raise ValueError("Invalid IIRP transmission or emissivity")

    received = pow(_float32(apparent + _KELVIN_OFFSET), 4)
    air = pow(ambient + _KELVIN_OFFSET, 4)
    throughput = transmission * emissivity
    if reflected is None:
        received -= air * (1 - throughput / _TRANSMISSION_SCALE)
    else:
        reflected = _float32(reflected)
        reflected_weight = (1 - emissivity) * transmission
        received -= pow(reflected + _KELVIN_OFFSET, 4) * reflected_weight / _TRANSMISSION_SCALE
        received -= air * (1 - transmission / _TRANSMISSION_SCALE)

    divisor = _float32(_float32(throughput) / _TRANSMISSION_SCALE)
    kelvin = _float32(sqrt(sqrt(max(received, 0.0) / divisor)))
    return _float32(kelvin - _KELVIN_OFFSET)


@dataclass
class IirpFrame:
    width: int
    height: int
    raw_values: array
    measurement_fields: tuple
    calibration_payload: bytes

    def uncorrected_celsius(self):
        """Return sensor temperatures before the IIRP environment correction."""
        return array("f", (value / 16 - _KELVIN_OFFSET for value in self.raw_values))


def _jpeg_segments(data):
    if data[:2] != b"\xff\xd8":
        raise ValueError("Expected a JPEG start marker")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            raise ValueError("Invalid JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset == len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0xDA:
            return
        if marker == 0xD9 or offset + 2 > len(data):
            break
        length = unpack_from(">H", data, offset)[0]
        end = offset + length
        if length < 2 or end > len(data):
            raise ValueError("Invalid JPEG segment length")
        yield marker, data[offset + 2:end]
        offset = end
    raise ValueError("Missing JPEG start of scan")


def decode_iirp(data):
    """Read IIRP raw samples and metadata from the primary JPEG header."""
    samples = bytearray()
    header = None
    legacy_header = None
    calibration = bytearray()
    for marker, payload in _jpeg_segments(data):
        if marker == 0xE3:
            samples.extend(payload)
        elif marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffiirp"):
            if header is not None:
                raise ValueError("Multiple IIRP headers")
            header = payload
        elif marker == 0xE4 and len(payload) == 224 and payload[101:109] == b"Mini_640":
            if legacy_header is not None:
                raise ValueError("Multiple legacy IIRP headers")
            legacy_header = payload
        elif marker == 0xE5:
            calibration.extend(payload)

    if header is not None and len(header) >= 161:
        width, height = unpack_from("<HH", header, 157)
        measurement_fields = unpack_from("<5f", header, 32)
    elif header is None and legacy_header is not None:
        width, height = 640, 512
        measurement_fields = unpack_from("<5f", legacy_header)
        calibration = restore_legacy_m2ea_calibration(calibration)
    else:
        raise ValueError("Missing or truncated IIRP header")
    if width == 0 or height == 0 or len(samples) != 2 * width * height:
        raise ValueError("IIRP dimensions do not match the thermal payload")
    words = array("H")
    words.frombytes(samples)
    if sys.byteorder != "little":
        words.byteswap()
    raw_values = array("H", (word >> 2 for word in words))
    return IirpFrame(
        width, height, raw_values, measurement_fields, bytes(calibration)
    )


def measure_float32(data):
    """Measure a supported IIRP R-JPEG entirely in Python."""
    frame = decode_iirp(data)
    calibration = frame.calibration_payload
    table = TransmissionTable(calibration)
    calibration_version = 0
    emissivity_enabled = emissivity_version = humidity_enabled = humidity_version = 0
    reflection_enabled = reflection_version = 0
    if calibration[4:8] == b"\xff" * 4:
        calibration_version = unpack_from("<H", calibration, 2)[0]
        emissivity_enabled, emissivity_version, humidity_enabled, humidity_version = (
            unpack_from("<4H", calibration, 8)
        )
        reflection_enabled, reflection_version = unpack_from("<2H", calibration, 20)
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
    reflected = reflection if reflection_enabled == 1 else None

    # All pixels sharing the same raw code have the same inputs to correction.
    corrected = {}
    for raw in set(frame.raw_values):
        apparent = _float32(raw / 16 - _KELVIN_OFFSET)
        effective = lookup_emissivity(
            apparent, emissivity, emissivity_version if emissivity_enabled == 1 else 0
        )
        transmission = table.lookup(apparent, distance)
        estimate = correct_temperature(apparent, effective, transmission, ambient, reflected)
        transmission = table.lookup(estimate, distance)
        if humidity_enabled == 1:
            transmission = correct_humidity(
                transmission, estimate, distance, humidity, ambient, humidity_version
            )
        corrected[raw] = correct_temperature(
            apparent, effective, transmission, ambient, reflected
        )
    return array("f", (corrected[raw] for raw in frame.raw_values)), (frame.height, frame.width)
