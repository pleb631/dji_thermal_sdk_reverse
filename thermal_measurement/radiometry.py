"""SDK-free DJI R-JPEG float32 measurement using object-oriented handlers."""

import argparse
import sys
from pathlib import Path

from .dirp_radiance import DirpHandler
from .girp_radiance import GirpHandler
from .hirp_rjpeg import HirpHandler
from .iirp_rjpeg import IirpHandler, _jpeg_segments
from .xts_radiance import ThermalArray, XtsHandler


def measure_float32(data: bytes) -> tuple[ThermalArray, tuple[int, int]]:
    """Measure R-JPEG bytes and return (temperatures, (height, width)).

    Temperatures are a row-major ThermalArray (NumPy float32 ndarray) in degrees Celsius.
    """
    segments = list(_jpeg_segments(data))
    if XtsHandler.can_handle(segments):
        return XtsHandler.get_instance().measure(segments)
    if GirpHandler.can_handle(segments):
        return GirpHandler.get_instance().measure(segments)
    if HirpHandler.can_handle(segments):
        return HirpHandler.get_instance().measure(data, segments)
    if DirpHandler.can_handle(segments):
        return DirpHandler.get_instance().measure(segments)
    return IirpHandler.get_instance().measure(data, segments)


def main():
    parser = argparse.ArgumentParser(
        description="Measure DJI R-JPEG temperatures as float32."
    )
    parser.add_argument("-s", required=True, type=Path, metavar="R-JPEG")
    parser.add_argument("-o", required=True, type=Path, metavar="RAW")
    parser.add_argument("-a", choices=["measure"], default="measure")
    parser.add_argument("--measurefmt", choices=["float32"], default="float32")
    args = parser.parse_args()

    try:
        temperatures, (height, width) = measure_float32(args.s.read_bytes())
        if sys.byteorder != "little":
            temperatures.byteswap()
        args.o.write_bytes(temperatures.tobytes())
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print(f"image width : {width}")
    print(f"image height : {height}")


if __name__ == "__main__":
    main()
