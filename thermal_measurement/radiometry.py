"""SDK-free DJI R-JPEG float32 measurement."""

import argparse
from pathlib import Path
import sys

if __package__:
    from .dirp_radiance import measure_dirp
    from .girp_radiance import measure_girp
    from .hirp_rjpeg import measure_hirp
    from .iirp_rjpeg import _jpeg_segments, measure_float32 as measure_iirp
    from .xts_radiance import measure_xts
else:
    from dirp_radiance import measure_dirp
    from girp_radiance import measure_girp
    from hirp_rjpeg import measure_hirp
    from iirp_rjpeg import _jpeg_segments, measure_float32 as measure_iirp
    from xts_radiance import measure_xts


def measure_float32(data):
    """Measure R-JPEG bytes and return (temperatures, (height, width)).

    Temperatures are a row-major array('f') in degrees Celsius.
    """
    segments = list(_jpeg_segments(data))
    if any(marker == 0xE2 and payload.startswith(b"Plug") for marker, payload in segments):
        return measure_xts(segments)
    if any(marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffgirp")
           for marker, payload in segments):
        return measure_girp(segments)
    if any(marker == 0xE4 and payload.startswith(b"\xff\xd2\xd1\xffhirp")
           for marker, payload in segments):
        return measure_hirp(data)
    if any(marker == 0xE4 and payload.startswith(b"\xaa\x55\x12\x06")
           for marker, payload in segments):
        return measure_dirp(segments)
    return measure_iirp(data)


def main():
    parser = argparse.ArgumentParser(description="Measure DJI R-JPEG temperatures as float32.")
    parser.add_argument("-s", required=True, type=Path, metavar="R-JPEG")
    parser.add_argument("-o", required=True, type=Path, metavar="RAW")
    parser.add_argument("-a", choices=["measure"], default="measure")
    parser.add_argument("--measurefmt", choices=["float32"], default="float32")
    args = parser.parse_args()

    try:
        temperatures, (height, width) = measure_float32(args.s.read_bytes())
        if sys.byteorder != "little":
            temperatures.byteswap()
        args.o.write_bytes(temperatures)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print(f"image width : {width}")
    print(f"image height : {height}")


if __name__ == "__main__":
    main()
