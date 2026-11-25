"""Compare Linux DJI Thermal SDK and repository float32 measurement."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

from thermal_measurement import measure_float32

ROOT = Path(__file__).resolve().parent


def load_sdk(sdk_dir):
    lib = ct.CDLL(str(sdk_dir.resolve() / "libdirp.so"))
    lib.dirp_create_from_rjpeg.argtypes = [
        ct.c_char_p, ct.c_int32, ct.POINTER(ct.c_void_p)
    ]
    lib.dirp_measure_ex.argtypes = [
        ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32
    ]
    lib.dirp_destroy.argtypes = [ct.c_void_p]

    def check(operation, *args):
        status = operation(*args)
        if status:
            raise RuntimeError(f"{operation.__name__} returned {status}")

    def measure(data, shape):
        handle = ct.c_void_p()
        output = (ct.c_float * (shape[0] * shape[1]))()
        check(lib.dirp_create_from_rjpeg, data, len(data), ct.byref(handle))
        try:
            check(lib.dirp_measure_ex, handle, output, ct.sizeof(output))
        finally:
            check(lib.dirp_destroy, handle)
        return output

    return measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-dir", type=Path, default=ROOT / ".dji-sdk")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    sdk_measure = load_sdk(args.sdk_dir)
    fixture = ROOT / "tests/fixtures/dji_tsdk_1_8"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    samples = [(sample, (fixture / sample["image"]).read_bytes()) for sample in manifest]

    for sample, data in samples:
        shape = tuple(sample["shape"])
        output, actual_shape = measure_float32(data)
        raw = bytes(sdk_measure(data, shape))
        if (
            actual_shape != shape
            or raw != output.tobytes()
            or hashlib.sha256(raw).hexdigest() != sample["float32_le_sha256"]
        ):
            raise ValueError(f"Float32 reference mismatch for {sample['image']}")
    print(f"{len(samples)} images matched byte for byte", flush=True)

    times = {}
    for round_index in range(args.runs):
        for sample, data in samples:
            model = Path(sample["image"]).parent.name
            shape = tuple(sample["shape"])
            timings = times.setdefault(model, ([], []))
            for index in (round_index % 2, 1 - round_index % 2):
                start = perf_counter_ns()
                result = sdk_measure(data, shape) if index == 0 else measure_float32(data)
                elapsed_ms = (perf_counter_ns() - start) / 1e6
                del result
                timings[index].append(elapsed_ms)

    print(f"{'Model':<10} {'SDK ms':>10} {'Repo ms':>10} {'Speedup':>10}")
    for model, (sdk_times, repo_times) in sorted(times.items()):
        print_row(model, sdk_times, repo_times)
    print_row(
        "Overall",
        [t for sdk_times, _ in times.values() for t in sdk_times],
        [t for _, repo_times in times.values() for t in repo_times],
    )


def print_row(label, sdk_times, repo_times):
    sdk_ms, repo_ms = mean(sdk_times), mean(repo_times)
    print(f"{label:<10} {sdk_ms:10.3f} {repo_ms:10.3f} {sdk_ms / repo_ms:9.1f}x")


if __name__ == "__main__":
    main()
