"""Benchmark latency and throughput for DJI R-JPEG radiometry measurement."""

import argparse
import time
from pathlib import Path

from thermal_measurement import measure_float32
from thermal_measurement.dirp_radiance import DirpHandler
from thermal_measurement.girp_radiance import GirpHandler
from thermal_measurement.hirp_rjpeg import HirpHandler
from thermal_measurement.iirp_rjpeg import IirpHandler, _jpeg_segments
from thermal_measurement.xts_radiance import XtsHandler


def detect_protocol(segments: list[tuple[int, bytes]]) -> str:
    """Identify protocol type from JPEG segments."""
    if XtsHandler.can_handle(segments):
        return "XT S"
    if GirpHandler.can_handle(segments):
        return "GIRP"
    if HirpHandler.can_handle(segments):
        return "HIRP"
    if DirpHandler.can_handle(segments):
        return "DIRP"
    return "IIRP"


def run_benchmark(dataset_dir: Path, runs: int = 5, warmup: int = 1) -> None:
    images = sorted(dataset_dir.glob("*/*.JPG"))
    if not images:
        images = sorted(dataset_dir.glob("*/*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG images found in {dataset_dir}")

    print(f"Loading {len(images)} sample images into memory...")
    image_entries = []
    for img in images:
        model = img.parent.name
        data = img.read_bytes()
        segments = list(_jpeg_segments(data))
        proto = detect_protocol(segments)
        _, (height, width) = measure_float32(data)
        resolution = f"{width}x{height}"
        image_entries.append((model, proto, resolution, img.name, data))

    # Warm-up phase
    if warmup > 0:
        print(f"Warming up ({warmup} round{'s' if warmup > 1 else ''})...")
        for _ in range(warmup):
            for _, _, _, _, data in image_entries:
                measure_float32(data)

    # Benchmark phase
    print(f"Running benchmark ({runs} rounds, {len(image_entries) * runs} total measurements)...")
    durations: list[float] = []
    model_durations: dict[str, list[float]] = {}
    model_info: dict[str, tuple[set[str], set[str], int]] = {}

    for _ in range(runs):
        for model, proto, res, name, data in image_entries:
            t0 = time.perf_counter()
            measure_float32(data)
            t1 = time.perf_counter()

            dt_ms = (t1 - t0) * 1000.0
            durations.append(dt_ms)
            model_durations.setdefault(model, []).append(dt_ms)

            if model not in model_info:
                model_info[model] = (set(), set(), 0)
            model_info[model][0].add(proto)
            model_info[model][1].add(res)

    for model in model_info:
        model_info[model] = (
            model_info[model][0],
            model_info[model][1],
            len(model_durations[model]) // runs,
        )

    avg_ms = sum(durations) / len(durations)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print("\n" + "=" * 76)
    print("                      DJI Thermal SDK Benchmark Results")
    print("=" * 76)
    print(f"Total Unique Samples   : {len(image_entries)}")
    print(f"Benchmark Rounds       : {runs}")
    print(f"Total Invocations      : {len(durations)}")
    print(f"Average Latency        : {avg_ms:.2f} ms")
    print(f"Min Latency            : {min(durations):.2f} ms")
    print(f"Max Latency            : {max(durations):.2f} ms")
    print(f"Overall Throughput     : {fps:.1f} FPS")
    print("-" * 76)
    print(
        f"{'Model':<12} {'Protocol':<12} {'Resolution':<12} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Count':<6}"
    )
    print("-" * 76)

    for model, times in sorted(model_durations.items()):
        protos = "/".join(sorted(model_info[model][0]))
        resolutions = "/".join(sorted(model_info[model][1]))
        m_avg = sum(times) / len(times)
        m_min = min(times)
        m_max = max(times)
        count = model_info[model][2]
        print(
            f"{model:<12} {protos:<12} {resolutions:<12} {m_avg:<10.2f} {m_min:<10.2f} {m_max:<10.2f} {count:<6}"
        )
    print("=" * 76)


def main() -> None:
    default_dataset = (
        Path(__file__).resolve().parent
        / "tests"
        / "fixtures"
        / "dji_tsdk_1_8"
        / "dataset"
    )

    parser = argparse.ArgumentParser(
        description="Benchmark latency and throughput for DJI R-JPEG radiometry measurement."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset,
        help="Path to the sample dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up iterations (default: 1)",
    )
    args = parser.parse_args()
    run_benchmark(args.dataset_dir, runs=args.runs, warmup=args.warmup)


if __name__ == "__main__":
    main()
