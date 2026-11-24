"""Regression test against saved DJI Thermal SDK 1.8 measurements."""

from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

from thermal_measurement import measure_float32


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "dji_tsdk_1_8"


class MeasureFloat32Test(unittest.TestCase):
    def test_official_rjpeg_samples(self):
        manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 49)
        for sample in manifest:
            with self.subTest(image=sample["image"]):
                data = (FIXTURE_ROOT / sample["image"]).read_bytes()
                self.assertEqual(sha256(data).hexdigest(), sample["image_sha256"])

                temperatures, shape = measure_float32(data)
                self.assertEqual(shape, tuple(sample["shape"]))
                self.assertEqual(temperatures.typecode, "f")
                self.assertEqual(temperatures.itemsize, 4)
                self.assertEqual(len(temperatures), shape[0] * shape[1])
                if sys.byteorder != "little":
                    temperatures.byteswap()
                self.assertEqual(
                    sha256(temperatures.tobytes()).hexdigest(), sample["float32_le_sha256"]
                )
