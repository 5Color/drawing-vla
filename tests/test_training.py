"""Training caches and episode splits must follow the data actually loaded."""
import contextlib
import io
import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from train.dataset import DrawingVLADataset, encode_png_gray
from train import train_policy


class StopBeforeTraining(Exception):
    pass


class TestTrainingData(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "demo_test_0.json"
        self.cache = self.root / "cache.npz"
        self.traj = [[0.2, 0.2, 0], [0.2, 0.2, 1], [0.6, 0.2, 1], [0.6, 0.2, 0]]
        self.write_demo(self.path)

    def write_demo(self, path, *, traj=None, target=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "meta": {"target": "test"},
            "trajectory": self.traj if traj is None else traj,
            "target_128_png": encode_png_gray(np.full((128, 128), target, np.uint8)),
        }), encoding="utf-8")

    def arrays(self, **kwargs):
        return train_policy.build_arrays(DrawingVLADataset(files=[self.path], **kwargs), self.cache)

    def test_cache_invalidates_when_coordinates_change_without_length_change(self):
        before = self.arrays()
        changed = [[0.2, 0.2, 0], [0.2, 0.2, 1], [0.8, 0.2, 1], [0.8, 0.2, 0]]
        self.write_demo(self.path, traj=changed)
        after = self.arrays()
        self.assertAlmostEqual(float(after["act"][0, 2, 0]), 0.8)
        self.assertFalse(np.array_equal(after["canv"][3], before["canv"][3]))

    def test_cache_invalidates_when_target_pixels_change(self):
        self.arrays()
        self.write_demo(self.path, target=37)
        np.testing.assert_array_equal(self.arrays()["targets"], np.full((1, 128, 128), 37, np.uint8))

    def test_cache_invalidates_when_chunk_size_changes(self):
        self.arrays()
        self.assertEqual(self.arrays(chunk_size=4)["act"].shape, (4, 4, 3))

    def test_cache_invalidates_when_render_width_changes(self):
        thin = self.arrays()["canv"][3]
        thick = self.arrays(stroke_width_px=8)["canv"][3]
        self.assertGreater(int(thick.sum()), int(thin.sum()) * 2)

    def test_cache_invalidates_when_observation_size_changes(self):
        self.arrays()
        resized = self.arrays(obs_size=64)
        self.assertEqual(resized["canv"].shape, (4, 64, 64))
        self.assertEqual(resized["targets"].shape, (1, 64, 64))
        self.assertEqual(resized["act"].shape, (4, 16, 3))

    def test_cache_invalidates_when_normalization_reorders_same_length_trajectory(self):
        bottom_first = [[0.2, 0.8, 0], [0.2, 0.8, 1], [0.6, 0.8, 1], [0.6, 0.8, 0],
                        [0.2, 0.2, 0], [0.2, 0.2, 1], [0.6, 0.2, 1], [0.6, 0.2, 0]]
        self.write_demo(self.path, traj=bottom_first)
        self.arrays()
        normalized = self.arrays(normalize_strokes=True)
        self.assertAlmostEqual(float(normalized["act"][0, 0, 1]), 0.2)

    def test_unchanged_data_reuses_cached_canvases(self):
        expected = self.arrays()
        ds = DrawingVLADataset(files=[self.path])
        with patch.object(ds, "canvas_at", side_effect=AssertionError("cached canvases were rerendered")):
            actual = train_policy.build_arrays(ds, self.cache)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_reused_cache_archive_is_closed_before_return(self):
        self.arrays()
        archive = np.load(self.cache)
        self.addCleanup(archive.close)
        stream = archive.fid
        with patch.object(train_policy.np, "load", return_value=archive):
            self.arrays()
        self.assertTrue(stream.closed)

    def test_missing_valid_validation_episodes_stops_before_model_construction(self):
        raw = self.root / "data" / "raw"
        files = [raw / f"demo_test_{i}.json" for i in range(4)]
        for path in files:
            self.write_demo(path)
        shuffled = files[:]
        random.Random(0).shuffle(shuffled)
        for path in shuffled[:2]:
            path.write_text("{}", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.object(train_policy, "ROOT", self.root), \
                    patch.object(train_policy, "OUT_DIR", self.root / "outputs"), \
                    patch.object(train_policy, "pick_device", return_value=torch.device("cpu")), \
                    patch.object(train_policy, "build_policy", side_effect=AssertionError("empty validation reached model construction")), \
                    patch("sys.argv", ["train_policy.py", "--target", "test", "--val-ratio", "0.5"]):
                with self.assertRaises(ValueError):
                    train_policy.main()

    def test_skipped_training_episode_keeps_validation_samples_held_out(self):
        raw = self.root / "data" / "raw"
        files = [raw / f"demo_test_{i}.json" for i in range(4)]
        for path in files:
            self.write_demo(path)
        shuffled = files[:]
        random.Random(0).shuffle(shuffled)
        shuffled[-1].write_text("{}", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.object(train_policy, "ROOT", self.root), \
                    patch.object(train_policy, "OUT_DIR", self.root / "outputs"), \
                    patch.object(train_policy, "pick_device", return_value=torch.device("cpu")), \
                    patch.object(train_policy, "build_policy", side_effect=StopBeforeTraining), \
                    patch("sys.argv", ["train_policy.py", "--target", "test", "--val-ratio", "0.5"]):
                with self.assertRaises(StopBeforeTraining):
                    train_policy.main()
        # One of two training files was skipped; both validation episodes remain held out.
        self.assertRegex(output.getvalue(), r"샘플 train 4 / val 8\b")


if __name__ == "__main__":
    unittest.main()
