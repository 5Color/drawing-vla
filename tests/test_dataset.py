"""로드 시 획 정규화: 순서·방향 독립성, 원본 보존, 관측/행동 일치."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from train.dataset import DrawingVLADataset, encode_png_gray, pen_down_runs, render_canvas


def bracket(strokes):
    rows = []
    for stroke in strokes:
        rows.extend([[*stroke[0], 0], *[[*p, 1] for p in stroke], [*stroke[-1], 0]])
    return np.asarray(rows, dtype=np.float32)


class TestStrokeNormalization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "demo_test_0.json"
        self.top = [[0.2, 0.1], [0.4, 0.1], [0.6, 0.2]]
        self.bottom = [[0.1, 0.7], [0.4, 0.8], [0.8, 0.9]]

    def write_demo(self, strokes):
        traj = bracket(strokes)
        image = encode_png_gray(render_canvas(traj))
        data = {"meta": {"target": "test"}, "trajectory": traj.tolist(),
                "strokes": [{"points": s} for s in strokes],
                "target_128_png": image, "canvas_128_png": image}
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return traj, data

    def load(self, **kwargs):
        return DrawingVLADataset(files=[self.path], **kwargs)

    def test_default_preserves_recorded_order_and_direction(self):
        traj, _ = self.write_demo([self.bottom[::-1], self.top])
        np.testing.assert_array_equal(self.load().episodes[0].traj, traj)

    def test_order_and_direction_normalize_without_changing_source(self):
        expected = bracket([self.top, self.bottom])
        for strokes in ([self.bottom[::-1], self.top], [self.top[::-1], self.bottom],
                        [self.bottom, self.top[::-1]]):
            with self.subTest(strokes=strokes):
                _, raw = self.write_demo(strokes)
                before = self.path.read_bytes()
                ds = self.load(normalize_strokes=True)
                ep = ds.episodes[0]
                np.testing.assert_array_equal(ep.traj, expected)
                self.assertEqual(ep.raw, raw)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(len(ep.runs), 2)
                np.testing.assert_array_equal(ep.run_cache[-1], render_canvas(expected))

    def test_closed_stroke_and_single_point_keep_pen_brackets(self):
        loop = [[0.2, 0.2], [0.5, 0.3], [0.3, 0.5], [0.2, 0.2]]
        dot = [[0.7, 0.8]]
        self.write_demo([dot, loop[::-1]])
        ep = self.load(normalize_strokes=True).episodes[0]
        np.testing.assert_array_equal(ep.traj, bracket([loop, dot]))
        self.assertEqual(pen_down_runs(ep.traj).tolist(), [[1, 5], [7, 8]])

    def test_resampled_loader_observations_follow_normalized_actions(self):
        self.write_demo([self.bottom[::-1], self.top[::-1]])
        ds = self.load(normalize_strokes=True, resample_step=0.05)
        ep = ds.episodes[0]
        batch = next(iter(DataLoader(ds, batch_size=4)))
        self.assertEqual(tuple(batch["obs"].shape), (4, 2, 128, 128))
        self.assertEqual(tuple(batch["action"].shape), (4, 16, 3))
        for t in (0, 2, int(ep.runs[1, 0]) + 2, len(ep) - 1):
            np.testing.assert_array_equal(ds.canvas_at(ep, t), render_canvas(ep.traj[:t]))
        np.testing.assert_array_equal(ep.traj[0], [np.float32(0.2), np.float32(0.1), 0])
        self.assertTrue(ds[-1]["action_is_pad"][1:].all())

    def test_pen_up_only_trajectory_is_preserved(self):
        self.write_demo([self.top])
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["trajectory"] = [[0.2, 0.1, 0], [0.8, 0.7, 0]]
        self.path.write_text(json.dumps(data), encoding="utf-8")
        ep = self.load(normalize_strokes=True).episodes[0]
        np.testing.assert_array_equal(ep.traj, np.asarray(data["trajectory"], dtype=np.float32))
        self.assertEqual(len(ep.runs), 0)


if __name__ == "__main__":
    unittest.main()
