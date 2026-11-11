"""붓 렌더러(train/brush.py)와 필압 판정 확인 — python -m unittest tests.test_brush"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "train")]

import brush  # noqa: E402


class TestBrush(unittest.TestCase):
    def test_spline_passes_through_control_points(self):
        pts = np.array([[0.1, 0.1], [0.3, 0.4], [0.5, 0.2], [0.8, 0.6]])
        curve = brush.catmull_rom(pts, per_seg=10)
        for p in pts:   # 보간이지 피팅이 아님: 모든 제어점을 지난다
            self.assertLess(np.linalg.norm(curve - p, axis=1).min(), 1e-9)

    def test_taper_thin_ends_thick_middle(self):
        curve = brush.catmull_rom(np.c_[np.linspace(0.1, 0.9, 30), np.full(30, 0.5)], per_seg=4)
        w = brush.widths(curve, base=8.0)
        self.assertLess(w[0], w[len(w) // 2] * 0.4)
        self.assertLess(w[-1], w[len(w) // 2] * 0.4)
        self.assertAlmostEqual(w[len(w) // 2], 8.0, delta=0.01)

    def test_pressure_and_slow_speed_make_thicker(self):
        curve = np.c_[np.linspace(0.1, 0.9, 50), np.full(50, 0.5)]
        light = brush.widths(curve, 8.0, taper=0, pressure=np.full(50, 0.2))
        heavy = brush.widths(curve, 8.0, taper=0, pressure=np.full(50, 1.0))
        self.assertTrue((heavy > light).all())
        speed = np.r_[np.full(25, 2.0), np.full(25, 0.5)]   # 뒤쪽이 느림
        w = brush.widths(curve, 8.0, taper=0, speed=speed)
        self.assertGreater(w[40], w[10])

    def test_smoothing_removes_step_jitter_keeps_ends(self):
        x = np.linspace(0.1, 0.9, 60)
        run = np.c_[x, 0.5 + 0.004 * (-1) ** np.arange(60)]   # 스텝마다 지그재그
        sm = brush.smooth_run(run)
        self.assertLess(np.abs(np.diff(sm[5:-5, 1])).max(), np.abs(np.diff(run[:, 1])).max() * 0.3)
        np.testing.assert_allclose(sm[[0, -1]], run[[0, -1]])

    def test_render_from_trajectory(self):
        traj = np.array([[0.2, 0.2, 0], [0.2, 0.2, 1], [0.5, 0.6, 1], [0.8, 0.3, 1], [0.8, 0.3, 0]])
        runs = brush.runs_from_traj(traj)
        self.assertEqual(len(runs), 1)
        img = np.asarray(brush.render(runs, 128).convert("L"))
        self.assertEqual(img.shape, (128, 128))
        self.assertLess(img.min(), 100)   # 잉크가 그려짐


if __name__ == "__main__":
    unittest.main()
