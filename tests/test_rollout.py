"""롤아웃 루프(Temporal Ensembling 포함)와 매끈함 지표 확인 — 가짜 정책으로 빠르게. python -m unittest tests.test_rollout"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import rollout_eval as R  # noqa: E402


class LinePolicy:
    """현재 펜 위치에서 오른쪽으로 0.005씩 가는 16스텝 (펜 다운). 호출 횟수를 센다."""
    chunk_size, head_type = 16, "mlp"

    def __init__(self, wobble: float = 0.0):
        self.calls, self.wobble = 0, wobble

    def predict(self, obs, st):
        self.calls += 1
        x0, y0 = float(st[0, 0]), float(st[0, 1])
        k = torch.arange(1, 17, dtype=torch.float32)
        y = y0 + self.wobble * (1 if self.calls % 2 else -1) * torch.ones(16)   # 호출마다 위아래로 어긋남
        xy = torch.stack([(x0 + 0.005 * k).clamp(0, 1), y], 1)[None]
        return xy, torch.ones(1, 16)


class TestEnsembleRollout(unittest.TestCase):
    def run_policy(self, pol, te_m, q, iters):
        target = np.zeros((R.OBS_SIZE, R.OBS_SIZE), np.uint8)
        return R.rollout(pol, target, "cpu", max_iters=iters, exec_steps=12, patience=999, te_m=te_m, query_freq=q)

    def test_query_freq_controls_calls(self):
        pol = LinePolicy()
        ex, hist, _ = self.run_policy(pol, 0.01, 4, 10)
        self.assertEqual((pol.calls, len(ex), len(hist)), (10, 40, 10))      # 4스텝마다 1번 호출

    def test_ensemble_averages_overlapping_chunks(self):
        # 호출마다 ±0.01 어긋나는 정책: 하드 컷은 계단처럼 튀고, 앙상블은 겹친 예측을 평균해 덜 튄다
        hard, _, _ = self.run_policy(LinePolicy(0.01), None, 12, 6)
        soft, _, _ = self.run_policy(LinePolicy(0.01), 0.01, 4, 18)
        jump = lambda e: np.abs(np.diff(e[:, 1])).max()
        self.assertLess(jump(soft), jump(hard))

    def test_smoothness_metric(self):
        t = np.linspace(0, np.pi, 200)
        arc = np.c_[0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t), np.ones(200)]
        zig = arc.copy()
        zig[::2, 1] += 0.004
        a, z = R.smoothness(arc), R.smoothness(zig)
        self.assertLess(a["wave_deg"], 0.5)
        self.assertGreater(z["wave_deg"], a["wave_deg"] * 10)
        self.assertLess(R.smoothness(zig, display=True)["wave_deg"], z["wave_deg"])   # 붓 렌더러 다듬기 후


if __name__ == "__main__":
    unittest.main()
