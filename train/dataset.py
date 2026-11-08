"""DrawingVLADataset: data/raw/demo_*.json → PyTorch 텐서.

샘플 1개 = (에피소드 e, 스텝 t), t ∈ [0, T_e - 1]
  obs            float32 (2, 128, 128) [0,1]  ch0 = traj[:t] 렌더 캔버스, ch1 = 목표 도안
  action         float32 (16, 3)       [0,1]  traj[t : t+16] = (x, y, pen_down), 끝 경계 복제 패딩
  state          float32 (3,)                 현재 펜 상태 traj[t-1] (t=0이면 HOME_STATE)
  action_is_pad  bool    (16,)                패딩 스텝 표시 (loss 마스킹용)
  target_name    str

검증:  python train/dataset.py [--root data/raw] [--resample-step 0.004] [--normalize-strokes]
규격은 CLAUDE.md 참고. 렌더 상수는 collector/index.html(OBS, OBS_W)과 동기화.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

OBS_SIZE = 128
CHUNK_SIZE = 16
STROKE_WIDTH_PX = 2.0      # 128 해상도 기준 (= 512 좌표계 8 단위, JS OBS_W)
SUPERSAMPLE = 4            # 512에서 그린 뒤 BOX 다운샘플 → 안티앨리어싱
HOME_STATE = (0.5, 0.5, 0.0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# 궤적 유틸
# ---------------------------------------------------------------------------
def slice_action_chunk(traj: np.ndarray, t: int, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """traj[t : t+H], 끝을 넘으면 마지막 포인트 복제. collector의 sliceChunk와 동일."""
    idx = np.minimum(np.arange(t, t + chunk_size), len(traj) - 1)
    return traj[idx]


def pen_down_runs(traj: np.ndarray) -> np.ndarray:
    """pen_down=1이 연속된 구간들의 [start, end) 인덱스, shape (R, 2)."""
    pen = (traj[:, 2] > 0.5).astype(np.int8)
    edges = np.diff(np.concatenate([[0], pen, [0]]))
    return np.stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)], axis=1)


def normalize_stroke_order(traj: np.ndarray) -> np.ndarray:
    """획을 위→아래, 왼쪽→오른쪽 좌표 순으로 정렬하고 방향도 통일한다.

    양 방향의 (y, x) 점열 중 사전순으로 앞선 것을 선택한다. 끝점이 같은 닫힌
    획도 안쪽 점으로 방향을 결정한다. pen-up bracketing을 다시 만들며 입력은 보존한다.
    """
    def key(stroke):
        return tuple(stroke[:, [1, 0]].ravel())

    strokes = []
    for a, b in pen_down_runs(traj):
        stroke = traj[a:b]
        strokes.append(min((stroke, stroke[::-1]), key=key))
    if not strokes:
        return traj.copy()
    bracketed = []
    for stroke in sorted(strokes, key=key):
        start, end = stroke[0].copy(), stroke[-1].copy()
        start[2] = end[2] = 0
        bracketed.extend((start[None], stroke, end[None]))
    return np.concatenate(bracketed).astype(np.float32)


def resample_trajectory(traj: np.ndarray, step: float) -> np.ndarray:
    """같은 pen 상태가 이어지는 구간마다 호 길이 `step` 간격으로 균일 재샘플.

    구간 양 끝점(획 시작/끝, 펜-업 이동의 출발/도착)은 보존되므로 bracketing 규약이 유지된다.
    원시 입력의 속도 편차를 없애고 T를 줄여, 16스텝 청크가 일정한 공간 범위를 덮게 한다.
    """
    if step <= 0:
        raise ValueError("resample step must be > 0")
    traj = np.asarray(traj, dtype=np.float64)
    change = np.flatnonzero(np.diff(traj[:, 2]) != 0) + 1
    bounds = np.concatenate([[0], change, [len(traj)]])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = traj[a:b]
        xy = seg[:, :2]
        if len(xy) > 1:  # 제자리 중복 포인트 제거 (np.interp는 증가하는 xp 필요)
            keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9])
            xy = xy[keep]
        if len(xy) == 1:
            out.append(np.array([[xy[0, 0], xy[0, 1], seg[0, 2]]]))
            continue
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        n = max(1, int(np.ceil(s[-1] / step)))
        q = np.linspace(0.0, s[-1], n + 1)
        out.append(np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1]), np.full(n + 1, seg[0, 2])], 1))
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# 관측 렌더링 (canonical: 학습과 추론 시뮬레이터 모두 이 함수로 캔버스를 그린다)
# ---------------------------------------------------------------------------
def render_polyline(
    xy: np.ndarray,
    size: int = OBS_SIZE,
    width_px: float = STROKE_WIDTH_PX,
    supersample: int = SUPERSAMPLE,
) -> np.ndarray:
    """펜 다운 run 하나(정규화 좌표 [n,2])를 size×size uint8로 렌더 (둥근 cap/join)."""
    S = size * supersample
    img = Image.new("L", (S, S), 0)
    draw = ImageDraw.Draw(img)
    w = max(1, int(round(width_px * supersample)))
    r = w / 2.0
    pts = [(float(x) * S, float(y) * S) for x, y in xy]
    if len(pts) >= 2:
        draw.line(pts, fill=255, width=w, joint="curve")
    for x, y in (pts[0], pts[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(img.resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def render_canvas(traj: np.ndarray, size: int = OBS_SIZE, width_px: float = STROKE_WIDTH_PX,
                  supersample: int = SUPERSAMPLE) -> np.ndarray:
    """궤적 [T,3] 전체를 128×128 uint8 캔버스로 렌더. run끼리는 max 합성."""
    canvas = np.zeros((size, size), np.uint8)
    for a, b in pen_down_runs(np.asarray(traj)):
        np.maximum(canvas, render_polyline(traj[a:b, :2], size, width_px, supersample), out=canvas)
    return canvas


def decode_png_gray(data_url: str, size: int = OBS_SIZE) -> np.ndarray:
    b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.uint8)


def encode_png_gray(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 에피소드 / 데이터셋
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    path: Path
    target_name: str
    traj: np.ndarray            # (T, 3) float32
    target: np.ndarray          # (128, 128) uint8
    runs: np.ndarray            # (R, 2) pen-down run [start, end)
    run_cache: np.ndarray       # (R+1, 128, 128) uint8, [k] = 앞의 k개 run 완료 후 캔버스
    browser_canvas: np.ndarray | None
    raw: dict

    def __len__(self) -> int:
        return len(self.traj)


def load_episode(path: Path, resample_step: float | None = None, obs_size: int = OBS_SIZE,
                 width_px: float = STROKE_WIDTH_PX, supersample: int = SUPERSAMPLE,
                 normalize_strokes: bool = False) -> Episode:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    traj = np.asarray(d["trajectory"], dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] != 3 or len(traj) == 0:
        raise ValueError(f"trajectory shape {traj.shape}, expected (T, 3)")
    traj[:, :2] = np.clip(traj[:, :2], 0.0, 1.0)
    traj[:, 2] = (traj[:, 2] > 0.5).astype(np.float32)
    if normalize_strokes:
        traj = normalize_stroke_order(traj)
    if resample_step:
        traj = resample_trajectory(traj, resample_step)

    target_name = str(d.get("meta", {}).get("target") or path.stem.split("_")[1])
    target = decode_png_gray(d["target_128_png"], obs_size)
    browser = decode_png_gray(d["canvas_128_png"], obs_size) if d.get("canvas_128_png") else None

    runs = pen_down_runs(traj)
    cache = np.zeros((len(runs) + 1, obs_size, obs_size), np.uint8)
    for k, (a, b) in enumerate(runs):
        cache[k + 1] = np.maximum(cache[k], render_polyline(traj[a:b, :2], obs_size, width_px, supersample))
    return Episode(path, target_name, traj, target, runs, cache, browser, d)


class DrawingVLADataset(Dataset):
    """data/raw/demo_*.json 시연들을 (obs, action chunk) 샘플로 펼친 데이터셋.

    Args:
        root: demo_*.json 폴더 (files를 주면 무시)
        files: 사용할 파일 목록 (에피소드 단위 train/val 분할용)
        chunk_size: action chunk 길이 H (기본 16)
        resample_step: None이면 원시 궤적 그대로, 값이 있으면 정규화 호 길이 간격으로 재샘플
                       (예: 1/256 ≈ 0.004 → 128 관측에서 0.5px 간격)
        targets: 특정 도안만 사용 (예: ["apple", "cat"])
        normalize_strokes: True이면 로드할 때만 획 순서·방향을 좌표 기준으로 통일 (기본 False)
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        files: Sequence[str | Path] | None = None,
        chunk_size: int = CHUNK_SIZE,
        resample_step: float | None = None,
        targets: Iterable[str] | None = None,
        obs_size: int = OBS_SIZE,
        stroke_width_px: float = STROKE_WIDTH_PX,
        supersample: int = SUPERSAMPLE,
        normalize_strokes: bool = False,
    ):
        self.chunk_size = chunk_size
        self.obs_size = obs_size
        self.stroke_width_px = stroke_width_px
        self.supersample = supersample
        self.normalize_strokes = normalize_strokes
        paths = sorted(Path(root).glob("demo_*.json")) if files is None else [Path(f) for f in files]
        keep = set(targets) if targets else None

        self.episodes: list[Episode] = []
        for p in paths:
            try:
                ep = load_episode(p, resample_step, obs_size, stroke_width_px, supersample,
                                  normalize_strokes=normalize_strokes)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                warnings.warn(f"skip {p.name}: {e}")
                continue
            if keep is None or ep.target_name in keep:
                self.episodes.append(ep)

        self._offsets = np.cumsum([0] + [len(ep) for ep in self.episodes])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def locate(self, idx: int) -> tuple[int, int]:
        """전역 인덱스 → (에피소드 인덱스, 스텝 t)."""
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        e = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        return e, idx - int(self._offsets[e])

    def canvas_at(self, ep: Episode, t: int) -> np.ndarray:
        """traj[:t]를 렌더한 캔버스. 완료된 run은 캐시, 진행 중인 run만 새로 그린다."""
        k = int(np.searchsorted(ep.runs[:, 1], t, side="right")) if len(ep.runs) else 0
        canvas = ep.run_cache[k]
        if k < len(ep.runs) and ep.runs[k, 0] < t:
            partial = render_polyline(ep.traj[ep.runs[k, 0]:t, :2], self.obs_size,
                                      self.stroke_width_px, self.supersample)
            canvas = np.maximum(canvas, partial)
        return canvas

    def __getitem__(self, idx: int) -> dict:
        e, t = self.locate(idx)
        ep = self.episodes[e]
        obs = np.stack([self.canvas_at(ep, t), ep.target]).astype(np.float32) / 255.0
        action = slice_action_chunk(ep.traj, t, self.chunk_size)
        is_pad = np.arange(t, t + self.chunk_size) > len(ep.traj) - 1
        state = ep.traj[t - 1] if t > 0 else np.asarray(HOME_STATE, np.float32)
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(np.ascontiguousarray(action, dtype=np.float32)),
            "state": torch.as_tensor(state, dtype=torch.float32),
            "action_is_pad": torch.from_numpy(is_pad),
            "target_name": ep.target_name,
        }


# ---------------------------------------------------------------------------
# shape 검증 (__main__)
# ---------------------------------------------------------------------------
def canvas_iou(a: np.ndarray, b: np.ndarray, thr: int = 64) -> float:
    A, B = a > thr, b > thr
    union = np.logical_or(A, B).sum()
    return 1.0 if union == 0 else float(np.logical_and(A, B).sum() / union)


def make_synthetic_demo(path: Path) -> None:
    """data/raw가 비었을 때 파이프라인 검증용 합성 시연 (원 + 가로선 2획)."""
    th = np.linspace(0, 2 * np.pi, 240)
    strokes = [
        np.stack([0.5 + 0.28 * np.cos(th), 0.5 + 0.28 * np.sin(th)], 1),
        np.stack([np.linspace(0.3, 0.7, 80), np.full(80, 0.5)], 1),
    ]
    traj = []
    for s in strokes:
        traj += [[*s[0], 0]] + [[*p, 1] for p in s] + [[*s[-1], 0]]
    traj = np.round(np.asarray(traj), 5)
    img = encode_png_gray(render_canvas(traj))
    path.write_text(json.dumps({
        "schema_version": 1,
        "meta": {"target": "synthetic", "num_strokes": len(strokes), "num_steps": len(traj)},
        "trajectory": traj.tolist(),
        "target_128_png": img,
        "canvas_128_png": img,
    }), encoding="utf-8")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="DrawingVLADataset shape 검증")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--resample-step", type=float, default=None)
    ap.add_argument("--normalize-strokes", action="store_true", help="로드 시 획 순서·방향 정규화 (원본 보존)")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    root = Path(args.root)
    tmp = None
    if not any(root.glob("demo_*.json")):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        make_synthetic_demo(root / "demo_synthetic_0.json")
        print(f"[info] {args.root} 에 demo_*.json이 없어 합성 시연 1개로 검증합니다.\n")

    t_load = time.perf_counter()
    ds = DrawingVLADataset(root, resample_step=args.resample_step, normalize_strokes=args.normalize_strokes)
    t_load = time.perf_counter() - t_load
    _check(len(ds.episodes) > 0, "로드된 에피소드가 없습니다")

    print(f"episodes={len(ds.episodes)}  samples={len(ds)}  resample_step={args.resample_step}"
          f"  normalize_strokes={args.normalize_strokes}"
          f"  load={t_load:.2f}s")
    print("per target:", dict(Counter(ep.target_name for ep in ds.episodes)))
    print(f"\n{'file':<44} {'target':<10} {'T':>6} {'runs':>5} {'IoU(py vs browser)':>19}")
    for ep in ds.episodes:
        iou = "-" if ep.browser_canvas is None else f"{canvas_iou(render_canvas(ep.traj), ep.browser_canvas):.3f}"
        print(f"{ep.path.name:<44} {ep.target_name:<10} {len(ep):>6} {len(ep.runs):>5} {iou:>19}")

    # --- 샘플 단위 검증 ---
    rng = np.random.default_rng(0)
    probe = sorted({0, len(ds) - 1, *rng.integers(0, len(ds), size=min(64, len(ds))).tolist()})
    t0 = time.perf_counter()
    for i in probe:
        s = ds[i]
        _check(tuple(s["obs"].shape) == (2, OBS_SIZE, OBS_SIZE), f"obs shape {tuple(s['obs'].shape)}")
        _check(tuple(s["action"].shape) == (ds.chunk_size, 3), f"action shape {tuple(s['action'].shape)}")
        _check(tuple(s["state"].shape) == (3,), "state shape")
        _check(s["obs"].dtype == torch.float32 and s["action"].dtype == torch.float32, "dtype")
        _check(0.0 <= float(s["obs"].min()) and float(s["obs"].max()) <= 1.0, "obs range")
        _check(0.0 <= float(s["action"].min()) and float(s["action"].max()) <= 1.0, "action range")
        _check(set(s["action"][:, 2].unique().tolist()) <= {0.0, 1.0}, "pen_down must be 0/1")
    per_item_ms = (time.perf_counter() - t0) / len(probe) * 1e3

    # 경계 복제 패딩: 각 에피소드 마지막 스텝은 마지막 포인트 16개
    for e, ep in enumerate(ds.episodes):
        last = ds[int(ds._offsets[e + 1]) - 1]
        _check(bool(torch.all(last["action"] == torch.from_numpy(ep.traj[-1]))), "edge padding")
        _check(bool(last["action_is_pad"][1:].all()) and not bool(last["action_is_pad"][0]), "is_pad mask")
        first = ds[int(ds._offsets[e])]
        _check(float(first["obs"][0].max()) == 0.0, "t=0 캔버스는 비어 있어야 함")
        _check(torch.allclose(first["state"], torch.tensor(HOME_STATE)), "t=0 state = HOME_STATE")
        # 수집기가 action_chunks를 같이 저장했다면 JS 슬라이싱과 일치하는지 교차 검증
        js = ep.raw.get("action_chunks")
        if js is not None and args.resample_step is None and not args.normalize_strokes:
            js = np.asarray(js, dtype=np.float32)
            _check(js.shape == (len(ep), ds.chunk_size, 3), f"action_chunks shape {js.shape}")
            for t in {0, len(ep) // 2, len(ep) - 1}:
                _check(np.allclose(js[t], slice_action_chunk(ep.traj, t, ds.chunk_size)), "JS/Python chunk 불일치")
            print(f"[ok] {ep.path.name}: JS action_chunks == Python slice_action_chunk")

    # --- DataLoader 배치 ---
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    B = batch["obs"].shape[0]
    _check(tuple(batch["obs"].shape) == (B, 2, OBS_SIZE, OBS_SIZE), "batch obs shape")
    _check(tuple(batch["action"].shape) == (B, ds.chunk_size, 3), "batch action shape")

    print("\nbatch:")
    for k, v in batch.items():
        desc = f"{tuple(v.shape)} {v.dtype}" if torch.is_tensor(v) else f"list[{type(v[0]).__name__}] x{len(v)}"
        print(f"  {k:<14} {desc}")
    print(f"\n__getitem__ avg {per_item_ms:.2f} ms ({len(probe)} probes)")
    print("ALL CHECKS PASSED")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
