"""표시용 붓 렌더러: 정책이 낸 제어점(펜 다운 구간)을 사람 일러스트처럼 매끈하고 굵기가 변하는 선으로 그린다.

  - 보간: centripetal Catmull-Rom 스플라인 (α=0.5). 제어점을 '그대로 지나가는' 보간이라 곡선 맞춤(fitting)이 아니다.
          → CLAUDE.md의 '베지에 곡선 피팅/세그멘테이션 금지'(학습 데이터 전처리 규칙)와 충돌하지 않는다.
  - 굵기: 기본 굵기 × 끝선 빼기(시작·끝을 가늘게) × 속도(느릴수록 굵게, 종 모양 가감속을 굵기로) × 필압(있으면)
  - 렌더: 4배 슈퍼샘플로 원+사다리꼴을 그리고 줄여서 안티에일리어싱.

주의: 이 렌더러는 '보여 주기' 전용이다. 정책의 관측(obs ch0)은 반드시 dataset.render_canvas(학습과 같은 규격)로 만든다.
      붓 그림을 관측으로 넣으면 학습 분포와 달라져 정책이 무너진다.

  python train/brush.py --demo data/raw/demo_apple_….json   사람 시연을 붓으로 다시 그려 보기 (outputs/brush_demo.png)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SUPER = 4


def catmull_rom(pts: np.ndarray, per_seg: int = 8, alpha: float = 0.5) -> np.ndarray:
    """[N, D] 제어점 → [M, D] 곡선 위 점 (모든 제어점을 지남). D의 앞 두 열을 좌표로 거리 계산, 나머지 열(필압 등)도 같이 보간."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 3:
        return pts.copy()
    # 연속 중복 점 제거 (0 길이 구간은 매개변수가 정의되지 않음)
    keep = np.r_[True, np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) > 1e-9]
    pts = pts[keep]
    if len(pts) < 3:
        return pts.copy()
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])   # 양 끝 가상 점
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1[:2] - p0[:2]) ** alpha, 1e-9)
        t2 = t1 + max(np.linalg.norm(p2[:2] - p1[:2]) ** alpha, 1e-9)
        t3 = t2 + max(np.linalg.norm(p3[:2] - p2[:2]) ** alpha, 1e-9)
        for t in np.linspace(t1, t2, per_seg, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(pts[-1])
    return np.asarray(out)


def widths(curve: np.ndarray, base: float, taper: float = 0.18, min_ratio: float = 0.25,
           speed: np.ndarray | None = None, pressure: np.ndarray | None = None, speed_gain: float = 0.35) -> np.ndarray:
    """곡선 위 점마다 선 굵기.
    taper    : 선 길이 중 시작·끝에서 가늘어지는 비율 (0.18 = 앞뒤 18%)
    speed    : 점마다 속도(아무 단위). 평균보다 느리면 굵게, 빠르면 가늘게 (±speed_gain)
    pressure : 점마다 필압 0~1 (스타일러스 시연). 주어지면 굵기에 곱함 (0.35~1.0)"""
    n = len(curve)
    if n < 2:
        return np.full(n, base)
    seg = np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = max(s[-1], 1e-9)
    u = s / L
    ease = lambda v: np.sin(np.clip(v, 0, 1) * np.pi / 2) ** 0.8     # 부드럽게 0 → 1
    tp = np.minimum(ease(u / taper), ease((1 - u) / taper)) if taper > 0 else np.ones(n)
    w = base * (min_ratio + (1 - min_ratio) * tp)
    if speed is not None and len(speed) == n:
        v = np.asarray(speed, np.float64)
        rel = v / max(np.median(v[v > 0]) if (v > 0).any() else 1.0, 1e-9)
        w *= 1 + speed_gain * np.clip(1 - rel, -1, 1)
    if pressure is not None and len(pressure) == n:
        w *= 0.35 + 0.65 * np.clip(np.asarray(pressure, np.float64), 0, 1)
    return w


def smooth_run(run: np.ndarray, passes: int = 2, width: int = 5) -> np.ndarray:
    """제어점 잔떨림 제거: 이동평균을 passes번 (양 끝점 유지). 정책 출력의 스텝 간 흔들림이 구슬 모양으로 보이지 않게."""
    out = np.asarray(run, np.float64).copy()
    if len(out) < width or passes <= 0:
        return out
    k = np.ones(width) / width
    h = width // 2
    for _ in range(passes):
        padded = np.vstack([np.repeat(out[:1], h, 0), out, np.repeat(out[-1:], h, 0)])
        sm = np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(out.shape[1])], 1)
        sm[0], sm[-1] = out[0], out[-1]
        out = sm
    return out


def curvature_speed(curve: np.ndarray) -> np.ndarray:
    """시간 정보가 없을 때(정책 출력) 쓰는 속도 추정: 꺾이는 곳에서 느려진다는 가정 (2/3 멱법칙 근사)."""
    n = len(curve)
    if n < 3:
        return np.ones(n)
    d1 = np.gradient(curve[:, :2], axis=0)
    d2 = np.gradient(d1, axis=0)
    k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
    v = (1.0 / (1.0 + k * 0.02)) ** (1 / 3)
    win = min(len(v), 15)
    return np.convolve(np.pad(v, win // 2, mode="edge"), np.ones(win) / win, mode="valid")[:n]


def draw_stroke(dr: ImageDraw.ImageDraw, curve: np.ndarray, w: np.ndarray, size: int, color) -> None:
    """곡선과 굵기 → 원(관절) + 사다리꼴(구간)."""
    S = size * SUPER
    P = curve[:, :2] * S
    R = w * SUPER / 2
    for (x, y), r in zip(P, R):
        dr.ellipse((x - r, y - r, x + r, y + r), fill=color)
    for (a, b, ra, rb) in zip(P[:-1], P[1:], R[:-1], R[1:]):
        d = b - a
        nrm = math.hypot(*d)
        if nrm < 1e-9:
            continue
        nx, ny = -d[1] / nrm, d[0] / nrm
        dr.polygon([(a[0] + nx * ra, a[1] + ny * ra), (b[0] + nx * rb, b[1] + ny * rb),
                    (b[0] - nx * rb, b[1] - ny * rb), (a[0] - nx * ra, a[1] - ny * ra)], fill=color)


def runs_from_traj(traj: np.ndarray) -> list[np.ndarray]:
    """[N, 3+] (x, y, pen, …) → 펜 다운 구간 목록 (각 [M, 2+])."""
    out, cur = [], []
    for row in np.asarray(traj):
        if row[2] > 0.5:
            cur.append(np.r_[row[:2], row[3:]])
        elif cur:
            out.append(np.asarray(cur)); cur = []
    if cur:
        out.append(np.asarray(cur))
    return out


def render(runs: list[np.ndarray], size: int = 512, base_px: float | None = None, bg=(250, 248, 243), color=(25, 25, 32),
           per_seg: int = 8, taper: float = 0.18, times: list[np.ndarray] | None = None,
           pressures: list[np.ndarray] | None = None, smooth: int = 2) -> Image.Image:
    """펜 다운 구간들([M, 2] 정규화 좌표) → 붓 그림 RGB.
    times     : 구간별 점 시각(ms) — 있으면 실제 속도로 굵기 (사람 시연), 없으면 곡률로 추정 (정책 출력)
    pressures : 구간별 점 필압 0~1 (스타일러스 시연)
    smooth    : 제어점 이동평균 횟수 (5점 창). 0이면 제어점을 그대로 지남"""
    base = base_px if base_px is not None else size / 512 * 7
    img = Image.new("RGB", (size * SUPER, size * SUPER), bg)
    dr = ImageDraw.Draw(img)
    for i, run in enumerate(runs):
        run = np.asarray(run, np.float64)
        extra = [run[:, :2]]
        if times is not None:
            extra.append(np.asarray(times[i], np.float64)[:, None])
        if pressures is not None:
            extra.append(np.asarray(pressures[i], np.float64)[:, None])
        curve = catmull_rom(smooth_run(np.hstack(extra), smooth), per_seg)
        col = 2
        speed = None
        if times is not None:
            t = curve[:, col]; col += 1
            dist = np.r_[0, np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1)]
            dt = np.r_[1, np.diff(t)]
            speed = dist / np.maximum(dt, 1e-3)
            speed = np.convolve(speed, np.ones(9) / 9, mode="same")    # 측정 잡음 완화
        else:
            speed = curvature_speed(curve)
        pr = curve[:, col] if pressures is not None else None
        draw_stroke(dr, curve, widths(curve, base, taper, speed=speed, pressure=pr), size, color)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser(description="사람 시연을 붓 렌더러로 다시 그려 보기")
    ap.add_argument("--demo", required=True, help="data/raw/demo_*.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "outputs" / "brush_demo.png"))
    args = ap.parse_args()
    d = json.loads(Path(args.demo).read_text(encoding="utf-8"))
    st = d["strokes"]
    runs = [np.asarray([p[:2] for p in s["points"]]) for s in st]
    times = [np.asarray([p[2] for p in s["points"]]) for s in st]
    has_p = (d.get("meta") or {}).get("has_pressure")
    pres = [np.asarray([p[3] for p in s["points"]]) for s in st] if has_p else None
    lines = render(runs, args.size)                       # 곡률 기반 (정책 출력과 같은 조건)
    real = render(runs, args.size, times=times, pressures=pres)   # 실제 시간·필압
    sheet = Image.new("RGB", (args.size * 2 + 10, args.size), (20, 20, 26))
    sheet.paste(lines, (0, 0)); sheet.paste(real, (args.size + 10, 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"[brush] {args.out}  (왼쪽: 곡률로 굵기 추정 · 오른쪽: 실제 속도{'·필압' if has_p else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
