"""폐루프(Closed-loop) 롤아웃 평가: 빈 캔버스 + 목표 도안에서 출발해 정책이 스스로 그림을 완성한다.

Receding horizon (최대 100회):
  1) (현재 캔버스, 목표 도안) → 미래 16스텝 [16, 3] 예측
  2) 앞 8스텝만 실행 (pen_down = sigmoid(logit) > 0.5, 연속한 펜 다운 구간만 선으로 연결)
  3) 실행 결과를 학습과 같은 렌더러(render_canvas)로 다시 그려 다음 관측으로 사용

실행:  python train/rollout_eval.py [--ckpt outputs/apple_vla_policy.pth] [--exec-steps 8] [--brush]
출력:  outputs/rollout_apple.gif, outputs/rollout_apple_final.png, outputs/rollout_apple.json
       --brush: outputs/rollout_apple_brush.png, outputs/rollout_apple_brush.gif
                (저주파 추론 루프 + 고주파 붓 렌더: 청크마다 실행한 제어점을 Catmull-Rom 스플라인·끝선 빼기로 그림.
                 관측은 그대로 render_canvas — 붓 그림은 보여 주기 전용)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import HOME_STATE, OBS_SIZE, decode_png_gray, pen_down_runs, render_canvas, render_polyline  # noqa: E402
from brush import render as brush_render, runs_from_traj, smooth_run  # noqa: E402
from policies import build_policy  # noqa: E402
from train_policy import OUT_DIR, ROOT, pick_device, tag_suffix  # noqa: E402

TILE = 384
SCALE = TILE // OBS_SIZE


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 평가 지표: 2px(@128) 허용 오차로 도안 커버리지 / 정밀도
# ---------------------------------------------------------------------------
def dilate(m: np.ndarray, r: int) -> np.ndarray:
    out, (H, W) = m.copy(), m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            s = np.zeros_like(m)
            s[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
            out |= s
    return out


def match_scores(canvas: np.ndarray, target: np.ndarray, tol: int = 2) -> dict:
    R, T = canvas > 64, target > 64
    cov = float((T & dilate(R, tol)).sum() / max(1, T.sum()))
    prec = float((R & dilate(T, tol)).sum() / max(1, R.sum())) if R.any() else 0.0
    f1 = 2 * cov * prec / (cov + prec) if cov + prec else 0.0
    return {"coverage": cov, "precision": prec, "f1": f1}


# ---------------------------------------------------------------------------
# 롤아웃
# ---------------------------------------------------------------------------
class IncrementalCanvas:
    """render_canvas(executed)와 같은 결과를, 완료된 획은 캐시하고 진행 중인 획만 다시 그려서 빠르게."""

    def __init__(self):
        self.done = np.zeros((OBS_SIZE, OBS_SIZE), np.uint8)
        self.cur: list[tuple[float, float]] = []

    def add(self, x: float, y: float, pen: float) -> None:
        if pen > 0.5:
            self.cur.append((x, y))
        elif self.cur:
            self.done = np.maximum(self.done, render_polyline(np.asarray(self.cur)))
            self.cur = []

    def image(self) -> np.ndarray:
        return np.maximum(self.done, render_polyline(np.asarray(self.cur))) if self.cur else self.done


INFER_MS: list[float] = []   # 청크 추론 시간 (관측 텐서 준비 + 모델 + CPU로 복사, ms)


def predict_chunk(model, canvas, tgt, state, dev, noise_scale: float = 1.0) -> np.ndarray:
    """→ [16, 3] (x, y, pen 확률). 디퓨전은 호출할 때마다 새로 샘플링."""
    t0 = time.perf_counter()
    obs = torch.from_numpy(np.stack([canvas.astype(np.float32) / 255, tgt]))[None].to(dev)
    st = torch.from_numpy(np.asarray(state, np.float32))[None].to(dev)
    kw = {"noise_scale": noise_scale} if getattr(model, "head_type", "mlp") == "diffusion" else {}
    xy, pen_prob = model.predict(obs, st, **kw)
    out = torch.cat([xy[0], pen_prob[0][:, None]], 1).cpu().numpy()   # .cpu()가 장치 동기화 → 실제 소요 시간
    INFER_MS.append((time.perf_counter() - t0) * 1000)
    return out


def smoothness(traj: np.ndarray, step: float = 0.005, display: bool = False) -> dict:
    """선의 매끈함 (펜 다운 구간을 호 길이 step 간격으로 재샘플한 뒤):
    turn_deg = 이웃 스텝 사이 평균 꺾임 각도(작을수록 매끈), wave = 꺾임 각도의 '변화' 평균(물결·지그재그일수록 큼)."""
    turns, waves = [], []
    T = np.asarray(traj, np.float64)
    for a0, a1 in pen_down_runs(T.astype(np.float32)):   # [start, end) 인덱스
        P = T[a0:a1, :2]
        if display:   # 붓 렌더러가 실제로 그리는 선 (제어점 이동평균 후) 기준
            P = smooth_run(P)
        if len(P) < 3:
            continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s_ = np.r_[0, np.cumsum(seg)]
        if s_[-1] < step * 4:
            continue
        u = np.arange(0, s_[-1], step)
        R = np.c_[np.interp(u, s_, P[:, 0]), np.interp(u, s_, P[:, 1])]
        d = np.diff(R, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        da = np.angle(np.exp(1j * np.diff(ang)))
        turns.append(np.abs(da))
        waves.append(np.abs(np.diff(da)))
    t = np.concatenate(turns) if turns else np.zeros(1)
    w = np.concatenate(waves) if waves else np.zeros(1)
    return {"turn_deg": float(np.degrees(t.mean())), "wave_deg": float(np.degrees(w.mean()))}


@torch.no_grad()
def rollout(model, target: np.ndarray, dev, max_iters: int = 100, exec_steps: int = 8, patience: int = 6,
            skip_pauses: bool = False, te_m: float | None = None, noise_scale: float = 1.0,
            query_freq: int = 1, te_weight: str = "exp"):
    """반환: (실행 궤적 [N,3], 기록, 종료 사유).

    te_m=None : receding horizon — 청크를 예측해 앞 exec_steps개 실행 후 다시 예측 (반복 1회 = exec_steps스텝)
    te_m=값   : ACT식 Temporal Ensembling — query_freq스텝마다 예측해 겹치는 청크를 만들고, 지금 스텝 t에 대한
                예측들(최대 ⌈16/query_freq⌉개)을 가중 평균해 1스텝씩 실행 (반복 1회 = query_freq스텝).
                te_weight="exp": w_i = exp(-m·i) (i=0이 가장 오래된 예측, m<0이면 최신 예측을 더 믿음)
                te_weight="linear": 최신 예측일수록 선형으로 큰 가중치
    """
    H = model.chunk_size
    per_iter = query_freq if te_m is not None else exec_steps
    executed: list[list[float]] = []
    ink = IncrementalCanvas()
    canvas = ink.image()
    state = np.asarray(HOME_STATE, np.float32)  # 학습 데이터의 t=0 상태와 동일
    history, still, stale, buffer = [], 0, 0, []
    reason = f"최대 {max_iters}회 도달"
    tgt = target.astype(np.float32) / 255
    for it in range(max_iters):
        if te_m is None:
            chunk = predict_chunk(model, canvas, tgt, state, dev, noise_scale)
            chunk[:, 2] = (chunk[:, 2] > 0.5).astype(np.float32)
            run = chunk
            if skip_pauses:  # 진단용: 펜 다운인 채 거의 안 움직이는 스텝(사람의 머뭇거림)을 건너뜀
                prev, keep = state, []
                for a in chunk:
                    if a[2] != prev[2] or np.linalg.norm(a[:2] - prev[:2]) * 512 >= 0.5:
                        keep.append(a)
                        prev = a
                run = np.asarray(keep) if keep else chunk[-1:]
            for a in run[:exec_steps]:
                executed.append(a.tolist())
                ink.add(*a)
            state = np.asarray(executed[-1], np.float32)  # 다음 관측의 펜 상태 = 마지막으로 실행한 점
            canvas = ink.image()
        else:
            for j in range(per_iter):
                t = len(executed)
                if j == 0:   # query_freq스텝마다 한 번만 정책 호출
                    chunk = predict_chunk(model, canvas, tgt, state, dev, noise_scale)
                    buffer = [(ti, c) for ti, c in buffer if t - ti < H] + [(t, chunk)]
                buffer = [(ti, c) for ti, c in buffer if t - ti < H]
                cands = np.stack([c[t - ti] for ti, c in buffer])  # 오래된 예측부터
                if te_weight == "linear":
                    w = np.arange(1, len(cands) + 1, dtype=np.float64)
                else:
                    w = np.exp(-te_m * np.arange(len(cands)))
                a = (w[:, None] * cands).sum(0) / w.sum()
                a[2] = float(a[2] > 0.5)
                executed.append(a.tolist())
                ink.add(*a)
                state = a.astype(np.float32)
                canvas = ink.image()
            chunk = chunk.copy()
            chunk[:, 2] = (chunk[:, 2] > 0.5).astype(np.float32)
        prev_canvas = history[-1]["canvas"] if history else np.zeros_like(canvas)
        new_ink = int((canvas.astype(np.int16) - prev_canvas > 32).sum())
        history.append({"chunk": chunk, "canvas": canvas.copy(), "n_exec": len(executed), "new_ink": new_ink})
        # 종료 1: 16스텝 전부 펜-업 + 거의 정지 = 시연 끝(마지막 점 복제 패딩)을 재현 → 스스로 끝냄
        idle = chunk[:, 2].max() < 0.5 and np.abs(chunk[:, :2] - chunk[0, :2]).max() < 0.01
        still = still + 1 if idle and it > 2 else 0
        if still >= 2:
            reason = "스스로 종료 (펜 업 정지)"
            break
        # 종료 2: 새 잉크가 patience회 연속 없음 = 같은 자리 맴돌기
        stale = stale + 1 if new_ink == 0 and it > 2 else 0
        if stale >= patience:
            reason = f"진행 없음 ({patience}회 연속 새 잉크 0)"
            break
    return np.asarray(executed, np.float32), history, reason


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def to_rgb(gray: np.ndarray, tint=(255, 255, 255), scale: int = SCALE) -> Image.Image:
    g = Image.fromarray(gray).resize((gray.shape[1] * scale, gray.shape[0] * scale), Image.Resampling.NEAREST)
    a = np.asarray(g, np.float32)[..., None] / 255
    return Image.fromarray((a * np.array(tint, np.float32)).astype(np.uint8))


def frame(target: np.ndarray, h: dict, it: int, exec_steps: int) -> Image.Image:
    base = np.maximum(np.asarray(to_rgb(target, (70, 70, 85))), np.asarray(to_rgb(h["canvas"])))
    img = Image.fromarray(base)
    dr = ImageDraw.Draw(img)
    for k, (x, y, p) in enumerate(h["chunk"]):
        cx, cy = x * TILE, y * TILE
        executed = k < exec_steps
        col = ((255, 80, 80) if p > 0.5 else (80, 230, 120)) if executed else ((150, 70, 70) if p > 0.5 else (60, 140, 90))
        r = 3 if executed else 2
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
    x, y, p = h["chunk"][exec_steps - 1]
    dr.ellipse((x * TILE - 7, y * TILE - 7, x * TILE + 7, y * TILE + 7), outline=(255, 220, 60), width=2)
    out = Image.new("RGB", (TILE, TILE + 28), (20, 20, 26))
    out.paste(img, (0, 28))
    ImageDraw.Draw(out).text((8, 6), f"iter {it + 1:3d}  step {h['n_exec']:4d}  pen {'DOWN' if p > 0.5 else 'up'}",
                             fill=(230, 230, 230), font=_font(15))
    return out


def panel(img: Image.Image, title: str, sub: str = "") -> Image.Image:
    out = Image.new("RGB", (TILE, TILE + 52), (20, 20, 26))
    out.paste(img, (0, 52))
    d = ImageDraw.Draw(out)
    d.text((8, 6), title, fill=(240, 240, 240), font=_font(17))
    d.text((8, 30), sub, fill=(170, 170, 180), font=_font(13))
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    dev = pick_device()  # 콘솔 첫 줄
    ap = argparse.ArgumentParser(
        description="폐루프 롤아웃 평가",
        epilog="권장 프리셋 (v4 MLP·사과 기준, F1 0.973 · 화면 물결 1.59° · 추론 2배): "
               "--exec-steps 12 --ensemble --brush  (= --query-freq 6 --temporal-ensemble 0.01)")
    ap.add_argument("--ckpt", default=str(OUT_DIR / "apple_vla_policy.pth"))
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--exec-steps", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6, help="새 잉크 없는 반복이 이만큼 이어지면 중단")
    ap.add_argument("--seeds", type=int, default=None, help="롤아웃 반복 횟수 (기본: mlp 1, diffusion 5). GIF는 seed 0")
    ap.add_argument("--skip-pauses", action="store_true",
                    help="진단용: 예측 청크에서 펜 다운 정지 스텝(<0.5px)을 건너뛰고 실행. 출력 이름에 _skip 추가")
    ap.add_argument("--out-suffix", default="", help="출력 파일 이름 뒤에 붙일 문자열")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="M",
                    help="ACT식 Temporal Ensembling 가중치 exp(-M·i) (i=0 가장 오래된 예측). 예: 0.01")
    ap.add_argument("--ensemble", action="store_true",
                    help="Temporal Ensembling 켜기 (--query-freq스텝마다 예측, 겹치는 청크 가중 평균). "
                         "기본 q=6, m=0.01 = 벤치마크 최적 (추론 2배)")
    ap.add_argument("--query-freq", type=int, default=6, help="[앙상블] 정책 호출 주기(스텝, 기본 6). 12/주기 = 추론 횟수 배율")
    ap.add_argument("--te-weight", choices=["exp", "linear"], default="exp", help="[앙상블] 가중치 방식")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="[diffusion] 초기 잡음 배율 (저온 샘플링, 예: 0.5)")
    ap.add_argument("--brush", action="store_true", help="붓 렌더(스플라인·끝선 빼기) 결과 PNG·GIF도 저장")
    ap.add_argument("--brush-fps", type=int, default=30, help="붓 GIF 프레임률 (화면 렌더 주기)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.ensemble and args.temporal_ensemble is None:
        args.temporal_ensemble = 0.01
    qf = args.query_freq if args.temporal_ensemble is not None else args.exec_steps
    if args.temporal_ensemble is not None and not args.ensemble:
        qf = 1   # 예전 동작: 매 스텝 예측
    # 앙상블은 반복 1회 = qf스텝 → 같은 그림 길이·같은 '진행 없음' 시간이 되도록 반복 수·patience를 환산
    scale = args.exec_steps / qf
    max_iters, patience = int(np.ceil(args.max_iters * scale)), int(np.ceil(args.patience * scale))

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=True)
    cfg = ckpt["config"]
    model = build_policy(cfg).to(dev).eval()
    model.load_state_dict(ckpt["model"])
    head = cfg.get("head", "mlp")
    n_seeds = args.seeds or (5 if head == "diffusion" else 1)
    target_name = cfg["target"]
    name = (f"{target_name}{tag_suffix(cfg.get('tag', ''))}{'_skip' if args.skip_pauses else ''}"
            f"{f'_te{args.temporal_ensemble:g}q{qf}' + ('lin' if args.te_weight == 'linear' else '') if args.temporal_ensemble is not None else ''}"
            f"{f'_ns{args.noise_scale:g}' if args.noise_scale != 1.0 else ''}{args.out_suffix}")
    print(f"[ckpt] {Path(args.ckpt).name}: head {head}, epoch {ckpt['epoch']}, val xy {ckpt['val']['xy'] * 512:.1f}px,"
          f" state_dim {cfg.get('state_dim', 0)}", flush=True)

    # 목표 도안(브라우저가 렌더한 128px)과 비교용 사람 시연(검증셋, 학습에 안 쓰인 에피소드)
    ref_path = ROOT / "data" / "raw" / ckpt["val_files"][0]
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    target = decode_png_gray(ref["target_128_png"])
    human = render_canvas(np.asarray(ref["trajectory"], np.float32))
    s_human = match_scores(human, target)

    runs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        ex, hist, why = rollout(model, target, dev, max_iters, args.exec_steps, patience, args.skip_pauses,
                                args.temporal_ensemble, args.noise_scale, qf, args.te_weight)
        sc = match_scores(hist[-1]["canvas"], target)
        sm = smoothness(ex)
        sm.update({f"disp_{k}": v for k, v in smoothness(ex, display=True).items()})
        runs.append({"seed": seed, "executed": ex, "history": hist, "reason": why, "score": sc, "smooth": sm,
                     "strokes": len(pen_down_runs(ex)), "sec": time.perf_counter() - t0})
        print(f"[rollout] seed {seed}: {len(hist):3d}회 · {len(ex):4d}스텝 · 획 {runs[-1]['strokes']}개 · {why}"
              f" | 커버리지 {sc['coverage']:.1%}  정밀도 {sc['precision']:.1%}  F1 {sc['f1']:.3f}"
              f" | 꺾임 {sm['turn_deg']:.2f}° 물결 {sm['wave_deg']:.2f}° (화면 {sm['disp_turn_deg']:.2f}°/{sm['disp_wave_deg']:.2f}°)",
              flush=True)
    if n_seeds > 1:
        f1s = np.array([r["score"]["f1"] for r in runs])
        print(f"[score]   롤아웃 F1 평균 {f1s.mean():.3f} (최저 {f1s.min():.3f} / 최고 {f1s.max():.3f}),"
              f" 스스로 종료 {sum(r['reason'].startswith('스스로') for r in runs)}/{n_seeds}", flush=True)
    if INFER_MS:
        ms = np.asarray(INFER_MS[3:] or INFER_MS)   # 첫 몇 번은 워밍업(커널 컴파일) 제외
        per_step = ms.mean() / qf                         # 실행 1스텝당 추론 비용
        base = ms.mean() / args.exec_steps                 # 기준: 청크마다 exec_steps 실행 (앙상블 없음)
        print(f"[speed]   추론 {ms.mean():.1f} ms/호출 (p95 {np.percentile(ms, 95):.1f}) · {qf}스텝마다 호출 →"
              f" 스텝당 {per_step:.3f} ms = 기준(12스텝마다)의 {per_step / base:.1f}배"
              f" · 최대 {1000 / per_step:.0f} 스텝/초 ({dev})", flush=True)
    sm_h, sm_hd = smoothness(np.asarray(ref["trajectory"], np.float32)), smoothness(np.asarray(ref["trajectory"], np.float32), display=True)
    print(f"[smooth]  사람시연 꺾임 {sm_h['turn_deg']:.2f}° 물결 {sm_h['wave_deg']:.2f}°"
          f" (화면 {sm_hd['turn_deg']:.2f}°/{sm_hd['wave_deg']:.2f}°)", flush=True)
    print(f"[score]   사람시연 커버리지 {s_human['coverage']:.1%}  정밀도 {s_human['precision']:.1%}  F1 {s_human['f1']:.3f}"
          f"  ({ref_path.name}, 검증셋)", flush=True)

    r0 = runs[0]
    executed, history, reason, s_roll, n_strokes = r0["executed"], r0["history"], r0["reason"], r0["score"], r0["strokes"]
    final = history[-1]["canvas"]
    done = reason.startswith("스스로")

    # GIF: 반복마다 한 프레임 (흰색 = 캔버스, 큰 점 = 실행한 8스텝, 작은 점 = 버린 8스텝, 빨강 = 펜 다운, 초록 = 펜 업)
    frames = [frame(target, h, i, qf) for i, h in enumerate(history)]
    frames += [frames[-1]] * 12
    gif = OUT_DIR / f"rollout_{name}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)

    overlay = np.zeros((OBS_SIZE * SCALE, OBS_SIZE * SCALE, 3), np.uint8)
    overlay = np.maximum(overlay, np.asarray(to_rgb(target, (255, 70, 70))))
    overlay = np.maximum(overlay, np.asarray(to_rgb(final, (90, 220, 255))))
    panels = [
        panel(to_rgb(target), "target", "obs[1]"),
        panel(to_rgb(final), "policy rollout (closed-loop)",
              f"cov {s_roll['coverage']:.0%}  prec {s_roll['precision']:.0%}  F1 {s_roll['f1']:.2f}  · {n_strokes} strokes"),
        panel(Image.fromarray(overlay), "overlay", "red = target, cyan = rollout"),
        panel(to_rgb(human), "human demo (val, unseen)",
              f"cov {s_human['coverage']:.0%}  prec {s_human['precision']:.0%}  F1 {s_human['f1']:.2f}"),
    ]
    sheet = Image.new("RGB", (TILE * 4 + 30, TILE + 52), (20, 20, 26))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * (TILE + 10), 0))
    png = OUT_DIR / f"rollout_{name}_final.png"
    sheet.save(png)
    outs = [gif, png]

    if args.brush:   # 고주파 렌더: 추론 1회(청크)당 여러 프레임으로 실행 제어점을 붓으로 이어 그림
        size = 512
        bpng = OUT_DIR / f"rollout_{name}_brush.png"
        brush_render(runs_from_traj(executed), size).save(bpng)
        sub = max(1, round(args.brush_fps * 0.1))       # 청크 하나를 약 0.1초에 걸쳐 그린다고 보고 나눔
        frames_b, n = [], len(executed)
        for h in history:
            end = min(h["n_exec"], n)
            start = max(0, end - (qf if args.temporal_ensemble is not None else args.exec_steps))
            for j in range(1, sub + 1):
                k = start + max(1, round((end - start) * j / sub))
                frames_b.append(brush_render(runs_from_traj(executed[:k]), 256))
        frames_b += [frames_b[-1]] * args.brush_fps
        bgif = OUT_DIR / f"rollout_{name}_brush.gif"
        frames_b[0].save(bgif, save_all=True, append_images=frames_b[1:], duration=round(1000 / args.brush_fps),
                         loop=0, optimize=True)
        outs += [bpng, bgif]

    if n_seeds > 1:  # 시드별 최종 그림: 디퓨전이 매번 어떤 스타일을 고르는지
        T2 = OBS_SIZE * 2
        grid = Image.new("RGB", (n_seeds * (T2 + 8), T2 + 44), (20, 20, 26))
        for k, r in enumerate(runs):
            img = Image.fromarray(np.maximum(np.asarray(to_rgb(target, (70, 70, 85), 2)), np.asarray(to_rgb(r["history"][-1]["canvas"], scale=2))))
            x0 = k * (T2 + 8)
            grid.paste(img, (x0, 44))
            dd = ImageDraw.Draw(grid)
            dd.text((x0 + 4, 4), f"seed {k}  F1 {r['score']['f1']:.2f}  {r['strokes']} strokes", fill=(235, 235, 235), font=_font(13))
            dd.text((x0 + 4, 24), "self-stop" if r["reason"].startswith("스스로") else
                    "stuck" if r["reason"].startswith("진행") else "max iters", fill=(170, 170, 180), font=_font(12))
        seeds_png = OUT_DIR / f"rollout_{name}_seeds.png"
        grid.save(seeds_png)
        outs.append(seeds_png)

    (OUT_DIR / f"rollout_{name}.json").write_text(json.dumps({
        "ckpt": Path(args.ckpt).name, "head": head, "epoch": ckpt["epoch"], "human_val": s_human,
        "runs": [{"seed": r["seed"], "iters": len(r["history"]), "steps": len(r["executed"]), "strokes": r["strokes"],
                  "stop_reason": r["reason"], "self_terminated": r["reason"].startswith("스스로"), "score": r["score"],
                  "smooth": r["smooth"], "query_freq": qf,
                  "executed": np.round(r["executed"], 5).tolist()} for r in runs],
    }), encoding="utf-8")
    print("\n".join(f"[out] {p}" for p in outs), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
