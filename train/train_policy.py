"""사과 Action Chunking 정책 학습: 경량 ConvNet 인코더 + (MLP 또는 디퓨전) 헤드 → [16, 3] action chunk.

  obs    (B, 2, 128, 128)  ch0 = traj[:t] 렌더 캔버스(t번 액션 직전, 진행 중인 그림), ch1 = 목표 도안
  state  (B, 3)            현재 펜 상태 traj[t-1] = (x, y, pen)
  action (B, 16, 3)        traj[t : t+16] = (x, y, pen_down)

버전
  v1: --head mlp --no-state --pen-up-weight 1 --no-aug      이미지만, L1 + 0.2·BCE
  v2: --head mlp (기본값)                                    + 펜 상태 입력, 펜 업 가중치 ×10, 캔버스 증강·상태 잡음
  v3: --head diffusion                                       v2 입력 + 디퓨전 헤드 (EMA, lr 3e-4 코사인)
  v4: + --resample-step 0.004 --state-embed 64               거리 기준 재샘플(감속·머뭇거림 제거) + 펜 상태 임베딩

실행:  python train/train_policy.py --tag v3 --head diffusion --epochs 100 --stride 4
출력:  outputs/apple_vla_policy[_tag].pth, outputs/loss_curve[_tag].png, outputs/train_log[_tag].json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import HOME_STATE, OBS_SIZE, DrawingVLADataset, slice_action_chunk  # noqa: E402
from policies import DiffusionPolicy, TransformerPolicy, VisuomotorPolicy, build_policy  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PX = 512  # 오차를 512px 캔버스 기준 픽셀로 표시


# ---------------------------------------------------------------------------
# 디바이스: Intel Arc XPU → CUDA → CPU
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    if os.environ.get("DEVICE") != "cpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        dev, name = torch.device("xpu"), torch.xpu.get_device_name(0)
    elif torch.cuda.is_available():
        dev, name = torch.device("cuda"), torch.cuda.get_device_name(0)
    else:
        dev, name = torch.device("cpu"), platform.processor() or "CPU"
    print(f"[device] {dev.type.upper()}: {name}", flush=True)
    if dev.type == "cpu" and hasattr(torch, "xpu") and "+cpu" in torch.__version__:
        print(f"         (XPU 사용 불가: 설치된 PyTorch가 CPU 전용 빌드 {torch.__version__})", flush=True)
    return dev


def sync(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def tag_suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


# ---------------------------------------------------------------------------
# 증강: 롤아웃에서 정책이 스스로 그린 '약간 어긋난 그림'에서도 도안을 따라가도록
# ---------------------------------------------------------------------------
def augment(obs: torch.Tensor, state: torch.Tensor, scale: float = 0.05, shift_px: float = 3.0,
            state_noise: float = 0.005) -> tuple[torch.Tensor, torch.Tensor]:
    """캔버스 채널만 무작위 확대/축소·이동 (목표 도안과 액션은 그대로) + 펜 위치 잡음."""
    B, dev = obs.shape[0], obs.device
    s = 1 + (torch.rand(B, device=dev) * 2 - 1) * scale
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * shift_px * 2 / obs.shape[-1]
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, (B, 1, *obs.shape[-2:]), align_corners=False)
    canv = F.grid_sample(obs[:, :1], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    state = state.clone()
    state[:, :2] = (state[:, :2] + torch.randn(B, 2, device=dev) * state_noise).clamp(0, 1)
    return torch.cat([canv, obs[:, 1:]], 1), state


# ---------------------------------------------------------------------------
# 데이터: 진행 중인 캔버스를 스텝마다 미리 렌더해 uint8로 캐시 → 학습 시 디바이스 메모리에 상주
# ---------------------------------------------------------------------------
def build_arrays(ds: DrawingVLADataset, cache_path: Path) -> dict[str, np.ndarray]:
    # Cache the processed inputs, not filenames: edits and stroke normalization can keep T unchanged.
    digest = hashlib.sha256(json.dumps({
        "version": 2, "chunk_size": ds.chunk_size, "obs_size": ds.obs_size,
        "stroke_width_px": ds.stroke_width_px, "supersample": ds.supersample,
        "lengths": [len(ep) for ep in ds.episodes],
    }, sort_keys=True).encode())
    for ep in ds.episodes:
        digest.update(ep.traj.tobytes())
        digest.update(ep.target.tobytes())
    key = digest.hexdigest()
    if cache_path.exists():
        with np.load(cache_path) as z:
            if str(z["key"]) == key:
                print(f"[data] 캐시 사용: {cache_path.name}", flush=True)
                return add_state({k: z[k] for k in z.files if k != "key"})
    N = len(ds)
    canv = np.zeros((N, ds.obs_size, ds.obs_size), np.uint8)
    act = np.zeros((N, ds.chunk_size, 3), np.float32)
    ep_id = np.zeros(N, np.int32)
    tt = np.zeros(N, np.int32)
    t0, i = time.perf_counter(), 0
    for e, ep in enumerate(ds.episodes):
        for t in range(len(ep)):
            canv[i] = ds.canvas_at(ep, t)
            act[i] = slice_action_chunk(ep.traj, t, ds.chunk_size)
            ep_id[i], tt[i] = e, t
            i += 1
    targets = np.stack([ep.target for ep in ds.episodes])
    print(f"[data] 진행 중 캔버스 {N}장 렌더 {time.perf_counter() - t0:.1f}s → {cache_path.name}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, key=key, canv=canv, act=act, ep_id=ep_id, t=tt, targets=targets)
    return add_state({"canv": canv, "act": act, "ep_id": ep_id, "t": tt, "targets": targets})


def add_state(arr: dict) -> dict:
    """state[i] = traj[t-1] = 같은 에피소드 직전 샘플의 첫 액션 (t=0이면 HOME_STATE). 샘플은 에피소드 순서로 연속 저장됨."""
    prev = np.concatenate([np.asarray([HOME_STATE], np.float32), arr["act"][:-1, 0]])
    arr["state"] = np.where((arr["t"] > 0)[:, None], prev, np.asarray(HOME_STATE, np.float32)).astype(np.float32)
    return arr


def to_device(arr: dict, dev: torch.device) -> dict[str, torch.Tensor]:
    return {
        "canv": torch.from_numpy(arr["canv"]).to(dev),          # uint8 상주 (~400MB)
        "act": torch.from_numpy(arr["act"]).to(dev),
        "state": torch.from_numpy(arr["state"]).to(dev),
        "targets": torch.from_numpy(arr["targets"]).to(dev),
        "ep_id": torch.from_numpy(arr["ep_id"].astype(np.int64)).to(dev),
    }


def make_batch(d: dict[str, torch.Tensor], idx: torch.Tensor):
    canv = d["canv"][idx].float().div_(255)
    tgt = d["targets"][d["ep_id"][idx]].float().div_(255)
    return torch.stack([canv, tgt], 1), d["state"][idx], d["act"][idx]


# ---------------------------------------------------------------------------
# 평가: 학습 손실(헤드별) + 실제 예측(디퓨전은 샘플링)으로 좌표 오차·펜 정확도
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_loss(model, d, idx: torch.Tensor, bs: int = 512) -> float:
    model.eval()
    tot = torch.zeros((), device=idx.device)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        obs, state, action = make_batch(d, b)
        tot += model.loss(obs, state, action)[0] * len(b)
    return tot.item() / len(idx)


@torch.no_grad()
def val_predict(model, d, idx: torch.Tensor, bs: int = 512) -> dict:
    model.eval()
    dev = idx.device
    acc = {k: torch.zeros((), device=dev) for k in ("xy", "pen_ok", "up_hit", "up_n", "first")}
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        obs, state, action = make_batch(d, b)
        xy, pen_prob = model.predict(obs, state)
        up = action[..., 2] < 0.5
        acc["xy"] += (xy - action[..., :2]).abs().mean() * len(b)
        acc["pen_ok"] += ((pen_prob > 0.5) == ~up).float().mean() * len(b)
        acc["up_hit"] += ((pen_prob <= 0.5) & up).sum()
        acc["up_n"] += up.sum()
        acc["first"] += (xy[:, 0] - action[:, 0, :2]).norm(dim=-1).sum() * PX
    v = {k: t.item() for k, t in acc.items()}
    N = len(idx)
    return {"xy": v["xy"] / N, "pen_acc": v["pen_ok"] / N, "pen_up_recall": v["up_hit"] / max(1, v["up_n"]),
            "first_px": v["first"] / N}


def plot_curves(log: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in log]
    pr = [r for r in log if "xy" in r["val"]]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(ep, [r["train"]["loss"] for r in log], label="train")
    ax[0].plot(ep, [r["val"]["loss"] for r in log], label="val")
    ax[0].set_title(f"{title}: training loss"); ax[0].set_yscale("log"); ax[0].legend()
    ax[1].plot([r["epoch"] for r in pr], [r["val"]["xy"] * PX for r in pr], "o-", label="val (predicted)")
    if "xy" in log[0]["train"]:
        ax[1].plot(ep, [r["train"]["xy"] * PX for r in log], label="train")
    ax[1].set_title("xy L1 (px @512)"); ax[1].legend()
    ax[2].plot([r["epoch"] for r in pr], [r["val"]["pen_acc"] * 100 for r in pr], "o-", label="val pen acc")
    ax[2].plot([r["epoch"] for r in pr], [r["val"]["pen_up_recall"] * 100 for r in pr], "o-", label="val pen-up recall")
    ax[2].set_title("pen_down (%)"); ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    dev = pick_device()  # 콘솔 첫 줄
    ap = argparse.ArgumentParser(description="사과 Action Chunking 정책 학습")
    ap.add_argument("--target", default="apple")
    ap.add_argument("--tag", default="", help="출력 파일 접미사 (예: v3 → apple_vla_policy_v3.pth)")
    ap.add_argument("--head", choices=["mlp", "diffusion", "transformer"], default="mlp")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="기본: mlp 1e-3 고정, diffusion 3e-4 + 코사인 감쇠")
    ap.add_argument("--stride", type=int, default=1,
                    help="에폭마다 k스텝 중 1개만 사용(시작 위치는 매 에폭 랜덤). 연속 스텝은 거의 같은 샘플")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--channels", default="32,64,128,256")
    ap.add_argument("--no-state", action="store_true", help="펜 상태 입력 끄기 (v1)")
    ap.add_argument("--pen-up-weight", type=float, default=10.0, help="[mlp] 펜 업 스텝 BCE 가중치 (v1 = 1)")
    ap.add_argument("--no-aug", action="store_true", help="캔버스 증강·상태 잡음 끄기 (v1)")
    ap.add_argument("--eval-every", type=int, default=5, help="[diffusion] 샘플링 검증 주기(에폭)")
    ap.add_argument("--resample-step", type=float, default=None,
                    help="궤적을 호 길이 간격으로 재샘플 (예: 0.004 ≈ 512px 기준 2px). 사람의 감속·머뭇거림 제거 (v4)")
    ap.add_argument("--normalize-strokes", action="store_true", help="로드 시 획 순서·방향 정규화 (원본 보존)")
    ap.add_argument("--state-embed", type=int, default=0, help="펜 상태를 이 차원으로 키워서 붙임 (0 = 그대로, v4는 64)")
    ap.add_argument("--resume", action="store_true", help="outputs/train_state[_tag].pt에서 이어서 학습")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files-from", help="이 텍스트 파일에 적힌 시연 파일 이름만 사용 (한 줄에 하나, data/raw 기준)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    diffusion = args.head == "diffusion"
    is_decay = diffusion or args.head == "transformer"
    lr = args.lr or (3e-4 if is_decay else 1e-3)

    # 에피소드 단위 Train/Val 분할 (같은 그림의 인접 스텝이 양쪽에 섞이지 않게)
    files = sorted((ROOT / "data" / "raw").glob(f"demo_{args.target}_*.json"))
    if args.files_from:   # 일부 시연만 (예: 획 구성이 다른 시연을 빼고 비교 학습)
        keep = {ln.strip() for ln in Path(args.files_from).read_text(encoding="utf-8").splitlines() if ln.strip()}
        files = [f for f in files if f.name in keep]
    shuffled = files[:]
    random.Random(args.seed).shuffle(shuffled)
    n_val = max(1, round(len(files) * args.val_ratio))
    val_files, train_files = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])
    ds = DrawingVLADataset(files=train_files + val_files, resample_step=args.resample_step,
                          normalize_strokes=args.normalize_strokes)
    loaded_paths = {ep.path for ep in ds.episodes}
    train_files = [f for f in train_files if f in loaded_paths]
    val_files = [f for f in val_files if f in loaded_paths]
    if not train_files or not val_files:
        raise ValueError("training and validation each require at least one valid episode")
    print(f"[data] {args.target}: 에피소드 {len(ds.episodes)}개 → train {len(train_files)} / val {len(val_files)}", flush=True)
    rs = f"_r{args.resample_step:g}" if args.resample_step else ""
    if args.normalize_strokes:
        rs += "_normalized"
    arr = build_arrays(ds, OUT_DIR / f"cache_{args.target}{rs}.npz")
    val_paths = set(val_files)
    val_ep_ids = [e for e, ep in enumerate(ds.episodes) if ep.path in val_paths]
    is_val = np.isin(arr["ep_id"], val_ep_ids)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    d = to_device(arr, dev)
    val_idx_t = torch.from_numpy(val_idx).to(dev)
    # 디퓨전 샘플링 검증은 비용이 커서 고정된 1024개 부분집합으로
    pred_idx_t = val_idx_t if not diffusion else torch.from_numpy(
        np.sort(np.random.default_rng(0).choice(val_idx, min(1024, len(val_idx)), replace=False))).to(dev)
    print(f"[data] 샘플 train {len(train_idx)} / val {len(val_idx)}  (에폭당 train {len(train_idx) // args.stride}개,"
          f" stride {args.stride}) · 펜 업 비율 {1 - arr['act'][..., 2].mean():.1%}", flush=True)

    cfg = {"head": args.head, "chunk_size": 16, "channels": [int(c) for c in args.channels.split(",")],
           "obs_size": OBS_SIZE, "state_dim": 0 if args.no_state else 3, "target": args.target, "tag": args.tag,
           "pen_up_weight": args.pen_up_weight if not diffusion else None, "aug": not args.no_aug,
           "state_embed": 0 if args.no_state else args.state_embed, "resample_step": args.resample_step,
           "normalize_strokes": args.normalize_strokes}
    if diffusion:
        cfg["diffusion"] = {"hidden": 512, "n_blocks": 4, "train_steps": 100, "infer_steps": 10, "noise_repeats": 4}
    elif args.head == "transformer":
        cfg["transformer"] = {"d_model": 256, "nhead": 4, "num_layers": 3, "dim_feedforward": 512}
    model = build_policy(cfg).to(dev)
    ema = copy.deepcopy(model).eval() if diffusion else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(len(train_idx) / args.stride / args.batch_size)
    total_steps, warmup = steps_per_epoch * args.epochs, 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, (lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
                                              if is_decay else (lambda s: 1.0))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] head={args.head} channels={tuple(cfg['channels'])} state_dim={cfg['state_dim']}"
          f" state_embed={cfg['state_embed']} resample={args.resample_step} aug={cfg['aug']}"
          f" lr={lr:g}{' cosine+EMA' if diffusion else ''}"
          f"{'' if diffusion else f' pen_up_weight={args.pen_up_weight}'}  params={n_params / 1e6:.2f}M  FP32", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    sfx = tag_suffix(args.tag)
    ckpt_path = OUT_DIR / f"{args.target}_vla_policy{sfx}.pth"
    state_path = OUT_DIR / f"train_state{sfx}.pt"  # 에폭마다 덮어씀: XPU 장치 손실 등으로 죽으면 --resume
    log, best, step, start = [], float("inf"), 0, 1
    t_all = time.perf_counter()
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if ema is not None:
            ema.load_state_dict(st["ema"])
        log, best, step, start = st["log"], st["best"], st["step"], st["epoch"] + 1
        np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"].cpu())
        print(f"[resume] {state_path.name}: {st['epoch']}에폭까지 학습된 상태에서 이어서 시작", flush=True)

    def save(net, epoch, va):
        torch.save({"model": net.state_dict(), "config": cfg, "epoch": epoch, "val": va,
                    "train_files": [f.name for f in train_files], "val_files": [f.name for f in val_files]}, ckpt_path)

    for epoch in range(start, args.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        phase = np.random.randint(args.stride)
        sel = train_idx[arr["t"][train_idx] % args.stride == phase]
        perm = torch.from_numpy(np.random.permutation(sel)).to(dev)
        tr = {"loss": torch.zeros((), device=dev)}
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            if len(b) < 2:  # BatchNorm은 배치 2 이상 필요
                continue
            obs, state, action = make_batch(d, b)
            if cfg["aug"]:
                obs, state = augment(obs, state)
            loss, parts = model.loss(obs, state, action)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if ema is not None:  # 지수 이동 평균 가중치 (디퓨전 샘플 품질 안정화)
                decay = min(0.999, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), model.parameters()):
                        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tr["loss"] += loss.detach() * len(b)
            for k, v in parts.items():
                tr[k] = tr.get(k, torch.zeros((), device=dev)) + v.detach() * len(b)
        tr = {k: v.item() / len(perm) for k, v in tr.items()}  # 에폭당 1회만 CPU로 동기화
        net = ema if diffusion else model
        va = {"loss": val_loss(net, d, val_idx_t)}
        if not diffusion or epoch % args.eval_every == 0 or epoch == args.epochs:
            va.update(val_predict(net, d, pred_idx_t))
        sync(dev)
        dt = time.perf_counter() - t_ep
        # 체크포인트 기준: mlp = 검증 손실, diffusion = 샘플링한 청크의 좌표 오차
        score = va.get("xy", float("inf")) if diffusion else va["loss"]
        mark = ""
        if score < best:
            best, mark = score, "  *best"
            save(net, epoch, va)
        log.append({"epoch": epoch, "train": tr, "val": va, "sec": dt, "lr": sched.get_last_lr()[0]})
        pred = (f", xy {va['xy'] * PX:4.1f}px, 첫스텝 {va['first_px']:4.1f}px, pen acc {va['pen_acc']:.1%},"
                f" 펜업 재현율 {va['pen_up_recall']:.0%}") if "xy" in va else ""
        trs = f"xy {tr['xy'] * PX:4.1f}px, pen {tr['pen']:.3f}" if "xy" in tr else f"eps {tr['eps']:.4f}"
        print(f"epoch {epoch:03d}/{args.epochs} | train {tr['loss']:.4f} ({trs}) | val {va['loss']:.4f}{pred}"
              f" | {dt:4.1f}s{mark}", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "log": log, "best": best, "step": step,
                    "epoch": epoch, "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, state_path)

    plot_curves(log, OUT_DIR / f"loss_curve{sfx}.png", args.tag or "v1")
    (OUT_DIR / f"train_log{sfx}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    scored = [r for r in log if "xy" in r["val"]]
    best_ep = min(scored, key=lambda r: r["val"]["xy"] if diffusion else r["val"]["loss"])
    print(f"\n[done] {time.perf_counter() - t_all:.0f}s · best epoch {best_ep['epoch']} (xy {best_ep['val']['xy'] * PX:.1f}px,"
          f" 펜업 재현율 {best_ep['val']['pen_up_recall']:.0%}) → {ckpt_path}", flush=True)
    print(f"[done] {OUT_DIR / f'loss_curve{sfx}.png'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
