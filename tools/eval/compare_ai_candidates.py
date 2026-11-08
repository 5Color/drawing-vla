"""Small, matched closed-loop comparison; output artifacts only."""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw
import rollout_eval as ev
from dataset import HOME_STATE, decode_png_gray, pen_down_runs
from policies import build_policy
from train_policy import pick_device

OUT = ROOT / "outputs"

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    dev = pick_device()
    checkpoints = {tag: torch.load(OUT / f"apple_vla_policy_{tag}.pth", map_location=dev, weights_only=True)
                   for tag in ("v4", "v5b")}
    ref_name = checkpoints["v4"]["val_files"][0]
    ref = json.loads((ROOT / "data" / "raw" / ref_name).read_text(encoding="utf-8"))
    target = decode_png_gray(ref["target_128_png"])
    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
    summary = {"target_source": ref_name, "target_sha256": target_hash, "seed": 0,
               "exec_steps": 12, "max_iters_baseline": 100, "patience_baseline": 6,
               "raw_sources_changed": False, "candidates": []}
    settings = [("v4", 12, None), ("v4", 6, .01), ("v5b", 12, None),
                ("v5b", 6, .01), ("v5b", 4, .01)]
    model_tag, model = None, None
    for tag, q, te_m in settings:
        ckpt = checkpoints[tag]
        cfg = ckpt["config"]
        if tag != model_tag:
            model = build_policy(cfg).to(dev).eval()
            model.load_state_dict(ckpt["model"])
            model_tag = tag
        own_ref = json.loads((ROOT / "data" / "raw" / ckpt["val_files"][0]).read_text(encoding="utf-8"))
        own_hash = hashlib.sha256(decode_png_gray(own_ref["target_128_png"]).tobytes()).hexdigest()
        print(f"[config] {tag}: epoch={ckpt['epoch']} resample={cfg.get('resample_step')} own_target_equal={own_hash == target_hash}", flush=True)
        with torch.no_grad():
            for _ in range(4):
                ev.predict_chunk(model, np.zeros_like(target), target.astype(np.float32)/255,
                                 HOME_STATE, dev)
        ev.INFER_MS.clear()
        torch.manual_seed(0)
        start = time.perf_counter()
        scale = 12 / q
        ex, hist, why = ev.rollout(model, target, dev, max_iters=int(np.ceil(100*scale)),
                                  exec_steps=12, patience=int(np.ceil(6*scale)),
                                  te_m=te_m, query_freq=q)
        sec = time.perf_counter() - start
        infer = np.asarray(ev.INFER_MS)
        sm = ev.smoothness(ex)
        sm.update({f"disp_{k}": v for k, v in ev.smoothness(ex, display=True).items()})
        name = f"{tag}_matched_x12" if te_m is None else f"{tag}_matched_te{te_m:g}q{q}"
        row = {"name": name, "seed": 0, "iters": len(hist), "steps": len(ex),
               "strokes": len(pen_down_runs(ex)), "stop_reason": why,
               "self_terminated": why.startswith("스스로"),
               "score": ev.match_scores(hist[-1]["canvas"], target), "smooth": sm,
               "query_freq": q, "temporal_ensemble": te_m, "te_weight": "exp", "exec_steps": 12,
               "elapsed_sec": sec, "inference_ms_mean": float(infer.mean()),
               "inference_ms_p95": float(np.percentile(infer, 95)),
               "inference_ms_per_step": float(infer.mean()/q),
               "executed": np.round(ex, 5).tolist()}
        artifact = {"ckpt": f"apple_vla_policy_{tag}.pth", "head": cfg.get("head", "mlp"),
                    "epoch": ckpt["epoch"], "config": cfg, "target_source": ref_name,
                    "target_sha256": target_hash, "own_reference_same_target": own_hash == target_hash,
                    "runs": [row]}
        path = OUT / f"rollout_apple_{name}.json"
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep a standard rollout GIF and overlay PNG for each candidate.
        frames = [ev.frame(target, h, i, q) for i, h in enumerate(hist)]
        frames += [frames[-1]] * 12
        frames[0].save(OUT / f"rollout_apple_{name}.gif", save_all=True,
                       append_images=frames[1:], duration=90, loop=0, optimize=True)
        final = hist[-1]["canvas"]
        overlay = np.maximum(np.asarray(ev.to_rgb(target, (255,70,70))),
                             np.asarray(ev.to_rgb(final, (90,220,255))))
        panels = [ev.panel(ev.to_rgb(target), "target", "matched target for every rollout"),
                  ev.panel(ev.to_rgb(final), name, f"F1 {row['score']['f1']:.4f}, {row['strokes']} strokes"),
                  ev.panel(Image.fromarray(overlay), "overlay", "red=target cyan=rollout")]
        sheet = Image.new("RGB", (ev.TILE*3+20, ev.TILE+52), (20,20,26))
        for i, panel in enumerate(panels):
            sheet.paste(panel, (i*(ev.TILE+10),0))
        sheet.save(OUT / f"rollout_apple_{name}_final.png")
        small = {k:v for k,v in row.items() if k != "executed"}
        small.update({"checkpoint": artifact["ckpt"], "epoch": ckpt["epoch"],
                      "resample_step": cfg.get("resample_step"), "rollout_path": str(path)})
        summary["candidates"].append(small)
        (OUT / "ai_comparison_matched.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(small, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
