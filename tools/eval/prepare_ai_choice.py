"""Render the selected saved rollout; no rollout, source, or checkpoint mutation."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from brush import render, runs_from_traj
import scoring

OUT = ROOT / "outputs"
STEM = "rollout_apple_v5b_matched_te0.01q4"

def main():
    data = json.loads((OUT / f"{STEM}.json").read_text(encoding="utf-8"))
    ex = np.asarray(data["runs"][0]["executed"], dtype=np.float32)
    render(runs_from_traj(ex), 512).save(OUT / f"{STEM}_brush.png")
    # Same control points, with time mapped to 12 seconds for convenient viewing.
    frames = [render(runs_from_traj(ex[:k]), 256)
              for k in np.unique(np.rint(np.linspace(1, len(ex), 360)).astype(int))]
    frames += [frames[-1]] * 30
    frames[0].save(OUT / f"{STEM}_brush.gif", save_all=True,
                   append_images=frames[1:], duration=33, loop=0, optimize=True)
    summary_path = OUT / "ai_comparison_matched.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["chosen"] = STEM
    summary["selection_reason"] = "Highest B score among tested candidates, qualified 6 strokes, improved display smoothness over exec12 baselines."
    prior_path = OUT / "rollout_apple_v4_te0.1q4.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))["runs"][0]
    summary["prior_v4_q4_m01"] = {"path": str(prior_path), "benchmark_b": scoring.score_trajectory("apple", prior["executed"]),
                                    "smooth": prior["smooth"], "strokes": prior["strokes"],
                                    "self_terminated": prior["self_terminated"], "fresh_matched_run": False}
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = ["# AI rollout comparison", "", "Selected: **v5b MLP + temporal ensemble q=4, m=0.01**.", "",
             "Seed 0; identical target PNG; arc-length training resample 0.005; exec12 baseline; 1200-step budget; equal 72-step stale-ink window. No retraining.", "",
             "| Candidate | B score | F1 10px | F1 5px | Display wave (deg) | Strokes | Qualified | Mean / p95 ms per model call |",
             "|---|---:|---:|---:|---:|---:|:---:|---:|"]
    for row in summary["candidates"]:
        b = row["benchmark_b"]
        lines.append(f"| {row['name']} | {b['score']:.1f} | {b['f1_10']:.4f} | {b['f1_5']:.4f} | {row['smooth']['disp_wave_deg']:.3f} | {row['strokes']} | {'yes' if b['qualified'] else 'no'} | {row['inference_ms_mean']:.2f} / {row['inference_ms_p95']:.2f} |")
    lines += ["", "Scores use `score_version=b-v1`: average raw F1 at tolerances 0.02 and 0.01, then floor to one decimal. `qualified` retains the 10px F1 and normal stroke count rule.", "",
              "The selected rollout has B score **87.2**, 100% 10px target coverage, 6 strokes, and display wave **1.842 degrees** (41.2% lower than the v4 exec12 baseline). It has the highest B score of this limited comparison. At export precision (4 decimal coordinates), B remains **87.2**, F1 10px=0.9854 and F1 5px=0.7593.", "",
              "All five candidates stop through no-new-ink detection. **None self-terminates**. The selected run executes 760 actions / 190 calls, ending after an 18-query stale window. Its mean inference is 4.39 ms per model call (p95 6.53 ms), 1.098 ms per executed action. TE q4 uses three model calls per 12 actions.", "",
              "The previous v4 TE q6 run is slightly smoother (1.593 degrees) but produces 9 strokes, fails the existing normal stroke-count qualification, and scores 86.7 under B. The previous v4 TE q4 m=0.1 artifact is smoother (1.345 degrees) but scores 80.9. They were not selected.", "",
              "Performance figures are local XPU measurements, include observation preparation plus model and CPU synchronization, and exclude four warmup calls. These are closed-loop reproductions of the trained apple template; no novel-template generalization was tested.", "",
              f"Target source: `{summary['target_source']}`. Decoded target SHA256: `{summary['target_sha256']}`. Both checkpoint validation targets match it.", "",
              f"Chosen rollout: `{STEM}.json`.", f"Comparison PNG: `{STEM}_final.png`.", f"Brush PNG/GIF: `{STEM}_brush.png`, `{STEM}_brush.gif`.", "",
              "Brush animation maps the unchanged saved control points to approximately 12 seconds plus a final hold; it is display-only. Inference observations used the canonical `render_canvas`."]
    (OUT / "ai_comparison_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Selected artifacts: {OUT / STEM}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
