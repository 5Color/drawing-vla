# AI rollout comparison

Selected: **v5b MLP + temporal ensemble q=4, m=0.01**.

Seed 0; identical target PNG; arc-length training resample 0.005; exec12 baseline; 1200-step budget; equal 72-step stale-ink window. No retraining.

| Candidate | B score | F1 10px | F1 5px | Display wave (deg) | Strokes | Qualified | Mean / p95 ms per model call |
|---|---:|---:|---:|---:|---:|:---:|---:|
| v4_matched_x12 | 86.8 | 0.9812 | 0.7548 | 3.135 | 6 | yes | 3.44 / 4.38 |
| v4_matched_te0.01q6 | 86.7 | 0.9595 | 0.7754 | 1.593 | 9 | no | 4.16 / 6.98 |
| v5b_matched_x12 | 84.3 | 0.9790 | 0.7086 | 2.579 | 5 | yes | 2.62 / 3.63 |
| v5b_matched_te0.01q6 | 83.5 | 0.9821 | 0.6881 | 2.373 | 6 | yes | 5.25 / 8.90 |
| v5b_matched_te0.01q4 | 87.2 | 0.9854 | 0.7588 | 1.842 | 6 | yes | 4.39 / 6.53 |

Scores use `score_version=b-v1`: average raw F1 at tolerances 0.02 and 0.01, then floor to one decimal. `qualified` retains the 10px F1 and normal stroke count rule.

The selected rollout has B score **87.2**, 100% 10px target coverage, 6 strokes, and display wave **1.842 degrees** (41.2% lower than the v4 exec12 baseline). It has the highest B score of this limited comparison. At export precision (4 decimal coordinates), B remains **87.2**, F1 10px=0.9854 and F1 5px=0.7593.

All five candidates stop through no-new-ink detection. **None self-terminates**. The selected run executes 760 actions / 190 calls, ending after an 18-query stale window. Its mean inference is 4.39 ms per model call (p95 6.53 ms), 1.098 ms per executed action. TE q4 uses three model calls per 12 actions.

The previous v4 TE q6 run is slightly smoother (1.593 degrees) but produces 9 strokes, fails the existing normal stroke-count qualification, and scores 86.7 under B. The previous v4 TE q4 m=0.1 artifact is smoother (1.345 degrees) but scores 80.9. They were not selected.

Performance figures are local XPU measurements, include observation preparation plus model and CPU synchronization, and exclude four warmup calls. These are closed-loop reproductions of the trained apple template; no novel-template generalization was tested.

Target source: `demo_apple_20261107-151648-848.json`. Decoded target SHA256: `c9c94d1a0b33ca6e9eb11cac63ebdf5cbfcd2300ebfc223da163af9ab2d0d598`. Both checkpoint validation targets match it.

Chosen rollout: `rollout_apple_v5b_matched_te0.01q4.json`.
Comparison PNG: `rollout_apple_v5b_matched_te0.01q4_final.png`.
Brush PNG/GIF: `rollout_apple_v5b_matched_te0.01q4_brush.png`, `rollout_apple_v5b_matched_te0.01q4_brush.gif`.

Brush animation maps the unchanged saved control points to approximately 12 seconds plus a final hold; it is display-only. Inference observations used the canonical `render_canvas`.
