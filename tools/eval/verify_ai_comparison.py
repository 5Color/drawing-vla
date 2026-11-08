import json
from pathlib import Path
import sys
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scoring

out = ROOT / "outputs"
summary = json.loads((out / "ai_comparison_matched.json").read_text(encoding="utf-8"))
assert len(summary["candidates"]) == 5
hashes = set()
for row in summary["candidates"]:
    path = Path(row["rollout_path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    hashes.add(data["target_sha256"])
    assert data["own_reference_same_target"]
    data["runs"][0]["benchmark_b"] = row["benchmark_b"]
    data["runs"][0]["export_precision_b"] = row["export_precision_b"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
assert len(hashes) == 1
name = "rollout_apple_v5b_matched_te0.01q4"
data = json.loads((out / (name + ".json")).read_text(encoding="utf-8"))
run = data["runs"][0]
traj = [[round(x, 4), round(y, 4), int(z > .5)] for x, y, z in run["executed"]]
score = scoring.score_trajectory("apple", traj)
assert score["score"] == 87.2 and score["qualified"] and score["strokes"] == 6
assert not run["self_terminated"]
assert score["score_version"] == "b-v1"
print("chosen export score", json.dumps(score, ensure_ascii=True))
for suffix in ["_final.png", "_brush.png", ".gif", "_brush.gif"]:
    path = out / (name + suffix)
    image = Image.open(path)
    print(path.name, image.size, getattr(image, "n_frames", 1), path.stat().st_size)
image = Image.open(out / "ai_comparison_brush.png")
print("comparison panel", image.size)
print("matched candidates", len(summary["candidates"]), "unique targets", len(hashes))
