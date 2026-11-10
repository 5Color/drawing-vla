"""AI(학습된 정책)의 폐루프 롤아웃 궤적을 collector/index.html의 AI_DEMOS에 넣는다 — 결과 창의 "AI vs 나" 비교용.

점수는 사람과 똑같이 scoring.py로 다시 매긴다 (공정한 비교).

  python tools/export_ai_demo.py                                        기본: 사과 = outputs/rollout_apple_v5b_matched_te0.01q4.json
  python tools/export_ai_demo.py --rollout outputs/rollout_sprout_v5.json --template sprout --model "v5 MLP"
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import collect_secrets  # noqa: F401  (루트 경로를 sys.path에 추가)
import scoring

ROOT = collect_secrets.ROOT
HTML = ROOT / "collector" / "index.html"
BEGIN, END = "// AI_DEMOS_BEGIN", "// AI_DEMOS_END"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default=str(ROOT / "outputs" / "rollout_apple_v5b_matched_te0.01q4.json"))
    ap.add_argument("--template", default="apple")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="v5b MLP · 앙상블 q4")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    run = next(r for r in json.loads(open(args.rollout, encoding="utf-8").read())["runs"] if r["seed"] == args.seed)
    traj = [[round(x, 4), round(y, 4), int(p > 0.5)] for x, y, p in run["executed"]]
    while len(traj) > 1 and traj[-1][2] == 0:   # 끝에 펜을 든 채 맴도는 구간 제거 → 다시 보기 재생이 깔끔하게 끝나도록
        traj.pop()
    sc = scoring.score_trajectory(args.template, traj)

    src = HTML.read_text(encoding="utf-8")
    m = re.search(r"const AI_DEMOS = (\{.*?\});", src)
    if not m:
        raise SystemExit(f"{HTML.name}에서 const AI_DEMOS 선언을 찾을 수 없습니다")
    demos = json.loads(m.group(1))
    demos[args.template] = {"score": sc["score"], "model": args.model, "traj": traj}   # 사람과 같은 버림 규칙
    new_decl = f"const AI_DEMOS = {json.dumps(demos, separators=(',', ':'))};"
    src = src[:m.start()] + new_decl + src[m.end():]
    HTML.write_text(src, encoding="utf-8")
    print(f"[ai-demo] {args.template}: {args.model} {sc['score']:.1f}점 ({len(traj)}스텝) → {HTML.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
