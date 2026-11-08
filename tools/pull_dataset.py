"""Supabase에 쌓인 시연 중 학습 데이터로 채택된 것(is_qualified = true)만 data/raw/로 동기화하고,
곧바로 품질 검사(train/check_demos.py)와 학습 데이터셋 로드 검증(train/dataset.py)을 실행한다.

  python tools/pull_dataset.py                 새 시연만 받기 → 품질 검사 → 데이터셋 shape 검증
  python tools/pull_dataset.py --no-check      받기만
  python tools/pull_dataset.py --sqlite outputs/local_bench.sqlite --dest <폴더>   (로컬 시뮬레이터 테스트)
  python tools/pull_dataset.py --trusted-only  개인 링크 참여자(contributor) 시연만
  python tools/pull_dataset.py --min-f1 0.9     학습 기준을 더 엄격하게 (기본 F1·정밀도 0.85 = A 티어 수준)

리더보드 채택(F1 ≥ 0.70)보다 학습 기준을 높게 둔다: 도안 밖 낙서가 섞인 시연·조작된 시연이 학습에 들어오지 않게.

접속 정보: 환경변수 SUPABASE_URL, SUPABASE_KEY(service_role) 또는 .collect-secrets.json의 supabase_url, supabase_key.
파일명 demo_<도안>_<YYYYMMDD-HHMMSS>-<id 앞 8자>.json. data/raw 또는 data/rejected에 같은 id가 있으면 건너뛴다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import collect_secrets
import db as dbmod
from collect_common import KST, sanitize_target, validate_payload

ROOT = collect_secrets.ROOT
ID_RE = re.compile(r"-([0-9a-f]{8})\.json$")


def file_name(row: dict) -> str:
    created = dt.datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00")).astimezone(KST)
    return f"demo_{sanitize_target(row['template_name'])}_{created:%Y%m%d-%H%M%S}-{str(row['id'])[:8]}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="채택된 시연만 data/raw로 동기화")
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--rejected", default=str(ROOT / "data" / "rejected"))
    ap.add_argument("--sqlite", help="Supabase 대신 로컬 SQLite (시뮬레이터 테스트)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--min-f1", type=float, default=0.85, help="학습에 쓸 최소 F1 (기본 0.85)")
    ap.add_argument("--min-precision", type=float, default=0.85, help="학습에 쓸 최소 선 정밀도 (기본 0.85)")
    ap.add_argument("--min-coverage", type=float, default=0.85, help="학습에 쓸 최소 도안 채움률 (기본 0.85)")
    ap.add_argument("--min-score", type=float, default=70.0, help="학습에 쓸 최소 점수 (기본 70.0)")
    ap.add_argument("--trusted-only", action="store_true", help="개인 링크 참여자 시연만")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.sqlite:
        db = dbmod.SqliteDB(args.sqlite)
    else:
        sec = collect_secrets.load()
        url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
        key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")
        if not (url and key):
            sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 없습니다 (환경변수 또는 .collect-secrets.json의 supabase_url, supabase_key)")
        db = dbmod.SupabaseDB(url, key)

    dest, rejected = Path(args.dest), Path(args.rejected)
    dest.mkdir(parents=True, exist_ok=True)
    have = {m.group(1) for d in (dest, rejected) if d.is_dir() for p in d.glob("demo_*.json") if (m := ID_RE.search(p.name))}

    got, skipped, bad, offset, low = Counter(), 0, [], 0, 0
    while True:
        rows = db.list_qualified(offset=offset, limit=100)
        if not rows:
            break
        offset += len(rows)
        for row in rows:
            payload = row["trajectory_data"]
            sc_meta = (payload.get("meta") or {}).get("score") or {}
            f1_val = row.get("f1") if row.get("f1") is not None else sc_meta.get("f1_10", sc_meta.get("f1", 0))
            prec_val = row.get("precision") if row.get("precision") is not None else sc_meta.get("precision", 0)
            cov_val = row.get("coverage") if row.get("coverage") is not None else sc_meta.get("coverage", 0)
            score_val = row.get("score") if row.get("score") is not None else sc_meta.get("score", 0)
            if (args.trusted_only and not row.get("contributor")) or \
                    f1_val < args.min_f1 or \
                    prec_val < args.min_precision or \
                    cov_val < args.min_coverage or \
                    score_val < args.min_score:
                low += 1
                continue
            seed = (payload.get("meta") or {}).get("seeded_from")   # seed_supabase.py로 올린 로컬 원본
            if str(row["id"])[:8] in have or (seed and ((dest / seed).exists() or (rejected / seed).exists())):
                skipped += 1
                continue
            err = validate_payload(payload)
            if err:
                bad.append((row["id"], err))
                continue
            payload.setdefault("meta", {}).update(db_id=str(row["id"]), nickname=row.get("nickname"),
                                                  contributor=row.get("contributor"))
            path = dest / file_name(row)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
            got[(row["template_name"], row.get("contributor") or "공개")] += 1

    print(f"[pull] 채택 시연 {offset}개 · 학습 기준 미달 {low}개 · 이미 있음 {skipped}개 · 새로 받음 {sum(got.values())}개",
          flush=True)
    for (tgt, who), n in sorted(got.items()):
        print(f"         {tgt:<8} {who:<14} {n}")
    for rid, err in bad:
        print(f"[skip] {rid}: {err}")
    total = Counter(p.name.split("_")[1] for p in dest.glob("demo_*.json"))
    print(f"[pull] {dest} 현재: " + ", ".join(f"{k} {v}" for k, v in sorted(total.items())), flush=True)

    if got and not args.no_check:
        py = sys.executable
        print("\n[check] 품질 검사 (train/check_demos.py)\n", flush=True)
        subprocess.call([py, str(ROOT / "train" / "check_demos.py"), "--root", str(dest)])
        print("\n[check] 학습 데이터셋 로드 검증 (train/dataset.py)\n", flush=True)
        return subprocess.call([py, str(ROOT / "train" / "dataset.py"), "--root", str(dest)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
