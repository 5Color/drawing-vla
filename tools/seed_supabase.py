"""로컬 data/raw/의 검증된 시연을 Supabase drawing_demos에 초기 등록(시드) — 빈 리더보드로 시작하지 않게.

  python tools/seed_supabase.py --dry-run      무엇을 올릴지 점수·티어만 출력 (DB에 쓰지 않음)
  python tools/seed_supabase.py                실제 등록 (이미 올린 파일은 건너뜀 → 여러 번 실행해도 중복 없음)
  python tools/seed_supabase.py --nickname siat --sqlite outputs/local_bench.sqlite   (로컬 시뮬레이터에 시드)

- 점수는 서버와 같은 scoring.py로 다시 매기고, 채택 기준(F1 ≥ 0.70, 정상 획 수)을 통과한 것만 올린다.
- 닉네임: meta.nickname/contributor가 있으면 그것, 없으면 --nickname (기본 siat).
- trajectory_data.meta.seeded_from = 원본 파일명. pull_dataset.py는 이 파일이 로컬에 있으면 다시 받지 않는다.
- created_at은 원래 그린 시각(meta.created_at)을 쓴다.
- data/raw에는 pull_dataset.py가 DB에서 받아 온 시연도 섞여 있다(meta.db_id, 배포 서버가 만든 기록은
  meta.trajectory_source == "server"). 이미 DB에 있는 기록이므로 올리지 않는다 (다시 올리면 'siat' 중복 기록이 생김).
  meta.server_received_at은 로컬 수집 서버(server.py)도 쓰므로 기준으로 삼지 않는다.
- 접속: 환경변수 SUPABASE_URL/SUPABASE_KEY 또는 .collect-secrets.json의 supabase_url/supabase_key (service_role).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import collect_secrets
import db as dbmod
import scoring
from collect_common import sanitize_target, validate_payload

ROOT = collect_secrets.ROOT
BATCH = 10   # 한 번에 보낼 행 수 (행마다 수십 KB)


def input_type(meta: dict) -> str:
    kinds = (meta.get("device") or {}).get("pointer_types") or []
    t = meta.get("input_type") or (kinds[0] if len(kinds) == 1 else None)
    return t if t in ("touch", "pen", "mouse") else "unknown"


def from_db(meta: dict) -> bool:
    """DB에서 온 시연인가: pull_dataset.py가 붙인 db_id, 또는 배포 서버(api_core)가 만든 기록."""
    return bool(meta.get("db_id")) or meta.get("trajectory_source") == "server"


def main() -> int:
    ap = argparse.ArgumentParser(description="로컬 검증 시연 → Supabase 초기 등록")
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--nickname", default="siat")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sqlite", help="Supabase 대신 로컬 SQLite (시뮬레이터 테스트)")
    ap.add_argument("--rescore-errors", action="store_true",
                    help="등록된 모든 기록의 avg_error를 지금 채점 방식으로 다시 계산 (채점 코드가 바뀌었을 때)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    nick_default = args.nickname[:20]
    files = sorted(Path(args.root).glob("demo_*.json"))
    templates = scoring.load_templates()
    rows, skipped, pulled = [], Counter(), 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        meta = d.get("meta", {})
        if from_db(meta):   # pull_dataset.py로 받은 시연 → 이미 DB에 있음
            pulled += 1
            continue
        target = sanitize_target(meta.get("target") or f.name.split("_")[1])
        if validate_payload(d) or target not in templates:
            skipped["형식 오류/모르는 도안"] += 1
            continue
        sc = scoring.score_trajectory(target, d["trajectory"])
        if not sc["qualified"]:
            skipped["채택 기준 미달"] += 1
            continue
        nick = (meta.get("nickname") or meta.get("contributor") or nick_default)[:20]
        d.pop("action_chunks", None)
        meta.update(target=target, nickname=nick, score=sc, seeded_from=f.name)
        rows.append({
            "nickname": nick, "contributor": meta.get("contributor") or nick_default, "template_name": target,
            "score": sc["score"], "tier": sc["tier"], "f1": sc["f1"], "coverage": sc["coverage"],
            "precision": sc["precision"], "strokes": sc["strokes"], "is_qualified": True, "avg_error": sc["avg_error"],
            "trajectory_data": d, "ip_hash": None, "input_type": input_type(meta),
            **({"created_at": meta["created_at"]} if meta.get("created_at") else {}),
        })

    by_t = defaultdict(list)
    for r in rows:
        by_t[r["template_name"]].append(r["score"])
    print(f"[seed] {args.root}: 파일 {len(files)}개 → 올릴 후보 {len(rows)}개" + (f" (제외 {dict(skipped)})" if skipped else ""))
    print(f"[seed] DB에서 받아 온 시연 {pulled}개 건너뜀 (pull_dataset.py로 받은 것 = 이미 DB에 있음)")
    for t, v in sorted(by_t.items()):
        tiers = Counter(scoring.tier_for(s)[0] for s in v)
        print(f"         {t:<8} {len(v):>3}개 · 최고 {max(v):.1f} · 중앙 {sorted(v)[len(v) // 2]:.1f} · 티어 {dict(sorted(tiers.items()))}")
    if args.dry_run:
        print("[seed] --dry-run: DB에 쓰지 않았습니다")
        return 0

    if args.sqlite:
        db = dbmod.SqliteDB(args.sqlite)
    else:
        sec = collect_secrets.load()
        url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
        key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")
        if not (url and key):
            sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 없습니다 (python tools/make_link.py --supabase-url … --supabase-key …)")
        db = dbmod.SupabaseDB(url, key)

    try:
        db.missing_avg_error(1)
    except dbmod.DBError as e:
        if "avg_error" in str(e):
            sys.exit("[error] DB에 avg_error 열이 없습니다. Supabase SQL Editor에서 database/schema.sql을 다시 실행하세요"
                     " (여러 번 실행해도 안전)")
        raise
    done = db.seeded_files()
    todo = [r for r in rows if r["trajectory_data"]["meta"]["seeded_from"] not in done]
    print(f"[seed] 이미 등록됨 {len(rows) - len(todo)}개 · 새로 등록 {len(todo)}개", flush=True)
    for i in range(0, len(todo), BATCH):
        db.insert_many(todo[i:i + BATCH])
        print(f"         {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)

    fixed = 0   # avg_error 열이 생기기 전에 올라간 기록 채우기 (시드·실사용자 기록 모두)
    while True:
        missing = db.missing_avg_error(50)
        if not missing:
            break
        for r in missing:
            traj = (r["trajectory_data"] or {}).get("trajectory") or []
            db.update_demo(r["id"], {"avg_error": scoring.score_trajectory(r["template_name"], traj)["avg_error"]})
            fixed += 1
    if fixed:
        print(f"[seed] 평균 오차 채움 {fixed}개")
    if args.rescore_errors:
        changed, offset = 0, 0
        while True:
            batch = db.rows_for_rescore(offset, 100)
            if not batch:
                break
            offset += len(batch)
            for r in batch:
                if not r.get("trajectory"):
                    continue
                e = scoring.score_trajectory(r["template_name"], r["trajectory"])["avg_error"]
                if r.get("avg_error") is None or abs(e - r["avg_error"]) > 1e-6:
                    db.update_demo(r["id"], {"avg_error": e})
                    changed += 1
        print(f"[seed] 평균 오차 다시 계산: {offset}개 중 {changed}개 갱신")
    print("[seed] 리더보드 확인 (도안별 TOP 5):")
    for t in sorted(templates):
        top = db.leaderboard(t, 5)
        line = ", ".join(f"{r['rank']}위 {r['nickname']} {r['score']:.1f} (오차 {r.get('avg_error')}px)"
                         for r in top) or "(기록 없음)"
        print(f"         {t:<8} {line}")
    print(f"[seed] 도안별 등록 수: {db.counts()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
