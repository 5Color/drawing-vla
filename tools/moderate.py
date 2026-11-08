"""리더보드 운영: 상위 기록·검토 대기 기록 보기, 승인·내리기.

  python tools/moderate.py                     도안별 TOP 10 + 검토 대기 목록 (측정값 포함)
  python tools/moderate.py --approve <id>      검토 대기 기록 승인 → 리더보드·학습 데이터에 포함
  python tools/moderate.py --hide <id>         기록 내리기 (is_qualified=false). 데이터는 지우지 않음
  python tools/moderate.py --sqlite outputs/local_bench.sqlite   (로컬 시뮬레이터)

'검토 대기' = 사람 기록보다 훨씬 정밀(평균 오차 < 1.5px)해서 자동으로 보류된 기록. 사람 시연 실측 최소는 2.0px.
측정값(checks): draw_ms 그린 시간, avg_speed 평균 속도(한 변/초, 사람 0.4~0.65), peak_speed, rate(초당 점 수).
접속 정보: SUPABASE_URL/SUPABASE_KEY 환경변수 또는 .collect-secrets.json (service_role/secret 키).
"""
from __future__ import annotations

import argparse
import os
import sys

import collect_secrets
import db as dbmod
import scoring


def fmt(r: dict) -> str:
    c = r.get("checks") or {}
    who = r["nickname"] + (f" (링크:{r['contributor']})" if r.get("contributor") else "")
    return (f"{r['id']}  {who:<22} {r['score']:>5.1f}점  오차 {r.get('avg_error') or 0:>5.2f}px  "
            f"{r.get('input_type') or '?':<6} 시간 {c.get('draw_ms', 0) / 1000:>4.1f}s  "
            f"속도 {c.get('avg_speed', 0):.2f}/{c.get('peak_speed', 0):.2f}  {str(r['created_at'])[:16]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="리더보드 운영 (보기·승인·내리기)")
    ap.add_argument("--approve", metavar="ID")
    ap.add_argument("--hide", metavar="ID")
    ap.add_argument("--sqlite")
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
            sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 없습니다 (python tools/make_link.py --supabase-url … --supabase-key …)")
        db = dbmod.SupabaseDB(url, key)

    if args.approve or args.hide:
        rid, ok = args.approve or args.hide, bool(args.approve)
        db.update_demo(rid, {"is_qualified": ok})
        print(f"[moderate] {rid} → {'승인 (리더보드·학습에 포함)' if ok else '내림 (리더보드·학습 제외)'}")
        return 0

    pending = db.review_rows()
    print(f"[검토 대기] {len(pending)}개" + ("" if pending else " — 없음"))
    for r in pending:
        print("  " + fmt(r) + f"  [{r['template_name']}]")
    for t in sorted(scoring.load_templates()):
        rows = db.top_rows(t, 10)
        print(f"\n[{t}] TOP {len(rows)}")
        for r in rows:
            print("  " + fmt(r))
    print("\n승인: python tools/moderate.py --approve <id>   내리기: python tools/moderate.py --hide <id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
