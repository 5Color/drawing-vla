"""DB 전체 백업 (읽기 전용) — drawing_demos 모든 행(채택 안 된 기록·의심 기록·검토 대기 포함, 모든 열)
+ funnel_counts(퍼널 통계) + nick_owners(닉네임 주인) → backups/<표>_<YYYYMMDD-HHMMSS>.jsonl.gz

  python tools/backup.py                                   Supabase (환경변수 또는 .collect-secrets.json)
  python tools/backup.py --out D:/drawing-backup           저장 폴더 바꾸기 (기본: 저장소/backups)
  python tools/backup.py --sqlite outputs/local_bench.sqlite   로컬 시뮬레이터 DB

- 행 하나 = JSON 한 줄(gzip). 시각은 한국 시간 기준 파일명. 받는 도중 실패하면 .part 파일만 남고 완성 파일은 안 생긴다.
- DB에는 읽기만 한다(복원 기능 없음 → 실수로 덮어쓰지 않게). 아직 없는 표(schema.sql 다시 실행 전)는 건너뛴다.
- 백업 파일에는 닉네임·IP 해시·기기 해시가 들어 있다. 공유·배포하지 말 것 (.vercelignore 화이트리스트라 배포엔 안 올라감).
- 접속: SUPABASE_URL/SUPABASE_KEY 환경변수 또는 .collect-secrets.json (service_role/secret 키). 키·주소는 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
from pathlib import Path

import collect_secrets
import db as dbmod

ROOT = collect_secrets.ROOT
PAGE = 100            # drawing_demos 한 번에 받을 행 수 (행마다 수십 KB: 원시 입력·이미지 포함)
SMALL_PAGE = 1000     # funnel_counts, nick_owners
TABLES = ("drawing_demos", "funnel_counts", "nick_owners")
SQLITE_ORDER = {"drawing_demos": "created_at, id", "funnel_counts": "day, event", "nick_owners": "nick"}


# --- 행 읽기 (페이지 단위 제너레이터) -----------------------------------------
def supabase_rows(db: dbmod.SupabaseDB, table: str):
    if table == "drawing_demos":   # id 기준 키셋 페이지: 받는 도중 새 기록이 들어와도 빠지거나 겹치지 않음
        last = None
        while True:
            params = {"select": "*", "order": "id.asc", "limit": PAGE, **({"id": f"gt.{last}"} if last else {})}
            res, _ = db._req("GET", table, params)
            if not res:
                return
            yield from res
            if len(res) < PAGE:
                return
            last = res[-1]["id"]
    order = {"funnel_counts": "day.asc,event.asc", "nick_owners": "nick.asc"}[table]
    offset = 0
    while True:
        res, _ = db._req("GET", table, {"select": "*", "order": order, "offset": offset, "limit": SMALL_PAGE})
        if not res:
            return
        yield from res
        if len(res) < SMALL_PAGE:
            return
        offset += len(res)


def sqlite_rows(db: dbmod.SqliteDB, table: str):
    offset = 0
    while True:
        rows = db._rows(f"select * from {table} order by {SQLITE_ORDER[table]} limit ? offset ?", (PAGE, offset))
        if not rows:
            return
        for r in rows:
            if table == "drawing_demos":   # Supabase 백업과 같은 모양으로 (jsonb → 객체, boolean)
                r["trajectory_data"] = json.loads(r["trajectory_data"]) if r.get("trajectory_data") else None
                r["is_qualified"] = bool(r.get("is_qualified"))
            yield r
        offset += len(rows)


# --- 저장 -------------------------------------------------------------------
def dump(rows, path: Path) -> int:
    """rows → path(.jsonl.gz). 끝까지 받은 뒤에만 완성 파일 이름으로 바꾼다."""
    part = path.with_name(path.name + ".part")
    n = 0
    try:
        with gzip.open(part, "wt", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
                n += 1
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return n


def show(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def connect(args):
    if args.sqlite:
        if not Path(args.sqlite).exists():
            sys.exit(f"[error] SQLite 파일이 없습니다: {args.sqlite}")
        return dbmod.SqliteDB(args.sqlite)
    sec = collect_secrets.load()
    url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
    key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")
    if not (url and key):
        sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 없습니다 (python tools/make_link.py --supabase-url … --supabase-key …)")
    return dbmod.SupabaseDB(url, key, timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser(description="DB 전체 백업 (읽기 전용) → backups/<표>_<시각>.jsonl.gz")
    ap.add_argument("--out", default=str(ROOT / "backups"), help="저장 폴더 (기본: 저장소/backups)")
    ap.add_argument("--sqlite", help="Supabase 대신 로컬 SQLite (시뮬레이터 DB)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    db = connect(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dbmod.KST).strftime("%Y%m%d-%H%M%S")
    src = "SQLite " + Path(args.sqlite).name if args.sqlite else "Supabase"
    print(f"[backup] {src} → {show(out)}  (읽기 전용)", flush=True)
    total, missing = 0, []
    for table in TABLES:
        path = out / f"{table}_{stamp}.jsonl.gz"
        rows = sqlite_rows(db, table) if args.sqlite else supabase_rows(db, table)
        try:
            n = dump(rows, path)
        except dbmod.DBError as e:
            if table != "drawing_demos" and dbmod.is_missing(e, table):
                missing.append(table)
                print(f"[backup] {table:<14} 표가 아직 없어 건너뜀 (database/schema.sql을 다시 실행하면 생김)")
                continue
            print(f"[error] {table} 받기 실패: {str(e)[:200]}", file=sys.stderr)
            return 1
        total += n
        print(f"[backup] {table:<14} {n:>7,}행 → {show(path)} ({path.stat().st_size / 1e6:.2f}MB)", flush=True)
    print(f"[backup] 완료: {total:,}행" + (f" (없는 표: {', '.join(missing)})" if missing else "")
          + " · 닉네임·IP 해시가 들어 있으니 공유하지 마세요")
    return 0


if __name__ == "__main__":
    sys.exit(main())
