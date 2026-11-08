"""Supabase / SQLite 시연 데이터를 pandas DataFrame으로 불러오기.

사용법:
  # 1. 전체 데이터(280+건) 한 번에 CSV로 저장 (소요 시간, 포인트 수 등 확장 메타 포함)
  python tools/load_df.py --all --csv all_demos.csv

  # 2. 최근 50개 확인
  python tools/load_df.py --limit 50

  # 3. 특정 도안(사과) 전체 조회
  python tools/load_df.py --target apple --all --csv apple_demos.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import collect_secrets
import db as dbmod


def get_df(
    target: str | None = None,
    limit: int = 100,
    all_rows: bool = False,
    include_meta: bool = True,
    include_raw_traj: bool = False
) -> pd.DataFrame:
    """Supabase 또는 로컬 DB에서 데이터를 읽어와 pandas DataFrame으로 반환."""
    sec = collect_secrets.load()
    url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
    key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")

    if url and key:
        db = dbmod.SupabaseDB(url, key)
    else:
        sqlite_path = ROOT / "outputs" / "local_bench.sqlite"
        if sqlite_path.exists():
            db = dbmod.SqliteDB(str(sqlite_path))
        else:
            raise RuntimeError("DB 인증 정보나 로컬 SQLite 파일이 없습니다.")

    cols = [
        "id", "nickname", "contributor", "template_name", "score", "tier",
        "f1", "coverage", "precision", "strokes", "avg_error", "input_type",
        "is_qualified", "created_at"
    ]
    # 메타데이터나 궤적 확장을 위해 trajectory_data 필요
    if include_meta or include_raw_traj:
        cols.append("trajectory_data")

    fetch_limit = 5000 if all_rows else limit
    params = {
        "select": ",".join(cols),
        "order": "created_at.desc",
        "limit": fetch_limit
    }
    if target:
        params["template_name"] = f"eq.{target}"

    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "drawing_demos", params)
        rows = res or []
    else:
        where_clause = f"where template_name='{target}'" if target else ""
        rows = db._rows(f"select {','.join(cols)} from drawing_demos {where_clause} order by created_at desc limit {fetch_limit}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 날짜 시간 파싱
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])

    # trajectory_data 내부의 유용한 메타데이터를 컬럼으로 언팩
    if include_meta and "trajectory_data" in df.columns:
        def extract_meta(td):
            if not isinstance(td, dict):
                return pd.Series({"duration_sec": None, "num_steps": None, "screen": None})
            meta = td.get("meta", {})
            dur = meta.get("duration_ms")
            dur_sec = round(dur / 1000.0, 2) if dur is not None else None
            steps = meta.get("num_steps") or len(td.get("trajectory", []))
            device = meta.get("device", {})
            screen = device.get("screen_css_px")
            screen_str = f"{screen[0]}x{screen[1]}" if isinstance(screen, list) and len(screen) >= 2 else None
            return pd.Series({
                "duration_sec": dur_sec,
                "num_steps": steps,
                "screen_res": screen_str
            })

        meta_df = df["trajectory_data"].apply(extract_meta)
        insert_idx = df.columns.get_loc("avg_error") + 1
        df.insert(insert_idx, "duration_sec", meta_df["duration_sec"])
        df.insert(insert_idx + 1, "num_steps", meta_df["num_steps"])
        df.insert(insert_idx + 2, "screen_res", meta_df["screen_res"])

    if not include_raw_traj and "trajectory_data" in df.columns:
        df = df.drop(columns=["trajectory_data"])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="drawing-vla 데이터를 pandas DataFrame으로 조회")
    parser.add_argument("--target", help="도안 이름 필터 (apple, heart, cat 등)")
    parser.add_argument("--limit", type=int, default=100, help="조회할 행 수 (기본 100)")
    parser.add_argument("--all", action="store_true", help="전체 데이터(280+건) 모두 가져오기")
    parser.add_argument("--csv", help="CSV 파일 저장 경로 (예: all_demos.csv)")
    parser.add_argument("--full", action="store_true", help="원시 궤적 좌표(trajectory_data) 포함")
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.unicode.east_asian_width", True)

    df = get_df(
        target=args.target,
        limit=args.limit,
        all_rows=args.all,
        include_meta=True,
        include_raw_traj=args.full
    )

    print(f"\n[데이터 로드 완료: 총 {len(df)}개 행, {len(df.columns)}개 열]")
    summary_cols = ["nickname", "template_name", "score", "tier", "duration_sec", "coverage", "avg_error", "input_type", "is_qualified", "created_at"]
    print(df[summary_cols].head(15))

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\n[CSV 파일 저장 완료: 총 {len(df)}건] -> {args.csv}")
