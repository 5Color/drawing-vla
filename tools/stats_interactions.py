"""사용자 상호작용 및 드로잉 통계 분석 도구 (버튼 vs Ctrl+Z, 고민 시간, 획 수 등).

실행:
  python tools/stats_interactions.py
  python tools/stats_interactions.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "tools")
import collect_secrets
import db as dbmod


def main():
    ap = argparse.ArgumentParser(description="드로잉 상호작용 통계 분석")
    ap.add_argument("--limit", type=int, default=100, help="조회할 최근 데이터 개수")
    ap.add_argument("--sqlite", help="로컬 SQLite 파일 경로")
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
            sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 필요합니다.")
        db = dbmod.SupabaseDB(url, key)

    res, _ = db._req(
        "GET",
        "drawing_demos",
        {
            "select": "id,nickname,template_name,score,avg_error,input_type,is_qualified,created_at,trajectory_data",
            "order": "created_at.desc",
            "limit": str(args.limit),
        },
    )

    if isinstance(res, str):
        rows = json.loads(res)
    elif isinstance(res, list):
        rows = res
    else:
        rows = []

    print(f"\n=======================================================")
    print(f"📊 드로잉 인터랙션 및 상호작용 통계 (최근 {len(rows)}건 분석)")
    print(f"=======================================================\n")

    if not rows:
        print("데이터가 없습니다.")
        return

    total_count = len(rows)
    total_btn_undo = 0
    total_key_undo = 0
    total_clear = 0
    zero_undo_count = 0
    has_telemetry_count = 0

    durations = []
    pauses = []
    scores = []
    errors = []
    input_counts = {"mouse": 0, "pen": 0, "touch": 0, "other": 0}

    for r in rows:
        traj_data = r.get("trajectory_data") or {}
        if isinstance(traj_data, str):
            try:
                traj_data = json.loads(traj_data)
            except Exception:
                traj_data = {}
        meta = traj_data.get("meta") or {}
        checks = meta.get("checks") or {}

        # Undo 통계
        btn_u = meta.get("undo_button_count")
        key_u = meta.get("undo_key_count")
        clr = meta.get("clear_count", 0)

        if btn_u is not None or key_u is not None:
            has_telemetry_count += 1
            btn_val = btn_u or 0
            key_val = key_u or 0
            total_btn_undo += btn_val
            total_key_undo += key_val
            total_clear += clr
            if btn_val == 0 and key_val == 0:
                zero_undo_count += 1

        # 시간 통계
        draw_ms = checks.get("draw_ms") or meta.get("duration_ms")
        if draw_ms:
            durations.append(draw_ms / 1000)
        span_ms = checks.get("span_ms")
        if span_ms and draw_ms:
            pauses.append(max(0.0, span_ms - draw_ms) / 1000)

        # 점수/오차
        if r.get("score") is not None:
            scores.append(float(r["score"]))
        if r.get("avg_error") is not None:
            errors.append(float(r["avg_error"]))

        inp = r.get("input_type") or "other"
        input_counts[inp] = input_counts.get(inp, 0) + 1

    print(f"1. 🎨 입력 기기 비율:")
    for k, v in input_counts.items():
        if v > 0:
            print(f"   - {k:<6}: {v:>3}건 ({v / total_count * 100:>5.1f}%)")

    print(f"\n2. ⏱ 시간 지표 (평균):")
    if durations:
        avg_dur = sum(durations) / len(durations)
        print(f"   - 순수 드로잉 시간: {avg_dur:.1f}초 (최대 {max(durations):.1f}초)")
    if pauses:
        avg_pause = sum(pauses) / len(pauses)
        print(f"   - 획 사이 고민 시간: {avg_pause:.1f}초 (인간 지문 망설임)")

    print(f"\n3. 🎯 점수 및 정밀도:")
    if scores:
        print(f"   - 평균 점수: {sum(scores) / len(scores):.1f}점 (최고 {max(scores):.1f}점)")
    if errors:
        print(f"   - 평균 오차: {sum(errors) / len(errors):.2f}px (최소 {min(errors):.2f}px)")

    print(f"\n4. 🔄 [한 획 지우기] 버튼 vs Ctrl+Z (새 텔레메트리 적용분 {has_telemetry_count}건):")
    if has_telemetry_count > 0:
        total_undo = total_btn_undo + total_key_undo
        print(f"   - '한 획 지우기' UI 버튼 클릭 : {total_btn_undo:>3}회 ({total_btn_undo / max(total_undo, 1) * 100:.1f}%)")
        print(f"   - 'Ctrl+Z' 단축키 입력       : {total_key_undo:>3}회 ({total_key_undo / max(total_undo, 1) * 100:.1f}%)")
        print(f"   - '모두 지우기(휴지통)' 횟수  : {total_clear:>3}회")
        print(f"   - 실수 없이 한 번에 그린 비율 : {zero_undo_count / has_telemetry_count * 100:.1f}%")
    else:
        print("   - (방금 배포되었으므로 앞으로 들어오는 새 제출부터 자동 집계됩니다!)")

    print(f"\n=======================================================\n")


if __name__ == "__main__":
    main()
