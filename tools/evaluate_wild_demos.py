"""비전 AI 임베딩 기반 코사인 유사도 평가 및 DB 자동 승격기.

  # 1. 10개 전체 도안의 미인정 데이터 중 코사인 유사도 80% 이상을 DB에서 골드로 승격!
  python tools/evaluate_wild_demos.py --all-targets --threshold 0.80 --promote-db

  # 2. 특정 도안(사과)만 미리보기
  python tools/evaluate_wild_demos.py --target apple --threshold 0.80 --no-save
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tools"))

import collect_secrets
import db as dbmod
import scoring
from policies import VisuomotorPolicy

OBS_SIZE = 128
OUT_DIR = ROOT / "data" / "wild_gold"


def render_trajectory_image(traj: list, size: int = OBS_SIZE) -> Image.Image:
    """궤적을 (size, size) 흑백 이미지로 렌더링 (펜 다운 선: 255, 배경: 0)."""
    scale = size / 512.0
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cur = None
    for item in traj:
        x, y, p = item[0], item[1], item[2]
        px, py = x * 512.0 * scale, y * 512.0 * scale
        if p >= 0.5:
            if cur is not None:
                draw.line([cur, (px, py)], fill=255, width=max(1, int(2.0 * scale * 4)))
            cur = (px, py)
        else:
            cur = None
    return img


def render_template_image(target_name: str, size: int = OBS_SIZE) -> Image.Image:
    """도안 템플릿을 (size, size) 흑백 이미지로 렌더링."""
    parts = scoring.template_parts(target_name)
    scale = size / 512.0
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for part in parts:
        pts = [(x * 512.0 * scale, y * 512.0 * scale) for x, y in part]
        if len(pts) > 1:
            draw.line(pts, fill=255, width=max(1, int(2.0 * scale * 4)))
    return img


def load_encoder(device: torch.device):
    """사전학습된 VLA 인코더 로드."""
    ckpt_path = ROOT / "outputs" / "apple_vla_policy_act_v1.pth"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "outputs" / "apple_vla_policy_v2_resample.pth"

    policy = VisuomotorPolicy(channels=(32, 64, 128, 256), feat_dim=256, obs_size=OBS_SIZE, state_dim=3)
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(state.get("model", state), strict=False)
            print(f"[encoder] 사전학습 가중치 로드 완료: {ckpt_path.name}")
        except Exception as e:
            print(f"[encoder] 가중치 로드 주의 ({e}) → 구조 인코더 사용")
    policy.to(device).eval()
    return policy.encoder


def compute_cosine_similarity(img1: Image.Image, img2: Image.Image, encoder, device: torch.device) -> float:
    """두 흑백 이미지의 256차원 Feature 벡터 간의 코사인 유사도 계산."""
    t1 = torch.from_numpy(np.array(img1, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(np.array(img2, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    x1 = torch.cat([t1, t1], dim=1).to(device)
    x2 = torch.cat([t2, t2], dim=1).to(device)

    with torch.no_grad():
        feat1 = encoder(x1)
        feat2 = encoder(x2)
        cos_sim = F.cosine_similarity(feat1, feat2, dim=-1).item()
    return float(cos_sim)


def get_database(sqlite_path: str | None = None):
    """Supabase 또는 SQLite 데이터베이스 인스턴스 반환."""
    if sqlite_path:
        return dbmod.SqliteDB(sqlite_path)
    sec = collect_secrets.load()
    url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
    key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")
    if url and key:
        return dbmod.SupabaseDB(url, key)
    local_sqlite = ROOT / "outputs" / "local_bench.sqlite"
    if local_sqlite.exists():
        return dbmod.SqliteDB(str(local_sqlite))
    sys.exit("[error] Supabase 인증 정보 또는 로컬 SQLite DB가 없습니다.")


def fetch_submissions(db, target: str, limit: int = 500) -> list[dict]:
    """해당 도안의 최근 제출 데이터 조회."""
    cols = "id,nickname,template_name,score,avg_error,is_qualified,trajectory_data,created_at"
    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "drawing_demos", {
            "select": cols,
            "template_name": f"eq.{target}",
            "order": "created_at.desc",
            "limit": limit
        })
        return res or []
    rows = db._rows(
        f"select id, nickname, template_name, score, avg_error, is_qualified, trajectory_data, created_at "
        f"from drawing_demos where template_name=? order by created_at desc limit ?",
        (target, limit)
    )
    return rows or []


def evaluate_single_target(
    target: str,
    db,
    encoder,
    device: torch.device,
    threshold: float = 0.80,
    save_gold: bool = True,
    promote_db: bool = False
) -> tuple[int, int, int]:
    """단일 도안에 대해 평가 및 승격 수행. 반환: (총건수, 골드채택건수, DB승격건수)"""
    tpl_img = render_template_image(target, OBS_SIZE)
    target_rows = fetch_submissions(db, target, limit=500)
    if not target_rows:
        return 0, 0, 0

    gold_count = 0
    promoted_count = 0

    print(f"\n--- [{target}] 총 {len(target_rows)}건 평가 중 ---")
    for r in target_rows:
        demo_id = r["id"]
        nick = r.get("nickname", "익명")
        score = r.get("score", 0.0)
        is_qual = r.get("is_qualified", False)

        traj_data = r.get("trajectory_data")
        if isinstance(traj_data, str):
            try:
                traj_data = json.loads(traj_data)
            except Exception:
                continue
        if not isinstance(traj_data, dict) or "trajectory" not in traj_data:
            continue
        traj = traj_data["trajectory"]

        drawn_img = render_trajectory_image(traj, OBS_SIZE)
        cos_sim = compute_cosine_similarity(tpl_img, drawn_img, encoder, device)

        is_gold = cos_sim >= threshold
        if is_gold and not is_qual:
            gold_count += 1
            new_score = max(score, round(cos_sim * 85.0, 1))
            new_tier, _, _ = scoring.tier_for(new_score)

            action_desc = "선별"
            if promote_db:
                try:
                    db.update_demo(demo_id, {
                        "is_qualified": True,
                        "score": new_score,
                        "tier": new_tier
                    })
                    promoted_count += 1
                    action_desc = "DB 승격"
                except Exception as e:
                    action_desc = f"승격 실패({e})"

            print(f"  ✨ {nick:<12} (기존 {score:4.1f}점) → 유사도 {cos_sim*100:4.1f}% | 보정 {new_score:4.1f}점 [{new_tier}] | {action_desc}")

            if save_gold:
                out_file = OUT_DIR / f"gold_{target}_{demo_id[:8]}.json"
                out_file.write_text(json.dumps({
                    "id": demo_id, "nickname": nick, "target": target,
                    "score": new_score, "tier": new_tier, "cosine_similarity": cos_sim,
                    "trajectory": traj, "saved_at": dt.datetime.now().isoformat()
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(target_rows), gold_count, promoted_count


def evaluate_submissions(
    target: str = "apple",
    all_targets: bool = False,
    threshold: float = 0.80,
    save_gold: bool = True,
    promote_db: bool = False,
    sqlite_path: str | None = None
):
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"[device] {device}")

    db = get_database(sqlite_path)
    encoder = load_encoder(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(scoring.load_templates().keys()) if all_targets else [target]
    print(f"[eval] 대상 도안: {', '.join(targets)}")
    if promote_db:
        print("[mode] 🚀 DB 자동 승격 모드 ON: 미인정된 골드 시연의 is_qualified를 True로 업데이트합니다.")

    total_scanned = 0
    total_gold = 0
    total_promoted = 0

    for t in targets:
        n_scanned, n_gold, n_promoted = evaluate_single_target(
            t, db, encoder, device, threshold, save_gold, promote_db
        )
        total_scanned += n_scanned
        total_gold += n_gold
        total_promoted += n_promoted

    print("\n" + "=" * 80)
    print(f"[최종 집계] 총 {total_scanned}건 스캔 완료")
    print(f"  - 코사인 유사도 {int(threshold*100)}% 이상 미인정작 발굴: {total_gold}건")
    if promote_db:
        print(f"  - Supabase DB 공식 인정(is_qualified=True) 승격 완료: {total_promoted}건")
    if save_gold:
        print(f"  - 로컬 학습용 JSON 저장 위치: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="비전 임베딩 코사인 유사도 평가 및 DB 승격기")
    parser.add_argument("--target", default="apple", help="대상 도안 (기본: apple)")
    parser.add_argument("--all-targets", action="store_true", help="10개 모든 도안 전체 스캔")
    parser.add_argument("--threshold", type=float, default=0.80, help="코사인 유사도 기준 (기본: 0.80)")
    parser.add_argument("--promote-db", action="store_true", help="DB의 is_qualified를 True로 자동 승격")
    parser.add_argument("--sqlite", help="로컬 SQLite DB 경로")
    parser.add_argument("--no-save", action="store_true", help="골드 파일 저장 끄기")
    args = parser.parse_args()

    evaluate_submissions(
        target=args.target,
        all_targets=args.all_targets,
        threshold=args.threshold,
        save_gold=not args.no_save,
        promote_db=args.promote_db,
        sqlite_path=args.sqlite
    )
