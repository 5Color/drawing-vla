"""비전 AI 임베딩 기반 코사인 유사도 평가기 (노트북 로컬 전용).

  python tools/evaluate_wild_demos.py --target apple --threshold 0.80
  python tools/evaluate_wild_demos.py --watch --interval 10

원리:
  1. 도안 원본 템플릿과 사용자의 드로잉 궤적을 128×128 캔버스 이미지로 렌더링.
  2. 사전학습된 VLA 시각 인코더(ConvNet/XPU)를 통과시켜 256차원 의미 임베딩(Feature) 추출.
  3. 두 임베딩 벡터 간의 코사인 유사도(Cosine Similarity) 계산.
  4. 코사인 유사도가 높은 데이터(예: 0.80+)는 크기/위치가 가이드선과 달라도
     '자유 예술가 골드 데이터(Wild Gold)'로 판정하여 학습 데이터셋으로 자동 선별/저장!
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
    """사전학습된 VLA 인코더 로드 (가중치가 없으면 초기화된 인코더 사용)."""
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
    t1 = torch.from_numpy(np.array(img1, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    t2 = torch.from_numpy(np.array(img2, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    # 인코더는 2채널(캔버스+도안) 입력을 받으므로 2채널로 복제
    x1 = torch.cat([t1, t1], dim=1).to(device)
    x2 = torch.cat([t2, t2], dim=1).to(device)

    with torch.no_grad():
        feat1 = encoder(x1)  # [1, 256]
        feat2 = encoder(x2)  # [1, 256]
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


def fetch_submissions(db, target: str, limit: int = 100) -> list[dict]:
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


def evaluate_submissions(target: str = "apple", threshold: float = 0.80, save_gold: bool = True, sqlite_path: str | None = None):
    """DB 시연들을 스캔하여 임베딩 코사인 유사도로 숨겨진 명작 선별."""
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"[device] {device}")

    db = get_database(sqlite_path)
    encoder = load_encoder(device)
    tpl_img = render_template_image(target, OBS_SIZE)

    # 전체 데이터 조회
    print(f"[eval] '{target}' 도안 제출 데이터 분석 중...")
    target_rows = fetch_submissions(db, target, limit=100)
    print(f"[eval] 대상 데이터: {len(target_rows)}건")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_count = 0

    print("\n" + "=" * 80)
    print(f"{'닉네임':<12} {'기존 점수':<10} {'기존 상태':<10} {'코사인 유사도':<14} {'최종 판정'}")
    print("-" * 80)

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
        status_str = "인정" if is_qual else "미인정"
        verdict = "✨ 골드 채택" if is_gold and not is_qual else ("기존 합격" if is_qual else "탈락")

        print(f"{nick:<12} {score:<10.1f} {status_str:<10} {cos_sim*100:<13.1f}% {verdict}")

        if is_gold and save_gold:
            gold_count += 1
            out_file = OUT_DIR / f"gold_{target}_{demo_id[:8]}.json"
            out_file.write_text(json.dumps({
                "id": demo_id, "nickname": nick, "target": target,
                "score": score, "cosine_similarity": cos_sim,
                "trajectory": traj, "saved_at": dt.datetime.now().isoformat()
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"[done] 총 {len(target_rows)}건 중 임베딩 코사인 유사도 {int(threshold*100)}% 이상: {gold_count}건 선별 완료 → {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="비전 임베딩 코사인 유사도 평가기")
    parser.add_argument("--target", default="apple", help="대상 도안 (기본: apple)")
    parser.add_argument("--threshold", type=float, default=0.80, help="코사인 유사도 기준 (기본: 0.80)")
    parser.add_argument("--sqlite", help="로컬 SQLite DB 경로")
    parser.add_argument("--no-save", action="store_true", help="골드 파일 저장 끄기")
    args = parser.parse_args()

    evaluate_submissions(target=args.target, threshold=args.threshold, save_gold=not args.no_save, sqlite_path=args.sqlite)
