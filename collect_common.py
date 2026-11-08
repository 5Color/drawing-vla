"""로컬 서버(server.py)와 Vercel 함수(api/*.py)가 함께 쓰는 검증·토큰·파일명 로직.

표준 라이브러리만 사용한다.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import math
import re
import secrets

DEMO_NAME_RE = re.compile(r"^demo_([a-z0-9-]+)_.*\.json$")
NAME_RE = re.compile(r"^[a-z0-9-]{1,24}$")
KST = dt.timezone(dt.timedelta(hours=9))


# ---------------------------------------------------------------------------
# 시연 payload 검증
# ---------------------------------------------------------------------------
def sanitize_target(name: object) -> str:
    s = re.sub(r"[^a-z0-9-]", "", str(name or "").lower())[:32]
    return s or "unknown"


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_payload(p: object) -> str | None:
    """문제가 있으면 오류 메시지, 정상이면 None."""
    if not isinstance(p, dict):
        return "payload는 JSON 객체여야 합니다"
    if not isinstance(p.get("meta"), dict):
        return "meta 객체가 없습니다"
    traj = p.get("trajectory")
    if not isinstance(traj, list) or not traj:
        return "trajectory가 비어 있습니다"
    for i, pt in enumerate(traj):
        if not (isinstance(pt, list) and len(pt) == 3 and all(_is_num(v) for v in pt)):
            return f"trajectory[{i}]는 [x, y, pen_down] 숫자 3개여야 합니다"
        x, y, pen = pt
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return f"trajectory[{i}] 좌표가 [0, 1] 범위를 벗어났습니다"
        if pen not in (0, 1):
            return f"trajectory[{i}] pen_down은 0 또는 1이어야 합니다"
    tgt = p.get("target_128_png")
    if not (isinstance(tgt, str) and tgt.startswith("data:image/png;base64,")):
        return "target_128_png(PNG data URL)가 없습니다"
    return None


def demo_filename(target: str, contributor: str | None = None, now: dt.datetime | None = None) -> str:
    """demo_<target>_<YYYYMMDD-HHMMSS>-<ms>[-<rand>_<contributor>].json (KST)."""
    now = now or dt.datetime.now(KST)
    stem = f"demo_{sanitize_target(target)}_{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}"
    if contributor:
        stem += f"-{secrets.token_hex(2)}_{contributor}"
    return stem + ".json"


def count_by_target(names) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        m = DEMO_NAME_RE.match(name)
        key = m.group(1) if m else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# 참여자 개인 링크 토큰: k = HMAC(secret, name) — 비밀키 하나로 사람마다 링크 발급
# ---------------------------------------------------------------------------
def normalize_name(name: object) -> str | None:
    s = str(name or "").strip().lower()
    return s if NAME_RE.match(s) else None


def sign_name(secret: str, name: str) -> str:
    mac = hmac.new(secret.encode(), f"v1:{name}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")[:22]


def verify_name(secret: str, name: object, token: object) -> str | None:
    """서명이 맞으면 정규화된 이름, 아니면 None."""
    n = normalize_name(name)
    if not (secret and n and isinstance(token, str) and token.isascii() and len(token) <= 64):
        return None
    return n if hmac.compare_digest(sign_name(secret, n), token.strip()) else None
