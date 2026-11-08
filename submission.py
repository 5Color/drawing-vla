"""배포 벤치마크 제출 검증 (표준 라이브러리 전용) — 부정행위·자동화 방지.

원칙: 클라이언트가 보낸 값은 아무것도 믿지 않는다.
  1. 형식·크기 검사       : 허용한 필드만 남기고, 점 개수·이미지 크기·meta 크기를 제한한다.
  2. 궤적은 서버가 만든다  : 원시 입력 strokes[].points → (수집기와 같은) ma3x2 스무딩 → trajectory.
                            클라이언트가 보낸 trajectory는 버린다. 채점·학습 모두 서버가 만든 궤적을 쓴다.
  3. 사람 손 검사         : 시간이 흐르는지, 속도가 사람 손으로 가능한지, 너무 완벽하지 않은지.
                            걸리면 저장은 하되 기록으로 인정하지 않는다 (리더보드·학습 제외, meta.flags에 사유).
                            HARD_FLAGS(속도·시간·점 빈도 = 자동 입력이 분명함)면 원시 입력·이미지는 버리고 meta만,
                            SOFT_FLAGS(너무 정확함·시간 역행 = 사람 오탐 가능)면 원시 기록을 그대로 남긴다.
  4. 제출 티켓            : 서버가 서명한 발급 시각. 그림 그리는 데 걸린 시간보다 먼저 발급돼 있어야 하고 한 번만 쓴다.
  5. 같은 그림 재제출 금지 : 원시 입력의 해시(traj_hash)를 DB 고유 키로.

기준은 보수적으로(사람을 잘못 거르지 않는 쪽으로) 잡았다. 사람 시연 68개(터치) 측정값 대비:
  평균 속도 0.38~0.65 (캔버스 한 변/초) → 한도 4.0 (6배) · 50ms 최고 속도 ≤ 1.8 → 한도 10.0 (5.5배)
  평균 오차 2.0~5.0px → 0.5px 미만만 '너무 완벽' · 초당 점 수 107~120 → 5~2000 허용 (멈춰 있으면 점이 안 생김)
  그린 시간 3.3초 이상 → 0.5초 미만만 · 시간 역행은 1ms까지 허용
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import time
from pathlib import Path

from collect_common import _is_num

# --- 크기 제한 (계산량·저장량 상한. 사람 시연 실측: 점 최대 735개, 한 획 길이 최대 2.25, 점 간격 최대 0.015) ---
MAX_STROKES = 40
MAX_POINTS = 8000             # 240Hz 펜으로 30초 넘게 그려도 충분
MAX_STEP = 0.25               # 연속한 두 점 사이 거리 (캔버스 한 변 기준). 사람 최대 0.015
MAX_PATH_LEN = 12.0           # 전체 선 길이 합. 사람 최대 약 2.7
MAX_STROKE_MS = 10 * 60 * 1000
MAX_PNG_BYTES = 16 * 1024     # 128×128 PNG는 실측 3~4.5KB
META_STR = {"target", "target_label", "session_id", "created_at", "input_type", "smoothing", "nickname", "notice",
            "guide_hint"}   # guide_hint: 다음 획 안내 버전('next-v1') 또는 'off' — 획 순서별로 학습 데이터를 나눌 때
META_NUM = {"episode_index", "num_strokes", "num_steps", "duration_ms", "guide_opacity"}
META_BOOL = {"guide_order_marks", "has_pressure"}
DEVICE_KEYS = {"user_agent", "pointer_types", "device_pixel_ratio", "board_css_px", "screen_css_px"}
SPEC = {   # 수집기 spec과 같은 값 (클라이언트가 보낸 spec은 쓰지 않음)
    "coord": "normalized [0,1], origin top-left, y down", "action_format": ["x", "y", "pen_down"],
    "pen_convention": "state, each stroke bracketed by pen-up points", "stroke_point_format": ["x", "y", "t_ms", "pressure"],
    "chunk_size": 16, "chunk_padding": "edge", "obs_size": 128, "obs_stroke_width_px": 2, "canvas_units": 512,
}
TARGETS_FILE = Path(__file__).resolve().parent / "targets_128.json"   # 도안별 표준 목표 이미지 (브라우저 렌더)

# --- 사람 손 기준 -----------------------------------------------------------
MIN_DRAW_MS = 500             # 펜이 화면에 닿아 있던 시간 합
MAX_AVG_SPEED = 4.0           # 캔버스 한 변 / 초
MAX_PEAK_SPEED = 10.0         # 50ms 창
PEAK_WINDOW_MS = 50
MIN_RATE, MAX_RATE = 5, 2000   # 초당 점 수 (멈춘 채 누르고 있으면 점이 안 생김 ~ 1000Hz 마우스)
MIN_AVG_ERROR_PX = 0.5        # 이보다 정확하면 사람 손으로 보기 어려움 (사람 최소 2.0px)
MAX_TIME_REPAIR_MS = 500      # 이만큼까지 시간이 거꾸로 간 점은 고쳐서 받음 (Safari 이벤트 시각 섞임). 넘으면 의심
REVIEW_AVG_ERROR_PX = 0.8     # 이보다 정확하면 '확인 후 리더보드에' (사람 극한 1.0px, 0.8px 미만은 기계 정밀도 의심)
HARD_FLAGS = frozenset({"too_fast_total", "speed_avg", "speed_peak", "sample_rate"})   # 봇이 분명 → meta만 저장
SOFT_FLAGS = frozenset({"too_perfect", "time_backwards", "time_order"})                # 사람 오탐 가능 → 원시 기록 보존

# --- 제출 티켓 ---------------------------------------------------------------
TICKET_TTL_S = 6 * 3600
TICKET_SLACK_S = 10           # 티켓 발급 직후 그리기 시작한 경우의 네트워크 지연 여유


class Rejected(ValueError):
    """형식 오류 → 400."""


# ---------------------------------------------------------------------------
# 티켓: "<발급 ms>.<nonce>.<서명>"
# ---------------------------------------------------------------------------
def _sig(secret: str, body: str) -> str:
    mac = hmac.new(secret.encode(), f"t1:{body}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")[:22]


def issue_ticket(secret: str, now: float | None = None) -> str:
    body = f"{int((now or time.time()) * 1000)}.{secrets.token_urlsafe(9)}"
    return f"{body}.{_sig(secret, body)}"


def ticket_age(secret: str, ticket: object, now: float | None = None) -> float | None:
    """서명이 맞고 만료 전이면 발급 후 경과 초, 아니면 None."""
    if not (secret and isinstance(ticket, str) and len(ticket) <= 80 and ticket.isascii() and ticket.count(".") == 2):
        return None
    issued, nonce, sig = ticket.split(".")
    if not (issued.isdigit() and hmac.compare_digest(_sig(secret, f"{issued}.{nonce}"), sig)):
        return None
    age = (now or time.time()) - int(issued) / 1000
    return age if -5 <= age <= TICKET_TTL_S else None


# ---------------------------------------------------------------------------
# 형식 검사 + 정리
# ---------------------------------------------------------------------------
def _png_ok(v: object, required: bool) -> bool:
    if v is None:
        return not required
    if not (isinstance(v, str) and v.startswith("data:image/png;base64,")):
        return False
    try:
        raw = base64.b64decode(v[22:], validate=True)
    except ValueError:
        return False
    # PNG 서명 + IHDR 가로·세로 128
    return (len(raw) <= MAX_PNG_BYTES and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR"
            and int.from_bytes(raw[16:20], "big") == 128 and int.from_bytes(raw[20:24], "big") == 128)


def _clean_meta(meta: dict) -> dict:
    """알려진 키만, 정해진 타입·길이로 (나머지는 버림 → 저장량·오염 방지)."""
    out = {}
    for k, v in meta.items():
        if k in META_STR and isinstance(v, str):
            out[k] = v[:80]
        elif k in META_NUM and _is_num(v):
            out[k] = v
        elif k in META_BOOL and isinstance(v, bool):
            out[k] = v
    dev = meta.get("device")
    if isinstance(dev, dict):
        d = {}
        if isinstance(dev.get("user_agent"), str):
            d["user_agent"] = dev["user_agent"][:300]
        if isinstance(dev.get("pointer_types"), list):
            d["pointer_types"] = [str(t)[:10] for t in dev["pointer_types"][:4]]
        for k in ("device_pixel_ratio", "board_css_px"):
            if _is_num(dev.get(k)):
                d[k] = dev[k]
        sc = dev.get("screen_css_px")
        if isinstance(sc, list) and len(sc) == 2 and all(_is_num(v) for v in sc):
            d["screen_css_px"] = sc
        out["device"] = d
    return out


_TARGETS: dict | None = None


def canonical_target(target: str) -> str | None:
    """도안별 표준 목표 이미지(PNG data URL). 학습 관측 ch1은 클라이언트가 아니라 이 값을 쓴다."""
    global _TARGETS
    if _TARGETS is None:
        try:
            _TARGETS = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _TARGETS = {}
    return _TARGETS.get(target)


def clean_payload(p: object) -> dict:
    """허용한 필드만, 값은 반올림해서 새로 만든 사본. 문제가 있으면 Rejected."""
    if not isinstance(p, dict):
        raise Rejected("payload는 JSON 객체여야 합니다")
    if not isinstance(p.get("meta"), dict):
        raise Rejected("meta 객체가 없습니다")
    out = {"schema_version": 1, "meta": _clean_meta(p["meta"]), "spec": dict(SPEC)}
    if not _png_ok(p.get("target_128_png"), True):
        raise Rejected("target_128_png는 128×128 PNG여야 합니다")
    if not _png_ok(p.get("canvas_128_png"), False):
        raise Rejected("canvas_128_png는 128×128 PNG여야 합니다")
    out["target_128_png"] = p["target_128_png"]
    if p.get("canvas_128_png"):
        out["canvas_128_png"] = p["canvas_128_png"]

    strokes = p.get("strokes")
    if not (isinstance(strokes, list) and 1 <= len(strokes) <= MAX_STROKES):
        raise Rejected(f"strokes는 1~{MAX_STROKES}개여야 합니다")
    total, length, clean = 0, 0.0, []
    prev_t, back = -1.0, 0.0   # 시각은 획을 넘어 단조 증가로 고친다 (거꾸로 간 만큼 기록)
    for i, s in enumerate(strokes):
        pts = s.get("points") if isinstance(s, dict) else None
        if not (isinstance(pts, list) and pts):
            raise Rejected(f"strokes[{i}].points가 비어 있습니다")
        total += len(pts)
        if total > MAX_POINTS:
            raise Rejected("점이 너무 많습니다")
        P = []
        for pt in pts:
            if not (isinstance(pt, list) and len(pt) == 4 and all(_is_num(v) for v in pt)):
                raise Rejected(f"strokes[{i}] 점은 [x, y, t_ms, pressure] 숫자 4개여야 합니다")
            x, y, t, pr = pt
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= t <= MAX_STROKE_MS and 0 <= pr <= 1):
                raise Rejected(f"strokes[{i}] 값이 허용 범위를 벗어났습니다")
            q = [_r5(x), _r5(y), round(t, 2), round(pr, 3)]
            if q[2] < prev_t:
                back = max(back, prev_t - q[2])
                q[2] = prev_t
            prev_t = q[2]
            if P:
                step = math.dist(P[-1][:2], q[:2])
                if step > MAX_STEP:
                    raise Rejected("선이 끊겨 튀었어요")
                length += step
                if length > MAX_PATH_LEN:
                    raise Rejected("선이 너무 길어요")
            P.append(q)
        kind = s.get("pointer_type")
        clean.append({"pointer_type": kind if kind in ("touch", "pen", "mouse") else "unknown", "points": P})
    out["strokes"] = clean
    out["meta"]["time_repair_ms"] = round(back, 2)
    return out


# ---------------------------------------------------------------------------
# 궤적 생성 (collector/index.html buildTrajectory · smoothPts와 같은 계산)
# ---------------------------------------------------------------------------
def _r5(v: float) -> float:
    return math.floor(v * 1e5 + 0.5) / 1e5     # JS Math.round(v * 1e5) / 1e5 (양수)


def _ma3(P):
    if len(P) < 3:
        return [[p[0], p[1]] for p in P]
    out = [[P[0][0], P[0][1]]]
    for i in range(1, len(P) - 1):
        out.append([_r5((P[i - 1][0] + P[i][0] + P[i + 1][0]) / 3), _r5((P[i - 1][1] + P[i][1] + P[i + 1][1]) / 3)])
    out.append([P[-1][0], P[-1][1]])
    return out


def build_trajectory(strokes) -> list[list[float]]:
    """획마다 3점 이동평균 2회(ma3x2, 양 끝 유지) → 펜-업 점으로 감싸기."""
    traj = []
    for s in strokes:
        Q = _ma3(_ma3(s["points"]))
        traj.append([Q[0][0], Q[0][1], 0])
        traj += [[q[0], q[1], 1] for q in Q]
        traj.append([Q[-1][0], Q[-1][1], 0])
    return traj


def has_pressure(strokes) -> bool:
    """진짜 필압 여부 (클라이언트 값은 믿지 않고 서버가 다시 판단). 수집기 hasPressure()와 같은 규칙."""
    ps = [p[3] for s in strokes if s["pointer_type"] == "pen" for p in s["points"]]
    return bool(ps) and len(set(ps)) > 2 and any(v not in (0, 0.5, 1) for v in ps)


def traj_hash(strokes) -> str:
    """모양만 본 해시 (0.5px 단위 좌표, 시간·압력 제외) → 시간만 살짝 바꾼 재제출도 같은 그림으로 본다."""
    raw = json.dumps([[(round(p[0] * 1000), round(p[1] * 1000)) for p in s["points"]] for s in strokes],
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# 사람 손 검사
# ---------------------------------------------------------------------------
def human_checks(strokes, avg_error_px: float) -> tuple[list[str], dict]:
    """(걸린 항목 목록, 측정값). 목록이 비어 있으면 통과."""
    flags = []
    draw_ms, length, peak, prev_end = 0.0, 0.0, 0.0, -1.0
    npts = sum(len(s["points"]) for s in strokes)
    for s in strokes:
        P = s["points"]
        if P[0][2] < prev_end - 1:
            flags.append("time_order")          # 획 시작이 앞 획 끝보다 이름 (clean_payload가 고친 뒤라 정상이면 없음)
        prev_end = P[-1][2]
        if any(b[2] < a[2] - 1 for a, b in zip(P, P[1:])):
            flags.append("time_backwards")
        draw_ms += P[-1][2] - P[0][2]
        length += sum(math.dist(a[:2], b[:2]) for a, b in zip(P, P[1:]))
        j = 0
        for i in range(len(P)):                 # 50ms 창 최고 속도
            while P[i][2] - P[j][2] > PEAK_WINDOW_MS:
                j += 1
            dt = P[i][2] - P[j][2]
            if dt >= PEAK_WINDOW_MS * 0.6:
                peak = max(peak, math.dist(P[i][:2], P[j][:2]) / (dt / 1000))
    t_all = [p[2] for s in strokes for p in s["points"]]
    span_ms = max(t_all) - min(t_all)
    avg_speed = length / (draw_ms / 1000) if draw_ms > 0 else math.inf
    rate = npts / (draw_ms / 1000) if draw_ms > 0 else math.inf
    if draw_ms < MIN_DRAW_MS:
        flags.append("too_fast_total")
    if avg_speed > MAX_AVG_SPEED:
        flags.append("speed_avg")
    if peak > MAX_PEAK_SPEED:
        flags.append("speed_peak")
    if not MIN_RATE <= rate <= MAX_RATE:
        flags.append("sample_rate")
    if avg_error_px < MIN_AVG_ERROR_PX:
        flags.append("too_perfect")
    m = {"draw_ms": round(draw_ms, 1), "span_ms": round(span_ms, 1), "avg_speed": round(avg_speed, 3),
         "peak_speed": round(peak, 3), "rate": round(rate, 1)}
    return sorted(set(flags)), m
