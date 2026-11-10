"""드로잉 벤치마크 API 로직 (Vercel 함수 api/*.py와 로컬 시뮬레이터 tools/vercel_local.py가 공유).

  GET  /api/stats         도안별 제출 수 (+ 개인 링크면 내 제출 수)
  GET  /api/ticket        제출 티켓 (서버 서명 발급 시각, 한 번만 사용) — submission.py
  POST /save              검증 → 서버가 궤적 생성·채점 → Supabase 저장 → 점수·티어·상위 %·공유 링크
                          (vercel.json: /save → /api/save). 부정행위·자동화 방지 규칙은 submission.py
  GET  /api/leaderboard   ?template=apple  도안별 TOP 5
  GET  /api/og            ?score=96&tier=S&template=apple  공유 썸네일 PNG (1200×630)
  GET  /s                 ?score=…  공유 링크 미리보기 페이지 (OG 태그) → 메인으로 이동   (vercel.json: /s → /api/share)
  POST /api/event         퍼널 통계 (sendBeacon, text/plain 이벤트 이름) → 항상 204
  POST /api/delete        헤더 X-Owner의 기기로 저장한 기록 전부 삭제 + 닉네임 반납
  GET  /api/cron          Supabase 무료 플랜 일시정지 방지 (vercel.json crons가 매일 부름, CRON_SECRET이 있으면 Bearer 확인)

참여 방식
  개인 링크: 헤더 X-Collect-User / X-Collect-Token (HMAC 서명, COLLECT_SECRET). 닉네임 = 링크 이름.
  공개 참여: payload.meta.nickname (한글 닉네임은 헤더에 못 실어서 본문으로 받음). 활용 안내는 닉네임 창의 고지 문구(meta.notice).
            헤더 X-Owner = 기기 비밀값(localStorage 'vla.owner', 소문자 hex 32자) 필수. 서버는 sha256만 저장(owner 열).
            공개 닉네임은 처음 저장한 기기가 주인 (nick_owners) → 다른 기기는 같은 닉네임을 못 쓴다.
환경변수: SUPABASE_URL, SUPABASE_KEY(service_role), COLLECT_SECRET, (선택) IP_HASH_SALT, CRON_SECRET
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import html
import io
import ipaddress
import json
import math
import os
import re
import sys
import time
import traceback
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlencode, urlparse

import db as dbmod
import scoring
import submission as sub
from collect_common import KST, sanitize_target, verify_name

MAX_BODY_BYTES = 512 * 1024             # 사람 시연 payload 30~50KB (240Hz 펜으로 오래 그려도 ~250KB)
# 시도 횟수 한도 (초, 최대 시도 수) — 저장 실패한 시도도 센다
RATE_LIMITS = ((60, 20), (3600, 200))   # IP 해시 기준 (교실 하나가 같은 와이파이 = 같은 IP로 참여해도 되게)
NICK_LIMITS = ((60, 6), (3600, 60))     # 공개 닉네임(소문자) 기준 (한 사람이 IP를 바꿔 가며 몰아 제출하는 것 방지)
GLOBAL_LIMIT = (60, 600)                # 전체 1분 600회 (봇 폭주 시 DB 보호, 개인 링크 참여자는 제외)
TICKET_TRIES = 5                        # 티켓 하나로 시도 가능한 횟수 (네트워크 실패 후 재시도 허용)
GET_CACHE_S = 10                        # 함수 인스턴스 안 캐시 (CDN 캐시를 못 탄 요청도 DB를 덜 치게)
OWNER_RE = re.compile(r"[0-9a-f]{32}")  # 기기 비밀값 (crypto.getRandomValues 16바이트 → 소문자 hex)
KEPT = "그린 그림은 그대로 있어요"
NICK_TAKEN = "이미 다른 사람이 쓰고 있는 닉네임이에요. 다른 닉네임을 써 주세요"
NEED_RELOAD = "페이지를 새로고침한 뒤 다시 [채점하기]를 눌러 주세요"
EVENTS = frozenset(("visit", "start", "submit", "result", "retry", "next", "share", "card",
                    "demo_done", "demo_skip", "hint_off"))   # 퍼널 이벤트 (모르는 이름은 무시)
EVENT_MAX_BYTES = 32
EVENT_IP_PER_MIN = 300                  # 인스턴스 안 IP당 이벤트 상한 (폭주 시 DB 보호, 넘으면 조용히 무시)
NICK_RE = re.compile(r"^[0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ _\-.]{1,13}$")
BANNED = ("시발", "씨발", "씨팔", "ㅅㅂ", "ㅆㅂ", "병신", "ㅂㅅ", "좆", "존나", "지랄", "개새", "새끼", "니미", "느금",
          "엠창", "애미", "애비", "섹스", "보지", "자지", "fuck", "shit", "bitch", "sex", "porn", "nigg")
RESERVED = ("siat", "admin", "administrator", "관리자", "운영자", "운영진", "system", "root", "official", "ai")
CACHE_PUBLIC = "public, max-age=0, s-maxage=15, stale-while-revalidate=60"   # Vercel CDN이 15초 캐시
CACHE_LONG = "public, max-age=3600, s-maxage=86400"
CACHE_TRAJ = "public, max-age=60, s-maxage=300"   # 기록 궤적: '내 기록 삭제'·운영자 숨김이 5분 안에 반영되게
COPY_TIMEOUT_S = 4                                 # 복제 검사용 상위 기록 조회가 늦으면 검사를 건너뛰고 저장
INPUTS = ("touch", "pen", "mouse")
SITE_URL = "https://handvsai.vercel.app"   # 공유 링크·OG 이미지의 절대 주소 (요청 헤더로 바꿀 수 없게 고정)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
}
_pool = ThreadPoolExecutor(max_workers=4)   # DB 왕복을 겹쳐서 응답 시간 단축

Response = tuple[int, bytes, str]
_db = None


_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn):
    """GET 응답용 짧은 캐시 (저장 응답은 항상 DB에서 새로)."""
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < GET_CACHE_S:
        return hit[1]
    val = fn()
    _cache[key] = (time.monotonic(), val)
    return val


def get_db():
    global _db
    if _db is None:
        _db = dbmod.from_env()
    return _db


def _json(status: int, obj: dict) -> Response:
    return status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def _hdr(headers, key: str) -> str | None:
    v = headers.get(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def _guard(fn):
    """예외 내용은 서버 로그(vercel logs)에만 남기고, 응답에는 오류 번호만 (내부 구조 노출 방지)."""
    def wrapped(*a, **kw) -> Response:
        try:
            return fn(*a, **kw)
        except Exception:
            ref = uuid.uuid4().hex[:8]
            print(f"[error {ref}] {fn.__name__}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            msg = "서버 오류가 났어요. 잠시 후 다시 시도해 주세요"
            if os.environ.get("DEBUG_ERRORS") == "1":
                msg += f" ({traceback.format_exc(limit=1).strip().splitlines()[-1]})"
            return _json(500, {"ok": False, "error": msg, "ref": ref})
    return wrapped


def site_url(headers) -> str:
    """절대 주소. 배포에서는 항상 SITE_URL(헤더로 바꿀 수 없음 → 캐시 오염 방지), 로컬 개발 주소만 그대로."""
    host = _hdr(headers, "host") or ""
    if re.fullmatch(r"(localhost|127\.0\.0\.1)(:\d{1,5})?", host):
        return f"http://{host}"
    return os.environ.get("SITE_URL", SITE_URL).rstrip("/")


def auth(headers) -> tuple[str | None, bool]:
    """(개인 링크 이름 또는 None, 링크를 보냈는데 틀렸는가)."""
    user, token = _hdr(headers, "x-collect-user"), _hdr(headers, "x-collect-token")
    if not (user or token):
        return None, False
    who = verify_name(os.environ.get("COLLECT_SECRET", ""), user, token)
    return who, who is None


def clean_nickname(raw) -> tuple[str | None, str | None]:
    """(닉네임, 오류). NFKC 정규화(전각·호환 문자 → 일반 문자) 후 허용 문자만, 욕설·예약어 거절."""
    nick = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(raw or ""))).strip()
    if not NICK_RE.match(nick):
        return None, "닉네임은 1~13자 한글·영문·숫자로 적어 주세요"
    squashed = re.sub(r"[\s_.\-]", "", nick.lower())
    if any(b in squashed for b in BANNED):
        return None, "다른 닉네임을 써 주세요"
    if squashed in RESERVED:
        return None, "이미 쓰고 있는 이름이에요. 다른 닉네임을 써 주세요"
    return nick, None


def client_ip(headers, peer: str | None) -> str:
    """Vercel이 채우는 헤더(클라이언트가 보낸 값은 덮어씀) 우선. 로컬은 소켓 주소.
    IPv6는 /64 단위로 묶는다 (가정 회선 하나가 주소를 수없이 바꿔 한도를 피하지 못하게)."""
    raw = (_hdr(headers, "x-vercel-forwarded-for") or _hdr(headers, "x-real-ip")
           or _hdr(headers, "x-forwarded-for") or peer or "?").split(",")[0].strip()
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return raw[:64]
    if ip.version == 6:
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def ip_hash(headers, peer: str | None) -> str:
    salt = os.environ.get("IP_HASH_SALT") or os.environ.get("COLLECT_SECRET", "")
    return hashlib.sha256(f"{salt}|{client_ip(headers, peer)}".encode()).hexdigest()[:16]


def _secret() -> str:
    return os.environ.get("COLLECT_SECRET", "")


def owner_hash(secret: str) -> str:
    """기기 비밀값 → 저장용 해시 (sha256 hex 64자). 비밀값 자체는 어디에도 저장하지 않는다."""
    return hashlib.sha256(secret.encode()).hexdigest()


def owner_of(headers) -> str | None:
    """헤더 X-Owner(소문자 hex 32자)의 해시. 없거나 형식이 틀리면 None."""
    raw = _hdr(headers, "x-owner")
    return owner_hash(raw) if raw and OWNER_RE.fullmatch(raw) else None


def stats_cache(headers) -> str:
    """개인 링크 요청은 사람마다 응답이 달라 CDN에 캐시하지 않는다."""
    return "private, no-store" if _hdr(headers, "x-collect-user") else CACHE_PUBLIC


# ---------------------------------------------------------------------------
@_guard
def stats(headers) -> Response:
    who, bad = auth(headers)
    if bad:
        return _json(401, {"ok": False, "hosted": True, "error": "개인 링크가 올바르지 않습니다"})
    db = get_db()
    counts = _cached("counts", db.counts)
    out = {"ok": True, "hosted": True, "contributor": who,
           "counts": {k: v["total"] for k, v in counts.items()},
           "qualified": {k: v["qualified"] for k, v in counts.items()},
           "total": sum(v["total"] for v in counts.values())}
    if who:
        out["mine"] = db.contributor_count(who)
    return _json(200, out)


@_guard
def ticket(headers) -> Response:
    return _json(200, {"ok": True, "ticket": sub.issue_ticket(_secret())})


def _limits(db, iph: str, tk: str, trusted: bool, nick: str | None = None, owner: str | None = None) -> Response | None:
    """시도 횟수 한도 + 닉네임 주인 확인. 저장 성공 여부와 무관하게 '시도'를 DB에서 원자적으로 센다(동시 요청으로도 못 넘김).
    IP별 1분·1시간, 공개 닉네임별 1분·1시간, 티켓당 시도, 전체 1분(개인 링크 참여자는 전체 한도에서 제외).
    nick = 공개 참여 닉네임 소문자(개인 링크는 None), owner = 기기 비밀값의 해시."""
    try:
        h = db.hit_rate(iph, tk, nick, owner)
    except dbmod.DBError as e:
        if "hit_rate" not in str(e):
            raise
        print(f"[warn] hit_rate 함수를 쓸 수 없어 저장된 행 기준으로 셉니다 (닉네임 확인 없음): {str(e)[:160]}",
              file=sys.stderr, flush=True)
        h = None   # 스키마(hit_rate 함수)가 아직 없으면 예전 방식(저장된 행 기준)으로
    if h is None:
        longest = max(sec for sec, _ in RATE_LIMITS)
        f_ip = _pool.submit(db.ip_recent, iph, longest, max(n for _, n in RATE_LIMITS))
        f_all = _pool.submit(db.global_recent_count, GLOBAL_LIMIT[0])
        now, recent = dt.datetime.now(dt.timezone.utc), f_ip.result()
        h = {"ip_min": sum((now - t).total_seconds() < 60 for t in recent) + 1, "ip_hour": len(recent) + 1,
             "all_min": f_all.result() + 1, "ticket_n": 1, "nick_min": 0, "nick_hour": 0, "nick_ok": True}
    if not h.get("nick_ok", True):
        return _json(400, {"ok": False, "need": "nickname", "error": NICK_TAKEN})
    if h["ip_min"] > RATE_LIMITS[0][1]:
        return _json(429, {"ok": False, "error": f"같은 인터넷(와이파이)에서 제출이 몰렸어요. 1분 뒤에 다시 [채점하기]를 눌러 주세요. {KEPT}"})
    if h["ip_hour"] > RATE_LIMITS[1][1]:
        return _json(429, {"ok": False, "error": f"같은 인터넷(와이파이)에서 1시간 동안 너무 많이 제출했어요. "
                                                 f"잠시 후 다시 [채점하기]를 눌러 주세요. {KEPT}"})
    if h.get("nick_min", 0) > NICK_LIMITS[0][1]:
        return _json(429, {"ok": False, "error": f"이 닉네임으로 너무 빨리 제출하고 있어요. 1분 뒤에 다시 [채점하기]를 눌러 주세요. {KEPT}"})
    if h.get("nick_hour", 0) > NICK_LIMITS[1][1]:
        return _json(429, {"ok": False, "error": f"이 닉네임으로 1시간 동안 너무 많이 제출했어요. "
                                                 f"잠시 후 다시 [채점하기]를 눌러 주세요. {KEPT}"})
    if h["ticket_n"] > TICKET_TRIES:
        return _json(400, {"ok": False, "need": "ticket", "ticket": sub.issue_ticket(_secret()),
                           "error": "같은 그림으로 너무 여러 번 시도했어요. 그림을 지우고 다시 그려 주세요"})
    if not trusted and h["all_min"] > GLOBAL_LIMIT[1]:
        return _json(503, {"ok": False, "error": f"지금 참여자가 너무 많아요. 1분 뒤에 다시 [채점하기]를 눌러 주세요. {KEPT}"})
    return None


_traj_cache: dict[str, list] = {}   # 기록 id → 궤적 (저장된 궤적은 바뀌지 않으므로 인스턴스 안에서 재사용)


def _copy_candidates(db, target: str) -> list[tuple[str, float, list]]:
    """복제 검사 대상 = 이 도안 리더보드 상위 기록들 [(id, 점수, 궤적), ...]."""
    rows = _cached(f"lb{dbmod.LB_SIZE}:{target}", lambda: db.leaderboard(target, dbmod.LB_SIZE))
    need = [r["id"] for r in rows if r.get("id") and r["id"] not in _traj_cache]
    if need:
        got = db.trajectories(need)
        if len(_traj_cache) > 500:
            _traj_cache.clear()
        _traj_cache.update(got)
    return [(r["id"], r.get("score") or 0, _traj_cache.get(r["id"])) for r in rows if r.get("id")]


def _copy_check(f_cands, score: float, traj) -> tuple[str, float] | None:
    """상위 기록과 선 모양이 거의 같으면 (그 기록 id, 거리 px). 조회·계산 실패는 저장을 막지 않고 건너뛴다."""
    try:
        cands = f_cands.result(timeout=COPY_TIMEOUT_S)
        if len(cands) >= dbmod.LB_SIZE and score < min(s for _, s, _ in cands):
            return None   # 상위권에 들지 못하는 기록은 복제해도 얻는 것이 없음 → 계산 생략
        return sub.find_copy(traj, [(cid, t) for cid, _, t in cands if t])
    except Exception as e:   # noqa: BLE001
        print(f"[warn] 복제 검사 생략: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
        return None


def _input_type(strokes) -> str:
    n = {k: sum(len(s["points"]) for s in strokes if s["pointer_type"] == k) for k in ("touch", "pen", "mouse")}
    best = max(n, key=n.get)
    return best if n[best] else "unknown"


@_guard
def save(headers, body: bytes, peer: str | None = None) -> Response:
    if len(body) > MAX_BODY_BYTES:
        return _json(413, {"ok": False, "error": "요청이 너무 큽니다"})
    try:
        payload = sub.clean_payload(json.loads(body.decode("utf-8")))
    except sub.Rejected as e:              # (ValueError의 하위 클래스라 먼저 잡아야 구체적인 안내가 나감)
        return _json(400, {"ok": False, "error": str(e)})
    except (ValueError, RecursionError):   # JSONDecodeError·UnicodeDecodeError·거대한 정수
        return _json(400, {"ok": False, "error": "요청 형식이 올바르지 않습니다"})
    meta = payload["meta"]
    target = sanitize_target(meta.get("target"))
    if target not in scoring.load_templates():
        return _json(400, {"ok": False, "error": "알 수 없는 도안입니다"})

    who, bad = auth(headers)
    if bad:
        return _json(401, {"ok": False, "error": "개인 링크가 올바르지 않습니다"})
    owner = owner_of(headers)   # 개인 링크는 없어도 됨 (있으면 저장 → 이 기기에서 '내 기록 지우기' 가능)
    if who:
        nickname = who
    else:
        nickname, nerr = clean_nickname(meta.get("nickname"))
        if nerr:
            return _json(400, {"ok": False, "error": nerr, "need": "nickname"})
        if not owner:   # 기기 비밀값이 없는 옛 화면(캐시) → 새로고침하면 새 수집기가 비밀값을 만든다 (그림은 자동 보관)
            return _json(400, {"ok": False, "need": "reload", "error": NEED_RELOAD})

    # 제출 티켓: 서버가 서명한 발급 시각. 펜이 닿아 있던 시간보다 먼저 발급됐어야 한다.
    # (쉬는 시간·새로고침 후 이어 그리기는 빼고 실제로 그린 시간만 비교 → 사람이 걸리지 않게 느슨하게)
    tk = _hdr(headers, "x-draw-ticket")
    age = sub.ticket_age(_secret(), tk)
    draw_s = sum(s["points"][-1][2] - s["points"][0][2] for s in payload["strokes"]) / 1000
    if age is None or age + sub.TICKET_SLACK_S < draw_s:
        return _json(400, {"ok": False, "need": "ticket",
                           "error": "보안 확인이 만료됐어요. 그림을 지우고 다시 그려 주세요"})

    db = get_db()
    iph = ip_hash(headers, peer)
    limited = _limits(db, iph, tk, trusted=bool(who), nick=None if who else nickname.lower(),
                      owner=None if who else owner)   # nick = lower(닉네임) = 리더보드가 묶는 기준
    if limited:
        return limited

    # 복제 검사용 상위 기록 조회는 채점과 겹쳐서 (DB 왕복 시간을 숨김)
    f_cands = _pool.submit(_copy_candidates, db, target)
    # 궤적은 서버가 원시 입력으로 직접 만든다 (클라이언트 trajectory는 받지 않음).
    # 목표 이미지(학습 관측 ch1)도 서버의 표준 이미지로 바꾼다 (클라이언트 이미지로 학습 데이터 오염 방지).
    payload["trajectory"] = sub.build_trajectory(payload["strokes"])
    payload["target_128_png"] = sub.canonical_target(target) or payload["target_128_png"]
    sc = scoring.score_trajectory(target, payload["trajectory"])
    flags, measures = sub.human_checks(payload["strokes"], sc["avg_error"], blur_ms=float(meta.get("blur_ms") or 0.0))
    measures["time_repair_ms"] = meta.pop("time_repair_ms", 0)
    if measures["time_repair_ms"] > sub.MAX_TIME_REPAIR_MS:   # 0.5초 넘게 거꾸로 = 이벤트 섞임으로 볼 수 없음
        flags = sorted(set(flags) | {"time_backwards"})
    copy = _copy_check(f_cands, sc["score"], payload["trajectory"]) if sc["qualified"] and not flags else None
    if copy:       # 남의 상위 기록을 살짝 옮기거나 흔든 복제 의심 → 운영자가 보고 판단 (tools/moderate.py)
        measures.update(copy_of=copy[0], copy_px=copy[1])
    review = not flags and sc["qualified"] and (sc["avg_error"] < sub.REVIEW_AVG_ERROR_PX or copy is not None)
    if flags:      # 저장은 하되 기록으로 인정하지 않음 (리더보드·학습 제외)
        sc.update(qualified=False, reason="검증 실패")
    elif review:   # 사람 기록보다 훨씬 정밀하거나 상위 기록과 거의 같음 → 운영자 확인 후 리더보드에 (tools/moderate.py)
        sc.update(qualified=False, reason="검토 대기")
    input_type = _input_type(payload["strokes"])
    meta.update(target=target, nickname=nickname, contributor=who, score=sc, smoothing="ma3x2",
                trajectory_source="server", input_type=input_type, num_steps=len(payload["trajectory"]),
                num_strokes=len(payload["strokes"]), flags=flags, checks=measures, review=review,
                has_pressure=sub.has_pressure(payload["strokes"]),
                server_received_at=dt.datetime.now(KST).isoformat(timespec="milliseconds"))
    if any(f not in sub.SOFT_FLAGS for f in flags):
        # 자동 입력이 분명함(속도·시간·점 빈도): 원시 입력·이미지는 저장하지 않음 (DB 채우기 공격 방지).
        # 약한 신호(너무 정확함·시간 역행)만이면 원시 기록을 남긴다 — 사람 오탐일 수 있음 (아이패드 사용자 사례)
        payload = {"schema_version": 1, "meta": meta}
    try:
        row_id = db.insert_demo({
            "nickname": nickname, "contributor": who, "template_name": target,
            "score": sc["score"], "tier": sc["tier"], "f1": sc["f1"], "coverage": sc["coverage"],
            "precision": sc["precision"], "strokes": sc["strokes"], "is_qualified": sc["qualified"],
            "trajectory_data": payload, "ip_hash": iph, "avg_error": sc["avg_error"], "input_type": input_type,
            "traj_hash": sub.traj_hash(payload["strokes"]) if "strokes" in payload else None, "ticket": tk,
            **({"owner": owner} if owner else {}),
        })
    except dbmod.Duplicate:
        return _json(409, {"ok": False, "duplicate": True, "ticket": sub.issue_ticket(_secret()),
                           "error": "이미 저장된 그림이에요. 새로 그려 주세요"})

    draw_sec = (measures.get("draw_ms") or 0) / 1000
    time_qs = f"&time={draw_sec:.1f}" if draw_sec > 0 else ""
    out = {"ok": True, "hosted": True, "id": row_id, "nickname": nickname, "contributor": who,
           "input_type": input_type, "template": target, **sc, "draw_ms": measures.get("draw_ms", 0), "flags": flags, "review": review, "ticket": sub.issue_ticket(_secret()),
           "share_url": f"{site_url(headers)}/s?score={sc['score']:.1f}&tier={sc['tier']}&template={target}{time_qs}"}
    f_rank = _pool.submit(db.rank_of, target, sc["score"], sc["avg_error"]) if sc["qualified"] else None
    f_counts = _pool.submit(db.counts)
    f_mine = _pool.submit(db.contributor_count, who) if who else None
    f_top = _pool.submit(db.leaderboard, target, 5)
    if f_rank:
        better, total = f_rank.result()
        out.update(rank=better + 1, ranked_total=total, top_percent=max(1, round(100 * (better + 1) / max(total, 1))))
    out["counts"] = {k: v["total"] for k, v in f_counts.result().items()}
    out["total"] = sum(out["counts"].values())
    if f_mine:
        out["mine"] = f_mine.result()
    _cache.clear()
    out["top"] = [_lb_row(r) for r in f_top.result()]
    return _json(200, out)


@_guard
def leaderboard(query: dict) -> Response:
    target = sanitize_target((query.get("template") or [""])[0])
    if target not in scoring.load_templates():
        return _json(400, {"ok": False, "error": "template 파라미터가 필요합니다"})
    inp = (query.get("input") or [""])[0]
    if inp in INPUTS:   # 입력 장치별 순위 (펜·터치·마우스를 따로)
        rows = _cached(f"lb:{target}:{inp}", lambda: get_db().leaderboard_by_input(target, inp, 5))
        return _json(200, {"ok": True, "template": target, "input": inp, "top": [_lb_row(r) for r in rows]})
    rows = _cached(f"lb:{target}", lambda: get_db().leaderboard(target, 5))
    return _json(200, {"ok": True, "template": target, "top": [_lb_row(r) for r in rows]})


def _lb_row(r: dict) -> dict:
    return {**{k: r.get(k) for k in ("id", "rank", "nickname", "score", "avg_error", "tier", "input_type")},
            "score_estimated": r.get("score_estimated") is True}


@_guard
def trajectory(query: dict) -> Response:
    raw_id = (query.get("id") or [""])[0].strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", raw_id):
        return _json(400, {"ok": False, "error": "올바른 id가 아닙니다"})

    # 인정된 기록(is_qualified)만: 검증 실패·검토 대기·운영자가 내린 기록(--hide)·낙서는 도전장·고스트로 보여 주지 않는다
    db = get_db()
    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "drawing_demos", {
            "select": "id,nickname,template_name,score,avg_error,input_type,trajectory_data",
            "id": f"eq.{raw_id}",
            "is_qualified": "eq.true",
            "limit": 1
        })
        row = res[0] if (res and isinstance(res, list)) else None
    else:
        rows = db._rows(
            "select id, nickname, template_name, score, avg_error, input_type, trajectory_data from drawing_demos "
            "where id=? and is_qualified=1",
            (raw_id,)
        )
        row = rows[0] if rows else None

    if not row:
        return _json(404, {"ok": False, "error": "기록을 찾을 수 없습니다"})

    traj = row.get("trajectory_data") or {}
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except Exception:
            traj = {}

    strokes = traj.get("strokes") or []
    return _json(200, {
        "ok": True,
        "id": row.get("id"),
        "nickname": row.get("nickname"),
        "template": row.get("template_name"),
        "score": row.get("score"),
        "avg_error": row.get("avg_error"),
        "input_type": row.get("input_type"),
        "strokes": strokes,
    })


# ---------------------------------------------------------------------------
# 퍼널 통계 · 내 기록 지우기 · 깨우기(cron)
# ---------------------------------------------------------------------------
_ev: dict = {"minute": -1, "hits": {}}


def _event_allowed(key: str) -> bool:
    """인스턴스 안 1분 창 IP당 상한 (정확할 필요 없는 보호막. 넘으면 세지 않고 조용히 무시)."""
    minute = int(time.time() // 60)
    if _ev["minute"] != minute:
        _ev.update(minute=minute, hits={})
    n = _ev["hits"][key] = _ev["hits"].get(key, 0) + 1
    return n <= EVENT_IP_PER_MIN


def event(headers, body: bytes | None, peer: str | None = None) -> Response:
    """POST /api/event — 본문 = 이벤트 이름(text/plain, navigator.sendBeacon). 오늘(KST) 횟수 +1.
    모르는 이름·큰 본문은 무시. DB 함수가 없거나 DB 오류여도 항상 204 (통계가 화면 흐름을 막지 않게)."""
    try:
        name = body.decode("ascii", "replace").strip() if body and len(body) <= EVENT_MAX_BYTES else ""
        if name in EVENTS and _event_allowed(ip_hash(headers, peer)):
            get_db().bump_event(name)
    except Exception as e:   # noqa: BLE001 — 통계 실패는 로그만
        print(f"[event] 기록 실패: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
    return 204, b"", "text/plain; charset=utf-8"


@_guard
def delete(headers) -> Response:
    """POST /api/delete — 헤더 X-Owner(이 기기의 비밀값)로 저장한 기록 전부 삭제 + 이 기기가 가진 닉네임 반납."""
    owner = owner_of(headers)
    if not owner:
        return _json(400, {"ok": False, "error": "이 기기의 확인 정보가 없어요. 페이지를 새로고침한 뒤 다시 시도해 주세요"})
    n = get_db().delete_by_owner(owner)
    _cache.clear()   # 이 인스턴스의 리더보드·통계 캐시 (CDN 캐시는 15초 안에 갱신)
    return _json(200, {"ok": True, "deleted": n})


@_guard
def cron(headers) -> Response:
    """GET /api/cron — DB를 한 번 가볍게 읽는다 (Supabase 무료 플랜은 일주일 동안 요청이 없으면 일시정지).
    환경변수 CRON_SECRET이 있으면 Vercel Cron이 보내는 'Authorization: Bearer <CRON_SECRET>'만 받는다."""
    want = os.environ.get("CRON_SECRET", "").strip()
    if want:
        got = (_hdr(headers, "authorization") or "").encode("utf-8", "replace")
        if not hmac.compare_digest(got, f"Bearer {want}".encode()):
            return _json(401, {"ok": False, "error": "unauthorized"})
    get_db().keepalive()
    return _json(200, {"ok": True})


# ---------------------------------------------------------------------------
# 공유: OG 이미지 + 미리보기 페이지
# ---------------------------------------------------------------------------
TIER_COLORS = {"S": (255, 196, 50), "A": (90, 200, 255), "B": (140, 220, 140), "C": (170, 170, 180)}


@lru_cache(maxsize=1)
def ai_scores() -> dict[str, float]:
    """도안별 AI 점수 — index.html의 AI_DEMOS('AI vs 나' 결과 창과 같은 값)를 읽는다 (정의는 한 곳에만)."""
    try:
        src = scoring.INDEX_HTML.read_text(encoding="utf-8")
        i = src.index("const AI_DEMOS")
        line = src[i:src.index("\n", i)]
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in re.findall(r'"([a-z0-9-]+)":\{"score":([0-9]+(?:\.[0-9]+)?)', line)}


def share_params(query: dict) -> dict | None:
    """?score=96&tier=S&template=apple&time=12.4 검증. 하나라도 이상하면 None (기본 카드)."""
    try:
        score = math.floor(float((query.get("score") or [""])[0][:12]) * 10 + 1e-6) / 10   # 소수 첫째 자리까지 (버림)
        target = sanitize_target((query.get("template") or [""])[0])
    except (ValueError, OverflowError):
        return None
    if not (math.isfinite(score) and 0 <= score <= 100 and target in scoring.load_templates()):
        return None

    time_val = None
    time_raw = (query.get("time") or [""])[0][:12]
    if time_raw:
        try:
            t = float(time_raw)
            if math.isfinite(t) and 0.1 <= t <= 600.0:
                time_val = round(t, 1)
        except (ValueError, OverflowError):
            pass

    tier = scoring.tier_for(score)[0]   # 주소의 tier 값은 무시하고 점수로 계산
    return {"score": score, "score_text": f"{score:.1f}", "tier": tier, "template": target,
            "time": time_val, "time_text": f"{time_val:.1f}" if time_val is not None else None,
            "label": scoring.tier_for(score)[1], "label_en": scoring.tier_for(score)[2],
            "template_label": scoring.load_templates()[target]["label"]}


@_guard
def og_image(query: dict) -> Response:
    from PIL import Image, ImageDraw, ImageFont

    p = share_params(query)
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), (13, 17, 23))
    d = ImageDraw.Draw(img)
    font = lambda s: ImageFont.load_default(size=s)
    # 왼쪽: 도안 선화
    box, bx, by = 480, 70, 75
    d.rounded_rectangle((bx - 10, by - 10, bx + box + 10, by + box + 10), 24, fill=(0, 0, 0), outline=(43, 52, 64), width=2)
    color = TIER_COLORS[p["tier"]] if p else (125, 211, 252)
    for part in scoring.template_parts(p["template"] if p else "apple"):
        d.line([(bx + x * box, by + y * box) for x, y in part], fill=color, width=7, joint="curve")
    # 오른쪽: 점수·티어
    x0 = 620
    d.text((x0, 70), "BEAT THE AI", font=font(44), fill=(230, 237, 243))
    d.text((x0, 124), "Drawing Precision Test", font=font(30), fill=(141, 153, 168))
    if p:
        d.text((x0, 185), p["score_text"], font=font(150), fill=color)
        d.text((x0 + 12 + d.textlength(p["score_text"], font=font(150)), 295), "/100", font=font(40), fill=(141, 153, 168))

        # 소요 시간 & AI 비교 배지
        sub_y = 350
        tx = x0
        if p.get("time_text"):
            d.text((tx, sub_y), f"TIME {p['time_text']}s", font=font(28), fill=(200, 210, 220))
            tx += int(d.textlength(f"TIME {p['time_text']}s", font=font(28))) + 28

        ai_ref = ai_scores().get(p["template"])   # 이 도안의 AI 점수 (없으면 비교 배지 생략)
        if ai_ref is not None:
            diff = p["score"] - ai_ref
            if diff >= 0:
                ai_badge = f"BEAT AI (+{diff:.1f}pts)"
                badge_color = (52, 211, 153)
            else:
                ai_badge = f"VS AI ({diff:.1f}pts)"
                badge_color = (248, 113, 113)
            d.text((tx, sub_y), ai_badge, font=font(28), fill=badge_color)

        d.rounded_rectangle((x0, 420, x0 + 90, 510), 18, fill=color)
        d.text((x0 + 45, 466), p["tier"], font=font(64), fill=(13, 17, 23), anchor="mm")
        d.text((x0 + 110, 440), f"{p['tier']}-TIER", font=font(34), fill=(230, 237, 243))
        d.text((x0 + 110, 480), p["label_en"], font=font(26), fill=(141, 153, 168))
    else:
        d.text((x0, 230), "How precise\nis your hand?", font=font(72), fill=(255, 196, 50), spacing=10)
        d.text((x0, 440), "Trace the template. Get your F1 score.", font=font(28), fill=(141, 153, 168))
    d.text((x0, 560), "handvsai.vercel.app", font=font(26), fill=(76, 141, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return 200, buf.getvalue(), "image/png"


@_guard
def share_page(headers, query: dict) -> Response:
    site, p = site_url(headers), share_params(query)
    if p:
        ai_ref = ai_scores().get(p["template"])   # 이 도안의 AI 점수 (없으면 비교 문구 생략)
        if ai_ref is None:
            ai_status = ""
        else:
            diff = p["score"] - ai_ref
            ai_status = f"AI 격파 (+{diff:.1f}점)!" if diff >= 0 else f"AI 기록({ai_ref:.1f}점)까지 {abs(diff):.1f}점!"
        time_info = f" ({p['time_text']}초)" if p.get("time_text") else ""
        time_qs = f"&time={p['time_text']}" if p.get("time_text") else ""

        title = f"{p['score_text']}점{time_info} · {p['tier']}급 {p['label']} | AI를 이겨라! 내 손 드로잉 정밀도 테스트"
        desc = f"{ai_status + ' ' if ai_status else ''}{p['template_label']} 도안 정밀도 {p['score_text']}점{time_info}. 내 손은 상위 몇 %일까? 지금 도전해 보세요."
        og = f"{site}/api/og?score={p['score_text']}&tier={p['tier']}&template={p['template']}{time_qs}"
    else:
        title = "AI를 이겨라! 내 손 드로잉 정밀도 테스트"
        desc = "화면을 보고 직접 선을 긋는 시각 제어 AI의 기록은 100.0점(S급 신의 손)! 당신의 손가락은 몇 점인가요?"
        og = f"{site}/api/og"
    e = html.escape
    page = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(title)}</title>
<meta name="description" content="{e(desc)}"><meta name="robots" content="noindex">
<meta property="og:type" content="website"><meta property="og:site_name" content="AI를 이겨라! 드로잉 테스트">
<meta property="og:title" content="{e(title)}"><meta property="og:description" content="{e(desc)}">
<meta property="og:image" content="{e(og)}"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta property="og:url" content="{e(site)}/"><meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{e(title)}"><meta name="twitter:description" content="{e(desc)}"><meta name="twitter:image" content="{e(og)}">
<link rel="canonical" href="{e(site)}/"><meta http-equiv="refresh" content="0; url=/">
</head><body style="background:#0d1117;color:#e6edf3;font-family:sans-serif">
<p><a href="/" style="color:#4c8dff">{e(title)} — 도전하러 가기</a></p></body></html>"""
    return 200, page.encode("utf-8"), "text/html; charset=utf-8"


# ---------------------------------------------------------------------------
def respond(handler, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
    """BaseHTTPRequestHandler에 응답 쓰기. 오류 응답은 캐시하지 않는다."""
    if status == 204:   # 본문 없음 (204에는 Content-Length·Content-Type을 보내지 않는다)
        return respond_empty(handler)
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", cache if status == 200 else "no-store")
    for k, v in SECURITY_HEADERS.items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


def respond_empty(handler, status: int = 204) -> None:
    handler.send_response(status)
    handler.send_header("Cache-Control", "no-store")
    for k, v in SECURITY_HEADERS.items():
        handler.send_header(k, v)
    handler.end_headers()


# ---------------------------------------------------------------------------
# HTTP 진입점 (api/*.py와 tools/vercel_local.py가 공유)
#   GET은 정규 주소(허용한 파라미터만, 정해진 형식)로만 실제 처리한다. 다른 모양이면 정규 주소로 308.
#   → 쿼리를 바꿔 CDN 캐시를 우회하며 DB 조회·이미지 생성을 반복시키는 공격 방지. 정상 사용자 주소는 이미 정규형.
# ---------------------------------------------------------------------------
GET_PATHS = {"stats": "/api/stats", "leaderboard": "/api/leaderboard", "og": "/api/og", "share": "/s", "ticket": "/api/ticket", "trajectory": "/api/trajectory"}


def canonical_query(name: str, headers, q: dict) -> str | None:
    """정규 쿼리 문자열. None이면 정규화하지 않음(오류 응답 등)."""
    if name == "stats":
        who = _hdr(headers, "x-collect-user")
        return urlencode({"me": who}) if who else ""
    if name == "leaderboard":
        t = sanitize_target((q.get("template") or [""])[0])
        inp = (q.get("input") or [""])[0]
        if t not in scoring.load_templates():
            return None
        return f"template={t}" + (f"&input={inp}" if inp in INPUTS else "")
    if name == "trajectory":
        raw_id = (q.get("id") or [""])[0].strip().lower()
        return f"id={raw_id}" if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", raw_id) else None
    if name == "og":   # 공유 페이지(/s)는 Vercel 경로 재작성을 거치고 생성 비용도 작아 정규화하지 않음
        p = share_params(q)
        if not p:
            return ""
        time_qs = f"&time={p['time_text']}" if p.get("time_text") else ""
        return f"score={p['score_text']}&tier={p['tier']}&template={p['template']}{time_qs}"
    if name == "ticket":
        return ""
    return None


def _cache_for(name: str, headers) -> str:
    return {"stats": stats_cache(headers), "leaderboard": CACHE_PUBLIC, "og": CACHE_LONG,
            "share": CACHE_LONG, "trajectory": CACHE_TRAJ}.get(name, "no-store")


def serve_get(handler, name: str) -> None:
    u = urlparse(handler.path)
    q = parse_qs(u.query, keep_blank_values=True)
    canon = canonical_query(name, handler.headers, q)
    # 쿼리만 본다 (경로까지 보면 Vercel이 넘기는 경로 모양에 따라 반복 리다이렉트 위험). 경로를 바꿔도 쿼리는 정규형만 통과.
    if canon is not None and u.query != canon:
        loc = GET_PATHS[name] + (f"?{canon}" if canon else "")
        handler.send_response(308)
        handler.send_header("Location", loc)
        handler.send_header("Content-Length", "0")
        handler.send_header("Cache-Control", "no-store" if name in ("stats", "ticket") else CACHE_LONG)
        for k, v in SECURITY_HEADERS.items():
            handler.send_header(k, v)
        handler.end_headers()
        return
    resp = {"stats": lambda: stats(handler.headers), "leaderboard": lambda: leaderboard(q),
            "og": lambda: og_image(q), "share": lambda: share_page(handler.headers, q),
            "ticket": lambda: ticket(handler.headers),
            "trajectory": lambda: trajectory(q)}[name]()
    respond(handler, *resp, cache=_cache_for(name, handler.headers))


def serve_save(handler) -> None:
    body = read_body(handler)
    if body is None:
        return respond(handler, *_json(413, {"ok": False, "error": "요청이 너무 큽니다"}))
    respond(handler, *save(handler.headers, body, handler.client_address[0]))


def serve_event(handler) -> None:
    body = read_body(handler)              # 읽어서 비워 둔다 (32바이트 넘는 본문은 event()가 무시)
    event(handler.headers, body, handler.client_address[0])
    respond_empty(handler)                 # 무엇이 와도 204 (no-store)


def serve_delete(handler) -> None:
    read_body(handler)                     # 본문은 쓰지 않음 (헤더 X-Owner만)
    respond(handler, *delete(handler.headers))


def serve_cron(handler) -> None:
    respond(handler, *cron(handler.headers))   # 정규 주소 리다이렉트 없음, 항상 no-store


def _method_not_allowed(handler) -> None:
    read_body(handler)   # Windows TCP RST(WinError 10053) 방지: 요청 본문을 비운 뒤 405 응답
    respond(handler, *_json(405, {"ok": False, "error": "지원하지 않는 요청입니다"}))


POST_ROUTES = {"save": serve_save, "event": serve_event, "delete": serve_delete}
GET_ROUTES = {"cron": serve_cron}   # 나머지 GET은 serve_get (정규 주소 308·캐시)
HANDLER_NAMES = ("save", "stats", "leaderboard", "og", "share", "ticket", "event", "delete", "cron", "trajectory")


def make_handler(name: str):
    """Vercel 함수용 BaseHTTPRequestHandler 클래스. name = HANDLER_NAMES 중 하나 (api/<name>.py)."""
    if name not in HANDLER_NAMES:
        raise ValueError(f"unknown handler: {name}")

    class Handler(BaseHTTPRequestHandler):
        server_version, sys_version = "", ""   # Server 헤더에 파이썬 버전을 싣지 않음

        def do_GET(self):
            if name in POST_ROUTES:
                return _method_not_allowed(self)
            if name in GET_ROUTES:
                return GET_ROUTES[name](self)
            serve_get(self, name)

        def do_POST(self):
            if name not in POST_ROUTES:
                return _method_not_allowed(self)
            POST_ROUTES[name](self)

        do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed

        def log_message(self, fmt, *a):   # 요청 주소(공유 파라미터 등)를 로그에 남기지 않음
            pass
    return Handler


def read_body(handler, limit: int = MAX_BODY_BYTES) -> bytes | None:
    """Content-Length만큼 읽기. 한도를 넘으면 읽지 않고 None."""
    try:
        n = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        n = 0
    if n > limit:
        return None
    return handler.rfile.read(n) if n > 0 else b""
