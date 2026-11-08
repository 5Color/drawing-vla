"""Supabase 실시간 모니터 — 요약·오늘 퍼널·도안별 현황·최신 제출을 5초마다 갱신 (Ctrl+C로 종료, 읽기 전용).

  python tools/monitor.py                 기본 5초 주기, 최신 15건
  python tools/monitor.py --interval 10 --limit 20
  python tools/monitor.py --no-color      색 끄기 (로그로 남길 때)
  python tools/monitor.py --sqlite outputs/local_bench.sqlite   (로컬 시뮬레이터)

한글·기호 폭을 계산해 열을 맞추고(한글 = 2칸), 이모지 대신 글자와 색으로 표시한다(터미널마다 이모지 폭이 달라 줄이 밀림).
접속 정보: SUPABASE_URL/SUPABASE_KEY 환경변수 또는 .collect-secrets.json (service_role/secret 키).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import unicodedata

import collect_secrets
import db as dbmod
import scoring

KST = dt.timezone(dt.timedelta(hours=9))
COLS = ("nickname,contributor,template_name,score,tier,avg_error,input_type,is_qualified,created_at,"
        "flags:trajectory_data->meta->flags,review:trajectory_data->meta->review")
INPUT = {"touch": "터치", "pen": "펜", "mouse": "마우스"}
FUNNEL = (("visit", "방문"), ("start", "첫 획"), ("submit", "채점"), ("result", "결과"))   # 앞 단계 대비 %
AFTER = (("retry", "다시"), ("next", "다음"), ("share", "공유"))                         # 결과 대비 %
EXTRA = (("card", "카드"), ("demo_done", "AI 시연 끝"), ("demo_skip", "시연 건너뜀"), ("hint_off", "안내 끔"))
USE_COLOR = True


def c(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOR else text


TIER_COLOR = {"S": "1;33", "A": "1;36", "B": "32", "C": "90"}


# --- 한글 폭 맞추기 -----------------------------------------------------------
def dw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 0 if unicodedata.combining(ch) else 1 for ch in s)


def fit(s: str, width: int, right: bool = False) -> str:
    """표시 폭 width에 맞춰 자르거나(…) 채운다."""
    s = str(s)
    if dw(s) > width:
        out = ""
        for ch in s:
            if dw(out + ch) > width - 1:
                break
            out += ch
        s = out + "…"
    pad = " " * (width - dw(s))
    return pad + s if right else s + pad


# --- 데이터 -----------------------------------------------------------------
def recent(db, limit: int) -> list[dict]:
    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "drawing_demos", {"select": COLS, "order": "created_at.desc", "limit": limit})
        return res or []
    rows = db._rows("select nickname, contributor, template_name, score, tier, avg_error, input_type, is_qualified, created_at, "
                    "json_extract(trajectory_data, '$.meta.flags') flags, json_extract(trajectory_data, '$.meta.review') review "
                    "from drawing_demos order by created_at desc limit ?", (limit,))
    return [{**r, "flags": None if r["flags"] in (None, "[]") else r["flags"]} for r in rows]


def funnel_counts(db, day: str) -> dict[str, int]:
    """funnel_counts 표의 그날(KST) 이벤트별 횟수."""
    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "funnel_counts", {"select": "event,n", "day": f"eq.{day}"})
    else:
        res = db._rows("select event, n from funnel_counts where day=?", (day,))
    return {r["event"]: int(r["n"] or 0) for r in res or []}


def pct(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "-"


def funnel_line(db, now: dt.datetime) -> str:
    """오늘 퍼널 한 줄: 방문 → 첫 획 → 채점 → 결과 (앞 단계 대비 %) → 다시·다음·공유 (결과 대비 %)."""
    head = c("  오늘 퍼널", "1") + "  "
    try:
        n = funnel_counts(db, now.date().isoformat())
    except dbmod.DBError as e:
        if dbmod.is_missing(e, "funnel_counts"):
            return head + c("funnel_counts 표가 없어요 → Supabase SQL Editor에서 database/schema.sql을 다시 실행하세요", "33")
        return head + c("읽기 실패 (다음 갱신 때 다시)", "90")
    steps, prev = [], None
    for key, label in FUNNEL:
        v = n.get(key, 0)
        steps.append(f"{label} {c(str(v), '1;36')}" + (f" ({pct(v, prev)})" if prev is not None else ""))
        prev = v
    res = n.get("result", 0)
    after = " · ".join(f"{label} {c(str(n.get(k, 0)), '1;36')} ({pct(n.get(k, 0), res)})" for k, label in AFTER)
    extra = " · ".join(f"{label} {n[k]}" for k, label in EXTRA if n.get(k))
    return head + " → ".join(steps) + " → " + after + (c(f"   [{extra}]", "90") if extra else "")


def ts(v) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(KST)
    except ValueError:
        return None


def ago(t: dt.datetime | None, now: dt.datetime) -> str:
    if not t:
        return "?"
    s = (now - t).total_seconds()
    return f"{int(s)}초 전" if s < 60 else f"{int(s // 60)}분 전" if s < 3600 else f"{s / 3600:.1f}시간 전" if s < 86400 \
        else f"{int(s // 86400)}일 전"


def status(r: dict) -> str:
    if r.get("is_qualified"):   # 운영자가 승인한 검토 대기 기록도 '인정'
        return c("인정", "32")
    if r.get("flags"):
        return c("의심", "1;31")
    if r.get("review"):
        return c("검토대기", "1;33")
    return c("미인정", "90")


# --- 화면 -------------------------------------------------------------------
def render(db, limit: int, interval: int, seen: set) -> str:
    now = dt.datetime.now(KST)
    counts, rows = db.counts(), recent(db, 300)
    labels = {k: v["label"] for k, v in scoring.load_templates().items()}
    total = sum(v["total"] for v in counts.values())
    qual = sum(v["qualified"] for v in counts.values())
    times = [(r, ts(r["created_at"])) for r in rows]
    hour = [r for r, t in times if t and (now - t).total_seconds() < 3600]
    today = [r for r, t in times if t and t.date() == now.date()]
    people_today = {r["nickname"].lower() for r in today if not r.get("contributor")}
    last = times[0][1] if times else None
    pub = [r for r in rows[:50] if not r.get("contributor")]
    avg = sum(r["score"] for r in pub) / len(pub) if pub else 0
    sus = sum(1 for r in rows if not r.get("is_qualified") and (r.get("flags") or r.get("review")))

    W = 84
    out = [c(f" 드로잉 벤치마크 모니터 ", "1;97;44") + c(f"  {now:%m-%d %H:%M:%S} · {interval}초마다 · Ctrl+C 종료", "90"), ""]
    # 요약 카드
    out.append(f"  전체 {c(f'{total:,}', '1')}개   인정 {c(f'{qual:,}', '1;32')}개 ({qual / max(total, 1):.0%})"
               f"   최근 1시간 {c(str(len(hour)), '1;36')}건   오늘 참여 {c(str(len(people_today)), '1;36')}명")
    out.append(f"  공개 참여 평균 {c(f'{avg:.1f}', '1')}점 (최근 {len(pub)}건)   마지막 제출 {c(ago(last, now), '1')}"
               + (f"   {c(f'확인 필요 {sus}건 → tools/moderate.py', '1;33')}" if sus else ""))
    out.append(funnel_line(db, now))
    out.append(c("  " + "─" * W, "90"))
    # 도안별
    out.append(c(f"  {fit('도안', 12)}{fit('전체', 6, True)}{fit('인정', 6, True)}  {fit('인정률', 6, True)}  분포", "1"))
    biggest = max([v["total"] for v in counts.values()] or [1])
    for k in sorted(labels, key=lambda t: -counts.get(t, {}).get("total", 0)):
        v = counts.get(k, {"total": 0, "qualified": 0})
        n = round(36 * v["total"] / biggest)
        nq = round(36 * v["qualified"] / biggest)
        bar = c("█" * nq, "32") + c("█" * (n - nq), "90")
        rate = f"{v['qualified'] / v['total']:.0%}" if v["total"] else "-"
        out.append(f"  {fit(labels[k], 12)}{fit(v['total'], 6, True)}{fit(v['qualified'], 6, True)}  {fit(rate, 6, True)}  {bar}")
    out.append(c("  " + "─" * W, "90"))
    # 최신 제출
    out.append(c(f"  {fit('', 2)}{fit('언제', 10)}{fit('닉네임', 21)}{fit('도안', 12)}{fit('점수', 6, True)}  티어"
                 f"{fit('오차', 7, True)}  {fit('입력', 7)}상태", "1"))
    for r, t in times[:limit]:
        key = (r["created_at"], r["nickname"])
        new = seen and key not in seen
        name = r["nickname"] + (" (링크)" if r.get("contributor") else "")
        err = f"{r['avg_error']:.2f}" if r.get("avg_error") is not None else "-"
        line = (f"  {c('★', '1;35') if new else ' '} {fit(ago(t, now), 10)}{fit(name, 20)} {fit(labels.get(r['template_name'], r['template_name']), 12)}"
                f"{fit(f'{r['score']:.1f}', 6, True)}   {c(fit(r['tier'], 3), TIER_COLOR.get(r['tier'], '0'))}"
                f"{fit(err, 7, True)}  {fit(INPUT.get(r.get('input_type'), '?'), 7)}{status(r)}")
        out.append(c(line, "1") if new else line)
    if not rows:
        out.append("  (아직 제출이 없어요)")
    out.append(c(f"\n  ★ = 지난 갱신 이후 새 제출 · 오차 = 평균 오차(px, 작을수록 정확) · 인정 = 70점 이상", "90"))
    seen.clear()
    seen.update((r["created_at"], r["nickname"]) for r in rows[:limit])
    return "\n".join(out)


def main() -> int:
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Supabase 실시간 모니터 (읽기 전용)")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--sqlite")
    ap.add_argument("--once", action="store_true", help="한 번만 출력하고 종료")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    USE_COLOR = not args.no_color and (sys.stdout.isatty() or os.environ.get("FORCE_COLOR"))
    if args.sqlite:
        db = dbmod.SqliteDB(args.sqlite)
    else:
        sec = collect_secrets.load()
        url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
        key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")
        if not (url and key):
            sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 없습니다 (python tools/make_link.py --supabase-url … --supabase-key …)")
        db = dbmod.SupabaseDB(url, key)
    if os.name == "nt":
        os.system("")   # Windows 콘솔 ANSI 이스케이프 켜기
    seen: set = set()
    try:
        while True:
            try:
                screen = render(db, args.limit, args.interval, seen)
            except dbmod.DBError as e:
                screen = f"[연결 오류] {e}  — {args.interval}초 뒤 다시 시도"
            if args.once:
                print(screen)
                return 0
            sys.stdout.write("\x1b[H\x1b[2J\x1b[3J" + screen + "\n")   # 화면·스크롤 지우고 다시 그림
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[monitor] 종료")
        return 0


if __name__ == "__main__":
    sys.exit(main())
