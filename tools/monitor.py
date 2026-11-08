"""Supabase / SQLite 실시간 모니터 — 대시보드 카드·도안별 달성도 및 품질 게이지·최신 제출 실시간 모니터링.

  python tools/monitor.py                 기본 5초 주기, 최신 15건
  python tools/monitor.py --interval 10 --limit 20
  python tools/monitor.py --no-color      색 끄기 (로그로 남길 때)
  python tools/monitor.py --sqlite outputs/local_bench.sqlite   (로컬 시뮬레이터)
  python tools/monitor.py --once          1회 출력 후 종료

한글 및 전각 문자 폭을 엄밀하게 계산하여 어떤 터미널에서도 열 정렬이 어긋나지 않도록 처리합니다.
접속 정보: SUPABASE_URL / SUPABASE_KEY 환경변수 또는 .collect-secrets.json
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
INPUT_LABELS = {"touch": "터치", "pen": "펜", "mouse": "마우스"}
FUNNEL_STEPS = (("visit", "방문"), ("start", "첫 획"), ("submit", "채점"), ("result", "결과"))
AFTER_ACTIONS = (("retry", "다시"), ("next", "다음"), ("share", "공유"))
EXTRA_EVENTS = (("card", "카드"), ("demo_done", "AI 시연 끝"), ("demo_skip", "시연 건너뜀"), ("hint_off", "안내 끔"))

TARGET_DEMOS_GOAL = 30  # 도안별 목표 수집 시연 수 (CLAUDE.md 기준 30~50회)
USE_COLOR = True


def c(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOR else text


# --- 한글 및 전각 문자 폭 계산 ------------------------------------------------
def char_width(ch: str) -> int:
    """단일 문자의 터미널 표시 폭 계산 (한글=2, 일반문자=1)."""
    if unicodedata.combining(ch):
        return 0
    w = unicodedata.east_asian_width(ch)
    if w in ("W", "F"):
        return 2
    return 1


def str_width(s: str) -> int:
    """문자열의 실제 터미널 표시 폭 계산."""
    return sum(char_width(ch) for ch in str(s))


def pad_to(s: str, width: int, right: bool = False) -> str:
    """문자열의 표시 폭을 계산하여 정확한 공백을 채운다."""
    s = str(s)
    sw = str_width(s)
    if sw >= width:
        return s
    pad = " " * (width - sw)
    return pad + s if right else s + pad


def clip_str(s: str, max_width: int) -> str:
    """최대 너비를 초과하지 않도록 안전하게 자르고 필요시 '..' 추가."""
    s = str(s)
    if str_width(s) <= max_width:
        return s
    target_w = max_width - 2
    cur_w = 0
    buf = []
    for ch in s:
        cw = char_width(ch)
        if cur_w + cw > target_w:
            break
        buf.append(ch)
        cur_w += cw
    return "".join(buf) + ".."


def format_cell(s: str, width: int, right: bool = False) -> str:
    """셀 너비에 맞춰 안전하게 자르고 정확한 폭으로 공백을 패딩."""
    clipped = clip_str(s, width)
    return pad_to(clipped, width, right=right)


# --- 게이지 바 생성 -----------------------------------------------------------
def make_gauge_bar(ratio: float, length: int = 10) -> str:
    """0.0~1.0 비율을 바탕으로 10칸 미니 게이지 바 생성."""
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * length)
    empty = length - filled
    return c("█" * filled, "32") + c("░" * empty, "90")


def make_ratio_bar(qualified: int, total: int, length: int = 10) -> str:
    """인정/미인정 비율 바."""
    if total <= 0:
        return c("░" * length, "90")
    rate = qualified / total
    return make_gauge_bar(rate, length)


# --- 데이터 조회 및 가공 -------------------------------------------------------
def recent_submissions(db, limit: int) -> list[dict]:
    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "drawing_demos", {"select": COLS, "order": "created_at.desc", "limit": limit})
        return res or []
    rows = db._rows(
        "select nickname, contributor, template_name, score, tier, avg_error, input_type, is_qualified, created_at, "
        "json_extract(trajectory_data, '$.meta.flags') flags, json_extract(trajectory_data, '$.meta.review') review "
        "from drawing_demos order by created_at desc limit ?",
        (limit,),
    )
    return [{**r, "flags": None if r["flags"] in (None, "[]") else r["flags"]} for r in rows]


def get_funnel_counts(db, day: str) -> dict[str, int]:
    if isinstance(db, dbmod.SupabaseDB):
        res, _ = db._req("GET", "funnel_counts", {"select": "event,n", "day": f"eq.{day}"})
    else:
        res = db._rows("select event, n from funnel_counts where day=?", (day,))
    return {r["event"]: int(r["n"] or 0) for r in res or []}


def parse_timestamp(v) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(KST)
    except Exception:
        return None


def format_ago(t: dt.datetime | None, now: dt.datetime) -> str:
    if not t:
        return "-"
    sec = max(0, (now - t).total_seconds())
    if sec < 60:
        return f"{int(sec)}초 전"
    if sec < 3600:
        return f"{int(sec // 60)}분 전"
    if sec < 86400:
        return f"{sec / 3600:.1f}시간 전"
    days = int(sec // 86400)
    return f"{days}일 전"


def center_cell(s: str, width: int) -> str:
    """너비 안에서 가운데 정렬."""
    sw = str_width(s)
    if sw >= width:
        return s
    left = (width - sw) // 2
    right = width - sw - left
    return (" " * left) + s + (" " * right)


def format_tier_badge(tier: str) -> str:
    t = (tier or "C").strip().upper()
    colors = {
        "S": "1;93",    # 골드
        "A": "1;36",    # 시안
        "B": "1;32",    # 연초록
        "C": "90",      # 어두운 회색
    }
    return c(f"[{t}]", colors.get(t, "90"))


def format_status_badge(r: dict) -> str:
    if r.get("is_qualified"):
        return c("[인정]", "1;32")
    if r.get("flags"):
        return c("[의심]", "1;97;41")
    if r.get("review"):
        return c("[검토대기]", "1;33")
    return c("[미인정]", "90")


def pct_str(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "-"


# --- 렌더링 -------------------------------------------------------------------
def render_screen(db, limit: int, interval: int, seen: set) -> str:
    now = dt.datetime.now(KST)
    counts = db.counts()
    rows = recent_submissions(db, 300)
    templates = scoring.load_templates()
    template_labels = {k: v["label"] for k, v in templates.items()}

    total_submissions = sum(v["total"] for v in counts.values())
    qualified_submissions = sum(v["qualified"] for v in counts.values())
    overall_qual_rate = (qualified_submissions / total_submissions) if total_submissions else 0.0

    times = [(r, parse_timestamp(r["created_at"])) for r in rows]
    past_1h = [r for r, t in times if t and (now - t).total_seconds() < 3600]
    today_rows = [r for r, t in times if t and t.date() == now.date()]
    today_participants = {r["nickname"].strip().lower() for r in today_rows if not r.get("contributor")}
    last_time = times[0][1] if times else None

    public_recent = [r for r in rows[:50] if not r.get("contributor")]
    avg_score = (sum(r["score"] for r in public_recent) / len(public_recent)) if public_recent else 0.0
    needs_review = sum(1 for r in rows if not r.get("is_qualified") and (r.get("flags") or r.get("review")))

    lines: list[str] = []
    dash_width = 86

    # 1. 헤더 배너
    title = " 🎨 DRAWING VLA BENCHMARK MONITOR "
    status_info = f" {now:%Y-%m-%d %H:%M:%S} KST · {interval}초 갱신 · Ctrl+C 종료 "
    space = " " * max(1, dash_width - str_width(title) - str_width(status_info))
    lines.append(c(title, "1;97;44") + space + c(status_info, "90"))
    lines.append(c("═" * dash_width, "34"))

    # 2. 요약 지표 카드 (2행 그리드)
    # 카드 1행: 전체 수집 현황 | 활동 지표
    tot_str = f"전체 {total_submissions:,}건  (인정 {qualified_submissions:,}건, {overall_qual_rate:.0%})"
    act_str = f"최근 1시간 {len(past_1h)}건  ·  오늘 참여자 {len(today_participants)}명"
    lines.append(f"  {c('📊 총 수집:', '1')} {c(tot_str, '1;32')}" +
                 " " * max(2, 46 - str_width(f"  📊 총 수집: {tot_str}")) +
                 f"{c('⚡ 활성도:', '1')} {c(act_str, '1;36')}")

    # 카드 2행: 품질 지표 | 마지막 제출 / 알림
    avg_str = f"{avg_score:.1f}점 (최근 {len(public_recent)}건)"
    last_str = format_ago(last_time, now)
    rev_str = f"  {c('⚠️ 확인 필요 ' + str(needs_review) + '건 → tools/moderate.py', '1;97;43')}" if needs_review else ""
    lines.append(f"  {c('🎯 공개 평균:', '1')} {c(avg_str, '1;33')}" +
                 " " * max(2, 46 - str_width(f"  🎯 공개 평균: {avg_str}")) +
                 f"{c('⏱️ 최근 제출:', '1')} {c(last_str, '1')}{rev_str}")

    # 3. 퍼널 지표
    try:
        funnel_data = get_funnel_counts(db, now.date().isoformat())
        f_steps, prev_val = [], None
        for key, lbl in FUNNEL_STEPS:
            val = funnel_data.get(key, 0)
            p_text = f" ({pct_str(val, prev_val)})" if prev_val is not None else ""
            f_steps.append(f"{lbl} {c(str(val), '1;36')}{p_text}")
            prev_val = val
        res_val = funnel_data.get("result", 0)
        after_parts = [f"{lbl} {c(str(funnel_data.get(k, 0)), '36')} ({pct_str(funnel_data.get(k, 0), res_val)})"
                       for k, lbl in AFTER_ACTIONS]
        extra_parts = " · ".join(f"{label} {funnel_data[k]}" for k, label in EXTRA_EVENTS if funnel_data.get(k))
        funnel_line_str = " → ".join(f_steps) + " → " + " · ".join(after_parts)
        if extra_parts:
            funnel_line_str += f"   [{extra_parts}]"
        lines.append(f"  {c('오늘 퍼널:', '90')} {funnel_line_str}")
    except Exception:
        pass

    lines.append(c("─" * dash_width, "90"))

    # 4. 도안별 수집 달성도 및 품질 게이지 테이블
    # 컬럼 정의: 도안(12)  수집수(7)  인정(7)  인정률(8)  품질게이지(12)  목표달성도(30개 기준)
    lines.append(c(
        f"  {format_cell('도안', 12)}  "
        f"{format_cell('수집', 6, right=True)}  "
        f"{format_cell('인정', 6, right=True)}  "
        f"{format_cell('인정률', 7, right=True)}  "
        f"{format_cell('품질(인정률)', 12)}  "
        f"{format_cell('목표 달성도 (목표 30개)', 24)}",
        "1;37"
    ))

    sorted_targets = sorted(template_labels.keys(), key=lambda t: -counts.get(t, {}).get("total", 0))
    for t_id in sorted_targets:
        c_info = counts.get(t_id, {"total": 0, "qualified": 0})
        tot = c_info["total"]
        qual = c_info["qualified"]
        rate_val = (qual / tot) if tot > 0 else 0.0
        rate_disp = f"{rate_val:.0%}" if tot > 0 else "-"

        # 품질 게이지 (10칸)
        q_bar = make_ratio_bar(qual, tot, length=10)

        # 목표 달성도 (30개 기준 진척 바)
        goal_ratio = min(1.0, tot / TARGET_DEMOS_GOAL)
        goal_bar_len = 10
        goal_filled = round(goal_ratio * goal_bar_len)
        goal_color = "1;32" if tot >= TARGET_DEMOS_GOAL else "33" if tot >= 15 else "90"
        goal_bar = c("━" * goal_filled, goal_color) + c("─" * (goal_bar_len - goal_filled), "90")
        goal_pct_disp = f"{tot}/{TARGET_DEMOS_GOAL} ({tot / TARGET_DEMOS_GOAL:.0%})"

        t_name = template_labels.get(t_id, t_id)
        lines.append(
            f"  {format_cell(t_name, 12)}  "
            f"{format_cell(str(tot), 6, right=True)}  "
            f"{format_cell(str(qual), 6, right=True)}  "
            f"{format_cell(rate_disp, 7, right=True)}  "
            f"{q_bar}  "
            f"[{goal_bar}] {c(goal_pct_disp, goal_color)}"
        )

    lines.append(c("─" * dash_width, "90"))

    # 5. 최신 제출 내역 테이블
    # 컬럼 정의:
    # [NEW](2)  시간(11)  닉네임(18)  도안(11)  점수(7)  티어(5)  오차(8)  입력(6)  상태(8)
    header_line = (
        f"  {format_cell('', 2)}  "
        f"{format_cell('언제', 11)}  "
        f"{format_cell('닉네임', 18)}  "
        f"{format_cell('도안', 11)}  "
        f"{format_cell('점수', 6, right=True)}  "
        f"{center_cell('티어', 5)}  "
        f"{format_cell('오차', 8, right=True)}  "
        f"{format_cell('입력', 6)}  "
        f"{format_cell('상태', 8)}"
    )
    lines.append(c(header_line, "1;37"))

    for r, t in times[:limit]:
        key = (r.get("created_at"), r.get("nickname"))
        is_new = bool(seen and key not in seen)

        time_str = format_ago(t, now)
        raw_nick = str(r.get("nickname") or "익명")
        if r.get("contributor"):
            nick_display = raw_nick + c("(링크)", "36")
            nick_plain = raw_nick + "(링크)"
        else:
            nick_display = raw_nick
            nick_plain = raw_nick

        t_code = r.get("template_name", "")
        t_label = template_labels.get(t_code, t_code)

        sc = f"{r.get('score', 0.0):.1f}"
        tier_badge = format_tier_badge(r.get("tier", "C"))

        err_val = r.get("avg_error")
        err_str = f"{err_val:.2f}px" if err_val is not None else "-"

        in_type = INPUT_LABELS.get(r.get("input_type"), r.get("input_type") or "-")
        stat_badge = format_status_badge(r)

        star = c("★", "1;35") if is_new else " "
        star_cell = f" {star}"

        # ANSI 태그가 포함된 셀의 패딩 보정
        nick_pad = " " * max(0, 18 - str_width(nick_plain))
        nick_cell = clip_str(nick_plain, 18)
        if r.get("contributor") and "(링크)" in nick_cell:
            nick_cell = nick_cell.replace("(링크)", c("(링크)", "36"))
        nick_cell = nick_cell + nick_pad

        tier_cell = center_cell(tier_badge, 5)

        data_row = (
            f"  {star_cell}  "
            f"{format_cell(time_str, 11)}  "
            f"{nick_cell}  "
            f"{format_cell(t_label, 11)}  "
            f"{format_cell(sc, 6, right=True)}  "
            f"{tier_cell}  "
            f"{format_cell(err_str, 8, right=True)}  "
            f"{format_cell(in_type, 6)}  "
            f"{stat_badge}"
        )
        lines.append(c(data_row, "1") if is_new else data_row)

    if not rows:
        lines.append(c("  (아직 기록된 시연이 없습니다)", "90"))

    lines.append(c("═" * dash_width, "34"))
    lines.append(c("  ★ = 신규 제출  │  품질 게이지: [██████░░░░] 인정률  │  목표: 30개 시연 기준 달성률", "90"))

    seen.clear()
    seen.update((r.get("created_at"), r.get("nickname")) for r in rows[:limit])
    return "\n".join(lines)


def main() -> int:
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Supabase / SQLite 실시간 드로잉 벤치마크 모니터")
    ap.add_argument("--interval", type=int, default=5, help="갱신 주기 (초, 기본 5초)")
    ap.add_argument("--limit", type=int, default=15, help="표시할 최신 제출 수 (기본 15건)")
    ap.add_argument("--sqlite", help="로컬 SQLite DB 파일 경로 (예: outputs/local_bench.sqlite)")
    ap.add_argument("--once", action="store_true", help="1회만 출력하고 즉시 종료")
    ap.add_argument("--no-color", action="store_true", help="ANSI 색상 끄기")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    USE_COLOR = not args.no_color and (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR")))

    if args.sqlite:
        db = dbmod.SqliteDB(args.sqlite)
    else:
        sec = collect_secrets.load()
        url = os.environ.get("SUPABASE_URL") or sec.get("supabase_url")
        key = os.environ.get("SUPABASE_KEY") or sec.get("supabase_key")
        if not (url and key):
            sys.exit("[error] SUPABASE_URL / SUPABASE_KEY가 설정되지 않았습니다.")
        db = dbmod.SupabaseDB(url, key)

    if os.name == "nt":
        os.system("")  # Windows 콘솔 가상 터미널 시퀀스 활성화

    seen: set = set()

    # 1회 출력 모드
    if args.once:
        try:
            screen = render_screen(db, args.limit, args.interval, seen)
            print(screen)
            return 0
        except dbmod.DBError as e:
            sys.exit(f"[DB 오류] {e}")

    # 실시간 모니터링 모드 (플리커 없는 렌더링)
    # 터미널 커서 숨김
    if USE_COLOR:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.write("\x1b[2J")  # 최초 1회 화면 클리어
        sys.stdout.flush()

    try:
        while True:
            try:
                screen = render_screen(db, args.limit, args.interval, seen)
            except dbmod.DBError as e:
                screen = f"[연결 오류] {e}  — {args.interval}초 뒤 재시도..."

            # 줄별로 덮어쓰기 + 라인 끝 지우기 (\x1b[K)로 플리커 방지
            out_buf = ["\x1b[H"]  # 커서 홈으로 이동
            for line in screen.split("\n"):
                out_buf.append(line + "\x1b[K\n")
            sys.stdout.write("".join(out_buf))
            sys.stdout.flush()

            time.sleep(args.interval)
    except KeyboardInterrupt:
        if USE_COLOR:
            sys.stdout.write("\x1b[?25h\n")  # 커서 복원
            sys.stdout.flush()
        print("[monitor] 모니터링이 안전하게 종료되었습니다.")
        return 0
    finally:
        if USE_COLOR:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
