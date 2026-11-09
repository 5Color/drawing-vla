"""Supabase / SQLite 실시간 모니터 — 전문적인 TUI 대시보드 & 대화형 스크롤 로그 뷰어.

  python tools/monitor.py                 기본 5초 자동 갱신, ↑/↓ 대화형 스크롤
  python tools/monitor.py --interval 10   10초 갱신
  python tools/monitor.py --no-color      ANSI 색상 끄기
  python tools/monitor.py --sqlite outputs/local_bench.sqlite   (로컬 DB)
  python tools/monitor.py --once          1회 출력 후 종료 (스크립트/파이프용)

단축키:
  ↑ / k           1줄 위로 스크롤 (이전 기록)
  ↓ / j           1줄 아래로 스크롤 (과거 기록)
  PgUp / Space    반 페이지 스크롤
  PgDn            반 페이지 아래로 스크롤
  Home            맨 위(라이브 최신 기록)로 즉시 복귀
  End             맨 아래(가장 오래된 기록)로 이동
  r / R           즉시 새로고침
  q / Q / ESC     종료
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
import unicodedata

import collect_secrets
import db as dbmod
import scoring

KST = dt.timezone(dt.timedelta(hours=9))
COLS = (
    "nickname,contributor,template_name,score,tier,avg_error,input_type,is_qualified,created_at,"
    "flags:trajectory_data->meta->flags,review:trajectory_data->meta->review"
)
INPUT_LABELS = {"touch": "touch", "pen": "pen", "mouse": "mouse"}
FUNNEL_STEPS = (("visit", "방문"), ("start", "첫 획"), ("submit", "채점"), ("result", "결과"))
AFTER_ACTIONS = (("retry", "다시"), ("next", "다음"), ("share", "공유"))
EXTRA_EVENTS = (("card", "카드"), ("demo_done", "AI 시연 끝"), ("demo_skip", "시연 건너뜀"), ("hint_off", "안내 끔"))

TARGET_DEMOS_GOAL = 30  # 도안별 목표 수집 시연 수
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


def center_cell(s: str, width: int) -> str:
    """너비 안에서 가운데 정렬."""
    sw = str_width(s)
    if sw >= width:
        return s
    left = (width - sw) // 2
    right = width - sw - left
    return (" " * left) + s + (" " * right)


# --- 게이지 바 생성 -----------------------------------------------------------
def make_gauge_bar(ratio: float, length: int = 8) -> str:
    """0.0~1.0 비율을 바탕으로 미니 게이지 바 생성."""
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * length)
    empty = length - filled
    return c("█" * filled, "32") + c("░" * empty, "90")


def make_ratio_bar(qualified: int, total: int, length: int = 8) -> str:
    """인정/미인정 비율 바."""
    if total <= 0:
        return c("░" * length, "90")
    rate = qualified / total
    return make_gauge_bar(rate, length)


# --- 데이터 조회 및 포맷팅 ---------------------------------------------------
def recent_submissions(db, limit: int = 200) -> list[dict]:
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
    try:
        if isinstance(db, dbmod.SupabaseDB):
            res, _ = db._req("GET", "funnel_counts", {"select": "event,n", "day": f"eq.{day}"})
        else:
            res = db._rows("select event, n from funnel_counts where day=?", (day,))
        return {r["event"]: int(r["n"] or 0) for r in res or []}
    except Exception:
        return {}


def parse_timestamp(v) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(KST)
    except Exception:
        return None


def format_ago(t: dt.datetime | None, now: dt.datetime) -> str:
    """영문 전문 스타일 상대 시간 (공백/한글 폭 깨짐 0%)."""
    if not t:
        return "-"
    sec = max(0, (now - t).total_seconds())
    if sec < 60:
        return f"{int(sec)}s ago"
    if sec < 3600:
        return f"{int(sec // 60)}m ago"
    if sec < 86400:
        return f"{sec / 3600:.1f}h ago"
    days = int(sec // 86400)
    return f"{days}d ago"


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
        return c("[PASS]", "1;32")
    if r.get("flags"):
        return c("[WARN]", "1;97;41")
    if r.get("review"):
        return c("[REVIEW]", "1;33")
    return c("[FAIL]", "90")


def pct_str(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "-"


# --- 키보드 입력 핸들러 (Cross-platform non-blocking) ------------------------
def read_key() -> str | None:
    if os.name == "nt":
        import msvcrt
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            ch2 = msvcrt.getch()
            mapping = {
                b"H": "UP",
                b"P": "DOWN",
                b"I": "PGUP",
                b"Q": "PGDN",
                b"G": "HOME",
                b"O": "END",
            }
            return mapping.get(ch2)
        if ch in (b"q", b"Q", b"\x1b", b"\x03"):
            return "QUIT"
        if ch in (b"r", b"R"):
            return "REFRESH"
        if ch in (b"k", b"K"):
            return "UP"
        if ch in (b"j", b"J"):
            return "DOWN"
        if ch == b" ":
            return "PGDN"
        return None
    else:
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch in ("q", "Q", "\x03"):
            return "QUIT"
        if ch in ("r", "R"):
            return "REFRESH"
        if ch in ("k", "K"):
            return "UP"
        if ch in ("j", "J"):
            return "DOWN"
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "UP"
            if seq == "[B":
                return "DOWN"
            if seq == "[5~":
                return "PGUP"
            if seq == "[6~":
                return "PGDN"
            if seq == "[H":
                return "HOME"
            if seq == "[F":
                return "END"
        return None


# --- 대시보드 및 로그 렌더러 -------------------------------------------------
class MonitorState:
    def __init__(self, db, interval: int):
        self.db = db
        self.interval = interval
        self.now = dt.datetime.now(KST)
        self.counts = {}
        self.rows = []
        self.templates = scoring.load_templates()
        self.template_labels = {k: v["label"] for k, v in self.templates.items()}
        self.funnel_data = {}
        self.seen_keys = set()
        self.new_keys = set()
        self.is_first_fetch = True
        self.scroll_offset = 0
        self.last_fetch_time = 0.0
        self.error_msg = None

    def refresh_data(self):
        try:
            self.now = dt.datetime.now(KST)
            self.counts = self.db.counts()
            self.rows = recent_submissions(self.db, 250)
            self.funnel_data = get_funnel_counts(self.db, self.now.date().isoformat())
            self.error_msg = None
            self.last_fetch_time = time.time()

            cur_keys = {(r.get("created_at"), r.get("nickname")) for r in self.rows}
            if self.is_first_fetch:
                self.seen_keys = cur_keys
                self.new_keys = set()
                self.is_first_fetch = False
            else:
                self.new_keys = cur_keys - self.seen_keys
                self.seen_keys.update(cur_keys)
        except dbmod.DBError as e:
            self.error_msg = str(e)


def render_tui(state: MonitorState, term_cols: int, term_rows: int) -> list[str]:
    now = state.now
    counts = state.counts
    rows = state.rows
    dash_width = min(78, max(68, term_cols - 1))

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

    # 1. 상단 고정 대시보드 (Pinned Header)
    top_lines: list[str] = []

    # 헤더 배너
    title = " 🎨 DRAWING VLA BENCHMARK MONITOR "
    status_info = f" {now:%Y-%m-%d %H:%M:%S} KST · {state.interval}s poll · ↑/↓ scroll "
    space = " " * max(1, dash_width - str_width(title) - str_width(status_info))
    top_lines.append(c(title, "1;97;44") + space + c(status_info, "90"))
    top_lines.append(c("═" * dash_width, "34"))

    # 카드 1행: 전체 수집 현황 | 활동 지표
    tot_str = f"전체 {total_submissions:,}건  (인정 {qualified_submissions:,}건, {overall_qual_rate:.0%})"
    act_str = f"최근 1시간 {len(past_1h)}건  ·  오늘 참여자 {len(today_participants)}명"
    top_lines.append(
        f"  {c('📊 총 수집:', '1')} {c(tot_str, '1;32')}"
        + " " * max(2, 42 - str_width(f"  📊 총 수집: {tot_str}"))
        + f"{c('⚡ 활성도:', '1')} {c(act_str, '1;36')}"
    )

    # 카드 2행: 품질 지표 | 마지막 제출 / 알림
    avg_str = f"{avg_score:.1f}점 (최근 {len(public_recent)}건)"
    last_str = format_ago(last_time, now)
    rev_str = f"  {c('⚠️ 확인 필요 ' + str(needs_review) + '건 → tools/moderate.py', '1;97;43')}" if needs_review else ""
    top_lines.append(
        f"  {c('🎯 공개 평균:', '1')} {c(avg_str, '1;33')}"
        + " " * max(2, 42 - str_width(f"  🎯 공개 평균: {avg_str}"))
        + f"{c('⏱️ 최근 제출:', '1')} {c(last_str, '1')}{rev_str}"
    )

    # 퍼널
    if state.funnel_data:
        f_steps, prev_val = [], None
        for key, lbl in FUNNEL_STEPS:
            val = state.funnel_data.get(key, 0)
            p_text = f" ({pct_str(val, prev_val)})" if prev_val is not None else ""
            f_steps.append(f"{lbl} {c(str(val), '1;36')}{p_text}")
            prev_val = val
        res_val = state.funnel_data.get("result", 0)
        after_parts = [
            f"{lbl} {c(str(state.funnel_data.get(k, 0)), '36')} ({pct_str(state.funnel_data.get(k, 0), res_val)})"
            for k, lbl in AFTER_ACTIONS
        ]
        extra_parts = " · ".join(f"{label} {state.funnel_data[k]}" for k, label in EXTRA_EVENTS if state.funnel_data.get(k))
        funnel_line_str = " → ".join(f_steps) + " → " + " · ".join(after_parts)
        if extra_parts:
            funnel_line_str += f"   [{extra_parts}]"
        top_lines.append(f"  {c('오늘 퍼널:', '90')} {funnel_line_str}")

    top_lines.append(c("─" * dash_width, "90"))

    # 도안별 수집 현황 게이지 (터미널 높이에 따라 반응형 전환)
    # term_rows < 30 이면 2열 콤팩트 모드 (6줄), term_rows >= 30 이면 1열 상세 게이지 (11줄)
    compact_mode = term_rows < 30
    sorted_targets = sorted(state.template_labels.keys(), key=lambda t: -counts.get(t, {}).get("total", 0))

    if compact_mode:
        top_lines.append(c(f"  {format_cell('도안별 수집 및 인정률 (목표 30개)', dash_width - 4)}", "1;37"))
        half = (len(sorted_targets) + 1) // 2
        left_targets = sorted_targets[:half]
        right_targets = sorted_targets[half:]
        col_w = (dash_width - 5) // 2

        for i in range(max(len(left_targets), len(right_targets))):
            t_left = left_targets[i] if i < len(left_targets) else None
            t_right = right_targets[i] if i < len(right_targets) else None

            def make_compact_col(t_id: str | None) -> str:
                if not t_id:
                    return " " * col_w
                c_info = counts.get(t_id, {"total": 0, "qualified": 0})
                tot = c_info["total"]
                qual = c_info["qualified"]
                rate_val = (qual / tot) if tot > 0 else 0.0
                rate_str = f"{rate_val:.0%}" if tot > 0 else "-"
                t_name = state.template_labels.get(t_id, t_id)
                g_col = "1;32" if tot >= TARGET_DEMOS_GOAL else "33" if tot >= 15 else "90"
                g_ratio = min(1.0, tot / TARGET_DEMOS_GOAL)
                g_filled = round(g_ratio * 4)
                g_bar = c("━" * g_filled, g_col) + c("─" * (4 - g_filled), "90")
                name_cell = format_cell(t_name, 11)
                tot_cell = format_cell(f"{tot}건", 5, right=True)
                rate_cell = format_cell(rate_str, 5, right=True)
                content = f"  {name_cell} {tot_cell} {rate_cell} [{g_bar}]"
                return content

            l_str = pad_to(make_compact_col(t_left), col_w)
            r_str = pad_to(make_compact_col(t_right), col_w)
            top_lines.append(f"{l_str} {c('│', '90')} {r_str}")
    else:
        top_lines.append(
            c(
                f"  {format_cell('도안', 11)} "
                f"{format_cell('수집', 5, right=True)} "
                f"{format_cell('인정', 5, right=True)} "
                f"{format_cell('인정률', 6, right=True)} "
                f"{format_cell('품질', 8)} "
                f"{format_cell('목표 달성도 (30개)', 20)}",
                "1;37",
            )
        )
        for t_id in sorted_targets:
            c_info = counts.get(t_id, {"total": 0, "qualified": 0})
            tot = c_info["total"]
            qual = c_info["qualified"]
            rate_val = (qual / tot) if tot > 0 else 0.0
            rate_disp = f"{rate_val:.0%}" if tot > 0 else "-"

            q_bar = make_ratio_bar(qual, tot, length=8)

            goal_ratio = min(1.0, tot / TARGET_DEMOS_GOAL)
            goal_bar_len = 8
            goal_filled = round(goal_ratio * goal_bar_len)
            goal_color = "1;32" if tot >= TARGET_DEMOS_GOAL else "33" if tot >= 15 else "90"
            goal_bar = c("━" * goal_filled, goal_color) + c("─" * (goal_bar_len - goal_filled), "90")
            goal_pct_disp = f"{tot}/{TARGET_DEMOS_GOAL} ({tot / TARGET_DEMOS_GOAL:.0%})"

            t_name = state.template_labels.get(t_id, t_id)
            top_lines.append(
                f"  {format_cell(t_name, 11)} "
                f"{format_cell(str(tot), 5, right=True)} "
                f"{format_cell(str(qual), 5, right=True)} "
                f"{format_cell(rate_disp, 6, right=True)} "
                f"{q_bar} "
                f"[{goal_bar}] {c(goal_pct_disp, goal_color)}"
            )

    top_lines.append(c("─" * dash_width, "90"))

    # 2. 하단 스크롤 가능 로그 뷰포트 (Scrollable Log Viewport)
    fixed_height = len(top_lines) + 2  # top + header + status bar
    viewport_height = max(1, term_rows - fixed_height)

    total_logs = len(rows)
    # 스크롤 오프셋 경계 보정
    max_offset = max(0, total_logs - viewport_height)
    state.scroll_offset = max(0, min(state.scroll_offset, max_offset))

    # 전문적인 영문 TUI 테이블 헤더
    header_line = (
        f" {format_cell('', 1)} "
        f"{format_cell('TIME', 8)} "
        f"{format_cell('USER', 11)} "
        f"{format_cell('TARGET', 11)} "
        f"{format_cell('SCORE', 5, right=True)} "
        f"{center_cell('TIER', 5)} "
        f"{format_cell('ERROR', 8, right=True)} "
        f"{format_cell('INPUT', 6)} "
        f"{format_cell('STATUS', 6)}"
    )
    log_lines = [c(header_line, "1;37")]

    visible_slice = times[state.scroll_offset : state.scroll_offset + viewport_height]
    for r, t in visible_slice:
        key = (r.get("created_at"), r.get("nickname"))
        is_new = bool(key in state.new_keys)

        time_str = format_ago(t, now)
        raw_nick = str(r.get("nickname") or "익명")
        if r.get("contributor"):
            nick_display = raw_nick + c("(링크)", "36")
            nick_plain = raw_nick + "(링크)"
        else:
            nick_display = raw_nick
            nick_plain = raw_nick

        t_code = r.get("template_name", "")
        t_label = state.template_labels.get(t_code, t_code)

        sc = f"{r.get('score', 0.0):.1f}"
        tier_badge = format_tier_badge(r.get("tier", "C"))

        err_val = r.get("avg_error")
        err_str = f"{err_val:.2f}px" if err_val is not None else "-"

        in_type = INPUT_LABELS.get(r.get("input_type"), r.get("input_type") or "-")
        stat_badge = format_status_badge(r)

        star = c("★", "1;35") if is_new else " "
        star_cell = f"{star}"

        nick_pad = " " * max(0, 11 - str_width(nick_plain))
        nick_cell = clip_str(nick_plain, 11)
        if r.get("contributor") and "(링크)" in nick_cell:
            nick_cell = nick_cell.replace("(링크)", c("(링크)", "36"))
        nick_cell = nick_cell + nick_pad

        tier_cell = center_cell(tier_badge, 5)

        data_row = (
            f" {star_cell} "
            f"{format_cell(time_str, 8)} "
            f"{nick_cell} "
            f"{format_cell(t_label, 11)} "
            f"{format_cell(sc, 5, right=True)} "
            f"{tier_cell} "
            f"{format_cell(err_str, 8, right=True)} "
            f"{format_cell(in_type, 6)} "
            f"{stat_badge}"
        )
        log_lines.append(c(data_row, "1") if is_new else data_row)

    # 빈 줄 채우기 (높이 유지)
    while len(log_lines) < viewport_height + 1:
        log_lines.append(" ")

    # 3. 하단 상태 바 (Status Bar)
    pos_start = state.scroll_offset + 1 if total_logs > 0 else 0
    pos_end = min(state.scroll_offset + viewport_height, total_logs)

    if state.scroll_offset == 0:
        bar_text = f" [LIVE] Log {pos_start}-{pos_end}/{total_logs} │ ▲/▼: Scroll PgUp/Dn Home: Top R: Refresh Q: Quit "
        bar_line = c(pad_to(bar_text, dash_width), "1;37;44")
    else:
        bar_text = f" [▲ PAST LOGS] {pos_start}-{pos_end}/{total_logs} (Home: Jump to Live) │ ▲/▼: Scroll Q: Quit "
        bar_line = c(pad_to(bar_text, dash_width), "1;97;43")

    all_lines = top_lines + log_lines + [bar_line]
    return all_lines[:term_rows]


def main() -> int:
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Supabase / SQLite 실시간 대화형 TUI 모니터")
    ap.add_argument("--interval", type=int, default=5, help="자동 갱신 주기 (초, 기본 5초)")
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
        os.system("")  # Windows 가상 터미널 시퀀스 활성화

    state = MonitorState(db, args.interval)
    state.refresh_data()

    # 1회 출력 모드
    if args.once:
        size = shutil.get_terminal_size((80, 32))
        lines = render_tui(state, size.columns, size.lines)
        print("\n".join(lines))
        return 0

    # TUI 인터랙티브 모드: Alternate Screen Buffer 진입, 커서 숨김, Auto-wrap 끔
    if USE_COLOR:
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J\x1b[H")
        sys.stdout.flush()

    need_redraw = True

    try:
        while True:
            cur_time = time.time()
            size = shutil.get_terminal_size((80, 24))

            # 1. 주기적 백그라운드 데이터 갱신 (5초)
            if cur_time - state.last_fetch_time >= state.interval:
                state.refresh_data()
                need_redraw = True

            # 2. 키 입력 감지 (즉각적인 스크롤 반응)
            key = read_key()
            if key == "QUIT":
                break
            elif key == "REFRESH":
                state.refresh_data()
                need_redraw = True
            elif key == "UP":
                if state.scroll_offset > 0:
                    state.scroll_offset -= 1
                    need_redraw = True
            elif key == "DOWN":
                total_logs = len(state.rows)
                compact_m = size.lines < 30
                fixed_h = 16 if compact_m else (len(state.template_labels) + 9)
                vp_h = max(1, size.lines - fixed_h)
                if state.scroll_offset < max(0, total_logs - vp_h):
                    state.scroll_offset += 1
                    need_redraw = True
            elif key == "PGUP":
                state.scroll_offset = max(0, state.scroll_offset - 8)
                need_redraw = True
            elif key == "PGDN":
                total_logs = len(state.rows)
                compact_m = size.lines < 30
                fixed_h = 16 if compact_m else (len(state.template_labels) + 9)
                vp_h = max(1, size.lines - fixed_h)
                state.scroll_offset = min(max(0, total_logs - vp_h), state.scroll_offset + 8)
                need_redraw = True
            elif key == "HOME":
                if state.scroll_offset != 0:
                    state.scroll_offset = 0
                    need_redraw = True
            elif key == "END":
                total_logs = len(state.rows)
                compact_m = size.lines < 30
                fixed_h = 16 if compact_m else (len(state.template_labels) + 9)
                vp_h = max(1, size.lines - fixed_h)
                state.scroll_offset = max(0, total_logs - vp_h)
                need_redraw = True

            # 3. 화면 렌더링 (플리커 없음, 스크롤 누적 방지)
            if need_redraw:
                lines = render_tui(state, size.columns, size.lines)
                lines = lines[: size.lines]
                # 각 줄 끝 지우기(\x1b[K)를 붙이고 \n으로 결합하되, 맨 마지막 줄 뒤에는 \n을 찍지 않음!
                body = "\n".join(l + "\x1b[K" for l in lines)
                out_buf = ["\x1b[H", body, "\x1b[J"]
                sys.stdout.write("".join(out_buf))
                sys.stdout.flush()
                need_redraw = False

            time.sleep(0.03)  # CPU 0% 유지 및 30fps 반응성
    except KeyboardInterrupt:
        pass
    finally:
        if USE_COLOR:
            # Auto-wrap 복원, 커서 복원, Alternate Screen Buffer 복귀
            sys.stdout.write("\x1b[?7h\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
        print("[monitor] 모니터링이 안전하게 종료되었습니다.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
