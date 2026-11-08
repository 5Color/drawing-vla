"""드로잉 채점 (표준 라이브러리 전용) — 서버(로컬 server.py, Vercel api)가 궤적만 받아 직접 계산한다.

  coverage  = 도안 위 샘플점 중, 그린 선(펜 다운 점)과 TOL 이내인 비율   (빠뜨린 부분이 있으면 낮아짐)
  precision = 그린 펜 다운 점 중, 도안과 TOL 이내인 비율                  (도안 밖 낙서가 있으면 낮아짐)
  B score = 100·(F1_10px + F1_5px)/2를 소수 첫째 자리까지 버림 → 티어 S/A/B/C
  avg_error = 도안 선 ↔ 그린 선 평균 거리 (양방향 평균, TOL에서 자름, 512px 캔버스 기준 px, 소수 둘째 자리)
              점수가 같을 때 순위를 가르는 보조 지표 (작을수록 위)
  학습 데이터 채택(is_qualified) = F1_10px ≥ 0.70 이고 획 수가 정상 범위(1 ~ 2·부분수+2)

학습 채택 기준은 train/check_demos.py와 같다 (TOL = 0.02 = 512px 캔버스에서 약 10px).
게임 점수에는 절반 허용오차(NARROW_TOL = 0.01, 약 5px)를 함께 쓴다.
도안 좌표는 collector/index.html의 TEMPLATES를 그대로 읽는다 (정의는 한 곳에만).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "collector" / "index.html"
UNITS = 512.0
TOL = 0.02
NARROW_TOL = 0.01
SCORE_VERSION = "b-v1"
SAMPLE_STEP = 0.004
QUALIFY_F1 = 0.70
QUALIFY_SCORE = 70.0
QUALIFY_COVERAGE = 0.75
QUALIFY_PRECISION = 0.75
QUALIFY_PART_COVERAGE = 0.50  # 여러 파트(몸통·꼭지·잎 등)로 이루어진 도안의 파트별 최소 채움률
DRAW_CELL_CAP = 64      # avg_error 선분 격자 칸당 상한 (사람 최대 44)

TIERS = [  # (최소 점수, 티어, 한국어 라벨, OG 이미지용 영어 라벨)
    (96, "S", "신의 손 (AI 압도)", "GOD'S HAND"),     # 운영 실측 최고 기록 26개 중 2개(7.7%), 선택 AI 기록 87.2 초과
    (85, "A", "인간 로봇 (정밀 제어)", "HUMAN ROBOT"),
    (70, "B", "자유로운 예술가", "FREE ARTIST"),
    (0, "C", "화백 지망생 (다시 도전!)", "ASPIRING PAINTER"),
]


def to_score(f1: float) -> float:
    """100·F1을 소수 첫째 자리까지 버림. (부동소수 오차로 98.9999…가 98.9가 되지 않게 아주 작은 값을 더함)"""
    return math.floor(f1 * 1000 + 1e-6) / 10


def tier_for(score: float) -> tuple[str, str, str]:
    for lo, t, ko, en in TIERS:
        if score >= lo:
            return t, ko, en
    return TIERS[-1][1:]


# ---------------------------------------------------------------------------
# 템플릿: index.html TEMPLATES 파싱 + SVG path 샘플링
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_templates(html_path: str = str(INDEX_HTML)) -> dict[str, dict]:
    """{key: {"label": str, "paths": [d, ...]}}"""
    src = Path(html_path).read_text(encoding="utf-8")
    block = src[src.index("const TEMPLATES"):]
    block = block[: block.index("\n};") + 3]
    out = {}
    for m in re.finditer(r"(\w+):\s*\{\s*label:\s*'([^']*)',\s*paths:\s*\[(.*?)\]", block, re.S):
        out[m.group(1)] = {"label": m.group(2), "paths": re.findall(r"'([^']+)'", m.group(3))}
    return out


def _bez(p, n):
    pts = []
    for i in range(n + 1):
        t = i / n
        if len(p) == 4:
            a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
            pts.append((a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0],
                        a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1]))
        else:
            a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
            pts.append((a * p[0][0] + b * p[1][0] + c * p[2][0], a * p[0][1] + b * p[1][1] + c * p[2][1]))
    return pts


def _arc(p0, rx, ry, phi, fa, fs, p1, n=48):
    (x1, y1), (x2, y2) = p0, p1
    phi = math.radians(phi)
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = c * dx + s * dy, -s * dx + c * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    coef = math.sqrt(max(0.0, num / den)) * (-1 if fa == fs else 1)
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx, cy = c * cxp - s * cyp + (x1 + x2) / 2, s * cxp + c * cyp + (y1 + y2) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    th1, dth = ang(1, 0, ux, uy), ang(ux, uy, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dth > 0:
        dth -= 2 * math.pi
    elif fs and dth < 0:
        dth += 2 * math.pi
    out = []
    for i in range(n + 1):
        t = th1 + dth * i / n
        out.append((cx + rx * math.cos(t) * c - ry * math.sin(t) * s, cy + rx * math.cos(t) * s + ry * math.sin(t) * c))
    return out


def sample_path(d: str, step: float = SAMPLE_STEP) -> list[tuple[float, float]]:
    """SVG path(M L H V C Q A Z, 절대좌표) → 정규화 좌표 균일 샘플."""
    toks = re.findall(r"[MLHVCQAZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    poly: list[tuple[float, float]] = []
    cur = start = (0.0, 0.0)
    i, cmd = 0, None
    nums = lambda k: [float(v) for v in toks[i:i + k]]
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd == "Z":
                poly += [cur, start]
                cur = start
                continue
        if cmd == "M":
            cur = start = tuple(nums(2)); i += 2; poly.append(cur); cmd = "L"
        elif cmd in ("L", "H", "V"):
            k = 2 if cmd == "L" else 1
            v = nums(k); i += k
            p = tuple(v) if cmd == "L" else ((v[0], cur[1]) if cmd == "H" else (cur[0], v[0]))
            poly += [cur, p]; cur = p
        elif cmd == "C":
            v = nums(6); i += 6
            poly += _bez([cur, (v[0], v[1]), (v[2], v[3]), (v[4], v[5])], 64); cur = (v[4], v[5])
        elif cmd == "Q":
            v = nums(4); i += 4
            poly += _bez([cur, (v[0], v[1]), (v[2], v[3])], 48); cur = (v[2], v[3])
        elif cmd == "A":
            v = nums(7); i += 7
            poly += _arc(cur, v[0], v[1], v[2], int(v[3]), int(v[4]), (v[5], v[6])); cur = (v[5], v[6])
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
    poly = [(x / UNITS, y / UNITS) for x, y in poly]
    seg = [math.dist(a, b) for a, b in zip(poly, poly[1:])]
    total = sum(seg)
    n = max(1, math.ceil(total / step))
    out, acc, j = [], 0.0, 0
    for k in range(n + 1):
        target = total * k / n
        while j < len(seg) - 1 and acc + seg[j] < target:
            acc += seg[j]; j += 1
        t = 0.0 if seg[j] == 0 else min(1.0, (target - acc) / seg[j])
        a, b = poly[j], poly[j + 1]
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


@lru_cache(maxsize=None)
def template_parts(key: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    tpl = load_templates().get(key)
    if not tpl:
        raise KeyError(key)
    return tuple(tuple(sample_path(d)) for d in tpl["paths"])


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
class _Grid:
    """TOL 크기 격자로 '반경 TOL 안에 점이 있는가'를 빠르게.
    1px(0.002) 격자로 중복 점을 합친다 → 한 칸에 점이 아무리 몰려도 최대 100개 (조작된 낙서로 계산이 폭증하지 않게)."""

    def __init__(self, pts, tol):
        self.tol, self.cells, seen = tol, {}, set()
        for p in pts:
            q = (round(p[0] * 500), round(p[1] * 500))
            if q in seen:
                continue
            seen.add(q)
            self.cells.setdefault((int(p[0] // tol), int(p[1] // tol)), []).append(p)

    def near(self, p) -> bool:
        cx, cy = int(p[0] // self.tol), int(p[1] // self.tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in self.cells.get((cx + dx, cy + dy), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < self.tol ** 2:
                        return True
        return False


def _seg_dist(p, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = vx * vx + vy * vy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L))
    return math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy)


class _SegGrid:
    """선분 격자: 점에서 가장 가까운 선분까지의 거리 (TOL보다 멀면 TOL로 자름).
    cap: 한 칸에 담는 선분 수 상한. 사람 시연은 칸당 최대 44개(호 길이 재샘플 기준), 그보다 빽빽한 칸은
    조작된 낙서뿐이라 앞쪽 cap개만으로 거리를 재도 결과가 사실상 같다 → 계산량 상한."""

    def __init__(self, lines, tol, cap: int | None = None):
        self.tol, self.cells = tol, {}
        for line in lines:
            for a, b in (zip(line, line[1:]) if len(line) > 1 else [(line[0], line[0])]):
                for cx in range(int(min(a[0], b[0]) // tol), int(max(a[0], b[0]) // tol) + 1):
                    for cy in range(int(min(a[1], b[1]) // tol), int(max(a[1], b[1]) // tol) + 1):
                        cell = self.cells.setdefault((cx, cy), [])
                        if cap is None or len(cell) < cap:
                            cell.append((a, b))

    def dist(self, p) -> float:
        cx, cy = int(p[0] // self.tol), int(p[1] // self.tol)
        best = self.tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for a, b in self.cells.get((cx + dx, cy + dy), ()):
                    best = min(best, _seg_dist(p, a, b))
        return best


def pen_down_runs(traj) -> list[list[tuple[float, float]]]:
    """연속한 펜 다운 점 묶음 (획마다 하나)."""
    runs, cur = [], []
    for x, y, p in traj:
        if p >= 0.5:
            cur.append((x, y))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs


def _resample(line, step: float = SAMPLE_STEP) -> list[tuple[float, float]]:
    """호 길이 기준 균일 샘플 (천천히 그린 곳에 점이 몰려 가중치가 커지지 않게)."""
    out, carry = [line[0]], 0.0
    for a, b in zip(line, line[1:]):
        d = math.dist(a, b)
        t = step - carry
        while t <= d:
            out.append((a[0] + (b[0] - a[0]) * t / d, a[1] + (b[1] - a[1]) * t / d))
            t += step
        carry = (carry + d) % step if d else carry
    if len(line) > 1 and out[-1] != tuple(line[-1]):
        out.append(tuple(line[-1]))      # 끝점 포함
    return out


@lru_cache(maxsize=32)
def _tpl_seg_grid(parts: tuple, tol: float) -> _SegGrid:
    return _SegGrid(parts, tol)


@lru_cache(maxsize=32)
def _tpl_grid(tpts: tuple, tol: float) -> _Grid:
    return _Grid(tpts, tol)


def avg_error(parts, runs) -> float:
    """도안 → 그린 선, 그린 선 → 도안 평균 거리의 평균 (512px 기준 px, 소수 둘째 자리)."""
    if not runs:
        return round(TOL * UNITS, 2)
    lines = [_resample(run) for run in runs]   # 입력 기기의 샘플링 속도와 무관하게 0.004 간격 선분
    g_draw, g_tpl = _SegGrid(lines, TOL, cap=DRAW_CELL_CAP), _tpl_seg_grid(tuple(tuple(p) for p in parts), TOL)
    tpts = [p for part in parts for p in part]
    dpts = [p for line in lines for p in line]
    e = (sum(map(g_draw.dist, tpts)) / len(tpts) + sum(map(g_tpl.dist, dpts)) / len(dpts)) / 2
    return round(e * UNITS, 2)


def pen_down_points(traj) -> list[tuple[float, float]]:
    """펜 다운 점 + 연속한 펜 다운 점 사이를 보간 (점 간격이 넓어도 선 전체를 평가)."""
    out, prev = [], None
    for x, y, p in traj:
        if p < 0.5:
            prev = None
            continue
        if prev is not None:
            n = int(math.dist(prev, (x, y)) / (TOL / 2))
            out += [(prev[0] + (x - prev[0]) * k / (n + 1), prev[1] + (y - prev[1]) * k / (n + 1)) for k in range(1, n + 1)]
        out.append((x, y))
        prev = (x, y)
    return out


def count_strokes(traj) -> int:
    n, prev = 0, 0
    for _, _, p in traj:
        if p >= 0.5 and prev < 0.5:
            n += 1
        prev = p
    return n


def _agreement(tpts, drawn, tol: float) -> tuple[float, float, float]:
    if not drawn:
        return 0.0, 0.0, 0.0
    g_draw, g_tpl = _Grid(drawn, tol), _tpl_grid(tuple(tpts), tol)
    cov = sum(g_draw.near(p) for p in tpts) / len(tpts)
    prec = sum(g_tpl.near(p) for p in drawn) / len(drawn)
    f1 = 0.0 if cov + prec == 0 else 2 * cov * prec / (cov + prec)
    return cov, prec, f1


def shape_similarity(tpts: list[tuple[float, float]], drawn: list[tuple[float, float]]) -> tuple[float, float, float]:
    """도안과 그린 선의 중심/스케일을 정규화하여 형태 자체의 일치도를 측정 (자유 드로잉 형태 평가)."""
    if not drawn or not tpts:
        return 0.0, 0.0, 0.0
    dxs, dys = [p[0] for p in drawn], [p[1] for p in drawn]
    txs, tys = [p[0] for p in tpts], [p[1] for p in tpts]

    dw, dh = max(dxs) - min(dxs), max(dys) - min(dys)
    tw, th = max(txs) - min(txs), max(tys) - min(tys)

    if dw < 0.08 or dh < 0.08 or tw < 0.05 or th < 0.05:
        return 0.0, 0.0, 0.0

    aspect_ratio = (dw / dh) / (tw / th) if dh > 0 and th > 0 else 1.0
    if not (0.45 <= aspect_ratio <= 2.2):
        return 0.0, 0.0, 0.0

    dcx, dcy = (min(dxs) + max(dxs)) / 2, (min(dys) + max(dys)) / 2
    tcx, tcy = (min(txs) + max(txs)) / 2, (min(tys) + max(tys)) / 2
    scale = (tw / dw + th / dh) / 2
    aligned = [(tcx + (x - dcx) * scale, tcy + (y - dcy) * scale) for x, y in drawn]
    return _agreement(tpts, aligned, TOL * 1.5)


def score_trajectory(template: str, traj) -> dict:
    parts = template_parts(template)
    tpts = [p for part in parts for p in part]
    drawn = pen_down_points(traj)
    strokes = count_strokes(traj)
    cov, prec, f1_10 = _agreement(tpts, drawn, TOL)
    _, _, f1_5 = _agreement(tpts, drawn, NARROW_TOL)
    score = to_score((f1_10 + f1_5) / 2)
    err = avg_error(parts, pen_down_runs(traj))
    max_strokes = 2 * len(parts) + 2

    part_covs = []
    if drawn:
        g_draw = _Grid(drawn, TOL)
        for part in parts:
            part_covs.append(sum(g_draw.near(p) for p in part) / len(part) if part else 0.0)
    else:
        part_covs = [0.0] * len(parts)
    min_part_cov = min(part_covs) if part_covs else 0.0

    # 형태 유사도 측정 (가이드선과 크기/위치가 살짝 달라도 형태가 예쁘면 구제)
    s_cov, s_prec, s_f1 = shape_similarity(tpts, drawn)
    is_freestyle_qualified = (
        s_f1 >= 0.70
        and cov >= 0.40
        and prec >= 0.40
        and (len(parts) <= 1 or min_part_cov >= 0.30)
        and (1 <= strokes <= max_strokes)
    )

    reason = None
    if not 1 <= strokes <= max_strokes:
        reason = f"획 수 {strokes}개 (정상 1~{max_strokes}개)"
    elif cov < QUALIFY_COVERAGE:
        if not is_freestyle_qualified:
            reason = f"도안 채움률 {int(QUALIFY_COVERAGE * 100)}% 미만"
    elif len(parts) > 1 and min_part_cov < QUALIFY_PART_COVERAGE:
        if not is_freestyle_qualified:
            reason = f"도안 일부 미완성 (부위별 채움률 {int(min_part_cov * 100)}% < 50%)"
    elif prec < QUALIFY_PRECISION:
        if not is_freestyle_qualified:
            reason = f"도안 정밀도 {int(QUALIFY_PRECISION * 100)}% 미만"
    elif score < QUALIFY_SCORE:
        if not is_freestyle_qualified:
            reason = f"{int(QUALIFY_SCORE)}점 미만"
    elif f1_10 < QUALIFY_F1:
        if not is_freestyle_qualified:
            reason = f"10px 일치도 {int(QUALIFY_F1 * 100)}% 미만"

    # 자유 예술가 구제 시 점수/티어 보정
    if is_freestyle_qualified and score < 70.0:
        score = max(score, round(s_f1 * 80.0, 1))

    tier, label, label_en = tier_for(score)
    return {"score": score, "tier": tier, "tier_label": label, "tier_label_en": label_en,
            "score_version": SCORE_VERSION,
            "f1": round(f1_10, 4), "f1_10": round(f1_10, 4), "f1_5": round(f1_5, 4),
            "coverage": round(cov, 4), "part_coverage": [round(c, 4) for c in part_covs], "precision": round(prec, 4), "avg_error": err,
            "shape_f1": round(s_f1, 4),
            "strokes": strokes, "qualified": reason is None, "reason": reason}
