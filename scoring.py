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


HEART_ALT_PATHS = {
    "hand": "M247 212 C254 216 169 169 128 177 C86 186 88 245 109 282 C129 319 158 352 189 381 C220 411 269 404 302 378 C336 351 362 316 383 279 C404 243 425 201 421 159 C417 117 359 110 324 134 C290 159 240 208 247 212 Z",
    "chubby": "M256 176 C236 122 174 106 130 136 C82 170 84 248 136 312 C178 362 226 392 256 412 C286 392 334 362 376 312 C428 248 430 170 382 136 C338 106 276 122 256 176 Z",
}


def transform_variant_pt(x: float, y: float, variant: str) -> tuple[float, float]:
    """도안 스타일 변형(hand: 손그림풍 기울임, chubby: 통통한 비율) 2D 좌표 변환 (정규화 0..1 좌표)."""
    if variant == "hand":
        dx, dy = (x - 0.5) * 0.96, (y - 0.5) * 1.03
        ang = -0.104719755  # -6 deg
        c, s = math.cos(ang), math.sin(ang)
        return (0.5 + dx * c - dy * s, 0.5 + dx * s + dy * c)
    if variant == "chubby":
        return (0.5 + (x - 0.5) * 1.08, 0.5 + (y - 0.5) * 0.92)
    return (x, y)


@lru_cache(maxsize=None)
def _heart_alt_part(variant: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    return (tuple(sample_path(HEART_ALT_PATHS[variant])),)


def template_variant_parts(key: str, variant: str = "base") -> tuple[tuple[tuple[float, float], ...], ...]:
    base = template_parts(key)
    if variant == "base":
        return base
    if key == "heart" and variant in HEART_ALT_PATHS:
        return _heart_alt_part(variant)
    return tuple(tuple(transform_variant_pt(x, y, variant) for x, y in part) for part in base)


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


def _ring(cx: int, cy: int, r: int, bounds: tuple[int, int, int, int]):
    """(cx, cy)에서 체비셰프 거리가 정확히 r인 칸들 (색인 범위 밖은 건너뜀)."""
    bx0, by0, bx1, by1 = bounds
    if r == 0:
        yield cx, cy
        return
    x0, x1 = max(cx - r, bx0), min(cx + r, bx1)
    for y in (cy - r, cy + r):
        if by0 <= y <= by1:
            for x in range(x0, x1 + 1):
                yield x, y
    y0, y1 = max(cy - r + 1, by0), min(cy + r - 1, by1)
    for x in (cx - r, cx + r):
        if bx0 <= x <= bx1:
            for y in range(y0, y1 + 1):
                yield x, y


class _NearIndex:
    """전수 비교(`if d < best`로 끝까지 훑기)와 똑같은 답 — 가장 작은 거리, 같으면 앞 번호 — 을 내는 격자 색인.
    물체(선분·점)를 경계 상자가 걸치는 모든 칸에 넣고, 질의 점의 칸에서 고리를 한 겹씩 넓혀 가며 찾는다.
    r번째 고리까지 보면 남은 물체는 질의 점에서 r·CELL보다 멀다 → 지금 최솟값이 그보다 확실히 작으면 멈춘다.
    도안 근처 점은 주변 몇 칸만 본다. (도안 한가운데처럼 모든 선이 비슷하게 먼 점은 결국 전부 보게 되어 예전과 같은 비용)
    채점 공식은 바꾸지 않는다: 같은 거리 함수로 같은 후보 중 최솟값을 고를 뿐이다."""
    CELL = 0.01
    EPS = 1e-9   # 칸 경계의 부동소수 오차 여유 (멈출지 판단할 때만 쓴다)

    def __init__(self, boxes):
        c, cells = self.CELL, {}
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for cx in range(math.floor(x0 / c), math.floor(x1 / c) + 1):
                for cy in range(math.floor(y0 / c), math.floor(y1 / c) + 1):
                    cells.setdefault((cx, cy), []).append(i)
        self.cells = cells
        self.bounds = ((min(k[0] for k in cells), min(k[1] for k in cells), max(k[0] for k in cells),
                        max(k[1] for k in cells)) if cells else None)

    def nearest(self, p, dist) -> tuple[float, int]:
        """(최소 거리, 물체 번호). dist(i) = 물체 i까지의 거리 (전수 비교에서 쓰던 식 그대로)."""
        best, bi = math.inf, -1
        if not self.cells:
            return best, bi
        c, cells, bounds = self.CELL, self.cells, self.bounds
        cx, cy = math.floor(p[0] / c), math.floor(p[1] / c)
        bx0, by0, bx1, by1 = bounds
        r = max(bx0 - cx, cx - bx1, by0 - cy, cy - by1, 0)      # 이보다 안쪽 고리에는 물체가 없음
        r_all = max(cx - bx0, bx1 - cx, cy - by0, by1 - cy, r)   # 이 고리까지 보면 모든 칸을 본 것
        while True:
            for key in _ring(cx, cy, r, bounds):
                for i in cells.get(key, ()):
                    d = dist(i)
                    if d < best or (d == best and i < bi):
                        best, bi = d, i
            if r >= r_all or best < r * c - self.EPS:
                return best, bi
            r += 1


def _parts_key(parts) -> tuple:
    return tuple(tuple(tuple(p) for p in part) for part in parts)


@lru_cache(maxsize=32)
def _tpl_tangent_index(parts: tuple):
    """도안 선분(길이 1e-4 초과)과 단위 접선 + 선분 색인 — tangent_cosine_similarity 전용."""
    segs = []
    for part in parts:
        for a, b in zip(part, part[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L > 1e-4:
                segs.append((a, b, dx / L, dy / L))
    boxes = [(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])) for a, b, _, _ in segs]
    return segs, _NearIndex(boxes)


@lru_cache(maxsize=32)
def _tpl_point_index(tpts: tuple) -> _NearIndex:
    return _NearIndex([(x, y, x, y) for x, y in tpts])


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


def tangent_cosine_similarity(parts: tuple, runs: list[list[tuple[float, float]]]) -> float:
    """궤적의 진행 방향 벡터와 도안의 접선 벡터 간의 평균 코사인 유사도 (0.0 ~ 1.0).
    사용자가 도안보다 작거나 위치가 달라도, 곡선이 흘러가는 각도/회전이 도안과 닮았는가를 평가한다."""
    if not runs:
        return 0.0
    # 도안 선분 단위 벡터 풀 (+ 가장 가까운 선분을 빠르게 찾는 색인. 답은 전수 비교와 같다)
    tpl_segs, index = _tpl_tangent_index(_parts_key(parts))
    if not tpl_segs:
        return 0.0

    cos_sims = []
    for run in runs:
        if len(run) < 2:
            continue
        for p1, p2 in zip(run, run[1:]):
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            v_len = math.hypot(vx, vy)
            if v_len < 1e-3:
                continue
            vx, vy = vx / v_len, vy / v_len
            # 가장 가까운 도안 선분(거리가 같으면 앞 선분)의 접선 찾기
            mid_p = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            _, i = index.nearest(mid_p, lambda i: _seg_dist(mid_p, tpl_segs[i][0], tpl_segs[i][1]))
            _, _, ux, uy = tpl_segs[i]
            # 양방향 그리기(시계/반시계) 모두 인정: abs(dot)
            cos_sims.append(abs(ux * vx + uy * vy))

    return sum(cos_sims) / len(cos_sims) if cos_sims else 0.0


def shape_similarity(tpts: list[tuple[float, float]], drawn: list[tuple[float, float]], parts: tuple = ()) -> tuple[float, float, float]:
    """도안과 그린 선의 중심/스케일을 정규화하여 형태 자체의 일치도를 측정 (자유 드로잉 형태 평가)."""
    if not drawn or not tpts:
        return 0.0, 0.0, 0.0
    dxs, dys = [p[0] for p in drawn], [p[1] for p in drawn]
    txs, tys = [p[0] for p in tpts], [p[1] for p in tpts]

    dw, dh = max(dxs) - min(dxs), max(dys) - min(dys)
    tw, th = max(txs) - min(txs), max(tys) - min(tys)

    if dw < 0.05 or dh < 0.05 or tw < 0.05 or th < 0.05:
        return 0.0, 0.0, 0.0

    aspect_ratio = (dw / dh) / (tw / th) if dh > 0 and th > 0 else 1.0
    if not (0.35 <= aspect_ratio <= 2.8):
        return 0.0, 0.0, 0.0

    dcx, dcy = (min(dxs) + max(dxs)) / 2, (min(dys) + max(dys)) / 2
    tcx, tcy = (min(txs) + max(txs)) / 2, (min(tys) + max(tys)) / 2
    scale = (tw / dw + th / dh) / 2
    aligned = [(tcx + (x - dcx) * scale, tcy + (y - dcy) * scale) for x, y in drawn]
    return _agreement(tpts, aligned, TOL * 1.8)


def find_spurious_stroke(parts, runs) -> str | None:
    """도안 가이드라인과 무관한 낙서/이탈 획을 감지한다."""
    tpts = [p for part in parts for p in part]
    if not tpts or not runs:
        return None
    index = _tpl_point_index(tuple(tuple(p) for p in tpts))   # 가장 가까운 도안 점 (답은 전수 비교와 같다)
    for idx, run in enumerate(runs):
        if not run:
            continue
        dists = [index.nearest((rx, ry), lambda i: math.hypot(rx - tpts[i][0], ry - tpts[i][1]))[0] * 512
                 for rx, ry in run]
        # 1. 3점 이하 짧은 고립 점: 도안에서 30px 이상 떨어짐
        if len(run) <= 3:
            min_d = min(dists)
            if min_d > 30.0:
                return f"도안과 무관한 점 찍기 ({min_d:.1f}px 이탈)"
            continue
        # 2. 실질적 획: 획 점의 45% 이상이 18px 이상 떨어져 있고, 평균 오차 14px 이상 & 최대 이탈 25px 이상
        avg_d = sum(dists) / len(dists)
        far_ratio = sum(1 for d in dists if d > 18.0) / len(dists)
        max_d = max(dists)
        if far_ratio >= 0.45 and avg_d >= 14.0 and max_d >= 25.0:
            return f"도안과 무관한 낙서/이탈 획 감지 (평균 {avg_d:.1f}px 오차)"
    return None


def _align_runs(tpts: list[tuple[float, float]], drawn: list[tuple[float, float]], runs: list[list[tuple[float, float]]]):
    """그린 궤적(runs)을 도안 바운딩 박스 중심·스케일에 맞게 정렬해 반환한다."""
    if not drawn or not tpts:
        return []
    dxs, dys = [p[0] for p in drawn], [p[1] for p in drawn]
    txs, tys = [p[0] for p in tpts], [p[1] for p in tpts]
    dw, dh = max(dxs) - min(dxs), max(dys) - min(dys)
    tw, th = max(txs) - min(txs), max(tys) - min(tys)
    if dw < 0.05 or dh < 0.05 or tw < 0.05 or th < 0.05:
        return []
    dcx, dcy = (min(dxs) + max(dxs)) / 2, (min(dys) + max(dys)) / 2
    tcx, tcy = (min(txs) + max(txs)) / 2, (min(tys) + max(tys)) / 2
    scale = (tw / dw + th / dh) / 2
    return [[(tcx + (x - dcx) * scale, tcy + (y - dcy) * scale) for x, y in r] for r in runs]


def _score_against_parts(parts: tuple, drawn: list[tuple[float, float]], runs: list[list[tuple[float, float]]], strokes: int) -> dict:
    tpts = [p for part in parts for p in part]
    cov, prec, f1_10 = _agreement(tpts, drawn, TOL)
    _, _, f1_5 = _agreement(tpts, drawn, NARROW_TOL)
    score = to_score((f1_10 + f1_5) / 2)
    err = avg_error(parts, runs)
    max_strokes = 2 * len(parts) + 2

    part_covs = []
    if drawn:
        g_draw = _Grid(drawn, TOL)
        for part in parts:
            part_covs.append(sum(g_draw.near(p) for p in part) / len(part) if part else 0.0)
    else:
        part_covs = [0.0] * len(parts)
    min_part_cov = min(part_covs) if part_covs else 0.0

    # 1. 크기·위치 정규화 기하 형태 유사도 (Procrustes)
    s_cov, s_prec, s_f1 = shape_similarity(tpts, drawn, parts)
    # 2. 진행 방향 벡터 코사인 유사도 (Tangent Cosine)
    tan_cos = tangent_cosine_similarity(parts, runs)

    # 통합 형태 유사도 (0.0 ~ 1.0)
    shape_score = 0.6 * s_f1 + 0.4 * tan_cos

    # 정렬된 궤적 기반 부위별 채움률 및 낙서 검사 (기억/자유 그리기 모드에서 위치·크기가 다른 정상 그림 구제)
    aligned_runs = _align_runs(tpts, drawn, runs)
    aligned_pts = [p for r in aligned_runs for p in r]
    if aligned_pts and len(parts) > 1:
        g_align = _Grid(aligned_pts, TOL * 1.8)
        aligned_part_covs = [sum(g_align.near(p) for p in part) / len(part) if part else 0.0 for part in parts]
        min_aligned_part_cov = min(aligned_part_covs)
    else:
        min_aligned_part_cov = 1.0

    raw_spurious = find_spurious_stroke(parts, runs)
    aligned_spurious = find_spurious_stroke(parts, aligned_runs) if aligned_runs else raw_spurious

    # 가이드 위에서 직접 그린 경우(cov >= 0.70)에는 raw_spurious를 엄격 적용(고양이 얼굴 안 낙서 차단),
    # 기억/자유 그리기처럼 위치·크기가 이동된 경우에는 정렬 후에도 낙서 획이 남는지(aligned_spurious)로 판정
    spurious_reason = raw_spurious if cov >= 0.70 else (aligned_spurious if aligned_spurious else None)

    is_freestyle_qualified = (
        (
            ((s_f1 >= 0.65 or shape_score >= 0.65) and cov >= 0.35 and prec >= 0.35)
            or (shape_score >= 0.78 and s_cov >= 0.68 and s_prec >= 0.68 and cov >= 0.12 and prec >= 0.12)
        )
        and (len(parts) <= 1 or max(min_part_cov, min_aligned_part_cov) >= QUALIFY_PART_COVERAGE)
        and (1 <= strokes <= max_strokes)
        and (spurious_reason is None)
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
    elif spurious_reason:
        reason = spurious_reason
    elif score < QUALIFY_SCORE:
        if not is_freestyle_qualified:
            reason = f"{int(QUALIFY_SCORE)}점 미만"
    elif f1_10 < QUALIFY_F1:
        if not is_freestyle_qualified:
            reason = f"10px 일치도 {int(QUALIFY_F1 * 100)}% 미만"

    if spurious_reason:
        score = min(score, 65.0)  # 무관한 낙서 획이 있으면 C급(65점 이하)으로 감점
    elif is_freestyle_qualified and score < 70.0:
        score = max(score, round(shape_score * 88.0, 1))

    tier, label, label_en = tier_for(score)
    return {"score": score, "tier": tier, "tier_label": label, "tier_label_en": label_en,
            "score_version": SCORE_VERSION,
            "f1": round(f1_10, 4), "f1_10": round(f1_10, 4), "f1_5": round(f1_5, 4),
            "coverage": round(cov, 4), "part_coverage": [round(c, 4) for c in part_covs], "precision": round(prec, 4), "avg_error": err,
            "shape_f1": round(s_f1, 4), "tangent_cos": round(tan_cos, 4), "shape_score": round(shape_score, 4),
            "strokes": strokes, "qualified": reason is None, "reason": reason}


def score_trajectory(template: str, traj) -> dict:
    drawn = pen_down_points(traj)
    runs = pen_down_runs(traj)
    strokes = count_strokes(traj)
    base_res = _score_against_parts(template_variant_parts(template, "base"), drawn, runs, strokes)
    best = base_res
    # 변형 도안(손그림풍 hand / 통통풍 chubby)을 따라 그린 경우, 통과 기준을 만족하거나 기본 도안보다 현저히 높은 일치도일 때 채택
    if template in load_templates():
        for var in ("hand", "chubby"):
            cand = _score_against_parts(template_variant_parts(template, var), drawn, runs, strokes)
            if not best["qualified"] and cand["qualified"]:
                best = cand
            elif cand["qualified"] and cand["f1_10"] > best["f1_10"] + 0.02 and cand["score"] > best["score"]:
                best = cand
            elif (not best["qualified"] or best["f1_10"] < 0.70) and cand["shape_score"] > best["shape_score"] + 0.03 and cand["score"] > best["score"]:
                best = cand
    return best
