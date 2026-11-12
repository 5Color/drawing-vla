"""수집한 시연 데이터 품질 검사 (형식은 dataset.py, 여기서는 '내용'을 본다).

  - 획마다 도안의 어느 부분(#1, #2 …: index.html TEMPLATES의 path 순서)을 그렸는지
  - 획 순서 패턴이 시연끼리 일관적인지 (예: 1-2-3, 1-23 = 2·3번을 한 획에)
  - 부분별 그리는 방향(시계/반시계, 정/역방향), 부분별 커버리지(빠뜨린 부분),
    정밀도(도안 밖 낙서), 체계적 오프셋, 짧은 실수 획, 중복 파일
  - data/qa/contact_<target>.png 에 전체 시연 한눈에 보기
    (획 색: 1 빨강 → 2 노랑 → 3 초록 → 4 파랑 → 5 보라, 점 = 획 시작점;
     테두리: 빨강 = 내용 문제, 노랑 = 획 순서만 최다 패턴과 다름)

실행:  python train/check_demos.py [--target apple] [--tol 0.02]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "collector" / "index.html"
TEMPLATES_JSON = ROOT / "templates.json"
UNITS = 512.0
STROKE_COLORS = [(255, 90, 90), (255, 200, 60), (90, 220, 120), (80, 170, 255), (210, 120, 255), (255, 255, 255)]


# ---------------------------------------------------------------------------
# 템플릿: index.html의 TEMPLATES를 그대로 파싱 (없으면 templates.json 폴백)
# ---------------------------------------------------------------------------
def load_templates(html_path: Path = INDEX_HTML) -> dict[str, list[str]]:
    if html_path.exists():
        src = html_path.read_text(encoding="utf-8")
        block = src[src.index("const TEMPLATES"):]
        block = block[: block.index("\n};") + 3]
        out = {}
        # paths 배열만 (뒤에 오는 order: [...] 안내 문구는 제외). SVG path 문자열에는 ']'가 없다.
        for m in re.finditer(r"(\w+):\s*\{\s*label:\s*'[^']*',\s*paths:\s*\[(.*?)\]", block, re.S):
            out[m.group(1)] = re.findall(r"'([^']+)'", m.group(2))
        return out
    if TEMPLATES_JSON.exists():
        data = json.loads(TEMPLATES_JSON.read_text(encoding="utf-8"))
        return {k: v["paths"] for k, v in data.items()}
    raise FileNotFoundError(f"Neither {html_path} nor {TEMPLATES_JSON} found")


def _arc(p0, rx, ry, phi, fa, fs, p1, n=48):
    """SVG elliptical arc (endpoint 표기) → 점 목록."""
    (x1, y1), (x2, y2) = p0, p1
    phi = math.radians(phi)
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = c * dx + s * dy, -s * dx + c * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1p**2 / rx**2 + y1p**2 / ry**2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    coef = math.sqrt(max(0.0, num / den)) * (-1 if fa == fs else 1)
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx, cy = c * cxp - s * cyp + (x1 + x2) / 2, s * cxp + c * cyp + (y1 + y2) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    th1 = ang(1, 0, ux, uy)
    dth = ang(ux, uy, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dth > 0:
        dth -= 2 * math.pi
    elif fs and dth < 0:
        dth += 2 * math.pi
    t = th1 + dth * np.linspace(0, 1, n)
    return np.stack([cx + rx * np.cos(t) * c - ry * np.sin(t) * s, cy + rx * np.cos(t) * s + ry * np.sin(t) * c], 1)


def sample_path(d: str, step: float = 0.004) -> np.ndarray:
    """SVG path(M L H V C Q A Z, 절대좌표) → 정규화 좌표 균일 샘플 [n,2]."""
    toks = re.findall(r"[MLHVCQAZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    pts, cur, start, i, cmd = [], (0.0, 0.0), (0.0, 0.0), 0, None
    num = lambda k: [float(v) for v in toks[i:i + k]]
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd == "Z":
                pts.append(np.array([cur, start]))
                cur = start
                continue
        if cmd == "M":
            cur = start = tuple(num(2)); i += 2; pts.append(np.array([cur])); cmd = "L"
        elif cmd == "L":
            p = tuple(num(2)); i += 2; pts.append(np.array([cur, p])); cur = p
        elif cmd == "H":
            p = (num(1)[0], cur[1]); i += 1; pts.append(np.array([cur, p])); cur = p
        elif cmd == "V":
            p = (cur[0], num(1)[0]); i += 1; pts.append(np.array([cur, p])); cur = p
        elif cmd == "C":
            x1, y1, x2, y2, x, y = num(6); i += 6
            t = np.linspace(0, 1, 64)[:, None]
            P = [np.array(cur), np.array([x1, y1]), np.array([x2, y2]), np.array([x, y])]
            pts.append((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t**2 * P[2] + t**3 * P[3])
            cur = (x, y)
        elif cmd == "Q":
            x1, y1, x, y = num(4); i += 4
            t = np.linspace(0, 1, 48)[:, None]
            P = [np.array(cur), np.array([x1, y1]), np.array([x, y])]
            pts.append((1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t**2 * P[2])
            cur = (x, y)
        elif cmd == "A":
            rx, ry, phi, fa, fs, x, y = num(7); i += 7
            pts.append(_arc(cur, rx, ry, phi, int(fa), int(fs), (x, y)))
            cur = (x, y)
        else:
            raise ValueError(f"unsupported path command {cmd!r} in {d!r}")
    poly = np.concatenate(pts) / UNITS
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], 1)


def nearest(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a의 각 점에서 b까지 최단거리와 그 b 인덱스."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    j = d.argmin(1)
    return d[np.arange(len(a)), j], j


def stroke_direction(xy: np.ndarray, part: np.ndarray) -> str:
    """닫힌 도형은 화면 기준 시계/반시계, 열린 선은 path 정방향/역방향."""
    dist, j = nearest(xy, part)
    on = dist < 0.03
    if on.sum() < 5:
        return "?"
    if np.linalg.norm(part[0] - part[-1]) < 0.01:
        seg = xy[on]
        area = 0.5 * np.sum(seg[:-1, 0] * seg[1:, 1] - seg[1:, 0] * seg[:-1, 1])
        return "시계" if area > 0 else "반시계"  # y가 아래로 증가하므로 area>0 = 화면상 시계방향
    jj = j[on]
    return "정방향" if jj[-1] > jj[0] else "역방향"


# ---------------------------------------------------------------------------
# 에피소드 분석
# ---------------------------------------------------------------------------
def analyze(path: Path, parts: list[np.ndarray], tol: float) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    strokes = [np.asarray(s["points"], dtype=np.float64) for s in d["strokes"]]
    traj = np.asarray(d["trajectory"], dtype=np.float64)
    all_parts = np.concatenate(parts)
    part_id = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])
    drawn = np.concatenate([s[:, :2] for s in strokes])

    # 획 → 도안 부분 매칭: 이 획 하나가 그 부분의 40% 이상을 덮으면 그 부분을 그린 것
    # (짧은 꼭지를 몸통에 이어 그려도 '12'로 잡힘). 어느 부분도 40% 못 덮으면 가장 가까운 부분.
    labels, tiny, offpart, dirs = [], 0, 0, []
    for s in strokes:
        xy = s[:, :2]
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        ks = [k for k, p in enumerate(parts) if (nearest(p, xy)[0] < tol).mean() >= 0.4]
        if not ks:
            dist, j = nearest(xy, all_parts)
            near = part_id[j][dist < 3 * tol]
            if len(near) >= 0.5 * len(xy):
                ks = [Counter(near.tolist()).most_common(1)[0][0]]
        if length < 0.02:
            tiny += 1
            labels.append("x")
        elif not ks:
            offpart += 1
            labels.append("?")
        else:
            labels.append("".join(str(k + 1) for k in ks))
            for k in ks:
                dirs.append((k, stroke_direction(xy, parts[k])))

    d_draw, j_draw = nearest(drawn, all_parts)
    part_cov = [float((nearest(p, drawn)[0] < tol).mean()) for p in parts]
    offset = (drawn - all_parts[j_draw]).mean(0) * UNITS  # 512px 기준 평균 어긋남 (dx, dy)
    hz = [(len(s) - 1) * 1000 / (s[-1, 2] - s[0, 2]) for s in strokes if len(s) > 8 and s[-1, 2] > s[0, 2]]
    gaps = [(b[0, 2] - a[-1, 2]) / 1000 for a, b in zip(strokes, strokes[1:])]

    # 구조 검사: trajectory가 strokes와 일치하고 펜-업 bracketing을 지키는지
    pen = traj[:, 2]
    downs = int(((pen[1:] == 1) & (pen[:-1] == 0)).sum())
    structure_ok = pen[0] == 0 and pen[-1] == 0 and downs == len(strokes) \
        and len(traj) == sum(len(s) + 2 for s in strokes)

    return {
        "file": path.name,
        "meta_target": d.get("meta", {}).get("target"),
        "n_strokes": len(strokes),
        "pattern": "-".join(labels),
        "tiny": tiny,
        "offpart": offpart,
        "part_cov": part_cov,
        "precision": float((d_draw < tol).mean()),
        "offset": offset,
        "duration_s": float(strokes[-1][-1, 2] / 1000) if strokes else 0.0,
        "hz": float(np.median(hz)) if hz else float("nan"),
        "max_gap_s": max(gaps) if gaps else 0.0,
        "pointer": ",".join(sorted({s["pointer_type"] for s in d["strokes"]})),
        "board_px": d.get("meta", {}).get("device", {}).get("board_css_px"),
        "structure_ok": bool(structure_ok),
        "dirs": dirs,
        "hash": hashlib.sha1(json.dumps(d["trajectory"]).encode()).hexdigest(),
        "strokes": strokes,
        "target_png": d.get("target_128_png"),
    }


def flags_for(r: dict, canon: str, min_cov: float, min_prec: float) -> list[str]:
    f = []
    if not r["structure_ok"]:
        f.append("구조오류")
    for k, c in enumerate(r["part_cov"]):
        if c < min_cov:
            f.append(f"#{k + 1}누락({c:.0%})")
    if r["precision"] < min_prec:
        f.append(f"도안밖({1 - r['precision']:.0%})")
    if r["tiny"]:
        f.append(f"짧은획{r['tiny']}")
    if r["offpart"]:
        f.append(f"엉뚱한획{r['offpart']}")
    if r["pattern"] != canon:
        f.append("순서다름")
    return f


# ---------------------------------------------------------------------------
# 한눈에 보기 이미지
# ---------------------------------------------------------------------------
def contact_sheet(rows: list[dict], out: Path, tile: int = 200, cols: int = 7) -> None:
    import base64, io
    n = len(rows)
    rws = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tile, rws * tile), (24, 24, 28))
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    for i, r in enumerate(rows):
        x0, y0 = (i % cols) * tile, (i // cols) * tile
        img = Image.new("RGB", (tile, tile), (0, 0, 0))
        if r["target_png"]:
            t = Image.open(io.BytesIO(base64.b64decode(r["target_png"].split(",", 1)[1]))).convert("L")
            t = t.resize((tile, tile), Image.Resampling.BILINEAR).point(lambda v: v * 0.3)
            img.paste(Image.merge("RGB", (t, t, t)))
        dr = ImageDraw.Draw(img)
        for k, s in enumerate(r["strokes"]):
            col = STROKE_COLORS[min(k, len(STROKE_COLORS) - 1)]
            pts = [(float(x) * tile, float(y) * tile) for x, y in s[:, :2]]
            if len(pts) > 1:
                dr.line(pts, fill=col, width=2)
            dr.ellipse((pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3), fill=col)
        content = [f for f in r["flags"] if f != "순서다름"]
        border = (230, 60, 60) if content else (230, 180, 40) if r["flags"] else (60, 60, 70)
        dr.rectangle((0, 0, tile - 1, tile - 1), outline=border, width=1 if border[0] == 60 else 3)
        dr.text((5, 4), f"{r['idx']:02d}  {r['pattern']}", fill=(255, 255, 255), font=font)
        dr.text((5, tile - 18), f"cov {min(r['part_cov']):.0%}  prec {r['precision']:.0%}", fill=(200, 200, 200), font=font)
        sheet.paste(img, (x0, y0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="시연 데이터 품질 검사")
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--target", default=None, help="특정 도안만 (기본: 전부)")
    ap.add_argument("--tol", type=float, default=0.02, help="도안 일치 허용 거리 (정규화, 0.02 ≈ 512px 기준 10px)")
    ap.add_argument("--min-cov", type=float, default=0.8)
    ap.add_argument("--min-prec", type=float, default=0.85)
    ap.add_argument("--out", default=str(ROOT / "data" / "qa"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    templates = load_templates()
    files = sorted(Path(args.root).glob("demo_*.json"))
    by_target: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        by_target[f.name.split("_")[1]].append(f)

    for target, paths in sorted(by_target.items()):
        if args.target and target != args.target:
            continue
        if target not in templates:
            print(f"\n[{target}] index.html에 템플릿이 없어 건너뜀 ({len(paths)}개)")
            continue
        parts = [sample_path(p) for p in templates[target]]
        rows = [analyze(p, parts, args.tol) for p in paths]
        patt = Counter(r["pattern"] for r in rows)
        canon = patt.most_common(1)[0][0]  # 기준 = 가장 많이 쓴 획 순서 (일관성 확인용)
        dup = Counter(r["hash"] for r in rows)
        for i, r in enumerate(rows, 1):
            r["idx"] = i
            r["flags"] = flags_for(r, canon, args.min_cov, args.min_prec)
            if dup[r["hash"]] > 1:
                r["flags"].append("중복")
            if r["meta_target"] != target:
                r["flags"].append(f"meta.target={r['meta_target']}")

        print(f"\n{'=' * 78}\n[{target}] {len(rows)}개  ·  도안 부분 {len(parts)}개 (#1~#{len(parts)} = index.html path 순서)")
        print(f"  획 순서 패턴: " + ", ".join(f"{p} ×{c}" for p, c in patt.most_common()))
        print(f"  (기준 = 최다 패턴 {canon} {patt[canon]}/{len(rows)}. '12' = 1·2번을 한 획에, "
              f"'3-3' = 3번을 두 획으로 나눔, x = 짧은 실수 획, ? = 도안과 무관한 획)")
        cov = np.array([min(r["part_cov"]) for r in rows])
        prec = np.array([r["precision"] for r in rows])
        off = np.array([r["offset"] for r in rows])
        dur = np.array([r["duration_s"] for r in rows])
        hz = np.array([r["hz"] for r in rows])
        dir_by_part: dict[int, Counter] = defaultdict(Counter)
        for r in rows:
            for k, dname in r["dirs"]:
                dir_by_part[k][dname] += 1
        print("  그리는 방향: " + " · ".join(
            f"#{k + 1} " + "/".join(f"{n} {c}" for n, c in dir_by_part[k].most_common()) for k in sorted(dir_by_part)))
        print(f"  최소 부분 커버리지 중앙값 {np.median(cov):.0%} (최저 {cov.min():.0%})  ·  정밀도 중앙값 {np.median(prec):.0%} (최저 {prec.min():.0%})")
        print(f"  평균 어긋남 dx {off[:, 0].mean():+.1f}px  dy {off[:, 1].mean():+.1f}px  (512px 기준, +y = 아래)")
        print(f"  그리기 시간 {np.median(dur):.1f}s (범위 {dur.min():.1f}~{dur.max():.1f}s)  ·  샘플링 {np.nanmedian(hz):.0f}Hz"
              f"  ·  입력 {dict(Counter(r['pointer'] for r in rows))}  ·  캔버스 {dict(Counter(r['board_px'] for r in rows))}px")
        fw = min(60, max(len(r["file"]) for r in rows))  # 배포 수집 파일명은 참여자 이름까지 붙어 길다
        print(f"\n  {'#':>3} {'file':<{fw}} {'획':>2} {'패턴':<12} {'부분 커버리지':<22} {'정밀':>5} {'시간':>5}  문제")
        for r in rows:
            covs = " ".join(f"{c:.0%}" for c in r["part_cov"])
            print(f"  {r['idx']:>3} {r['file'][:fw]:<{fw}} {r['n_strokes']:>2} {r['pattern']:<12} {covs:<22} "
                  f"{r['precision']:>5.0%} {r['duration_s']:>4.1f}s  {' '.join(r['flags']) or '-'}")
        bad = [r for r in rows if any(not f.startswith("순서다름") for f in r["flags"])]
        order_only = [r for r in rows if r["flags"] == ["순서다름"]]
        print(f"\n  요약: 문제 없음 {sum(not r['flags'] for r in rows)} · 순서만 다름 {len(order_only)} · 내용 문제 {len(bad)}")
        out = Path(args.out) / f"contact_{target}.png"
        contact_sheet(rows, out)
        print(f"  한눈에 보기: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
