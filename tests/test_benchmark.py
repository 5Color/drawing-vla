"""드로잉 벤치마크 테스트 (표준 라이브러리 unittest + Pillow).

  python -m unittest tests.test_benchmark -v

Supabase는 실제로 호출하지 않는다: API 로직은 SQLite 백엔드로, Supabase REST 요청은 가짜 urlopen으로 검증.
"""
from __future__ import annotations

import datetime as dt
import glob
import gzip
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from email.message import Message
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SECRET = "test-secret"

import scoring  # noqa: E402
import submission  # noqa: E402
from collect_common import sign_name  # noqa: E402


def _from_db(path: str) -> bool:
    """pull_dataset.py가 DB에서 받아 온 시연 (meta.db_id / 배포 서버 궤적) — 로컬 원본과 구분."""
    meta = json.loads(Path(path).read_text(encoding="utf-8")).get("meta", {})
    return bool(meta.get("db_id")) or meta.get("trajectory_source") == "server"


# 로컬 수집 원본만 (data/raw에는 DB에서 받아 온 시연도 섞여 있다)
DEMO_FILES = [f for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "demo_apple_*.json"))) if not _from_db(f)]
_uniq = iter(range(1, 10 ** 6))
T0 = dt.datetime(2026, 11, 8, 3, 0, 30, tzinfo=dt.timezone.utc)   # 시각 고정용 (분 경계에 걸려 한도 테스트가 흔들리지 않게)


def device(nick) -> str:
    """닉네임마다 고정된 기기 비밀값 (X-Owner: 소문자 hex 32자)."""
    return hashlib.md5(f"device:{nick}".encode()).hexdigest()


def frozen(at: dt.datetime = T0):
    """db의 현재 시각 고정 (시도 횟수 버킷·created_at)."""
    return mock.patch("db._now", return_value=at)


def demo_payload(i: int = 0, unique: bool = True) -> dict:
    """실제 시연. unique=True면 첫 점 위치를 1px 안팎 옮겨 매번 다른 그림으로 (같은 모양 재제출 금지 때문)."""
    p = json.loads(Path(DEMO_FILES[i]).read_text(encoding="utf-8"))
    if unique:
        k = next(_uniq)
        pt = p["strokes"][0]["points"][0]
        pt[0] = min(1, pt[0] + (k % 25) * 0.0011)
        pt[1] = min(1, pt[1] + (k // 25 % 25) * 0.0011)
    return p


def old_ticket(age_s: float = 3600) -> str:
    import time
    return submission.issue_ticket(SECRET, time.time() - age_s)


def as_strokes(pts, dt_ms: float = 10.0) -> list[dict]:
    """[(x, y), ...] → 한 획짜리 strokes (dt_ms 간격 시간)."""
    return [{"pointer_type": "touch", "points": [[x, y, round(k * dt_ms, 2), 0] for k, (x, y) in enumerate(pts)]}]


def headers(**kw) -> Message:
    m = Message()
    for k, v in kw.items():
        m[k.replace("_", "-")] = v
    return m


def link(name: str) -> Message:
    return headers(x_collect_user=name, x_collect_token=sign_name(SECRET, name))


@unittest.skipUnless(DEMO_FILES, "data/raw/demo_apple_*.json 필요")
class TestScoring(unittest.TestCase):
    def test_real_demos_keep_high_loose_agreement_and_qualification(self):
        sc = [scoring.score_trajectory("apple", demo_payload(i)["trajectory"]) for i in range(5)]
        self.assertTrue(all(s["f1_10"] >= .85 and s["qualified"] for s in sc), sc)

    def test_scribble_and_wrong_template_fail(self):
        random.seed(0)
        scrib = [[random.random(), random.random(), 1] for _ in range(300)]
        self.assertEqual(scoring.score_trajectory("apple", scrib)["tier"], "C")
        wrong = scoring.score_trajectory("cat", demo_payload()["trajectory"])
        self.assertFalse(wrong["qualified"])

    def test_tiers(self):
        self.assertEqual([scoring.tier_for(s)[0] for s in (100, 96, 95.9, 85, 84.9, 70, 69.9, 0)], list("SSAABBCC"))

    def test_score_is_truncated_to_one_decimal(self):
        # 반올림하지 않고 소수 첫째 자리까지 버림 → 실력보다 높게 표시되지 않음
        self.assertEqual([scoring.to_score(f) for f in (0.98962, 0.98999999, 0.99, 1.0, 0.6999)], [98.9, 98.9, 99.0, 100.0, 69.9])
        s = scoring.score_trajectory("apple", demo_payload()["trajectory"])
        self.assertAlmostEqual(s["score"], 50 * (s["f1_10"] + s["f1_5"]), delta=.11)
        self.assertEqual(s["score"], round(s["score"], 1))

    def test_avg_error(self):
        # 도안 경로를 그대로 따라 그리면 오차 0, 사람 시연은 몇 px, 살짝 밀면 커짐
        parts = scoring.template_parts("apple")
        exact = []
        for part in parts:
            exact += [[part[0][0], part[0][1], 0]] + [[x, y, 1] for x, y in part] + [[part[-1][0], part[-1][1], 0]]
        self.assertEqual(scoring.score_trajectory("apple", exact)["avg_error"], 0.0)
        shifted = [[x + 0.004, y, pd] for x, y, pd in exact]              # 가로로 2.05px 밀기 → 선에 수직인 거리는 0~2.05px
        self.assertTrue(0.5 < scoring.score_trajectory("apple", shifted)["avg_error"] <= 2.05)
        human = scoring.score_trajectory("apple", demo_payload()["trajectory"])["avg_error"]
        self.assertTrue(0 < human < scoring.TOL * scoring.UNITS, human)

    def test_real_demos_pass_human_checks_and_server_trajectory_matches(self):
        for f in DEMO_FILES:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            c = submission.clean_payload(d)
            traj = submission.build_trajectory(c["strokes"])
            if d["meta"].get("smoothing") == "ma3x2":
                self.assertEqual(traj, d["trajectory"], f)                               # JS buildTrajectory와 같은 결과
            sc = scoring.score_trajectory("apple", traj)
            self.assertEqual(submission.human_checks(c["strokes"], sc["avg_error"])[0], [], f)

    def test_score_does_not_depend_on_stroke_order_or_direction(self):
        # 획 순서 안내를 없앴으므로: 획 순서를 뒤집고 각 획 방향을 거꾸로 그려도 점수·오차가 같아야 함
        import submission as sub_
        d = demo_payload(unique=False)
        fwd = sub_.build_trajectory(d["strokes"])
        rev = sub_.build_trajectory([{**s_, "points": s_["points"][::-1]} for s_ in d["strokes"][::-1]])
        a, b = scoring.score_trajectory("apple", fwd), scoring.score_trajectory("apple", rev)
        self.assertEqual((a["score"], a["coverage"], a["precision"]), (b["score"], b["coverage"], b["precision"]))
        self.assertAlmostEqual(a["avg_error"], b["avg_error"], delta=0.02)

    def test_too_many_strokes_not_qualified(self):
        traj = demo_payload()["trajectory"]
        # 같은 그림을 획마다 잘게 끊으면 (펜 업을 많이 넣으면) 획 수 초과
        chopped = []
        for i, (x, y, p) in enumerate(traj):
            chopped.append([x, y, 0 if (p and i % 20 == 0) else p])
        s = scoring.score_trajectory("apple", chopped)
        self.assertFalse(s["qualified"])
        self.assertIn("획 수", s["reason"])


@unittest.skipUnless(DEMO_FILES, "data/raw/demo_apple_*.json 필요")
class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"COLLECT_LOCAL_DB": str(Path(self.tmp.name) / "t.sqlite"),
                                                "COLLECT_SECRET": SECRET}, clear=False)
        self.env.start()
        os.environ.pop("SUPABASE_URL", None)
        os.environ.pop("SUPABASE_KEY", None)
        import api_core
        api_core._db = None
        api_core._cache.clear()
        api_core._ev.update(minute=-1, hits={})
        self.api = api_core
        self.ip = iter(f"10.0.0.{i}" for i in range(1, 250))

    def tearDown(self):
        self.api._db = None
        self.env.stop()
        self.tmp.cleanup()

    def save(self, payload, hdr=None, ip=None, ticket="auto", owner="auto"):
        hdr = hdr or headers()
        if ticket == "auto":
            ticket = old_ticket()
        if ticket and "x-draw-ticket" not in hdr:
            hdr["X-Draw-Ticket"] = ticket
        if owner == "auto":   # 공개 참여는 기기 비밀값(X-Owner) 필수 → 닉네임마다 같은 기기로
            owner = device((payload.get("meta") or {}).get("nickname"))
        if owner and "x-owner" not in hdr:
            hdr["X-Owner"] = owner
        st, body, _ = self.api.save(hdr, json.dumps(payload).encode(), ip or next(self.ip))
        return st, json.loads(body)

    def rows(self) -> dict:
        """닉네임 → 저장된 trajectory_data (SQLite)."""
        return {r["nickname"]: json.loads(r["trajectory_data"]) for r in
                self.api.get_db()._rows("select nickname, trajectory_data from drawing_demos order by created_at")}

    def public(self, nick="선따기장인", i=0, **meta):
        p = demo_payload(i)
        p["meta"].update(nickname=nick, notice="v1", **meta)
        return p

    def test_stats_public_and_bad_link(self):
        st, b, _ = self.api.stats(headers())
        self.assertEqual((st, json.loads(b)["contributor"]), (200, None))
        st, _, _ = self.api.stats(headers(x_collect_user="minsu", x_collect_token="forged"))
        self.assertEqual(st, 401)

    def test_public_save_requires_valid_nickname_only(self):
        p = demo_payload()
        st, b = self.save(p)                                   # 닉네임 없음
        self.assertEqual((st, b.get("need")), (400, "nickname"))
        p = demo_payload()
        p["meta"]["nickname"] = "체크박스없이"                     # 동의 체크 없이 닉네임만으로 통과
        self.assertEqual(self.save(p)[0], 200)
        for bad in ("", "a" * 14, "<script>", "씨발놈"):
            st, b = self.save(self.public(bad))
            self.assertEqual(st, 400, bad)

    def test_public_save_scores_and_ranks(self):
        st, b = self.save(self.public("선따기장인"))
        self.assertEqual(st, 200, b)
        self.assertTrue(b["qualified"])
        self.assertEqual(b["rank"], 1)
        self.assertIn("/s?score=", b["share_url"])
        self.assertIsNone(b["contributor"])

    def test_input_type_recorded_and_sanitized(self):
        p = self.public("펜쟁이", input_type="mouse")          # meta 값은 무시, 실제 입력(strokes)으로 판단
        for s_ in p["strokes"]:
            s_["pointer_type"] = "pen"
        st, b = self.save(p)
        self.assertEqual((st, b["input_type"]), (200, "pen"))
        p = self.public("이상한입력", 1)
        for s_ in p["strokes"]:
            s_["pointer_type"] = "<img>"
        self.save(p)
        top = json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]
        kinds = {r["nickname"]: r["input_type"] for r in top}
        self.assertEqual(kinds["펜쟁이"], "pen")
        self.assertEqual(kinds["이상한입력"], "unknown")

    def test_client_score_is_ignored(self):
        p = self.public("hacker")
        p["meta"]["score"] = {"score": 100, "tier": "S"}
        p["strokes"] = as_strokes([(0.9, 0.9), (0.92, 0.92), (0.95, 0.95)], 200)
        st, b = self.save(p)
        self.assertEqual((st, b["tier"], b["qualified"]), (200, "C", False))

    def test_link_save_uses_link_name(self):
        st, b = self.save(demo_payload(), link("minsu"))
        self.assertEqual((st, b["nickname"], b["mine"]), (200, "minsu", 1))
        st, b = self.save(demo_payload(), headers(x_collect_user="minsu", x_collect_token="forged"))
        self.assertEqual(st, 401)

    def test_unknown_template_rejected(self):
        p = self.public()
        p["meta"]["target"] = "../../etc"
        self.assertEqual(self.save(p)[0], 400)

    def test_limit_values(self):
        # 교실 하나가 같은 와이파이(IP)로 참여해도 되게 IP 한도를 올리고, 닉네임별 한도를 따로 둔다
        self.assertEqual(self.api.RATE_LIMITS, ((60, 20), (3600, 200)))
        self.assertEqual(self.api.NICK_LIMITS, ((60, 6), (3600, 60)))
        self.assertEqual((self.api.TICKET_TRIES, self.api.GLOBAL_LIMIT), (5, (60, 600)))

    def test_rate_limit_per_ip(self):
        n = self.api.RATE_LIMITS[0][1]
        with frozen():
            res = [self.save(self.public(f"u{i}"), ip="1.2.3.4") for i in range(n + 2)]
            self.assertEqual([st for st, _ in res], [200] * n + [429] * 2)
            self.assertIn("그림은 그대로", res[-1][1]["error"])                              # 그림이 지워지지 않는다고 안내
            self.assertEqual(self.save(self.public("other"), ip="5.6.7.8")[0], 200)
            # 프록시 헤더(x-forwarded-for)가 있으면 그 IP 기준
            hdr = headers(x_forwarded_for="1.2.3.4, 10.0.0.1")
            self.assertEqual(self.save(self.public("via-proxy"), hdr, ip="9.9.9.9")[0], 429)

    def test_leaderboard_best_per_nickname(self):
        for i, nick in enumerate(["alice", "bob", "alice", "carol", "dave", "erin", "frank"]):
            self.save(self.public(nick, i))
        p = self.public("scribbler")
        p["strokes"] = as_strokes([(0.1 + 0.008 * k, 0.1 + 0.008 * k) for k in range(100)], 20)
        self.save(p)
        st, b, _ = self.api.leaderboard({"template": ["apple"]})
        top = json.loads(b)["top"]
        names = [r["nickname"] for r in top]
        self.assertEqual(st, 200)
        self.assertEqual(len(top), 5)                              # 공동 순위 없음 → 항상 5명 이하
        self.assertEqual([r["rank"] for r in top], [1, 2, 3, 4, 5])
        self.assertEqual(len(set(n.lower() for n in names)), len(names))   # 닉네임당 1개
        self.assertNotIn("scribbler", names)
        self.assertEqual([r["score"] for r in top], sorted([r["score"] for r in top], reverse=True))
        self.assertEqual(self.api.leaderboard({"template": ["nope"]})[0], 400)

    def test_trajectory_endpoint(self):
        p = self.public("장인")
        st, b = self.save(p)
        self.assertEqual(st, 200)
        top = json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]
        demo_id = top[0]["id"]
        self.assertTrue(demo_id)

        st, body, _ = self.api.trajectory({"id": [demo_id]})
        self.assertEqual(st, 200)
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["nickname"], "장인")
        self.assertEqual(len(data["strokes"]), len(p["strokes"]))

        self.assertEqual(self.api.trajectory({"id": ["invalid-uuid"]})[0], 400)
        self.assertEqual(self.api.trajectory({"id": ["00000000-0000-0000-0000-000000000000"]})[0], 404)

    def test_ticket_required_fresh_signed_single_use(self):
        p = self.public("티켓")
        self.assertEqual(self.save(p, ticket=None)[1].get("need"), "ticket")             # 없음
        self.assertEqual(self.save(p, ticket=old_ticket()[:-3] + "AAA")[1].get("need"), "ticket")   # 위조
        slow = self.public("느림")
        for s_ in slow["strokes"]:                                                        # 같은 그림을 5배 천천히 (약 24초)
            for pt in s_["points"]:
                pt[2] = round(pt[2] * 5, 2)
        self.assertEqual(self.save(slow, ticket=old_ticket(0))[1].get("need"), "ticket")  # 그린 시간보다 늦게 발급
        self.assertEqual(self.save(slow, ticket=old_ticket(30))[0], 200)                  # 그리기 전에 받은 티켓은 통과
        self.assertEqual(self.save(p, ticket=old_ticket(7 * 3600))[1].get("need"), "ticket")   # 만료
        tk = old_ticket()
        st, b = self.save(p, ticket=tk)
        self.assertEqual(st, 200, b)
        self.assertTrue(submission.ticket_age(SECRET, b["ticket"]) is not None)            # 다음 티켓
        st, b = self.save(self.public("티켓2", 1), ticket=tk)                              # 같은 티켓 재사용
        self.assertEqual((st, b.get("duplicate")), (409, True))

    def test_same_drawing_cannot_be_resubmitted(self):
        p = demo_payload(2, unique=False)
        p["meta"]["nickname"] = "복붙"
        self.assertEqual(self.save(p)[0], 200)
        p["meta"]["nickname"] = "복붙2"
        self.assertEqual(self.save(p)[0], 409)

    def test_client_trajectory_is_ignored(self):
        p = self.public("궤적조작")
        honest = scoring.score_trajectory("apple", submission.build_trajectory(p["strokes"]))
        parts = scoring.template_parts("apple")
        p["trajectory"] = [[x, y, 1] for part in parts for x, y in part]                  # 완벽한 가짜 궤적
        st, b = self.save(p)
        self.assertEqual((st, b["score"], b["avg_error"]), (200, honest["score"], honest["avg_error"]))

    def test_bot_like_drawings_not_counted(self):
        parts = scoring.template_parts("apple")
        perfect = [{"pointer_type": "pen", "points": [[x, y, round(k * 8.0, 2), 0.5] for k, (x, y) in enumerate(part)]}
                   for part in parts]
        t = 0.0
        for s in perfect:   # 획 사이 시간이 이어지게
            for pt in s["points"]:
                pt[2] = round(t, 2); t += 8.0
            t += 300
        for name, strokes in [("느린완벽봇", perfect),
                              ("빠른봇", [{"pointer_type": "pen", "points": [[x, y, k * 0.5, 0.5] for k, (x, y) in enumerate(s_)]}
                                       for s_ in parts])]:
            p = self.public(name)
            p["strokes"] = strokes
            st, b = self.save(p)
            self.assertEqual(st, 200, b)
            self.assertFalse(b["qualified"], name)
            self.assertEqual(b["reason"], "검증 실패")
        names = [r["nickname"] for r in json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]]
        self.assertNotIn("느린완벽봇", names)
        self.assertNotIn("빠른봇", names)

    def test_reserved_and_disguised_nicknames(self):
        for bad in ("siat", "SIAT", "ｓｉａｔ", "관리자", "씨.발", "s i a t"):
            st, b = self.save(self.public(bad))
            self.assertEqual((st, b.get("need")), (400, "nickname"), bad)

    def test_errors_do_not_leak_internals(self):
        with mock.patch.object(self.api, "get_db", side_effect=RuntimeError("secret-internal-detail")):
            st, b = self.save(self.public("오류"))
        self.assertEqual(st, 500)
        self.assertNotIn("secret-internal-detail", json.dumps(b))
        self.assertIn("ref", b)

    def test_payload_limits(self):
        p = self.public("큰거")
        p["strokes"][0]["points"] = [[0.5, 0.5, 0, 0]] * (submission.MAX_POINTS + 1)
        self.assertEqual(self.save(p)[0], 400)                                            # 점 개수 한도
        p = self.public("이미지")
        p["target_128_png"] = "data:image/png;base64,AAAA"
        self.assertEqual(self.save(p)[0], 400)                                            # 128×128 PNG 아님
        st, b, _ = self.api.save(headers(), b"x" * (self.api.MAX_BODY_BYTES + 1), "1.1.1.1")
        self.assertEqual(st, 413)
        p = self.public("밖")
        p["strokes"][0]["points"][3][0] = 1.5
        self.assertEqual(self.save(p)[0], 400)                                            # 좌표 범위

    def test_global_rate_limit(self):
        with mock.patch.object(self.api, "GLOBAL_LIMIT", (60, 3)), frozen():
            codes = [self.save(self.public(f"g{i}"))[0] for i in range(4)]
        self.assertEqual(codes, [200, 200, 200, 503])

    def test_save_response_includes_fresh_leaderboard(self):
        st, b = self.save(self.public("응답리더"))
        self.assertEqual(st, 200)
        self.assertEqual(b["top"][0]["nickname"], "응답리더")

    def test_client_ip_rules(self):
        ip = self.api.client_ip
        self.assertEqual(ip(headers(x_vercel_forwarded_for="1.2.3.4", x_real_ip="9.9.9.9"), "5.5.5.5"), "1.2.3.4")
        self.assertEqual(ip(headers(), "5.5.5.5"), "5.5.5.5")
        # 같은 /64의 IPv6는 한 사람으로
        self.assertEqual(ip(headers(x_real_ip="2001:db8:1:2:aaaa::1"), None), ip(headers(x_real_ip="2001:db8:1:2:ffff::9"), None))
        self.assertNotEqual(ip(headers(x_real_ip="2001:db8:1:2::1"), None), ip(headers(x_real_ip="2001:db8:1:3::1"), None))
        self.assertEqual(ip(headers(x_real_ip="::ffff:1.2.3.4"), None), "1.2.3.4")

    def test_site_url_cannot_be_spoofed(self):
        self.assertEqual(self.api.site_url(headers(host="evil.example", x_forwarded_host="evil.example")),
                         "https://handvsai.vercel.app")
        self.assertEqual(self.api.site_url(headers(host="127.0.0.1:8100")), "http://127.0.0.1:8100")

    def test_huge_number_is_bad_request(self):
        st, b, _ = self.api.save(headers(), ('{"meta":{},"x":' + "9" * 5000 + "}").encode(), "1.1.1.1")
        self.assertEqual(st, 400)

    def test_attempts_count_even_when_not_saved(self):
        p = self.public("재전송")
        tk = old_ticket()
        n = self.api.RATE_LIMITS[0][1]
        with frozen():
            codes = [self.save(p, ip="3.3.3.3", ticket=tk)[0] for _ in range(4)]      # 같은 그림·같은 티켓 반복
            self.assertEqual(codes, [200, 409, 409, 409])
            codes = [self.save(self.public(f"r{i}"), ip="3.3.3.3")[0] for i in range(n - 3)]
        self.assertEqual(codes, [200] * (n - 4) + [429])                               # 실패한 시도도 1분 한도에 포함

    def test_ticket_try_limit(self):
        tk = old_ticket()
        for i in range(5):
            self.api.get_db().hit_rate(f"other{i}", tk)                                  # 이 티켓으로 5번 시도한 상태
        st, b = self.save(self.public("티켓남용"), ticket=tk)
        self.assertEqual((st, b.get("need")), (400, "ticket"))
        self.assertIn("ticket", b)

    def test_safari_out_of_order_timestamps_are_repaired_not_flagged(self):
        # iOS Safari처럼 시각이 몇~수십 ms 앞뒤로 섞인 사람 시연 → 고쳐서 받고 인정
        p = self.public("아이패드")
        pts = p["strokes"][0]["points"]
        for i in range(5, len(pts) - 1, 7):
            pts[i][2], pts[i + 1][2] = pts[i + 1][2] + 30, pts[i][2]   # 30ms 거꾸로
        st, b = self.save(p)
        self.assertEqual(st, 200, b)
        self.assertTrue(b["qualified"], b)
        row = self.api.get_db().list_qualified(0, 100)[-1]["trajectory_data"]
        self.assertGreater(row["meta"]["checks"]["time_repair_ms"], 0)
        # 0.5초 넘게 거꾸로 가면 여전히 의심
        p = self.public("시간조작")
        p["strokes"][0]["points"][10][2] = 0
        p["strokes"][0]["points"][9][2] = 2000
        st, b = self.save(p)
        self.assertEqual((st, b["reason"]), (200, "검증 실패"))

    def test_superhuman_record_held_for_review(self):
        parts = scoring.template_parts("apple")
        t, strokes = 0.0, []
        for part in parts:   # 도안을 3px 정도만 흔들며 사람 속도로 → 오차 0.5~1.5px
            pts = []
            for k, (x, y) in enumerate(part):
                pts.append([x + 0.004 * ((k // 7) % 2), y, round(t, 2), 0.5]); t += 8.0   # 오차 약 0.63px (0.5~0.8px 검토 대기)
            strokes.append({"pointer_type": "pen", "points": pts}); t += 300
        p = self.public("정밀왕")
        p["strokes"] = strokes
        st, b = self.save(p)
        self.assertEqual(st, 200, b)
        self.assertTrue(b["review"])
        self.assertFalse(b["qualified"])
        self.assertEqual(b["reason"], "검토 대기")
        self.assertNotIn("정밀왕", [r["nickname"] for r in b["top"]])

    def test_target_image_replaced_and_meta_whitelisted(self):
        p = self.public("이미지바꿈")
        p["meta"]["evil"] = "x" * 5000
        p["meta"]["device"]["user_agent"] = "u" * 2000
        p["spec"] = {"chunk_size": 999}
        self.assertEqual(self.save(p)[0], 200)
        row = self.api.get_db().list_qualified(0, 100)[-1]["trajectory_data"]
        import submission as sub_
        if sub_.canonical_target("apple"):
            self.assertEqual(row["target_128_png"], sub_.canonical_target("apple"))
        self.assertNotIn("evil", row["meta"])
        self.assertEqual(len(row["meta"]["device"]["user_agent"]), 300)
        self.assertEqual(row["spec"]["chunk_size"], 16)

    def test_shape_caps_and_specific_messages(self):
        p = self.public("튐")
        p["strokes"][0]["points"][5][0] = min(1, p["strokes"][0]["points"][4][0] + 0.5)
        st, b = self.save(p)
        self.assertEqual((st, b["error"]), (400, "선이 끊겨 튀었어요"))                    # 구체적인 안내 (500·뭉뚱그림 아님)

    def test_non_ascii_tokens_are_rejected_cleanly(self):
        st, b = self.save(demo_payload(), headers(x_collect_user="minsu", x_collect_token="토큰"))
        self.assertEqual(st, 401)
        st, b = self.save(self.public("티켓한글"), ticket="1.2.한글")
        self.assertEqual((st, b.get("need")), (400, "ticket"))

    def test_leaderboard_by_input_device(self):
        for nick, kind, i in [("펜왕", "pen", 0), ("손가락", "touch", 1), ("펜왕", "pen", 2), ("쥐돌이", "mouse", 3)]:
            p = self.public(nick, i)
            for s_ in p["strokes"]:
                s_["pointer_type"] = kind
            self.assertEqual(self.save(p)[0], 200)
        top = lambda q: json.loads(self.api.leaderboard(q)[1])["top"]
        self.assertEqual([r["nickname"] for r in top({"template": ["apple"], "input": ["pen"]})], ["펜왕"])   # 닉네임당 1개
        self.assertEqual({r["input_type"] for r in top({"template": ["apple"], "input": ["touch"]})}, {"touch"})
        self.assertEqual(len(top({"template": ["apple"]})), 3)                                            # 전체
        self.assertEqual(self.api.canonical_query("leaderboard", headers(), {"template": ["apple"], "input": ["evil"]}),
                         "template=apple")                                                                   # 모르는 값은 버림

    def test_tie_broken_by_avg_error(self):
        db = self.api.get_db()
        base = {"template_name": "apple", "tier": "S", "f1": 1, "coverage": 1, "precision": 1, "strokes": 2,
                "is_qualified": True, "trajectory_data": {}, "contributor": None, "ip_hash": None, "input_type": "pen"}
        for nick, sc, err in [("siat", 100.0, 3.29), ("rival", 100.0, 2.10), ("third", 99.9, 1.0), ("twin", 100.0, 2.10)]:
            db.insert_demo({**base, "nickname": nick, "score": sc, "avg_error": err})
        top = json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]
        self.assertEqual([(r["nickname"], r["rank"]) for r in top],
                         [("rival", 1), ("twin", 2), ("siat", 3), ("third", 4)])   # 점수·오차까지 같으면 먼저 달성한 쪽이 위
        self.assertEqual(top[0]["avg_error"], 2.10)
        self.assertEqual(db.rank_of("apple", 100.0, 2.5), (2, 4))   # 100점·오차 2.5 → rival·twin 다음
        self.assertEqual(db.rank_of("apple", 100.0, 1.0), (0, 4))
        self.assertEqual(db.rank_of("apple", 99.9, 0.5), (3, 4))

    def test_save_stores_avg_error(self):
        st, b = self.save(self.public("오차확인"))
        self.assertEqual(st, 200)
        self.assertIsInstance(b["avg_error"], float)
        top = json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]
        self.assertEqual(top[0]["avg_error"], b["avg_error"])
        missing = self.api.get_db().missing_avg_error()
        self.assertEqual(missing, [])

    def test_og_image_and_share_page(self):
        st, png, ct = self.api.og_image({"score": ["96"], "tier": ["S"], "template": ["apple"]})
        self.assertEqual((st, ct, png[:8]), (200, "image/png", b"\x89PNG\r\n\x1a\n"))
        from PIL import Image
        self.assertEqual(Image.open(io.BytesIO(png)).size, (1200, 630))
        self.assertEqual(self.api.og_image({})[0], 200)            # 기본 카드
        self.assertEqual(self.api.og_image({"score": ["x"]})[0], 200)
        st, page, _ = self.api.share_page(headers(host="handvsai.vercel.app"),
                                          {"score": ["96"], "tier": ["S"], "template": ["apple"]})
        page = page.decode()
        # 티어는 점수로 계산. B 채점의 S 기준은 96점.
        self.assertIn('content="https://handvsai.vercel.app/api/og?score=96.0&amp;tier=S&amp;template=apple"', page)
        st, page2, _ = self.api.share_page(headers(host="x"), {"score": ["98.96"], "tier": ["A"], "template": ["apple"]})
        self.assertIn("98.9점", page2.decode())   # 공유 링크 점수도 버림
        self.assertEqual(self.api.og_image({"score": ["nan"], "tier": ["S"], "template": ["apple"]})[0], 200)
        self.assertIn("summary_large_image", page)
        st, page, _ = self.api.share_page(headers(host="x"), {"score": ["96"], "tier": ["S"], "template": ['"><script>']})
        st, page3, _ = self.api.share_page(headers(host="handvsai.vercel.app"),
                                           {"score": ["96"], "tier": ["S"], "template": ["apple"], "time": ["12.4"]})
        self.assertIn("&amp;time=12.4", page3.decode())
        self.assertIn("12.4초", page3.decode())
        st, png_time, _ = self.api.og_image({"score": ["96"], "tier": ["S"], "template": ["apple"], "time": ["12.4"]})
        self.assertEqual(st, 200)
        self.assertNotIn("<script>", page.decode())

    # --- 기기 비밀값(X-Owner) · 닉네임 주인 ---------------------------------------
    def test_public_save_requires_owner_header(self):
        for bad in (None, "", "A" * 32, "a" * 31, "a" * 33, "g" * 32, "한" * 32):
            hdr = headers() if bad is None else headers(x_owner=bad)
            st, b = self.save(self.public("기기확인"), hdr, owner=None)
            self.assertEqual((st, b.get("need"), b.get("error")), (400, "reload", self.api.NEED_RELOAD), bad)
        self.assertEqual(self.api.NEED_RELOAD, "페이지를 새로고침한 뒤 다시 [채점하기]를 눌러 주세요")
        self.assertEqual(self.save(demo_payload(), link("minsu"), owner=None)[0], 200)     # 개인 링크는 없어도 됨
        secret = "0123456789abcdef" * 2
        self.assertEqual(self.save(self.public("기기확인"), headers(x_owner=secret))[0], 200)
        stored = self.api.get_db()._rows("select nickname, owner, trajectory_data from drawing_demos")
        by_nick = {r["nickname"]: r for r in stored}
        self.assertEqual(by_nick["기기확인"]["owner"], hashlib.sha256(secret.encode()).hexdigest())   # 해시만 저장
        self.assertFalse(any(secret in r["trajectory_data"] for r in stored))                   # 비밀값 자체는 어디에도 없음
        self.assertIsNone(by_nick["minsu"]["owner"])

    def test_nickname_belongs_to_first_device(self):
        a, b = device("first"), device("second")
        self.assertEqual(self.save(self.public("Alice", 0), owner=a)[0], 200)
        st, body = self.save(self.public("alice", 1), owner=b)       # 대소문자만 다른 같은 닉네임 (리더보드가 묶는 기준)
        self.assertEqual((st, body.get("need"), body.get("error")),
                         (400, "nickname", "이미 다른 사람이 쓰고 있는 닉네임이에요. 다른 닉네임을 써 주세요"))
        self.assertEqual(self.save(self.public("ALICE", 2), owner=a)[0], 200)                  # 주인 기기는 계속 사용
        self.assertEqual(self.save(self.public("Alice2", 3), owner=b)[0], 200)                 # 다른 닉네임은 됨
        # 이 기능 전부터 DB에 있던 닉네임(주인 없음)은 처음 저장한 기기가 주인이 된다
        self.api.get_db().insert_demo({"nickname": "옛닉", "template_name": "apple", "score": 90.0, "tier": "A",
                                       "f1": 0.9, "coverage": 0.9, "precision": 0.9, "strokes": 2,
                                       "is_qualified": True, "trajectory_data": {}})
        self.assertEqual(self.save(self.public("옛닉", 4), owner=b)[0], 200)
        self.assertEqual(self.save(self.public("옛닉", 5), owner=a)[1].get("need"), "nickname")
        owners = {r["nick"]: r["owner"] for r in self.api.get_db()._rows("select nick, owner from nick_owners")}
        sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
        self.assertEqual(owners, {"alice": sha(a), "alice2": sha(b), "옛닉": sha(b)})

    def test_nickname_rate_limit(self):
        a, b = device("한도주인"), device("방해꾼")
        with frozen():
            ok = [self.save(self.public("한도", i), owner=a)[0] for i in range(3)]
            # 주인이 아닌 기기의 시도는 거절되고, 주인의 닉네임 한도도 소진시키지 못한다
            bad = [self.save(self.public("한도", 0), owner=b)[1].get("need") for _ in range(5)]
            ok += [self.save(self.public("한도", i), owner=a)[0] for i in range(3, 6)]
            st, body = self.save(self.public("한도"), owner=a)
            other = self.save(self.public("한도2"), owner=a)[0]
        self.assertEqual((ok, bad), ([200] * 6, ["nickname"] * 5))
        self.assertEqual(st, 429)                                                                # 닉네임당 1분 6회
        self.assertIn("그림은 그대로", body["error"])
        self.assertEqual(other, 200)                                                             # 닉네임마다 따로 센다

    def test_hourly_limits_for_ip_and_nickname(self):
        db, a = self.api.get_db(), device("시간한도")
        iph = self.api.ip_hash(headers(), "7.7.7.7")
        for k in range(10):   # 10분 동안: 같은 IP 분당 20번(=200), 같은 닉네임 분당 6번(=60) → 분 한도 안, 1시간 한도 꽉 참
            with frozen(T0 + dt.timedelta(minutes=k)):
                for j in range(20):
                    db.hit_rate(iph, f"ip{k}-{j}")
                for j in range(6):
                    db.hit_rate(f"other-ip-{k}", f"nk{k}-{j}", "시간한도", hashlib.sha256(a.encode()).hexdigest())
        with frozen(T0 + dt.timedelta(minutes=30)):
            st_ip, b_ip = self.save(self.public("와이파이"), ip="7.7.7.7")
            st_nick, b_nick = self.save(self.public("시간한도"), owner=a)
            st_ok = self.save(self.public("새사람"))[0]
        self.assertEqual((st_ip, st_nick, st_ok), (429, 429, 200))
        self.assertIn("인터넷", b_ip["error"])
        self.assertIn("닉네임", b_nick["error"])
        self.assertTrue(all("1시간" in b["error"] and "그림은 그대로" in b["error"] for b in (b_ip, b_nick)))

    def test_save_still_works_without_hit_rate_function(self):
        # schema.sql을 다시 실행하기 전: hit_rate가 없으면 저장된 행 기준으로 세고 닉네임 확인은 건너뛴다
        import db as dbmod
        err = dbmod.DBError('Supabase POST rpc/hit_rate → HTTP 404: {"code":"PGRST202"}')
        with mock.patch.object(dbmod.SqliteDB, "hit_rate", side_effect=err):
            self.assertEqual(self.save(self.public("대체", 0), owner=device("x"))[0], 200)
            self.assertEqual(self.save(self.public("대체", 1), owner=device("y"))[0], 200)

    def test_soft_flags_keep_raw_data_hard_flags_strip_it(self):
        parts = scoring.template_parts("apple")
        perfect, t = [], 0.0
        for part in parts:                                   # 사람 속도로 도안을 그대로 → '너무 정확함'(약한 신호)만
            pts = []
            for x, y in part:
                pts.append([x, y, round(t, 2), 0.5]); t += 8.0
            perfect.append({"pointer_type": "pen", "points": pts}); t += 300
        fast = [{"pointer_type": "pen", "points": [[x, y, k * 0.5, 0.5] for k, (x, y) in enumerate(part)]}
                for part in parts]                           # 사람 손으로 불가능한 속도 (강한 신호)
        back = self.public("시간역행", 1)
        pts = back["strokes"][0]["points"]
        k = next(i for i in range(1, len(pts)) if pts[i - 1][2] >= 700)
        pts[k][2] = round(pts[k - 1][2] - 600, 2)            # 한 점만 0.6초 거꾸로 (Safari 오탐 같은 약한 신호)
        cases = [("완벽손", perfect, "too_perfect"), ("빠른손", fast, "speed_avg"), ("시간역행", None, "time_backwards")]
        for nick, strokes, _ in cases:
            p = back if strokes is None else self.public(nick)
            if strokes is not None:
                p["strokes"] = strokes
            st, b = self.save(p)
            self.assertEqual((st, b["qualified"], b["reason"]), (200, False, "검증 실패"), nick)
        rows = self.rows()
        for nick, _, flag in cases:
            self.assertIn(flag, rows[nick]["meta"]["flags"], nick)
        for nick in ("완벽손", "시간역행"):                   # 약한 신호만: 원시 입력·궤적·이미지를 그대로 남긴다
            d = rows[nick]
            self.assertEqual(set(d["meta"]["flags"]) - submission.SOFT_FLAGS, set(), nick)
            self.assertTrue(d["strokes"] and d["trajectory"] and d["target_128_png"], nick)
        self.assertEqual(set(rows["빠른손"]), {"schema_version", "meta"})                       # 강한 신호: meta만
        self.assertTrue(set(rows["빠른손"]["meta"]["flags"]) & submission.HARD_FLAGS)
        names = {r["nickname"] for r in json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]}
        self.assertFalse({"완벽손", "빠른손", "시간역행"} & names)                              # 기록으로는 인정 안 함

    def test_guide_hint_meta_is_kept(self):
        p = self.public("안내켬", guide_hint="next-v1", guide_order_marks=True)
        self.assertEqual(self.save(p)[0], 200)
        p = self.public("안내끔", 1, guide_hint="off" + "x" * 200)
        self.assertEqual(self.save(p)[0], 200)
        rows = self.rows()
        self.assertEqual((rows["안내켬"]["meta"]["guide_hint"], rows["안내켬"]["meta"]["guide_order_marks"]), ("next-v1", True))
        self.assertEqual(len(rows["안내끔"]["meta"]["guide_hint"]), 80)                         # 다른 문자열처럼 잘라서

    # --- 퍼널 이벤트 · 내 기록 지우기 · 깨우기 ------------------------------------
    def test_event_endpoint_counts_known_events_only(self):
        import db as dbmod
        ev = lambda body, ip="1.2.3.4": self.api.event(headers(), body, ip)
        self.assertEqual(ev(b"visit"), (204, b"", "text/plain; charset=utf-8"))
        for body in (b"visit", b"start\n", b"result", b"hack", b"x" * 40, b"", None, "방문".encode(), b"visit" + b" " * 40):
            self.assertEqual(ev(body)[:2], (204, b""))
        rows = self.api.get_db()._rows("select day, event, n from funnel_counts")
        self.assertEqual({r["day"] for r in rows}, {dbmod.kst_today()})                           # 한국 시간 날짜별
        self.assertEqual({r["event"]: r["n"] for r in rows}, {"visit": 2, "start": 1, "result": 1})  # 모르는 이름·큰 본문 무시
        with mock.patch.object(dbmod.SqliteDB, "bump_event", side_effect=dbmod.DBError("PGRST202 bump_event")):
            self.assertEqual(ev(b"visit")[:2], (204, b""))                                         # DB 함수가 없어도 204
        with mock.patch.object(self.api, "EVENT_IP_PER_MIN", 2):
            for _ in range(5):
                ev(b"share", "8.8.8.8")                                                            # 한 IP 폭주 → 상한까지만
        n = {r["event"]: r["n"] for r in self.api.get_db()._rows("select event, n from funnel_counts")}
        self.assertEqual((n["visit"], n["share"]), (2, 2))
        self.assertEqual(set(self.api.EVENTS), {"visit", "start", "submit", "result", "retry", "next", "share", "card",
                                                 "demo_done", "demo_skip", "hint_off"})

    def test_delete_my_records_and_release_nickname(self):
        a, b, c = device("지우는기기"), device("남는기기"), device("새기기")
        for i in range(2):
            self.assertEqual(self.save(self.public("지울닉", i), owner=a)[0], 200)
        self.assertEqual(self.save(self.public("남을닉", 2), owner=b)[0], 200)
        self.assertEqual(self.save(self.public("지울닉", 3), owner=c)[1].get("need"), "nickname")
        for hdr in (headers(), headers(x_owner="A" * 32), headers(x_owner="zz")):
            st, body, _ = self.api.delete(hdr)
            self.assertEqual((st, json.loads(body)["ok"]), (400, False))
        names = lambda: {r["nickname"] for r in json.loads(self.api.leaderboard({"template": ["apple"]})[1])["top"]}
        self.assertEqual(names(), {"지울닉", "남을닉"})
        st, body, _ = self.api.delete(headers(x_owner=a))
        self.assertEqual((st, json.loads(body)), (200, {"ok": True, "deleted": 2}))
        self.assertEqual(names(), {"남을닉"})                                                     # 인스턴스 캐시도 비움
        self.assertEqual([r["nickname"] for r in self.api.get_db()._rows("select nickname from drawing_demos")], ["남을닉"])
        self.assertEqual(self.save(self.public("지울닉", 4), owner=c)[0], 200)                    # 반납된 닉네임은 새 기기가
        self.assertEqual(json.loads(self.api.delete(headers(x_owner=a))[1]), {"ok": True, "deleted": 0})
        sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
        owners = {r["nick"]: r["owner"] for r in self.api.get_db()._rows("select nick, owner from nick_owners")}
        self.assertEqual(owners, {"남을닉": sha(b), "지울닉": sha(c)})

    def test_cron_keepalive(self):
        with mock.patch.dict(os.environ, {"CRON_SECRET": "cron-secret-value"}):
            for hdr in (headers(), headers(authorization="Bearer wrong"), headers(authorization="cron-secret-value"),
                        headers(authorization="Bearer 크론")):
                st, body, _ = self.api.cron(hdr)
                self.assertEqual((st, json.loads(body)["ok"]), (401, False))
            st, body, _ = self.api.cron(headers(authorization="Bearer cron-secret-value"))
            self.assertEqual((st, json.loads(body)), (200, {"ok": True}))
        with mock.patch.dict(os.environ):
            os.environ.pop("CRON_SECRET", None)
            self.assertEqual(self.api.cron(headers())[:2], (200, b'{"ok": true}'))               # 비밀값이 없으면 확인 없이


class TestVercelEntrypoints(unittest.TestCase):
    def test_each_function_defines_top_level_handler_class(self):
        # Vercel CLI는 최상위 `class handler` 정의가 있어야 api/*.py를 함수로 인식한다 (handler = ... 대입은 배포 실패)
        import re
        import api_core
        files = {f.stem: f for f in sorted((ROOT / "api").glob("*.py"))}
        self.assertEqual(set(files), set(api_core.HANDLER_NAMES))
        self.assertTrue({"event", "delete", "cron"} <= set(files))
        for name, f in files.items():
            src = f.read_text(encoding="utf-8")
            self.assertRegex(src, re.compile(r"^class handler\(", re.M), f.name)
            self.assertRegex(src, re.compile(rf'^class handler\(api_core\.make_handler\("{name}"\)\):\n    pass$', re.M), f.name)
            self.assertTrue(hasattr(api_core.make_handler(name), "do_POST"))
        with self.assertRaises(ValueError):
            api_core.make_handler("nope")
        vj = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
        self.assertIn({"path": "/api/cron", "schedule": "0 3 * * *"}, vj["crons"])                 # 매일 깨우기
        self.assertEqual(vj["regions"], ["sin1"])
        self.assertIn("api/*.py", vj["functions"])


class TestSupabaseRequests(unittest.TestCase):
    """Supabase REST(PostgREST) 요청 형식 확인 — 실제 네트워크 없이 urlopen을 가로챈다."""

    def setUp(self):
        import db
        self.db = db.SupabaseDB("https://abc.supabase.co", "service-key")
        self.calls = []

    def fake(self, body=b"[]", hdrs=None):
        test = self

        class Resp(io.BytesIO):
            def __init__(self):
                super().__init__(body)
                self.headers = hdrs or {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def send(self_, method, url, hdr, data):
            test.calls.append((method, url, dict(hdr), data))
            return 200, {k.lower(): v for k, v in (hdrs or {}).items()}, body
        return mock.patch("db.SupabaseDB._send", send)

    def test_insert(self):
        with self.fake(b'[{"id": "0b1e"}]'):
            rid = self.db.insert_demo({"nickname": "a", "trajectory_data": {"k": "한글"}})
        m, url, h, data = self.calls[0]
        self.assertEqual((rid, m), ("0b1e", "POST"))
        self.assertEqual(url, "https://abc.supabase.co/rest/v1/drawing_demos?select=id")
        self.assertEqual((h["apikey"], h["Authorization"], h["Prefer"]), ("service-key", "Bearer service-key", "return=representation"))
        self.assertEqual(json.loads(data)["trajectory_data"]["k"], "한글")

    def test_count_from_content_range(self):
        with self.fake(b"[]", {"Content-Range": "0-0/42"}):
            n = self.db.recent_count("abcd", 60)
        m, url, h, _ = self.calls[0]
        self.assertEqual(n, 42)
        self.assertIn("ip_hash=eq.abcd", url)
        self.assertIn("created_at=gte.", url)
        self.assertEqual((h["Prefer"], h["Range"]), ("count=exact", "0-0"))

    def test_rank_of_tie_filter_and_update(self):
        with self.fake(b"[]", {"Content-Range": "0-0/3"}):
            self.db.rank_of("apple", 100.0, 2.1)
            self.db.update_demo("abc", {"avg_error": 1.5})
        better = urllib.parse.unquote(self.calls[0][1])
        self.assertIn("or=(score.gt.100.0,and(score.eq.100.0,avg_error.lt.2.1))", better)
        m, url, _, data = self.calls[2]
        self.assertEqual((m, json.loads(data)), ("PATCH", {"avg_error": 1.5}))
        self.assertIn("id=eq.abc", url)

    def test_leaderboard_and_list(self):
        with self.fake(b'[{"nickname":"a","score":99,"tier":"S","rank":1,"created_at":"x"}]'):
            self.db.leaderboard("apple", 5)
            self.db.list_qualified(100, 50)
        lb, lst = self.calls[0][1], self.calls[1][1]
        self.assertTrue(lb.startswith("https://abc.supabase.co/rest/v1/drawing_demos?"))
        self.assertIn("template_name=eq.apple", lb)
        self.assertIn("is_qualified=eq.true", lb)
        self.assertIn("score_meta:trajectory_data->meta->score", urllib.parse.unquote(lb))
        self.assertIn("is_qualified=eq.true", lst)
        self.assertIn("offset=100", lst)
        self.assertIn("limit=50", lst)

    def fake_seq(self, *responses):
        """호출마다 (HTTP 상태, 본문)을 하나씩 돌려주는 가짜 전송."""
        test, queue = self, list(responses)

        def send(self_, method, url, hdr, data):
            test.calls.append((method, url, dict(hdr), data))
            status, body = queue.pop(0)
            return status, {}, body
        return mock.patch("db.SupabaseDB._send", send)

    MISSING_FN = (b'{"code":"PGRST202","message":"Could not find the function '
                  b'public.hit_rate(p_ip, p_nick, p_owner, p_ticket) in the schema cache"}')

    def test_hit_rate_named_params_and_fallbacks(self):
        import db
        row4 = b'[{"ip_min":1,"ip_hour":2,"all_min":3,"ticket_n":1,"nick_min":1,"nick_hour":4,"nick_ok":false}]'
        with self.fake_seq((200, row4)):
            h = self.db.hit_rate("iph", "tk", "alice", "o" * 64)
        m, url, _, data = self.calls[0]
        self.assertEqual((m, url), ("POST", "https://abc.supabase.co/rest/v1/rpc/hit_rate"))
        self.assertEqual(json.loads(data), {"p_ip": "iph", "p_ticket": "tk", "p_nick": "alice", "p_owner": "o" * 64})
        self.assertEqual(h, {"ip_min": 1, "ip_hour": 2, "all_min": 3, "ticket_n": 1, "nick_min": 1, "nick_hour": 4,
                             "nick_ok": False})
        # 4개 인자 함수가 아직 없음(schema.sql 다시 실행 전) → 예전 2개 인자 함수 (닉네임 확인 없음)
        self.calls.clear()
        with self.fake_seq((404, self.MISSING_FN), (200, b'[{"ip_min":1,"ip_hour":1,"all_min":1,"ticket_n":2}]')):
            h = self.db.hit_rate("iph", "tk", "alice", "o" * 64)
        self.assertEqual(json.loads(self.calls[1][3]), {"p_ip": "iph", "p_ticket": "tk"})
        self.assertEqual((h["ticket_n"], h["nick_ok"], h["nick_min"], h["nick_hour"]), (2, True, 0, 0))
        # 둘 다 없음 → DBError (메시지에 hit_rate → api_core가 저장된 행 기준으로 대체)
        with self.fake_seq((404, self.MISSING_FN), (404, self.MISSING_FN)), self.assertRaises(db.DBError) as cm:
            self.db.hit_rate("iph", "tk")
        self.assertIn("hit_rate", str(cm.exception))
        # 함수 안의 다른 오류는 예전 함수로 넘기지 않는다
        self.calls.clear()
        with self.fake_seq((400, b'{"code":"42702","message":"column reference is ambiguous"}')), \
                self.assertRaises(db.DBError):
            self.db.hit_rate("iph", "tk", "a", "b")
        self.assertEqual(len(self.calls), 1)

    def test_insert_retries_without_owner_when_column_missing(self):
        import db
        nocol = b'{"code":"PGRST204","message":"Could not find the \'owner\' column of \'drawing_demos\' in the schema cache"}'
        with self.fake_seq((400, nocol), (201, b'[{"id": "r1"}]')):
            rid = self.db.insert_demo({"nickname": "a", "owner": "f" * 64, "trajectory_data": {}})
        self.assertEqual(rid, "r1")
        self.assertEqual((json.loads(self.calls[0][3])["owner"], "owner" in json.loads(self.calls[1][3])), ("f" * 64, False))
        self.calls.clear()
        with self.fake_seq((409, b'{"code":"23505","message":"duplicate key"}')), self.assertRaises(db.Duplicate):
            self.db.insert_demo({"nickname": "a", "owner": "f" * 64})
        with self.fake_seq((400, b'{"code":"23514","message":"check constraint"}')), self.assertRaises(db.DBError):
            self.db.insert_demo({"nickname": "a", "owner": "f" * 64})
        self.assertEqual(len(self.calls), 2)                                               # 다른 오류는 다시 보내지 않음

    def test_delete_by_owner(self):
        h = "e" * 64
        with self.fake_seq((200, b'[{"id":"1"},{"id":"2"}]'), (204, b"")):
            self.assertEqual(self.db.delete_by_owner(h), 2)
        (m1, u1, h1, _), (m2, u2, _, _) = self.calls
        self.assertEqual((m1, m2, h1["Prefer"]), ("DELETE", "DELETE", "return=representation"))
        self.assertIn(f"/rest/v1/drawing_demos?owner=eq.{h}&select=id", u1)
        self.assertIn(f"/rest/v1/nick_owners?owner=eq.{h}", u2)
        with self.fake_seq((400, b'{"code":"42703","message":"column drawing_demos.owner does not exist"}')):
            self.assertEqual(self.db.delete_by_owner(h), 0)                                 # owner 열이 아직 없음
        notable = b'{"code":"PGRST205","message":"Could not find the table \'public.nick_owners\' in the schema cache"}'
        with self.fake_seq((200, b'[{"id":"1"}]'), (404, notable)):
            self.assertEqual(self.db.delete_by_owner(h), 1)                                 # nick_owners 표가 아직 없음
        n = len(self.calls)
        self.assertEqual(self.db.delete_by_owner(""), 0)                                    # 빈 값으로는 요청하지 않음
        self.assertEqual(len(self.calls), n)

    def test_bump_event_and_keepalive(self):
        with self.fake_seq((204, b""), (200, b'[{"id":"x"}]')):
            self.db.bump_event("visit")
            self.assertTrue(self.db.keepalive())
        (m1, u1, _, d1), (m2, u2, _, _) = self.calls
        self.assertEqual((m1, u1, json.loads(d1)), ("POST", "https://abc.supabase.co/rest/v1/rpc/bump_event", {"p_event": "visit"}))
        self.assertEqual((m2, u2), ("GET", "https://abc.supabase.co/rest/v1/drawing_demos?select=id&limit=1"))


@unittest.skipUnless(DEMO_FILES, "data/raw/demo_apple_*.json 필요")
class TestVercelHandlersAndPull(unittest.TestCase):
    """api/*.py의 실제 handler 클래스를 HTTP로 띄워 확인 + pull_dataset.py 동기화."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dbpath = str(Path(cls.tmp.name) / "h.sqlite")
        cls.env = mock.patch.dict(os.environ, {"COLLECT_LOCAL_DB": cls.dbpath, "COLLECT_SECRET": SECRET})
        cls.env.start()
        import api_core
        api_core._db = None
        api_core._cache.clear()
        cls.servers = {}
        for i, name in enumerate(("save", "stats", "leaderboard", "og", "share", "ticket", "event", "delete", "cron")):
            spec = importlib.util.spec_from_file_location(f"api_{name}", ROOT / "api" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.handler.log_message = lambda *a, **k: None
            srv = HTTPServer(("127.0.0.1", 0), mod.handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            cls.servers[name] = srv

    @classmethod
    def tearDownClass(cls):
        for s in cls.servers.values():
            s.shutdown()
            s.server_close()
        import api_core
        api_core._db = None
        cls.env.stop()
        cls.tmp.cleanup()

    def req(self, name, path="/", data=None, hdr=None):
        url = f"http://127.0.0.1:{self.servers[name].server_port}{path}"
        r = urllib.request.Request(url, data=data, headers=hdr or {})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type")

    def test_handlers_and_pull(self):
        p = demo_payload()
        p["meta"].update(nickname="핸들러")
        p.pop("trajectory")                                    # 배포 수집기는 trajectory를 보내지 않는다
        st, b, _ = self.req("ticket", "/api/ticket")
        self.assertEqual(st, 200)
        self.assertIsNotNone(submission.ticket_age(SECRET, json.loads(b)["ticket"]))
        st, b, _ = self.req("save", "/api/save", json.dumps(p).encode(),
                            {"Content-Type": "application/json", "X-Draw-Ticket": old_ticket()})
        self.assertEqual((st, json.loads(b).get("need")), (400, "reload"))                  # 기기 비밀값 없음 → 새로고침
        st, b, _ = self.req("save", "/api/save", json.dumps(p).encode(),
                            {"Content-Type": "application/json", "X-Draw-Ticket": old_ticket(), "X-Owner": device("핸들러")})
        self.assertEqual(st, 200, b)
        self.assertEqual(self.req("stats", "/api/stats")[0], 200)
        r = urllib.request.urlopen(f"http://127.0.0.1:{self.servers['leaderboard'].server_port}/api/leaderboard?template=apple")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")                 # 보안 헤더
        self.assertNotIn("Python", r.headers.get("Server", ""))                            # 서버 버전 숨김
        self.assertIn("s-maxage", r.headers["Cache-Control"])                             # CDN 캐시
        st, b, _ = self.req("leaderboard", "/api/leaderboard?template=apple")
        self.assertEqual(json.loads(b)["top"][0]["nickname"], "핸들러")
        st, b, ct = self.req("og", "/api/og?score=97.0&tier=S&template=apple")
        self.assertEqual((st, ct), (200, "image/png"))
        # 정규형이 아닌 주소(캐시 우회용 파라미터, 다른 표기, 가짜 티어)는 처리하지 않고 정규 주소로 308
        for path, loc in [("/api/og?score=97&tier=A&template=apple&x=1", "/api/og?score=97.0&tier=S&template=apple"),
                          ("/api/og?junk=1", "/api/og"),
                          ("/api/leaderboard?template=apple&t=123", "/api/leaderboard?template=apple"),
                          ("/api/stats?me=someone", "/api/stats")]:
            name = path.split("?")[0].rsplit("/", 1)[-1]
            conn = __import__("http.client").client.HTTPConnection("127.0.0.1", self.servers[name].server_port)
            conn.request("GET", path)
            r = conn.getresponse()
            self.assertEqual((r.status, r.getheader("Location")), (308, loc), path)
            conn.close()
        st, b, ct = self.req("save", "/api/save")                                      # GET → 405 JSON
        self.assertEqual((st, json.loads(b)["ok"]), (405, False))
        st, b, ct = self.req("stats", "/api/stats", b"{}")                              # POST → 405
        self.assertEqual(st, 405)
        st, b, ct = self.req("share", "/s?score=97.0&tier=A&template=apple", hdr={"Host": "handvsai.vercel.app"})
        self.assertIn(b"og:image", b)

        # 채택 안 된 시연도 하나 넣고 pull → 채택된 것만 받아야 함
        bad = demo_payload(1)
        bad["meta"].update(nickname="낙서")
        bad["strokes"] = as_strokes([(0.1 + 0.008 * k, 0.1 + 0.008 * k) for k in range(100)], 20)
        self.req("save", "/api/save", json.dumps(bad).encode(),
                 {"Content-Type": "application/json", "X-Forwarded-For": "7.7.7.7", "X-Draw-Ticket": old_ticket(),
                  "X-Owner": device("낙서")})
        dest = Path(self.tmp.name) / "raw"
        cmd = [sys.executable, str(ROOT / "tools" / "pull_dataset.py"), "--sqlite", self.dbpath,
               "--dest", str(dest), "--rejected", str(Path(self.tmp.name) / "rej"), "--no-check"]
        out1 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
        files = list(dest.glob("demo_apple_*.json"))
        self.assertEqual(len(files), 1, out1)
        saved = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(saved["meta"]["nickname"], "핸들러")
        out2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
        self.assertIn("이미 있음 1개", out2)

    def test_event_delete_cron_over_http(self):
        import http.client
        import db as dbmod

        def call(name, method, path, data=None, hdr=None):
            conn = http.client.HTTPConnection("127.0.0.1", self.servers[name].server_port, timeout=10)
            conn.request(method, path, body=data, headers=hdr or {})
            r = conn.getresponse()
            out = (r.status, r.read(), {k.lower(): v for k, v in r.getheaders()})
            conn.close()
            return out
        st, body, h = call("event", "POST", "/api/event", b"visit", {"Content-Type": "text/plain;charset=UTF-8"})
        self.assertEqual((st, body, h.get("cache-control"), h.get("x-content-type-options")), (204, b"", "no-store", "nosniff"))
        self.assertNotIn("content-length", h)                                                  # 204는 본문 길이 헤더 없음
        self.assertEqual(call("event", "POST", "/api/event", b"x" * 100_000)[:2], (204, b""))  # 큰 본문도 204 (무시)
        self.assertEqual(call("event", "POST", "/api/event", b"nope")[:2], (204, b""))
        self.assertEqual(call("event", "GET", "/api/event")[0], 405)
        rows = dbmod.SqliteDB(self.dbpath)._rows("select event, n from funnel_counts")
        self.assertEqual({r["event"]: r["n"] for r in rows}, {"visit": 1})
        st, body, _ = call("delete", "POST", "/api/delete", b"")
        self.assertEqual((st, json.loads(body)["ok"]), (400, False))
        st, body, h = call("delete", "POST", "/api/delete", b"", {"X-Owner": device("아무것도안함")})
        self.assertEqual((st, json.loads(body), h["cache-control"]), (200, {"ok": True, "deleted": 0}, "no-store"))
        self.assertEqual(call("delete", "GET", "/api/delete")[0], 405)
        with mock.patch.dict(os.environ):
            os.environ.pop("CRON_SECRET", None)
            st, body, h = call("cron", "GET", "/api/cron?x=1")                                  # 정규 주소 리다이렉트 없음
            self.assertEqual((st, json.loads(body), h["cache-control"]), (200, {"ok": True}, "no-store"))
            os.environ["CRON_SECRET"] = "cron-s3cret"
            self.assertEqual(call("cron", "GET", "/api/cron")[0], 401)
            self.assertEqual(call("cron", "GET", "/api/cron", hdr={"Authorization": "Bearer cron-s3cret"})[0], 200)
        self.assertEqual(call("cron", "POST", "/api/cron", b"{}")[0], 405)


@unittest.skipUnless(DEMO_FILES, "data/raw/demo_apple_*.json 필요")
class TestSeed(unittest.TestCase):
    """tools/seed_supabase.py: 로컬 시연 일괄 등록 → 리더보드 갱신, 재실행 시 중복 없음, pull은 시드 원본을 다시 받지 않음."""

    def test_seed_idempotent_and_pull_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            for f in DEMO_FILES[:5]:
                (src / Path(f).name).write_bytes(Path(f).read_bytes())
            dbp = str(Path(tmp) / "s.sqlite")
            cmd = [sys.executable, str(ROOT / "tools" / "seed_supabase.py"), "--root", str(src), "--sqlite", dbp]
            out1 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
            self.assertIn("새로 등록 5개", out1)
            self.assertIn("1위 siat", out1)
            out2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
            self.assertIn("이미 등록됨 5개", out2)
            import db
            rows = db.SqliteDB(dbp).list_qualified(0, 100)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["trajectory_data"]["meta"]["seeded_from"] for r in rows))
            # 원본이 로컬에 있으면 pull_dataset은 다시 받지 않음
            pull = [sys.executable, str(ROOT / "tools" / "pull_dataset.py"), "--sqlite", dbp, "--dest", str(src),
                    "--rejected", str(Path(tmp) / "rej"), "--no-check"]
            out3 = subprocess.run(pull, capture_output=True, text=True, encoding="utf-8").stdout
            self.assertIn("새로 받음 0개", out3)
            self.assertEqual(len(list(src.glob("demo_*.json"))), 5)

    def test_seed_skips_demos_pulled_from_db(self):
        # data/raw에는 pull_dataset.py가 DB에서 받아 온 시연도 있다 → 다시 올리면 'siat' 중복 기록이 되므로 건너뛴다
        import db
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            src.mkdir()
            for f in DEMO_FILES[:3]:
                (src / Path(f).name).write_bytes(Path(f).read_bytes())
            for i, f in enumerate(DEMO_FILES[3:5]):
                d = json.loads(Path(f).read_text(encoding="utf-8"))
                d["meta"].update({"db_id": "0b1e0000-0000-4000-8000-00000000000a"} if i == 0 else {"trajectory_source": "server"})
                (src / f"demo_apple_20261108-00000{i}-0b1e000{i}.json").write_text(json.dumps(d), encoding="utf-8")
            dbp = str(Path(tmp) / "p.sqlite")
            cmd = [sys.executable, str(ROOT / "tools" / "seed_supabase.py"), "--root", str(src), "--sqlite", dbp]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            dry = subprocess.run(cmd + ["--dry-run"], capture_output=True, text=True, encoding="utf-8", env=env).stdout
            self.assertIn("DB에서 받아 온 시연 2개 건너뜀", dry)
            self.assertIn("올릴 후보 3개", dry)
            out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env).stdout
            self.assertIn("새로 등록 3개", out)
            seeded = {r["trajectory_data"]["meta"]["seeded_from"] for r in db.SqliteDB(dbp).list_qualified(0, 100)}
            self.assertEqual(seeded, {Path(f).name for f in DEMO_FILES[:3]})


class TestBackupAndMonitor(unittest.TestCase):
    """tools/backup.py (읽기 전용 전체 백업) + tools/monitor.py 오늘 퍼널 줄 — 임시 SQLite로만."""

    def test_backup_writes_all_tables_and_monitor_shows_funnel(self):
        import re
        import db
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with tempfile.TemporaryDirectory() as tmp:
            dbp = str(Path(tmp) / "b.sqlite")
            d = db.SqliteDB(dbp)
            base = {"template_name": "apple", "tier": "A", "f1": 0.9, "coverage": 0.9, "precision": 0.9, "strokes": 2}
            d.insert_demo({**base, "nickname": "채택", "score": 90.0, "is_qualified": True, "owner": "o" * 64,
                           "trajectory_data": {"meta": {"k": "한글"}, "strokes": [[0.1, 0.2]]}})
            d.insert_demo({**base, "nickname": "의심", "score": 10.0, "tier": "C", "is_qualified": False,
                           "trajectory_data": {"meta": {"flags": ["speed_avg"]}}})
            d.hit_rate("iph", "tk", "채택", "o" * 64)                                       # 닉네임 주인 1명
            for e in ("visit", "visit", "visit", "visit", "start", "start", "submit", "result", "next", "share", "hint_off"):
                d.bump_event(e)
            out_dir = Path(tmp) / "bk"
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "backup.py"), "--sqlite", dbp, "--out", str(out_dir)],
                               capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            files = {re.match(r"(.+)_\d{8}-\d{6}\.jsonl\.gz$", p.name).group(1): p for p in out_dir.glob("*.jsonl.gz")}
            self.assertEqual(set(files), {"drawing_demos", "funnel_counts", "nick_owners"})
            self.assertEqual(list(out_dir.glob("*.part")), [])
            read = lambda p: [json.loads(line) for line in gzip.open(p, "rt", encoding="utf-8")]
            demos = {row["nickname"]: row for row in read(files["drawing_demos"])}
            self.assertEqual(set(demos), {"채택", "의심"})                                      # 채택 안 된 기록도 전부
            self.assertEqual(demos["채택"]["trajectory_data"]["meta"]["k"], "한글")              # 모든 열, jsonb는 객체로
            self.assertEqual((demos["채택"]["owner"], demos["채택"]["is_qualified"], demos["의심"]["is_qualified"]),
                             ("o" * 64, True, False))
            self.assertEqual({(x["event"], x["n"]) for x in read(files["funnel_counts"])},
                             {("visit", 4), ("start", 2), ("submit", 1), ("result", 1), ("next", 1), ("share", 1), ("hint_off", 1)})
            self.assertEqual([x["nick"] for x in read(files["nick_owners"])], ["채택"])
            self.assertIn("drawing_demos", r.stdout)
            self.assertEqual(d.counts()["apple"]["total"], 2)                                     # 읽기만 (DB 그대로)

            mon = subprocess.run([sys.executable, str(ROOT / "tools" / "monitor.py"), "--sqlite", dbp, "--once", "--no-color"],
                                 capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertEqual(mon.returncode, 0, mon.stderr)
            line = next(x for x in mon.stdout.splitlines() if "오늘 퍼널" in x)
            self.assertIn("방문 4 → 첫 획 2 (50%) → 채점 1 (50%) → 결과 1 (100%) → 다시 0 (0%) · 다음 1 (100%) · 공유 1 (100%)", line)
            self.assertIn("안내 끔 1", line)


if __name__ == "__main__":
    unittest.main()
