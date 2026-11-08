"""추정 점수 표시와 닉네임별 최고 기록의 페이지 경계를 검증한다."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api_core
import db


class TestLeaderboardScores(unittest.TestCase):
    def test_sqlite_exposes_estimate_of_selected_best_record(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = db.SqliteDB(str(Path(tmp.name) / "test.sqlite"))
        base = {"template_name": "apple", "tier": "A", "is_qualified": True,
                "avg_error": 2.0, "input_type": "touch"}
        store.insert_demo({**base, "nickname": "추정", "score": 90,
                           "trajectory_data": {"meta": {"score": {"estimated": True}}}})
        store.insert_demo({**base, "nickname": "실측", "score": 88,
                           "trajectory_data": {"meta": {"score": {"estimated": False}}}})
        rows = [api_core._lb_row(r) for r in store.leaderboard("apple")]
        self.assertTrue(rows[0]["score_estimated"])
        self.assertFalse(rows[1]["score_estimated"])
        filtered = [api_core._lb_row(r) for r in store.leaderboard_by_input("apple", "touch")]
        self.assertEqual(rows, filtered)

    def test_supabase_reads_next_page_when_one_nickname_fills_first_page(self):
        store = db.SupabaseDB("https://test.supabase.co", "test-key")
        common = {"nickname": "반복", "score": 95, "avg_error": 1.0, "tier": "S",
                  "input_type": "touch", "created_at": "2026-11-08T00:00:00Z"}
        other = {**common, "nickname": "다른사람", "score": 90,
                 "score_meta": {"estimated": True}}
        with patch.object(store, "_req", side_effect=[([common] * 100, {}), ([other], {})]) as req:
            top = store.leaderboard("apple", 2)
        self.assertEqual([r["nickname"] for r in top], ["반복", "다른사람"])
        self.assertTrue(api_core._lb_row(top[1])["score_estimated"])
        self.assertEqual(req.call_args_list[1].args[2]["offset"], 100)

    def test_old_records_do_not_claim_to_be_estimated(self):
        self.assertFalse(api_core._lb_row({"rank": 1, "nickname": "기존", "score": 90})["score_estimated"])


if __name__ == "__main__":
    unittest.main()
