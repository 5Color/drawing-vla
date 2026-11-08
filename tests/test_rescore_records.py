"""Safety checks for the explicit, backup-first historical score migration."""
from __future__ import annotations

import copy
import contextlib
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import db
import scoring
from tools import rescore_records as rr


def example_trajectory(offset=0):
    out = []
    for part in scoring.template_parts("apple"):
        points = [[min(1, max(0, x + offset)), y, 1] for x, y in part]
        out.extend([[points[0][0], points[0][1], 0], *points,
                    [points[-1][0], points[-1][1], 0]])
    return out


def record(rid, *, offset=0, qualified=True, raw=True, error=2, flags=None):
    payload = {"schema_version": 1, "meta": {"target": "apple", "flags": flags or [],
               "review": False, "custom": "preserve"}, "target_128_png": "original-target"}
    if raw:
        payload["trajectory"] = example_trajectory(offset)
        payload["strokes"] = [{"points": [[.1, .2, 0, .5], [.2, .3, 10, .5]]}]
    return {"id": rid, "nickname": "person" + rid, "template_name": "apple", "score": 100,
            "tier": "S", "f1": .99, "coverage": .98, "precision": .97, "avg_error": error,
            "strokes": 3, "is_qualified": qualified, "trajectory_data": payload}


class CalibrationTests(unittest.TestCase):
    def test_isotonic_fit_pools_violation_and_interpolates_monotonically(self):
        samples = [{"id": str(i), "avg_error": x, "score": y}
                   for i, (x, y) in enumerate([(1, 90), (2, 70), (3, 80), (4, 40)])]
        model = rr.fit_calibration(samples, "template:apple")
        self.assertEqual(model["knots"], [[1.0, 90.0], [2.0, 75.0], [3.0, 75.0], [4.0, 40.0]])
        estimates = [rr.estimate_score(model, x) for x in [0, 1, 1.5, 2, 2.5, 3, 4, 8]]
        self.assertEqual(estimates, sorted(estimates, reverse=True))
        self.assertEqual(rr.estimate_score(model, 1.5), 82.5)
        self.assertEqual(rr.fit_calibration(list(reversed(samples)), "template:apple"), model)


class PlanningTests(unittest.TestCase):
    def calibration_rows(self):
        return [record("a", offset=.005), record("b", offset=.012), record("c", offset=.02)]

    def test_raw_rescore_preserves_source_and_never_readmits_excluded_records(self):
        row = record("excluded", qualified=False, flags=["time_backwards"])
        row["trajectory_data"]["meta"]["review"] = True
        before = copy.deepcopy(row)
        plan = rr.build_plan([row])
        change = plan["changes"][0]
        self.assertEqual(change["kind"], "exact")
        self.assertFalse(change["fields"]["is_qualified"])
        saved = change["fields"]["trajectory_data"]
        self.assertEqual(saved["trajectory"], before["trajectory_data"]["trajectory"])
        self.assertEqual(saved["strokes"], before["trajectory_data"]["strokes"])
        self.assertEqual(saved["target_128_png"], "original-target")
        self.assertTrue(saved["meta"]["review"])
        self.assertEqual(saved["meta"]["flags"], ["time_backwards"])
        self.assertEqual(saved["meta"]["score"]["score_version"], "b-v1")
        self.assertFalse(saved["meta"]["score"]["estimated"])
        self.assertEqual(row, before)

    def test_estimate_preserves_quality_columns_and_qualification(self):
        missing = record("missing", raw=False, qualified=False, error=2.84, flags=["speed_avg"])
        plan = rr.build_plan([*self.calibration_rows(), missing])
        estimated = next(c for c in plan["changes"] if c["id"] == "missing")
        self.assertEqual(estimated["kind"], "estimated")
        self.assertEqual(set(estimated["fields"]), {"score", "tier", "trajectory_data"})
        saved = estimated["fields"]["trajectory_data"]
        self.assertNotIn("trajectory", saved)
        self.assertNotIn("strokes", saved)
        self.assertTrue(saved["meta"]["score"]["estimated"])
        self.assertEqual(saved["meta"]["score"]["method"], "avg-error-isotonic-v1")
        self.assertEqual(saved["meta"]["score"]["calibration"]["scope"], "template:apple")
        self.assertNotIn("f1_5", saved["meta"]["score"])
        self.assertFalse(missing["is_qualified"])

    def test_missing_source_without_calibration_is_reported_unscorable(self):
        plan = rr.build_plan([record("missing", raw=False)])
        self.assertEqual(plan["changes"], [])
        self.assertEqual(plan["summary"]["unscorable"], 1)
        self.assertIn("calibration", plan["unscorable"][0]["reason"])

    def test_versioned_rows_are_skipped_and_force_is_idempotent(self):
        row = record("done")
        first = rr.build_plan([row])["changes"][0]
        updated = {**row, **first["fields"]}
        self.assertEqual(rr.build_plan([updated])["summary"]["skipped_version"], 1)
        self.assertEqual(rr.build_plan([updated], force=True)["changes"], [])
        self.assertEqual(rr.build_plan([updated])["exact_personal_bests"]["count"], 1)

    def test_strokes_only_source_is_scored_without_inventing_saved_trajectory(self):
        row = record("strokes")
        del row["trajectory_data"]["trajectory"]
        plan = rr.build_plan([row])
        saved = plan["changes"][0]["fields"]["trajectory_data"]
        self.assertNotIn("trajectory", saved)
        self.assertEqual(saved["strokes"], row["trajectory_data"]["strokes"])
        self.assertEqual(saved["meta"]["score"]["method"], "raw-strokes-ma3x2")

    def test_invalid_stored_source_is_not_silently_estimated(self):
        row = record("invalid")
        row["trajectory_data"]["trajectory"] = [[float("nan"), .1, 1]]
        plan = rr.build_plan([*self.calibration_rows(), row])
        self.assertFalse(any(c["id"] == "invalid" for c in plan["changes"]))
        self.assertEqual(plan["summary"]["unscorable"], 1)


class ApplyTests(unittest.TestCase):
    def test_default_cli_preview_does_not_migrate_or_write_existing_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            with contextlib.closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("create table drawing_demos (id text, created_at text, trajectory_data text)")
            before = path.read_bytes()
            result = rr.main(["--sqlite", str(path), "--report", str(Path(tmp) / "preview.json")])
            self.assertEqual(result, 0)
            self.assertEqual(path.read_bytes(), before)

    def test_apply_requires_complete_backup_and_stores_sqlite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            connection = db.SqliteDB(str(Path(tmp) / "test.sqlite"))
            row = record("original")
            rid = connection.insert_demo({k: v for k, v in row.items() if k != "id"})
            rows = list(rr.read_rows(connection))
            plan = rr.build_plan(rows)
            backup_path = Path(tmp) / "before.jsonl.gz"
            applied = rr.apply_plan(connection, rows, plan, backup_path)
            self.assertEqual(applied, 1)
            with gzip.open(backup_path, "rt", encoding="utf-8") as stream:
                originals = [json.loads(line) for line in stream]
            self.assertEqual(originals[0]["id"], rid)
            self.assertEqual(originals[0]["score"], 100)
            after = list(rr.read_rows(connection))[0]
            self.assertEqual(after["trajectory_data"]["meta"]["score"]["score_version"], "b-v1")
            self.assertEqual(rr.build_plan([after])["changes"], [])

    def test_existing_backup_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            connection = db.SqliteDB(str(Path(tmp) / "test.sqlite"))
            connection.insert_demo(record("source"))
            rows = list(rr.read_rows(connection))
            path = Path(tmp) / "original.jsonl.gz"
            path.write_bytes(b"existing backup")
            with self.assertRaises(FileExistsError):
                rr.apply_plan(connection, rows, rr.build_plan(rows), path)
            self.assertEqual(path.read_bytes(), b"existing backup")
            self.assertNotIn("score", list(rr.read_rows(connection))[0]["trajectory_data"]["meta"])


if __name__ == "__main__":
    unittest.main()
