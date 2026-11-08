"""Preview or explicitly apply B-v1 scores to every historical database record.

  python tools/rescore_records.py                  # GET/read only; writes a local preview
  python tools/rescore_records.py --dry-run --sqlite outputs/local_bench.sqlite
  python tools/rescore_records.py --apply          # full original-row backup BEFORE updates

Source trajectories are rescored without changing the original drawing. A record
without source data can receive an explicitly labeled avg_error-based estimate;
its measured quality columns and qualification remain unchanged. No record that
was excluded before this migration is admitted by it. No schema changes/deletes.
Backups contain private record data and stay in the ignored backups/ directory.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_secrets
import backup
import db as dbmod
import scoring
import submission

VERSION = "b-v1"
ESTIMATE_METHOD = "avg-error-isotonic-v1"
MIN_DISTINCT_ERRORS = 3


class ExistingSqliteDB(dbmod.SqliteDB):
    """Open an existing file without the application's constructor migrations."""

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        self.lock = threading.Lock()
        if not Path(self.path).is_file():
            raise ValueError("existing SQLite database required")

    def _rows(self, sql: str, args=()) -> list[dict]:
        uri = Path(self.path).as_uri() + "?mode=ro"
        with self.lock, contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, args)]


def read_rows(connection):
    """Read all columns/statuses using the existing full-backup paginated reader."""
    reader = backup.sqlite_rows if isinstance(connection, dbmod.SqliteDB) else backup.supabase_rows
    yield from reader(connection, "drawing_demos")


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def fit_calibration(samples: list[dict], scope: str) -> dict:
    """Least-squares decreasing isotonic fit, then linear interpolation of knots.

    Repeated avg_error values have equal weight per demonstration. Adjacent
    violations are pooled (PAVA); predictions outside observed errors clamp to
    the nearest endpoint rather than extrapolating unsupported performance.
    """
    ordered = sorted(samples, key=lambda r: (float(r["avg_error"]), str(r["id"])))
    grouped = defaultdict(list)
    for row in ordered:
        grouped[float(row["avg_error"])].append(float(row["score"]))
    blocks = []
    for error, scores in sorted(grouped.items()):
        blocks.append({"errors": [error], "total": sum(scores), "n": len(scores)})
        while len(blocks) > 1 and blocks[-2]["total"] / blocks[-2]["n"] < blocks[-1]["total"] / blocks[-1]["n"]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append({"errors": left["errors"] + right["errors"],
                           "total": left["total"] + right["total"], "n": left["n"] + right["n"]})
    knots = [[error, block["total"] / block["n"]] for block in blocks for error in block["errors"]]
    fingerprint = [{"id": str(r["id"]), "avg_error": float(r["avg_error"]), "score": float(r["score"])}
                   for r in ordered]
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"method": ESTIMATE_METHOD, "scope": scope, "samples": len(samples),
            "distinct_errors": len(grouped), "knots": knots, "samples_sha256": digest,
            "endpoint_policy": "clamp", "sample_policy": "existing-qualified-unflagged-source"}


def estimate_score(calibration: dict, error: float) -> float:
    knots = calibration["knots"]
    if not knots:
        raise ValueError("empty calibration")
    score = knots[0][1] if error <= knots[0][0] else knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= error <= x1:
            score = y0 + (y1 - y0) * (error - x0) / (x1 - x0)
            break
    return math.floor(min(100.0, max(0.0, score)) * 10 + 1e-7) / 10


def _trajectory(payload: dict) -> tuple[list | None, str]:
    if payload.get("trajectory"):
        trajectory, source = payload["trajectory"], "stored-trajectory"
    elif payload.get("strokes"):
        trajectory, source = submission.build_trajectory(payload["strokes"]), "raw-strokes-ma3x2"
    else:
        return None, "missing-source"
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("invalid source trajectory")
    for point in trajectory:
        if not isinstance(point, (list, tuple)) or len(point) != 3 or not all(_finite(v) and 0 <= v <= 1 for v in point):
            raise ValueError("invalid source trajectory")
    if not any(point[2] >= .5 for point in trajectory):
        raise ValueError("source contains no pen-down points")
    return trajectory, source


def _row_preview(row: dict) -> dict:
    keys = ("score", "tier", "avg_error", "f1", "coverage", "precision", "strokes", "is_qualified")
    return {key: row.get(key) for key in keys}


def _distribution(changes: list[dict]) -> dict:
    best = {}
    for change in changes:
        fields = change["fields"]
        if change["kind"] != "exact" or not fields["is_qualified"]:
            continue
        key = (change["template"], str(change.get("nickname") or change["id"]).casefold())
        best[key] = max(best.get(key, 0), fields["score"])
    scores = sorted(best.values())
    return {"policy": "exact-qualified-best-per-nickname-template", "count": len(scores), "scores": scores,
            "q90_nearest_rank": scores[max(0, math.ceil(.9 * len(scores)) - 1)] if scores else None,
            "max": scores[-1] if scores else None}


def build_plan(rows: list[dict], force: bool = False) -> dict:
    """Pure planning: no DB/file writes, source mutation, or invented measurements."""
    exact, missing, unscorable = {}, [], []
    clean_samples, by_template = [], defaultdict(list)
    for row in rows:
        payload = row.get("trajectory_data") or {}
        meta = payload.get("meta") or {} if isinstance(payload, dict) else {}
        try:
            if not isinstance(payload, dict) or not isinstance(meta, dict):
                raise ValueError("invalid stored payload")
            trajectory, source = _trajectory(payload)
            if trajectory is None:
                missing.append(row)
                continue
            measured = scoring.score_trajectory(row["template_name"], trajectory)
            if measured.get("score_version") != VERSION:
                raise RuntimeError("scorer must return score_version=b-v1")
            exact[row["id"]] = (measured, source)
            if row.get("is_qualified") and not meta.get("flags"):
                sample = {"id": row["id"], "avg_error": measured["avg_error"], "score": measured["score"]}
                clean_samples.append(sample)
                by_template[row["template_name"]].append(sample)
        except (ValueError, TypeError, KeyError, IndexError):
            unscorable.append({"id": row["id"], "template": row.get("template_name"), "reason": "invalid source or template"})

    calibrations = {}
    for template, samples in sorted(by_template.items()):
        model = fit_calibration(samples, "template:" + template)
        if model["distinct_errors"] >= MIN_DISTINCT_ERRORS:
            calibrations[template] = model
    pooled = fit_calibration(clean_samples, "all-templates")
    if pooled["distinct_errors"] < MIN_DISTINCT_ERRORS:
        pooled = None

    changes, skipped, unchanged = [], 0, 0
    for row in rows:
        payload = row.get("trajectory_data") or {}
        meta = payload.get("meta") or {} if isinstance(payload, dict) else {}
        old_score = meta.get("score") or {} if isinstance(meta, dict) else {}
        if not force and isinstance(old_score, dict) and old_score.get("score_version") == VERSION:
            skipped += 1
            continue
        if row["id"] in exact:
            measured, source = exact[row["id"]]
            # Existing false includes manual hides, pending review, and flags. Never re-admit.
            qualified = bool(row.get("is_qualified")) and bool(measured["qualified"])
            score_meta = {**measured, "qualified": qualified, "estimated": False, "method": source,
                          "qualification_policy": "preserve-existing-exclusions"}
            fields = {key: measured[key] for key in ("score", "tier", "f1", "coverage", "precision", "avg_error", "strokes")}
            fields["is_qualified"] = qualified
            kind = "exact"
        elif row in missing:
            error = row.get("avg_error")
            model = calibrations.get(row.get("template_name")) or pooled
            if not _finite(error) or error < 0 or model is None:
                unscorable.append({"id": row["id"], "template": row.get("template_name"),
                                   "reason": "missing finite avg_error or sufficient source calibration"})
                continue
            estimated = estimate_score(model, float(error))
            # Retain any legacy measured quality metadata; do not manufacture f1_5 or trajectories.
            score_meta = {**(old_score if isinstance(old_score, dict) else {}), "score": estimated,
                          "tier": scoring.tier_for(estimated)[0], "score_version": VERSION, "estimated": True,
                          "method": ESTIMATE_METHOD, "input_avg_error": float(error), "calibration": model,
                          "qualification_policy": "unchanged", "quality_metrics_version": "legacy-unchanged"}
            fields = {"score": estimated, "tier": score_meta["tier"]}
            kind = "estimated"
        else:
            continue
        saved = copy.deepcopy(payload)
        saved.setdefault("meta", {})["score"] = score_meta
        fields["trajectory_data"] = saved
        if all(row.get(key) == value for key, value in fields.items()):
            unchanged += 1
            continue
        changes.append({"id": row["id"], "nickname": row.get("nickname"), "template": row.get("template_name"),
                        "kind": kind, "before": _row_preview(row), "fields": fields})

    counts = Counter(change["kind"] for change in changes)
    exact_population = [{"id": row["id"], "nickname": row.get("nickname"), "template": row.get("template_name"),
                         "kind": "exact", "fields": {"score": exact[row["id"]][0]["score"],
                         "is_qualified": bool(row.get("is_qualified")) and exact[row["id"]][0]["qualified"]}}
                        for row in rows if row["id"] in exact]
    return {"score_version": VERSION, "summary": {"total": len(rows), "exact": counts["exact"],
            "estimated": counts["estimated"], "unscorable": len(unscorable), "skipped_version": skipped,
            "unchanged": unchanged, "changes": len(changes)}, "changes": changes, "unscorable": unscorable,
            "calibrations": calibrations, "pooled_calibration": pooled, "exact_personal_bests": _distribution(exact_population)}


def _write_snapshot(rows: list[dict], path: Path) -> None:
    """Exclusive, fsynced complete snapshot; no write is allowed before this finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as output:
            created = True
            with gzip.GzipFile(fileobj=output, mode="wb") as compressed:
                for row in rows:
                    compressed.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        with gzip.open(path, "rt", encoding="utf-8") as saved:
            restored = [json.loads(line) for line in saved]
        if restored != rows:
            raise ValueError("backup verification failed")
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def apply_plan(connection, rows: list[dict], plan: dict, backup_path: Path) -> int:
    """Back up every original row, then journal updates. Explicit caller action only."""
    _write_snapshot(rows, backup_path)
    count = 0
    journal = backup_path.with_name(backup_path.name + ".journal.jsonl")
    with journal.open("x", encoding="utf-8", newline="\n") as output:
        for change in plan["changes"]:
            fields = copy.deepcopy(change["fields"])
            if isinstance(connection, dbmod.SqliteDB):
                fields["trajectory_data"] = json.dumps(fields["trajectory_data"], ensure_ascii=False)
            connection.update_demo(change["id"], fields)
            output.write(json.dumps({"id": change["id"], "kind": change["kind"], "status": "applied"}) + "\n")
            output.flush()
            os.fsync(output.fileno())
            count += 1
    return count


def write_report(plan: dict, path: Path) -> None:
    """Reviewable scores/provenance only; omit raw drawing, owner, ticket, and IP data."""
    report = {key: value for key, value in plan.items() if key != "changes"}
    report["changes"] = [{key: value for key, value in change.items() if key != "fields"} |
                         {"after": _row_preview({**change["before"], **change["fields"]}),
                          "score_provenance": change["fields"]["trajectory_data"]["meta"]["score"]}
                         for change in plan["changes"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="B-v1 historical score preview (default) or backup-first explicit update")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="read-only database preview (default)")
    mode.add_argument("--apply", action="store_true", help="back up all original rows then update scores")
    parser.add_argument("--sqlite", help="use this existing local SQLite database")
    parser.add_argument("--force", action="store_true", help="recompute records already marked b-v1")
    parser.add_argument("--report", default=str(collect_secrets.ROOT / "outputs" / "rescore-b-v1-preview.json"))
    parser.add_argument("--backup-dir", default=str(collect_secrets.ROOT / "backups"))
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        connection = ExistingSqliteDB(args.sqlite) if args.sqlite else backup.connect(args)
        rows = list(read_rows(connection))
        plan = build_plan(rows, force=args.force)
        write_report(plan, Path(args.report))
        print("[rescore] " + ("APPLY" if args.apply else "DRY RUN (database unchanged)") + ": " +
              json.dumps(plan["summary"], ensure_ascii=False))
        print(f"[rescore] preview: {backup.show(Path(args.report))}")
        distribution = plan["exact_personal_bests"]
        print(f"[rescore] exact qualified personal bests: {distribution['count']}; q90={distribution['q90_nearest_rank']}")
        if args.apply and plan["changes"]:
            stamp = dt.datetime.now(dbmod.KST).strftime("%Y%m%d-%H%M%S")
            path = Path(args.backup_dir) / f"rescore_{VERSION}_before_{stamp}_{uuid.uuid4().hex[:8]}.jsonl.gz"
            count = apply_plan(connection, rows, plan, path)
            print(f"[rescore] applied {count}; full original snapshot: {backup.show(path)}")
        return 0
    except (dbmod.DBError, OSError, ValueError, RuntimeError) as error:
        # Connection settings/service response bodies must not enter logs.
        print(f"[rescore] stopped: {type(error).__name__}; no further updates attempted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
