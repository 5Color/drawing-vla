"""B scoring: both tolerances affect points; collection eligibility stays loose."""
from __future__ import annotations

import math
import unittest
from unittest import mock

import scoring


def horizontal_parts(*ys):
    """Known parallel lines, with target samples spaced 0.004 apart."""
    return tuple(tuple((0.2 + k * 0.004, y) for k in range(151)) for y in ys)


def trajectory(lines):
    out = []
    for line in lines:
        out.append([*line[0], 0])
        out.extend([x, y, 1] for x, y in line)
        out.append([*line[-1], 0])
    return out


class TestBScoring(unittest.TestCase):
    def score_lines(self, target, drawn):
        # Replace only the template input; all geometry and scoring remain real.
        with mock.patch.object(scoring, "template_parts", return_value=target):
            return scoring.score_trajectory("fixture", trajectory(drawn))

    def test_exact_template_lines_score_100_at_both_tolerances(self):
        for name in scoring.load_templates():
            with self.subTest(template=name):
                result = scoring.score_trajectory(name, trajectory(scoring.template_parts(name)))
                self.assertEqual(result["score"], 100.0)
                self.assertEqual((result["f1_10"], result["f1_5"]), (1.0, 1.0))
                self.assertEqual(result["avg_error"], 0.0)
                self.assertEqual(result["score_version"], "b-v1")
                self.assertTrue(result["qualified"])

    def test_between_tolerances_lowers_score_and_fails_qualification(self):
        target = horizontal_parts(0.5)
        # 0.015 = 7.68px: every point passes nominal 10px and fails nominal 5px.
        result = self.score_lines(target, horizontal_parts(0.515))
        self.assertEqual(result["score"], 50.0)
        self.assertEqual((result["f1_10"], result["f1_5"]), (1.0, 0.0))
        self.assertEqual((result["f1"], result["coverage"], result["precision"]), (1.0, 1.0, 1.0))
        self.assertEqual(result["avg_error"], 7.68)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "70점 미만")

    def test_low_coverage_and_low_precision_fail_qualification(self):
        target = horizontal_parts(0.25, 0.5, 0.75, 0.85)
        # Draw only 2 of 4 lines (coverage ~ 50% < 75%)
        low_cov = self.score_lines(target, horizontal_parts(0.25, 0.5))
        self.assertFalse(low_cov["qualified"])
        self.assertEqual(low_cov["reason"], "도안 채움률 75% 미만")

        # Draw 1 target line + 2 extra doodle lines outside (precision ~ 33% < 75%)
        one_target = horizontal_parts(0.5)
        low_prec = self.score_lines(one_target, horizontal_parts(0.5, 0.1, 0.9))
        self.assertFalse(low_prec["qualified"])
        self.assertEqual(low_prec["reason"], "도안 정밀도 75% 미만")

    def test_partial_narrow_agreement_uses_arithmetic_mean_and_truncation(self):
        target = horizontal_parts(0.25, 0.5, 0.75)
        result = self.score_lines(target, horizontal_parts(0.25, 0.5, 0.765))
        # Loose F1=1, narrow F1=2/3, so 100*(1+2/3)/2 truncates to 83.3.
        self.assertEqual(result["score"], 83.3)
        self.assertEqual((result["f1_10"], result["f1_5"]), (1.0, 0.6667))
        self.assertTrue(result["qualified"])

    def test_five_pixel_tolerance_is_half_the_existing_loose_tolerance(self):
        target = horizontal_parts(0.5)
        close = self.score_lines(target, horizontal_parts(0.5075))
        far = self.score_lines(target, horizontal_parts(0.525))
        self.assertEqual((close["score"], close["f1_5"]), (100.0, 1.0))
        self.assertEqual((far["score"], far["f1_10"], far["f1_5"]), (0.0, 0.0, 0.0))
        self.assertFalse(far["qualified"])
        self.assertEqual(far["avg_error"], 10.24)

    def test_perfect_score_still_requires_normal_stroke_count(self):
        target = horizontal_parts(0.5)
        result = self.score_lines(target, target * 5)
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["strokes"], 5)
        self.assertFalse(result["qualified"])
        self.assertIn("획 수", result["reason"])

    def test_score_is_independent_of_stroke_order_and_direction(self):
        target = horizontal_parts(0.25, 0.5, 0.75)
        drawn = horizontal_parts(0.25, 0.5, 0.765)
        forward = self.score_lines(target, drawn)
        reverse = self.score_lines(target, tuple(line[::-1] for line in drawn[::-1]))
        for key in ("score", "f1", "f1_10", "f1_5", "coverage", "precision", "qualified"):
            self.assertEqual(forward[key], reverse[key], key)

    def test_missing_part_fails_qualification_even_with_high_total_coverage(self):
        # 3 parts: main body (80 pts), part2 (10 pts), part3 (10 pts)
        p1 = tuple((0.2 + k * 0.004, 0.2) for k in range(80))
        p2 = tuple((0.2 + k * 0.004, 0.4) for k in range(10))
        p3 = tuple((0.2 + k * 0.004, 0.6) for k in range(10))
        target = (p1, p2, p3)
        # Draw p1 only (global cov = 80/100 = 80% >= 75%, but p2, p3 are 0% < 50%)
        result = self.score_lines(target, (p1,))
        self.assertFalse(result["qualified"])
        self.assertIn("도안 일부 미완성", result["reason"])

    def test_empty_pen_up_and_dense_off_target_input_stays_bounded(self):
        for traj in ([], [[0.5, 0.5, 0]], [[0.0, 0.0, 1]] * 8000):
            with self.subTest(points=len(traj)):
                result = scoring.score_trajectory("apple", traj)
                for key in ("f1", "f1_10", "f1_5", "coverage", "precision"):
                    self.assertTrue(math.isfinite(result[key]), key)
                    self.assertEqual(result[key], 0.0)
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["avg_error"], 10.24)
                self.assertFalse(result["qualified"])


if __name__ == "__main__":
    unittest.main()
