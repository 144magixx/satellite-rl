"""Small arithmetic checks for the S1-03-R1 action curve."""

import unittest

from satellite_rl.coverage.report_action_curve import accumulate


def action(step, operation, accepted):
    return {"env_step": step, "requested_action_fields": {"operation": operation},
            "accepted": accepted}


class ActionCurveTests(unittest.TestCase):
    def test_stop_is_excluded_from_acceptance_denominator(self):
        bins = accumulate([action(1, "assign", True), action(2, "assign", False),
                           action(3, "stop", False), action(4, "assign", True)],
                          budget=4, bin_size=2)
        self.assertEqual(bins[0]["acceptance_rate_non_stop"], .5)
        self.assertEqual(bins[1]["non_stop"], 1)
        self.assertEqual(bins[1]["accepted_non_stop"], 1)
        self.assertEqual(bins[1]["stop"], 1)

    def test_duplicate_or_missing_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "env_step"):
            accumulate([action(1, "assign", True), action(1, "assign", True)],
                       budget=2, bin_size=2)


if __name__ == "__main__":
    unittest.main()
