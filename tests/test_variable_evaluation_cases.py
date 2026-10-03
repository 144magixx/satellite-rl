"""S2-P17 completed allocations are immutable and tied to frozen inputs."""

import json
import tempfile
import unittest

from satellite_rl.workbench.variable_evaluation import VariableEvaluationCases


class VariableEvaluationCasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.contract = {"experiment_id": "S2-P17", "run_id": "test",
                         "scene_sha256": "a" * 64, "action_seed_base": 53001}
        self.store = VariableEvaluationCases(self.temp.name, self.contract)
        self.expected = {"kind": "model", "scene_index": 0,
                         "scene_sha256": "a" * 64, "beam_count": 100,
                         "repeat_index": 0, "action_seed": 53001}
        self.result = {"scene": {"source_sha256": "a" * 64},
                       "beam_count": 100, "repeat_index": 0,
                       "action_seed": 53001, "metrics": {"final_si_mean": 0.5},
                       "beams": [], "trace": []}

    def test_save_reuse_and_refuse_second_write(self):
        self.assertIsNone(self.store.load("model_00000_00", self.expected))
        self.store.save("model_00000_00", self.expected, self.result)
        self.assertEqual(self.store.load("model_00000_00", self.expected), self.result)
        with self.assertRaises(ValueError):
            self.store.save("model_00000_00", self.expected, self.result)
        self.assertEqual(list(self.store.require_all([("model_00000_00", self.expected)]))[0][1],
                         self.result)

    def test_reject_tamper_or_changed_case_identity(self):
        self.store.save("model_00000_00", self.expected, self.result)
        with self.assertRaises(ValueError):
            self.store.load("model_00000_00", {**self.expected, "action_seed": 53002})
        path = self.store.path("model_00000_00")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["body"]["result"]["metrics"]["final_si_mean"] = 0.9
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.load("model_00000_00", self.expected)

    def test_reject_changed_contract_and_unknown_case(self):
        with self.assertRaises(ValueError):
            VariableEvaluationCases(self.temp.name,
                                    {**self.contract, "action_seed_base": 53002})
        self.store.save("model_00000_00", self.expected, self.result)
        with self.assertRaises(ValueError):
            list(self.store.require_all([]))
        with self.assertRaises(ValueError):
            self.store.path("../outside")


if __name__ == "__main__":
    unittest.main()
