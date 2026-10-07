"""I0 direct validation respects six real 72-scene nodes after an early pilot cap."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.sac_improvement import ImprovementEnv, default_config
from satellite_rl.sac_improvement.agent import ImprovementSAC, pack_actions
from satellite_rl.sac_improvement import runner
from tests.test_s2p19_i0 import scene


def saved_node(directory, step):
    path = Path(directory) / "validation" / f"step_{step:09d}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env_step": step, "count": 72,
                                "cases": [{"sha256": "a" * 64}] * 72}), encoding="utf-8")


class I0ValidationBudgetTests(unittest.TestCase):
    def test_early_97112_node_keeps_200k_and_actual_terminal_skips_300k(self):
        with tempfile.TemporaryDirectory() as temp:
            for node in (0, 50000, 97112, 100000):
                saved_node(temp, node)
            self.assertTrue(runner._validation_due(temp, 200000))
            saved_node(temp, 200000)
            self.assertFalse(runner._validation_due(temp, 300000))
            self.assertTrue(runner._training_checkpoint_due(
                300000, 299999, 10000, default_config(), False, False))
            with patch.object(runner, "_validate_node", side_effect=AssertionError(
                    "300k must not allocate another 72 cases")):
                for _ in range(2):  # same 300k checkpoint after interrupted resume
                    self.assertTrue(runner._scheduled_validation(
                        Path(temp), {"run_id": "i0_same_id"}, {}, None, 300000,
                        "checkpoint300k", None))
            skip = Path(temp) / "validation_skips/step_000300000.json"
            record = json.loads(skip.read_text(encoding="utf-8"))
            self.assertEqual(record["checkpoint_sha256"], "checkpoint300k")
            self.assertEqual(record["saved_validation_steps_before_skip"],
                             [0, 50000, 97112, 100000, 200000])
            self.assertEqual(record["cases_generated"], 0)
            self.assertFalse((Path(temp) / "validation/step_000300000/summary.json").exists())
            self.assertTrue(runner._validation_due(temp, 350000, terminal=True))
            saved_node(temp, 350000)
            self.assertEqual(len(runner._saved_validation_nodes(temp)), 6)
            self.assertTrue(runner._validation_due(temp, 350000, terminal=True))
            with self.assertRaisesRegex(ValueError, "no remaining"):
                runner._validation_due(temp, 360000, terminal=True)

    def test_ordinary_exact_pilot_retains_original_six_nodes(self):
        with tempfile.TemporaryDirectory() as temp:
            for node in (0, 50000, 100000):
                saved_node(temp, node)
            self.assertTrue(runner._validation_due(temp, 200000))
            saved_node(temp, 200000)
            self.assertTrue(runner._validation_due(temp, 300000))
            saved_node(temp, 300000)
            self.assertTrue(runner._validation_due(temp, 400000, terminal=True))
            self.assertTrue(runner._validation_due(temp, 300000, terminal=True))
            saved_node(temp, 400000)
            self.assertEqual(runner._saved_validation_nodes(temp),
                             [0, 50000, 100000, 200000, 300000, 400000])

    def test_skip_rejects_partial_case_and_changes_no_rng_or_real_update(self):
        config = default_config()
        config.update(device="cpu", batch_size=4, warmup_steps=0)
        env = ImprovementEnv(config)
        env.reset(scene(16))
        state = env.state()
        features, mask, identities = pack_actions(env)
        next_state, reward, done, _ = env.step(identities[0])
        next_features, next_mask, _ = pack_actions(env)
        reference = ImprovementSAC(config)
        for _ in range(config["batch_size"]):
            reference.replay.add(state, features, mask, identities, 0, reward,
                                 next_state, next_features, next_mask, done)
        resumed = ImprovementSAC(config)
        resumed.load_state_dict(copy.deepcopy(reference.state_dict()))
        with tempfile.TemporaryDirectory() as temp:
            for node in (0, 50000, 97112, 100000, 200000):
                saved_node(temp, node)
            random.seed(31); np.random.seed(31); torch.manual_seed(31)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.get_rng_state().clone()
            with patch.object(runner, "_validate_node", side_effect=AssertionError(
                    "skip cannot call validation")):
                runner._scheduled_validation(Path(temp), {"run_id": "i0_same_id"}, config,
                                             resumed, 300000, "checkpoint300k", None)
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            torch.set_rng_state(torch_state.clone())
            original_update = reference.update()
            torch.set_rng_state(torch_state.clone())
            resumed_update = resumed.update()
            self.assertEqual(original_update, resumed_update)
            self.assertEqual(original_update["updates"], 1)
            for original, actual in zip(reference.actor.parameters(), resumed.actor.parameters()):
                self.assertTrue(torch.equal(original, actual))
            for original, actual in zip(reference.critic1.parameters(), resumed.critic1.parameters()):
                self.assertTrue(torch.equal(original, actual))
            partial = Path(temp) / "validation/step_000300000/scene_00000.json"
            partial.parent.mkdir(parents=True)
            partial.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "saved cases"):
                runner._record_validation_skip(Path(temp), {"run_id": "i0_same_id"},
                                               300000, "checkpoint300k")


if __name__ == "__main__":
    unittest.main()
