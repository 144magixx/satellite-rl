"""Read-only SAC learning snapshots for later Critic diagnosis."""

from copy import deepcopy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.envs.regions import profile_metadata
from satellite_rl.workbench.config import config_summary, default_config
from satellite_rl.workbench.learning_snapshots import FORMAT, save_learning_snapshot


class LearningSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = default_config()
        self.config["network"].update(hidden_sizes=[8], dropout=0.1)
        summary = config_summary(self.config)
        self.agent = build_agent(self.config, summary["state_dim"], summary["action_dims"], "cpu")
        self.provenance = {"run_id": "snapshot_fixture", "git_commit": "f" * 40,
                           "scene_sets": {}, "environment_profile": profile_metadata("china_legacy")}
        self.state_dim = summary["state_dim"]

    def add_rows(self, terminal_count, nonterminal_count, masked=False):
        action_mask = {"frequency_slots": np.ones((100, 10), dtype=np.bool_),
                       "power": np.ones(10, dtype=np.bool_)}
        for index in range(terminal_count + nonterminal_count):
            state = np.full(self.state_dim, index / 100, dtype=np.float32)
            next_state = state + 0.01
            terminal = index < terminal_count
            masks = (action_mask, None if terminal else action_mask) if masked else (None, None)
            self.agent.replay.add(state, (index % 100, index % 10, index % 10),
                                  index / 10, next_state, terminal, *masks)

    def test_snapshot_is_bounded_and_leaves_rng_weights_and_modes_unchanged(self):
        self.add_rows(90, 100)
        self.agent.actor.eval()
        self.agent.critic1.train()
        self.agent.update_step = 23
        networks = ("actor", "critic1", "critic2", "target1", "target2")
        weights_before = {name: {key: value.detach().clone() for key, value in
                                 getattr(self.agent, name).state_dict().items()} for name in networks}
        modes_before = {name: getattr(self.agent, name).training for name in networks}
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        agent_rng_before = self.agent.rng.getstate()
        replay_rng_before = self.agent.replay.rng.getstate()

        path = Path(save_learning_snapshot(self.directory, self.agent, self.config,
                                           self.provenance, 1200))
        self.assertTrue(path.is_file())
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(saved["format"], FORMAT)
        self.assertEqual(saved["env_step"], 1200)
        self.assertEqual(saved["update_step"], 23)
        self.assertEqual(saved["provenance"], self.provenance)
        self.assertEqual(saved["network_training_modes"], modes_before)
        self.assertEqual(set(saved["networks"]), set(networks))
        self.assertNotIn("optimizers", saved)
        self.assertNotIn("replay", saved)
        sample = saved["replay_sample"]
        self.assertEqual(sample["replay_size"], 190)
        self.assertEqual(len(sample["terminal_indices"]), 64)
        self.assertEqual(len(sample["nonterminal_indices"]), 64)
        self.assertEqual(len(sample["transitions"]), 128)
        self.assertEqual(len({row["replay_index"] for row in sample["transitions"]}), 128)
        self.assertEqual(sum(row["done"] for row in sample["transitions"]), 64)
        self.assertTrue(all(row["state"].device.type == "cpu" for row in sample["transitions"]))

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        if cuda_before is not None:
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(torch.cuda.get_rng_state_all(), cuda_before)))
        self.assertEqual(self.agent.rng.getstate(), agent_rng_before)
        self.assertEqual(self.agent.replay.rng.getstate(), replay_rng_before)
        self.assertEqual({name: getattr(self.agent, name).training for name in networks}, modes_before)
        for name in networks:
            for key, value in getattr(self.agent, name).state_dict().items():
                self.assertTrue(torch.equal(value, weights_before[name][key]))

        original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(save_learning_snapshot(self.directory, self.agent, self.config,
                                                 self.provenance, 1200), str(path))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_sha)
        with torch.no_grad():
            next(self.agent.critic1.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "differs"):
            save_learning_snapshot(self.directory, self.agent, self.config,
                                   self.provenance, 1200)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_sha)

    def test_masked_transition_masks_survive_snapshot_and_non_sac_is_skipped(self):
        self.add_rows(2, 3, masked=True)
        path = Path(save_learning_snapshot(self.directory, self.agent, self.config,
                                           self.provenance, 0))
        sample = torch.load(path, map_location="cpu", weights_only=True)["replay_sample"]
        self.assertEqual(len(sample["transitions"]), 5)
        for row in sample["transitions"]:
            self.assertEqual(set(row["action_mask"]), {"frequency_slots", "power"})
            self.assertEqual(tuple(row["action_mask"]["frequency_slots"].shape), (100, 10))
            if row["done"]:
                self.assertIsNone(row["next_action_mask"])
            else:
                self.assertIsNotNone(row["next_action_mask"])
        other = deepcopy(self.config)
        other["algorithm"] = "dqn"
        self.assertIsNone(save_learning_snapshot(self.directory, self.agent, other,
                                                 self.provenance, 1))
        self.assertFalse((self.directory / "learning_snapshots" / "step_000000001.pt").exists())


if __name__ == "__main__":
    unittest.main()
