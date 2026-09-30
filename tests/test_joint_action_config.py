"""J action structures are frozen, validated and distinguishable from old models."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.workbench.checkpoints import (config_hash, export_policy, load_checkpoint,
                                                load_policy, save_checkpoint)
from satellite_rl.workbench.config import config_summary, default_config, validate_config
from satellite_rl.workbench.service import Application


def j_config(policy="factorized", critic="additive"):
    config = default_config()
    config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0,
                             sac_policy_structure=policy, sac_critic_structure=critic)
    config["state"].update(context_level="compact",
                           features=["beam", "power", "interference", "occupancy",
                                     "natural_global_context"])
    config["environment"].update(profile="europe_v1", action_mask="joint_feasible")
    config["training"].update(scene_order="shuffle_each_cycle", total_steps=400000)
    config["evaluation"]["extra_steps"] = [0, 2000, 5000]
    return validate_config(config)


class JointActionConfigTests(unittest.TestCase):
    def test_four_structures_validate_and_reach_service_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(directory)
            for policy, critic in (("factorized", "additive"), ("factorized", "joint"),
                                   ("conditional", "additive"), ("conditional", "joint")):
                with self.subTest(policy=policy, critic=critic):
                    config = j_config(policy, critic)
                    preview = app.api("POST", ["validate"], {}, config)
                    self.assertEqual(preview["summary"]["state_dim"], 207)
                    self.assertEqual(preview["summary"]["sac_policy_structure"], policy)
                    self.assertEqual(preview["summary"]["sac_critic_structure"], critic)
                    self.assertEqual(preview["summary"]["evaluation_extra_steps"], [0, 2000, 5000])
                    self.assertEqual(preview["config"]["network"]["sac_policy_structure"], policy)
                    self.assertEqual(preview["config"]["network"]["sac_critic_structure"], critic)
                    self.assertGreater(preview["model"]["parameter_count"], 0)
                    self.assertIn("策略", preview["model"]["branches"])
                    self.assertIn("价值网络1", preview["model"]["branches"])
                    self.assertEqual(config_summary(config)["state_dim"], 207)

    def test_nondefault_requires_masked_sac(self):
        for structure in ("sac_policy_structure", "sac_critic_structure"):
            config = j_config()
            config["network"][structure] = ("conditional" if structure == "sac_policy_structure"
                                             else "joint")
            for section, key, value in (("environment", "action_mask", "none"),
                                        ("algorithm", None, "dqn")):
                bad = copy.deepcopy(config)
                if key is None:
                    bad[section] = value
                else:
                    bad[section][key] = value
                with self.subTest(structure=structure, change=(section, key)):
                    with self.assertRaisesRegex(ValueError, "SAC.*掩码"):
                        validate_config(bad)

    def test_extra_steps_strict_and_default_hash_compatible(self):
        config = j_config()
        for bad_steps in ([-1], [2000, 0], [0, 0], [400001], [0.0], [True], "0,2000"):
            bad = copy.deepcopy(config)
            bad["evaluation"]["extra_steps"] = bad_steps
            with self.subTest(steps=bad_steps), self.assertRaisesRegex(ValueError, "额外验证步数"):
                validate_config(bad)
        old = default_config()
        old["network"].pop("sac_policy_structure")
        old["network"].pop("sac_critic_structure")
        old["evaluation"].pop("extra_steps")
        self.assertEqual(config_hash(old), config_hash(default_config()))
        hashes = {config_hash(j_config(policy, critic)) for policy, critic in
                  (("factorized", "additive"), ("factorized", "joint"),
                   ("conditional", "additive"), ("conditional", "joint"))}
        self.assertEqual(len(hashes), 4)
        no_early = j_config()
        no_early["evaluation"]["extra_steps"] = []
        self.assertNotEqual(config_hash(config), config_hash(no_early))

    def test_policy_and_checkpoint_reject_metadata_mismatch(self):
        config = default_config()
        config["network"].update(kind="mlp", hidden_sizes=[8],
                                 sac_policy_structure="conditional", sac_critic_structure="joint")
        config["environment"]["action_mask"] = "joint_feasible"
        config["evaluation"]["extra_steps"] = [0, 2]
        config["training"]["total_steps"] = 5
        config = validate_config(config)
        agent = build_agent(config, 204, [100, 10, 10], "cpu")
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            export_policy(model_dir, agent, config, {"purpose": "unit_test"})
            metadata_path = model_dir / "model.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual((metadata["sac_policy_structure"], metadata["sac_critic_structure"]),
                             ("conditional", "joint"))
            self.assertEqual(metadata["evaluation_extra_steps"], [0, 2])
            self.assertEqual(load_policy(model_dir).config["network"]["sac_critic_structure"], "joint")
            metadata["sac_critic_structure"] = "additive"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action structures"):
                load_policy(model_dir)

            checkpoint_path = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint_path, agent, {"env_step": 1}, config, {"boundary": "next_episode"})
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            payload["sac_policy_structure"] = "factorized"
            torch.save(payload, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "SAC action structures"):
                load_checkpoint(checkpoint_path, agent, config)


if __name__ == "__main__":
    unittest.main()
