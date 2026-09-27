"""Regional physics and input semantics cannot be silently swapped in a policy."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.workbench.checkpoints import (
    config_hash, export_policy, load_checkpoint, load_policy, save_checkpoint,
)
from satellite_rl.workbench.config import config_summary, default_config, validate_config


class ProfileCheckpointTests(unittest.TestCase):
    def config(self, profile="china_legacy"):
        config = default_config()
        config["environment"]["profile"] = profile
        config["network"].update(kind="mlp", hidden_sizes=[8])
        return validate_config(config)

    def agent(self, config):
        summary = config_summary(config)
        return build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")

    def test_old_china_hash_and_package_remain_loadable(self):
        config = self.config()
        historical = copy.deepcopy(config)
        historical["environment"].pop("profile")
        self.assertEqual(config_hash(historical), config_hash(config))
        with tempfile.TemporaryDirectory() as tmp:
            export_policy(tmp, self.agent(config), historical, {})
            path = Path(tmp) / "model.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata.pop("environment_profile")
            path.write_text(json.dumps(metadata), encoding="utf-8")
            loaded = load_policy(tmp, config=config)
            self.assertEqual(loaded.config["environment"].get("profile", "china_legacy"), "china_legacy")

    def test_cross_profile_load_and_metadata_tampering_rejected(self):
        china, europe = self.config(), self.config("europe_v1")
        self.assertNotEqual(config_hash(china), config_hash(europe))
        with tempfile.TemporaryDirectory() as tmp:
            export_policy(tmp, self.agent(europe), europe, {})
            self.assertEqual(load_policy(tmp, config=europe).config["environment"]["profile"], "europe_v1")
            with self.assertRaisesRegex(ValueError, "config differs"):
                load_policy(tmp, config=china)
            path = Path(tmp) / "model.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["environment_profile"]["satellite_longitude_deg"] = 122.2
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profile metadata"):
                load_policy(tmp)
            metadata.pop("environment_profile")
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "profile metadata missing"):
                load_policy(tmp)

    def test_checkpoint_recovery_keeps_profile_and_rejects_mismatch(self):
        china, europe = self.config(), self.config("europe_v1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, self.agent(europe), {"env_step": 7}, europe, {"scene": "europe"})
            progress, environment = load_checkpoint(path, self.agent(europe), europe)
            self.assertEqual((progress["env_step"], environment["scene"]), (7, "europe"))
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                load_checkpoint(path, self.agent(china), china)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["environment_profile"]["satellite_longitude_deg"] = 122.2
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "profile metadata"):
                load_checkpoint(path, self.agent(europe), europe)
            payload.pop("environment_profile")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "profile metadata missing"):
                load_checkpoint(path, self.agent(europe), europe)


if __name__ == "__main__":
    unittest.main()
