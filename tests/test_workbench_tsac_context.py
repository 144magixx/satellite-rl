"""The 207-state context layout extends T-SAC without changing the old layout."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.agents.workbench_networks import Encoder, MultiHeadNetwork, model_description
from satellite_rl.workbench.checkpoints import export_policy, load_policy
from satellite_rl.workbench.config import config_summary, default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv


def tsac_config(layout):
    return {"algorithm": "sac", "network": {
        "kind": "transformer", "transformer_layout": layout,
        "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
        "feedforward_dim": 512, "activation": "relu", "dropout": 0,
    }}


def workbench_config(layout):
    config = default_config()
    config["network"].update(tsac_config(layout)["network"])
    config["environment"].update(profile="europe_v1", action_mask="joint_feasible")
    if layout == "tsac_14_context":
        config["state"].update(features=["beam", "power", "interference", "occupancy",
                                         "beta", "progress", "remaining_demand"],
                               demand_visibility="known")
    return validate_config(config)


class TsacContextTests(unittest.TestCase):
    def test_old_layout_weights_and_forward_are_unchanged(self):
        config = tsac_config("tsac_13_tokens")
        torch.manual_seed(51001)
        old = MultiHeadNetwork(config, 204, [100, 10, 10]).eval()
        state = torch.randn(2, 204)
        with torch.no_grad():
            expected = old(state)
        weights = copy.deepcopy(old.state_dict())

        # Constructing the new variant must not alter the old module's keys or
        # its strict same-version loading behavior.
        _ = MultiHeadNetwork(tsac_config("tsac_14_context"), 207, [100, 10, 10])
        restored = MultiHeadNetwork(config, 204, [100, 10, 10]).eval()
        restored.load_state_dict(weights, strict=True)
        self.assertEqual(set(restored.state_dict()), set(weights))
        self.assertFalse(any("context" in key for key in weights))
        with torch.no_grad():
            actual = restored(state)
        for output, reference in zip(actual, expected):
            self.assertTrue(torch.equal(output, reference))

    def test_new_token_receives_three_features_and_changes_output(self):
        torch.manual_seed(51001)
        encoder = Encoder(tsac_config("tsac_14_context"), 207).eval()
        state = torch.zeros(1, 207)
        changed = state.clone()
        changed[0, 204:207] = torch.tensor([0.6, 0.5, 0.25])
        seen = []
        hook = encoder.context_embed.register_forward_pre_hook(
            lambda _, args: seen.append(args[0].clone()))
        try:
            with torch.no_grad():
                original_output = encoder(state)
                changed_output = encoder(changed)
        finally:
            hook.remove()
        self.assertEqual(tuple(original_output.shape), (1, 128))
        self.assertEqual(tuple(seen[0].shape), (1, 3))
        torch.testing.assert_close(seen[0], torch.zeros(1, 3))
        torch.testing.assert_close(seen[1], changed[:, 204:207])
        self.assertFalse(torch.allclose(original_output, changed_output))

    def test_variant_trace_parameter_delta_and_shape_guards(self):
        old = model_description(tsac_config("tsac_13_tokens"), 204, [100, 10, 10])
        new = model_description(tsac_config("tsac_14_context"), 207, [100, 10, 10])
        self.assertEqual(new["variant"], "workbench_sac_transformer_tsac_14_context_v1")
        self.assertEqual(new["parameter_count"] - old["parameter_count"], 3200)
        layers = new["branches"]["策略"]
        token = next(layer for layer in layers if layer["type"] == "TokenAssembly")
        attention = next(layer for layer in layers if layer["type"] == "MultiheadAttention")
        self.assertEqual(token["input_shape"], [207])
        self.assertEqual(token["shape"], [14, 128])
        self.assertEqual(attention["input_shape"], [14, 128])
        self.assertEqual(attention["shape"], [14, 128])
        with self.assertRaisesRegex(ValueError, "207"):
            Encoder(tsac_config("tsac_14_context"), 204)
        with self.assertRaisesRegex(ValueError, "207"):
            Encoder(tsac_config("tsac_14_context"), 207)(torch.zeros(1, 204))

    def test_new_layout_rejects_old_weights_and_preserves_common_initialization(self):
        torch.manual_seed(51001)
        old = MultiHeadNetwork(tsac_config("tsac_13_tokens"), 204, [100, 10, 10])
        torch.manual_seed(51001)
        new = MultiHeadNetwork(tsac_config("tsac_14_context"), 207, [100, 10, 10])
        old_weights = old.state_dict()
        new_weights = new.state_dict()
        self.assertEqual(set(new_weights) - set(old_weights),
                         {"encoder.context_embed.weight", "encoder.context_embed.bias",
                          "encoder.context_pos_embed"})
        for key, old_value in old_weights.items():
            self.assertTrue(torch.equal(old_value, new_weights[key]), key)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            new.load_state_dict(old_weights, strict=True)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            old.load_state_dict(new_weights, strict=True)

    def test_config_and_real_environment_feature_order(self):
        config = workbench_config("tsac_14_context")
        summary = config_summary(config)
        self.assertEqual((summary["state_dim"], summary["network_token_count"]), (207, 14))
        self.assertEqual([(part["id"], part["offset"]) for part in summary["state_layout"]],
                         [("beam", 0), ("power", 3), ("interference", 4),
                          ("occupancy", 104), ("beta", 204), ("progress", 205),
                          ("remaining_demand", 206)])
        environment = WorkbenchEnv(config)
        state = environment.reset(seed=103)
        demands = environment.scene()["rate"].reshape(-1)
        self.assertEqual(state.shape, (207,))
        self.assertAlmostEqual(float(state[204]), float(demands.mean() / 6e8), places=6)
        self.assertEqual(float(state[205]), 0)
        self.assertAlmostEqual(float(state[206]), float(demands.sum() / (100 * 6e8)), places=6)
        state_after, _, done, _ = environment.step([0, 0, 0])
        self.assertFalse(done)
        self.assertAlmostEqual(float(state_after[205]), 0.01, places=6)
        self.assertAlmostEqual(float(state_after[206]), float(demands[1:].sum() / (100 * 6e8)), places=6)

    def test_future_demand_counterexample_distinguished_by_new_state(self):
        config = workbench_config("tsac_14_context")
        environment = WorkbenchEnv(config)
        first = environment.reset(seed=103)
        scene = environment.scene()
        changed = {key: value.copy() for key, value in scene.items()}
        changed["rate"][1:] *= 0.5
        second = environment.reset(scene=changed)
        np.testing.assert_array_equal(first[:204], second[:204])
        self.assertEqual(float(first[205]), float(second[205]))
        self.assertNotEqual(float(first[204]), float(second[204]))
        self.assertNotEqual(float(first[206]), float(second[206]))

    def test_policy_packages_roundtrip_both_layouts_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            for layout, dim in (("tsac_13_tokens", 204), ("tsac_14_context", 207)):
                with self.subTest(layout=layout):
                    config = workbench_config(layout)
                    summary = config_summary(config)
                    self.assertEqual(summary["state_dim"], dim)
                    agent = build_agent(config, dim, summary["action_dims"], "cpu")
                    path = Path(tmp) / layout
                    export_policy(path, agent, config, {"kind": "unit_test"})
                    loaded = load_policy(path, config=config)
                    self.assertIn(layout, loaded.metadata["variant"])
                    for key, value in agent.policy_network().state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.agent.policy_network().state_dict()[key]))
                    metadata_path = path / "model.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["variant"] = "workbench_sac_transformer_wrong_v1"
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "variant"):
                        load_policy(path)


if __name__ == "__main__":
    unittest.main()
