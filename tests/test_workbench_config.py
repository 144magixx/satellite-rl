import math
import unittest

from satellite_rl.workbench.config import catalog, config_summary, default_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_default_and_layout(self):
        config = validate_config({})
        self.assertEqual(config, default_config())
        summary = config_summary(config)
        self.assertEqual(summary["state_dim"], 204)
        self.assertEqual(summary["action_dims"], [100, 10, 10])
        self.assertEqual([part["offset"] for part in summary["state_layout"]], [0, 3, 4, 104])
        self.assertEqual(config["environment"]["action_mask"], "none")
        self.assertEqual(summary["action_mask"], "none")
        self.assertEqual(config["network"]["transformer_layout"], "scalar_tokens")

    def test_tsac_layout_preset_and_backward_compatibility(self):
        legacy = default_config()
        del legacy["network"]["transformer_layout"]
        self.assertEqual(validate_config(legacy)["network"]["transformer_layout"], "scalar_tokens")
        preset = catalog()["network_presets"][0]
        self.assertEqual(preset["id"], "tsac_13_tokens")
        self.assertEqual(preset["algorithm"], "sac")
        self.assertEqual(preset["state_features"], ["beam", "power", "interference", "occupancy"])
        self.assertEqual(preset["network"], {
            "kind": "transformer", "transformer_layout": "tsac_13_tokens",
            "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
            "feedforward_dim": 512, "activation": "relu", "dropout": 0.1})
        raw = {"algorithm": preset["algorithm"], "state": {"features": preset["state_features"]},
               "network": preset["network"], "environment": {"action_mask": "joint_feasible"}}
        config = validate_config(raw)
        summary = config_summary(config)
        self.assertEqual(summary["state_dim"], 204)
        self.assertEqual(summary["transformer_layout"], "tsac_13_tokens")
        self.assertEqual(summary["network_token_count"], 13)
        self.assertIn("10个频率槽令牌", summary["network_layout_description"])
        self.assertEqual(validate_config(config), config)
        scalar = config_summary({"network": {"kind": "transformer"}})
        self.assertEqual(scalar["network_token_count"], 204)
        self.assertIn("每个标量", scalar["network_layout_description"])

    def test_tsac_layout_rejects_wrong_state_or_algorithm(self):
        base = {"network": {"kind": "transformer", "transformer_layout": "tsac_13_tokens"}}
        for change in ({"algorithm": "ppo"}, {"algorithm": "dqn"},
                       {"network": {"kind": "mlp"}},
                       {"state": {"features": ["power", "beam", "interference", "occupancy"]}},
                       {"state": {"features": ["beam", "power", "interference"]}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "13令牌"):
                raw = {"network": dict(base["network"])}
                for key, value in change.items():
                    raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
                validate_config(raw)
        with self.assertRaisesRegex(ValueError, "输入布局"):
            validate_config({"network": {"transformer_layout": "unknown"}})

    def test_joint_action_mask_config_and_legacy_default(self):
        legacy = default_config()
        del legacy["environment"]["action_mask"]
        self.assertEqual(validate_config(legacy)["environment"]["action_mask"], "none")
        for network in ("mlp", "cnn", "transformer"):
            with self.subTest(network=network):
                config = validate_config({"network": {"kind": network},
                                          "environment": {"action_mask": "joint_feasible"}})
                self.assertEqual(config_summary(config)["action_mask"], "joint_feasible")
                self.assertEqual(validate_config(config), config)
        choices = catalog()["action_masks"]
        self.assertEqual([item["id"] for item in choices], ["none", "joint_feasible"])
        self.assertEqual(choices[1]["algorithms"], ["sac"])

    def test_joint_action_mask_rejects_unsupported_combinations(self):
        invalid = [
            ({"algorithm": "ppo"}, "SAC"),
            ({"algorithm": "dqn"}, "SAC"),
            ({"environment": {"action_mode": "feasible_projection"}}, "投影"),
            ({"environment": {"power_budget_w": 499}}, "100束"),
            ({"training": {"target_entropy": -0.1}}, "目标熵"),
            ({"training": {"target_entropy": math.log(10000) + 0.01}}, "目标熵"),
        ]
        for change, message in invalid:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                raw = {"environment": {"action_mask": "joint_feasible"}}
                for key, value in change.items():
                    raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
                validate_config(raw)
        with self.assertRaisesRegex(ValueError, "掩码"):
            validate_config({"environment": {"action_mask": "unexpected"}})
        valid = validate_config({"environment": {"action_mask": "joint_feasible", "power_budget_w": 500},
                                 "training": {"target_entropy": math.log(10000)}})
        self.assertEqual(valid["environment"]["power_budget_w"], 500)

    def test_feature_order_and_visibility(self):
        config = validate_config({"state": {"features": ["progress", "beam", "other_occupancy"]}})
        self.assertEqual(config_summary(config)["state_dim"], 704)
        self.assertEqual([p["id"] for p in config_summary(config)["state_layout"]],
                         ["progress", "beam", "other_occupancy"])
        with self.assertRaises(ValueError):
            validate_config({"state": {"features": ["beta"], "demand_visibility": "online"}})
        for fid in ("remaining_demand", "remaining_demand_mean"):
            with self.subTest(feature=fid), self.assertRaises(ValueError):
                validate_config({"state": {"features": [fid], "demand_visibility": "online"}})

    def test_online_sgm_warning(self):
        config = validate_config({"state": {"demand_visibility": "online"},
                                  "reward": {"preset": "custom", "terms": [
                                      {"id": "sgm_delta", "weight": 1, "scale": 1, "timing": "step"}]}})
        self.assertTrue(any("在线" in text for text in config_summary(config)["warnings"]))

    def test_sac_entropy_out_of_range_warns(self):
        config = validate_config({"training": {"target_entropy": -6.9}})
        self.assertTrue(any("目标熵" in text for text in config_summary(config)["warnings"]))

    def test_strict_validation(self):
        bad = [
            {"unknown": 1}, {"network": {"hidden_sizes": []}},
            {"reward": {"preset": "custom", "terms": []}},
            {"environment": {"power_levels_w": [5, 5]}},
            {"training": {"total_steps": True}},
            {"training": {"seed": 2 ** 32}},
            {"training": {"seed": 2 ** 32 - 100000, "total_steps": 1000}},
            {"training": {"tau": 1.01}},
            {"training": {"batch_size": 512, "replay_capacity": 256}},
            {"training": {"value_coef": -0.1}},
            {"network": {"num_layers": 4096}},
            {"network": {"embedding_dim": 4096}},
            {"network": {"kind": "mlp", "hidden_sizes": [1024] * 8}},
            {"network": {"kind": "transformer", "embedding_dim": 1024,
                         "num_heads": 32, "num_layers": 12, "feedforward_dim": 8192}},
            {"evaluation": {"validation_seed": 2 ** 32 - 2, "validation_count": 3}},
            {"algorithm": "ppo", "network": {"dropout": 0.1}},
            {"reward": {"terms": [{"id": "terminal_si", "weight": 1, "scale": 1, "timing": "step"}]}},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_config(raw)

    def test_catalog_matches_features(self):
        entries = catalog()
        self.assertGreaterEqual(len(entries["features"]), 14)
        self.assertGreaterEqual(len(entries["reward_terms"]), 20)
        self.assertTrue(all(item["dim"] > 0 for item in entries["features"]))
        self.assertTrue(all(item["implemented"] for item in entries["reward_terms"]))
        self.assertTrue(all(item["formula"] != item["id"] for item in entries["reward_terms"]))

    def test_formula_sign_and_malformed_json_types(self):
        config = validate_config({"reward": {"preset": "custom", "terms": [
            {"id": "si_delta", "weight": 2, "scale": 4, "timing": "step"},
            {"id": "power_cost", "weight": 1, "scale": 1, "timing": "step"}]}})
        formula = config_summary(config)["reward_formula"]
        self.assertIn("2×(ΔΣmin", formula)
        self.assertIn("1×(−本步实际辐射功率", formula)
        for raw in ({"state": {"features": [{"id": "beam"}]}},
                    {"schema_version": 1.0},
                    {"reward": {"preset": []}},
                    {"reward": {"terms": [{"id": ["si_delta"], "weight": 1,
                                            "scale": 1, "timing": "step"}]}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_config(raw)


if __name__ == "__main__":
    unittest.main()
