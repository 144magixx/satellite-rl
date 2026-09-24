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
