"""Scientific input and mode contracts for temperature, memory and graph runs."""

import copy
import unittest

from satellite_rl.state_layouts import natural_shapes
from satellite_rl.workbench.config import catalog, config_summary, default_config, validate_config


def tmg_config(temporal="none", graph="none", level="compact", alpha="adaptive"):
    c = default_config()
    c["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                        embedding_dim=128, num_heads=8, num_layers=2, feedforward_dim=512,
                        temporal_mode=temporal, graph_mode=graph)
    c["state"].update(context_level=level, features=list(natural_shapes(level)))
    if temporal != "none":
        c["state"]["features"].append("previous_actual_action")
    c["environment"].update(profile="europe_v1", action_mask="joint_feasible")
    c["training"].update(scene_order="shuffle_each_cycle", initial_alpha=0.01,
                         alpha_mode=alpha, fixed_alpha=0.01, total_steps=400000)
    return c


class TmgConfigTests(unittest.TestCase):
    def test_actual_input_dimensions_tokens_and_effective_history(self):
        for temporal, graph, level, dim, tokens, window in (
            ("none", "none", "compact", 207, 14, 1),
            ("previous_action", "none", "compact", 211, 15, 1),
            ("gru", "none", "compact", 211, 15, 16),
            ("causal_transformer", "none", "compact", 211, 15, 16),
            ("none", "none", "full", 2946, 32, 1),
            ("none", "beam_tokens", "full", 2946, 122, 1),
            ("none", "interference_bias", "full", 2946, 122, 1),
        ):
            with self.subTest(temporal=temporal, graph=graph, level=level):
                c = tmg_config(temporal, graph, level)
                summary = config_summary(c)
                self.assertEqual((summary["state_dim"], summary["network_token_count"],
                                  summary["temporal_window"]), (dim, tokens, window))
                self.assertEqual(summary["temporal_mode"], temporal)
                self.assertEqual(summary["graph_mode"], graph)
                if temporal != "none":
                    self.assertEqual(summary["state_layout"][-1],
                                     dict(id="previous_actual_action", offset=207, dim=4, shape=[4]))

    def test_reject_inconsistent_or_unverified_information_combinations(self):
        examples = [tmg_config("gru", "beam_tokens", "full"),
                    tmg_config("gru", "none", "full"),
                    tmg_config("none", "beam_tokens", "compact")]
        for section, key, value in [("network", "sac_critic_structure", "joint"),
                                    ("network", "sac_policy_structure", "conditional"),
                                    ("network", "temporal_window", 17),
                                    ("network", "temporal_window", True),
                                    ("network", "temporal_window", 0),
                                    ("environment", "action_mask", "none")]:
            c = tmg_config("gru")
            c[section][key] = value
            examples.append(c)
        c = tmg_config()
        c["state"]["features"].append("previous_actual_action")
        examples.append(c)
        c = tmg_config("previous_action")
        c["state"]["features"].pop()
        examples.append(c)
        for c in examples:
            with self.subTest(config=c), self.assertRaises(ValueError):
                validate_config(c)

    def test_fixed_temperature_is_explicit_positive_and_independent_of_critic(self):
        for critic in ("additive", "joint"):
            c = tmg_config(alpha="fixed")
            c["network"]["sac_critic_structure"] = critic
            s = config_summary(c)
            self.assertEqual(s["alpha_mode"], "fixed")
            self.assertEqual(s["fixed_alpha"], 0.01)
        for value in (0, -0.1, float("nan"), True, "0.01"):
            c = tmg_config(alpha="fixed")
            c["training"]["fixed_alpha"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config(c)
        c = tmg_config(alpha="auto")
        with self.assertRaises(ValueError):
            validate_config(c)
        self.assertIsNone(config_summary(tmg_config())["fixed_alpha"])

    def test_capabilities_advertise_modes_and_meaningful_action_fields(self):
        caps = catalog()
        self.assertEqual({x["id"] for x in caps["temporal_modes"]},
                         {"none", "previous_action", "gru", "causal_transformer"})
        self.assertEqual({x["id"] for x in caps["graph_modes"]},
                         {"none", "beam_tokens", "interference_bias"})
        feature = next(x for x in caps["features"] if x["id"] == "previous_actual_action")
        self.assertEqual(feature["shape"], [4])
        self.assertEqual(feature["timing"], "动作前")


if __name__ == "__main__":
    unittest.main()
