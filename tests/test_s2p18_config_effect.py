"""S2-P18 配置生效链路、状态维数与前端数据源的端到端检查（不训练）。"""

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.state_layouts import (S2P18_VARIABLE_FEATURES, natural_shapes,
                                        natural_shapes_actual_count)
from satellite_rl.workbench.config import (catalog, config_summary, default_config,
                                           expected_state_dim, resolve_gamma, validate_config)
from satellite_rl.workbench.environment import WorkbenchEnv

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "satellite_rl" / "workbench" / "static"


def scene_of(count, seed=51001):
    np.random.seed(seed)
    source = BeamInfo(profile="europe_v1", count=count)
    scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
    scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
    return scene


def _variable_base(total_steps=400000):
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                  action_mask="joint_feasible")
    config["training"].update(seed=51001, scene_order="variable_beams",
                              beam_count_mode="mixed_four_buckets", beam_count_min=50,
                              beam_count_max=200, held_out_beam_counts=[60, 90, 140, 180],
                              total_steps=total_steps, batch_size=256, replay_capacity=20000,
                              warmup_steps=2000, update_every=1, gradient_steps=1,
                              alpha_mode="fixed", fixed_alpha=0.01, gamma=0.99, tau=0.005,
                              actor_lr=1e-5, critic_lr=1e-4, gradient_clip=10,
                              checkpoint_interval=10000)
    config["evaluation"].update(validation_count=144, validation_seed=53001,
                                interval_steps=20000, extra_steps=[0], test_count=0,
                                action_mode="stochastic")
    return config


def arm_raw(kind):
    """与前端 ”套用 S2-P18 预设“ 写入的键保持一致。"""
    config = _variable_base()
    if kind in ("V2", "V3"):
        config["network"].update(kind="transformer",
                                 transformer_layout="tsac_14_variable_context",
                                 embedding_dim=128, num_heads=8, num_layers=2,
                                 feedforward_dim=512, dropout=0,
                                 context_beam_count="fixed_100", readout_mode="first_token",
                                 beam_count_conditioning="none")
        config["state"]["features"] = list(S2P18_VARIABLE_FEATURES)
        config["reward"]["shaping"] = dict(w_throughput=50 if kind == "V2" else 0,
                                           w_spectrum=1 if kind == "V2" else 0,
                                           w_shape=0.5 if kind == "V3" else 0,
                                           throughput_reference="mean_demand")
        if kind == "V2":
            config["reward"]["preset"] = "legacy_repaired_throughput"
            config["training"]["progress_strata"] = 4
        else:
            config["reward"]["preset"] = "legacy_repaired"
            config["training"]["gamma_schedule"] = dict(
                mode="by_beam_count", default=0.995,
                by_beam_count=[dict(max_beam_count=100, gamma=0.99)])
        return config
    config["environment"]["beam_count_max"] = 120
    config["training"].update(beam_count_max=120, beam_count_mode="none",
                              scene_order="shuffle_each_cycle")
    config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0, context_beam_count="actual",
                             readout_mode="attention_pooling", beam_count_conditioning="film")
    config["state"].update(context_level="layout",
                           features=["beam", "power", "interference", "occupancy",
                                     "natural_global_context", "natural_beam_context",
                                     "natural_resource_context"])
    config["reward"]["preset"] = "legacy_repaired"
    config["reward"]["shaping"] = dict(w_throughput=0, w_spectrum=0, w_shape=0,
                                       throughput_reference="mean_demand")
    return config


def _walk(raw, validated, path=""):
    """递归对比原始配置与校验后配置，返回被静默丢弃的键路径。"""
    dropped = []
    if isinstance(raw, dict):
        if not isinstance(validated, dict):
            return [path or "<root>"]
        for key, value in raw.items():
            if key not in validated:
                dropped.append(f"{path}{key}")
                continue
            dropped.extend(_walk(value, validated[key], f"{path}{key}."))
    return dropped


class ConfigEffectTests(unittest.TestCase):
    def test_no_configured_key_is_silently_dropped(self):
        for kind in ("V2", "V3", "G0"):
            with self.subTest(arm=kind):
                raw = arm_raw(kind)
                validated = validate_config(raw)
                self.assertEqual(_walk(raw, validated), [])

    def test_validated_config_carries_every_new_key_into_effect(self):
        v2 = validate_config(arm_raw("V2"))
        self.assertEqual(v2["reward"]["preset"], "legacy_repaired_throughput")
        self.assertEqual(v2["reward"]["shaping"]["w_throughput"], 50)
        self.assertEqual(v2["reward"]["shaping"]["w_spectrum"], 1)
        self.assertEqual(v2["training"]["progress_strata"], 4)
        self.assertEqual(v2["network"]["context_beam_count"], "fixed_100")
        self.assertEqual(v2["state"]["features"], list(S2P18_VARIABLE_FEATURES))
        self.assertEqual(config_summary(v2)["state_dim"], 213)

        v3 = validate_config(arm_raw("V3"))
        self.assertEqual(v3["training"]["gamma_schedule"]["mode"], "by_beam_count")
        self.assertEqual(resolve_gamma(v3["training"], 200), 0.995)
        self.assertEqual(v3["reward"]["shaping"]["w_shape"], 0.5)
        self.assertEqual(config_summary(v3)["effective_gamma"]["100"], 0.99)
        self.assertEqual(config_summary(v3)["effective_gamma"]["200"], 0.995)

        g0 = validate_config(arm_raw("G0"))
        self.assertEqual(g0["network"]["context_beam_count"], "actual")
        self.assertEqual(g0["network"]["readout_mode"], "attention_pooling")
        self.assertEqual(g0["network"]["beam_count_conditioning"], "film")
        summary = config_summary(g0)
        self.assertEqual(summary["readout_mode"], "attention_pooling")
        self.assertEqual(summary["beam_context_count"], "actual")
        self.assertEqual(summary["state_dim"], expected_state_dim(g0))

    def test_configured_state_features_change_the_environment(self):
        """状态特征不是“只写在配置里”：维数与取值必须随之改变。"""
        legacy = validate_config(arm_raw("V2"))
        legacy["state"]["features"] = ["beam", "power", "interference", "occupancy", "beta",
                                       "progress", "remaining_demand", "beam_count_normalized",
                                       "remaining_beams_normalized", "power_pressure"]
        self.assertEqual(config_summary(legacy)["state_dim"], 210)
        env_legacy = WorkbenchEnv(legacy)
        env_new = WorkbenchEnv(validate_config(arm_raw("V2")))
        left = env_legacy.reset(scene=scene_of(100))
        right = env_new.reset(scene=scene_of(100))
        self.assertEqual(left.shape[0], 210)
        self.assertEqual(right.shape[0], 213)
        np.testing.assert_array_equal(left, right[:210])
        # 新增三项分别是 组号/7、本组剩余预留义务、剩余合法槽上限。
        # N=100 时本组还有 13 束（0,8,...,96），可用槽为 100-(13-1)=88。
        self.assertAlmostEqual(float(right[210]), 0.0)
        self.assertAlmostEqual(float(right[211]), 13 / 100)
        self.assertAlmostEqual(float(right[212]), 88 / 100)

    def test_progress_strata_reach_the_replay_and_are_recorded(self):
        config = validate_config(arm_raw("V2"))
        summary = config_summary(config)
        agent = build_agent(config, summary["state_dim"], tuple(summary["action_dims"]), "cpu")
        self.assertEqual(agent.replay.progress_strata, 4)
        self.assertIsNotNone(agent.replay.segment_indices)
        state = np.zeros(summary["state_dim"], dtype=np.float32)
        mask = {"frequency_slots": np.ones((100, 10), dtype=bool),
                "power": np.ones(10, dtype=bool)}
        for bucket in agent.replay.bucket_indices:
            for segment in range(4):
                for _ in range(4):
                    agent.replay.add(state, (0, 0, 0), 0.0, state, False, mask, mask,
                                     bucket=bucket, progress_segment=segment)
        agent.replay.sample(256, "cpu")
        self.assertEqual(agent.replay.sampled_by_bucket,
                         {key: 64 for key in agent.replay.bucket_indices})
        self.assertEqual(len(agent.replay.sampled_by_stratum), 16)
        self.assertEqual(set(agent.replay.sampled_by_stratum.values()), {16})
        self.assertTrue(agent.replay.replacement_draws_by_stratum)
        saved = agent.replay.state_dict()
        self.assertEqual(saved["progress_strata"], 4)
        self.assertEqual(len(saved["slot_segments"]), len(agent.replay))

    def test_default_arm_keeps_legacy_sampling_and_gamma(self):
        config = validate_config(default_config())
        summary = config_summary(config)
        agent = build_agent(config, summary["state_dim"], tuple(summary["action_dims"]), "cpu")
        self.assertEqual(agent.replay.progress_strata, 1)
        self.assertIsNone(agent.replay.segment_indices)
        self.assertEqual(agent.gamma_for(200), 0.99)
        self.assertIsNone(summary["effective_gamma"])

    def test_entity_encoder_arm_accepts_multiple_counts_and_permutations(self):
        """G0：N=50/100/200 前向与掩码都不报错，实体置换后决策一致。"""
        for count in (50, 100, 200):
            with self.subTest(count=count):
                config = validate_config(arm_raw("G0"))
                config["environment"]["beam_count_max"] = count
                config["training"]["beam_count_max"] = count
                config = validate_config(config)
                summary = config_summary(config)
                env = WorkbenchEnv(config)
                state = env.reset(scene=scene_of(count))
                self.assertEqual(state.shape[0], expected_state_dim(config))
                mask = env.action_mask()
                self.assertEqual(mask["frequency_slots"].shape, (100, 10))
                agent = build_agent(config, summary["state_dim"],
                                    tuple(summary["action_dims"]), "cpu")
                with torch.no_grad():
                    # 逐束实体档位：观测按束数上限补零，选择动作时必须给出真实束数。
                    action = agent.act(state, deterministic=True, action_mask=mask,
                                       beam_entity_count=count)
                self.assertEqual(len(action), 3)
                width = natural_shapes(config["state"]["context_level"])["natural_beam_context"][-1]
                offset = 204 + natural_shapes(config["state"]["context_level"])["natural_global_context"][0]
                rows = state[offset:offset + count * width].reshape(count, width).copy()
                order = list(range(count))[::-1]
                permuted = state.copy()
                permuted[offset:offset + count * width] = rows[order].reshape(-1)
                with torch.no_grad():
                    left = agent.policy_network()(
                        torch.as_tensor(state, dtype=torch.float32).reshape(1, -1), count)
                    right = agent.policy_network()(
                        torch.as_tensor(permuted, dtype=torch.float32).reshape(1, -1), count)
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=1e-4)
                self.assertEqual(tuple(int(row.argmax()) for row in left),
                                 tuple(int(row.argmax()) for row in right))
                # 观测不按 100 填充：N 不同则状态宽度不同。
                self.assertEqual(natural_shapes_actual_count("layout", count),
                                 {"beam": [3], "power": [1], "interference": [100],
                                  "occupancy": [100], "natural_global_context": [10],
                                  "natural_beam_context": [count, 5],
                                  "natural_resource_context": [8, 4]})


class FrontendSourceTests(unittest.TestCase):
    def test_catalog_exposes_every_new_option(self):
        data = catalog()
        features = {row["id"] for row in data["features"]}
        for fid in ("group", "group_remaining_obligation", "remaining_slot_limit"):
            self.assertIn(fid, features)
        self.assertIn("legacy_repaired_throughput", {row["id"] for row in data["presets"]})
        self.assertEqual([row["id"] for row in data["s2p18_arms"]], ["V2", "V3", "G0"])
        for name in ("readout_modes", "beam_count_conditioning_modes", "context_beam_counts",
                     "progress_strata_options", "gamma_schedule_modes", "optional_state_features"):
            self.assertTrue(data[name], name)
        self.assertEqual({row["id"] for row in data["readout_modes"]},
                         {"first_token", "attention_pooling"})
        self.assertEqual({row["id"] for row in data["context_beam_counts"]},
                         {"fixed_100", "actual"})

    def test_page_script_exposes_new_options_and_by_beam_panel(self):
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        for token in ("legacy_repaired_throughput", "progress_strata", "gamma_schedule",
                      "context_beam_count", "readout_mode", "beam_count_conditioning",
                      "applyS2P18Preset", "s2p18PresetButtons", "rewardShapingControls",
                      "gammaScheduleControls", "entityEncoderControls", "beamCountPanel"):
            self.assertIn(token, script, token)
        self.assertIn('id="monitor-by-beam"', page)
        self.assertIn("beamCountPanel(s)", script)

    def test_arm_presets_in_catalog_match_the_config_key_set(self):
        """catalog 里的三臂预设必须覆盖前端脚本写入的新键，避免两处漂移。"""
        presets = {row["id"]: row for row in catalog()["s2p18_arms"]}
        v2 = presets["V2"]
        self.assertEqual(v2["reward_preset"], "legacy_repaired_throughput")
        self.assertEqual(v2["reward_shaping"]["w_throughput"], 50)
        self.assertEqual(v2["progress_strata"], 4)
        self.assertEqual(v2["state_features"], list(S2P18_VARIABLE_FEATURES))
        v3 = presets["V3"]
        self.assertEqual(v3["gamma_schedule"]["mode"], "by_beam_count")
        self.assertEqual(v3["reward_shaping"]["w_shape"], 0.5)
        g0 = presets["G0"]
        self.assertEqual(g0["network"]["context_beam_count"], "actual")
        self.assertEqual(g0["network"]["readout_mode"], "attention_pooling")
        self.assertEqual(g0["network"]["beam_count_conditioning"], "film")


if __name__ == "__main__":
    unittest.main()
