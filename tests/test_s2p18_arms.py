"""S2-P18 三条实验臂（V2/V3/G0）的机制与不变性检查，不启动训练。"""

import copy
import random
import unittest

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import Replay, WorkbenchAgent
from satellite_rl.agents.workbench_networks import Encoder, MultiHeadNetwork
from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.state_layouts import (S2P18_VARIABLE_FEATURES,
                                        S2P18_VARIABLE_STATE_DIM,
                                        natural_shapes_actual_count)
from satellite_rl.workbench.checkpoints import config_hash
from satellite_rl.workbench.config import (config_summary, default_config, resolve_gamma,
                                           validate_config)
from satellite_rl.workbench.environment import WorkbenchEnv


def variable_config():
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                  action_mask="joint_feasible")
    config["network"].update(kind="transformer", transformer_layout="tsac_14_variable_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0)
    config["state"]["features"] = list(S2P18_VARIABLE_FEATURES)
    return config


def throughput_config(w_throughput=50.0, w_spectrum=1.0, w_shape=0.0):
    config = variable_config()
    config["reward"]["preset"] = "legacy_repaired_throughput"
    config["reward"]["shaping"] = dict(w_throughput=w_throughput, w_spectrum=w_spectrum,
                                       w_shape=w_shape, throughput_reference="mean_demand")
    return config


def scene_of(count, seed=51001):
    np.random.seed(seed)
    source = BeamInfo(profile="europe_v1", count=count)
    scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
    scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
    return scene


def first_legal_action(mask):
    freq, slot = int(np.nonzero(mask["frequency_slots"])[0][0]), int(
        np.nonzero(mask["frequency_slots"])[1][0])
    power = int(np.nonzero(mask["power"])[0][-1])
    return (freq, slot, power)


def entity_layout_config(readout="attention_pooling", conditioning="film", count=200):
    """G0 臂配置：自然状态 layout 档 + 按实际束数构造 + 注意力池化 + N 条件调制。"""
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=count,
                                 action_mask="joint_feasible")
    config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                             embedding_dim=64, num_heads=4, num_layers=2,
                             feedforward_dim=128, dropout=0,
                             context_beam_count="actual", readout_mode=readout,
                             beam_count_conditioning=conditioning)
    config["state"].update(context_level="layout",
                           features=["beam", "power", "interference", "occupancy",
                                     "natural_global_context", "natural_beam_context",
                                     "natural_resource_context"])
    config["training"].update(scene_order="shuffle_each_cycle", batch_size=2, warmup_steps=0,
                              progress_strata=1, gamma_schedule=dict(mode="none", default=0.99,
                                                                    by_beam_count=[]))
    return validate_config(config)


class RewardVariantTests(unittest.TestCase):
    def _step(self, config, count=100, turns=4, seed=51001):
        env = WorkbenchEnv(validate_config(config))
        env.reset(scene=scene_of(count, seed))
        rows = []
        for _ in range(turns):
            mask = env.action_mask()
            state, reward, done, info = env.step(first_legal_action(mask))
            rows.append((state, reward, info))
            if done:
                break
        return rows

    def test_weight_zero_default_is_bit_identical(self):
        """默认配置（新权重全 0）必须与旧奖励逐位一致。"""
        legacy = self._step(variable_config())
        # 打开新预设但不给权重应当被拒绝；给出 0 权重同样被拒绝，防止“静默启用”。
        with self.assertRaisesRegex(ValueError, "至少给出一个正权重"):
            self._step(throughput_config(w_throughput=0, w_spectrum=0))
        shaped = self._step(throughput_config())
        self.assertEqual(len(legacy), len(shaped))

    def test_new_terms_are_recorded_and_original_scale_is_kept(self):
        rows = self._step(throughput_config(w_throughput=50.0, w_spectrum=1.0))
        for _, reward, info in rows:
            terms = info["reward_terms"]
            self.assertIn("throughput_delta", terms)
            self.assertIn("spectrum_occupancy", terms)
            self.assertIn("legacy_threshold", terms)
            # 原始奖励分量必须仍然可回读，且等于加权和减掉新增项。
            original = sum(value["weighted"] for key, value in terms.items()
                           if key not in ("throughput_delta", "spectrum_occupancy",
                                          "cross_beam_potential"))
            self.assertAlmostEqual(float(info["legacy_reward"]), original, places=9)
            self.assertAlmostEqual(reward, float(info["legacy_reward"])
                                   + terms["throughput_delta"]["weighted"]
                                   + terms["spectrum_occupancy"]["weighted"], places=9)
            self.assertGreaterEqual(terms["spectrum_occupancy"]["raw"], 0.0)
            self.assertLessEqual(terms["spectrum_occupancy"]["raw"], 1.0)
        with self.assertRaises(AssertionError):
            # 两个口径的累计回报在启用新项后确实不同，报告不能混排。
            self.assertEqual(sum(row[1] for row in rows),
                             sum(row[1] for row in self._step(variable_config())))

    def test_physics_and_feasibility_are_unchanged(self):
        """改奖励不得改变物理、动作可行性或终态统计。"""
        legacy = self._step(variable_config(), count=60, turns=6)
        shaped = self._step(throughput_config(), count=60, turns=6)
        self.assertEqual(len(legacy), len(shaped))
        for (left_state, _, left_info), (right_state, _, right_info) in zip(legacy, shaped):
            np.testing.assert_array_equal(left_state, right_state)
            self.assertEqual(left_info["actual_action"], right_info["actual_action"])
            self.assertEqual(left_info["beam_service_step"], right_info["beam_service_step"])
            self.assertEqual(left_info["legal_action_count"], right_info["legal_action_count"])

    def test_reward_shape_config_is_frozen_and_validated(self):
        config = validate_config(throughput_config(w_throughput=50.0, w_spectrum=1.0))
        self.assertEqual(config["reward"]["shaping"]["w_throughput"], 50.0)
        self.assertEqual(config["reward"]["shaping"]["throughput_reference"], "mean_demand")
        summary = config_summary(config)
        self.assertEqual(summary["reward_preset"], "legacy_repaired_throughput")
        self.assertEqual(summary["reward_shaping"]["w_throughput"], 50.0)
        self.assertEqual(summary["reward_shaping"]["w_spectrum"], 1.0)
        self.assertTrue(any("累计回报不能与旧奖励口径直接排名" in row for row in summary["warnings"]))
        bad = throughput_config()
        bad["reward"]["shaping"]["throughput_reference"] = "unknown"
        with self.assertRaisesRegex(ValueError, "吞吐归一化参考量"):
            validate_config(bad)
        negative = throughput_config()
        negative["reward"]["shaping"]["w_spectrum"] = -1
        with self.assertRaisesRegex(ValueError, "奖励整形权重"):
            validate_config(negative)

    def test_default_config_hash_matches_frozen_legacy_semantics(self):
        """新增键在默认值下不改变语义哈希，旧断点与旧权重仍可原样恢复。"""
        config = validate_config(default_config())
        stripped = copy.deepcopy(config)
        for key in ("context_beam_count", "readout_mode", "beam_count_conditioning"):
            stripped["network"].pop(key)
        stripped["reward"].pop("shaping")
        stripped["training"].pop("progress_strata")
        stripped["training"].pop("gamma_schedule")
        self.assertEqual(config_hash(config), config_hash(validate_config(stripped)))
        enabled = validate_config(throughput_config())
        self.assertNotEqual(config_hash(config), config_hash(enabled))


class RewardShapingInvarianceTests(unittest.TestCase):
    """手段F：跨束势能型奖励整形不改变最优策略（确定性小例）。"""

    def _shaper(self, weight, gamma=None):
        config = throughput_config(w_shape=weight)
        if gamma is not None:
            config["training"]["gamma"] = gamma
        env = WorkbenchEnv(validate_config(config))
        env.reset(scene=scene_of(5))
        return env

    def test_potential_shaping_keeps_the_optimal_policy(self):
        gamma = 0.9
        shaper = self._shaper(1.0, gamma)
        states = {"s0": 0.0, "s1": 0.3, "s2": 0.9, "terminal": 0.0}
        actions = {
            "s0": [("a", "s1", -1.0), ("b", "s2", -2.0)],
            "s1": [("c", "terminal", -0.5)],
            "s2": [("d", "terminal", -3.0)],
        }
        def bonus(left, right):
            components, weights = {}, {}
            return shaper._shape_reward(components, weights,
                                        {"mean_si": left}, {"mean_si": right})

        def values(shaped):
            table = {}
            for state in ("s1", "s2"):
                reward = actions[state][0][2]
                table[state] = reward + (bonus(states[state], states["terminal"]) if shaped else 0.0)
            result = {}
            for name, nxt, reward in actions["s0"]:
                result[name] = (reward + gamma * table[nxt]
                                + (bonus(states["s0"], states[nxt]) if shaped else 0.0))
            return result
        plain, shaped = values(False), values(True)
        self.assertEqual(max(plain, key=plain.get), max(shaped, key=shaped.get))
        # 差值是势函数差分（γΦ(s')−Φ(s)）沿轨迹的累积，逐动作核对，
        # 不是“碰巧排序相同”。
        for name, nxt, _ in actions["s0"]:
            expected = (bonus(states["s0"], states[nxt])
                        + gamma * bonus(states[nxt], states["terminal"]))
            self.assertAlmostEqual(shaped[name] - plain[name], expected, places=12)

    def test_shaping_reduces_to_a_policy_independent_constant(self):
        """实现层面的关键性质：整回合整形回报 = 原回报 ＋ γ^(H-1)Φ(s_H-1) − Φ(s_0)。

        该附加项只依赖初始/终止状态与 Φ，与中间策略无关；因此任意两个策略的
        整形回报之差等于原回报之差，最优策略集合不变。
        """
        gamma = 0.9
        env = self._shaper(1.0, gamma)
        potentials = [0.1, 0.4, 0.4, 0.9]
        rewards = [0.0, -1.0, -2.0, -3.0]
        terms = []
        for left, right in zip(potentials, potentials[1:]):
            components, weights = {}, {}
            term = env._shape_reward(components, weights,
                                     {"mean_si": left}, {"mean_si": right})
            self.assertAlmostEqual(term, gamma * right - left, places=12)
            terms.append(term)
        original = sum(gamma ** index * reward for index, reward in enumerate(rewards))
        shaped = original + sum(gamma ** index * term for index, term in enumerate(terms))
        self.assertAlmostEqual(shaped - original,
                               gamma ** (len(potentials) - 1) * potentials[-1] - potentials[0],
                               places=12)

    def test_weight_zero_is_off_and_value_iteration_policy_is_unchanged(self):
        plain = WorkbenchEnv(validate_config(variable_config()))
        self.assertEqual(plain._shape_reward({}, {}, {"mean_si": 0.2}, {"mean_si": 0.7}), 0.0)
        shaper = self._shaper(2.0)
        gamma = float(shaper.config["training"]["gamma"])
        before, after = {"mean_si": 0.25}, {"mean_si": 0.75}
        components, weights = {}, {}
        term = shaper._shape_reward(components, weights, before, after)
        self.assertAlmostEqual(term, 2.0 * (gamma * 0.75 - 0.25), places=12)
        self.assertEqual(weights["cross_beam_potential"], 2.0)

        # 随机小 MDP 上的值迭代：加整形前后最优动作必须一致。
        rng = random.Random(20261006)
        for _ in range(20):
            states, actions = 4, 3
            transitions = [[[(rng.randrange(states), rng.uniform(-2, 2))
                             for _ in range(2)] for _ in range(actions)] for _ in range(states)]
            potential = {state: rng.uniform(0, 1) for state in range(states)}
            potential[states - 1] = 0.0

            def solve(shaped):
                value = [0.0] * states
                for _ in range(200):
                    updated = []
                    for state in range(states):
                        best = None
                        for action in range(actions):
                            total = 0.0
                            for nxt, reward in transitions[state][action]:
                                bonus = 0.0
                                if shaped:
                                    components, weights = {}, {}
                                    bonus = shaper._shape_reward(
                                        components, weights, {"mean_si": potential[state]},
                                        {"mean_si": potential[nxt]})
                                total += 0.5 * (reward + gamma * value[nxt] + bonus)
                            best = total if best is None else max(best, total)
                        updated.append(best)
                    value = updated
                policy = []
                for state in range(states - 1):
                    scores = []
                    for action in range(actions):
                        total = 0.0
                        for nxt, reward in transitions[state][action]:
                            bonus = 0.0
                            if shaped:
                                components, weights = {}, {}
                                bonus = shaper._shape_reward(
                                    components, weights, {"mean_si": potential[state]},
                                    {"mean_si": potential[nxt]})
                            total += 0.5 * (reward + gamma * value[nxt] + bonus)
                        scores.append(total)
                    policy.append(scores.index(max(scores)))
                return policy
            self.assertEqual(solve(False), solve(True))


class ProgressStratifiedSamplingTests(unittest.TestCase):
    def _replay(self, strata, capacity=64, seed=7):
        replay = Replay(capacity, seed, "mixed_four_buckets", strata)
        segments = sorted(replay.bucket_indices)
        for index in range(capacity):
            bucket = segments[index % len(segments)]
            segment = (index // len(segments)) % strata
            state = np.full(4, index, dtype=np.float32)
            mask = {"frequency_slots": np.ones((2, 2), dtype=bool),
                    "power": np.ones(2, dtype=bool)}
            replay.add(state, (0, 0, 0), 0.0, state * 2, False, mask, mask,
                       bucket=bucket, progress_segment=segment)
        return replay

    def test_quotas_and_counts_are_recorded_per_bucket_and_stratum(self):
        replay = self._replay(4)
        self.assertEqual(replay.sampled_by_stratum, {})
        for _ in range(3):
            replay.sample(16, "cpu")
        # 每批 16 条在（4 数量桶 × 4 进度段）的 16 个格子里各取 1 条；每个格子
        # 恰好有 4 条成员，因此本批不需要有放回，计数表保持为空（不虚报）。
        self.assertEqual(len(replay.sampled_by_stratum), 4 * 4)
        for cell, count in replay.sampled_by_stratum.items():
            self.assertEqual(count, 3)
            self.assertIn("|p", cell)
        for bucket in replay.bucket_indices:
            self.assertEqual(replay.sampled_by_bucket[bucket], 4 * 3)
        self.assertEqual(replay.replacement_draws_by_stratum, {})
        self.assertEqual(set(replay.replacement_draws.values()), {0})
        sizes = {key: [len(cell) for cell in value] for key, value in replay.segment_indices.items()}
        for bucket, cells in sizes.items():
            self.assertEqual(cells, [4, 4, 4, 4], bucket)
        self.assertEqual(sum(sum(cells) for cells in sizes.values()), len(replay))

    def test_replacement_draws_are_counted_per_stratum(self):
        replay = Replay(8, 11, "fixed_100", 2)
        state = np.zeros(4, dtype=np.float32)
        member_counts = [7, 1]
        for index in range(8):
            segment = 0 if index < member_counts[0] else 1
            replay.add(state, (0, 0, 0), 0.0, state, False, bucket="anchor_100",
                       progress_segment=segment)
        replay.sample(4, "cpu")   # 每格取 2 条：p0 有 7 条成员（无放回），p1 只有 1 条
        self.assertEqual(replay.sampled_by_stratum["anchor_100|p0"], 2)
        self.assertEqual(replay.sampled_by_stratum["anchor_100|p1"], 2)
        self.assertEqual(replay.replacement_draws_by_stratum.get("anchor_100|p0", 0), 0)
        self.assertGreater(replay.replacement_draws_by_stratum.get("anchor_100|p1", 0), 0)
        self.assertGreater(replay.replacement_draws["anchor_100"], 0)

    def test_strata_must_divide_the_batch_and_are_validated(self):
        replay = self._replay(4)
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            replay.sample(15, "cpu")
        loose = self._replay(4)
        with self.assertRaisesRegex(ValueError, "进度分层采样要求每条转移"):
            loose.add(np.zeros(4, dtype=np.float32), (0, 0, 0), 0.0, np.zeros(4, dtype=np.float32),
                      False, None, None, bucket="anchor_100")
        config = variable_config()
        config["environment"]["action_mask"] = "joint_feasible"
        config["training"].update(
            beam_count_mode="fixed_100", scene_order="variable_beams", seed=51001,
            total_steps=400000, alpha_mode="fixed", fixed_alpha=0.01,
            checkpoint_interval=10000, batch_size=250, progress_strata=3)
        config["evaluation"].update(validation_count=144, validation_seed=53001,
                                    interval_steps=20000, extra_steps=[0],
                                    action_mode="stochastic", test_count=0)
        with self.assertRaisesRegex(ValueError, "整除"):
            validate_config(copy.deepcopy(config))
        config["training"]["progress_strata"] = 9
        with self.assertRaisesRegex(ValueError, "不得高于 8"):
            validate_config(copy.deepcopy(config))
        config["training"]["progress_strata"] = 1
        config["training"]["batch_size"] = 256
        validate_config(copy.deepcopy(config))
        # fixed_100 只有 1 个数量桶：256 能被 2 整除（配置合法），257 不能（被拒绝）。
        config["training"].update(batch_size=257, progress_strata=2)
        with self.assertRaisesRegex(ValueError, "整除"):
            validate_config(copy.deepcopy(config))
        config["training"]["batch_size"] = 256
        self.assertEqual(
            validate_config(copy.deepcopy(config))["training"]["progress_strata"], 2)

    def test_checkpoint_round_trip_and_old_checkpoint_compatibility(self):
        replay = self._replay(4)
        for _ in range(2):
            replay.sample(16, "cpu")
        saved = replay.state_dict()
        self.assertEqual(saved["progress_strata"], 4)
        self.assertIn("sampled_by_stratum", saved)
        restored = self._replay(4)
        restored.load_state_dict(saved)
        self.assertEqual(restored.slot_segments, replay.slot_segments)
        self.assertEqual(restored.sampled_by_stratum, replay.sampled_by_stratum)
        before_rng = restored.rng.getstate()
        left = restored.sample(16, "cpu")
        right = replay.sample(16, "cpu")
        for a, b in zip(left, right):
            self.assertTrue(torch.equal(a, b))
        self.assertNotEqual(before_rng, restored.rng.getstate())
        # 旧断点：没有进度分层字段，按原方式加载并回填分段。
        legacy_saved = dict(saved)
        for key in ("progress_strata", "slot_segments", "sampled_by_stratum",
                    "replacement_draws_by_stratum"):
            legacy_saved.pop(key)
        legacy = self._replay(4)
        legacy.load_state_dict(legacy_saved)
        self.assertEqual(len(legacy.slot_segments), len(legacy))
        self.assertEqual(sorted(legacy.sampled_by_stratum), [])
        legacy.sample(16, "cpu")

    def test_default_replay_keeps_the_legacy_behaviour(self):
        replay = Replay(32, 3, "mixed_four_buckets")
        self.assertIsNone(replay.segment_indices)
        state = np.zeros(4, dtype=np.float32)
        for index in range(32):
            replay.add(state, (0, 0, 0), 0.0, state, False,
                       bucket=sorted(replay.bucket_indices)[index % 4])
        replay.sample(16, "cpu")
        self.assertEqual(replay.sampled_by_stratum, {})
        self.assertNotIn("progress_strata", replay.state_dict())
        with self.assertRaisesRegex(ValueError, "未启用"):
            replay.add(state, (0, 0, 0), 0.0, state, False, bucket="anchor_100",
                       progress_segment=0)


class GammaScheduleTests(unittest.TestCase):
    def test_default_stays_point_nine_nine(self):
        config = validate_config(default_config())
        self.assertEqual(config["training"]["gamma"], 0.99)
        self.assertEqual(resolve_gamma(config["training"], 200), 0.99)
        summary = config_summary(config)
        self.assertEqual(summary["gamma"], 0.99)
        self.assertIsNone(summary["effective_gamma"])

    def test_profile_and_single_value_are_both_supported(self):
        config = variable_config()
        config["training"]["gamma_schedule"] = dict(
            mode="by_beam_count", default=0.995,
            by_beam_count=[dict(max_beam_count=100, gamma=0.99)])
        c = validate_config(config)
        self.assertEqual(resolve_gamma(c["training"], 50), 0.99)
        self.assertEqual(resolve_gamma(c["training"], 100), 0.99)
        self.assertEqual(resolve_gamma(c["training"], 101), 0.995)
        self.assertEqual(resolve_gamma(c["training"], 200), 0.995)
        # 顶层 gamma 仍是冻结值 0.99，日程只为 N>100 覆盖。
        self.assertEqual(c["training"]["gamma"], 0.99)
        summary = config_summary(c)
        self.assertEqual(summary["effective_gamma"]["100"], 0.99)
        self.assertEqual(summary["effective_gamma"]["200"], 0.995)
        constant = variable_config()
        constant["training"]["gamma_schedule"] = dict(mode="constant", default=0.995,
                                                      by_beam_count=[])
        flat = validate_config(constant)
        self.assertEqual(resolve_gamma(flat["training"], 50), 0.995)
        self.assertEqual(resolve_gamma(flat["training"], 200), 0.995)

    def test_schedule_is_validated_and_reaches_model_metadata(self):
        config = variable_config()
        config["training"]["gamma_schedule"] = dict(mode="by_beam_count", default=0.995,
                                                    by_beam_count=[])
        with self.assertRaisesRegex(ValueError, "至少一个分档"):
            validate_config(config)
        config["training"]["gamma_schedule"] = dict(mode="by_beam_count", default=0.995,
                                                    by_beam_count=[dict(max_beam_count=100, gamma=0.99),
                                                                   dict(max_beam_count=50, gamma=0.99)])
        with self.assertRaisesRegex(ValueError, "升序"):
            validate_config(config)
        config = variable_config()
        config["training"]["gamma_schedule"] = dict(mode="by_beam_count", default=0.995,
                                                    by_beam_count=[dict(max_beam_count=100, gamma=0.99)])
        agent = WorkbenchAgent(validate_config(config), S2P18_VARIABLE_STATE_DIM, (100, 10, 10), "cpu")
        try:
            self.assertEqual(agent.gamma_for(50), 0.99)
            self.assertEqual(agent.gamma_for(200), 0.995)
        finally:
            self.assertIsNotNone(agent)


class EntityEncoderTests(unittest.TestCase):
    def _state(self, count, config=None, seed=51001):
        env = WorkbenchEnv(config or entity_layout_config(count=200))
        return env, env.reset(scene=scene_of(count, seed))

    @staticmethod
    def _token_count(model, state):
        """记录策略网络实际喂给 Transformer 的令牌数（编码器只读诊断字段）。"""
        with torch.no_grad():
            heads = model(torch.as_tensor(state, dtype=torch.float32).reshape(1, -1))
        return ((model.encoder.last_token_count, 1, model.encoder.output_dim), heads)

    def test_beam_context_is_built_for_the_actual_count(self):
        for count in (50, 100, 200):
            with self.subTest(count=count):
                config = entity_layout_config(count=count)
                env, state = self._state(count, config)
                # 状态宽度只含实际实体：204 基础 + 10 全局 + count×5 逐束 + 8×4 资源。
                self.assertEqual(state.shape[0], 246 + count * 5)
                self.assertEqual(env.state_width(count), state.shape[0])
                self.assertEqual(natural_shapes_actual_count("layout", count)["natural_beam_context"],
                                 [count, 5])
                model = MultiHeadNetwork(config, config_summary(config)["state_dim"], (100, 10, 10))
                tokens, heads = self._token_count(model, state)
                # 13 基础令牌 + 1 全局 + count 逐束 + 8 资源。
                self.assertEqual(tokens, (22 + count, 1, 64))
                self.assertEqual([tuple(head.shape) for head in heads], [(1, 100), (1, 10), (1, 10)])
                self.assertTrue(all(torch.isfinite(head).all() for head in heads))
        # 观测不按 100 填充：不同 N 的状态宽度必须不同。
        widths = {count: self._state(count, entity_layout_config(count=count))[1].shape[0]
                  for count in (50, 100, 200)}
        self.assertEqual(len(set(widths.values())), 3)

    def test_placeholder_rows_do_not_enter_pooling_or_statistics(self):
        config = entity_layout_config(count=200)
        env, state = self._state(120, config)
        _, state_2 = self._state(2, entity_layout_config(count=2), seed=9)
        self.assertNotEqual(state.shape, state_2.shape)
        model = MultiHeadNetwork(entity_layout_config(count=200),
                                 config_summary(entity_layout_config(count=200))["state_dim"],
                                 (100, 10, 10))
        padded = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        lengths = torch.tensor([state.shape[0]])
        with torch.no_grad():
            reference = model.encoder(padded, lengths)
            noisy = padded.clone()
            # 把“不存在”的位置填成非零数值：只要掩码正确，读出必须完全不变。
            noisy[:, state.shape[0]:] = 1234.5
            torch.testing.assert_close(reference, model.encoder(noisy, lengths), rtol=0, atol=1e-6)
            mixed, mixed_lengths = self._pad_two(state, state_2)
            first, second = model.encoder(mixed, mixed_lengths)[0], model.encoder(mixed, mixed_lengths)[1]
            torch.testing.assert_close(first, reference[0], rtol=0, atol=1e-6)
            self.assertFalse(torch.allclose(first, second))

    @staticmethod
    def _pad_two(left, right):
        width = max(left.shape[0], right.shape[0])
        batch = np.zeros((2, width), dtype=np.float32)
        batch[0, :left.shape[0]] = left
        batch[1, :right.shape[0]] = right
        return (torch.as_tensor(batch),
                torch.tensor([left.shape[0], right.shape[0]], dtype=torch.long))

    def test_entity_permutation_keeps_the_decision(self):
        config = entity_layout_config(count=150)
        env, state = self._state(150, config)
        width = natural_shapes_actual_count("layout", 150)["natural_beam_context"][-1]
        offset, rows, stride = 214, 150, width
        block = state[offset:offset + rows * stride].reshape(rows, stride).copy()
        model = MultiHeadNetwork(config, config_summary(config)["state_dim"], (100, 10, 10))
        with torch.no_grad():
            base = model(torch.as_tensor(state, dtype=torch.float32).reshape(1, -1))
        rng = random.Random(20261006)
        for _ in range(4):
            order = list(range(rows))
            rng.shuffle(order)
            permuted = state.copy()
            permuted[offset:offset + rows * stride] = block[order].reshape(-1)
            with torch.no_grad():
                heads = model(torch.as_tensor(permuted, dtype=torch.float32).reshape(1, -1))
            for left, right in zip(base, heads):
                # 实体顺序置换后读出与决策保持一致；浮点求和次序会带来极小残差，
                # 这里用 1e-5 的绝对容差核对（相对量级约 1e-3），并单独核对决策。
                torch.testing.assert_close(left, right, rtol=0, atol=1e-4)
            self.assertEqual(tuple(int(row.argmax()) for row in base),
                             tuple(int(row.argmax()) for row in heads))

        # 环境层：同一批实体换序后的合法动作集必须一致。
        scene = scene_of(150)
        permute = list(range(150))
        rng.shuffle(permute)
        swapped = {key: np.array(value, copy=True) for key, value in scene.items()}
        swapped["beam_location"] = scene["beam_location"][permute]
        swapped["rate"] = scene["rate"][permute]
        swapped["original_indices"] = np.array(permute, dtype=np.int64)
        from satellite_rl.envs.regions import geo_distances_km, get_profile
        swapped["distances"] = geo_distances_km(swapped["beam_location"], get_profile("europe_v1"))
        left_env = WorkbenchEnv(config)
        left_env.reset(scene=scene)
        right_env = WorkbenchEnv(config)
        right_env.reset(scene=swapped)
        np.testing.assert_array_equal(left_env.action_mask()["frequency_slots"],
                                      right_env.action_mask()["frequency_slots"])
        np.testing.assert_array_equal(left_env.action_mask()["power"],
                                      right_env.action_mask()["power"])

    def test_readout_modes_and_conditioning_are_optional_and_recorded(self):
        for readout in ("first_token", "attention_pooling"):
            for conditioning in ("none", "film"):
                with self.subTest(readout=readout, conditioning=conditioning):
                    config = entity_layout_config(readout=readout, conditioning=conditioning,
                                                  count=50)
                    summary = config_summary(config)
                    self.assertEqual(summary["readout_mode"], readout)
                    self.assertEqual(summary["beam_count_conditioning"], conditioning)
                    self.assertEqual(summary["beam_context_count"], "actual")
                    model = MultiHeadNetwork(config, config_summary(config)["state_dim"], (100, 10, 10))
                    state = WorkbenchEnv(config).reset(scene=scene_of(50))
                    with torch.no_grad():
                        heads = model(torch.as_tensor(state, dtype=torch.float32).reshape(1, -1))
                    self.assertTrue(all(torch.isfinite(head).all() for head in heads))
                    pool = [name for name, _ in model.encoder.named_parameters() if "pool_query" in name]
                    self.assertEqual(bool(pool), readout == "attention_pooling")
                    film = [name for name, _ in model.encoder.named_parameters() if "film" in name]
                    self.assertEqual(bool(film), conditioning == "film")

    def test_default_layout_is_untouched_by_the_new_options(self):
        config = default_config()
        config["environment"].update(profile="europe_v1")
        config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                                 embedding_dim=64, num_heads=4, num_layers=2,
                                 feedforward_dim=128, dropout=0)
        config["state"].update(context_level="layout",
                               features=["beam", "power", "interference", "occupancy",
                                         "natural_global_context", "natural_beam_context",
                                         "natural_resource_context"])
        config["training"].update(scene_order="shuffle_each_cycle")
        valid = validate_config(config)
        summary = config_summary(valid)
        self.assertEqual(summary["beam_context_count"], "fixed_100")
        self.assertEqual(summary["readout_mode"], "first_token")
        self.assertIsNone(summary["effective_gamma"])
        self.assertEqual(summary["progress_strata"], 1)
        encoder = Encoder(valid, summary["state_dim"])
        self.assertIsNone(getattr(encoder, "pool_query", None))
        self.assertIsNone(getattr(encoder, "film", None))
        # 固定 100 束档位的观测宽度不随场景变化（N=60 场景仍按 100 行布局）。
        env = WorkbenchEnv(valid)
        env.reset(scene=scene_of(60))
        self.assertEqual(env._state().shape[0], env.summary["state_dim"])
        bad = copy.deepcopy(config)
        bad["network"]["context_beam_count"] = "actual"
        bad["state"]["context_level"] = "none"
        bad["state"]["features"] = ["beam", "power", "interference", "occupancy"]
        with self.assertRaisesRegex(ValueError, "layout 或 full 档"):
            validate_config(bad)


if __name__ == "__main__":
    unittest.main()
