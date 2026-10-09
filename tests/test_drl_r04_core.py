"""A-stage fixed-width executor and shared physical-budget checks."""

import pickle
from types import SimpleNamespace
import unittest

import numpy as np

from satellite_rl.coverage.drl_r04_config import DrlR04Config
from satellite_rl.coverage.drl_r04_core import (DrlR04Env, h1_teacher_label,
                                                 r04_objective, run_h1, run_h2)
from satellite_rl.coverage.drl_r04_observation import layout_sha256
from satellite_rl.coverage.variable_design import _objective, metrics


def scene(positions=None, demands=None):
    points = np.asarray(positions if positions is not None else
                        [[45, 10], [45.01, 10.01], [46, 11], [46.01, 11.01],
                         [47, 12], [47.01, 12.01]], dtype=float)
    values = np.asarray(demands if demands is not None else
                        [450e6, 400e6, 350e6, 500e6, 300e6, 280e6], dtype=float)
    return SimpleNamespace(positions=points, demands=values,
                           aircraft_ids=[f"flight-{i}" for i in range(len(points))],
                           scene_id="test-scene", source_hash="test-source")


class DrlR04CoreTests(unittest.TestCase):
    def test_initialization_and_score_match_frozen_r04(self):
        s = scene()
        env = DrlR04Env(s)
        obs, info = env.reset()
        self.assertEqual(info["status"], "ready")
        self.assertEqual(obs["beam"].shape[1], 12)
        self.assertEqual(obs["candidate"].shape[1], 16)
        self.assertEqual(obs["global"].shape, (14,))
        self.assertEqual(env.menu[0][0].operation, "stop")
        self.assertTrue(all(a.operation in ("stop", "split", "merge") for a, _ in env.menu))
        self.assertTrue(metrics(s, env.initial_layout, env.cfg)["coverage_100"])
        self.assertTrue(metrics(s, env.initial_layout, env.cfg)["demand_conserved"])
        self.assertTrue(np.all(env.initial_layout.widths == .5))
        self.assertAlmostEqual(env.initial_j, _objective(s, env.initial_layout, env.cfg, route="R04"))
        self.assertAlmostEqual(env.initial_j, r04_objective(env.initial_layout, s.demands,
                                                            env.cfg.reference_capacity_bps,
                                                            env.cfg.max_beams))

    def test_observation_cache_and_no_precognition(self):
        env = DrlR04Env(scene())
        obs, _ = env.reset()
        first_calls = env.geometry_calls
        first_trials = env.trials
        other, menu = env.observe()
        self.assertEqual(first_calls, env.geometry_calls)
        self.assertEqual(first_trials, env.trials)
        self.assertTrue(np.array_equal(obs["candidate"], other["candidate"]))
        if len(menu) > 1:
            env.preview(menu[1][0])
            self.assertEqual(env.trials, 1)
            self.assertEqual(first_calls + env._trial_cache[
                (layout_sha256(env.current_layout), menu[1][0])].info["geometry_budget_charges"],
                env.geometry_calls)
            env.observe()
            self.assertEqual(env.trials, 1)

    def test_stop_reward_and_pickle_resume(self):
        env = DrlR04Env(scene())
        env.reset()
        initial_sha = env.initial_sha256
        copy = pickle.loads(pickle.dumps(env))
        obs, reward, terminated, truncated, info = copy.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.0)
        self.assertEqual(copy.initial_sha256, initial_sha)
        self.assertEqual(copy.result()["reward_sum"], 0.0)
        self.assertEqual(info["best_sha256"], initial_sha)

    def test_rejected_action_is_atomic_and_charged(self):
        env = DrlR04Env(scene([[40, -10], [50, 10], [60, 30]], [1, 2, 3]))
        env.reset()
        i, action = next((i, a) for i, (a, _) in enumerate(env.menu)
                         if a.operation == "merge")
        before = layout_sha256(env.current_layout)
        old_calls = env.geometry_calls
        env.step(i)
        self.assertGreater(env.geometry_calls, old_calls)
        self.assertEqual(env.trials, 1)
        self.assertEqual(env.commits, 1)
        self.assertFalse(env.ledger[-1]["accepted"])
        self.assertEqual(layout_sha256(env.current_layout), before)
        self.assertEqual(env.current_j, env.initial_j)
        self.assertTrue(metrics(env.scene, env.current_layout, env.cfg)["demand_conserved"])

    def test_reward_telescopes_even_when_current_layout_gets_worse(self):
        env = DrlR04Env(scene())
        env.reset()
        first = next(i for i, (a, _) in enumerate(env.menu) if a.operation == "merge")
        _, reward, _, _, _ = env.step(first)
        self.assertEqual(reward, 0.0)
        self.assertGreater(env.current_j, env.best_j)
        self.assertEqual(layout_sha256(env.best_layout), env.initial_sha256)
        second = next(i for i, (a, _) in enumerate(env.menu) if a.operation == "split")
        env.step(second)
        env.step(0)
        result = env.result()
        self.assertAlmostEqual(result["reward_sum"], 100 * (env.initial_j - env.best_j))
        self.assertLessEqual(env.best_j, env.current_j + 1e-10)
        self.assertLessEqual(result["commits"], 32)
        self.assertLessEqual(result["trials"], 256)
        self.assertLessEqual(result["geometry_calls"], 8192)
        self.assertLess(result["best_j"], result["initial_j"])

    def test_h1_h2_share_budget_and_return_best_layout(self):
        config = DrlR04Config(max_commits=3, max_trials=12)
        for runner in (run_h1, run_h2):
            env = DrlR04Env(scene(), config)
            result = runner(env)
            self.assertLessEqual(result["trials"], 12)
            self.assertLessEqual(result["commits"], 3)
            self.assertLessEqual(result["best_j"], result["initial_j"] + 1e-10)
            self.assertEqual(layout_sha256(env.best_layout), result["best_sha256"])
            self.assertAlmostEqual(result["reward_sum"], 100 *
                                   (result["initial_j"] - result["best_j"]))
        env = DrlR04Env(scene(), config)
        env.reset()
        before = env.trials
        h1_teacher_label(env)
        self.assertGreaterEqual(env.trials, before)
        self.assertEqual(env.commits, 0)

    def test_raw_terminal_and_best_can_differ(self):
        env = DrlR04Env(scene())
        env.reset()
        merge = next(i for i, (a, _) in enumerate(env.menu) if a.operation == "merge")
        env.step(merge)
        env.step(0)
        result = env.result()
        self.assertGreater(result["current_j"], result["best_j"])
        self.assertNotEqual(result["raw_terminal_layout"], result["best_layout"])
        self.assertEqual(result["reward_sum"], 0.0)

    def test_initial_failure_keeps_scene_in_denominator(self):
        separated = scene([[40, -10], [50, 10], [60, 30]], [1, 2, 3])
        env = DrlR04Env(separated, DrlR04Config(max_beams=2))
        _, info = env.reset()
        self.assertEqual(info["status"], "initialization_failed")
        self.assertTrue(env.done)
        self.assertIsNone(env.best_j)
        self.assertIsNotNone(env.failure_reason)
        self.assertEqual(len(env.initial_layout.assignment), 3)
        self.assertLess(metrics(separated, env.initial_layout, env.cfg)["covered_users"], 3)

    def test_near_geometry_budget_never_overshoots(self):
        s = scene()
        too_small = DrlR04Env(s, DrlR04Config(max_geometry_calls=2))
        _, info = too_small.reset()
        self.assertEqual(info["status"], "initialization_failed")
        self.assertEqual(too_small.geometry_calls, 0)
        self.assertEqual(too_small.initialization["metrics"]["aircraft_count"], 6)
        exact_init = DrlR04Env(s, DrlR04Config(max_geometry_calls=3))
        exact_init.reset()
        self.assertTrue(exact_init.done)
        self.assertTrue(exact_init.truncated)
        self.assertLessEqual(exact_init.geometry_calls, 3)
        one_trial = DrlR04Env(s, DrlR04Config(max_geometry_calls=12))
        one_trial.reset()
        action = next(i for i, (a, _) in enumerate(one_trial.menu)
                      if a.operation != "stop")
        one_trial.step(action)
        self.assertTrue(one_trial.done)
        self.assertTrue(one_trial.truncated)
        self.assertLessEqual(one_trial.geometry_calls, 12)
        self.assertGreaterEqual(one_trial.observe()[0]["global"].min(), 0.0)

    def test_preview_cache_is_logged_and_commit_reuses_real_trial(self):
        env = DrlR04Env(scene())
        env.reset()
        index = next(i for i, (a, _) in enumerate(env.menu) if a.operation == "split")
        action = env.menu[index][0]
        first = env.preview(action)
        second = env.preview(action)
        self.assertIs(first, second)
        self.assertEqual(env.trials, 1)
        self.assertEqual(env.preview_requests, 2)
        self.assertEqual(env.preview_cache_hits, 1)
        env.step(index)
        self.assertEqual(env.trials, 1)
        self.assertEqual(env.preview_requests, 3)
        self.assertEqual(env.preview_cache_hits, 2)
        self.assertEqual(env.ledger[-1]["preview_cache_hits"], 2)

    def test_preview_request_denied_by_trial_budget_is_logged(self):
        env = DrlR04Env(scene(), DrlR04Config(max_trials=1))
        env.reset()
        actions = [a for a, _ in env.menu if a.operation != "stop"]
        self.assertGreaterEqual(len(actions), 2)
        env.preview(actions[0])
        with self.assertRaisesRegex(RuntimeError, "trial budget"):
            env.preview(actions[1])
        self.assertEqual(env.preview_requests, 2)
        self.assertEqual(env.trials, 1)
        self.assertEqual(env.preview_cache_hits, 0)
        self.assertEqual(env.preview_budget_rejections, 1)

    def test_h2_branches_share_one_counter(self):
        env = DrlR04Env(scene(), DrlR04Config(max_trials=3, max_commits=2))
        result = run_h2(env)
        self.assertLessEqual(result["trials"], 3)
        self.assertGreater(result["preview_requests"], result["preview_cache_hits"])
        self.assertEqual(result["trials"], len(env._trial_cache))
        self.assertLessEqual(result["geometry_calls"], 8192)


if __name__ == "__main__":
    unittest.main()
