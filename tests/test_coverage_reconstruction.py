"""Deterministic constraint checks for the new 100-beam reconstruction core."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.coverage.geometry import satellite_rays_numpy, squared_ray_limit
from satellite_rl.coverage.reconstruction import (
    ReconstructionAction, ReconstructionConfig, ReconstructionState,
    _geometry, action_candidates, apply_action, initial_state, state_metrics,
)
from satellite_rl.coverage.scenes import CoverageScene


def scene_fixture(*, impossible=False, isolated=False):
    # 97 distant ordinary beams plus the three beams used in a macro action.
    grid = np.array([(35.0 + 2.0 * r, -18.0 + 4.0 * c)
                     for r in range(10) for c in range(10)], dtype=float)
    centers = grid.copy()
    centers[:3] = ([40.0, 0.0], [40.10, 0.0], [50.0, 10.0])
    positions = centers.tolist()
    demands = [20.0] * 100
    positions.append([50.04, 10.0])
    demands.append(180.0)
    demands[0], demands[1], demands[2] = 30.0, 30.0, 90.0
    if impossible:
        positions[0] = [38.5, 0.0]
        positions[1] = [43.5, 0.0]
        centers[0] = [40.0, 0.0]
        centers[1] = [42.0, 0.0]
    if isolated:
        # Isolated tiny component: certificate protects its dedicated beam.
        positions[3] = [62.0, -10.0]
        centers[3] = positions[3]
        demands[3] = 1.0
    return CoverageScene("tiny", np.asarray(positions), np.asarray(demands),
                         centers, [f"aircraft-{i}" for i in range(len(positions))],
                         source_hash="fixture-sha")


def near_scene(*, for_action=False):
    base = scene_fixture()
    centers = base.centers.copy()
    positions = base.positions.copy()
    centers[2] = [61.0, -10.0] if for_action else [60.5, -10.0]
    positions[2] = [60.0, -10.0]
    positions[100] = [62.0, -10.0] if for_action else [61.0, -10.0]
    many_positions = np.repeat(positions, 10, axis=0)
    many_demands = np.repeat(base.demands, 10)
    missing = [65.0, -10.0]
    return CoverageScene("near", np.vstack((many_positions, missing)),
                         np.append(many_demands, 1.0), centers,
                         [f"near-{i}" for i in range(len(many_demands) + 1)],
                         source_hash="near-fixture-sha")


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.config = ReconstructionConfig(max_geometry_calls_per_action=512,
                                           max_geometry_calls_total=16384)

    def _state(self, scene):
        state = ReconstructionState.from_scene(scene, config=self.config)
        self.assertTrue(state.initialization_valid)
        self.assertEqual(state_metrics(state, self.config)["empty_beams"], 0)
        return state

    def test_merge_split_is_atomic_and_preserves_all_aircraft(self):
        state = self._state(scene_fixture())
        action = ReconstructionAction("merge_split", (0, 1, 2), 0)
        self.assertIn(action, action_candidates(state, self.config))
        changed, info, reward, done, metrics = apply_action(state, action, self.config)
        self.assertTrue(info["accepted"], info)
        self.assertFalse(done)
        self.assertEqual(len(changed.centers), 100)
        self.assertTrue(np.all(changed.loads > 0))
        self.assertEqual(metrics["coverage_users"], 1.0)
        self.assertEqual(metrics["coverage_demand"], 1.0)
        self.assertAlmostEqual(changed.loads.sum(), state.loads.sum())
        self.assertTrue(np.array_equal(state.assignment[:3], [0, 1, 2]))
        self.assertEqual(set(changed.assignment[[0, 1]]), {0})
        self.assertEqual(set(changed.assignment[[2, 100]]), {1, 2})
        rays = satellite_rays_numpy(scene_fixture().positions)
        beam_rays = satellite_rays_numpy(changed.centers)
        self.assertTrue(np.all(((rays - beam_rays[changed.assignment]) ** 2).sum(-1)
                               <= squared_ray_limit(.25) + 1e-14))
        self.assertAlmostEqual(reward, metrics["quality"] - state_metrics(state, self.config)["quality"])
        self.assertGreater(info["ray_pair_checks"], 0)

    def test_impossible_union_rolls_back_and_spends_attempt(self):
        state = self._state(scene_fixture(impossible=True))
        cache_before = state._distance_cache.copy()
        calls_before = state.geometry_calls
        action = ReconstructionAction("merge_split", (0, 1, 2), 0)
        self.assertIn(action, action_candidates(state, self.config))
        changed, info, reward, _, _ = apply_action(state, action, self.config)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "no_legal_center_for_group")
        self.assertTrue(np.array_equal(changed.centers, state.centers))
        self.assertTrue(np.array_equal(changed.assignment, state.assignment))
        self.assertTrue(np.array_equal(changed._distance_cache, cache_before))
        self.assertTrue(np.array_equal(state._distance_cache, cache_before))
        self.assertEqual(state.geometry_calls, calls_before)
        self.assertEqual(changed.steps, 1)
        self.assertEqual(changed.attempts, 1)
        self.assertGreater(info["geometry_calls"], 0)
        self.assertEqual(reward, 0.0)

    def test_fixed_edge_certificate_cannot_be_selected(self):
        state = self._state(scene_fixture(isolated=True))
        self.assertIn(3, state.edge_beam_ids)
        self.assertTrue(all(3 not in action.beams for action in action_candidates(state, self.config)))

    def test_stop_budget_reward_telescope_and_serialization(self):
        state = self._state(scene_fixture())
        first, _, r1, _, _ = apply_action(
            state, ReconstructionAction("merge_split", (0, 1, 2), 0), self.config)
        final, info, r2, done, metrics = apply_action(
            first, ReconstructionAction("stop"), self.config)
        self.assertTrue(done)
        self.assertEqual(info["terminal_reason"], "stop")
        self.assertAlmostEqual(r1 + r2,
                               metrics["quality"] - state_metrics(state, self.config)["quality"])
        restored = ReconstructionState.from_record(
            state.scene, json.loads(json.dumps(final.to_record())), self.config)
        self.assertTrue(np.array_equal(restored.assignment, final.assignment))
        self.assertEqual(restored.geometry_calls, final.geometry_calls)
        self.assertEqual(restored.best_quality, final.best_quality)
        self.assertEqual(action_candidates(restored, self.config), [])
        with self.assertRaises(ValueError):
            apply_action(restored, ReconstructionAction("stop"), self.config)
        one_step = ReconstructionConfig(max_macro_steps=1,
                                        max_geometry_calls_per_action=512)
        started = ReconstructionState.from_scene(scene_fixture(), config=one_step)
        ended, _, _, done, _ = apply_action(started, ReconstructionAction("stop"), one_step)
        self.assertTrue(done)
        self.assertEqual(ended.steps, 1)

    def test_common_start_recomputes_candidates_and_records_timing(self):
        scene = scene_fixture()
        state = initial_state(scene, self.config)
        self.assertTrue(state.initialization_valid)
        self.assertGreater(state.elapsed_initialization_s, 0)
        self.assertEqual({row["source"] for row in state.initialization_candidates},
                         {"scene_centers_seed", "recomputed_cluster", "recomputed_rule"})
        row = json.loads(json.dumps(state.to_record()))
        restored = ReconstructionState.from_record(scene, row, self.config)
        self.assertEqual(restored.initialization_source, state.initialization_source)

    def test_geometry_budget_and_certificate_record_are_hard_limits(self):
        tiny_budget = ReconstructionConfig(max_geometry_calls_per_action=1,
                                           max_geometry_calls_total=1)
        state = ReconstructionState.from_scene(scene_fixture(impossible=True),
                                               config=tiny_budget)
        rejected, info, reward, done, _ = apply_action(
            state, ReconstructionAction("merge_split", (0, 1, 2), 0), tiny_budget)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "action_geometry_budget_exhausted")
        self.assertEqual(rejected.geometry_calls, 1)
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)
        protected = self._state(scene_fixture(isolated=True))
        record = json.loads(json.dumps(protected.to_record()))
        record["edge_beam_ids"] = []
        with self.assertRaisesRegex(ValueError, "edge beam identity"):
            ReconstructionState.from_record(protected.scene, record, self.config)

    def test_covered_but_unassigned_is_rejected_on_create_and_restore(self):
        scene = scene_fixture()
        assignment = ReconstructionState.from_scene(scene, config=self.config).assignment
        assignment[0] = -1
        with self.assertRaisesRegex(ValueError, "geometrically covered"):
            ReconstructionState.from_scene(scene, assignment=assignment, config=self.config)
        state = ReconstructionState.from_scene(scene, config=self.config)
        record = json.loads(json.dumps(state.to_record()))
        record["assignment"][0] = -1
        with self.assertRaisesRegex(ValueError, "geometrically covered"):
            ReconstructionState.from_record(scene, record, self.config)

    def test_newly_covered_aircraft_gets_one_owner(self):
        scene = near_scene(for_action=True)
        state = ReconstructionState.from_scene(scene, config=self.config)
        self.assertTrue(state.initialization_valid)
        self.assertEqual(state.assignment[-1], -1)
        action = ReconstructionAction("merge_split", (0, 1, 2), 0)
        self.assertIn(action, action_candidates(state, self.config))
        changed, info, _, _, metrics = apply_action(state, action, self.config)
        self.assertTrue(info["accepted"], info)
        self.assertGreaterEqual(changed.assignment[-1], 0)
        self.assertTrue(metrics["coverage_100"])
        self.assertAlmostEqual(changed.loads.sum(), scene.demands.sum())

    def test_near_start_attempts_strict_repair_on_separate_budget(self):
        scene = near_scene()
        near = ReconstructionState.from_scene(scene, config=self.config)
        self.assertTrue(near.initialization_valid)
        self.assertFalse(state_metrics(near, self.config)["coverage_100"])
        with patch("satellite_rl.coverage.reconstruction._initial_centers",
                   return_value=(scene.centers.copy(), {"initializer": "fixture"})):
            repaired = initial_state(scene, self.config)
        self.assertTrue(state_metrics(repaired, self.config)["coverage_100"])
        self.assertGreater(repaired.initialization_repair_geometry_calls, 0)
        self.assertGreater(repaired.initialization_repair_attempts, 0)
        self.assertEqual(repaired.geometry_calls, 0)
        self.assertLessEqual(repaired.initialization_repair_geometry_calls,
                             self.config.max_initialization_geometry_calls)
        restored = ReconstructionState.from_record(
            scene, json.loads(json.dumps(repaired.to_record())), self.config)
        self.assertEqual(restored.initialization_repair_geometry_calls,
                         repaired.initialization_repair_geometry_calls)

    def test_assign_template_zero_updates_current_loads(self):
        scene = scene_fixture()
        scene.positions = np.vstack((scene.positions, [[40.02, 0.0], [40.04, 0.0],
                                                       [40.10, 0.0]]))
        scene.demands = np.append(scene.demands, [100.0, 100.0, 20.0])
        scene.demands[0] = 100.0
        scene.demands[1] = 20.0
        scene.aircraft_ids += ["extra-0", "extra-1", "extra-2"]
        state = ReconstructionState.from_scene(scene, config=self.config)
        before_gap = abs(state.loads[0] - state.loads[1])
        changed, info, _, _, _ = apply_action(
            state, ReconstructionAction("assign", (0, 1), 0), self.config)
        self.assertTrue(info["accepted"], info)
        self.assertGreater(changed.loads[0], 0)
        self.assertGreater(changed.loads[1], 0)
        self.assertLess(abs(changed.loads[0] - changed.loads[1]), before_gap)

    def test_complete_candidate_menu_survives_beam_id_permutation(self):
        scene = scene_fixture()
        # Most ordinary beams deliberately have equal demand; tie-breaking
        # must use their GEO positions rather than their incidental IDs.
        original = ReconstructionState.from_scene(scene, config=self.config)
        rng = np.random.default_rng(51001)
        permutation = rng.permutation(100)
        old_to_new = np.empty(100, dtype=np.int64)
        old_to_new[permutation] = np.arange(100)
        moved = ReconstructionState.from_scene(
            scene, centers=original.centers[permutation],
            assignment=np.where(original.assignment >= 0,
                                old_to_new[original.assignment.clip(min=0)], -1),
            config=self.config)
        expected = {ReconstructionAction(action.operation,
                                         tuple(int(old_to_new[b]) for b in action.beams),
                                         action.template)
                    for action in action_candidates(original, self.config)}
        actual = set(action_candidates(moved, self.config))
        self.assertEqual(actual, expected)

    def test_candidate_ties_with_identical_centres_are_symmetric(self):
        scene = scene_fixture()
        scene.centers[1] = scene.centers[0]
        scene.positions[1] = scene.positions[0]
        assignment = ReconstructionState.from_scene(scene_fixture(),
                                                    config=self.config).assignment
        state = ReconstructionState.from_scene(scene, assignment=assignment,
                                               config=self.config)
        permutation = np.arange(100)
        permutation[[0, 1]] = [1, 0]
        remap = permutation.copy()  # This swap is its own inverse.
        moved = ReconstructionState.from_scene(
            scene, centers=state.centers[permutation],
            assignment=np.where(state.assignment >= 0,
                                remap[state.assignment.clip(min=0)], -1),
            config=self.config)
        expected = {ReconstructionAction(action.operation,
                                         tuple(int(remap[b]) for b in action.beams),
                                         action.template)
                    for action in action_candidates(state, self.config)}
        self.assertEqual(set(action_candidates(moved, self.config)), expected)

    def test_accepted_geometry_cache_matches_restored_two_step_path(self):
        scene = scene_fixture()
        initial = self._state(scene)
        first_action = ReconstructionAction("merge_split", (0, 1, 2), 0)
        first, first_info, _, _, _ = apply_action(initial, first_action, self.config)
        self.assertTrue(first_info["accepted"])
        self.assertTrue(np.array_equal(first._distance_cache,
                                       _geometry(scene, first.centers)))
        restored = ReconstructionState.from_record(
            scene, json.loads(json.dumps(first.to_record())), self.config)
        second_action = ReconstructionAction("merge_split", (1, 2, 0), 0)
        source_cache = first._distance_cache.copy()
        direct, direct_info, direct_reward, direct_done, direct_metrics = apply_action(
            first, second_action, self.config)
        replay, replay_info, replay_reward, replay_done, replay_metrics = apply_action(
            restored, second_action, self.config)
        self.assertTrue(direct_info["accepted"])
        self.assertEqual(direct_info["accepted"], replay_info["accepted"])
        self.assertEqual(direct_info["reason"], replay_info["reason"])
        self.assertTrue(np.array_equal(direct.centers, replay.centers))
        self.assertTrue(np.array_equal(direct.assignment, replay.assignment))
        self.assertTrue(np.array_equal(direct.loads, replay.loads))
        self.assertTrue(np.array_equal(direct._distance_cache,
                                       _geometry(scene, direct.centers)))
        self.assertAlmostEqual(direct_reward, replay_reward)
        self.assertEqual(direct_done, replay_done)
        self.assertEqual(direct_metrics, replay_metrics)
        for field in ("geometry_calls", "fit_candidate_checks",
                      "assignment_candidate_checks", "ray_pair_checks", "attempts"):
            self.assertEqual(getattr(direct, field), getattr(replay, field))
        self.assertEqual(direct_info["geometry_calls"], replay_info["geometry_calls"])
        self.assertTrue(np.array_equal(first._distance_cache, source_cache))
        stopped, _, _, _, _ = apply_action(first, ReconstructionAction("stop"), self.config)
        self.assertTrue(np.array_equal(stopped._distance_cache, source_cache))
        self.assertTrue(np.array_equal(first._distance_cache, source_cache))


if __name__ == "__main__":
    unittest.main()
