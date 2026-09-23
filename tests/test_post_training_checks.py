"""CPU-only checks for the frozen 18-run post-training protocol."""

import json
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.evaluation.checkpoints import load_network_checkpoint
from satellite_rl.evaluation.post_training_checks import (
    ACTION_SEEDS, EXPECTED_RUN_IDS, SCENE_IDS, EvaluationAgent,
    _final_result_is_complete, _record_failure, legacy_environment_source,
    make_timing_states, require_completed, transform_scene, verify_cached_post_result,
)
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.run_monitor import RunMonitor


class PostTrainingChecksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="satellite_post_checks_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def arrays():
        index = np.arange(100)
        return {
            "beam_location": np.column_stack((index, -index)),
            "beam_G_Tx_max": (index + 1)[:, None],
            "UE_G_Rx_max": (index + 2)[:, None],
            "distances": (index + 3)[:, None],
            "rate": (index + 100)[:, None].astype(np.float64),
            "noise_temperature": (index + 4)[:, None],
            "original_indices": 99 - index,
        }

    def test_demand_changes_only_rate(self):
        original = self.arrays()
        for label, factor in (("demand_075", 0.75), ("demand_125", 1.25)):
            with self.subTest(label=label):
                changed = transform_scene(original, label, 81000)
                np.testing.assert_array_equal(changed["rate"], original["rate"] * factor)
                for key in original.keys() - {"rate"}:
                    np.testing.assert_array_equal(changed[key], original[key])
                np.testing.assert_array_equal(changed["new_position_to_old_position"],
                                              np.arange(100))

    def test_group_reorder_preserves_identity_group_and_coupled_fields(self):
        original = self.arrays()
        changed = transform_scene(original, "within_group_order", 81000)
        permutation = changed["new_position_to_old_position"]
        np.testing.assert_array_equal(permutation % 8, np.arange(100) % 8)
        self.assertEqual(len(np.unique(permutation)), 100)
        self.assertFalse(np.array_equal(permutation, np.arange(100)))
        for key, value in original.items():
            np.testing.assert_array_equal(changed[key], value[permutation])
        np.testing.assert_array_equal(changed["new_position_to_original_identity"],
                                      original["original_indices"][permutation])
        np.testing.assert_array_equal(
            transform_scene(original, "within_group_order", 81000)["new_position_to_old_position"],
            permutation,
        )

    def test_incomplete_queue_refuses_before_loading_any_weight(self):
        exp = self.root / "outputs" / "experiments" / "sac_comparison_20260924_r2"
        exp.mkdir(parents=True)
        plan = {"steps": 400000,
                "source": {"git_commit": "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8",
                           "source_root": str(self.root / "frozen")},
                "jobs": [{"run_id": run_id} for run_id in sorted(EXPECTED_RUN_IDS)]}
        (exp / "suite_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (exp / "suite_status.json").write_text(json.dumps({
            "counts": {"completed": 17, "pending": 0, "running": 1, "failed": 0}
        }), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "尚未全部完成"):
            require_completed(self.root, self.root / "frozen")

    def test_final_result_requires_every_scene_and_action_once(self):
        rows = [{"scene_id": scene, "action_mode": "sample", "action_seed": seed}
                for scene in range(81000, 81100) for seed in ACTION_SEEDS]
        rows += [{"scene_id": scene, "action_mode": "argmax", "action_seed": "argmax"}
                 for scene in range(81000, 81100)]
        path = self.root / "result.json"
        arrays_path = self.root / "arrays.npz"
        trace_path = self.root / "trace.jsonl.gz"
        arrays_path.write_bytes(b"arrays")
        trace_path.write_bytes(b"trace")
        metadata = {"checkpoint": "final_test", "env_step": 400000,
                    "include_argmax": True, "arrays_path": str(arrays_path),
                    "arrays_sha256": runner.sha256_file(arrays_path),
                    "trace_path": str(trace_path), "trace_sha256": runner.sha256_file(trace_path)}
        path.write_text(json.dumps({"rows": rows, **metadata}), encoding="utf-8")
        self.assertEqual(len(_final_result_is_complete(path)["rows"]), 400)
        rows[-1] = rows[0]
        path.write_text(json.dumps({"rows": rows, **metadata}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "缺失或重复"):
            _final_result_is_complete(path)

    def test_cached_post_result_requires_exact_60_pairs_and_matching_raw_hashes(self):
        refs = {str(scene): {"path": f"scene_{scene}.npz", "sha256": "fixed"}
                for scene in SCENE_IDS}
        arrays = self.root / "result.npz"
        trace = self.root / "result.jsonl.gz"
        arrays.write_bytes(b"actual arrays")
        trace.write_bytes(b"actual trace")
        rows = [{"scene_id": scene, "action_seed": seed, "action_mode": "sample"}
                for scene in SCENE_IDS for seed in ACTION_SEEDS]
        result = {"checkpoint": "final_test", "env_step": 400000,
                  "include_argmax": False, "action_seeds": list(ACTION_SEEDS),
                  "scene_references": refs, "rows": rows,
                  "arrays_path": str(arrays), "arrays_sha256": runner.sha256_file(arrays),
                  "trace_path": str(trace), "trace_sha256": runner.sha256_file(trace)}
        path = self.root / "final_test_400000.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        self.assertEqual(len(verify_cached_post_result(path, refs)["rows"]), 60)
        result["env_step"] = "unknown"
        path.write_text(json.dumps(result), encoding="utf-8")
        self.assertEqual(len(verify_cached_post_result(
            path, refs, expected_step="unknown")["rows"]), 60)
        result["env_step"] = 400000
        result["rows"][-1] = rows[0]
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "不完整"):
            verify_cached_post_result(path, refs)
        result["rows"][-1] = {"scene_id": 81019, "action_seed": 91003,
                               "action_mode": "sample"}
        path.write_text(json.dumps(result), encoding="utf-8")
        trace.write_bytes(b"changed trace")
        with self.assertRaisesRegex(RuntimeError, "校验失败"):
            verify_cached_post_result(path, refs)

    def test_unknown_historical_step_deduplicates_on_resume(self):
        directory = self.root / "historical"
        row = {"checkpoint": "final_test", "env_step": "unknown",
               "scene_id": "81000", "action_mode": "sample", "action_seed": 91001}
        first = RunMonitor(directory, "historical")
        first.row("evaluations", row)
        first.close()
        resumed = RunMonitor(directory, "historical", resume=True)
        resumed.row("evaluations", row)
        resumed.close()
        with (directory / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_failure_records_are_unique_and_latest_is_only_an_index(self):
        paths = []
        for index in range(2):
            try:
                raise RuntimeError(f"failure {index}")
            except RuntimeError as error:
                paths.append(_record_failure(self.root, "evaluation", error,
                                             {"trial": index}, latest_name="failure.json"))
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(paths[0].exists() and paths[1].exists())
        index = json.loads((self.root / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(index["latest_failure"], str(paths[1]))
        self.assertIn("failure 0", paths[0].read_text(encoding="utf-8"))

    def test_old_source_and_pretrained_transformer_load_strictly_on_cpu(self):
        module, path, digest = legacy_environment_source(self.root)
        self.assertTrue(path.is_file())
        self.assertEqual(len(digest), 64)
        self.assertTrue(hasattr(module, "Env"))
        agent = EvaluationAgent("transformer", torch.device("cpu"))
        weight = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "sac_transformer" / "actor.pth"
        load_network_checkpoint(agent.actor, weight, algorithm="pretrained-transformer",
                                map_location="cpu", evaluation_mode=True)
        self.assertFalse(agent.actor.training)
        np.random.seed(81000)
        beam_info = BeamInfo()
        frequency = module.SatelliteFreqState()
        current = module.CurrentBeamState(beam_info.beam_info)
        environment = module.Env(frequency, beam_info, current)
        state = environment.reset(beam_info)
        self.assertEqual(state.shape, (204,))
        next_state, reward, done, info = environment.step([0, 0, 5, 10 * np.log10(20)])
        self.assertEqual(next_state.shape, (204,))
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(int(np.asarray(info["out"]).item()), 0)

    def test_reference_actions_reach_actual_intermediate_states(self):
        np.random.seed(81000)
        beam_info = BeamInfo()
        path = self.root / "scene_81000.npz"
        digest = runner._save_scene(path, beam_info)
        plan = {"test_scenes": {"81000": {"path": str(path), "sha256": digest}}}
        saved, _, states = make_timing_states(plan, self.root, (81000,))
        self.assertEqual(states.shape, (4, 204))
        with np.load(saved) as data:
            np.testing.assert_array_equal(data["before_beam_index"], [0, 25, 50, 75])
            self.assertEqual(data["requested_reference_actions"].shape, (76, 6))
            np.testing.assert_array_equal(data["requested_reference_actions"],
                                          data["actual_reference_actions"])


if __name__ == "__main__":
    unittest.main()
