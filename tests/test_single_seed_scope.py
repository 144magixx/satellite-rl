"""CPU-only checks for the six-run completion gate."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from satellite_rl.evaluation.single_seed_scope import (
    EXPERIMENT_ID, RUN_IDS, TRAIN_COMMIT,
    _checkpoint_pair, _result, require_completed,
)
from satellite_rl.training import experiment_runner as runner


class SingleSeedScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="single_seed_scope_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _result_fixture(self, checkpoint):
        scene_refs = {str(scene): {"path": f"scene_{scene}.npz", "sha256": "fixed"}
                      for scene in range(81000, 81100)}
        seeds = list(runner.ACTION_SEEDS)
        rows = [{"scene_id": scene, "action_mode": "sample", "action_seed": seed}
                for scene in range(81000, 81100) for seed in seeds]
        rows += [{"scene_id": scene, "action_mode": "argmax", "action_seed": "argmax"}
                 for scene in range(81000, 81100)]
        arrays = self.root / "rates.npz"
        trace = self.root / "steps.jsonl.gz"
        arrays.write_bytes(b"all beams")
        trace.write_bytes(b"all requested and actual actions")
        data = {"checkpoint": checkpoint, "env_step": 400000,
                "include_argmax": True, "scene_references": scene_refs,
                "action_seeds": seeds, "rows": rows,
                "arrays_path": str(arrays), "arrays_sha256": runner.sha256_file(arrays),
                "trace_path": str(trace), "trace_sha256": runner.sha256_file(trace)}
        path = self.root / (checkpoint + ".json")
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, scene_refs, seeds, data

    def test_result_needs_exact_400_rows_and_verified_raw_files(self):
        path, scenes, seeds, data = self._result_fixture("final_test")
        self.assertEqual(len(_result(path, "final_test", scenes, seeds)["rows"]), 400)
        data["rows"][-1] = data["rows"][0]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "100场景"):
            _result(path, "final_test", scenes, seeds)
        data["rows"][-1] = {"scene_id": 81099, "action_mode": "argmax",
                            "action_seed": "argmax"}
        path.write_text(json.dumps(data), encoding="utf-8")
        (self.root / "steps.jsonl.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "校验失败"):
            _result(path, "final_test", scenes, seeds)

    def test_complete_checkpoint_and_actor_must_match(self):
        networks = {name: {"weight": torch.tensor([1.0])} for name in runner.NETWORKS}
        optimizers = {name: {} for name in runner.OPTIMIZERS}
        payload = {"format_version": 1, "config_hash": "fixed",
                   "progress": {"env_step": 400000},
                   "agent": {"networks": networks, "optimizers": optimizers,
                             "log_alpha": torch.tensor(-1.0)},
                   "replay_capacity": 20000, "replay": [], "rng": {}}
        full, actor = self.root / "final.pt", self.root / "final_actor.pt"
        torch.save(payload, full)
        torch.save(networks["actor"], actor)
        self.assertEqual(_checkpoint_pair(full, actor, "fixed", final=True)["checkpoint_step"],
                         400000)
        torch.save({"weight": torch.tensor([2.0])}, actor)
        with self.assertRaisesRegex(RuntimeError, "不一致"):
            _checkpoint_pair(full, actor, "fixed", final=True)

    def _plan_fixture(self):
        exp = self.root / "outputs" / "experiments" / EXPERIMENT_ID
        exp.mkdir(parents=True)
        frozen = self.root / "frozen"
        frozen.mkdir()
        scene = self.root / "scene.npz"
        scene.write_bytes(b"saved actual scene")
        jobs = [{"run_id": f"{model}_{formula}_s{seed}",
                 "model": model, "formula": formula, "seed": seed,
                 "state": "running", "pid": 123}
                for seed in runner.TRAIN_SEEDS
                for model in runner.MODELS for formula in ("legacy", "corrected")]
        plan = {"jobs": jobs, "steps": 400000,
                "source": {"git_commit": TRAIN_COMMIT, "source_root": str(frozen)},
                "test_scenes": {str(seed): {"path": str(scene),
                                            "sha256": runner.sha256_file(scene)}
                                for seed in range(81000, 81100)},
                "action_seeds": list(runner.ACTION_SEEDS)}
        (exp / "suite_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (exp / "suite_status.json").write_text(json.dumps({
            "counts": {"pending": 12, "running": 6, "completed": 0}}), encoding="utf-8")
        (exp / "suite.lock").write_text("stale", encoding="utf-8")
        for run_id in RUN_IDS:
            run_dir = exp / run_id
            run_dir.mkdir()
            (run_dir / "status.json").write_text(json.dumps({
                "stage": "completed", "env_step": 400000, "update_step": 398000,
            }), encoding="utf-8")
        return exp, frozen

    def test_stale_suite_status_and_lock_do_not_override_six_real_runs(self):
        _, frozen = self._plan_fixture()
        with mock.patch("satellite_rl.evaluation.single_seed_scope._frozen_source",
                        return_value=frozen), \
             mock.patch("satellite_rl.evaluation.single_seed_scope._completed_run",
                        side_effect=lambda *args: {"job": args[-1]}):
            _, filtered, final = require_completed(self.root, frozen)
        self.assertEqual(len(filtered["jobs"]), 6)
        self.assertEqual(set(final), RUN_IDS)
        self.assertTrue(all(job["state"] == "completed" and job["suite_plan_state"] == "running"
                            and job["pid"] is None for job in filtered["jobs"]))
        self.assertEqual(filtered["single_seed_output_dir"],
                         str(self.root / "outputs" / "experiments" / EXPERIMENT_ID / "single_seed_51001"))

    def test_one_incomplete_run_refuses_before_weight_loading(self):
        exp, frozen = self._plan_fixture()
        first = next(iter(RUN_IDS))
        (exp / first / "status.json").write_text(json.dumps({
            "stage": "training", "env_step": 399999, "update_step": 397999,
        }), encoding="utf-8")
        with mock.patch("satellite_rl.evaluation.single_seed_scope._frozen_source",
                        return_value=frozen), \
             mock.patch("satellite_rl.evaluation.single_seed_scope._completed_run") as weights:
            with self.assertRaisesRegex(RuntimeError, "尚未全部完成"):
                require_completed(self.root, frozen)
            weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
