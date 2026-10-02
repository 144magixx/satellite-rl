"""Short worker integration: frozen inputs, durable cases, and recovery."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.power_refinement.config import default_config as refinement_defaults
from satellite_rl.power_refinement.runner import evaluate_refinement
from satellite_rl.workbench.config import default_config as model_defaults
from satellite_rl.workbench.power_refinement import G11_M1_RUN_ID, G11_SOURCE_COMMIT
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import Store, digest, read_json


class FixedPolicy:
    def __init__(self, config, provenance):
        self.config = config
        self.metadata = {"provenance": provenance}
        self.agent = type("Agent", (), {"rng": random.Random(51001)})()

    def reset_episode(self):
        pass

    def act(self, state, deterministic=True, action_mask=None):
        if action_mask is None:
            return [0, 0, 3]
        frequency, slot = np.argwhere(action_mask["frequency_slots"])[0]
        power = 3 if action_mask["power"][3] else int(np.flatnonzero(action_mask["power"])[0])
        return [int(frequency), int(slot), power]


class RefinementRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = Application(self.root, source_root=self.root)
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        frozen = patch.object(Store, "freeze_source", return_value=(snapshot, "f" * 40))
        frozen.start()
        self.addCleanup(frozen.stop)
        model_config = model_defaults()
        model_config["name"] = "G11 M1"
        model_config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                                       temporal_mode="previous_action", dropout=0.0)
        model_config["state"].update(
            features=["beam", "power", "interference", "occupancy",
                      "natural_global_context", "previous_actual_action"],
            demand_visibility="known", context_level="compact")
        model_config["reward"]["preset"] = "legacy_repaired"
        model_config["environment"].update(profile="europe_v1", power_budget_w=3000,
                                            action_mask="joint_feasible")
        model_config["training"].update(seed=51001, total_steps=400000, device="cuda",
                                        scene_order="shuffle_each_cycle")
        model_dir = self.root / "source_model"
        model_dir.mkdir()
        (model_dir / "policy.pt").write_bytes(b"frozen short-test policy")
        (model_dir / "model.json").write_text("{}", encoding="utf-8")
        provenance = {"run_id": G11_M1_RUN_ID, "git_commit": G11_SOURCE_COMMIT,
                      "env_step": 400000, "scene_sets": {
            "train": {"source": {"source_dates": ["2024-01-08"]},
                      "files": [{"sha256": "a" * 64}]},
            "validation": {"source": {"source_dates": ["2024-01-15"]},
                           "files": [{"sha256": "b" * 64}]},
        }}
        self.model = {"id": "M1", "name": "G11 M1", "path": str(model_dir),
                      "config": model_config, "provenance": provenance,
                      "sha256": digest(model_dir / "policy.pt")}
        self.policy = FixedPolicy(model_config, provenance)
        model_list = patch.object(self.app, "models", return_value={"models": [self.model]})
        model_list.start()
        self.addCleanup(model_list.stop)
        loader = patch("satellite_rl.workbench.checkpoints.load_policy", return_value=self.policy)
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        scene = self.app.api("POST", ["scenes"], {},
                             {"name": "one short test scene", "seed": 94100,
                              "count": 1, "profile": "europe_v1"})
        scene["source"].update(partition_role="test", source_dates=["2024-02-03"])
        self.app.store.save_record("scenes", scene)
        controls = refinement_defaults()
        controls.update(repeats=1, maxiter=2, maxeval=4, max_time_seconds=1.0)
        self.job = self.app.api("POST", ["power-refinement", "runs"], {},
                                {"model_id": "M1", "scene_id": scene["id"],
                                 "config": controls, "request_id": "runner_short_fixture"})
        self.directory = self.app.store.job_dir(self.job["run_id"])
        self.config = read_json(self.directory / "config.json")
        self._solver_calls = 0

    def unchanged_solver(self, problem, initial_power, method="none", options=None,
                         callback=None, stop_requested=None):
        self._solver_calls += 1
        evaluation = problem.evaluate(np.asarray(initial_power, dtype=float))
        return {"power_w": np.asarray(initial_power, dtype=float).tolist(),
                "elapsed_seconds": 0.001, "evaluations": 1,
                "termination_reason": "unchanged", "fallback_used": False,
                "converged": True,
                "history": [{"elapsed_seconds": 0.0,
                             "final_si_mean": evaluation.si,
                             "final_sgm_mean": evaluation.sgm}]}

    def run_short(self):
        with patch("satellite_rl.power_refinement.optimizer.optimize_power",
                   side_effect=self.unchanged_solver):
            return evaluate_refinement(self.directory, self.job, self.config, device="cpu")

    def test_complete_worker_writes_paired_cases_and_final_beams(self):
        summary = self.run_short()
        self.assertEqual(set(summary), {"none", "discrete", "slsqp"})
        self.assertEqual(self._solver_calls, 3)
        result = read_json(self.directory / "results.json")
        self.assertEqual(len(result["scenes"]), 3)
        self.assertEqual({row["initial_allocation_sha256"] for row in result["scenes"]},
                         {result["scenes"][0]["initial_allocation_sha256"]})
        self.assertEqual(len(result["scenes"][0]["beams"]), 100)
        self.assertEqual(result["metadata"]["training_interactions"], 0)
        self.assertAlmostEqual(result["scenes"][0]["metrics"]["final_si_mean"],
                               result["scenes"][0]["initial_metrics"]["final_si_mean"], places=8)
        with (self.directory / "power_results.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)
        with (self.directory / "power_beams.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 300)
        self.assertEqual(len((self.directory / "power_cases.jsonl").read_text(encoding="utf-8").splitlines()), 3)
        self.assertTrue((self.directory / "power_report.md").is_file())
        self.assertEqual(read_json(self.directory / "status.json")["env_step"], 3)
        self.assertEqual(read_json(self.directory / "status.json")["stage"], "completed")

    def test_same_run_resume_does_not_repeat_inference_or_solve(self):
        from satellite_rl.power_refinement import runner as worker
        with patch.object(worker, "_evaluation_episode", wraps=worker._evaluation_episode) as inference, \
             patch("satellite_rl.power_refinement.optimizer.optimize_power",
                   side_effect=self.unchanged_solver):
            worker.evaluate_refinement(self.directory, self.job, self.config, device="cpu")
            self.assertEqual(inference.call_count, 1)
            self.assertEqual(self._solver_calls, 3)
            worker.evaluate_refinement(self.directory, self.job, self.config, device="cpu")
            self.assertEqual(inference.call_count, 1)
            self.assertEqual(self._solver_calls, 3)

    def test_stop_after_first_saved_case_then_resume(self):
        from satellite_rl.power_refinement import runner as worker
        original_status = worker._status
        def stop_after_first(directory, job, **fields):
            original_status(directory, job, **fields)
            if fields.get("stage") == "evaluating" and fields.get("env_step") == 1:
                (directory / "stop.request").write_text("stop", encoding="utf-8")
        with patch.object(worker, "_status", side_effect=stop_after_first), \
             patch("satellite_rl.power_refinement.optimizer.optimize_power",
                   side_effect=self.unchanged_solver):
            self.assertIsNone(worker.evaluate_refinement(self.directory, self.job, self.config, device="cpu"))
        self.assertEqual(self._solver_calls, 1)
        self.assertEqual(read_json(self.directory / "status.json")["stage"], "stopped")
        self.assertEqual(len(list((self.directory / "power_cases").glob("*.json"))), 1)
        (self.directory / "stop.request").unlink()
        self.run_short()
        self.assertEqual(self._solver_calls, 3)
        self.assertEqual(read_json(self.directory / "status.json")["stage"], "completed")

    def test_frozen_model_and_scene_sha_are_checked_before_evaluation(self):
        model_file = Path(self.job["model_path"]) / "policy.pt"
        original_model = model_file.read_bytes()
        model_file.write_bytes(b"tampered model")
        with self.assertRaisesRegex(ValueError, "冻结模型SHA"):
            self.run_short()
        self.assertEqual(self._solver_calls, 0)
        model_file.write_bytes(original_model)
        scene_file = Path(self.job["scene_sets"]["test"]["files"][0]["path"])
        scene_file.write_bytes(b"tampered scene")
        with self.assertRaisesRegex(ValueError, "校验"):
            self.run_short()
        self.assertEqual(self._solver_calls, 0)

    def test_saved_case_capacity_tampering_is_rejected_on_resume(self):
        self.run_short()
        case = next((self.directory / "power_cases").glob("*none.json"))
        row = json.loads(case.read_text(encoding="utf-8"))
        row["beams"][0]["rate_bps"] *= 1.5
        case.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "容量|校验|已保存"):
            self.run_short()

    def test_saved_case_metric_tampering_is_rejected_on_resume(self):
        self.run_short()
        case = next((self.directory / "power_cases").glob("*none.json"))
        row = json.loads(case.read_text(encoding="utf-8"))
        row["metrics"]["final_si_mean"] += 0.05
        case.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "指标|逐束重算"):
            self.run_short()


if __name__ == "__main__":
    unittest.main()
