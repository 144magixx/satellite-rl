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
from satellite_rl.power_refinement.runner import evaluate_refinement, verify_solution
from satellite_rl.workbench.config import default_config as model_defaults
from satellite_rl.workbench.power_refinement import (G11_T0_RUN_ID, G11_T0_SOURCE_COMMIT,
                                                     G11_T0_ENV_STEP)
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import Store, digest, read_json


class FixedPolicy:
    def __init__(self, config, provenance):
        self.config = config
        self.metadata = {"provenance": provenance,
                         "experiment_modes": {"alpha_mode": "fixed", "fixed_alpha": 0.01,
                                              "temporal_mode": "none", "graph_mode": "none"}}
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
        model_config["name"] = "G11 T0"
        model_config["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                                       temporal_mode="none", graph_mode="none", dropout=0.0)
        model_config["state"].update(
            features=["beam", "power", "interference", "occupancy",
                      "natural_global_context"],
            demand_visibility="known", context_level="compact")
        model_config["reward"]["preset"] = "legacy_repaired"
        model_config["environment"].update(profile="europe_v1", power_budget_w=3000,
                                            action_mask="joint_feasible")
        model_config["training"].update(seed=51001, total_steps=G11_T0_ENV_STEP, device="cuda",
                                        scene_order="shuffle_each_cycle", alpha_mode="fixed",
                                        fixed_alpha=0.01)
        model_dir = self.root / "source_model"
        model_dir.mkdir()
        (model_dir / "policy.pt").write_bytes(b"frozen short-test policy")
        (model_dir / "model.json").write_text("{}", encoding="utf-8")
        provenance = {"run_id": G11_T0_RUN_ID, "git_commit": G11_T0_SOURCE_COMMIT,
                      "env_step": G11_T0_ENV_STEP, "scene_sets": {
            "train": {"source": {"source_dates": ["2024-01-08"]},
                      "files": [{"sha256": "a" * 64}]},
            "validation": {"source": {"source_dates": ["2024-01-15"]},
                           "files": [{"sha256": "b" * 64}]},
        }}
        self.model = {"id": "T0", "name": "G11 T0", "path": str(model_dir),
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
        controls.update(methods=["none", "discrete", "slsqp"], repeats=1, maxiter=2, maxeval=4, max_time_seconds=1.0)
        self.job = self.app.api("POST", ["power-refinement", "runs"], {},
                                {"model_id": "T0", "scene_id": scene["id"],
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

    def test_seven_methods_keep_one_initial_action_and_store_real_stage_timing(self):
        controls = refinement_defaults()
        controls.update(repeats=1, maxiter=2, maxeval=4, max_time_seconds=1.0)
        self.job = self.app.api("POST", ["power-refinement", "runs"], {}, {
            "model_id": self.job["model_id"], "scene_id": self.job["scene_id"],
            "config": controls, "request_id": "runner_seven_routes"})
        self.directory = self.app.store.job_dir(self.job["run_id"])
        self.config = read_json(self.directory / "config.json")
        observed = {}
        def record_options(problem, initial_power, method="none", options=None, **kwargs):
            observed[method] = deepcopy(options)
            return self.unchanged_solver(problem, initial_power, method, options, **kwargs)
        with patch("satellite_rl.power_refinement.optimizer.optimize_power", side_effect=record_options):
            summary = evaluate_refinement(self.directory, self.job, self.config, device="cpu")
        self.assertEqual(set(summary), set(controls["methods"]))
        self.assertEqual(self._solver_calls, 7)
        self.assertEqual(observed["de"]["optimizer_seed"], 63002)
        self.assertEqual(observed["pso"]["optimizer_seed"], 63003)
        self.assertEqual(observed["de"]["population_size"], 24)
        self.assertNotIn("optimizer_seed", observed["sca"])
        cases = read_json(self.directory / "results.json")["scenes"]
        self.assertEqual(len({r["initial_allocation_sha256"] for r in cases}), 1)
        self.assertEqual(len(list((self.directory / "initial_allocations").glob("*.json"))), 1)
        for row in cases:
            t = row["timing"]
            self.assertGreater(t["initialization_seconds"], 0)
            self.assertAlmostEqual(t["total_pipeline_seconds"],
                t["initialization_seconds"] + t["physics_setup_seconds"]
                + t["optimizer_wall_seconds"] + t["verification_seconds"])
        with (self.directory / "power_scene_times.csv").open(encoding="utf-8-sig", newline="") as stream:
            table = list(csv.DictReader(stream))
        self.assertEqual(len(table), 7)
        self.assertIn("optimizer_excluding_callback_seconds_sum", table[0])
        for row in table:
            self.assertAlmostEqual(float(row["total_pipeline_seconds_sum"]),
                                   float(row["total_pipeline_seconds_mean_per_action"]))
        self.assertTrue((self.directory / "power_report.md").is_file())
        self.assertEqual(read_json(self.directory / "status.json")["env_step"], 7)
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

    def priority_job(self):
        controls = refinement_defaults()
        controls.update(methods=["sca", "sca_si_lp", "sca_si_lb"], repeats=1)
        for package, version in (("cvxpy", "1.6.5"), ("clarabel", "0.11.1")):
            distribution = self.root / f"{package}-{version}.dist-info"
            distribution.mkdir(exist_ok=True)
            for name in ("METADATA", "RECORD"):
                (distribution / name).write_text(f"{package} {version} {name}", encoding="utf-8")
        with patch.dict("os.environ", {"SATELLITE_RL_PRIORITY_DEPS": str(self.root)}):
            self.job = self.app.api("POST", ["power-refinement", "runs"], {}, {
                "model_id": self.job["model_id"], "scene_id": self.job["scene_id"],
                "config": controls, "request_id": "runner_si_priority_fixture"})
        self.directory = self.app.store.job_dir(self.job["run_id"])
        self.config = read_json(self.directory / "config.json")

    def test_priority_routes_share_one_reference_and_resume_without_recomputation(self):
        self.priority_job()
        observed = []
        def priority(problem, reference_power, method, options, **kwargs):
            observed.append((method, np.asarray(reference_power).copy(), deepcopy(options)))
            solved = self.unchanged_solver(problem, reference_power, method, options, **kwargs)
            solved["phase1_best_si_mean"] = problem.evaluate(reference_power).si
            solved["best_feasible_si_mean"] = solved["phase1_best_si_mean"]
            solved["phase2_si_floor"] = solved["phase1_best_si_mean"] - 1e-8
            return solved
        with patch("satellite_rl.power_refinement.optimizer.optimize_power",
                   side_effect=self.unchanged_solver) as old, \
             patch("satellite_rl.power_refinement.priority.optimize_priority",
                   side_effect=priority) as new:
            evaluate_refinement(self.directory, self.job, self.config, device="cpu")
            self.assertEqual(old.call_count, 1)
            self.assertEqual(new.call_count, 2)
            changed_job = deepcopy(self.job)
            changed_job["priority_solver_dependency_files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "恢复任务配置或来源"):
                evaluate_refinement(self.directory, changed_job, self.config, device="cpu")
            self.assertEqual([name for name, _, _ in observed], ["sca_si_lp", "sca_si_lb"])
            np.testing.assert_array_equal(observed[0][1], observed[1][1])
            self.assertNotIn("si_stall_window", old.call_args.kwargs["options"])
            self.assertEqual(observed[0][2]["si_stall_window"], 200)
            self.assertEqual(len(observed[0][2]["baseline_service_floor_bps"]), 100)
            cases = read_json(self.directory / "results.json")["scenes"]
            summary = read_json(self.directory / "results.json")["summary"]
            self.assertEqual(summary["sca_si_lp"]["convergence_scope"],
                             "not_certified_for_original_objective")
            self.assertEqual(summary["sca_si_lb"]["convergence_scope"],
                             "not_certified_for_original_objective")
            self.assertEqual(len({r["initial_allocation_sha256"] for r in cases}), 1)
            self.assertEqual(cases[1]["reference_case_sha256"], cases[2]["reference_case_sha256"])
            for row in cases[1:]:
                timing = row["timing"]
                self.assertAlmostEqual(timing["reference_optimizer_wall_seconds"],
                                       cases[0]["timing"]["optimizer_wall_seconds"])
                self.assertAlmostEqual(timing["total_pipeline_seconds"],
                    timing["initialization_seconds"] + timing["physics_setup_seconds"] +
                    timing["reference_total_seconds"] + timing["optimizer_wall_seconds"] +
                    timing["verification_seconds"])
            evaluate_refinement(self.directory, self.job, self.config, device="cpu")
            self.assertEqual(old.call_count, 1)
            self.assertEqual(new.call_count, 2)

    def test_priority_resume_rejects_changed_reference_record(self):
        self.priority_job()
        def priority(problem, reference_power, method, options, **kwargs):
            solved = self.unchanged_solver(problem, reference_power, method, options, **kwargs)
            solved["phase1_best_si_mean"] = problem.evaluate(reference_power).si
            solved["best_feasible_si_mean"] = solved["phase1_best_si_mean"]
            return solved
        with patch("satellite_rl.power_refinement.optimizer.optimize_power",
                   side_effect=self.unchanged_solver), \
             patch("satellite_rl.power_refinement.priority.optimize_priority", side_effect=priority):
            evaluate_refinement(self.directory, self.job, self.config, device="cpu")
            reference = next((self.directory / "power_cases").glob("*_sca.json"))
            row = json.loads(reference.read_text(encoding="utf-8"))
            row["timing"]["optimizer_wall_seconds"] += 1
            reference.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "来源"):
                evaluate_refinement(self.directory, self.job, self.config, device="cpu")

    def test_priority_independent_check_allows_sgm_decline_but_guards_locked_si(self):
        self.priority_job()
        with patch("satellite_rl.power_refinement.optimizer.optimize_power",
                   side_effect=self.unchanged_solver), \
             patch("satellite_rl.power_refinement.priority.optimize_priority") as priority:
            # Only A is needed for a real frozen initial allocation and beam table.
            from satellite_rl.power_refinement import runner as worker
            old_status = worker._status
            def stop_after_a(directory, job, **fields):
                old_status(directory, job, **fields)
                if fields.get("env_step") == 1 and fields.get("stage") == "evaluating":
                    (directory / "stop.request").write_text("stop", encoding="utf-8")
            with patch.object(worker, "_status", side_effect=stop_after_a):
                self.assertIsNone(evaluate_refinement(self.directory, self.job, self.config, device="cpu"))
            priority.assert_not_called()
        reference = read_json(next((self.directory / "power_cases").glob("*_sca.json")))
        initial = read_json(reference["initial_path"])
        inflated = deepcopy(initial)
        inflated["metrics"]["final_sgm_mean"] = 1.0
        cap = reference["metrics"]["power_budget_w"]
        with self.assertRaisesRegex(ValueError, "SGM"):
            verify_solution(inflated, reference["beams"], cap, self.config["power_refinement"])
        metrics = verify_solution(inflated, reference["beams"], cap,
            self.config["power_refinement"], reference=reference,
            phase1_best_si=reference["metrics"]["final_si_mean"])
        self.assertLess(metrics["final_sgm_mean"], inflated["metrics"]["final_sgm_mean"])
        with self.assertRaisesRegex(ValueError, "锁定门槛"):
            verify_solution(inflated, reference["beams"], cap,
                self.config["power_refinement"], reference=reference,
                locked_si_floor=metrics["final_si_mean"] + 1e-5)
        crossing = deepcopy(reference["beams"])
        crossed_reference = deepcopy(reference)
        candidate_index = next(i for i, beam in enumerate(crossing)
                               if beam["rate_bps"] >= 0.1 * beam["required_bps"])
        threshold = 0.1 * crossing[candidate_index]["required_bps"]
        crossed_reference["beams"][candidate_index]["rate_bps"] = threshold
        crossing[candidate_index]["rate_bps"] = threshold - 1e-4
        loose_initial = deepcopy(initial)
        loose_initial["metrics"]["final_si_mean"] = 0.0
        crossed_reference["metrics"]["final_si_mean"] = 0.0
        with self.assertRaisesRegex(ValueError, "严格基本服务跨下"):
            verify_solution(loose_initial, crossing, cap, self.config["power_refinement"],
                            reference=crossed_reference)


if __name__ == "__main__":
    unittest.main()
