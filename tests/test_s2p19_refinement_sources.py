"""Saved S2-P19 allocations are complete physical cases, not result summaries."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from satellite_rl.baselines.greedy import greedy_policy
from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.workbench.baselines import BaselinePolicy
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv
from satellite_rl.workbench.refinement_sources import source_rows
from satellite_rl.workbench.refinement_sources import prepare_source
from satellite_rl.workbench.power_refinement import PowerRefinementApplication
from satellite_rl.power_refinement.config import default_config as refinement_defaults, VARIABLE_BEAM_METHODS
from satellite_rl.power_refinement.physics import FixedAllocationPhysics
from satellite_rl.power_refinement.runner import evaluate_refinement
from satellite_rl.workbench.s2p19_service import constructive_config
from satellite_rl.workbench.runner import _s2p19_validation_case
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.storage import Store
from satellite_rl.envs.regions import profile_metadata


def _saved_scene(count=50):
    np.random.seed(51001)
    beam = BeamInfo(profile="europe_v1", count=count)
    arrays = {key: np.asarray(value).copy() for key, value in beam.beam_info.items()}
    arrays["original_indices"] = np.asarray(beam.shuffle_indices).copy()
    return arrays


def _greedy_result():
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                  action_mask="joint_feasible")
    env = WorkbenchEnv(config)
    env.reset(scene=_saved_scene())
    policy = BaselinePolicy("greedy", env)
    trace = []
    done = False
    while not done:
        mask = env.action_mask()
        action = policy.act(None, action_mask=mask)
        _, _, done, info = env.step(action)
        trace.append({"beam_id": info["beam_id"]})
    return env.beam_results(), env.metrics(), trace


class S2P19SavedSourceTests(unittest.TestCase):
    def test_test_random_baseline_requires_all_three_seeded_cases(self):
        beams, metrics, _ = _greedy_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scenes"
            source.mkdir()
            scene_path = source / "scene.npz"
            np.savez_compressed(scene_path, **_saved_scene())
            scene_sha = digest(scene_path)
            profile = profile_metadata("europe_v1")
            record = {"id": "final", "name": "final", "path": str(source), "count": 1,
                      "profile": "europe_v1", "environment_profile": profile,
                      "environment_version": profile["environment_version"],
                      "source": {"partition_role": "test"},
                      "files": [{"path": str(scene_path), "sha256": scene_sha, "beam_count": 50}]}
            jobdir = root / "outputs" / "workbench" / "jobs" / "i0_baselines"
            (jobdir / "cases").mkdir(parents=True)
            rows = []
            for repeat in range(3):
                case_path = jobdir / "cases" / f"scene_00000_random_r{repeat}.json"
                case = {"beams": beams, "metrics": {"si": metrics["final_si_mean"],
                        "sgm": metrics["final_sgm_mean"], "power_w": metrics["power_w"],
                        "total_seconds": 0.7, "initialization_seconds": 0.4,
                        "candidate_generation_seconds": 0.1, "probe_seconds": 0.2,
                        "decisions": 1, "probes": 1, "invalid": 0},
                        "trace": [{"step": 0}], "scene_sha256": scene_sha,
                        "seed": 53001 + repeat, "action_seed": 53001 + repeat,
                        "method": "random", "repeat_index": repeat}
                case_path.write_text(json.dumps(case), encoding="utf-8")
                rows.append({"method": "random", "repeat_index": repeat,
                             "action_seed": 53001 + repeat,
                             "scene_sha256": scene_sha, "case_path": str(case_path),
                             "case_sha256": digest(case_path), "case_id": case_path.stem})
            (jobdir / "job.json").write_text(json.dumps({
                "run_id": "i0_baselines", "kind": "evaluate", "name": "I0 final baselines",
                "experiment_id": "S2-P19", "research_stage": "sac_improvement",
                "mode": "baselines", "partition": "test", "action_seed": 53001,
                "methods": ["greedy", "rule", "random"],
                "method_repeats": {"greedy": 1, "rule": 1, "random": 3},
                "scene_sets": {"test": record}}), encoding="utf-8")
            (jobdir / "status.json").write_text('{"stage":"completed"}', encoding="utf-8")
            results = jobdir / "results.json"
            results.write_text(json.dumps({"scenes": rows}), encoding="utf-8")
            class App:
                def resolve_run(self, run_id):
                    if run_id != "i0_baselines":
                        raise FileNotFoundError(run_id)
                    return jobdir, False

                def scene_dates(self, _record):
                    return {"2024-02-01"}

            _, _, _, reference = prepare_source(App(), {
                "source_run_id": "i0_baselines", "source_group": "random", "partition": "test"})
            self.assertEqual(reference["repeats"], 3)
            self.assertEqual(set(source_rows(reference)), {(scene_sha, r) for r in range(3)})
            rows[2]["case_sha256"] = "0" * 64
            results.write_text(json.dumps({"scenes": rows}), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_source(App(), {"source_run_id": "i0_baselines",
                                       "source_group": "random", "partition": "test"})

    def test_gate_development_source_keeps_partition_and_complete_case(self):
        beams, metrics, _ = _greedy_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scenes"
            source.mkdir()
            scene_path = source / "scene.npz"
            np.savez_compressed(scene_path, **_saved_scene())
            scene_sha = digest(scene_path)
            profile = profile_metadata("europe_v1")
            record = {"id": "dev", "name": "dev", "path": str(source), "count": 1,
                      "profile": "europe_v1", "environment_profile": profile,
                      "environment_version": profile["environment_version"],
                      "source": {"partition_role": "development"},
                      "files": [{"path": str(scene_path), "sha256": scene_sha,
                                 "beam_count": 50, "mother_scene_id": "mother-1"}]}
            jobdir = root / "outputs" / "workbench" / "jobs" / "i0_gate"
            (jobdir / "cases").mkdir(parents=True)
            case_path = jobdir / "cases" / "scene_00000_greedy.json"
            case = {"beams": beams, "metrics": {"si": metrics["final_si_mean"],
                    "sgm": metrics["final_sgm_mean"], "power_w": metrics["power_w"],
                    "total_seconds": 0.7, "initialization_seconds": 0.4,
                    "candidate_generation_seconds": 0.1, "probe_seconds": 0.2,
                    "decisions": 1, "probes": 1, "invalid": 0},
                    "trace": [{"step": 0}], "scene_sha256": scene_sha,
                    "seed": 53001, "action_seed": 53001,
                    "method": "greedy", "repeat_index": 0}
            case_path.write_text(json.dumps(case), encoding="utf-8")
            (jobdir / "job.json").write_text(json.dumps({
                "run_id": "i0_gate", "name": "I0 gate", "kind": "evaluate",
                "research_stage": "sac_improvement", "experiment_id": "S2-P19",
                "mode": "gate", "i0_config": {"seed": 51001}, "action_seed": 53001,
                "scene_sets": {"development": record}}), encoding="utf-8")
            (jobdir / "status.json").write_text('{"stage":"completed"}', encoding="utf-8")
            (jobdir / "results.json").write_text(json.dumps({"scenes": [{
                "method": "greedy", "repeat_index": 0, "scene_sha256": scene_sha,
                "action_seed": 53001,
                "case_path": str(case_path), "case_sha256": digest(case_path),
                "case_id": case_path.stem, "beam_count": 50}]}), encoding="utf-8")
            class FakeApp:
                def __init__(self):
                    self.store = SimpleNamespace(create_job=lambda config, **kwargs:
                                                 {"config": config, **kwargs})

                def resolve_run(self, run_id):
                    self_run = run_id == "i0_gate"
                    if not self_run:
                        raise FileNotFoundError(run_id)
                    return jobdir, False

                def scene_dates(self, _record):
                    return {"2024-01-12"}

            app = FakeApp()
            body = {"source_run_id": "i0_gate", "source_group": "greedy",
                    "partition": "development", "request_id": "refine-i0-greedy"}
            job, _, dataset, reference = prepare_source(app, body)
            self.assertEqual(reference["partition"], "development")
            self.assertEqual(set(source_rows(reference)), {(scene_sha, 0)})
            controls = refinement_defaults()
            controls.update(methods=list(VARIABLE_BEAM_METHODS), repeats=1,
                            action_mode="deterministic", action_seed=53001)
            submitted = PowerRefinementApplication(app).submit_saved({**body, "config": controls})
            self.assertEqual(submitted["extra"]["scene_sets"]["development"]["id"], dataset["id"])
            self.assertEqual(submitted["extra"]["model_files"], [])
            self.assertEqual(submitted["extra"]["initial_source"]["format"], "s2p19_i0_case_v1")
            refinement_dir = root / "refinement_short_check"
            refinement_dir.mkdir()
            frozen_job = {**submitted["extra"], "run_id": "refinement_short_check",
                          "git_commit": "f" * 40, "source_root": str(root),
                          "archive_root": str(root)}

            def unchanged(_problem, power, *, method, options, callback, stop_requested):
                return {"method": method, "power_w": np.asarray(power).tolist(),
                        "evaluations": 1, "elapsed_seconds": 0.0,
                        "termination_reason": "short_correctness_check", "history": [],
                        "converged": True, "fallback_used": False}

            def unchanged_priority(_problem, power, *, method, options, callback, stop_requested):
                value = unchanged(_problem, power, method=method, options=options,
                                  callback=callback, stop_requested=stop_requested)
                value.update(phase1_best_si_mean=metrics["final_si_mean"],
                             best_feasible_si_mean=metrics["final_si_mean"],
                             phase2_si_floor=metrics["final_si_mean"] - options["si_tolerance"])
                return value

            with (patch("satellite_rl.power_refinement.optimizer.optimize_power", side_effect=unchanged),
                  patch("satellite_rl.power_refinement.priority.optimize_priority",
                        side_effect=unchanged_priority)):
                summary = evaluate_refinement(refinement_dir, frozen_job, submitted["config"], device="cpu")
            power_cases = sorted((refinement_dir / "power_cases").glob("*.json"))
            self.assertEqual(len(power_cases), 3)
            initial_files = list((refinement_dir / "initial_allocations").glob("*.json"))
            self.assertEqual(len(initial_files), 1)
            self.assertTrue(summary)

    def test_store_freezes_typed_saved_baseline_without_invented_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            np.savez_compressed(source / "scene.npz", **_saved_scene())
            scene_path = source / "scene.npz"
            profile = profile_metadata("europe_v1")
            record = {"id": "dev-test", "name": "dev-test", "path": str(source),
                      "count": 1, "profile": "europe_v1", "environment_version": profile["environment_version"],
                      "environment_profile": profile,
                      "source": {"partition_role": "development"},
                      "files": [{"path": str(scene_path), "sha256": digest(scene_path), "beam_count": 50}]}
            store = Store(root, source_root=root)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            config = {"schema_version": 1, "name": "S2-P19 saved baseline refinement",
                      "algorithm": "sac", "network": {"kind": "transformer"},
                      "environment": {"profile": "europe_v1", "beam_count_max": 200},
                      "training": {"total_steps": 3}, "research_stage": "power_refinement"}
            source_ref = {"experiment_id": "S2-P19", "format": "s2p19_i0_case_v1",
                          "group": "greedy", "partition": "development",
                          "case_files": [{"path": str(root / "case.json"), "sha256": "a" * 64}]}
            extra = {"experiment_id": "S2-P19", "research_stage": "power_refinement",
                     "initial_source": source_ref, "scene_sets": {"development": record},
                     "model_path": None, "model_files": [], "model_config": config}
            with patch.object(store, "freeze_source", return_value=(snapshot, "f" * 40)):
                job = store.create_job(config, kind="refine", request_id="s2p19-test-no-model", extra=extra)
            self.assertEqual(job["model_files"], [])
            self.assertIsNone(job["model_path"])
            self.assertEqual(job["scene_sets"]["development"]["files"][0]["sha256"], digest(scene_path))

    def test_i0_direct_case_and_inline_case_are_distinct_and_sha_bound(self):
        beams, metrics, trace = _greedy_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "cases" / "scene_00000_greedy.json"
            case.parent.mkdir()
            i0 = {"beams": beams, "metrics": {"si": metrics["final_si_mean"],
                   "sgm": metrics["final_sgm_mean"], "power_w": metrics["power_w"],
                   "total_seconds": 0.7, "initialization_seconds": 0.4,
                   "candidate_generation_seconds": 0.1, "probe_seconds": 0.2,
                   "decisions": 1, "probes": 1, "invalid": 0},
                  "trace": [{"step": 0}], "scene_sha256": "a" * 64,
                  "seed": 51001, "action_seed": 51001,
                  "method": "greedy", "repeat_index": 0}
            case.write_text(json.dumps(i0), encoding="utf-8")
            record = {"path": str(case), "sha256": digest(case), "scene_index": 0,
                      "repeat_index": 0, "scene_sha256": "a" * 64,
                      "action_seed": 51001, "beam_count": 50, "method": "greedy",
                      "case_id": "scene_00000_greedy", "scene_metadata": {"beam_count": 50}}
            reference = {"experiment_id": "S2-P19", "format": "s2p19_i0_case_v1",
                         "source_directory": str(root), "group": "greedy",
                         "scene_count": 1, "repeats": 1, "case_files": [record]}
            initial = source_rows(reference)[("a" * 64, 0)]
            self.assertEqual(initial["beam_count"], 50)
            self.assertAlmostEqual(initial["metrics"]["final_si_mean"], metrics["final_si_mean"])
            self.assertEqual(initial["original_case_file_sha256"], record["sha256"])
            self.assertEqual(initial["inference_seconds"], 0.7)
            physical = FixedAllocationPhysics.from_scene(_saved_scene(), constructive_config("C0"),
                                                          initial["beams"])
            checked = physical.evaluate(np.asarray([beam["power_w"] for beam in initial["beams"]]))
            self.assertTrue(np.allclose(checked.capacity_bps,
                                        [beam["rate_bps"] for beam in initial["beams"]],
                                        rtol=1e-8, atol=1e-3))
            with case.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                source_rows(reference)

            inline = root / "validation_cases" / "step_000100000" / "scene_00000_r0.json"
            contract = {"config_hash": "f" * 64, "env_step": 100000,
                        "scene_index": 0, "repeat_index": 0,
                        "source_scene_sha256": "a" * 64, "action_seed": 53001,
                        "policy_sha256": "b" * 64}
            payload = {"scene_id": "validation_53001_r0", "scene": {"source_sha256": "a" * 64},
                       "action_seed": 53001, "beams": beams, "beam_count": 50,
                       "metrics": metrics, "trace": trace,
                       "initial_allocation_seconds": 1.5, "inference_seconds": 0.9}
            _s2p19_validation_case(inline, contract, payload)
            inline_ref = {"experiment_id": "S2-P19", "format": "s2p19_inline_validation_v1",
                          "source_directory": str(root), "group": "model", "scene_count": 1,
                          "repeats": 1, "validation_step": 100000,
                          "source_config_hash": "f" * 64, "policy_sha256": "b" * 64,
                          "case_files": [{"path": str(inline), "sha256": digest(inline),
                                          "scene_index": 0, "repeat_index": 0,
                                          "scene_sha256": "a" * 64, "action_seed": 53001,
                                          "beam_count": 50, "case_id": inline.stem}]}
            second = source_rows(inline_ref)[("a" * 64, 0)]
            self.assertEqual(second["inference_seconds"], 1.5)
            self.assertEqual(second["policy_inference_seconds"], 0.9)
            inline_ref["policy_sha256"] = "c" * 64
            with self.assertRaises(ValueError):
                source_rows(inline_ref)


if __name__ == "__main__":
    unittest.main()
