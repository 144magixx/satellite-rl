"""Small saved-result fixtures for S2-P19 report aggregation; no evaluation."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

from scripts.s2p19.report import (N_VALUES, LOADS, AuditError, _beam_audit,
                                   _physical_contract,
                                   build, collapse_repeats, iter_saved_scenes, mother_pair,
                                   normalize_result, scene_content, task_role,
                                   training_curves)


def action(role, stage, scene_sha, mother, n, load, repeat, si):
    mother_number = int(mother.rsplit("_", 1)[-1])
    date = "2024-02-01" if mother_number < 20 else "2024-02-02"
    return {"role": role, "stage": stage, "scene_sha256": scene_sha,
            "scene_semantic_sha256": scene_sha,
            "scene_content_sha256": scene_sha,
            "physical_contract_sha256": "p" * 64,
            "mother_source_sha256": hashlib.sha256(mother.encode()).hexdigest(),
            "selected_time_utc": f"{date}T00:{mother_number % 20:02d}:00Z",
            "mother_scene_id": mother, "beam_count": n, "load_multiplier": load,
            "repeat_index": repeat, "run_id": "saved-run", "case_path": None,
            "case_sha256": None, "beam_audit_available": True,
            "si": si, "sgm": si, "basic_met_count": n,
            "effective_carried_bps": 100.0, "power_w": 10.0, "slots_total": n,
            "zero_capacity_count": 0,
            "rho_min": si, "rho_p05": si, "rho_median": si,
            "rho_p95": si, "rho_max": si,
            "rho_below_08_fraction": 0.0, "rho_below_1_fraction": 1.0,
            "initial_seconds": None, "candidate_seconds": None,
            "physical_probe_seconds": None, "physics_setup_seconds": None,
            "optimizer_seconds": None, "verification_seconds": None,
            "total_seconds": None}


class S2P19ReportTests(unittest.TestCase):
    def test_task_role_and_full_beam_audit(self):
        self.assertEqual(task_role("baseline_random_B_20418"), ("random", "final_B"))
        self.assertEqual(task_role("I0_final_direct_20418"), ("I0", "direct"))
        beams = [{"beam_id": 0, "original_index": 0, "group": 0,
                  "lat": 45.0, "lon": 10.0,
                  "required_bps": 100.0, "capacity_bps": 80.0,
                  "sgm": 0.5, "power_w": 10.0, "freq": 0, "slots": 2,
                  "capacity_demand_ratio": 0.8},
                 {"beam_id": 1, "original_index": 1, "group": 1,
                  "lat": 46.0, "lon": 11.0,
                  "required_bps": 100.0, "capacity_bps": 120.0,
                  "sgm": 0.7, "power_w": 15.0, "freq": 0, "slots": 3,
                  "capacity_demand_ratio": 1.2}]
        result = _beam_audit(beams, 2, {"final_si_mean": 0.9,
                                        "final_sgm_mean": 0.6,
                                        "power_w": 25.0,
                                        "allocated_slots": 5}, "fixture")
        self.assertAlmostEqual(result["si"], 0.9)
        self.assertEqual(result["effective_carried_bps"], 180.0)
        self.assertEqual(result["rho_below_08_fraction"], 0.0)
        with self.assertRaises(AuditError):
            _beam_audit(beams, 2, {"final_si_mean": 0.95,
                                   "final_sgm_mean": 0.6}, "fixture")

    def test_repeat_first_then_forty_mother_cluster(self):
        rows = []
        for mother in range(40):
            mother_id = f"mother_{mother:02d}"
            for n in N_VALUES:
                for load in LOADS:
                    scene = f"{mother:02d}{n:03d}{int(load * 100):03d}".ljust(64, "a")
                    rows.append(action("greedy", "final_B", scene, mother_id,
                                       n, load, 0, 0.90))
                    for repeat in range(3):
                        # A second instance may package the same arrays with a
                        # different NPZ byte SHA; the beam-static signature is
                        # the paired scientific identity.
                        import hashlib
                        alternate_sha = hashlib.sha256(("alternate:" + scene).encode()).hexdigest()
                        candidate = action("C0", "final_B", alternate_sha,
                                           mother_id, n, load, repeat,
                                           0.91 + (repeat - 1) * 0.001)
                        candidate["scene_semantic_sha256"] = scene
                        candidate["scene_content_sha256"] = scene
                        rows.append(candidate)
        scenes = collapse_repeats(rows)
        self.assertEqual(len(scenes), 1760)
        pair = mother_pair(scenes, "C0", "greedy", "final_B", draws=200)
        self.assertEqual(pair["mother_count"], 40)
        self.assertEqual(pair["derived_scenes"], 880)
        self.assertAlmostEqual(pair["mean_delta"], 0.01, places=10)
        self.assertAlmostEqual(pair["ci95_low"], 0.01, places=10)

    def test_partial_fixture_keeps_missing_and_no_formal_figure(self):
        manifest = {"experiment_id": "S2-P19", "authorization": {"scope": ["C0"]},
                    "tasks": {"C0": {"run_id": "train_one", "stage": "training",
                                     "env_step": 1234, "budget_env_steps": 400000}}}
        plan = {"experiment_id": "S2-P19", "tasks": [{"id": "C0_final_direct_20418",
                "port": 20418}, {"id": "C0_final_B_20418", "port": 20418}]}
        with tempfile.TemporaryDirectory() as temp:
            summary = build(manifest, {"tasks": {}}, plan, {}, Path(temp),
                            render_plots=True)
            self.assertFalse(summary["formal_ready"])
            self.assertEqual(summary["figures"], [])
            self.assertEqual(summary["training_status"]["C0"]["env_step"], 1234)
            self.assertGreaterEqual(len(summary["missing"]), 4)
            self.assertTrue(any("episode_return" in item for item in summary["missing"]))
            self.assertTrue((Path(temp) / "summary.json").is_file())
            self.assertFalse((Path(temp) / "figures").exists())

    def test_i0_saved_case_reference_and_missing_timing_remain_null(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = {"scene_sha256": "a" * 64,
                    "scene_metadata": {"mother_scene_id": "m0", "beam_count": 50,
                                       "load_multiplier": 0.75},
                    "method": "model", "repeat_index": 0,
                    "metrics": {"si": 0.8, "sgm": 0.5},
                    "beams": [{"beam_id": i, "original_index": i,
                               "group": i % 8, "lat": 45.0 + i * 0.001,
                               "lon": 10.0 + i * 0.001,
                               "required_bps": 100.0,
                               "capacity_bps": 80.0, "sgm": 0.5,
                               "power_w": 1.0, "freq": i // 8, "slots": 1} for i in range(50)]}
            path = root / "case.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            import hashlib
            case_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            ref = {"case_path": str(path), "case_sha256": case_sha,
                   "scene_sha256": "a" * 64, "method": "model",
                   "repeat_index": 0, "beam_count": 50}
            row = normalize_result("I0_final_direct_20418", "I0", "direct",
                                   ref, {}, "run", "b" * 64)
            self.assertAlmostEqual(row["si"], 0.8)
            self.assertIsNone(row["total_seconds"])
            self.assertEqual(row["case_sha256"], case_sha)

    def test_training_curve_keeps_raw_and_declares_smoothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episodes = root / "episodes.jsonl"
            episodes.write_text(''.join(json.dumps({"env_step": i * 50,
                                                    "episode": i,
                                                    "reward_total": float(i)}) + "\n"
                                        for i in range(1, 4)), encoding="utf-8")
            evaluation = root / "evaluations.jsonl"
            evaluation.write_text(json.dumps({"env_step": 0, "mean_si": 0.75}) + "\n",
                                  encoding="utf-8")
            import hashlib
            sources = {"training": {"I0": {"run_id": "train_i0",
                "episodes": {"path": str(episodes),
                             "sha256": hashlib.sha256(episodes.read_bytes()).hexdigest()},
                "evaluations": {"path": str(evaluation),
                                "sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()}}}}
            rows = training_curves(sources, {"tasks": {"I0": {"run_id": "train_i0"}}})
            self.assertEqual(len([r for r in rows if r["metric"] == "episode_return"]), 3)
            smooth = [r for r in rows if r["metric"] == "episode_return_trailing_mean"]
            self.assertEqual([r["smooth_window"] for r in smooth], [1, 2, 3])
            self.assertAlmostEqual(smooth[-1]["value"], 2.0)
            self.assertEqual(len([r for r in rows if r["metric"] == "validation_si"]), 1)

    def test_controller_source_index_sha_and_case_mirror(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = root / "cases"
            cases.mkdir()
            case = {"scene_sha256": "c" * 64, "method": "greedy",
                    "repeat_index": 0,
                    "scene_metadata": {"mother_scene_id": "m0", "beam_count": 50,
                                       "load_multiplier": 0.75},
                    "metrics": {"si": 0.8, "sgm": 0.5, "power_w": 50.0,
                                "slots_total": 50, "basic_met": 50},
                    "beams": [{"beam_id": i, "original_index": i,
                               "group": i % 8, "lat": 45.0 + i * 0.001,
                               "lon": 10.0 + i * 0.001,
                               "required_bps": 100.0,
                               "capacity_bps": 80.0, "sgm": 0.5,
                               "power_w": 1.0, "freq": i // 8, "slots": 1,
                               "basic_met": True} for i in range(50)]}
            local = cases / "greedy_case.json"
            local.write_text(json.dumps(case), encoding="utf-8")
            case_sha = hashlib.sha256(local.read_bytes()).hexdigest()
            results = root / "results.json"
            results.write_text(json.dumps({"scenes": [{"case_path": "/remote/greedy_case.json",
                "case_sha256": case_sha, "method": "greedy", "repeat_index": 0,
                "scene_sha256": "c" * 64, "beam_count": 50}]}), encoding="utf-8")
            task_id = "baseline_test_20418"
            manifest = {"experiment_id": "S2-P19", "authorization": {"scope": ["C0"]},
                        "tasks": {task_id: {"run_id": "evaluate_saved"}}}
            controller = {"tasks": {task_id: {"state": "completed",
                                             "run_id": "evaluate_saved"}}}
            plan = {"experiment_id": "S2-P19", "tasks": [{"id": task_id, "port": 20418}]}
            sources = {"jobs": {task_id: {"run_id": "evaluate_saved",
                "results_path": str(results),
                "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
                "case_root": str(cases)}}}
            summary = build(manifest, controller, plan, sources, root / "analysis",
                            render_plots=False)
            self.assertFalse(summary["formal_ready"])
            self.assertEqual(summary["per_action_count"], 1)
            self.assertEqual(summary["per_scene_count"], 1)
            self.assertEqual(summary["jobs"][task_id]["run_id"], "evaluate_saved")
            sources["jobs"][task_id]["results_sha256"] = "0" * 64
            with self.assertRaises(AuditError):
                build(manifest, controller, plan, sources, root / "bad", render_plots=False)

    def test_full_npz_content_ignores_zip_wrapper_but_checks_all_inputs(self):
        arrays = {"beam_location": np.array([[45.0, 10.0], [46.0, 11.0]]),
                  "beam_G_Tx_max": np.full((2, 1), 50.0),
                  "UE_G_Rx_max": np.full((2, 1), 40.0),
                  "distances": np.array([38000.0, 38100.0]),
                  "rate": np.array([[100.0], [200.0]]),
                  "noise_temperature": np.full((2, 1), 290.0),
                  "original_indices": np.array([0, 1])}
        with tempfile.TemporaryDirectory() as temp:
            plain, zipped = Path(temp) / "plain.npz", Path(temp) / "zipped.npz"
            np.savez(plain, **arrays)
            np.savez_compressed(zipped, **arrays)
            self.assertNotEqual(plain.read_bytes(), zipped.read_bytes())
            self.assertEqual(scene_content(str(plain)), scene_content(str(zipped)))
            changed = Path(temp) / "changed.npz"
            altered = dict(arrays, noise_temperature=np.array([[291.0], [290.0]]))
            np.savez_compressed(changed, **altered)
            self.assertNotEqual(scene_content(str(zipped))[0], scene_content(str(changed))[0])

    def test_physics_contract_requires_frozen_source_units_and_budget(self):
        source = {"git_commit": "a" * 40, "files": {
            "src/satellite_rl/envs/satellite.py": "1" * 64,
            "src/satellite_rl/envs/regions.py": "2" * 64,
            "src/satellite_rl/power_refinement/physics.py": "3" * 64}}
        job = {"git_commit": "a" * 40, "scene_sets": {"test": {
            "environment_version": "europe_v1", "environment_profile": {"id": "europe_v1"},
            "units": {"rate": "bit/s", "beam_location": "degrees"}}}}
        config = {"environment": {"profile": "europe_v1",
                                  "power_budget_w": 3000, "basic_fraction": 0.1}}
        accepted = _physical_contract(config, job, source)
        self.assertEqual(len(accepted), 64)
        changed = dict(source, files=dict(source["files"],
                 **{"src/satellite_rl/envs/satellite.py": "4" * 64}))
        self.assertNotEqual(accepted, _physical_contract(config, job, changed))
        self.assertIsNone(_physical_contract(config, job,
                          dict(source, git_commit="b" * 40)))
        wrong_units = dict(job, scene_sets={"test": dict(job["scene_sets"]["test"],
                               units={"rate": "Mbps", "beam_location": "degrees"})})
        self.assertIsNone(_physical_contract(config, wrong_units, source))

    def test_incremental_result_reader_crosses_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.json"
            value = {"summary": {"note": "small"},
                     "scenes": [{"id": 1, "long_note": "x" * 140000}, {"id": 2}],
                     "metadata": {"run_id": "saved_run"}}
            path.write_text(json.dumps(value), encoding="utf-8")
            info = {}
            rows = list(iter_saved_scenes(path, info))
            self.assertEqual([row["id"] for row in rows], [1, 2])
            self.assertEqual(info["metadata"]["run_id"], "saved_run")


if __name__ == "__main__":
    unittest.main()
