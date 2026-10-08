"""Small saved-result fixtures for S2-P19 report aggregation; no evaluation."""

import json
import hashlib
import gzip
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np

from scripts.s2p19.report import (N_VALUES, LOADS, AuditError, _beam_audit, _timing,
                                   _physical_contract, _physical_contract_with_evidence,
                                   _JsonStream, build, collapse_repeats, iter_saved_scenes, mother_pair,
                                   learning_diagnostics, normalize_result, optimizer_cap_stats,
                                   verified_beam_rows, scene_content, task_role,
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
    def test_learning_diagnostics_nested_top_level_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c0 = root / "c0_updates.jsonl"
            c0.write_text(json.dumps({"env_step": 2000, "update_step": 1,
                "values": {"actor_loss": 0.2, "critic_1_loss": 1.2,
                           "critic_2_loss": 1.1}}) + "\n", encoding="utf-8")
            i0 = root / "i0_updates.jsonl"
            i0.write_text(json.dumps({"env_step": 2000, "updates": 1,
                "actor_loss": 0.3, "critic1_loss": 0.8,
                "critic2_loss": 0.9, "alpha": 0.01}) + "\n", encoding="utf-8")
            sources = {"training": {arm: {"run_id": "run_" + arm,
                "updates": {"path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
                for arm, path in (("C0", c0), ("I0", i0))}}
            manifest = {"tasks": {arm: {"run_id": "run_" + arm}
                                  for arm in ("C0", "I0")}}
            summary, missing, figures = learning_diagnostics(
                sources, manifest, root, render_plots=True)
            self.assertEqual(summary["raw_row_count"], 7)
            self.assertIn("C0: updates.alpha真实记录缺失", missing)
            self.assertIn("C1: updates学习诊断来源缺失", missing)
            self.assertEqual(figures, [])
            self.assertFalse((root / "figures/i0_learning_raw.png").exists())
            text = (root / "learning_diagnostics.csv").read_text(encoding="utf-8-sig")
            self.assertIn("values.critic_1_loss", text)
            self.assertIn("macro_decision", text)
            self.assertNotIn("values.alpha", text)
            _, complete_missing, complete_figures = learning_diagnostics(
                sources, manifest, root, arms=("I0",), render_plots=True)
            self.assertEqual(complete_missing, [])
            self.assertEqual(len(complete_figures), 1)
            self.assertTrue(Path(complete_figures[0]).is_file())

    def test_reference_a_timing_is_preserved_without_double_charge(self):
        raw = {"timing": {"initialization_seconds": 2.0,
                          "physics_setup_seconds": 3.0,
                          "reference_optimizer_wall_seconds": 4.0,
                          "reference_verification_seconds": 5.0,
                          "reference_total_seconds": 9.0,
                          "optimizer_wall_seconds": 6.0,
                          "verification_seconds": 7.0,
                          "total_pipeline_seconds": 27.0}}
        values = _timing(raw, {})
        self.assertEqual(values["reference_total_seconds"], 9.0)
        self.assertEqual(values["total_seconds"], 27.0)
        self.assertEqual(sum(values[key] for key in (
            "initial_seconds", "physics_setup_seconds",
            "reference_optimizer_seconds", "reference_verification_seconds",
            "optimizer_seconds", "verification_seconds")), values["total_seconds"])
        missing = _timing({"timing": {"total_pipeline_seconds": 2.0}}, {})
        self.assertIsNone(missing["reference_optimizer_seconds"])
        self.assertIsNone(missing["reference_verification_seconds"])
        self.assertIsNone(missing["reference_total_seconds"])
        repeated = []
        for repeat in range(3):
            item = action("C0", "final_B", "a" * 64, "mother_00",
                          50, 0.75, repeat, 0.9)
            item.update(reference_optimizer_seconds=4.0,
                        reference_verification_seconds=5.0,
                        reference_total_seconds=9.0)
            repeated.append(item)
        grouped = collapse_repeats(repeated)[0]
        self.assertEqual(grouped["reference_total_seconds_mean_per_action"], 9.0)
        self.assertEqual(grouped["reference_total_seconds_sum_repeats"], 27.0)

    def test_optimizer_time_hit_uses_explicit_reason_only(self):
        rows = [{"role": "C0", "stage": "final_B", "solver_termination": reason,
                 "solver_elapsed_seconds": elapsed,
                 "optimizer_time_budget_seconds": 30.0}
                for reason, elapsed in (("time_limit", 29.8),
                                        ("phase_reserve", 29.9),
                                        ("iteration_limit", 12.0),
                                        ("failed_attempt_fallback_to_A", None),
                                        (None, None))]
        rows.append({"role": "C0", "stage": "direct", "solver_termination": None})
        stats = optimizer_cap_stats(rows)
        final = stats["C0/final_B"]
        self.assertEqual(final["eligible_optimizer_actions"], 5)
        self.assertEqual(final["explicit_time_limit_count"], 1)
        self.assertEqual(final["known_other_termination_count"], 3)
        self.assertEqual(final["unknown_termination_count"], 1)
        self.assertEqual(final["known_termination_denominator"], 4)
        self.assertEqual(final["explicit_time_limit_fraction"], 0.25)
        self.assertEqual(final["near_95pct_budget_count"], 2)
        self.assertEqual(final["near_budget_known_denominator"], 3)
        self.assertIsNone(stats["C0/direct"]["explicit_time_limit_fraction"])
        unknown = optimizer_cap_stats([{"role": "I0", "stage": "final_B",
                                       "solver_termination": None}])["I0/final_B"]
        self.assertEqual(unknown["unknown_termination_count"], 1)
        self.assertEqual(unknown["known_termination_denominator"], 0)
        self.assertIsNone(unknown["explicit_time_limit_fraction"])

    def test_verified_beam_rows_preserve_identity_and_uncapped_ratio(self):
        beams = [{"beam_id": 0, "original_index": 7, "group": 0,
                  "lat": 45.0, "lon": 10.0, "required_bps": 100.0,
                  "capacity_bps": 120.0, "sgm": 0.7, "power_w": 10.0,
                  "freq": 0, "slots": 2},
                 {"beam_id": 1, "original_index": 9, "group": 1,
                  "lat": 46.0, "lon": 11.0, "required_bps": 100.0,
                  "capacity_bps": 80.0, "sgm": 0.5, "power_w": 15.0,
                  "freq": 0, "slots": 3}]
        audit = _beam_audit(beams, 2, {"final_si_mean": 0.9,
                                       "final_sgm_mean": 0.6}, "fixture")
        self.assertTrue(audit["available"])
        action_row = {"beam_audit_available": True, "task_id": "C0_final_direct_20473",
                      "run_id": "run_c0", "role": "C0", "stage": "direct",
                      "scene_sha256": "a" * 64, "mother_scene_id": "mother_1",
                      "beam_count": 2, "load_multiplier": 0.75,
                      "repeat_index": 1, "case_path": "saved/case.json",
                      "case_sha256": "b" * 64, "result_path_sha256": "c" * 64}
        saved = list(verified_beam_rows(action_row, beams))
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["original_index"], 7)
        self.assertEqual(saved[0]["capacity_demand_ratio"], 1.2)
        self.assertEqual(saved[0]["si"], 1.0)
        self.assertEqual(saved[0]["effective_carried_bps"], 100.0)
        self.assertEqual(saved[1]["capacity_demand_ratio"], 0.8)
        self.assertEqual(saved[1]["effective_carried_bps"], 80.0)
        self.assertEqual(saved[1]["rate_unit"], "bit/s")

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
            self.assertTrue(any("updates原始学习日志缺失" in item for item in summary["missing"]))
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
            updates = root / "updates.jsonl"
            updates.write_text(json.dumps({"env_step": 50, "values": {"alpha": 0.01}}) + "\n",
                               encoding="utf-8")
            import hashlib
            sources = {"training": {"I0": {"run_id": "train_i0",
                "episodes": {"path": str(episodes),
                             "sha256": hashlib.sha256(episodes.read_bytes()).hexdigest()},
                "evaluations": {"path": str(evaluation),
                                "sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()},
                "updates": {"path": str(updates),
                            "sha256": hashlib.sha256(updates.read_bytes()).hexdigest()}}}}
            rows = training_curves(sources, {"tasks": {"I0": {"run_id": "train_i0"}}})
            self.assertEqual(len([r for r in rows if r["metric"] == "episode_return"]), 3)
            smooth = [r for r in rows if r["metric"] == "episode_return_trailing_mean"]
            self.assertEqual([r["smooth_window"] for r in smooth], [1, 2, 3])
            self.assertAlmostEqual(smooth[-1]["value"], 2.0)
            self.assertEqual(len([r for r in rows if r["metric"] == "validation_si"]), 1)
            sources["training"]["I0"]["updates"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                learning_diagnostics(sources, {"tasks": {"I0": {"run_id": "train_i0"}}},
                                     root, arms=("I0",))

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
            self.assertEqual(summary["per_beam_rows"], 50)
            with gzip.open(root / "analysis/per_beam.csv.gz", "rt",
                           encoding="utf-8-sig") as stream:
                self.assertEqual(len(stream.readlines()), 51)
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
        float_budget = {"environment": {**config["environment"],
                                        "power_budget_w": 3000.0}}
        self.assertEqual(accepted, _physical_contract(float_budget, job, source))
        for invalid in (2999.99, 3000.01, 3100):
            changed_budget = {"environment": {**config["environment"],
                                              "power_budget_w": invalid}}
            self.assertIsNone(_physical_contract(changed_budget, job, source))

    def test_physics_contract_units_from_sha_bound_record_or_frozen_source(self):
        physics = {"src/satellite_rl/envs/satellite.py": "1" * 64,
                   "src/satellite_rl/envs/regions.py": "2" * 64,
                   "src/satellite_rl/power_refinement/physics.py": "3" * 64}
        source_text = ('def run_preparation():\n'
                       '    dataset = {"units": {"rate": "bit/s", '
                       '"beam_location": "degrees", '
                       '"covered_demand_bps": "bit/s"}}\n')
        source_bytes = source_text.encode()
        source_name = "src/satellite_rl/workbench/s2p19_data.py"
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        source = {"git_commit": "a" * 40,
                  "files": {**physics, source_name: source_sha}}
        scene = {"id": "s2p19_final_20261007", "count": 880,
                 "profile": "europe_v1", "environment_version": "europe_geo13_v1",
                 "environment_profile": {"id": "europe_v1"},
                 "source": {"type": "aviation_variable_beams_s2p19",
                            "reimport": {"record_id": "s2p19_final_20261007"}}}
        job = {"git_commit": "a" * 40, "scene_sets": {"test": scene}}
        config = {"environment": {"profile": "europe_v1",
                                  "power_budget_w": 3000, "basic_fraction": 0.1}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "part_0001.zip"
            record = dict(scene)
            record_bytes = json.dumps(record).encode()
            members = {"scenes/test/record.json": record_bytes,
                       "source/" + source_name: source_bytes}
            with zipfile.ZipFile(archive, "w") as output:
                for name, payload in members.items():
                    output.writestr(name, payload)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "members": [{"member": name, "bytes": len(payload),
                             "sha256": hashlib.sha256(payload).hexdigest()}
                            for name, payload in members.items()],
                "volumes": [{"name": archive.name,
                             "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
            result, evidence = _physical_contract_with_evidence(
                config, job, source, receipt)
            self.assertEqual(len(result), 64)
            self.assertEqual(evidence[0]["sha256"], source_sha)
            unbound_source = {**source, "files": {**source["files"],
                                                  source_name: "0" * 64}}
            self.assertIsNone(_physical_contract(config, job, unbound_source, receipt))
            # A complete record is independently sufficient when the source
            # record's frozen units are present.
            record["units"] = {"rate": "bit/s", "beam_location": "degrees"}
            record_bytes = json.dumps(record).encode()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("scenes/test/record.json", record_bytes)
                output.writestr("source/" + source_name, source_bytes)
            receipt.write_text(json.dumps({
                "members": [{"member": "scenes/test/record.json",
                             "bytes": len(record_bytes),
                             "sha256": hashlib.sha256(record_bytes).hexdigest()},
                            {"member": "source/" + source_name,
                             "bytes": len(source_bytes), "sha256": source_sha}],
                "volumes": [{"name": archive.name,
                             "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
            from scripts.s2p19.archive_sources import receipt_members, _zip_digest
            receipt_members.cache_clear()
            _zip_digest.cache_clear()
            with_record, evidence = _physical_contract_with_evidence(
                config, job, source, receipt)
            self.assertEqual(with_record, result)
            self.assertEqual({row["kind"] for row in evidence},
                             {"frozen_scene_record", "frozen_preparation_source"})
            record["units"]["rate"] = "Mbps"
            record_bytes = json.dumps(record).encode()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("scenes/test/record.json", record_bytes)
                output.writestr("source/" + source_name, source_bytes)
            receipt.write_text(json.dumps({
                "members": [{"member": "scenes/test/record.json",
                             "bytes": len(record_bytes),
                             "sha256": hashlib.sha256(record_bytes).hexdigest()},
                            {"member": "source/" + source_name,
                             "bytes": len(source_bytes), "sha256": source_sha}],
                "volumes": [{"name": archive.name,
                             "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
            receipt_members.cache_clear()
            _zip_digest.cache_clear()
            self.assertIsNone(_physical_contract(config, job, source, receipt))
        self.assertIsNone(_physical_contract(config, job, source))

    def test_incremental_result_reader_crosses_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.json"
            value = {"summary": {"note": "small"},
                     "scenes": [{"id": 1,
                                 "long_note": "x" * (_JsonStream.READ_CHARS + 17) +
                                              '雪"\\\n',
                                 "nested": {"items": [1.25e-7, True, None,
                                                      {"value": -42}]}},
                                {"id": 2}],
                     "metadata": {"run_id": "saved_run"}}
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            info = {}
            rows = list(iter_saved_scenes(path, info))
            self.assertEqual(rows, value["scenes"])
            self.assertEqual(info["metadata"]["run_id"], "saved_run")

    def test_incremental_reader_number_boundary_and_malformed_tails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.json"
            prefix = '{"padding":"'
            before_number = '","counter":12'
            padding = "x" * (_JsonStream.READ_CHARS - len(prefix) -
                             len(before_number))
            valid = prefix + padding + before_number + 'e3,"scenes":[{"id":1}]}'
            path.write_text(valid, encoding="utf-8")
            self.assertEqual(list(iter_saved_scenes(path, {})), [{"id": 1}])
            for malformed in (valid[:-1], valid + " trailing",
                              valid.replace('e3,"scenes"', 'e,"scenes"')):
                path.write_text(malformed, encoding="utf-8")
                with self.assertRaises(AuditError):
                    list(iter_saved_scenes(path, {}))


if __name__ == "__main__":
    unittest.main()
