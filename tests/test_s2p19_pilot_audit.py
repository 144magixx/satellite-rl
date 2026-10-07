"""Small, non-experimental checks for the S2-P19 pilot evidence builder."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from scripts.s2p19.pilot_audit import (
    AuditError, _direct, _validation_set, assess, freeze_pilot_inputs, physical_result,
    scene_content, sha,
)


class PilotAuditTests(unittest.TestCase):
    @staticmethod
    def rows(value, mother_gains=None):
        result = {}
        for mother in range(6):
            for n in (50, 80, 100, 120, 160, 200):
                for load in (.75, 1.25):
                    key = (f"2024-01-12T{mother:02d}:00:00Z", n, load,
                           f"mother_sha_{mother}")
                    result[key] = {"si": value +
                                   (mother_gains[mother] if mother_gains else 0)}
        return result

    def test_c0_c1_joint_gate_does_not_select_reference_alone(self):
        greedy = self.rows(.75)
        c0 = self.rows(.76)
        c1 = self.rows(.7621)
        c0_b = self.rows(.81)
        c1_b = self.rows(.81001)
        criteria, effect = assess("C0", c0, c0_b, greedy, greedy,
                                  c0, c0_b, c1, c1_b)
        self.assertTrue(all(criteria.values()))
        self.assertAlmostEqual(effect["c1_minus_c0_direct"], .0021)
        failed, _ = assess("C0", c0, c0_b, greedy, greedy,
                           c0, c0_b, self.rows(.7619), c1_b)
        self.assertFalse(failed["c1_direct_gain_at_least_0_2pp"])

    def test_i0_requires_two_mother_final_b_gain(self):
        greedy = self.rows(.75)
        direct = self.rows(.751)
        final_b = self.rows(.82, [.0011, .0012, 0, 0, 0, 0])
        baseline_b = self.rows(.82)
        criteria, effect = assess("I0", direct, final_b, greedy, baseline_b)
        self.assertTrue(all(criteria.values()))
        self.assertEqual(effect["mothers_above_0_1pp"], 2)
        failed, _ = assess("I0", direct, self.rows(.82, [.0011, 0, 0, 0, 0, 0]),
                           greedy, baseline_b)
        self.assertFalse(failed["at_least_two_of_six_mothers_above_0_1pp"])

    def test_full_npz_content_hash_detects_physics_change(self):
        with tempfile.TemporaryDirectory() as temp:
            files = []
            for noise in (1.0, 2.0):
                path = Path(temp) / f"{noise}.npz"
                np.savez_compressed(path, beam_location=np.zeros((2, 2)),
                    beam_G_Tx_max=np.ones(2), UE_G_Rx_max=np.ones(2),
                    distances=np.ones((2, 2)), rate=np.ones(2),
                    noise_temperature=np.array([noise, noise]),
                    original_indices=np.arange(2))
                files.append(path)
            self.assertNotEqual(scene_content(files[0]), scene_content(files[1]))

    def test_snapshot_survives_live_status_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobdir = root / "I0"
            jobdir.mkdir()
            (jobdir / "checkpoint.pt").write_bytes(b"checkpoint")
            for filename, body in {
                "job.json": {"run_id": "real_pilot", "archive_root": str(root)},
                "config.json": {"experiment": {"id": "S2-P19", "arm": "I0"}},
                "status.json": {"run_id": "real_pilot", "env_step": 99968,
                    "checkpoint_step": 99968, "stage": "stopped",
                    "stop_reason": "s2p19_pilot_gate_pending",
                    "checkpoint_sha256": sha(jobdir / "checkpoint.pt")},
            }.items():
                (jobdir / filename).write_text(json.dumps(body), encoding="utf-8")
            sources = {"arm": "I0", "I0": {"run_id": "real_pilot",
                                                "job_dir": str(jobdir)}}
            for name in ("greedy", "greedy_B", "I0_B"):
                other = root / name
                other.mkdir()
                (other / "job.json").write_text(json.dumps({"run_id": name}), encoding="utf-8")
                (other / "config.json").write_text("{}", encoding="utf-8")
                (other / "status.json").write_text(json.dumps({"run_id": name,
                    "stage": "completed", "env_step": 72}), encoding="utf-8")
                (other / "results.json").write_text("{}", encoding="utf-8")
                sources[name] = {"run_id": name, "job_dir": str(other)}
            snapshot = freeze_pilot_inputs(sources, root)["I0"]
            manifest = json.loads(Path(snapshot["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["bindings"]), 4)
            (jobdir / "status.json").write_text("{}", encoding="utf-8")
            self.assertEqual(sha(snapshot["path"] + "/status.json"),
                             manifest["bindings"][2]["snapshot_sha256"])
            with self.assertRaises(AuditError):
                freeze_pilot_inputs(sources, root)

    def test_unserved_beam_and_summary_recalculation(self):
        beams = [{"beam_id": 0, "group": 0, "freq": None, "slots": 0,
                  "required_bps": 10.0, "rate_bps": 0.0, "power_w": 0.0,
                  "sgm": 0.0, "capacity_demand_ratio": 0.0}]
        self.assertEqual(physical_result(beams, {"si": 0, "sgm": 0}, 1)["basic_met"], 0)

    def test_greedy_saved_72_case_branch_without_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows, validation = [], {}
            for index in range(72):
                n = (50, 80, 100, 120, 160, 200)[(index // 2) % 6]
                mother = index // 12
                key = (f"2024-01-12T{mother:02d}:00:00Z", n,
                       (.75, 1.25)[index % 2], f"mother_{mother}")
                scene_sha = f"{index:064x}"
                metadata = {"beam_count": n, "source_sha256": key[3]}
                validation[key] = {"scene_index": index, "scene_sha256": scene_sha,
                                   "content_sha256": f"content_{index}",
                                   "metadata": metadata}
                beams = [{"beam_id": beam, "group": beam % 8, "freq": 0,
                          "slots": 0, "required_bps": 1.0, "rate_bps": 0.0,
                          "power_w": 0.0, "sgm": 0.0} for beam in range(n)]
                case = {"action_seed": 53001, "seed": 53001,
                        "repeat_index": 0, "method": "greedy",
                        "scene_sha256": scene_sha, "scene_metadata": metadata,
                        "beams": beams, "metrics": {"si": 0.0, "sgm": 0.0}}
                path = directory / f"scene_{index:05d}.json"
                path.write_text(json.dumps(case), encoding="utf-8")
                rows.append({"case_path": str(path), "case_sha256": sha(path),
                             "scene_sha256": scene_sha, "action_seed": 53001,
                             "repeat_index": 0, "method": "greedy"})
            summary = {"mode": "baselines", "partition": "validation",
                       "methods": ["greedy"], "method_repeats": {"greedy": 1},
                       "source_set_id": "validation_72", "scenes": rows}
            (directory / "results.json").write_text(json.dumps(summary), encoding="utf-8")
            bundle = {"name": "greedy", "directory": directory, "source": directory,
                      "job": {"mode": "baselines", "scene_sets": {"validation":
                          {"id": "validation_72"}}}, "status": {"stage": "completed"}}
            ledger = {}
            observed, info = _direct(bundle, validation, ledger)
            self.assertEqual(len(observed), 72)
            self.assertIsNone(info["step"])
            self.assertEqual(len(ledger), 73)  # summary plus each saved case
            rows[0]["action_seed"] += 1
            (directory / "results.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(AuditError):
                _direct(bundle, validation, {})

    def test_frozen_workbench_scene_set_without_units_field(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = []
            for mother in range(6):
                for n in (50, 80, 100, 120, 160, 200):
                    for load in (.75, 1.25):
                        path = root / f"scene_{len(files):03d}.npz"
                        np.savez_compressed(path, beam_location=np.zeros((n, 2)),
                            beam_G_Tx_max=np.ones(n), UE_G_Rx_max=np.ones(n),
                            distances=np.ones((n, n)), rate=np.ones(n) * load,
                            noise_temperature=np.ones(n), original_indices=np.arange(n))
                        files.append({"path": str(path), "sha256": sha(path),
                                      "selected_time_utc": f"2024-01-12T{mother:02d}:00:00Z",
                                      "beam_count": n, "load_multiplier": load,
                                      "source_sha256": f"{mother:064x}",
                                      "partition_role": "validation"})
            record = {"count": 72, "profile": "europe_v1",
                      "environment_profile": {"id": "europe_v1"},
                      "source": {"partition_role": "validation",
                                 "source_dates": ["2024-01-12"]}, "files": files}
            bundle = {"job": {"scene_sets": {"validation": record}},
                      "config": {"environment": {"profile": "europe_v1",
                          "power_budget_w": 3000, "basic_fraction": .1}}}
            self.assertEqual(len(_validation_set(bundle, {})), 72)


if __name__ == "__main__":
    unittest.main()
