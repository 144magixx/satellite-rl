"""S2-P19 task identity, dataset roles and scientific-release guards."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.config import validate_config
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.s2p19_service import (constructive_config, select_teacher_scenes,
                                               assert_constructive_contract, S2P19Application)


class S2P19WorkbenchTests(unittest.TestCase):
    def test_formal_contract_rejects_hidden_hyperparameter_or_node_drift(self):
        for section, field, value in (("network", "embedding_dim", 128),
                ("training", "fixed_alpha", 0.02), ("evaluation", "validation_seed", 123),
                ("evaluation", "extra_steps", [40000])):
            config = constructive_config()
            config[section][field] = value
            with self.assertRaises(ValueError, msg=field):
                assert_constructive_contract(config)

    def test_constructive_routes_only_differ_in_declared_demonstrations(self):
        c0 = constructive_config()
        raw = constructive_config("C1")
        raw["training"]["demonstrations"].update(manifest_path=str(Path("teacher.json").resolve()),
            manifest_sha256="1" * 64, source_set_sha256="2" * 64)
        c1 = validate_config(raw)
        for config in (c0, c1):
            config.pop("name")
            config.pop("experiment")
            config["training"].pop("demonstrations")
        self.assertEqual(c0, c1)
        self.assertEqual(c0["training"]["pilot_stop_steps"], 100000)
        self.assertEqual(c0["evaluation"]["validation_count"], 72)
        self.assertEqual(c0["evaluation"]["extra_steps"], [0, 50000])

    def test_teacher_selection_balances_mothers_and_rejects_missing_stratum(self):
        rows = []
        for i in range(576):
            for n in (50, 100, 125, 150):
                for load in (0.75, 1.25):
                    rows.append({"mother_scene_id": f"mother_{i:04d}", "beam_count": n,
                                 "load_multiplier": load, "sha256": f"{i}-{n}-{load}"})
        selected = select_teacher_scenes({"files": rows})
        self.assertEqual(len(set(selected)), 576)
        self.assertEqual(selected, select_teacher_scenes({"files": list(reversed(rows))}))
        bad = [r for r in rows if not (r["mother_scene_id"] == "mother_0003" and r["beam_count"] == 150)]
        with self.assertRaises(ValueError):
            select_teacher_scenes({"files": bad})

    def test_gate_cannot_be_released_by_bare_pass_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Application(temp)
            run = "train_sac_s2p19_test"
            directory = app.store.job_dir(run)
            directory.mkdir()
            (directory / "checkpoint.pt").write_bytes(b"an auditable fixture, not a model")
            atomic_json(directory / "job.json", {"run_id": run, "kind": "train", "experiment_id": "S2-P19"})
            atomic_json(directory / "config.json", {"research_stage": "sac_improvement"})
            atomic_json(directory / "status.json", {"stage": "stopped", "env_step": 100000,
                "stop_reason": "s2p19_pilot_gate_pending"})
            atomic_json(directory / "pilot_gate.json", {"pass": True, "run_id": run})
            with self.assertRaises(ValueError):
                app.scheduler.resume(run)

    def test_non_p19_cannot_use_prepare_kind(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Application(temp)
            with self.assertRaises(ValueError):
                app.store.create_job(constructive_config(), kind="prepare", request_id="rejected_preparation")
            self.assertEqual(app.store.jobs(), [])


if __name__ == "__main__":
    unittest.main()
