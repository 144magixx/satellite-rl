"""First-stage task/source/data contract, without a formal training run."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from satellite_rl.coverage.config import default_config, validate_config
from satellite_rl.workbench.coverage import CoverageApplication, freeze_dataset
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import digest


class CoverageWorkbenchTests(unittest.TestCase):
    def test_config_rejects_unimplemented_or_misleading_options(self):
        for mutate in (
            lambda c: c["coverage"].update(beam_width_deg=0.7),
            lambda c: c["training"].update(gamma=0.99),
            lambda c: c["network"].update(dropout=0.1),
            lambda c: c["training"].update(seed=51002),
            lambda c: c["coverage"].update(pretend_switch=True),
            lambda c: c["training"].update(total_steps=65, num_envs=64),
        ):
            c = default_config()
            mutate(c)
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_dataset_freeze_preserves_and_verifies_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "original"
            source.mkdir()
            data = source / "inputs.npz"
            data.write_bytes(b"known coverage fixture")
            (source / "record.json").write_text('{"source_dates":["2024-01-08"]}')
            reference = {"path": str(data), "sha256": digest(data), "dates": ["2024-01-08"]}
            frozen = freeze_dataset(reference, root / "frozen")
            data.write_bytes(b"changed input")
            self.assertEqual(Path(frozen["path"]).read_bytes(), b"known coverage fixture")
            with self.assertRaises(ValueError):
                freeze_dataset(reference, root / "corrupt")

    @unittest.skipUnless(os.name == "posix", "远端提交门槛使用POSIX绝对路径；在Linux验收")
    def test_baseline_is_evaluation_and_datasets_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Application(temporary)
            coverage = CoverageApplication(app)
            for role, date in (("train", "2024-01-08"), ("validation", "2024-01-12"), ("test", "2024-01-22")):
                app.store.save_record("scenes", {"id": role, "research_stage": "coverage",
                    "partition_role": role, "dates": [date], "initialization": "rule", "path": "fixture", "sha256": "fixture"})
            c = default_config()
            c["coverage"]["method"] = "none"
            patches = [mock.patch("satellite_rl.workbench.coverage.sys.prefix", "/home/u2024110224/first_stage_satellite_rl/envs/DQN"),
                       mock.patch("satellite_rl.workbench.coverage.getpass.getuser", return_value="u2024110224"),
                       mock.patch("torch.cuda.is_available", return_value=True),
                       mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4090 D")]
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(app.store, "create_job", return_value={"kind": "evaluate"}) as create:
                with self.assertRaises(ValueError):
                    coverage.submit({"config": c, "kind": "train", "dataset_ids": {"train":"train", "validation":"validation"}})
                result = coverage.submit({"config": c, "kind": "evaluate", "dataset_ids": {"test":"test"}})
                self.assertEqual(result["kind"], "evaluate")
                self.assertEqual(create.call_args.kwargs["kind"], "evaluate")
                self.assertEqual(create.call_args.args[0]["datasets"]["test"]["dates"], ["2024-01-22"])

    def test_resource_catalog_does_not_offer_coverage_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Application(temporary)
            app.store.save_record("scenes", {"id":"coverage_one", "research_stage":"coverage"})
            app.store.save_record("scenes", {"id":"resource_one", "profile":"europe_v1"})
            self.assertEqual([r["id"] for r in app.api("GET", ["scenes"], {}, {})["scenes"]], ["resource_one"])
            self.assertEqual([r["id"] for r in app.api("GET", ["coverage","datasets"], {}, {})["datasets"]], ["coverage_one"])

    def test_formal_coverage_submission_requires_remote_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            coverage = CoverageApplication(Application(temporary))
            with mock.patch("satellite_rl.workbench.coverage.sys.prefix", "wrong-prefix"):
                with self.assertRaisesRegex(ValueError, "4090D"):
                    coverage.submit({"config": default_config(), "kind":"train"})

    @unittest.skipUnless(Path("/proc/stat").is_file(), "Linux process creation identity")
    def test_linux_process_identity_matches_creation_time(self):
        from satellite_rl.workbench.runtime import pid_alive, process_start_time
        started = process_start_time(os.getpid())
        self.assertIsNotNone(started)
        self.assertTrue(pid_alive(os.getpid(), started))
        self.assertFalse(pid_alive(os.getpid(), started + 20))


if __name__ == "__main__":
    unittest.main()
