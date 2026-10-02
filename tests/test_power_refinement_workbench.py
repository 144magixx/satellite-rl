"""Third-stage API guards and durable source/input freezing without a long run."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.power_refinement.config import default_config, validate_config
from satellite_rl.workbench.config import default_config as model_default_config
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.power_refinement import G11_M1_RUN_ID, G11_SOURCE_COMMIT
from satellite_rl.workbench.storage import Store, digest, read_json


class PowerRefinementWorkbenchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = Application(self.root, source_root=self.root)
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        frozen_source = patch.object(Store, "freeze_source", return_value=(snapshot, "f" * 40))
        frozen_source.start()
        self.addCleanup(frozen_source.stop)
        self.model_config = model_default_config()
        self.model_config["name"] = "G11 M1"
        self.model_config["environment"].update(profile="europe_v1", power_budget_w=3000)
        self.model_config["network"]["temporal_mode"] = "previous_action"
        self.model_config["training"].update(seed=51001, total_steps=400000, device="cuda")
        model_path = self.root / "original_model"
        model_path.mkdir()
        (model_path / "policy.pt").write_bytes(b"fixed M1 fixture weights")
        (model_path / "model.json").write_text("{}", encoding="utf-8")
        self.model = {"id": "M1_fixture", "name": "G11 M1", "path": str(model_path),
                      "config": self.model_config, "sha256": digest(model_path / "policy.pt")}
        listed = patch.object(self.app, "models", return_value={"models": [self.model]})
        listed.start()
        self.addCleanup(listed.stop)
        self.provenance = {"run_id": G11_M1_RUN_ID, "git_commit": G11_SOURCE_COMMIT,
                           "env_step": 400000, "scene_sets": {
            "train": {"source": {"source_dates": ["2024-01-08"]},
                      "files": [{"sha256": "a" * 64}]},
            "validation": {"source": {"source_dates": ["2024-01-15"]},
                           "files": [{"sha256": "b" * 64}]},
        }}
        self.model["provenance"] = self.provenance
        policy = SimpleNamespace(config=deepcopy(self.model_config), metadata={"provenance": self.provenance})
        loader = patch("satellite_rl.workbench.checkpoints.load_policy", return_value=policy)
        loader.start()
        self.addCleanup(loader.stop)
        self.scene = self.app.api("POST", ["scenes"], {},
                                  {"name": "new independent test", "seed": 90021,
                                   "count": 1, "profile": "europe_v1"})
        self.scene["source"].update(partition_role="test", source_dates=["2024-02-03"])
        self.app.store.save_record("scenes", self.scene)

    def submit(self, **change):
        body = {"model_id": self.model["id"], "scene_id": self.scene["id"],
                "config": default_config(), "request_id": "refine_fixture"}
        body.update(change)
        return self.app.api("POST", ["power-refinement", "runs"], {}, body)

    def test_config_rejects_old_bounds_and_invalid_budgets(self):
        for change in ({"minimum_power_w": 5}, {"maximum_power_w": 50},
                       {"total_power_w": 3100}, {"discrete_step_w": 2},
                       {"methods": ["slsqp"]}, {"maxeval": 0},
                       {"protect_basic_service": False},
                       {"si_tolerance": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**default_config(), **change})

    def test_catalog_only_offers_m1_and_dated_europe_test(self):
        catalog = self.app.api("GET", ["power-refinement", "defaults"], {}, {})
        self.assertEqual(catalog["defaults"]["methods"], ["none", "discrete", "slsqp"])
        self.assertEqual(len(self.app.api("GET", ["power-refinement", "models"], {}, {})["models"]), 1)
        self.assertEqual([s["id"] for s in self.app.api("GET", ["power-refinement", "scenes"], {}, {})["scenes"]],
                         [self.scene["id"]])
        self.model_config["name"] = "G11 M0"
        self.assertEqual(self.app.api("GET", ["power-refinement", "models"], {}, {})["models"], [])

    def test_submission_is_idempotent_and_freezes_actual_inputs(self):
        first = self.submit()
        second = self.submit()
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["kind"], "refine")
        self.assertEqual(first["research_stage"], "power_refinement")
        self.assertEqual(first["model_config"], self.model_config)
        self.assertEqual(first["evaluation_units"], 9)
        self.assertEqual(first["model_provenance"], self.provenance)
        job_dir = self.app.store.job_dir(first["run_id"])
        self.assertEqual(read_json(job_dir / "config.json")["power_refinement"], default_config())
        self.assertEqual(digest(Path(first["model_path"]) / "policy.pt"), self.model["sha256"])
        frozen_scene = first["scene_sets"]["test"]["files"][0]
        self.assertEqual(digest(frozen_scene["path"]), self.scene["files"][0]["sha256"])
        self.assertTrue(Path(frozen_scene["path"]).is_relative_to(job_dir))
        self.assertEqual(first["scene_path"], first["scene_sets"]["test"]["path"])
        with self.assertRaisesRegex(ValueError, "同一请求编号"):
            self.submit(config={**default_config(), "maxiter": 101})

    def test_rejects_partition_overlap_and_changed_source_hash(self):
        self.scene["source"]["partition_role"] = "validation"
        self.app.store.save_record("scenes", self.scene)
        with self.assertRaisesRegex(ValueError, "test"):
            self.submit()
        self.scene["source"].update(partition_role="test", source_dates=["2024-01-15"])
        self.app.store.save_record("scenes", self.scene)
        with self.assertRaisesRegex(ValueError, "日期"):
            self.submit()
        self.scene["source"]["source_dates"] = ["2024-02-03"]
        self.app.store.save_record("scenes", self.scene)
        Path(self.scene["files"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "校验"):
            self.submit()
        self.assertEqual(self.app.store.jobs(), [])

    def test_rechecks_loaded_policy_provenance_not_only_catalog(self):
        switched = deepcopy(self.provenance)
        switched["env_step"] = 399999
        policy = SimpleNamespace(config=deepcopy(self.model_config), metadata={"provenance": switched})
        with patch("satellite_rl.workbench.checkpoints.load_policy", return_value=policy):
            with self.assertRaisesRegex(ValueError, "固定预算终点"):
                self.submit()
        self.assertEqual(self.app.store.jobs(), [])

    def test_catalog_rejects_model_before_fixed_400000_step_endpoint(self):
        self.model["provenance"] = {**self.provenance, "env_step": 399999}
        self.assertEqual(self.app.api("GET", ["power-refinement", "models"], {}, {})["models"], [])
        with self.assertRaisesRegex(ValueError, "G11 M1"):
            self.submit()


if __name__ == "__main__":
    unittest.main()
