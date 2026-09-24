import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch

from satellite_rl.workbench.config import default_config, validate_config
from satellite_rl.workbench.datasets import csv_scene, import_scenes
from satellite_rl.workbench.environment import validate_scene
from satellite_rl.workbench.legacy import (
    legacy_architecture, legacy_config, load_legacy_policy, validate_legacy_config,
)
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.server import WorkbenchServer
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import Store, inside, read_json, validate_id


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "frozen_source"
        self.snapshot.mkdir()
        self.config = default_config()
        self.application = Application(self.root, source_root=self.root)
        self.freeze = patch.object(Store, "freeze_source", return_value=(self.snapshot, "f" * 40))
        self.freeze.start()
        self.addCleanup(self.freeze.stop)

    def test_path_guards_and_download_allowlist(self):
        for identifier in ("../escape", "a/b", "", "..\\escape"):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                validate_id(identifier)
        with self.assertRaises(ValueError):
            inside(self.root / "outputs" / ".." / "elsewhere", self.root / "outputs")
        job = self.application.store.create_job(self.config)
        with self.assertRaises(ValueError):
            self.application.download(job["run_id"], "../config.json")
        with self.assertRaises(ValueError):
            self.application.download(job["run_id"], "worker.log/../job.json")

    def test_request_id_includes_training_scene_identity(self):
        first = self.application.api("POST", ["runs"], {},
                                     {"config": self.config, "request_id": "same_request"})
        repeated = self.application.api("POST", ["runs"], {},
                                        {"config": self.config, "request_id": "same_request"})
        self.assertEqual(first["run_id"], repeated["run_id"])
        fake_scene = self.application.store.save_record("scenes", {
            "id": "scene_fixture", "name": "fixture", "path": str(self.root / "scenes"), "count": 1})
        with self.assertRaises(ValueError):
            self.application.api("POST", ["runs"], {}, {"config": self.config,
                                 "request_id": "same_request", "scene_id": fake_scene["id"]})

    def test_request_id_includes_evaluation_inputs(self):
        scene = self.application.store.save_record("scenes", {
            "id": "scene_fixture", "name": "fixture", "path": str(self.root / "scenes"), "count": 1})
        second_scene = self.application.store.save_record("scenes", {
            "id": "scene_other", "name": "other", "path": str(self.root / "other_scenes"), "count": 1})
        model = {"id": "model_fixture", "name": "fixture", "path": str(self.root / "model"),
                 "config": self.config, "algorithm": "sac"}
        with patch.object(self.application, "get_model", return_value=model), \
             patch("satellite_rl.workbench.checkpoints.load_policy"):
            request = {"model_id": model["id"], "scene_id": scene["id"],
                       "request_id": "eval_request", "action_mode": "deterministic", "repeats": 1}
            first = self.application.start_evaluation(request)
            self.assertEqual(first["run_id"], self.application.start_evaluation(request)["run_id"])
            with self.assertRaises(ValueError):
                self.application.start_evaluation({**request, "action_mode": "stochastic"})
            with self.assertRaises(ValueError):
                self.application.start_evaluation({**request, "repeats": 2})
            with self.assertRaises(ValueError):
                self.application.start_evaluation({**request, "scene_id": second_scene["id"]})
            with self.assertRaises(ValueError):
                self.application.start_evaluation({**request, "baselines": ["fixed"]})

    def test_uncommitted_source_blocks_freeze_without_archiving(self):
        self.freeze.stop()
        store = Store(self.root, source_root=self.root)
        class Completed:
            stdout = b" M src/satellite_rl/workbench/config.py\n"
        with patch("satellite_rl.workbench.storage.subprocess.run", return_value=Completed()) as git:
            with self.assertRaisesRegex(ValueError, "未提交"):
                store.freeze_source()
        self.assertEqual(git.call_count, 1)

    def test_corrupt_frozen_snapshot_is_rejected(self):
        self.freeze.stop()
        store = Store(self.root, source_root=self.root)
        commit = "f" * 40
        destination = store.root / "source_snapshots" / commit
        (destination / "src").mkdir(parents=True)
        (destination / "src" / "important.py").write_text("tampered", encoding="utf-8")
        (destination / "source_manifest.json").write_text(json.dumps({
            "git_commit": commit, "files": {"src/important.py": "0" * 64}}), encoding="utf-8")
        class Completed:
            def __init__(self, stdout):
                self.stdout = stdout
        with patch("satellite_rl.workbench.storage.subprocess.run",
                   side_effect=[Completed(b""), Completed(commit.encode())]) as git:
            with self.assertRaisesRegex(ValueError, "校验失败"):
                store.freeze_source()
        self.assertEqual(git.call_count, 2)

    def test_csv_preserves_identity_order_group_and_units(self):
        source = self.root / "scene.csv"
        fieldnames = ["beam_id", "lat", "lon", "demand_mbps", "group"]
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(100):
                writer.writerow({"beam_id": (index + 31) % 100, "lat": 10 + index / 100,
                                 "lon": 100 + index / 100, "demand_mbps": 100 + index,
                                 "group": index % 8})
        arrays = validate_scene(csv_scene(source))
        self.assertEqual(arrays["original_indices"].tolist()[:3], [31, 32, 33])
        self.assertEqual(arrays["rate"][1, 0], 101e6)
        record = import_scenes(self.application.store, source, "CSV fixture")
        self.assertEqual(record["count"], 1)
        with np.load(record["files"][0]["path"], allow_pickle=False) as data:
            self.assertEqual(data["original_indices"].tolist()[:2], [31, 32])
        bad = source.read_text(encoding="utf-8").replace(",0\n", ",7\n", 1)
        source.write_text(bad, encoding="utf-8")
        with self.assertRaises(ValueError):
            csv_scene(source)

    def test_scheduler_stop_resume_and_stale_pid(self):
        job = self.application.store.create_job(self.config)
        run_id = job["run_id"]
        scheduler = Scheduler(self.application.store)
        cancelled = scheduler.request_stop(run_id)
        self.assertEqual(cancelled["stage"], "cancelled")
        with self.assertRaises(ValueError):
            scheduler.resume(run_id)
        directory = self.application.store.job_dir(run_id)
        status = read_json(directory / "status.json")
        status.update(stage="running", pid=99999999)
        (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
        with patch("satellite_rl.workbench.runtime.pid_alive", return_value=False):
            scheduler.tick()
        self.assertEqual(read_json(directory / "status.json")["stage"], "interrupted")
        with self.assertRaises(ValueError):
            scheduler.resume(run_id)
        (directory / "checkpoint.pt").write_bytes(b"fixture")
        with patch("satellite_rl.workbench.runtime.pid_alive", return_value=False):
            resumed = scheduler.resume(run_id)
        self.assertEqual(resumed["stage"], "queued")
        self.assertTrue(read_json(directory / "job.json")["resume"])

    def test_scheduler_launch_uses_frozen_source_and_no_real_process(self):
        job = self.application.store.create_job(self.config)
        scheduler = Scheduler(self.application.store, python_executable=Path(__file__).resolve())
        class FakeChild:
            pid = 12345
            def poll(self):
                return None
        gpu = {"name": "fixture", "memory_total_mb": 12000, "memory_used_mb": 1000,
               "utilization_percent": 0, "compute_pids": []}
        with patch("satellite_rl.workbench.runtime.gpu_snapshot", return_value=gpu), \
             patch("satellite_rl.workbench.runtime.subprocess.Popen", return_value=FakeChild()) as popen:
            scheduler.tick()
        self.assertEqual(popen.call_count, 1)
        args, kwargs = popen.call_args
        self.assertEqual(Path(kwargs["cwd"]), self.snapshot)
        self.assertEqual(kwargs["env"]["PYTHONPATH"], str(self.snapshot / "src"))
        self.assertEqual(kwargs["env"]["SATELLITE_RL_OUTPUT_ROOT"], str(self.root))
        self.assertEqual(read_json(self.application.store.job_dir(job["run_id"]) / "status.json")["pid"], 12345)

    def test_compare_native_workbench_metrics_on_same_actual_scene(self):
        first = self.application.store.create_job(self.config)
        second = self.application.store.create_job(self.config)
        for job, value in ((first, 0.4), (second, 0.6)):
            result = {"summary": {"final_si_mean": value, "final_sgm_mean": value / 2},
                      "metadata": {"action_mode": "deterministic"},
                      "scenes": [{"scene_id": "same_scene", "scene": {"sha256": "a" * 64},
                                  "metrics": {"final_si_mean": value, "final_sgm_mean": value / 2}}]}
            (self.application.store.job_dir(job["run_id"]) / "results.json").write_text(
                json.dumps(result), encoding="utf-8")
        comparison = self.application.compare([first["run_id"], second["run_id"]])
        self.assertEqual(comparison["rows"][1]["paired_scenes"], 1)
        self.assertAlmostEqual(comparison["rows"][1]["final_si_mean_paired_delta"], 0.2)

    def test_http_json_and_local_origin(self):
        server = WorkbenchServer(("127.0.0.1", 0), self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/catalog", timeout=5) as stream:
            payload = json.load(stream)
            self.assertEqual(payload["defaults"]["schema_version"], 1)
        valid = Request(base + "/api/validate", method="POST",
                        data=json.dumps(self.config).encode("utf-8"),
                        headers={"Content-Type": "application/json", "Origin": base})
        with urlopen(valid, timeout=5) as stream:
            self.assertEqual(json.load(stream)["summary"]["state_dim"], 204)
        for request, status in ((Request(base + "/api/missing"), 404),
                                (Request(base + "/api/catalog", headers={"Origin": "http://evil.example"}), 403),
                                (Request(base + "/api/validate", method="POST", data=b"{}",
                                         headers={"Content-Type": "text/plain"}), 400)):
            with self.subTest(status=status), self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, status)
            self.assertIn("error", json.load(caught.exception))


class LegacyWeightsTests(unittest.TestCase):
    def test_all_five_pretrained_weights_load_and_act_on_cpu(self):
        root = Path(__file__).resolve().parents[1] / "models" / "pretrained"
        models = {"sac_transformer": "actor.pth", "sac_cnn": "actor.pth",
                  "sac_mlp": "actor.pth", "ppo_direct": "actor.pth",
                  "dqn": "dqn_online.pth"}
        for kind, filename in models.items():
            with self.subTest(kind=kind):
                path = root / kind / filename
                self.assertTrue(path.is_file())
                policy = load_legacy_policy(path, kind, device="cpu")
                self.assertEqual(policy.architecture["variant"], kind)
                self.assertEqual(policy.architecture["parameter_count"],
                                 sum(parameter.numel() for parameter in policy.network.parameters()))
                indices = policy.act(np.zeros(204, dtype=np.float32))
                self.assertEqual(len(indices), 3)
                self.assertTrue(all(0 <= value < upper for value, upper in zip(indices, (100, 10, 10))))

    def test_legacy_network_descriptors_and_strict_network_validation(self):
        expected = {
            "sac_transformer": ("transformer", [256, 256], 128, 8, 2, 512),
            "sac_cnn": ("cnn", [256, 128], 64, 4, 2, 128),
            "sac_mlp": ("mlp", [128], 64, 4, 2, 128),
            "sac_cnn_matched": ("cnn", [128, 128], 64, 4, 2, 128),
            "sac_mlp_matched": ("mlp", [512, 512], 64, 4, 2, 128),
            "ppo_direct": ("mlp", [256, 128], 64, 4, 2, 128),
            "dqn": ("mlp", [512, 256], 64, 4, 2, 128),
        }
        for kind, (network_kind, hidden, embed, heads, layers, ff) in expected.items():
            with self.subTest(kind=kind):
                config = validate_config(legacy_config(kind))
                network = config["network"]
                self.assertEqual(network["kind"], network_kind)
                self.assertEqual(network["hidden_sizes"], hidden)
                self.assertEqual((network["embedding_dim"], network["num_heads"],
                                  network["num_layers"], network["feedforward_dim"]),
                                 (embed, heads, layers, ff))
                self.assertIn("不代表旧权重", legacy_architecture(kind)["training_provenance"])
                validate_legacy_config(config, kind)
                modified = json.loads(json.dumps(config))
                modified["network"]["dropout"] = 0.05 if kind != "sac_transformer" else 0
                with self.assertRaisesRegex(ValueError, "网络结构不兼容"):
                    validate_legacy_config(modified, kind)

    def test_all_pretrained_import_and_evaluation_preflight_without_gpu(self):
        root = Path(__file__).resolve().parents[1] / "models" / "pretrained"
        models = {"sac_transformer": "actor.pth", "sac_cnn": "actor.pth",
                  "sac_mlp": "actor.pth", "ppo_direct": "actor.pth",
                  "dqn": "dqn_online.pth"}
        with tempfile.TemporaryDirectory() as temporary:
            application = Application(temporary, source_root=Path(temporary))
            snapshot = Path(temporary) / "frozen"
            snapshot.mkdir()
            scene = application.store.save_record("scenes", {
                "id": "legacy_scene", "name": "fixture", "path": str(Path(temporary) / "scenes"), "count": 1})
            with patch.object(Store, "freeze_source", return_value=(snapshot, "f" * 40)):
                for kind, filename in models.items():
                    with self.subTest(kind=kind):
                        record = application.import_model({"path": str(root / kind / filename),
                                                           "legacy_type": kind})
                        self.assertTrue(record["compatible"])
                        validate_legacy_config(record["config"], kind)
                        job = application.start_evaluation({"model_id": record["id"],
                                                            "scene_id": scene["id"],
                                                            "action_mode": "deterministic"})
                        self.assertEqual(job["model_id"], record["id"])
                        self.assertEqual(job["scene_id"], scene["id"])

    def test_nonfinite_legacy_weights_are_rejected_before_import(self):
        original = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "sac_mlp" / "actor.pth"
        state = torch.load(original, map_location="cpu", weights_only=True)
        name = next(key for key, value in state.items() if isinstance(value, torch.Tensor) and value.is_floating_point())
        with tempfile.TemporaryDirectory() as temporary:
            application = Application(temporary, source_root=Path(temporary))
            for invalid in (float("nan"), float("inf")):
                with self.subTest(invalid=invalid):
                    corrupted = {key: value.clone() if isinstance(value, torch.Tensor) else value
                                 for key, value in state.items()}
                    corrupted[name].reshape(-1)[0] = invalid
                    path = Path(temporary) / "corrupt.pth"
                    torch.save(corrupted, path)
                    with self.assertRaisesRegex(ValueError, "NaN或Inf"):
                        load_legacy_policy(path, "sac_mlp", device="cpu")
                    with self.assertRaisesRegex(ValueError, "NaN或Inf"):
                        application.import_model({"path": str(path), "legacy_type": "sac_mlp"})


if __name__ == "__main__":
    unittest.main()
