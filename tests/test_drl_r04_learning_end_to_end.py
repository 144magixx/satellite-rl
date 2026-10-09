"""Bounded CUDA teacher → BC → P1 initialization/update correctness check."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.coverage.drl_r04_config import DrlR04Config
from satellite_rl.coverage.drl_r04_policy import DrlR04Policy, Transition, collate, ppo_update
from satellite_rl.coverage.drl_r04_runner import (_bc, _initialize_p1_from_bc,
                                                   _worker_teacher)
from satellite_rl.coverage.runner import _read, _save_torch, _sha256, _write_json
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene
from satellite_rl.coverage.variable_dataset import save_variable_dataset
from satellite_rl.workbench.coverage_drl_r04 import default_drl_r04_config


@unittest.skipUnless(torch.cuda.is_available(), "short D12-A GPU check requires CUDA")
class DrlR04EndToEndCudaTest(unittest.TestCase):
    def test_teacher_bc_p1_real_updates_without_formal_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = CoverageScene(
                "short-gpu-scene",
                np.asarray([[45, 10], [45.01, 10.01], [46, 11],
                            [46.01, 11.01], [47, 12], [47.01, 12.01]], dtype=float),
                np.asarray([450e6, 400e6, 350e6, 500e6, 300e6, 280e6]),
                np.empty((0, 2)), [f"f{i}" for i in range(6)],
                {}, "short-gpu-source")
            data_path = root / "data" / "train.npz"
            save_variable_dataset(CoverageDataset([scene], {"partition_role": "train"}),
                                  data_path)
            config = default_drl_r04_config()
            config["datasets"] = {"train": {"path": str(data_path),
                                               "sha256": _sha256(data_path)}}
            teacher_dir = root / "teacher"
            shard_dir = teacher_dir / "demonstrations"
            shard_dir.mkdir(parents=True)
            labels, audit = _worker_teacher((scene, DrlR04Config(), 4, 256))
            self.assertTrue(labels)
            shard = shard_dir / "scene_0000.pt"
            _save_torch(shard, labels)
            operations = {name: sum(item["selected_action"]["operation"] == name
                                    for item in labels)
                          for name in ("stop", "split", "merge")}
            _write_json(teacher_dir / "teacher_manifest.json", {
                "format": "d12a_teacher_v1", "run_id": "teacher-short",
                "completed": True, "dataset_sha256": _sha256(data_path),
                "labels": len(labels), "label_operations": operations,
                "trials": audit["result"]["trials"], "unverified_trials": 0,
                "scenes": [{"path": str(shard), "sha256": _sha256(shard)}]})
            bc_dir = root / "bc"
            bc_dir.mkdir()
            bc_job = {"kind": "train", "train_mode": "BC", "run_id": "bc-short",
                      "git_commit": "short-check", "model_dir": str(root / "models"),
                      "demonstrations_dir": str(shard_dir),
                      "teacher_source": {"manifest_path": str(
                          teacher_dir / "teacher_manifest.json")}}
            _bc(bc_dir, bc_job, config, False,
                {"gpu": torch.cuda.get_device_name(0)})
            metadata = _read(root / "models" / "bc_endpoint" / "model.json", {})
            self.assertEqual(metadata["rl_env_step"], 0)
            self.assertGreater(metadata["bc_updates"], 0)
            policy = DrlR04Policy().cuda()
            _initialize_p1_from_bc({"bc_model_dir": str(root / "models" /
                                                         "bc_endpoint")}, config, policy)
            before = [parameter.detach().clone() for parameter in policy.parameters()]
            item = labels[0]["observation"]
            transitions = []
            for i in range(4):
                with torch.no_grad():
                    action, logp, value = policy.act(collate([item], "cuda"),
                                                     deterministic=True)
                transitions.append(Transition(item, int(action.item()),
                                              float(logp.item()), float(value.item()),
                                              float(i), i == 3))
            optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
            result = ppo_update(policy, optimizer, transitions, [0., 0.],
                                device="cuda", gamma=1, num_envs=2,
                                epochs=1, minibatch_size=2)
            self.assertEqual(result["minibatch_updates"], 2)
            self.assertTrue(any(not torch.equal(old, new) for old, new in
                                zip(before, policy.parameters())))


if __name__ == "__main__":
    unittest.main()
