"""Short correctness checks for the D12-A policy and persistence contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import torch

from satellite_rl.coverage.drl_r04_policy import (
    DrlR04Policy, Transition, collate, ppo_update, returns_and_advantages)
from satellite_rl.coverage.drl_r04_runner import (_load_checkpoint, _save_checkpoint,
                                                   _load_model, _fresh_stream,
                                                   _resident_init, _resident_step,
                                                   _resident_state, _resident_tail,
                                                   _worker_teacher,
                                                   _evaluate_policy)
from satellite_rl.coverage.drl_r04_config import DrlR04Config
from satellite_rl.coverage.variable_dataset import save_variable_dataset, load_variable_dataset
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene
from satellite_rl.coverage.runner import _write_json, _save_torch, _sha256


def observation(n=3, m=4):
    return {"beam": np.ones((n, 12), dtype=np.float32),
            "global": np.arange(14, dtype=np.float32) / 10,
            "candidate": np.arange(m * 16, dtype=np.float32).reshape(m, 16) / 100,
            "candidate_mask": np.asarray([True] * (m - 1) + [False])}


class DrlR04LearningTest(unittest.TestCase):
    def test_resident_worker_keeps_environment_between_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data" / "tiny.npz"
            points = np.asarray([[45, 10], [45.01, 10.01], [46, 11],
                                 [46.01, 11.01], [47, 12], [47.01, 12.01]], dtype=float)
            scene = CoverageScene("resident-scene", points,
                                  np.asarray([450e6, 400e6, 350e6, 500e6, 300e6, 280e6]),
                                  np.empty((0, 2)), [f"f{i}" for i in range(6)],
                                  {}, "resident-source")
            save_variable_dataset(CoverageDataset([scene], {"partition_role": "train"}), path)
            dataset = load_variable_dataset(path)
            core = DrlR04Config(max_commits=3, max_trials=12)
            stream = _fresh_stream(dataset, core, 51002)
            original_serial = stream["episode_serial"]
            record = {"path": str(path), "sha256": _sha256(path)}
            with ProcessPoolExecutor(max_workers=1,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_resident_init,
                                     initargs=([(0, stream)], record, core)) as pool:
                first = pool.submit(_resident_step, (0, [(0, 1)])).result()[0]
                state = pool.submit(_resident_state, None).result()[0][1]
                tail = pool.submit(_resident_tail, 32).result()[0]
                continued = pool.submit(_resident_step, (32, [(0, 0)])).result()[0]
                final_state = pool.submit(_resident_state, None).result()[0][1]
            with ProcessPoolExecutor(max_workers=1,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_resident_init,
                                     initargs=([(0, state)], record, core)) as resumed:
                replay = resumed.submit(_resident_step, (32, [(0, 0)])).result()[0]
                replay_state = resumed.submit(_resident_state, None).result()[0][1]
            self.assertEqual(first[0], 0)
            self.assertFalse(tail["completed_episode"])
            self.assertEqual(tail["stop_reason"], "budget_truncated")
            self.assertFalse(tail["terminated"])
            self.assertFalse(tail["truncated"])
            self.assertTrue(continued[3])
            self.assertGreaterEqual(continued[5]["terminal_physics"]["physical_allocation_calls"], 1)
            physical = continued[5]["terminal_physics"]
            best = physical["layouts_by_sha256"][physical["layout_aliases"]["best"]]
            self.assertTrue(all({"demand_bps", "capacity_bps", "carried_bps",
                                 "unclipped_capacity_demand_ratio", "si", "sgm"}
                                <= set(beam) for beam in best["beams"]))
            self.assertEqual(continued[4]["info"]["best_sha256"],
                             replay[4]["info"]["best_sha256"])
            self.assertEqual(final_state["episode_serial"], original_serial + 1)
            self.assertEqual(replay_state["episode_serial"], final_state["episode_serial"])
            self.assertEqual(final_state["episode_start_step"], 65)

    def test_teacher_labels_charge_preview_and_record_action(self):
        points = np.asarray([[45, 10], [45.01, 10.01], [46, 11],
                             [46.01, 11.01], [47, 12], [47.01, 12.01]], dtype=float)
        scene = CoverageScene("teacher-scene", points,
                              np.asarray([450e6, 400e6, 350e6, 500e6, 300e6, 280e6]),
                              np.empty((0, 2)), [f"f{i}" for i in range(6)],
                              {}, "teacher-source")
        labels, audit = _worker_teacher((scene, DrlR04Config(max_commits=3, max_trials=12),
                                         3, 12))
        self.assertGreater(len(labels), 0)
        self.assertLessEqual(len(labels), 3)
        self.assertLessEqual(audit["result"]["trials"], 12)
        self.assertEqual(sum(row["teacher_trials"] for row in labels),
                         audit["result"]["trials"])
        self.assertTrue(all(row["menu_sha256"] and "selected_action" in row and
                            "step_info" in row for row in labels))

    def test_baseline_evaluation_respects_mother_scene_shard(self):
        points = np.asarray([[45, 10], [45.01, 10.01], [46, 11],
                             [46.01, 11.01], [47, 12], [47.01, 12.01]], dtype=float)
        scenes = [CoverageScene(f"scene-{index}", points,
                                np.asarray([450e6, 400e6, 350e6, 500e6, 300e6, 280e6]),
                                np.empty((0, 2)), [f"f{i}" for i in range(6)],
                                {}, f"source-{index}") for index in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results.jsonl"
            rows = _evaluate_policy(None, CoverageDataset(scenes, {}), DrlR04Config(),
                                    device="cpu", step=0, output=output,
                                    indices=[1], baseline="I0")
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["dataset_index"] for row in rows], [1, 1, 1])
            self.assertEqual([row["demand_multiplier"] for row in rows],
                             [.75, 1.0, 1.25])
            self.assertEqual(output.read_text(encoding="utf-8").count("\n"), 3)

    def test_padding_masks_and_real_ppo_update(self):
        torch.manual_seed(8)
        policy = DrlR04Policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        rows = [observation(2, 3), observation(5, 4)]
        batch = collate(rows, "cpu")
        self.assertEqual(tuple(batch["global"].shape), (2, 14))
        self.assertEqual(tuple(batch["beam"].shape), (2, 5, 12))
        logits, values = policy(batch)
        self.assertTrue(torch.isfinite(logits[:, :2]).all())
        self.assertLess(logits[1, -1].item(), -1e30)
        before = [value.detach().clone() for value in policy.parameters()]
        transitions = []
        for i in range(4):
            item = rows[i % 2]
            with torch.no_grad():
                choice, logp, value = policy.act(collate([item], "cpu"),
                                                  deterministic=True)
            transitions.append(Transition(item, int(choice.item()), float(logp.item()),
                                          float(value.item()), float(i == 3), i == 3))
        result = ppo_update(policy, optimizer, transitions, np.asarray([0.0, 0.0]),
                            device="cpu", gamma=1, num_envs=2, epochs=1,
                            minibatch_size=2)
        self.assertEqual(result["minibatch_updates"], 2)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, policy.parameters())))

    def test_true_terminal_and_rollout_bootstrap(self):
        item = observation()
        transitions = [Transition(item, 0, 0, 1.0, 2.0, True),
                       Transition(item, 0, 0, 1.0, 0.0, False)]
        returns, advantage = returns_and_advantages(
            transitions, [99.0, 4.0], num_envs=2, gamma=1, gae_lambda=.95)
        self.assertAlmostEqual(returns[0], 2.0)
        self.assertAlmostEqual(returns[1], 4.0)
        self.assertAlmostEqual(advantage[0], 1.0)
        self.assertAlmostEqual(advantage[1], 3.0)

    def test_bc_cross_entropy_changes_same_policy_as_ppo(self):
        torch.manual_seed(17)
        policy = DrlR04Policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        rows = [observation(3, 4) for _ in range(8)]
        batch = collate(rows, "cpu")
        before = [value.detach().clone() for value in policy.parameters()]
        logits, _ = policy(batch)
        labels = torch.as_tensor([0, 1, 2, 0, 1, 2, 0, 1])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, policy.parameters())))

    def test_checkpoint_identity_and_legacy_model_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            job = {"run_id": "d12_short", "git_commit": "abc"}
            config = {"research_stage": "coverage", "network":
                      {"kind": "drl_r04_set_ppo", "hidden_size": 128},
                      "training": {"cpu_workers": 16}}
            policy = DrlR04Policy()
            optimizer = torch.optim.Adam(policy.parameters())
            (directory / "actions.jsonl").write_text("{}\n", encoding="utf-8")
            _save_checkpoint(directory, job, config, policy, optimizer, streams=[],
                             env_step=0, update_step=0, episode=0,
                             last_validation=0, ledgers={"actions.jsonl": 1})
            loaded = _load_checkpoint(directory, job, config, DrlR04Policy(),
                                      torch.optim.Adam(DrlR04Policy().parameters()))
            self.assertEqual(loaded["env_step"], 0)
            with self.assertRaises(ValueError):
                _load_checkpoint(directory, {**job, "run_id": "other"}, config,
                                 DrlR04Policy(), torch.optim.Adam(DrlR04Policy().parameters()))
            model_dir = directory / "old"
            model_dir.mkdir()
            _save_torch(model_dir / "policy.pt", policy.state_dict())
            _write_json(model_dir / "model.json", {"policy_format": "variable_set_ppo_v1",
                        "route": "R08", "network": config["network"],
                        "policy_sha256": _sha256(model_dir / "policy.pt")})
            with self.assertRaises(ValueError):
                _load_model({"model_path": str(model_dir), "model_dir": str(model_dir)},
                            config, "cpu")


if __name__ == "__main__":
    unittest.main()
