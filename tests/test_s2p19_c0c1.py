"""C0/C1 correctness and provenance checks; no formal training is started."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.state_layouts import natural_shapes
from satellite_rl.workbench.config import config_summary, default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv
from satellite_rl.workbench.runner import s2p19_pilot_gate_passed
from satellite_rl.workbench import runner as workbench_runner
from satellite_rl.workbench.scene_order import beam_bucket
from satellite_rl.workbench.s2p19_demonstrations import (
    _bucket, generate_greedy_trajectory, load_demonstrations, pack_demonstrations,
    validate_manifest, write_manifest)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(arm="C0"):
    value = default_config()
    value["experiment"] = {"id": "S2-P19", "arm": arm}
    value["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                            context_beam_count="actual", embedding_dim=16, num_heads=4,
                            num_layers=1, feedforward_dim=32, dropout=0)
    value["state"].update(context_level="full", features=list(natural_shapes("full")))
    value["environment"].update(profile="europe_v1", beam_count_max=200,
                                action_mask="joint_feasible")
    value["reward"]["preset"] = "terminal_si_delta"
    value["training"].update(scene_order="variable_beams", beam_count_mode="mixed_four_buckets",
                             gamma=1, total_steps=400000, checkpoint_interval=10000,
                             alpha_mode="fixed", fixed_alpha=0.01)
    return value


def _scene(count, seed):
    with np.errstate(all="raise"):
        np.random.seed(seed)
        source = BeamInfo(profile="europe_v1", count=count)
    value = {key: np.array(item, copy=True) for key, item in source.beam_info.items()}
    value["original_indices"] = np.array(source.shuffle_indices, copy=True)
    return value


class C0RewardTests(unittest.TestCase):
    def test_all_target_beams_and_legacy_config_isolation(self):
        old = validate_config(default_config())
        self.assertNotIn("demonstrations", old["training"])
        config = validate_config(_config())
        self.assertEqual(config["training"]["pilot_stop_steps"], 100000)
        self.assertEqual(config["training"]["gamma"], 1)
        env = WorkbenchEnv(config)
        env.reset(scene=_scene(50, 51001))
        initial = env.metrics()["final_si_mean"]
        self.assertEqual(initial, 0)
        reward_sum = 0.0
        for _ in range(50):
            mask = env.action_mask()
            freq, slot = np.argwhere(mask["frequency_slots"])[0]
            power = np.flatnonzero(mask["power"])[0]
            _, reward, done, info = env.step((int(freq), int(slot), int(power)))
            reward_sum += reward
            self.assertEqual(info["reward_preset"], "terminal_si_delta")
            if done:
                break
        self.assertTrue(done)
        self.assertEqual(env.metrics()["processed_count"], 50)
        self.assertAlmostEqual(reward_sum, 100 * env.metrics()["final_si_mean"], places=5)

    def test_rejects_drift_or_unbound_teacher(self):
        raw = _config()
        raw["training"]["gamma"] = 0.99
        with self.assertRaises(ValueError):
            validate_config(raw)
        raw = _config("C1")
        with self.assertRaises(ValueError):
            validate_config(raw)


class C1TeacherTests(unittest.TestCase):
    def test_formal_teacher_identity_and_order_are_bound_to_completed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            run_id = "prepare_s2p19_teacher"
            teacher_dir = archive / "outputs" / "workbench" / "jobs" / run_id
            (teacher_dir / "teacher").mkdir(parents=True)
            files = []
            for index in range(576):
                count = (100, 50, 125, 175)[index % 4]
                files.append({"sha256": f"{index + 1:064x}", "mother_scene_id": f"m{index:03d}",
                              "beam_count": count,
                              "load_multiplier": (0.75, 1.25)[(index // 4) % 2]})
            train = {"id": "frozen-train", "files": files}
            selected = [row["sha256"] for row in files]
            job_path = teacher_dir / "job.json"
            job_path.write_text(json.dumps({"run_id": run_id, "kind": "prepare",
                                            "experiment_id": "S2-P19",
                                            "scene_sets": {"train": train}}), encoding="utf-8")
            (teacher_dir / "status.json").write_text('{"stage":"completed"}', encoding="utf-8")
            (teacher_dir / "config.json").write_text(json.dumps({"preparation": {
                "preparation_type": "demonstrations", "selected_sha256": selected}}), encoding="utf-8")
            manifest_path = teacher_dir / "teacher" / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            result_path = teacher_dir / "preparation_result.json"
            source_set_sha = "e" * 64
            result_path.write_text(json.dumps({"manifest_path": str(manifest_path),
                                               "manifest_sha256": _sha(manifest_path),
                                               "source_set_sha256": source_set_sha}), encoding="utf-8")
            source = {"run_id": run_id, "job_path": str(job_path), "job_sha256": _sha(job_path),
                      "result_path": str(result_path), "result_sha256": _sha(result_path),
                      "manifest_path": str(manifest_path), "manifest_sha256": _sha(manifest_path)}
            config = {"training": {"demonstrations": {"manifest_path": str(manifest_path),
                       "manifest_sha256": _sha(manifest_path),
                       "source_set_sha256": source_set_sha}}}
            fake_order = SimpleNamespace(source_set_sha256=source_set_sha)
            fake_agent = SimpleNamespace(demonstrations=SimpleNamespace(
                records=[{"scene_sha256": sha} for sha in selected]))
            job = {"archive_root": str(archive), "demonstration_source": source}
            workbench_runner._verify_c1_demonstration_provenance(
                job, config, train, fake_order, fake_agent)
            fake_agent.demonstrations.records[0], fake_agent.demonstrations.records[1] = (
                fake_agent.demonstrations.records[1], fake_agent.demonstrations.records[0])
            with self.assertRaises(ValueError):
                workbench_runner._verify_c1_demonstration_provenance(
                    job, config, train, fake_order, fake_agent)
            fake_agent.demonstrations.records[0], fake_agent.demonstrations.records[1] = (
                fake_agent.demonstrations.records[1], fake_agent.demonstrations.records[0])
            job_path.write_text('{}', encoding="utf-8")
            with self.assertRaises(ValueError):
                workbench_runner._verify_c1_demonstration_provenance(
                    job, config, train, fake_order, fake_agent)

    def test_teacher_and_online_buckets_agree_at_boundaries(self):
        for count in (50, 59, 75, 99, 100, 101, 125, 149, 150, 151, 175, 200):
            self.assertEqual(_bucket(count), beam_bucket(count))

    def test_complete_trajectories_sha_sampling_and_recovery(self):
        c0 = validate_config(_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for count, seed in ((50, 51001), (100, 51002), (125, 51003), (175, 51004)):
                scene = root / f"scene_{count}.npz"
                np.savez_compressed(scene, **_scene(count, seed))
                records.append(generate_greedy_trajectory(
                    c0, scene, _sha(scene), root / f"teacher_{count}.npz"))
            packed = pack_demonstrations(records, root)
            source_set = "a" * 64
            manifest_path = root / "manifest.json"
            manifest_sha = write_manifest(c0, source_set, records, manifest_path, packed=packed)
            c1_raw = _config("C1")
            c1_raw["training"]["demonstrations"] = dict(
                mode="half", manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
                source_set_sha256=source_set)
            c1 = validate_config(c1_raw)
            self.assertEqual(len(validate_manifest(manifest_path, manifest_sha, c1, source_set)["records"]), 4)
            dataset = load_demonstrations(manifest_path, manifest_sha, c1, source_set, 91)
            first = dataset.sample(8, "cpu", (100, 10, 10))
            self.assertEqual(tuple(first[0].shape), (8, config_summary(c1)["state_dim"]))
            self.assertEqual(dataset.draw_count, 8)
            state = dataset.state_dict()
            expected = dataset.sample(8, "cpu", (100, 10, 10))
            restored = load_demonstrations(manifest_path, manifest_sha, c1, source_set, 91)
            restored.load_state_dict(state)
            actual = restored.sample(8, "cpu", (100, 10, 10))
            self.assertTrue(np.array_equal(expected[1].numpy(), actual[1].numpy()))
            self.assertTrue(np.array_equal(expected[2].numpy(), actual[2].numpy()))
            # At least one actual SAC update uses 4 online and 4 teacher rows.
            tiny = deepcopy(c1)
            tiny["training"].update(batch_size=8, replay_capacity=32, warmup_steps=0)
            agent = build_agent(tiny, config_summary(c1)["state_dim"], (100, 10, 10), "cpu")
            for record in records:
                with np.load(root / record["path"], allow_pickle=False) as shard:
                    for i in range(2):
                        agent.replay.add(shard["state"][i], shard["action"][i],
                                         shard["reward"][i], shard["next_state"][i],
                                         shard["done"][i],
                                         {"frequency_slots": shard["mask_fl"][i],
                                          "power": shard["mask_power"][i]},
                                         {"frequency_slots": shard["next_fl"][i],
                                          "power": shard["next_power"][i]},
                                         bucket=record["bucket"], entity_count=record["beam_count"])
            self.assertTrue(agent.ready(1))
            diagnostics = agent.update()
            self.assertEqual(agent.update_step, 1)
            self.assertEqual(agent.demonstrations.draw_count, 4)
            self.assertTrue(np.isfinite(diagnostics["actor_loss"]))
            agent_state = agent.state_dict()
            resumed = build_agent(tiny, config_summary(c1)["state_dim"], (100, 10, 10), "cpu")
            resumed.load_state_dict(agent_state)
            self.assertEqual(resumed.demonstrations.draw_count, 4)
            self.assertEqual(resumed.update_step, 1)
            dataset.close()
            restored.close()
            agent.demonstrations.close()
            resumed.demonstrations.close()
            changed = root / packed["reward"]["path"]
            with changed.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                validate_manifest(manifest_path, manifest_sha, c1, source_set)

    def test_pilot_gate_binds_run_and_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-one")
            self.assertFalse(s2p19_pilot_gate_passed(root, "run-c0", checkpoint, 100000))


class PilotBoundaryTests(unittest.TestCase):
    def test_inline_case_cannot_be_reused_after_actor_weights_change(self):
        class Agent:
            def __init__(self):
                self.actor = torch.nn.Linear(2, 2)

            def policy_network(self):
                return self.actor

        agent = Agent()
        first_sha = workbench_runner._policy_parameters_sha256(agent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            contract = {"config_hash": "c", "env_step": 100000, "scene_index": 0,
                        "repeat_index": 0, "source_scene_sha256": "a" * 64,
                        "action_seed": 53001, "policy_sha256": first_sha}
            result = {"action_seed": 53001,
                      "scene": {"source_sha256": "a" * 64}, "beam_count": 1,
                      "beams": [{"beam_id": 0}], "trace": [{"step": 0}],
                      "initial_allocation_seconds": 0.4}
            workbench_runner._s2p19_validation_case(path, contract, result)
            self.assertEqual(workbench_runner._s2p19_validation_case(path, contract), result)
            with torch.no_grad():
                agent.actor.weight.add_(1)
            changed = {**contract, "policy_sha256": workbench_runner._policy_parameters_sha256(agent)}
            self.assertNotEqual(changed["policy_sha256"], first_sha)
            with self.assertRaises(ValueError):
                workbench_runner._s2p19_validation_case(path, changed)

    def test_actual_training_saves_before_and_at_pilot_then_resumes_same_config(self):
        class FakeOrder:
            mode, seed, source_set_sha256 = "variable_beams", 51001, "a" * 64

            def __init__(self, *_args):
                pass

            def checkpoint_state(self, episode):
                return {"next_episode": episode}

            def verify_checkpoint(self, saved, episode):
                if saved != self.checkpoint_state(episode):
                    raise ValueError("fake scene order changed")

            def selection(self, episode):
                return {"source_index": 0, "bucket": "anchor_100", "beam_count": 1,
                        "mother_scene_id": "fake", "load_multiplier": 0.75,
                        "episode": episode}

        class FakeEnv:
            def __init__(self, _config):
                self.raw_env = SimpleNamespace(beam_num=1, beam_idx=0)

            def reset(self, *, seed=None, scene=None):
                self.raw_env.beam_idx = 0
                self._scene = scene
                return np.zeros(1, dtype=np.float32)

            def scene(self):
                return self._scene

            def action_mask(self):
                return None

            def step(self, action):
                self.raw_env.beam_idx = 1
                return np.ones(1, dtype=np.float32), 1.0, True, {
                    "reward_terms": {}, "requested_action": {}, "actual_action": {}, "beam_id": 0}

            def beam_results(self):
                return [{"beam_id": 0}]

            def metrics(self, beams=None):
                return {"final_si_mean": 1.0, "final_sgm_mean": 1.0,
                        "complete": True, "processed_count": 1}

        class FakeAgent:
            update_step, variable_width, temporal_enabled = 0, False, False

            def __init__(self):
                self.replay = SimpleNamespace(bucket_indices={"anchor_100": [0]},
                    sampled_by_bucket={"anchor_100": 0}, segment_indices=None,
                    replacement_draws={"anchor_100": 0})
                self.last_action_info = {}

            def act(self, *_args, **_kwargs):
                return (0, 0, 0)

            def observe(self, *_args, **_kwargs):
                return None

            def ready(self, *_args):
                return False

            def gamma_for(self, *_args):
                return 1.0

        class FakeLedger:
            def __init__(self):
                self.status = {"preflight": {"gpu": {}}}
                self.events = []

            def save_status(self, **fields):
                self.status.update(fields)

            def event(self, name, **fields):
                self.events.append((name, fields))

            def record(self, *_args, **_kwargs):
                pass

            def flush(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path = root / "scene.npz"
            np.savez_compressed(scene_path, rate=np.ones((1, 1)))
            source_sha = _sha(scene_path)
            scene_set = {"id": "fake-train", "path": str(root),
                         "files": [{"path": str(scene_path), "sha256": source_sha,
                                    "beam_count": 1, "mother_scene_id": "fake",
                                    "load_multiplier": 0.75}]}
            config = {"experiment": {"id": "S2-P19", "arm": "C0"},
                      "network": {"context_beam_count": "fixed_100"},
                      "environment": {"beam_count_max": 1},
                      "algorithm": "sac",
                      "training": {"seed": 51001, "total_steps": 5,
                                   "pilot_stop_steps": 2, "checkpoint_interval": 1,
                                   "log_interval": 20, "gradient_steps": 1,
                                   "warmup_steps": 0, "beam_count_mode": "mixed_four_buckets",
                                   "scene_order": "variable_beams", "scene_order_seed": 51001,
                                   "gamma": 1},
                      "evaluation": {"validation_count": 0, "test_count": 0,
                                     "validation_seed": 53001, "repeats": 1}}
            job = {"run_id": "p19-short", "model_dir": str(root / "models"),
                   "source_root": str(root), "scene_sets": {"train": scene_set},
                   "git_commit": "f" * 40}
            saved = []

            def save_checkpoint(path, _agent, progress, _config, provenance):
                saved.append((progress["env_step"], provenance["boundary"]))
                Path(path).write_bytes(f"{progress['env_step']}:{provenance['boundary']}".encode())

            def load_checkpoint(path, _agent, _config):
                self.assertTrue(Path(path).is_file())
                return ({"env_step": 2, "episode": 2, "best_validation": None,
                         "last_validation_step": -1, "last_checkpoint_step": 2,
                         "wall_seconds": 0.0},
                        {"scene_order": {"next_episode": 2}, "scene_sets": {"train": scene_set}})

            ledger = FakeLedger()
            with (patch.object(workbench_runner, "build_agent", side_effect=lambda *_: FakeAgent()),
                  patch.object(workbench_runner, "VariableBeamSceneOrder", FakeOrder),
                  patch.object(workbench_runner, "save_checkpoint", side_effect=save_checkpoint),
                  patch.object(workbench_runner, "load_checkpoint", side_effect=load_checkpoint),
                  patch.object(workbench_runner, "save_learning_snapshot", return_value=None),
                  patch.object(workbench_runner, "export_policy", return_value=None),
                  patch.object(workbench_runner, "_artifacts", return_value=None),
                  patch.object(workbench_runner, "_ensure_finite_agent", return_value=None),
                  patch.object(workbench_runner, "_restore_logging", return_value=None),
                  patch.object(workbench_runner, "_job_provenance", return_value={}),
                  patch.object(workbench_runner, "config_hash", return_value="test-config"),
                  patch.object(workbench_runner, "_validation_due", return_value=False),
                  patch("satellite_rl.workbench.config.config_summary",
                        return_value={"state_dim": 1, "action_dims": (100, 10, 10)}),
                  patch("satellite_rl.workbench.datasets.scene_set_files",
                        return_value=[(scene_path, source_sha)]),
                  patch("satellite_rl.workbench.environment.WorkbenchEnv", FakeEnv)):
                workbench_runner._training(root, job, config, ledger, resume=False)
                self.assertIn((1, "next_episode"), saved)
                self.assertIn((2, "next_episode"), saved)
                self.assertEqual(ledger.status["stage"], "stopped")
                self.assertEqual(ledger.status["checkpoint_step"], 2)
                self.assertEqual(ledger.status["stop_reason"], "s2p19_pilot_gate_pending")
                checkpoint = root / "checkpoint.pt"
                pilot_sha = _sha(checkpoint)
                (root / "pilot_gate.json").write_text(json.dumps({
                    "pass": True, "run_id": job["run_id"], "checkpoint_step": 2,
                    "checkpoint_sha256": pilot_sha}), encoding="utf-8")
                workbench_runner._training(root, job, config, ledger, resume=True)
                self.assertIn((3, "next_episode"), saved)
                self.assertIn((5, "next_episode"), saved)
                self.assertEqual(ledger.status["stage"], "completed")
            (root / "pilot_gate.json").write_text(json.dumps({
                "pass": True, "run_id": "run-c0", "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_step": 100000}), encoding="utf-8")
            self.assertTrue(s2p19_pilot_gate_passed(root, "run-c0", checkpoint, 100000))
            self.assertFalse(s2p19_pilot_gate_passed(root, "run-c1", checkpoint, 100000))
            checkpoint.write_bytes(b"checkpoint-two")
            self.assertFalse(s2p19_pilot_gate_passed(root, "run-c0", checkpoint, 100000))


if __name__ == "__main__":
    unittest.main()
