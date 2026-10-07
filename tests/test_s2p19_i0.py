"""Targeted I0 resource, protection, SAC update and recovery checks."""

import unittest
import json
from hashlib import sha256
from pathlib import Path
import tempfile
from unittest.mock import patch
import random

import numpy as np
import torch

from satellite_rl.envs.regions import geo_distances_km, profile_metadata, synthetic_reference_locations
from satellite_rl.sac_improvement import ImprovementEnv, default_config
from satellite_rl.sac_improvement.agent import ImprovementSAC, pack_actions
from satellite_rl.sac_improvement.environment import STATE_WIDTH
from satellite_rl.sac_improvement.runner import (_baseline_job, _gate_on_resume, _load_evaluation_agent,
                                                 _last_jsonl, _train_job, _validate_node,
                                                 run_job)


def scene(count, seed=73):
    prior = np.random.get_state()
    try:
        np.random.seed(seed)
        positions, _ = synthetic_reference_locations(count=count)
    finally:
        np.random.set_state(prior)
    return {"beam_location": positions,
            "distances": geo_distances_km(positions, "europe_v1"),
            "rate": np.linspace(0.7e9, 1.5e9, count).reshape(-1, 1),
            "beam_G_Tx_max": np.full((count, 1), 50.0),
            "UE_G_Rx_max": np.full((count, 1), 40.0),
            "noise_temperature": np.full((count, 1), 290.0),
            "original_indices": np.arange(count)}


class ImprovementChecks(unittest.TestCase):
    @staticmethod
    def config():
        result = default_config()
        result.update(device="cpu", batch_size=4, warmup_steps=0)
        return result

    def test_full_200_slot_ledger_has_atomic_candidate(self):
        env = ImprovementEnv(self.config())
        state = env.reset(scene(200))
        self.assertEqual(state.size, STATE_WIDTH)
        self.assertEqual(int(env.current_slots.sum()), 800)
        choices = env.candidates()
        self.assertTrue(choices)
        self.assertTrue({choice.kind for choice in choices} & {"boundary", "swap", "window"})
        initial = env.base_eval.si
        state, reward, done, info = env.step(choices[0].id)
        self.assertEqual(info["probes"], 1)
        self.assertTrue(env._legal(env.current_starts, env.current_slots))
        self.assertTrue(env._protected(env.best_eval))
        self.assertAlmostEqual(reward, 100 * (env.best_eval.si - initial), places=8)
        self.assertFalse(done)

    def test_stop_and_recovery_keep_separate_current_and_best(self):
        fixed_scene = scene(16)
        env = ImprovementEnv(self.config())
        env.reset(fixed_scene)
        first = env.candidates()[0]
        env.step(first.id)
        saved = env.checkpoint_state()
        restored = ImprovementEnv(self.config())
        restored.reset(fixed_scene)
        restored.load_checkpoint_state(saved)
        self.assertTrue(np.array_equal(restored.state(), env.state()))
        self.assertEqual([c.id for c in restored.candidates()], [c.id for c in env.candidates()])
        result = restored.output()
        self.assertGreaterEqual(result["metrics"]["si"] + 1e-8,
                                result["metrics"]["initial_si"])
        self.assertEqual(len(result["beams"]), 16)
        _, reward, done, info = restored.step("STOP")
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        self.assertEqual(info["probes"], 1)
        self.assertEqual(info["decisions"], 2)

    def test_step_and_terminal_service_logs_bind_physics_and_scope(self):
        env = ImprovementEnv(self.config())
        env.reset(scene(16))
        candidate = env.candidates()[0]
        _, _, _, info = env.step(candidate.id)
        self.assertEqual(info["requested_action"]["kind"], candidate.kind)
        self.assertEqual(info["requested_action"]["candidate_id"], candidate.id)
        self.assertEqual(info["actual_action"]["changes"], info["requested_action"]["changes"])
        self.assertEqual(len(info["changed_beam_service"]), len(candidate.changes))
        self.assertEqual(info["service_scope"],
                         "changed_beams_only_after_full_all_beam_physics_recalculation")
        for beam in info["changed_beam_service"]:
            i = beam["beam_id"]
            self.assertAlmostEqual(beam["capacity_bps"], float(env.current_eval.capacity_bps[i]))
            self.assertAlmostEqual(beam["capacity_demand_ratio"],
                                   beam["capacity_bps"] / beam["required_bps"])
            self.assertEqual(beam["units"]["capacity_bps"], "bit/s")
        snapshot = env.terminal_snapshot()
        self.assertEqual(len(snapshot["last_exploration_state"]["beams"]), 16)
        self.assertEqual(len(snapshot["selected_best_output"]["beams"]), 16)
        self.assertAlmostEqual(snapshot["last_exploration_state"]["metrics"]["si"],
                               env.current_eval.si)
        self.assertAlmostEqual(snapshot["selected_best_output"]["metrics"]["si"],
                               env.output()["metrics"]["si"])
        _, _, _, stop = env.step("STOP")
        self.assertEqual(stop["requested_action"],
                         {"candidate_id": "STOP", "kind": "STOP", "changes": []})
        self.assertEqual(stop["actual_action"]["changes"], [])

    def test_terminal_snapshot_does_not_replace_worse_exploration_with_best(self):
        fixed = scene(16)
        config = self.config()
        probe = ImprovementEnv(config)
        probe.reset(fixed)
        divergent = None
        for candidate in probe.candidates():
            current = ImprovementEnv(config)
            current.reset(fixed)
            current.step(candidate.id)
            if (current.current_eval.si != current.best_eval.si or
                    current.current_eval.sgm != current.best_eval.sgm):
                divergent = current
                break
        self.assertIsNotNone(divergent)
        terminal = divergent.terminal_snapshot()
        self.assertAlmostEqual(terminal["last_exploration_state"]["metrics"]["si"],
                               divergent.current_eval.si)
        self.assertAlmostEqual(terminal["selected_best_output"]["metrics"]["si"],
                               divergent.best_eval.si)
        self.assertNotEqual(terminal["last_exploration_state"]["layout_sha256"],
                            terminal["selected_best_output"]["layout_sha256"])

    def test_real_transition_updates_twin_q_and_actor(self):
        config = self.config()
        env = ImprovementEnv(config)
        env.reset(scene(16))
        before = env.state()
        features, mask, identities = pack_actions(env)
        self.assertTrue(mask[-1])
        self.assertEqual(mask.sum(), len(env.candidates()) + 1)
        action = 0
        after, reward, done, _ = env.step(identities[action])
        next_features, next_mask, _ = pack_actions(env)
        agent = ImprovementSAC(config)
        actor_before = [x.detach().clone() for x in agent.actor.parameters()]
        q_before = [x.detach().clone() for x in agent.critic1.parameters()]
        for _ in range(config["batch_size"]):
            agent.replay.add(before, features, mask, identities, action, reward,
                             after, next_features, next_mask, done)
        metrics = agent.update()
        self.assertEqual(metrics["updates"], 1)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(q_before, agent.critic1.parameters())))
        restored = ImprovementSAC(config)
        restored.load_state_dict(agent.state_dict())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(len(restored.replay.rows), config["batch_size"])

    def test_workbench_dispatch_contract_writes_real_case(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            scenes = directory / "scenes"
            scenes.mkdir()
            scene_path = scenes / "scene_00000.npz"
            np.savez_compressed(scene_path, **scene(16))
            scene_sha = sha256(scene_path.read_bytes()).hexdigest()
            meta = profile_metadata("europe_v1")
            record = {"path": str(scenes), "count": 72, "profile": "europe_v1",
                      "environment_version": meta["environment_version"],
                      "environment_profile": meta,
                      "files": [{"path": str(scene_path), "sha256": scene_sha,
                                 "beam_count": 16, "mother_scene_id": f"smoke_{i:03d}"}
                                for i in range(72)]}
            job = {"run_id": "baselines_smoke", "kind": "evaluate", "mode": "baselines",
                   "partition": "validation", "methods": ["greedy"],
                   "method_repeats": {"greedy": 1},
                   "scene_sets": {"validation": record},
                   "action_seed": 53001, "i0_config": config}
            (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
            (directory / "config.json").write_text(json.dumps({"research_stage": "sac_improvement",
                                                                "i0": config}), encoding="utf-8")
            with patch("satellite_rl.sac_improvement.runner._cuda_preflight",
                       return_value={"verified_smoke_only": True}):
                run_job(directory)
            result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["scenes"]), 72)
            case = json.loads(Path(result["scenes"][0]["case_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(case["beams"]), 16)
            self.assertEqual(result["scenes"][0]["method"], "greedy")
            self.assertEqual(result["scenes"][0]["repeat_index"], 0)
            self.assertEqual(result["scenes"][0]["action_seed"], 53001)
            self.assertTrue(case["metrics"]["si"] + 1e-8 >= case["metrics"]["initial_si"])
            self.assertLessEqual(case["metrics"]["decisions"], 32)
            self.assertLessEqual(case["metrics"]["probes"], 32)
            case_sha = sha256(Path(result["scenes"][0]["case_path"]).read_bytes()).hexdigest()
            with patch("satellite_rl.sac_improvement.runner._cuda_preflight",
                       return_value={"verified_smoke_only": True}):
                run_job(directory, resume=True)
            self.assertEqual(case_sha, sha256(Path(result["scenes"][0]["case_path"]).read_bytes()).hexdigest())

    def test_standard_final_model_pair_loads_with_sha_and_source_run(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as root:
            final = Path(root) / "final"
            final.mkdir()
            policy = final / "policy.pt"
            metadata = final / "model.json"
            agent = ImprovementSAC(config)
            torch.save({"run_id": "source_smoke", "i0_config": config,
                        "actor": agent.actor.state_dict()}, policy)
            metadata.write_text(json.dumps({"run_id": "source_smoke"}), encoding="utf-8")
            job = {"model_path": str(final), "model_run_id": "source_smoke",
                   "model_files": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()}
                                   for p in (policy, metadata)]}
            restored, digest = _load_evaluation_agent(job, config)
            self.assertEqual(digest, sha256(policy.read_bytes()).hexdigest())
            self.assertEqual(restored.actor.state_dict().keys(), agent.actor.state_dict().keys())

    def test_test_baseline_contract_uses_880_times_five_cases_and_repeat_seeds(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            record = {"files": [{"beam_count": 16}] * 880}
            files = [(directory / "unused.npz", "a" * 64)] * 880
            job = {"run_id": "baseline_contract", "partition": "test",
                   "action_seed": 53001,
                   "methods": ["greedy", "rule", "random"],
                   "method_repeats": {"greedy": 1, "rule": 1, "random": 3}}

            def fake_baseline(_scene, method, seed, _config):
                return {"method": method, "action_seed": seed, "metrics": {"si": 0.5},
                        "beams": [{}] * 16}

            def fake_case(_directory, _id, result, repeat_index=0):
                return {"method": result["method"], "repeat_index": repeat_index,
                        "action_seed": result["action_seed"], "scene_sha256": "a" * 64}

            with patch("satellite_rl.sac_improvement.runner._scene_set", return_value=(record, files)), \
                    patch("satellite_rl.sac_improvement.runner._load_scene", return_value={}), \
                    patch("satellite_rl.sac_improvement.runner.run_baseline",
                          side_effect=fake_baseline) as baseline, \
                    patch("satellite_rl.sac_improvement.runner._saved_case", return_value=None), \
                    patch("satellite_rl.sac_improvement.runner._attempt_start", return_value=0), \
                    patch("satellite_rl.sac_improvement.runner._attempt_complete"), \
                    patch("satellite_rl.sac_improvement.runner._case", side_effect=fake_case), \
                    patch("satellite_rl.sac_improvement.runner._status"), \
                    patch("satellite_rl.sac_improvement.runner._atomic_json"), \
                    patch("satellite_rl.sac_improvement.runner._artifacts"), \
                    patch("satellite_rl.sac_improvement.runner._sha", return_value="b" * 64):
                _baseline_job(directory, job, config, "baselines")
            self.assertEqual(baseline.call_count, 4400)
            first = baseline.call_args_list[:5]
            self.assertEqual([(call.args[1], call.args[2]) for call in first],
                             [("greedy", 53001), ("rule", 53001),
                              ("random", 53001), ("random", 53002), ("random", 53003)])

    def test_fixed_validation_preserves_all_training_rngs(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            scenes = directory / "scenes"
            scenes.mkdir()
            path = scenes / "scene_0.npz"
            np.savez_compressed(path, **scene(16))
            digest = sha256(path.read_bytes()).hexdigest()
            record = {"id": "fixed_validation_smoke", "path": str(scenes), "count": 1,
                      "files": [{"path": str(path), "sha256": digest, "beam_count": 16}]}
            torch.manual_seed(123)
            np.random.seed(123)
            random.seed(123)
            agent = ImprovementSAC(config)
            py_before = random.getstate()
            np_before = np.random.get_state()
            torch_before = torch.get_rng_state().clone()
            complete = _validate_node(directory, {"action_seed": 53001}, config, agent,
                                      50000, "checkpoint_smoke_sha", (record, [(path, digest)]))
            self.assertTrue(complete)
            self.assertEqual(py_before, random.getstate())
            np_after = np.random.get_state()
            self.assertEqual(np_before[0], np_after[0])
            self.assertTrue(np.array_equal(np_before[1], np_after[1]))
            self.assertEqual(np_before[2:], np_after[2:])
            self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
            self.assertEqual(json.loads((directory / "validation" / "step_000050000" /
                                         "summary.json").read_text(encoding="utf-8"))["count"], 1)

    def test_pilot_checkpoint_gate_resume_same_run_id(self):
        config = self.config()
        # Reduced boundaries exercise the exact production transition without
        # launching a long formal task or changing validate_config's contract.
        config.update(total_steps=2, pilot_stop_steps=1, warmup_steps=100,
                      checkpoint_interval=1)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "source"
            source.mkdir()
            (source / "source_manifest.json").write_text("{}", encoding="utf-8")
            frozen_path = directory / "config.json"
            frozen_path.write_text(json.dumps({"research_stage": "sac_improvement",
                                               "i0": config}), encoding="utf-8")
            frozen_sha = sha256(frozen_path.read_bytes()).hexdigest()
            path = directory / "scene.npz"
            np.savez_compressed(path, **scene(16))
            digest = sha256(path.read_bytes()).hexdigest()
            train_files = [(path, digest)] * 8064
            validation_files = [(path, digest)] * 72
            train_record = {"files": [{"beam_count": 16, "mother_scene_id": "smoke",
                                       "load_multiplier": 0.75}] * 8064}
            validation_record = {"files": [{}] * 72}
            job = {"run_id": "train_smoke", "source_root": str(source),
                   "git_commit": "smoke", "model_dir": str(directory / "models")}

            class Order:
                def __init__(self, *args):
                    pass

                def selection(self, episode):
                    return {"source_index": 0}

            def selected(_job, role):
                return ((train_record, train_files) if role == "train"
                        else (validation_record, validation_files))

            with patch("satellite_rl.sac_improvement.runner._scene_set", side_effect=selected), \
                    patch("satellite_rl.sac_improvement.runner.VariableBeamSceneOrder", Order), \
                    patch("satellite_rl.sac_improvement.runner._validate_node", return_value=True), \
                    patch("satellite_rl.sac_improvement.environment.validate_config",
                          return_value=config):
                _train_job(directory, job, config, False, frozen_sha)
                status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["stage"], "stopped")
                self.assertEqual(status["stop_reason"], "s2p19_pilot_gate_pending")
                self.assertEqual(status["env_step"], 1)
                trace = json.loads((directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
                self.assertIn("requested_action", trace)
                self.assertIn("actual_action", trace)
                self.assertIn("changed_beam_service", trace)
                episode_file = (directory / "episodes.jsonl") if (directory / "episodes.jsonl").exists() \
                    else (directory / "episode_snapshots.jsonl")
                episode_log = json.loads(episode_file.read_text(encoding="utf-8").splitlines()[0])
                self.assertIn("last_exploration_state", episode_log["terminal_snapshot"])
                self.assertIn("selected_best_output", episode_log["terminal_snapshot"])
                for state_name in ("last_exploration_state", "selected_best_output"):
                    beams = episode_log["terminal_snapshot"][state_name]["beams"]
                    self.assertEqual(len(beams), 16)
                    self.assertEqual(beams[0]["service_units"]["capacity_bps"], "bit/s")
                (directory / "pilot_gate.json").write_text(json.dumps({"run_id": "train_smoke",
                    "checkpoint_sha256": status["checkpoint_sha256"], "pass": True}), encoding="utf-8")
                _train_job(directory, job, config, True, frozen_sha)
            final = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(final["stage"], "completed")
            self.assertEqual(final["env_step"], 2)
            self.assertTrue((directory / "models" / "final" / "policy.pt").is_file())
            model_meta = json.loads((directory / "models" / "final" / "model.json").read_text(
                encoding="utf-8"))
            self.assertEqual(model_meta["format"], "satellite_rl_s2p19_improvement_v1")
            self.assertEqual(model_meta["state_dim"], STATE_WIDTH)
            self.assertEqual(model_meta["config"]["research_stage"], "sac_improvement")

    def test_interrupted_pilot_validation_can_finish_before_gate(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as root:
            allowed = _gate_on_resume(Path(root), {"run_id": "train_smoke"},
                                      {"stop_reason": "user_stop_during_validation"},
                                      "sha", {"env_step": config["pilot_stop_steps"],
                                              "episodes": 1}, config)
            self.assertFalse(allowed)

    def test_damaged_terminal_jsonl_is_not_silently_completed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episodes.jsonl"
            path.write_bytes(b'{"episode":0}\n{"episode":1')
            with self.assertRaisesRegex(ValueError, "incomplete record"):
                _last_jsonl(path)

    def test_training_trace_write_failure_marks_job_failed(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            path = directory / "scene.npz"
            np.savez_compressed(path, **scene(16))
            digest = sha256(path.read_bytes()).hexdigest()
            train_files = [(path, digest)] * 8064
            validation_files = [(path, digest)] * 72
            record = {"files": [{"beam_count": 16, "mother_scene_id": "smoke",
                                 "load_multiplier": 0.75}] * 8064}
            job = {"run_id": "train_log_failure", "kind": "train",
                   "mode": "train", "i0_config": config}
            (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
            (directory / "config.json").write_text(json.dumps({
                "research_stage": "sac_improvement", "i0": config}), encoding="utf-8")

            class Order:
                def __init__(self, *args):
                    pass

                def selection(self, episode):
                    return {"source_index": 0}

            def selected(_job, role):
                return ((record, train_files) if role == "train"
                        else ({"files": [{}] * 72}, validation_files))

            with patch("satellite_rl.sac_improvement.runner._cuda_preflight",
                       return_value={"isolated_smoke": True}), \
                    patch("satellite_rl.sac_improvement.runner._scene_set", side_effect=selected), \
                    patch("satellite_rl.sac_improvement.runner.VariableBeamSceneOrder", Order), \
                    patch("satellite_rl.sac_improvement.runner._validate_node", return_value=True), \
                    patch("satellite_rl.sac_improvement.runner._jsonl", side_effect=OSError(
                          "trace disk failure")):
                with self.assertRaisesRegex(OSError, "trace disk failure"):
                    run_job(directory)
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stage"], "failed")
            self.assertIn("trace disk failure",
                          (directory / "failure.json").read_text(encoding="utf-8"))
            self.assertFalse((directory / "models" / "final" / "policy.pt").exists())


if __name__ == "__main__":
    unittest.main()
