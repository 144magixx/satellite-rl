"""True CPU update and package/recovery checks for the new workbench variants."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent, generalized_advantages
from satellite_rl.workbench.checkpoints import (describe_model, export_policy, load_checkpoint,
                                                 load_policy, save_checkpoint)
from satellite_rl.workbench.config import config_summary, default_config, validate_config
from satellite_rl.workbench.runner import (Ledger, _ensure_finite_agent, _evaluate_job, _fixed_evaluate,
                                           _reconcile, _restore_logging, isolated_rng, run_job, seed_training)


class WorkbenchLearningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.config = default_config()
        self.config["network"]["hidden_sizes"] = [8]
        self.config["training"].update(batch_size=2, warmup_steps=0, rollout_steps=3, ppo_epochs=2,
                                       target_update_interval=1, gradient_clip=1)
        self.config = validate_config(self.config)

    def _transition(self, index):
        state = np.full(4, index / 10, dtype=np.float32)
        return state, (index % 3, index % 2, index % 2), float(index), state + 0.01, index == 2

    def test_all_algorithms_change_parameters_on_real_update(self):
        for algorithm in ("sac", "ppo", "dqn"):
            for network in ("mlp", "cnn", "transformer"):
                with self.subTest(algorithm=algorithm, network=network):
                    self._assert_update(algorithm, network)

    def _assert_update(self, algorithm, network):
        config = copy.deepcopy(self.config)
        config["algorithm"] = algorithm
        config["network"].update(kind=network, cnn_channels=[4], embedding_dim=8,
                                 num_heads=2, num_layers=1, feedforward_dim=16)
        agent = build_agent(config, 4, [3, 2, 2], "cpu")
        before = [p.detach().clone() for p in agent.policy_network().parameters()]
        for index in range(3):
            transition = self._transition(index)
            action = agent.act(transition[0], env_step=index)
            agent.observe(transition[0], action, *transition[2:])
        result = agent.update()
        self.assertTrue(result)
        self.assertEqual(agent.update_step, 2 if algorithm == "ppo" else 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(before, agent.policy_network().parameters())))
        self.assertTrue(all(np.isfinite(float(value)) for value in result.values()))
        if algorithm == "sac":
            self.assertIn("alpha", result)
            self.assertIn("td_abs_mean", result)
        if algorithm == "ppo":
            self.assertIn("clip_fraction", result)
        if algorithm == "dqn":
            self.assertEqual(result["target_updated"], 1)

    def test_sac_temperature_moves_toward_entropy_target(self):
        for target, expected in ((0.0, -1), (100.0, 1)):
            config = copy.deepcopy(self.config)
            config["training"]["target_entropy"] = target
            config["training"]["alpha_lr"] = 0.01
            agent = build_agent(config, 4, [3, 2, 2], "cpu")
            for index in range(3):
                agent.observe(*self._transition(index))
            before = float(agent.log_alpha)
            agent.update()
            self.assertEqual(int(np.sign(float(agent.log_alpha) - before)), expected)

    def test_seed_controls_initial_weights(self):
        config = self.config
        def weights(seed):
            seed_training(seed)
            agent = build_agent(config, 4, [3, 2, 2], "cpu")
            return [parameter.detach().clone() for parameter in agent.policy_network().parameters()]
        first, same, different = weights(51001), weights(51001), weights(51002)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, same)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first, different)))

    def test_ppo_log_prob_matches_update_policy_before_learning(self):
        config = copy.deepcopy(self.config)
        config["algorithm"] = "ppo"
        agent = build_agent(config, 4, [3, 2, 2], "cpu")
        for index in range(3):
            transition = self._transition(index)
            action = agent.act(transition[0], env_step=index)
            agent.observe(transition[0], action, *transition[2:])
        states = torch.as_tensor(np.stack([row[0] for row in agent.rollout]))
        actions = torch.as_tensor([row[1] for row in agent.rollout])
        distributions = [torch.distributions.Categorical(logits=logits) for logits in agent.actor(states)]
        current = sum(dist.log_prob(actions[:, i]) for i, dist in enumerate(distributions))
        old = torch.as_tensor([row[5] for row in agent.rollout])
        self.assertTrue(torch.allclose(current, old, atol=1e-6))
        config["network"]["dropout"] = 0.2
        with self.assertRaises(ValueError):
            build_agent(config, 4, [3, 2, 2], "cpu")

    def test_gae_terminal_and_truncation_masks(self):
        terminal = generalized_advantages([1.0], [0.2], [True], 10.0, 0.9, 0.95)
        truncated = generalized_advantages([1.0], [0.2], [False], 10.0, 0.9, 0.95)
        self.assertAlmostEqual(float(terminal[0]), 0.8)
        self.assertAlmostEqual(float(truncated[0]), 9.8, places=6)

    def test_policy_package_strict_config_and_checkpoint_recovery(self):
        config = self.config
        summary = config_summary(config)
        agent = build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")
        initial = np.zeros(summary["state_dim"], dtype=np.float32)
        action = agent.act(initial, deterministic=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_policy(directory / "model", agent, config, {"test": True})
            loaded = load_policy(directory / "model", config)
            self.assertEqual(loaded.act(initial), action)
            changed = copy.deepcopy(config)
            changed["state"]["features"] = list(reversed(changed["state"]["features"]))
            with self.assertRaises(ValueError):
                load_policy(directory / "model", changed)
            checkpoint = directory / "checkpoint.pt"
            agent.observe(initial, (0, 0, 0), 1.0, initial, False)
            save_checkpoint(checkpoint, agent, {"env_step": 1}, config, {"boundary": "next_episode"})
            restored = build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")
            progress, environment = load_checkpoint(checkpoint, restored, config)
            self.assertEqual(progress["env_step"], 1)
            self.assertEqual(environment["boundary"], "next_episode")
            self.assertEqual(len(restored.replay.rows), 1)
            self.assertEqual(restored.act(initial, deterministic=True), action)

    def test_policy_rejects_nonfinite_weights_even_with_matching_checksum(self):
        summary = config_summary(self.config)
        agent = build_agent(self.config, summary["state_dim"], summary["action_dims"], "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_policy(directory, agent, self.config, {"test": True})
            weight_path = directory / "policy.pt"
            weights = torch.load(weight_path, map_location="cpu", weights_only=True)
            next(iter(weights.values())).view(-1)[0] = float("nan")
            torch.save(weights, weight_path)
            metadata_path = directory / "model.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["weights_sha256"] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                load_policy(directory, self.config)

    def test_checkpoint_reproduces_next_stochastic_action_and_update(self):
        for algorithm in ("sac", "ppo", "dqn"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temporary:
                config = copy.deepcopy(self.config)
                config["algorithm"] = algorithm
                agent = build_agent(config, 4, [3, 2, 2], "cpu")
                for index in range(3):
                    transition = self._transition(index)
                    action = agent.act(transition[0], env_step=index)
                    agent.observe(transition[0], action, *transition[2:])
                checkpoint = Path(temporary) / "checkpoint.pt"
                save_checkpoint(checkpoint, agent, {"env_step": 3}, config, {"boundary": "next_episode"})
                action_a = agent.act(np.zeros(4, dtype=np.float32), deterministic=False)
                metrics_a = agent.update()
                params_a = [p.detach().clone() for p in agent.policy_network().parameters()]
                restored = build_agent(config, 4, [3, 2, 2], "cpu")
                load_checkpoint(checkpoint, restored, config)
                action_b = restored.act(np.zeros(4, dtype=np.float32), deterministic=False)
                metrics_b = restored.update()
                params_b = [p.detach().clone() for p in restored.policy_network().parameters()]
                self.assertEqual(action_a, action_b)
                self.assertEqual(metrics_a, metrics_b)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(params_a, params_b)))

    def test_reconciliation_and_finite_guard_leave_checkpoint_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "updates.jsonl"
            path.write_text('{"env_step": 4}\n{"env_step": 5}\n', encoding="utf-8")
            _reconcile(directory, 4)
            self.assertEqual(path.read_text(encoding="utf-8").count("env_step"), 1)
            agent = build_agent(self.config, 4, [3, 2, 2], "cpu")
            checkpoint = directory / "checkpoint.pt"
            save_checkpoint(checkpoint, agent, {"env_step": 4}, self.config, {})
            original = checkpoint.read_bytes()
            next(agent.policy_network().parameters()).data.fill_(float("nan"))
            with self.assertRaises(FloatingPointError):
                _ensure_finite_agent(agent)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_tensorboard_resume_purges_only_after_checkpoint(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("TensorBoard not installed")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            job = {"run_id": "tb_resume"}
            preflight = {"gpu": "test"}
            first = Ledger(directory, job, self.config, preflight)
            for step in (1, 2, 4, 5, 6):
                first.record("learning", step, {"loss": float(step)})
            first.save_status(env_step=6, checkpoint_step=4)
            first.close()
            resumed = Ledger(directory, job, self.config, preflight, resume=True)
            self.assertIsNone(resumed.writer)
            # The checkpoint is at 4 while status incorrectly advanced to 6.
            _restore_logging(resumed, directory, 4)
            resumed.record("learning", 5, {"loss": 50.0})
            resumed.record("learning", 6, {"loss": 60.0})
            resumed.close()
            event_file = directory / "tensorboard"
            values = EventAccumulator(str(event_file)).Reload().Scalars("learning/loss")
            self.assertEqual([(x.step, x.value) for x in values],
                             [(1, 1.0), (2, 2.0), (4, 4.0), (5, 50.0), (6, 60.0)])
            rows = [__import__("json").loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(row["step"], row["values"]["loss"]) for row in rows],
                             [(1, 1.0), (2, 2.0), (4, 4.0), (5, 50.0), (6, 60.0)])

    def test_independent_evaluation_random_stream(self):
        import random
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with isolated_rng(100):
            random.random()
            np.random.random()
            torch.rand(1)
        self.assertEqual(before[0], random.getstate())
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_actual_validation_does_not_change_training_random_streams(self):
        import random
        config = copy.deepcopy(self.config)
        config["algorithm"] = "dqn"
        config["evaluation"]["action_mode"] = "stochastic"
        config["evaluation"]["repeats"] = 2
        dimensions = config_summary(config)
        agent = build_agent(config, dimensions["state_dim"], dimensions["action_dims"], "cpu")
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), agent.rng.getstate())
        class Sink:
            def record(self, *_args):
                pass
        with tempfile.TemporaryDirectory() as directory:
            scenes, metrics = _fixed_evaluate(config, agent, directory, "validation", 1, 94000, 0, Sink())
            self.assertEqual(len(scenes), 2)
            self.assertEqual(scenes[0]["metrics"]["beam_count"], 100)
            self.assertIn("final_si_mean", metrics)
            self.assertTrue(Path(scenes[0]["scene"]["path"]).exists())
            self.assertEqual(scenes[0]["scene"]["sha256"], scenes[1]["scene"]["sha256"])
            self.assertNotEqual(scenes[0]["action_seed"], scenes[1]["action_seed"])
        self.assertEqual(before[0], random.getstate())
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
        self.assertEqual(before[3], agent.rng.getstate())

    def test_real_environment_transitions_drive_update(self):
        from satellite_rl.workbench.environment import WorkbenchEnv
        config = self.config
        summary = config_summary(config)
        env = WorkbenchEnv(config)
        state = env.reset(seed=94001)
        self.assertEqual(state.shape, (summary["state_dim"],))
        agent = build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")
        for step in range(2):
            action = agent.act(state, env_step=step)
            next_state, reward, done, info = env.step(action)
            self.assertTrue(np.isfinite(reward))
            self.assertIn("actual_action", info)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
        self.assertTrue(agent.ready(2))
        self.assertTrue(all(np.isfinite(value) for value in agent.update().values()))

    def test_baselines_use_same_saved_scene_and_report_power_mapping(self):
        from satellite_rl.workbench.environment import WorkbenchEnv
        config = copy.deepcopy(self.config)
        config["environment"]["power_levels_w"] = [3.0, 7.0, 27.0]
        summary = config_summary(config)
        agent = build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")
        env = WorkbenchEnv(config)
        env.reset(seed=94002)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene_path = directory / "scene.npz"
            np.savez_compressed(scene_path, **env.scene())
            export_policy(directory / "model", agent, config, {"test": True})
            job = {"run_id": "baseline_test", "model_path": str(directory / "model"),
                   "scene_path": str(scene_path), "model_dir": str(directory / "model"),
                   "source_root": temporary, "git_commit": "test", "baselines": ["fixed", "greedy"],
                   "repeats": 2, "action_mode": "deterministic"}
            class Sink:
                def save_status(self, **_fields):
                    pass
                def record(self, *_args):
                    pass
            _evaluate_job(directory, job, config, Sink(), device="cpu")
            import json
            results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results["scenes"]), 2)
            for name in ("fixed", "greedy"):
                baseline = results["baselines"][name]
                self.assertEqual(len(baseline["scenes"]), 1)
                self.assertEqual(baseline["scenes"][0]["scene"]["sha256"], results["scenes"][0]["scene"]["sha256"])
                self.assertFalse(baseline["metadata"]["exact_power_match"])
                self.assertTrue(baseline["metadata"]["power_mappings"])
                self.assertEqual(baseline["metadata"]["repeats"], 1)

    def test_preflight_failure_writes_recoverable_failure_record(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = copy.deepcopy(self.config)
            config["training"]["device"] = "cpu"
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (directory / "job.json").write_text(json.dumps({"run_id": "failed_preflight", "kind": "train"}),
                                                 encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cuda"):
                run_job(directory)
            failure = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stage"], "failed")
            self.assertIn("traceback", failure)
            self.assertIn("config_hash", failure)
            self.assertIsNone(failure["last_normal_checkpoint"])

    def test_model_description_is_from_real_network(self):
        model = describe_model(self.config)
        self.assertGreater(model["parameter_count"], 0)
        actor_heads = [node for node in model["branches"]["策略"] if "heads." in node["label"]]
        self.assertEqual([node["shape"][0] for node in actor_heads], [100, 10, 10])
        self.assertTrue(any("encoder.net" in node["label"] for node in model["branches"]["策略"]))
        config = copy.deepcopy(self.config)
        config["network"].update(kind="transformer", activation="tanh", embedding_dim=8,
                                  num_heads=2, num_layers=1, feedforward_dim=16)
        agent = build_agent(config, 4, [3, 2, 2], "cpu")
        self.assertEqual(len(agent.act(np.zeros(4, dtype=np.float32), deterministic=True)), 3)
        torch.manual_seed(107)
        rng_before = torch.get_rng_state()
        graph = describe_model(config)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(any(layer["type"] == "MultiheadAttention" for layer in graph["branches"]["策略"]))


if __name__ == "__main__":
    unittest.main()
