"""S1-03 conditional probability, graph symmetry and charged search checks."""

import unittest
from pathlib import Path
import tempfile
import json
import hashlib
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.coverage.reconstruction import (
    ReconstructionAction, ReconstructionConfig, ReconstructionState,
    action_candidates, state_metrics,
)
from satellite_rl.coverage.reconstruction_ppo import (
    GraphRollout, RelationPolicy, action_identity, graph_observation,
    ppo_update, prefix_mask, stack_observations,
)
from satellite_rl.coverage.reconstruction_runner import (
    ReconstructionVectorEnv, _checkpoint, _choose_rule, _choose_search,
    _evaluate, _restore, _summary,
    _truncate_to_checkpoint, _train,
)
from satellite_rl.coverage.reconstruction_config import reconstruction_default_config
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene


def scene_100():
    lat, lon = np.meshgrid(np.linspace(45, 45.4, 10),
                           np.linspace(0, .4, 10), indexing="ij")
    positions = np.column_stack((lat.ravel(), lon.ravel()))
    return CoverageScene("reconstruction-small", positions,
                         np.arange(1, 101, dtype=np.float64) * 1e6,
                         positions.copy(), [f"ac-{i:03d}" for i in range(100)],
                         {}, "frozen-scene-sha")


class ReconstructionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scene = scene_100()
        cls.config = ReconstructionConfig(max_macro_steps=2,
                                          neighbor_count=1,
                                          split_target_count=4)
        cls.state = ReconstructionState.from_scene(cls.scene, config=cls.config)

    def test_conditional_joint_log_probability_reuses_sampled_masks(self):
        torch.manual_seed(21)
        candidates = action_candidates(self.state, self.config)
        observation = stack_observations([graph_observation(self.state, self.config)], "cpu")
        policy = RelationPolicy()
        action, choices, masks, old_logp, value, digests = policy.act(
            observation, [candidates])
        logp, entropy, reevaluated = policy.evaluate_choices(observation, choices, masks)
        self.assertEqual(len(digests[0]), 64)
        self.assertIn(action[0], candidates)
        self.assertTrue(bool(masks[0, torch.arange(5), choices[0]].all()))
        torch.testing.assert_close(logp, old_logp, rtol=0, atol=1e-6)
        torch.testing.assert_close(reevaluated, value, rtol=0, atol=1e-6)
        self.assertTrue(bool(torch.isfinite(entropy).all()))

    def test_real_gradient_update_and_rollout_boundary_bootstrap(self):
        torch.manual_seed(22)
        observation = stack_observations([graph_observation(self.state, self.config)], "cpu")
        candidates = action_candidates(self.state, self.config)
        policy = RelationPolicy()
        _, choices, masks, logp, value, _ = policy.act(observation, [candidates])
        rollout = GraphRollout(
            {key: tensor.unsqueeze(0).repeat(2, *([1] * tensor.ndim))
             for key, tensor in observation.items()},
            choices.unsqueeze(0).repeat(2, 1, 1),
            masks.unsqueeze(0).repeat(2, 1, 1, 1),
            logp.unsqueeze(0).repeat(2, 1),
            value.unsqueeze(0).repeat(2, 1),
            torch.tensor([[.2], [-.1]]),
            torch.tensor([[False], [False]]),
            torch.tensor([.7]))
        advantage, returns = rollout.advantages(gamma=1, gae_lambda=1)
        torch.testing.assert_close(returns[0], torch.tensor([.8]), atol=1e-5, rtol=0)
        self.assertTrue(bool(torch.isfinite(advantage).all()))
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        diagnostics = ppo_update(policy, optimizer, rollout, epochs=1,
                                 minibatch_size=2)
        self.assertEqual(diagnostics["updates"], 1)
        self.assertTrue(any(not torch.equal(left, right)
                            for left, right in zip(before, policy.parameters())))
        rollout.dones[-1] = True
        _, terminal_returns = rollout.advantages(gamma=1, gae_lambda=1)
        torch.testing.assert_close(terminal_returns[0], torch.tensor([.1]),
                                   atol=1e-5, rtol=0)

    def test_beam_and_user_permutation_equivariance(self):
        rng = np.random.default_rng(51001)
        beam_order = rng.permutation(100)
        inverse = np.empty_like(beam_order)
        inverse[beam_order] = np.arange(100)
        user_order = rng.permutation(100)
        scene = self.scene
        remapped = CoverageScene(scene.scene_id,
                                 scene.positions[user_order],
                                 scene.demands[user_order],
                                 scene.centers[beam_order],
                                 [scene.aircraft_ids[index] for index in user_order],
                                 {}, scene.source_hash)
        assignment = inverse[np.arange(100)][user_order]
        state = ReconstructionState.from_scene(remapped,
                                               assignment=assignment,
                                               config=self.config)
        first = graph_observation(self.state, self.config)
        second = graph_observation(state, self.config)
        np.testing.assert_allclose(second["beam"], first["beam"][beam_order],
                                   atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(second["relation"],
                                   first["relation"][np.ix_(beam_order, beam_order)],
                                   atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(second["global"], first["global"],
                                   atol=1e-5, rtol=1e-5)
        policy = RelationPolicy()
        with torch.no_grad():
            value_a = policy.value(stack_observations([first], "cpu"))
            value_b = policy.value(stack_observations([second], "cpu"))
        torch.testing.assert_close(value_a, value_b, atol=1e-5, rtol=1e-5)
        candidate_set = [ReconstructionAction("stop"),
                         ReconstructionAction("repartition2", (0, 1), 0),
                         ReconstructionAction("repartition2", (0, 2), 1),
                         ReconstructionAction("assign", (7, 8), 2)]
        mapped_set = [ReconstructionAction(action.operation,
                                           tuple(int(inverse[index]) for index in action.beams),
                                           action.template)
                      for action in candidate_set]
        rule_a = _choose_rule(self.state, candidate_set)
        rule_b = _choose_rule(state, mapped_set)
        self.assertEqual(rule_b.operation, rule_a.operation)
        self.assertEqual(rule_b.beams,
                         tuple(int(inverse[index]) for index in rule_a.beams))
        self.assertEqual(rule_b.template, rule_a.template)
        chosen = candidate_set[2]
        mapped = mapped_set[2]
        original_identity = action_identity(chosen)
        mapped_identity = action_identity(mapped)
        original_masks = torch.as_tensor(np.stack([
            prefix_mask(candidate_set, original_identity[:head])
            for head in range(5)]))[None]
        mapped_masks = torch.as_tensor(np.stack([
            prefix_mask(mapped_set, mapped_identity[:head])
            for head in range(5)]))[None]
        with torch.no_grad():
            probability_a, _, _ = policy.evaluate_choices(
                stack_observations([first], "cpu"),
                torch.tensor([original_identity]), original_masks)
            probability_b, _, _ = policy.evaluate_choices(
                stack_observations([second], "cpu"),
                torch.tensor([mapped_identity]), mapped_masks)
        torch.testing.assert_close(probability_a, probability_b,
                                   atol=1e-5, rtol=1e-5)

    def test_vector_environment_state_restore_and_search_charge(self):
        dataset = CoverageDataset([self.scene])
        env = ReconstructionVectorEnv(dataset, [self.state], self.config,
                                      num_envs=2, seed=51001)
        saved = env.state_dict()
        candidates = env.candidates()[0]
        stop = next(action for action in candidates if action.operation == "stop")
        rewards, dones, events, completed = env.step([stop, stop])
        self.assertEqual(len(completed), 2)
        self.assertTrue(dones.all())
        env.load_state_dict(saved)
        self.assertEqual(env.state_dict()["episode_serial"], [0, 0])
        self.assertEqual(env.scene_indices, [0, 0])
        action, searched, info, trials, metrics = _choose_search(
            self.state, candidates, self.config, width=2)
        self.assertGreaterEqual(trials, 1)
        self.assertEqual(searched.geometry_calls, info["geometry_calls_total"])
        self.assertEqual(searched.fit_candidate_checks,
                         info["fit_candidate_checks_total"])
        self.assertEqual(searched.assignment_candidate_checks,
                         info["assignment_candidate_checks_total"])
        self.assertEqual(searched.ray_pair_checks,
                         info["ray_pair_checks_total"])
        self.assertAlmostEqual(metrics["quality"], state_metrics(searched, self.config)["quality"])

    def test_checkpoint_restores_policy_optimizer_rng_and_environment(self):
        env = ReconstructionVectorEnv(CoverageDataset([self.scene]), [self.state],
                                      self.config, num_envs=1, seed=51001)
        policy = RelationPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        config = reconstruction_default_config()
        job = {"run_id": "test-reconstruction", "git_commit": "frozen-sha",
               "source_root": "frozen-source"}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            checksum = _checkpoint(directory, job, config, policy, optimizer, env,
                                   step=1, update=2, episode=3,
                                   last_validation=0, elapsed=.25,
                                   episode_returns=np.array([.1]),
                                   episode_traces=[[]], gate_passed=False)
            self.assertEqual(len(checksum), 64)
            expected = [parameter.detach().clone() for parameter in policy.parameters()]
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.add_(1)
            payload = _restore(directory, job, config, policy, optimizer, env,
                               device="cpu")
            self.assertEqual(payload["step"], 1)
            self.assertEqual(payload["update"], 2)
            self.assertEqual(payload["episode"], 3)
            self.assertEqual(env.state_dict()["scene_indices"], [0])
            for actual, saved in zip(policy.parameters(), expected):
                torch.testing.assert_close(actual, saved, rtol=0, atol=0)

    def test_summary_keeps_all_scene_denominator_and_missing_values(self):
        timing = {name: 0.0 for name in ("initialization_seconds", "policy_seconds",
                                         "reconstruction_seconds", "final_check_seconds",
                                         "total_seconds")}
        rows = [
            {"initialization_valid": True, "metrics": {"coverage_users": 1.0,
                 "coverage_demand": 1.0, "coverage_100": True, "target_met": False},
             "macro_steps": 1, "geometry_calls": 2, "attempts": 1, **timing},
            {"initialization_valid": False, "metrics": {"coverage_users": .5,
                 "coverage_demand": .6, "coverage_100": False, "target_met": False},
             "macro_steps": 0, "geometry_calls": 0, "attempts": 0, **timing},
            {"initialization_valid": False, "metrics": {}, "macro_steps": 0,
             "geometry_calls": 0, "attempts": 0, **timing},
        ]
        summary = _summary(rows)
        self.assertEqual(summary["scenes"], 3)
        self.assertEqual(summary["strict_coverage_scenes"], 1)
        self.assertEqual(summary["strict_coverage_missing_metrics"], 1)
        self.assertIsNone(summary["coverage_users_mean"])
        self.assertEqual(summary["coverage_users_available_count"], 2)
        self.assertAlmostEqual(summary["coverage_users_defined_mean"], .75)

    def test_resume_discards_uncommitted_validation_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            metrics = Path(temp) / "metrics.jsonl"
            rows = [
                {"step": 16384, "group": "validation", "scheduled_step": 16384},
                {"step": 32768, "group": "validation", "scheduled_step": 32768},
                {"step": 32768, "group": "ppo_update"},
            ]
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows),
                               encoding="utf-8")
            _truncate_to_checkpoint(metrics, "step", 32768,
                                    validation_maximum=16384)
            retained = [json.loads(line) for line in metrics.read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(retained, [rows[0], rows[2]])

    def test_evaluation_result_binds_raw_rows_sha(self):
        config = reconstruction_default_config()
        config["coverage"]["reconstruction"]["route"] = "R0"
        config["coverage"]["method"] = "none"
        config["datasets"] = {"test": {"sha256": "a" * 64}}
        config["reconstruction_starts"] = {"test": {"sha256": "b" * 64}}
        dataset = CoverageDataset([self.scene])
        start_records = [{"initialization_seconds": .1}]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            job = {"run_id": "isolated-evaluation-smoke", "git_commit": "test-only",
                   "execution_target": "local-test-only"}
            with (patch("satellite_rl.coverage.reconstruction_runner._dataset",
                        return_value=dataset),
                  patch("satellite_rl.coverage.reconstruction_runner._load_starts",
                        return_value=([self.state], start_records)),
                  patch("satellite_rl.coverage.reconstruction_runner.config_hash",
                        return_value="short-test-config")):
                _evaluate(directory, job, config, resume=False)
            result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
            artifacts = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
            raw_sha = hashlib.sha256((directory / "results.partial.jsonl").read_bytes()).hexdigest()
            self.assertEqual(result["rows_sha256"], raw_sha)
            self.assertEqual(artifacts["raw_results_sha256"], raw_sha)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA smoke requires a GPU")
    def test_isolated_cuda_train_gate_validation_and_same_id_resume(self):
        """Eight synthetic interactions verify mechanics, not learning quality."""
        config = reconstruction_default_config()
        config["coverage"]["reconstruction"].update(
            neighbor_count=1, split_target_count=4, max_macro_steps=2)
        config["training"].update(total_steps=8, num_envs=2, rollout_steps=2,
                                  minibatch_size=4, epochs=1,
                                  validation_interval=4, checkpoint_interval=4,
                                  mechanism_gate_step=4)
        config["datasets"] = {role: {"sha256": role * 64}
                              for role in ("train", "validation")}
        config["reconstruction_starts"] = {role: {"sha256": role * 64}
                                            for role in ("train", "validation")}
        core_config = ReconstructionConfig(max_macro_steps=2,
                                           neighbor_count=1,
                                           split_target_count=4)
        start = ReconstructionState.from_scene(self.scene, config=core_config)
        dataset = CoverageDataset([self.scene])
        starts = ([start], [{"initialization_seconds": .01}])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            job = {"run_id": "isolated-gpu-smoke", "git_commit": "test-only",
                   "source_root": str(directory), "model_dir": str(directory / "models"),
                   "execution_target": "local-gpu-smoke"}
            preflight = {"gpu": torch.cuda.get_device_name(0),
                         "python_prefix": "local-test-only",
                         "execution_target": "local-gpu-smoke"}
            with (patch("satellite_rl.coverage.reconstruction_runner._dataset",
                        return_value=dataset),
                  patch("satellite_rl.coverage.reconstruction_runner._load_starts",
                        return_value=starts),
                  patch("satellite_rl.coverage.reconstruction_runner.config_hash",
                        return_value="short-test-config")):
                _train(directory, job, config, False, preflight)
                status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["stop_reason"], "mechanism_gate_pending")
                self.assertEqual(status["env_step"], 4)
                node = json.loads((directory / "validation_nodes" / "4.json").read_text(
                    encoding="utf-8"))
                self.assertEqual(node["scene_count"], 1)
                audit = directory / "audit.json"
                audit.write_text('{"passed":true}', encoding="utf-8")
                audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
                (directory / "mechanism_gate.json").write_text(json.dumps({
                    "approved": True, "checkpoint_sha256": status["checkpoint_sha256"],
                    "audit_path": str(audit), "audit_sha256": audit_sha}), encoding="utf-8")
                _train(directory, job, config, True, preflight)
                status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["stage"], "completed")
                self.assertEqual(status["env_step"], 8)
                self.assertTrue((directory / "models" / "final" / "policy.pt").is_file())
                evaluations = [json.loads(line) for line in (directory /
                    "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["scheduled_step"] for row in evaluations], [4, 8])


if __name__ == "__main__":
    unittest.main()
