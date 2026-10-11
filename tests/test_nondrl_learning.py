"""Supervised objectives, information boundary, true updates and recovery."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.coverage.scenes import CoverageScene
from satellite_rl.coverage.variable_design import Layout
from satellite_rl.coverage.nondrl.config import NondrlConfig
from satellite_rl.coverage.nondrl.core import Problem
from satellite_rl.coverage.nondrl.learning import (collate, load_policy, retained_candidates,
                                                   train, validate_sample, LearningPolicy,
                                                   LEARNING_METHODS, run_learning, audit_ranker)
from satellite_rl.coverage.nondrl.models import (center_rays, construction_loss,
                                                linear_assignment, make_model, supervised_loss)
from satellite_rl.coverage.nondrl.teachers import (EvidenceWriter, generate_teacher,
                                                   mlm_sample, scene_features, state_sample,
                                                   trial_label)


def physical(scene, layout):
    return {"aircraft_service": {"demand_weighted_carried_fraction": 1.},
            "metrics": {"total_power_w": 100.}}


def fixture():
    positions = np.array([[45., 10.], [45.1, 10.1], [45.3, 10.3], [45.4, 10.5]])
    scene = CoverageScene("mother", positions, np.ones(4), np.empty((0, 2)),
                          ["a", "b", "c", "d"], {}, "source")
    initial = Layout(positions[[0, 3]], [.5, .5], [0, 0, 0, 1], tuple(scene.aircraft_ids))
    problem = Problem(scene, NondrlConfig(online_seconds=30.), initial=initial, physical_evaluator=physical)
    return problem


def examples():
    problem = fixture()
    mlm = mlm_sample(problem, problem.current, np.random.default_rng(1), .5)
    objects = problem.menu()
    bc = state_sample(problem, objects)
    trials = [problem.preview(obj) for obj in objects]
    positive = next(i for i, trial in enumerate(trials) if trial["accepted"])
    bc.update(positive_actions=[positive], stop_positive=False,
              trials=[{"result": {"accepted": trial["accepted"]}} for trial in trials])
    direct = {"mother_id": "mother", "aircraft": scene_features(problem.scene).tolist(),
              "positions": problem.scene.positions.tolist(), "target_n": 2,
              "references": [{"assignment": [0, 0, 1, 1], "centers": problem.current.centers.tolist()}]}
    rank = state_sample(problem, objects)
    rank.update(known=[True] * len(objects), utility=[1. if i == positive else -1. for i in range(len(objects))])
    region = copy.deepcopy(rank)
    return {"01": mlm, "02": bc, "03": direct, "04": rank, "05": region}


def evidence(folder, route, sample):
    writer = EvidenceWriter(folder, route)
    writer.append(sample)
    writer.manifest["status"] = "completed"
    writer.flush()
    return Path(folder) / "manifest.json"


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_five_distinct_objectives_really_update(self):
        values = examples()
        for route, sample in values.items():
            with self.subTest(route=route):
                self.assertTrue(validate_sample(sample, route))
                model = make_model(route, {"hidden": 16, "layers": 1, "heads": 4})
                batch = collate([sample, copy.deepcopy(sample)], route)
                before = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                optimizer.zero_grad()
                loss = supervised_loss(route, model(batch), batch, [sample, sample])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(p, old) for p, old in zip(model.parameters(), before)))
        self.assertEqual(len({type(make_model(route)) for route in values}), 5)

    @unittest.skipUnless(torch.cuda.is_available(), "actual CUDA unavailable")
    def test_five_cuda_batched_updates(self):
        for route, sample in examples().items():
            with self.subTest(route=route):
                model = make_model(route, {"hidden": 16, "layers": 1, "heads": 4}).cuda()
                batch = collate([sample, sample], route, "cuda")
                before = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.zero_grad()
                loss = supervised_loss(route, model(batch), batch, [sample, sample])
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                self.assertTrue(any(not torch.equal(p, old) for p, old in zip(model.parameters(), before)))

    def test_mlm_rejects_full_hidden_load_leak(self):
        sample = examples()["01"]
        self.assertTrue(validate_sample(sample, "01"))
        leak = copy.deepcopy(sample)
        leak["candidates"][0][5] += 1.
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_sample(leak, "01")

    def test_mlm_illegal_label_and_original_edge(self):
        sample = examples()["01"]
        sample["legal"][0][sample["labels"][0]] = False
        with self.assertRaisesRegex(ValueError, "outside legal"):
            validate_sample(sample, "01")
        mask = retained_candidates([[99., 2., 1.]], [[True, True, True]], [2], maximum=2)
        self.assertTrue(mask[0, 2])
        self.assertEqual(int(mask.sum()), 2)

    def test_mask_model_is_equivariant_to_both_set_orders(self):
        sample = examples()["01"]
        model = make_model("01", {"hidden": 16, "layers": 1, "heads": 4}).eval()
        original = model(collate([sample], "01"))["logits"].detach()
        changed = copy.deepcopy(sample)
        airplane, candidates = [3, 2, 0, 1], [1, 0]
        inverse = np.argsort(candidates)
        for key in ("aircraft", "hidden", "labels", "visible", "original_assignment"):
            changed[key] = np.asarray(sample[key])[airplane].tolist()
        for key in ("labels", "original_assignment"):
            changed[key] = inverse[changed[key]].tolist()
        changed["visible"] = [int(inverse[x]) if x >= 0 else -1 for x in changed["visible"]]
        changed["candidates"] = np.asarray(sample["candidates"])[candidates].tolist()
        changed["legal"] = np.asarray(sample["legal"])[np.ix_(airplane, candidates)].tolist()
        actual = model(collate([changed], "01"))["logits"].detach()
        torch.testing.assert_close(actual, original[:, airplane][:, :, candidates], atol=2e-5, rtol=2e-5)

    def test_whole_layout_matching_permutation_invariant(self):
        sample = examples()["03"]
        logits = torch.tensor([[[4., 0.], [4., 0.], [0., 4.], [0., 4.]]], requires_grad=True)
        centers = torch.tensor([sample["references"][0]["centers"]])
        output = {"logits": logits, "rays": center_rays(centers)}
        first = construction_loss(output, [sample])
        changed = copy.deepcopy(sample)
        changed["references"][0]["centers"].reverse()
        changed["references"][0]["assignment"] = [1 - x for x in sample["references"][0]["assignment"]]
        second = construction_loss(output, [changed])
        torch.testing.assert_close(first, second)
        leak = copy.deepcopy(sample)
        leak["beams"] = [[0.] * 8]
        with self.assertRaisesRegex(ValueError, "layout context"):
            validate_sample(leak, "03")

    def test_hungarian_matches_enumerated_optimum(self):
        cost = np.array([[3., 1., 9.], [2., 7., 4.], [8., 2., 1.]])
        actual = cost[np.arange(3), linear_assignment(cost)].sum()
        expected = min(sum(cost[i, p[i]] for i in range(3)) for p in itertools.permutations(range(3)))
        self.assertEqual(actual, expected)

    def test_unknown_rank_trials_excluded_and_all_negative_regions_retained(self):
        sample = examples()["04"]
        sample["known"][0] = False
        sample["utility"][0] = -1.
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            validate_sample(sample, "04")
        sample["utility"][0] = 0.
        region = examples()["05"]
        region["utility"] = [-1.] * len(region["utility"])
        self.assertTrue(validate_sample(region, "05"))
        label = trial_label(fixture(), {"reason": "rebuild_no_incumbent_unknown"}, .5)
        self.assertFalse(label["known"])
        self.assertEqual(label["utility"], 0.)

    def test_bc_refuses_truncated_stop_or_fabricated_action(self):
        sample = examples()["02"]
        sample.update(positive_actions=[], stop_positive=True, stop_reason="budget")
        with self.assertRaisesRegex(ValueError, "exhaustive"):
            validate_sample(sample, "02")

    def test_fixed_rank_audit_handles_unknown_and_zero_denominator(self):
        sample = examples()["04"]
        policy = LearningPolicy("04", make_model("04", {"hidden": 16, "layers": 1, "heads": 4}))
        result = audit_ranker(policy, sample)
        self.assertTrue(result["complete"])
        self.assertEqual(result["qualified_improvements"], 1)
        self.assertIn(result["model_topk_recall"], (0., 1.))
        sample["utility"] = [0.] * len(sample["utility"])
        result = audit_ranker(policy, sample)
        self.assertTrue(result["no_improvement_state"])
        self.assertIsNone(result["model_topk_recall"])
        sample["known"][0] = False
        result = audit_ranker(policy, sample)
        self.assertFalse(result["complete"])
        sample = examples()["02"]
        sample["trials"][sample["positive_actions"][0]]["result"]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "completed executable"):
            validate_sample(sample, "02")

    def test_checkpoint_resume_matches_uninterrupted_parameters(self):
        sample = examples()["02"]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = evidence(folder / "data", "02", sample)
            config = {"engineering": True, "device": "cpu", "updates": 4,
                      "batch_size": 2, "checkpoint_interval": 1,
                      "network": {"hidden": 16, "layers": 1, "heads": 4}}
            full = train(source, "02", config, folder / "full")
            calls = [0]
            def stop():
                calls[0] += 1
                return calls[0] > 2
            part = train(source, "02", config, folder / "partial", stop_callback=stop)
            self.assertEqual(part["updates"], 2)
            self.assertEqual(part["status"], "stopped")
            resumed = train(source, "02", config, folder / "partial", resume=True)
            expected, actual = load_policy(full["endpoint_path"]), load_policy(resumed["endpoint_path"])
            for left, right in zip(expected.model.parameters(), actual.model.parameters()):
                self.assertTrue(torch.equal(left, right))
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                load_policy(full["endpoint_path"], {"route": "01"})

    def test_teacher_bc_real_trace_and_same_id_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            count = [0]
            def progress(values):
                count[0] += 1
            stop = lambda: count[0] >= 1
            first = generate_teacher([fixture(), fixture()], "02", root,
                                     limits={"max_steps": 2, "per_problem_seconds": 10.},
                                     stop_callback=stop, progress_callback=progress)
            self.assertEqual(first["status"], "stopped")
            second = generate_teacher([fixture(), fixture()], "02", root,
                                      limits={"max_steps": 2, "per_problem_seconds": 10.}, resume=True)
            self.assertEqual(second["problems"], 2)
            self.assertGreaterEqual(second["trials"], first["trials"])
            from satellite_rl.coverage.nondrl.learning import EvidenceDataset
            data = EvidenceDataset(second["path"], "02")
            self.assertGreater(len(data.keys), 0)

    def test_parallel_teacher_reservations_and_deterministic_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = generate_teacher([fixture(), fixture()], "02", temporary,
                                      limits={"workers": 2, "max_steps": 2, "max_trials": 128,
                                              "per_problem_seconds": 10., "max_seconds": 20.})
            self.assertLessEqual(result["trials"], 128)
            self.assertEqual(result["workers"], 2)
            self.assertFalse(result["reservations"])
            self.assertEqual(result["problems"], 2)
            paths = [record["path"] for record in result["shards"]]
            self.assertEqual(paths, sorted(paths))

    def test_teacher_mid_menu_resume_preserves_completed_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = fixture()
            limits = {"max_steps": 2, "per_problem_seconds": 10.}
            full = generate_teacher([fixture()], "02", root / "full", limits=limits)
            partial = generate_teacher([problem], "02", root / "partial", limits=limits,
                                       stop_callback=lambda: problem.calls["candidates"] >= 3)
            self.assertEqual(partial["status"], "stopped")
            self.assertEqual(partial["trials"], 3)
            self.assertIsNotNone(partial["active"])
            recovered = generate_teacher([fixture()], "02", root / "partial", limits=limits, resume=True)
            self.assertEqual(recovered["trials"], full["trials"])
            self.assertEqual(recovered["rows"], full["rows"])

    def test_teacher_mlm_geometry_pool_frozen_before_solution(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = generate_teacher([fixture()], "01", temporary,
                                      limits={"max_trials": 1, "max_seconds": 2.})
            self.assertEqual(result["trials"], 1)
            from satellite_rl.coverage.nondrl.learning import EvidenceDataset
            dataset = EvidenceDataset(result["path"], "01")
            sample = dataset.shard(0)[0]
            self.assertGreater(len(sample["candidates"]), sample["target_n"])
            self.assertTrue(validate_sample(sample, "01"))
            self.assertEqual(sample["input_pool"], "original_and_scene_geometry_before_teacher")

    def test_all_online_methods_use_common_feasible_executor(self):
        values = examples()
        for route, methods in LEARNING_METHODS.items():
            sample = values[route]
            model = make_model(route, {"hidden": 16, "layers": 1, "heads": 4})
            batch = collate([sample], route)
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.zero_grad()
            supervised_loss(route, model(batch), batch, [sample]).backward()
            optimizer.step()
            policy = LearningPolicy(route, model)
            for method in methods:
                with self.subTest(route=route, method=method):
                    problem = fixture()
                    run_learning(problem, route, policy, method)
                    result = problem.result(route)
                    self.assertTrue(result["metrics_all"]["coverage_100"])
                    self.assertEqual(result["metrics_all"]["beam_count"], 2)
                    self.assertTrue(result["metrics_all"]["demand_conserved"])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            run_learning(fixture(), "05", None, "fake_method")


if __name__ == "__main__":
    unittest.main()
