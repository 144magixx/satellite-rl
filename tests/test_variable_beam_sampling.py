"""Focused checks for variable beam scene selection and real replay buckets."""

import unittest
from io import BytesIO

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import Replay
from satellite_rl.workbench.runner import _validation_due
from satellite_rl.workbench.scene_order import VariableBeamSceneOrder, beam_bucket


class VariableSceneOrderTests(unittest.TestCase):
    def _order(self, mode="mixed_four_buckets"):
        records = []
        for n in (50, 51, 100, 101, 102, 150, 151):
            for mother in ("m0", "m1"):
                for load in (0.75, 1.25):
                    records.append({"beam_count": n, "mother_scene_id": mother,
                                    "load_multiplier": load})
        return VariableBeamSceneOrder(mode, 51001,
                                      [f"sha-{i}" for i in range(len(records))], records)

    def test_bucket_rotation_and_deterministic_resume(self):
        order = self._order()
        first = [order.selection(i) for i in range(80)]
        self.assertEqual([row["bucket"] for row in first[:4]],
                         ["anchor_100", "lower", "middle", "upper"])
        self.assertEqual({bucket: sum(row["bucket"] == bucket for row in first)
                          for bucket in order.buckets}, {bucket: 20 for bucket in order.buckets})
        self.assertEqual(first[20:], [self._order().selection(i) for i in range(20, 80)])
        order.verify_checkpoint(order.checkpoint_state(32), 32)
        with self.assertRaises(ValueError):
            order.verify_checkpoint(order.checkpoint_state(31), 32)

    def test_v0_only_anchor_and_held_out_rejected(self):
        order = self._order("fixed_100")
        self.assertTrue(all(order.selection(i)["beam_count"] == 100 for i in range(20)))
        for n in (60, 90, 140, 180, 49, 201):
            with self.assertRaises(ValueError):
                beam_bucket(n)


class ReplayBucketTests(unittest.TestCase):
    @staticmethod
    def add(replay, bucket, marker):
        state = np.array([marker], dtype=np.float32)
        replay.add(state, (0, 0, 0), 0.0, state, False, bucket=bucket)

    def test_real_four_bucket_batch_and_full_resume(self):
        replay = Replay(20000, 123, "mixed_four_buckets")
        for marker, bucket in enumerate(replay.bucket_indices):
            self.add(replay, bucket, marker)
        batch = replay.sample(256, "cpu")[0][:, 0].numpy()
        self.assertEqual([int(np.sum(batch == marker)) for marker in range(4)], [64] * 4)
        self.assertEqual(len(replay), 4)
        self.assertTrue(all(n > 0 for n in replay.replacement_draws.values()))
        restored = Replay(20000, 123, "mixed_four_buckets")
        restored.load_state_dict(replay.state_dict())
        self.assertTrue(np.array_equal(replay.sample(256, "cpu")[0].numpy(),
                                       restored.sample(256, "cpu")[0].numpy()))

    def test_ring_eviction_keeps_exact_bucket_index(self):
        replay = Replay(4, 123, "mixed_four_buckets")
        for marker, bucket in enumerate(replay.bucket_indices):
            self.add(replay, bucket, marker)
        self.add(replay, "upper", 4)
        self.assertEqual(replay.bucket_indices["anchor_100"], [])
        self.assertEqual(set(replay.bucket_indices["upper"]), {0, 3})
        with self.assertRaises(ValueError):
            replay.sample(256, "cpu")
        saved = replay.state_dict()
        saved["bucket_indices"]["upper"] = [3]
        with self.assertRaises(ValueError):
            Replay(4, 123, "mixed_four_buckets").load_state_dict(saved)

    def test_v0_single_bucket_batch(self):
        replay = Replay(20000, 123, "fixed_100")
        self.add(replay, "anchor_100", 7)
        self.assertEqual(replay.sample(256, "cpu")[0].shape[0], 256)
        self.assertEqual(len(replay), 1)


class ValidationCadenceTests(unittest.TestCase):
    def test_zero_and_exact_twenty_thousand_nodes(self):
        evaluation = {"interval_steps": 20000, "extra_steps": [0]}
        progress = {"last_validation_step": -1}
        self.assertTrue(_validation_due(evaluation, progress, 0))
        progress["last_validation_step"] = 0
        self.assertFalse(_validation_due(evaluation, progress, 19999))
        self.assertTrue(_validation_due(evaluation, progress, 20000))
        progress["last_validation_step"] = 20000
        self.assertFalse(_validation_due(evaluation, progress, 20000))


class PartialEpisodeSerializationTests(unittest.TestCase):
    def test_environment_continues_identically_after_torch_roundtrip(self):
        from satellite_rl.workbench.config import default_config
        from satellite_rl.workbench.environment import WorkbenchEnv

        env = WorkbenchEnv(default_config())
        env.reset(seed=51001)
        action = (0, 0, 0)
        env.step(action)
        stream = BytesIO()
        torch.save(env, stream)
        stream.seek(0)
        resumed = torch.load(stream, weights_only=False)
        left = env.step(action)
        right = resumed.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        self.assertEqual(left[1], right[1])
        self.assertEqual(left[2], right[2])


if __name__ == "__main__":
    unittest.main()
