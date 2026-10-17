import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from satellite_rl.s2fix.search_env import SearchEnv, configuration, run_search, layout_key
from satellite_rl.sac_improvement.agent import ImprovementSAC, pack_actions
from satellite_rl.sac_improvement.config import default_config, validate_config


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.stage = SimpleNamespace(layout=dict(
            request_bps_by_beam_int64=np.array([100., 200., 100., 100.]),
            centres_lat_lon_deg=np.array([[0.,0.], [0.,2.], [1.,0.], [1.,2.]])))
        allocation = (np.array([3,3,4,4]), np.array([0,2,0,3]),
                      np.array([2,3,3,2]), np.full(4, 20.))
        self.greedy = patch("satellite_rl.s2fix.greedy.greedy_allocation", return_value=allocation)
        self.capacity = patch("satellite_rl.s2fix.evaluate.capacities",
                              side_effect=lambda inp,g,s,c,p: np.asarray(c)*30.)
        self.greedy.start()
        self.capacity.start()
        self.addCleanup(self.greedy.stop)
        self.addCleanup(self.capacity.stop)

    def test_groups_xswap_and_budget(self):
        env = SearchEnv(dict(max_decisions=2, max_probes=2))
        state = env.reset(self.stage)
        np.testing.assert_allclose(state[:2800].reshape(200,14)[:4,4], env.groups/7)
        candidates = env.candidates()
        self.assertIn("xswap", [c.kind for c in candidates])
        candidate = next(c for c in candidates if c.kind == "xswap")
        self.assertTrue(env._legal(np.array(candidate.starts), np.array(candidate.slots), np.array(candidate.groups)))
        old_resources = sorted(zip(env.groups, env.current_starts, env.current_slots))
        self.assertEqual(old_resources, sorted(zip(candidate.groups, candidate.starts, candidate.slots)))
        features, mask, ids = pack_actions(env)
        self.assertEqual(features.shape, (17,28))
        env.step(candidate.id)
        env.step("STOP")
        self.assertTrue(env.done)
        self.assertEqual(env.probe_count, 1)
        self.assertGreaterEqual(env.best_eval.W, env.base_eval.W)
        self.assertTrue(env._protected(env.best_eval))

    def test_protection_weighted_service_not_sgm(self):
        env = SearchEnv()
        env.reset(self.stage)
        bad = SimpleNamespace(W=env.base_eval.W-.01, capacity_bps=np.ones(4)*200)
        self.assertFalse(env._protected(bad))
        good = SimpleNamespace(W=env.base_eval.W+.01, capacity_bps=np.ones(4)*20, sgm=-1.)
        self.assertTrue(env._protected(good))
        good.capacity_bps[0] = 1
        self.assertFalse(env._protected(good))

    def test_shared_search_and_sac_update(self):
        for method in ("R", "S1", "S2"):
            result, info = run_search(self.stage, method, max_probes=5, return_info=True)
            self.assertLessEqual(info["probes"], 5)
            self.assertEqual(len(result), 4)
        cfg = configuration(device="cpu", batch_size=2, hidden_width=16)
        env = SearchEnv(cfg)
        state = env.reset(self.stage)
        features, mask, ids = pack_actions(env)
        agent = ImprovementSAC(cfg)
        action = int(np.flatnonzero(mask)[0])
        ns, reward, done, _ = env.step(ids[action])
        nf, nm, _ = pack_actions(env)
        for _ in range(2):
            agent.replay.add(state,features,mask,ids,action,reward,ns,nf,nm,done)
        self.assertTrue(np.isfinite(agent.update()["actor_loss"]))
        self.assertEqual(agent.updates, 1)

    def test_train_checkpoint_and_policy(self):
        from satellite_rl.s2fix.i0 import train, policy
        with tempfile.TemporaryDirectory() as directory, patch(
                "satellite_rl.s2fix.data.load_input", return_value=self.stage):
            checkpoint = train([{}], directory, steps=4, config=dict(
                warmup_steps=0, batch_size=2, hidden_width=16, max_decisions=2,
                max_probes=2, replay_capacity=10))
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved["step"], 4)
            self.assertEqual(saved["config"]["schema_version"], "s2fix_i0_v1")
            self.assertGreater(saved["agent"]["updates"], 0)
            allocation, info = policy(self.stage, checkpoint, return_info=True)
            self.assertEqual(len(allocation), 4)
            self.assertLessEqual(info["probes"], 2)

    def test_old_config_remains_frozen(self):
        cfg = default_config()
        cfg["max_probes"] = 320
        with self.assertRaises(ValueError):
            validate_config(cfg)
        old = ImprovementSAC(dict(default_config(), device="cpu", hidden_width=16))
        self.assertEqual(old.actor.action[0].in_features, 27)


if __name__ == "__main__":
    unittest.main()
