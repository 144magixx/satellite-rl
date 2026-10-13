"""Bounded algorithm qualification, with no scientific training or new physics."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import WorkbenchAgent
from satellite_rl.t0_comparison.algorithms import build_agent, masked_additive_argmax

CONFIG = Path(__file__).parent / 'fixtures' / 't0_original_config.json'


def config(algorithm):
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    c['algorithm'] = algorithm
    c['training'].update(batch_size=2, ppo_epochs=1, rollout_steps=2)
    return c


def mask():
    fl = np.zeros((100, 10), dtype=bool)
    fl[2, 3] = fl[7, 1] = True
    p = np.zeros(10, dtype=bool)
    p[1] = p[8] = True
    return {'frequency_slots': fl, 'power': p}


class T0ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_sac_and_identical_initial_policy(self):
        c = config('sac')
        torch.manual_seed(51001)
        original = WorkbenchAgent(c, 207, (100, 10, 10))
        torch.manual_seed(51001)
        actual = build_agent(c, 207)
        self.assertIs(type(actual), WorkbenchAgent)
        for arm in ('sac', 'ppo', 'dqn'):
            torch.manual_seed(51001)
            agent = build_agent(config(arm), 207)
            for key, tensor in original.actor.state_dict().items():
                self.assertTrue(torch.equal(tensor, agent.policy_network().state_dict()[key]), (arm, key))

    def test_legal_sampling_and_checkpoint_continuation(self):
        x, m = np.zeros(207, np.float32), mask()
        for arm in ('ppo', 'dqn'):
            torch.manual_seed(51001)
            agent = build_agent(config(arm), 207)
            for step in range(12):
                action = agent.act(x, env_step=step, action_mask=m)
                self.assertTrue(m['frequency_slots'][action[0], action[1]])
                self.assertTrue(m['power'][action[2]])
            saved = copy.deepcopy(agent.state_dict())
            random_state = torch.get_rng_state()
            expected = agent.act(x, action_mask=m)
            restored = build_agent(config(arm), 207)
            restored.load_state_dict(saved)
            torch.set_rng_state(random_state)
            self.assertEqual(expected, restored.act(x, action_mask=m))
            if arm == 'dqn':
                self.assertEqual(restored.last_action_info['epsilon'], .05)

    def test_true_updates_and_terminal_zero_bootstrap(self):
        x, m = np.zeros(207, np.float32), mask()
        for arm in ('sac', 'ppo', 'dqn'):
            torch.manual_seed(51001)
            agent = build_agent(config(arm), 207)
            before = {k: v.clone() for k, v in agent.policy_network().state_dict().items()}
            for step, reward in enumerate((1., -1.)):
                a = agent.act(x, action_mask=m, env_step=step)
                agent.observe(x, a, reward, x+.01, True, action_mask=m)
            result = agent.update()
            self.assertTrue(all(np.isfinite(v) for v in result.values()))
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in agent.policy_network().state_dict().items()), arm)
            if arm == 'dqn':
                self.assertAlmostEqual(result['target_mean'], 0., places=6)
            if arm == 'ppo':
                self.assertFalse(agent.rollout)

    def test_nonterminal_mask_and_budget_tail(self):
        x, m = np.zeros(207, np.float32), mask()
        ppo = build_agent(config('ppo'), 207)
        a = ppo.act(x, action_mask=m)
        with self.assertRaises(ValueError):
            ppo.observe(x, a, 1., x, False, action_mask=m)
        ppo.observe(x, a, 1., x, False, action_mask=m, next_action_mask=m)
        self.assertFalse(ppo.ready(1, False))
        self.assertTrue(ppo.ready(1, True))
        ppo.flush()
        self.assertEqual(ppo.interactions, 1)
        self.assertFalse(ppo.rollout)

    def test_double_dqn_online_selects_target_evaluates(self):
        agent = build_agent(config('dqn'), 207)
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for head in network.heads:
                    head.weight.zero_()
                    head.bias.zero_()
            agent.online.heads[0].bias[7] = 10.
            agent.online.heads[2].bias[8] = 10.
            agent.target.heads[0].bias[2] = 100.
            agent.target.heads[0].bias[7] = 2.
            agent.target.heads[1].bias[1] = 3.
            agent.target.heads[2].bias[8] = 4.
            agent.target.heads[2].bias[1] = 100.
        x, m = np.zeros(207, np.float32), mask()
        for _ in range(2):
            agent.observe(x, (2, 3, 1), 1., x, False, action_mask=m, next_action_mask=m)
        result = agent.update()
        self.assertAlmostEqual(result['target_mean'], 1.+.99*9., places=5)

    def test_additive_argmax_matches_complete_joint_enumeration(self):
        m = mask()
        fl, p = torch.tensor(m['frequency_slots'])[None], torch.tensor(m['power'])[None]
        generator = torch.Generator().manual_seed(51)
        heads = [torch.randn((1, d), generator=generator) for d in (100, 10, 10)]
        chosen = masked_additive_argmax(heads, fl, p)[0]
        table = heads[0][:, :, None, None]+heads[1][:, None, :, None]+heads[2][:, None, None, :]
        index = table.masked_fill(~(fl[:, :, :, None]&p[:, None, None, :]), -torch.inf).flatten(1).argmax(1).item()
        self.assertEqual(chosen.tolist(), [index//100, index//10%10, index%10])
        for h in heads:
            h.zero_()
        self.assertEqual(masked_additive_argmax(heads, fl, p)[0].tolist(), [2, 3, 1])

    def test_reject_f20_contract(self):
        c = config('ppo')
        c['training']['gamma'] = 1.
        with self.assertRaises(ValueError):
            build_agent(c, 207)

    def test_original_reward_and_selected_power_in_real_environment(self):
        from satellite_rl.t0_comparison.environment import make_env, summary
        c = config('ppo')
        self.assertEqual(summary(c)['state_dim'], 207)
        env = make_env(c)
        state = env.reset(seed=20260928)
        self.assertEqual(len(state), 207)
        self.assertEqual(env.raw_env.beam_num, 100)
        for action, expected_power in (((0, 0, 9), 50.), ((10, 1, 0), 5.)):
            _, reward, done, info = env.step(action)
            self.assertFalse(done)
            self.assertAlmostEqual(info['actual_action']['power_w'], expected_power, places=8)
            self.assertAlmostEqual(reward, info['legacy_reward'], places=12)
            sat = 1. if info['satisfaction_increment'] >= 1. else -1.
            capacity = float(env.raw_env.beam_bps[-1])
            demand = float(np.asarray(info['required_rate']).item())
            penalty = max(abs(capacity-demand)-5e7, 0.)/1e9
            expected = sat+float(np.asarray(info['sgm_rate']).item())-penalty-info['occupy']/info['origin_slots']
            self.assertAlmostEqual(reward, expected, places=10)


if __name__ == '__main__':
    unittest.main()
