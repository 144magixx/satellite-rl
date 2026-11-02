"""Pure/mock repair qualification; no environmental or scientific interactions."""
import copy
import unittest
import numpy as np
import torch
from test_t0_comparison import config, mask, build_agent
from satellite_rl.t0_comparison.algorithms import build_agent as science_build, verify_actual_action
from satellite_rl.t0_comparison.contract import canonical_config
from satellite_rl.agents.workbench_algorithms import WorkbenchAgent
from satellite_rl.agents.action_masking import mask_batch, masked_policy


class GuardQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formal_contract_and_explicit_engineering(self):
        c = config('sac')
        with self.assertRaises(ValueError):
            canonical_config(c)
        self.assertEqual(canonical_config(c, 'engineering'), c)
        c['training'].update(total_steps=150000, batch_size=256, ppo_epochs=4, rollout_steps=256)
        c['evaluation'].update(test_count=40)
        canonical_config(c)
        mutations = [('environment','action_mode','feasible_projection'),
            ('environment','basic_fraction',.2), ('network','activation','gelu'),
            ('network','temporal_mode','gru'), ('network','graph_mode','gat'),
            ('training','seed',42), ('training','total_steps',400000),
            ('training','actor_lr',.0003), ('training','update_every',4)]
        for section,key,value in mutations:
            changed = copy.deepcopy(c)
            changed[section][key] = value
            with self.assertRaises(ValueError, msg=key):
                canonical_config(changed)
        changed = copy.deepcopy(c)
        changed['state']['features'].reverse()
        with self.assertRaises(ValueError):
            canonical_config(changed)
        changed = config('sac')
        changed['training']['total_steps'] = 2049
        with self.assertRaises(ValueError):
            canonical_config(changed, 'engineering')

    def test_actual_record_guard_all_arms(self):
        x, m, a = np.zeros(207,np.float32), mask(), (2,3,1)
        valid = dict(indices=list(a),freq=2,slots=4,power_w=10.,charged_power_w=10.,selected_power_w=10.)
        verify_actual_action(a, valid, config('sac'))
        for arm in ('sac','ppo','dqn'):
            agent=build_agent(config(arm),207)
            for field,value in (('indices',[2,3,8]),('freq',3),('slots',3),('power_w',15.),('selected_power_w',5.)):
                bad=dict(valid);bad[field]=value
                with self.assertRaises(ValueError):
                    agent.observe(x,a,1.,x,True,action_mask=m,actual_action=bad)
            self.assertEqual(len(agent.rollout) if arm=='ppo' else len(agent.replay),0)

    def test_engineering_view_has_no_formal_evaluation(self):
        from satellite_rl.t0_comparison.environment import environment_config, summary
        c=config('ppo')
        view=environment_config(c,purpose='engineering')
        self.assertEqual(view['evaluation']['extra_steps'],[])
        self.assertEqual(view['evaluation']['validation_count'],0)
        self.assertEqual(view['evaluation']['test_count'],0)
        self.assertEqual(summary(c,purpose='engineering')['state_dim'],207)
        self.assertEqual(c['evaluation']['extra_steps'],[0,2000,5000])
        formal=config('sac')
        formal['training'].update(total_steps=150000,batch_size=256,ppo_epochs=4,rollout_steps=256)
        formal['evaluation'].update(test_count=40)
        science_view=environment_config(formal)
        self.assertEqual(science_view['evaluation']['extra_steps'],[0,2000,5000])
        self.assertEqual(science_view['evaluation']['interval_steps'],10000)

    def test_ppo_ratio_one_mask_ownership_and_update_restore(self):
        torch.manual_seed(51001)
        c=config('ppo');agent=build_agent(c,207);x=np.zeros(207,np.float32);m=mask()
        for reward in (1.,-1.):
            a=agent.act(x,action_mask=m)
            old=agent.last_action_info['log_prob']
            fm,pm=mask_batch([m],(100,10,10),'cpu')
            _,_,lf,lp,*_=masked_policy(agent.actor(torch.tensor(x)[None]),fm,pm)
            self.assertAlmostEqual(float((lf[0,a[0],a[1]]+lp[0,a[2]]-old).exp()),1.,places=6)
            agent.observe(x,a,reward,x,True,action_mask=m)
        saved=copy.deepcopy(agent.state_dict());m['frequency_slots'][:]=False;m['power'][:]=False
        self.assertTrue(agent.rollout[0][7]['frequency_slots'].any())
        agent.update();first=copy.deepcopy(agent.state_dict())
        restored=build_agent(c,207);restored.load_state_dict(saved);restored.update()
        for name,network in first['networks'].items():
            for key,value in network.items():
                self.assertTrue(torch.equal(value,restored.state_dict()['networks'][name][key]))

    def test_sac_one_update_exact_original_equivalence(self):
        c=config('sac');x=np.zeros(207,np.float32);m=mask()
        torch.manual_seed(51001);original=WorkbenchAgent(c,207,(100,10,10))
        torch.manual_seed(51001);guarded=build_agent(c,207)
        for agent in (original,guarded):
            for reward in (1.,-1.):
                agent.observe(x,(2,3,1),reward,x,True,action_mask=m)
        rng=torch.get_rng_state();expected=original.update()
        torch.set_rng_state(rng);actual=guarded.update()
        self.assertEqual(expected,actual)
        for name,network in original.state_dict()['networks'].items():
            for key,value in network.items():
                self.assertTrue(torch.equal(value,guarded.state_dict()['networks'][name][key]))
        saved=copy.deepcopy(guarded.state_dict())
        restored=build_agent(c,207);restored.load_state_dict(saved)
        rng=torch.get_rng_state();expected_action=guarded.act(x,action_mask=m)
        torch.set_rng_state(rng);self.assertEqual(expected_action,restored.act(x,action_mask=m))

    def test_nonterminal_dqn_update_restore_same_replay_rng(self):
        c=config('dqn');x=np.zeros(207,np.float32);m=mask()
        torch.manual_seed(51001);agent=build_agent(c,207)
        for reward in (1.,-1.):
            agent.observe(x,(2,3,1),reward,x+.01,False,action_mask=m,next_action_mask=m)
        agent.update()
        saved=copy.deepcopy(agent.state_dict())
        restored=build_agent(c,207);restored.load_state_dict(saved)
        self.assertEqual(agent.update(),restored.update())
        for name,network in agent.state_dict()['networks'].items():
            for key,value in network.items():
                self.assertTrue(torch.equal(value,restored.state_dict()['networks'][name][key]))


if __name__=='__main__':
    unittest.main()
