"""Pure arithmetic and mocked boundaries only: zero real PHY/network/update calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from satellite_rl.power30.menus import power_table, legal_power_mask, validate_actual_power, action_schema
from satellite_rl.power30.schema import state_width, schema_identity
from satellite_rl.power30.environment import Power30Env


class Power30PureTests(unittest.TestCase):
    def test_menus_and_reservations(self):
        self.assertEqual([len(power_table(m)) for m in ('P00','P10','P01','P11')],[10,91,11,100])
        for menu in ('P00','P10','P01','P11'):
            self.assertTrue(set(power_table('P00'))<=set(power_table(menu)))
            self.assertEqual(power_table(menu)[-1],50.)
            mask=legal_power_mask(menu,2990.,1)
            self.assertEqual([w for w,ok in zip(power_table(menu),mask) if ok],
                             [w for w in power_table(menu) if w<=5.])
            self.assertEqual(action_schema(menu)['future_reserve_power_w'],5.)
        validate_actual_power('P01',[.5,0.],active=np.array([True,False]))
        with self.assertRaises(ValueError):validate_actual_power('P01',[0.,0.],active=np.array([True,False]))
        with self.assertRaises(ValueError):validate_actual_power('P01',[.5,.5],active=np.array([True,False]))
        with self.assertRaises(ValueError):validate_actual_power('P00',[.5])
        with self.assertRaises(ValueError):validate_actual_power('P11',[50.5])
        with self.assertRaises(ValueError):validate_actual_power('P00',[50.]*61)

    def test_real_selected_schema(self):
        self.assertEqual(state_width('M0'),208);self.assertEqual(state_width('M1'),308)
        a=schema_identity('P00','M0');b=schema_identity('P11','M1')
        self.assertNotIn('margin_summary',a['observation']['fields'])
        self.assertEqual(b['observation']['fields']['margin_summary'],[208,308])
        self.assertNotEqual(a['sha256'],b['sha256'])

    def test_all_eight_schemas_json_roundtrip_exact(self):
        import json
        import hashlib
        from satellite_rl.power30.agent import build_config
        for menu in ('P00','P10','P01','P11'):
            for group in ('M0','M1'):
                value=schema_identity(menu,group)
                restored=json.loads(json.dumps(value))
                self.assertEqual(value,restored)
                body=dict(value);claimed=body.pop('sha256')
                self.assertEqual(hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),claimed)
                config=build_config(menu,group)
                self.assertEqual(config,json.loads(json.dumps(config)))
                self.assertEqual(config['evaluation']['interval_steps'],50000)
                self.assertEqual(config['evaluation']['extra_steps'],[])
                self.assertEqual(config['evaluation']['repeats'],3)
                self.assertEqual(config['evaluation']['action_mode'],'stochastic')

    def fixture(self):
        env=Power30Env.__new__(Power30Env)
        env.indices=np.array([0,1,2]);env.beam_idx=1
        env.groups=np.array([0,2,4]);env.powers=np.array([10.,0.,0.])
        env.rates=np.array([100.,0.,0.]);env.demands=np.array([100.,200.,300.])
        env.physics={'rx_gain_dBi':0.}
        env.kernel={'gain':np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]),'path':np.ones((3,100))}
        env.last_coefficients=SimpleNamespace(noise=np.array([1.,1.]),
            interference=np.array([[.1,0.,0.],[.2,0.,0.]]),slot_owner=np.array([0,0]),occupied_slots=np.array([3,4]))
        return env

    def test_summary_formula_direction_and_no_future(self):
        env=self.fixture()
        with patch('satellite_rl.power30.environment.capacity_bps',side_effect=AssertionError('PHY forbidden')), \
             patch('satellite_rl.power30.environment.raw_slot_coupling',side_effect=AssertionError('coupling forbidden')), \
             patch('satellite_rl.power30.environment.prepare_slot_kernel',side_effect=AssertionError('kernel forbidden')):
            got=env._margin_summary(1)
            self.assertAlmostEqual(got[3],np.log1p(2./2.))
            self.assertAlmostEqual(got[4],np.log1p(2./3.))
            self.assertEqual(np.count_nonzero(got),2)
            env.rates[2]=1e30;env.demands[2]=1.;env.kernel['gain'][2,:]=1e30
            np.testing.assert_array_equal(got,env._margin_summary(1))
            env.rates[0]=200.
            self.assertAlmostEqual(env._margin_summary(1)[3],np.log1p(np.exp(-1)*2./2.))
        env.last_coefficients=None
        np.testing.assert_array_equal(env._margin_summary(1),np.zeros(100))

    def test_hook_before_count_and_failure_not_hidden(self):
        env=Power30Env.__new__(Power30Env);env.call_counts={}
        events=[];env.call_hook=lambda kind,**metadata:events.append((kind,metadata))
        env._newcall('phy');self.assertEqual(events,[('phy',{})]);self.assertEqual(env.call_counts,{'phy':1})
        def reject(*args,**kwargs):raise RuntimeError('cap')
        env.call_hook=reject
        with self.assertRaises(RuntimeError):env._newcall('phy')
        self.assertEqual(env.call_counts,{'phy':1})

    def test_sac_update_and_act_match_workbench_formula_bitwise(self):
        # Power30 的 act/更新只改了主机与GPU的往来（异步上传、一次取回）；同初值同回放，参数与动作应逐位相同。
        import torch
        from satellite_rl.agents.workbench_algorithms import WorkbenchAgent
        from satellite_rl.power30.agent import Power30Agent
        from satellite_rl.s2fix.tsac import config_for
        size=dict(embedding_dim=32,num_heads=4,num_layers=1,feedforward_dim=64)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            agents=[]
            for _ in range(2):
                torch.manual_seed(0)
                agents.append(Power30Agent(config_for(network=size,device=device)))
            rng=np.random.default_rng(0)
            for i in range(300):
                state,following=rng.random(308).astype(np.float32),rng.random(308).astype(np.float32)
                mask=dict(frequency_slots=rng.random((100,10))<.5,power=rng.random(11)<.5)
                mask['frequency_slots'][3,2]=mask['power'][1]=True
                done,reward=i%5==0,float(rng.random())
                following_mask=None if done else dict(frequency_slots=rng.random((100,10))<.5,power=np.ones(11,bool))
                for agent in agents:
                    agent.observe(state,(3,2,1),reward,following,done,action_mask=mask,
                                  next_action_mask=following_mask)
            for _ in range(3):
                old=WorkbenchAgent._update_masked_sac(agents[0])
                new=agents[1].update()
                self.assertEqual(old['actor_loss'],new['actor_loss'])
            for name in ('actor','critic1','critic2','target1','target2'):
                for a,b in zip(getattr(agents[0],name).parameters(),getattr(agents[1],name).parameters()):
                    self.assertTrue(torch.equal(a,b),(device,name))
            torch.manual_seed(1)
            old=[WorkbenchAgent.act(agents[0],state,action_mask=mask) for _ in range(20)]
            old_info=dict(agents[0].last_action_info)
            torch.manual_seed(1)
            new=[agents[1].act(state,action_mask=mask) for _ in range(20)]
            self.assertEqual(old,new)
            self.assertEqual({k:old_info[k] for k in old_info},{k:agents[1].last_action_info[k] for k in old_info})

    def test_mocked_step_and_restore_use_one_phy_no_replay(self):
        from satellite_rl.shared_interface.stages import UniformStage2Input
        from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
        stage=UniformStage2Input.__new__(UniformStage2Input)
        stage.layout=dict(beam_ids=['a','b'],content_sha256='layout',
                          request_bps_by_beam_int64=np.array([100,200]),
                          centres_lat_lon_deg=np.array([[50.,5.],[51.,6.]]),full_width_deg=np.array([.5,.7]))
        stage.physics=dict(k_J_per_K=1.,noise_temperature_K=1.,slot_Hz=1.,rx_gain_dBi=0.)
        stage.processing_index_to_beam_id=['a','b'];stage.group_by_beam_id={'a':0,'b':2}
        kernel=dict(gain=np.ones((2,2)),path=np.ones((2,100)))
        coeff=SimpleNamespace(beam_ids=('a','b'),slot_owner=np.array([0]),occupied_slots=np.array([0]),
              signal=np.ones(1),noise=np.ones(1),interference=np.zeros((1,2)),powers_initial=np.array([.5,0.]),
              active=np.array([True,False]),demand_bps=np.array([100,200]),
              metadata=dict(parent_layout_SHA='layout',physics_config_SHA=PHYSICS_SHA))
        events=[]
        with patch('satellite_rl.power30.environment.prepare_slot_kernel',return_value=kernel) as k, \
             patch('satellite_rl.power30.environment.build_link_coefficients',return_value=coeff) as c, \
             patch('satellite_rl.power30.environment.capacity_bps',return_value=np.array([50.,0.])) as phy, \
             patch('satellite_rl.power30.environment.raw_slot_coupling',return_value=np.zeros((1,1,100))):
            env=Power30Env(stage,'P01','M1',lambda kind,**kw:events.append(kind))
            initial=env._observation();self.assertEqual(initial.shape,(308,))
            obs,reward,done,info=env.step((0,0,0))
            self.assertFalse(done);self.assertEqual(env.powers[0],.5)
            self.assertEqual((k.call_count,c.call_count,phy.call_count),(1,1,1))
            self.assertEqual(events.count('phy'),1);self.assertEqual(info['actual_action']['power_w'],.5)
            saved=env.state_dict()
            # Same env restores directly, without any capacity/kernel/coeff replay.
            env.reset();env.load_state_dict(saved)
            np.testing.assert_array_equal(obs,env._observation())
            self.assertEqual((k.call_count,c.call_count,phy.call_count),(1,1,1))
            corrupt=dict(saved,powers=np.array([5.,0.]))
            with self.assertRaises(ValueError):env.load_state_dict(corrupt)


if __name__=='__main__':unittest.main()
