"""Encoder size and a real masked SAC update on a small synthetic replay."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from satellite_rl.power30.agent import Power30Agent
from satellite_rl.s2fix.tsac import config_for, make_env


class TSACFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mlp_parameter_match_and_real_update(self):
        transformer = Power30Agent(config_for())
        mlp = Power30Agent(config_for(encoder='mlp'))
        counts = [sum(p.numel() for p in a.actor.encoder.parameters())
                  for a in (transformer, mlp)]
        self.assertLess(abs(counts[0] - counts[1]) / counts[0], .01)
        state = np.linspace(0, 1, 308, dtype=np.float32)
        mask = dict(frequency_slots=np.ones((100, 10), bool), power=np.ones(11, bool))
        before = mlp.actor.encoder.net[0].weight.detach().clone()
        for i in range(256):
            mlp.observe(state, (0, 0, 0), .1, state, True, action_mask=mask)
        result = mlp.update()
        self.assertEqual(mlp.update_step, 1)
        self.assertTrue(np.isfinite(result['actor_loss']))
        self.assertFalse(torch.equal(before, mlp.actor.encoder.net[0].weight))
        self.assertEqual(transformer.train['warmup_steps'], 2000)
        self.assertEqual(mlp.train['warmup_steps'], 2000)

    def test_network_size_override_and_policy_reload(self):
        import tempfile
        from pathlib import Path
        from satellite_rl.s2fix.tsac import save_policy, load_policy
        default = Power30Agent(config_for())
        self.assertEqual(default.parameter_counts()['actor'], 419449)
        self.assertNotIn('network_overrides', config_for())
        size = dict(embedding_dim=32, num_heads=4, num_layers=1, feedforward_dim=128)
        small = Power30Agent(config_for(network=size))
        self.assertEqual(small.actor.encoder.pos_embed.shape[-1], 32)
        self.assertEqual(len(small.actor.encoder.net.layers), 1)
        self.assertEqual(small.parameter_counts()['actor'], 18521)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_policy(small, small.config, path, 0)
            loaded = load_policy(path)
        for name in ('actor', 'critic1', 'critic2'):
            for a, b in zip(getattr(small, name).state_dict().values(),
                            getattr(loaded, name).state_dict().values()):
                self.assertTrue(torch.equal(a, b))

    def test_plain_v2_episode_and_restore_without_hashing(self):
        from satellite_rl.shared_interface.stages import make_case, make_layout, UniformStage2Input
        from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2
        metadata = dict(case_id='small', actual_role='development', source_role='development',
                        original_mother_id='small', UTC='2024-01-12T00:00:00Z',
                        source_asset_SHA='source.npz', source={})
        with patch('hashlib.sha256', side_effect=AssertionError('SHA called')):
            case = make_case(metadata,['a','b'],[[45.,10.],[46.,11.]],
                             request_bps=86400000,plain=True)
            layout = make_layout(case,beam_ids=['x','y'],source_slot_ids=['x','y'],
                anchor_aircraft_ids=['a','b'],centres_lat_lon_deg=case['lat_lon_deg'],
                full_width_deg=[.3,.3],owner_aircraft_to_beam_id=[['a','x'],['b','y']],
                Ncap=2,geometry_Omega_cap_sr=1.,geometry_Omega_sr=.01,
                geometry_audit={},source_and_cost={},stack_id=[-1,-1],plain=True)
            stage = UniformStage2Input(case,layout,None,PHYSICS_V2,plain=True)
            env = make_env(stage)
            first = env.step((0,0,0))
            saved = env.state_dict()
            restored = make_env(stage)
            restored.load_state_dict(saved)
            np.testing.assert_array_equal(first[0],restored._observation())
            # G1 places two distant beams in the same group; consume the next slot.
            f,k = np.argwhere(restored.action_mask()['frequency_slots'])[0]
            _,_,done,info = restored.step((int(f),int(k),0))
            self.assertTrue(done)
            self.assertEqual(info['allocation']['physics_version'],PHYSICS_V2['version'])


if __name__ == '__main__':
    unittest.main()
