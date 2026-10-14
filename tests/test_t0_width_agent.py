import copy
import unittest
import numpy as np
import torch
from satellite_rl.agents.workbench_algorithms import WorkbenchAgent
from satellite_rl.t0_width.agent import WidthAgent, build_config, state_dimension
from satellite_rl.t0_width.runner import seed_all


class WidthAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_common_parameters_and_only_real_input_columns(self):
        agents = {}
        for group in ("S00", "S10", "S01", "S11"):
            seed_all(51001)
            agents[group] = WidthAgent(build_config(group))
        base = agents["S00"]
        for name in ("actor","critic1","critic2"):
            torch.testing.assert_close(getattr(agents['S10'],name).encoder.geo_embed.weight[:,2:],
                                       getattr(agents['S11'],name).encoder.geo_embed.weight[:,2:],rtol=0,atol=0)
            torch.testing.assert_close(getattr(agents['S01'],name).encoder.slot_embed.weight[:,20:],
                                       getattr(agents['S11'],name).encoder.slot_embed.weight[:,20:],rtol=0,atol=0)
        for group, delta in (("S10",128),("S01",2560),("S11",2688)):
            agent = agents[group]
            for name in ("actor", "critic1", "critic2"):
                old = getattr(base,name).state_dict()
                new = getattr(agent,name).state_dict()
                for key,value in old.items():
                    if key in ("encoder.geo_embed.weight", "encoder.slot_embed.weight"):
                        torch.testing.assert_close(value,new[key][:,:value.shape[1]],rtol=0,atol=0)
                    else:
                        torch.testing.assert_close(value,new[key],rtol=0,atol=0)
                self.assertEqual(agent.parameter_counts()[name]-base.parameter_counts()[name],delta)
            output = agent.actor(torch.ones(1,state_dimension(group)))
            self.assertEqual([tuple(x.shape) for x in output],[(1,100),(1,10),(1,10)])
            self.assertEqual(agent.actor.encoder.last_token_count,14)
            for key in ("encoder.geo_embed.weight","encoder.slot_embed.weight"):
                before = base.actor.state_dict()[key]
                after = agent.actor.state_dict()[key]
                extra = after[:,before.shape[1]:]
                if extra.numel():
                    self.assertTrue(bool(torch.all(extra != 0)))

    def test_s00_matches_original_forward_and_masked_update(self):
        config=build_config("S00","cpu","correctness")
        config["training"].update(batch_size=2,warmup_steps=2)
        seed_all(51001);original=WorkbenchAgent(config,207,(100,10,10),"cpu")
        seed_all(51001);adapted=WidthAgent(config)
        state=np.ones(207,np.float32)
        torch.testing.assert_close(original.actor(torch.from_numpy(state)[None])[0],
                                   adapted.actor(torch.from_numpy(state)[None])[0],rtol=0,atol=0)
        mask=dict(frequency_slots=np.ones((100,10),bool),power=np.ones(10,bool))
        for agent in (original,adapted):
            for index in range(2):agent.observe(state,(index,0,0),float(index),state,index==1,mask,None if index==1 else mask)
        seed_all(88);expected=original.update()
        seed_all(88);actual=adapted.update()
        self.assertEqual(expected,actual)
        for key,value in original.actor.state_dict().items():
            torch.testing.assert_close(value,adapted.actor.state_dict()[key],rtol=0,atol=0)

    def test_strict_checkpoint_schema_and_fixed_alpha(self):
        agent=WidthAgent(build_config("S00"))
        saved=agent.state_dict()
        agent.load_state_dict(saved)
        saved=copy.deepcopy(saved);saved.pop("w26_schema")
        with self.assertRaisesRegex(ValueError,"schema"):agent.load_state_dict(saved)
        with self.assertRaisesRegex(ValueError,"schema"):
            WidthAgent(build_config("S10")).load_state_dict(agent.state_dict())
        config=build_config("S00");config["training"]["gamma"]=1.
        with self.assertRaises(ValueError):WidthAgent(config)

    def test_engineering_has_separate_base_and_extension_initialization(self):
        science = build_config('S11','cpu','scientific')
        engineering = build_config('S11','cpu','engineering')
        self.assertEqual(science['training']['seed'],51001)
        self.assertEqual(engineering['training']['seed'],73026)
        self.assertEqual(engineering['training']['scene_order_seed'],73026)
        seed_all(science['training']['seed']);a=WidthAgent(science)
        seed_all(engineering['training']['seed']);b=WidthAgent(engineering)
        self.assertFalse(torch.equal(a.actor.encoder.geo_embed.weight[:,:2],b.actor.encoder.geo_embed.weight[:,:2]))
        self.assertFalse(torch.equal(a.actor.encoder.geo_embed.weight[:,2:],b.actor.encoder.geo_embed.weight[:,2:]))
        self.assertFalse(torch.equal(a.actor.encoder.slot_embed.weight[:,20:],b.actor.encoder.slot_embed.weight[:,20:]))


if __name__ == "__main__":unittest.main()
