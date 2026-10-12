import unittest
import numpy as np
import torch
from satellite_rl.coverage.geoload.core import Scene,Layout
from satellite_rl.coverage.geoload.evidence import pool_features
from satellite_rl.coverage.geoload.learning import collate,loss,orderer
from satellite_rl.coverage.geoload.models import make_model
class G16LearningTest(unittest.TestCase):
    def fixture(self):
        scene=Scene([[50,10],[50.01,10.01],[50.02,10.02]], [1,2,3])
        initial=Layout([[50,10],[50.02,10.02]],[0,0,1])
        other=Layout(initial.centers,[0,1,1]);pool=[('transfer',other,{})]
        features,_=pool_features(scene,initial,pool)
        return scene,initial,pool,dict(features,labels=np.array([0,1,1]))
    def test_features_and_real_update(self):
        scene,initial,pool,s=self.fixture();model=make_model({'hidden':16,'layers':1,'heads':4})
        batch=collate([s],np.random.default_rng(1),'cpu')
        # Visible labels refer only to initial assignment.
        visible=batch['visible'][0].numpy();np.testing.assert_array_equal(visible[visible>=0],s['initial'][visible>=0])
        before=[p.detach().clone() for p in model.parameters()]
        optimizer=torch.optim.Adam(model.parameters());value=loss(model,batch);value.backward();optimizer.step()
        self.assertTrue(torch.isfinite(value));self.assertTrue(any(not torch.equal(x,p) for x,p in zip(before,model.parameters())))
    def test_complete_update_resume_equivalence(self):
        import copy
        _,_,_,sample=self.fixture();torch.manual_seed(51001)
        model=make_model({'hidden':16,'layers':1,'heads':4});opt=torch.optim.Adam(model.parameters());rng=np.random.default_rng(51001)
        def update(m,o,r):
            o.zero_grad();v=loss(m,collate([sample],r,'cpu'));v.backward();o.step();return v.detach()
        update(model,opt,rng)
        state=copy.deepcopy((model.state_dict(),opt.state_dict(),rng.bit_generator.state,torch.get_rng_state()))
        expected=update(model,opt,rng);weights=copy.deepcopy(model.state_dict())
        recovered=make_model({'hidden':16,'layers':1,'heads':4});ropt=torch.optim.Adam(recovered.parameters())
        recovered.load_state_dict(state[0]);ropt.load_state_dict(state[1]);rrng=np.random.default_rng();rrng.bit_generator.state=state[2];torch.set_rng_state(state[3])
        actual=update(recovered,ropt,rrng);self.assertTrue(torch.equal(expected,actual))
        for key,value in recovered.state_dict().items():self.assertTrue(torch.equal(value,weights[key]))

    def test_common_pool_permutation(self):
        scene,initial,pool,s=self.fixture()
        for method in ('rule','random','MLM'):
            result=orderer(method,model=make_model({'hidden':16,'layers':1,'heads':4}))(scene,initial,pool)
            self.assertIs(result[0],pool[0])
if __name__=='__main__':unittest.main()

