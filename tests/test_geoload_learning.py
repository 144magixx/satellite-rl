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

    def test_real_train_resume_and_nonfinite_guard(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from satellite_rl.coverage.geoload.learning import train
        from satellite_rl.coverage.geoload.evidence import write_sample
        _,_,_,sample=self.fixture();sample.pop('centers',None)
        class Stop(Exception):pass
        def stop_at(n):
            def progress(update,value):
                if update==n:raise Stop()
            return progress
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);record=write_sample(root/'sample.npz',sample);record['mother_id']='m'
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(status='completed',samples=[record])))
            protocol=dict(network=dict(hidden=16,layers=1,heads=4),updates=10000,batch=32,microbatch=4)
            with self.assertRaises(Stop):train(manifest,root/'baseline',protocol,device='cpu',progress=stop_at(3))
            with self.assertRaises(Stop):train(manifest,root/'resume',protocol,device='cpu',progress=stop_at(2))
            with self.assertRaises(Stop):train(manifest,root/'resume',protocol,device='cpu',resume=True,progress=stop_at(3))
            baseline=torch.load(root/'baseline/checkpoint.pt',weights_only=False);recovered=torch.load(root/'resume/checkpoint.pt',weights_only=False)
            self.assertEqual(recovered['update'],3);self.assertEqual(recovered['ledger']['unknown_attempts'],0)
            for key,value in baseline['model'].items():self.assertTrue(torch.equal(value,recovered['model'][key]))
            with patch('satellite_rl.coverage.geoload.learning.loss',return_value=torch.tensor(float('nan'),requires_grad=True)):
                with self.assertRaises(FloatingPointError):train(manifest,root/'bad',protocol,device='cpu')
            good=torch.load(root/'bad/checkpoint.pt',weights_only=False);self.assertEqual(good['update'],0)
            self.assertEqual(json.loads((root/'bad/update_wal.json').read_text())['status'],'failed')
            original_step=torch.optim.Adam.step
            def bad_step(optimizer,*args,**kwargs):
                original_step(optimizer,*args,**kwargs)
                optimizer.param_groups[0]['params'][0].data.fill_(float('nan'))
            with patch.object(torch.optim.Adam,'step',bad_step):
                with self.assertRaises(FloatingPointError):train(manifest,root/'bad_parameter',protocol,device='cpu')
            self.assertEqual(torch.load(root/'bad_parameter/checkpoint.pt',weights_only=False)['update'],0)
            def bad_gradient_model(network):
                m=make_model(network);next(m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')));return m
            with patch('satellite_rl.coverage.geoload.learning.make_model',side_effect=bad_gradient_model):
                with self.assertRaises(FloatingPointError):train(manifest,root/'bad_gradient',protocol,device='cpu')
            self.assertEqual(torch.load(root/'bad_gradient/checkpoint.pt',weights_only=False)['update'],0)
            # Simulate uncertain crash before checkpoint commit; preserve truthful extra-attempt accounting.
            wal=root/'resume/update_wal.json';wal.write_text(json.dumps(dict(status='pending',update=4,attempt_id=1)))
            with self.assertRaises(Stop):train(manifest,root/'resume',protocol,device='cpu',resume=True,progress=stop_at(4))
            accounted=torch.load(root/'resume/checkpoint.pt',weights_only=False)
            self.assertEqual(accounted['ledger']['unknown_attempts'],1);self.assertEqual(accounted['ledger']['replayed_attempts'],1)

    def test_teacher_late_trial_has_replayable_evidence_and_no_commit(self):
        import tempfile
        from unittest.mock import patch
        from satellite_rl.coverage.geoload.teacher import teacher_case
        from satellite_rl.coverage.geoload.core import validate_layout,layout_sha256
        scene,initial,pool,_=self.fixture();clock=[0.];calls=[0]
        def validate(*args,**kwargs):
            result=validate_layout(*args,**kwargs);calls[0]+=1
            if calls[0]>1:clock[0]=.02
            return result
        case=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),initial=initial.to_record(),mother_id='m')
        with tempfile.TemporaryDirectory() as root,patch('satellite_rl.coverage.geoload.teacher.perf_counter',side_effect=lambda:clock[0]),patch('satellite_rl.coverage.geoload.teacher.validate_layout',side_effect=validate),patch('satellite_rl.coverage.geoload.teacher.candidates',return_value=iter(pool)):
            result=teacher_case(case,root,dict(seconds=.01,max_attempts=1))
        self.assertEqual(result['budget']['late_completed_candidates'],1)
        trial=result['journal'][0];self.assertTrue(trial['late']);self.assertFalse(trial['pool_committed'])
        self.assertEqual(layout_sha256(Layout.from_record(result['layout'])),layout_sha256(initial))
        rebuilt=Layout(initial.centers,initial.assignment.copy());a=rebuilt.assignment.copy();delta=trial['delta'];a[delta['users']]=delta['owners'];centers=rebuilt.centers.copy();centers[delta['beams']]=np.asarray(delta['centers']).reshape(-1,2)
        self.assertEqual(layout_sha256(Layout(centers,a)),trial['candidate_sha256'])

    def test_pool_index_first_seen_exact_equivalence(self):
        from unittest.mock import patch
        import time
        from satellite_rl.coverage.geoload import evidence
        timings=[]
        for n in (50,100):
            centers=np.column_stack((np.linspace(50,51,n),np.linspace(10,11,n)))
            centers[0,1]=-0.0
            scene=Scene(centers,np.arange(1,n+1,dtype=float));current=Layout(centers,np.arange(n))
            pool=[]
            for i in range(64):
                repeated=centers.copy();repeated[0,1]=0.0
                if i%8==0:repeated[-1,0]+=1e-6*i
                pool.append(('test',Layout(repeated,np.roll(np.arange(n),i)),{}))
            # Execute original list-scan implementation independently, preserving feature precision.
            import inspect
            source=inspect.getsource(evidence.pool_features)
            begin=source.index('    centers = [];');end=source.index('    initial = mapping(current)',begin)
            original=source[:begin]+"""    centers = []
    for layout in [current] + [entry[1] for entry in pool]:
        for center in layout.centers:
            if not any(np.array_equal(center, prior) for prior in centers): centers.append(center.copy())
    centers = np.asarray(centers)
    def mapping(layout):
        ids = [next(i for i,c in enumerate(centers) if np.array_equal(c,x)) for x in layout.centers]
        return np.asarray(ids, dtype=np.int64)[layout.assignment]
"""+source[end:]
            namespace=dict(np=np,legal_matrix=evidence.legal_matrix);exec(original,namespace)
            started=time.perf_counter();old,oldmap=namespace['pool_features'](scene,current,pool);oldseconds=time.perf_counter()-started
            started=time.perf_counter();new,newmap=pool_features(scene,current,pool);newseconds=time.perf_counter()-started
            for key in old:
                self.assertEqual(old[key].dtype,new[key].dtype);self.assertEqual(old[key].tobytes(),new[key].tobytes())
            for entry in pool:self.assertEqual(oldmap(entry[1]).tobytes(),newmap(entry[1]).tobytes())
            timings.append(dict(N=n,pool=64,old_seconds=oldseconds,new_seconds=newseconds))
        print('engineering_feature_preparation_only',timings)

    def test_common_pool_permutation(self):
        scene,initial,pool,s=self.fixture()
        for method in ('rule','random','MLM'):
            result=orderer(method,model=make_model({'hidden':16,'layers':1,'heads':4}))(scene,initial,pool)
            self.assertIs(result[0],pool[0])
if __name__=='__main__':unittest.main()

