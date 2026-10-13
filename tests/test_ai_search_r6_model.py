import copy
from types import SimpleNamespace
import unittest
import tempfile
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from satellite_rl.coverage.ai_search_r6.features import build_features,FeatureStats,collate_features,record_sha
from satellite_rl.coverage.ai_search_r6.model import initialize_model,model_tensor_sha
from satellite_rl.coverage.ai_search_r6.learning import CaseSampler,TeacherTarget,TrainingSession,compute_loss_sums,global_loss,match_targets
from satellite_rl.coverage.ai_search_r6.ai_adapter import build_ai_proposal,audit_ai_proposal,REQUIRED_PROVENANCE
from satellite_rl.coverage.ai_search_r6.model_qualification import _source_binding,_consume_budget,_counted_optimizer_step


class ModelTests(unittest.TestCase):
    def scene(self,n=3):
        theta=np.arange(n)*.001
        return SimpleNamespace(rays=np.column_stack((np.sin(theta),np.zeros(n),np.cos(theta))),demands=np.arange(n)*1e6,
            aircraft_ids=tuple(reversed([str(i) for i in range(n)])),input_sha256='input')

    def test_features_zero_neighbor_and_population_stats(self):
        one=build_features(self.scene(1));self.assertTrue((one.neighbors==-1).all());self.assertEqual(one.cost['NN_ray_pairs'],0)
        self.assertTrue((one.values[:,5:8]==0).all())
        c=build_features(self.scene());self.assertEqual(c.cost['NN_ray_pairs'],3)
        stats=FeatureStats.fit([one,c],role='train',qualified_table_sha='table')
        np.testing.assert_allclose(stats.mean,np.concatenate((one.values,c.values)).mean(0))
        self.assertEqual(FeatureStats.from_record(stats.to_record()).sha,stats.sha)
        with self.assertRaises(ValueError):FeatureStats.fit([c],role='development',qualified_table_sha='x')
        constant=FeatureStats.fit([one],role='train',qualified_table_sha='x');np.testing.assert_array_equal(constant.scale,np.ones(10))

    def test_block_neighbors_equal_original_pair_loop(self):
        for n in (1,19,257):
            scene=self.scene(n);actual=build_features(scene);rows=[[] for _ in range(n)]
            for i in range(n):
                angles=2*np.arcsin(np.minimum(np.linalg.norm(scene.rays[i+1:]-scene.rays[i],axis=1)/2,1))
                for j,angle in enumerate(angles,i+1):rows[i].append((float(angle),scene.aircraft_ids[j],j));rows[j].append((float(angle),scene.aircraft_ids[i],i))
            expected=np.full((n,16),-1,np.int64)
            angle_mean=np.zeros(n);angle_max=np.zeros(n);neighbor_share=np.zeros(n)
            total=scene.demands.sum();shares=scene.demands/total if total else np.zeros(n)
            for i,row in enumerate(rows):
                sorted_row=sorted(row)[:16];expected[i,:len(sorted_row)]=[v[2] for v in sorted_row]
                if sorted_row:
                    angle_mean[i]=np.mean([v[0] for v in sorted_row])/np.deg2rad(1)
                    angle_max[i]=max(v[0] for v in sorted_row)/np.deg2rad(1)
                    neighbor_share[i]=shares[expected[i,:len(sorted_row)]].sum()
            expected_values=np.column_stack(((scene.rays-scene.rays.mean(0))/np.deg2rad(1),shares,
                np.log1p(scene.demands/1e6),angle_mean,angle_max,neighbor_share,np.full(n,shares.max()),np.full(n,np.log1p(n))))
            np.testing.assert_array_equal(actual.values,expected_values)
            np.testing.assert_array_equal(actual.neighbors,expected);self.assertEqual(actual.cost['NN_ray_pairs'],n*(n-1)//2)

    def test_model_padding_and_initialization(self):
        torch.set_num_threads(1)
        net=initialize_model();self.assertEqual(model_tensor_sha(net),model_tensor_sha(initialize_model()))
        a,b=build_features(self.scene(1)),build_features(self.scene(3));stats=FeatureStats.fit([a,b],role='train',qualified_table_sha='x')
        with torch.no_grad():
            single=net(**collate_features([a],stats));batched=net(**collate_features([a,b],stats))
        torch.testing.assert_close(single['enable'][0],batched['enable'][0],atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.isneginf(batched['pointer'][0,:,1:]).all())

    def test_global_loss_reference_and_real_aircraft_only(self):
        self.assertAlmostEqual(float(global_loss(dict(E=torch.tensor(640.),O=torch.tensor(32.),P=torch.tensor(16.)),64,32)),1.2,places=6)
        target=TeacherTarget(('z','a'),np.array([0,1]),('z','a'),('s0','s1'))
        out=dict(enable=torch.zeros((1,200)),owner=torch.zeros((1,3,200)),pointer=torch.zeros((1,200,3)),valid=torch.tensor([[1,1,0]],dtype=torch.bool))
        matches=[dict(mapping={0:4,1:9})];sums=compute_loss_sums(out,[target],matches)
        self.assertAlmostEqual(float(sums['O']),2*np.log(200),places=5)
        self.assertAlmostEqual(float(sums['P']),2*np.log(2),places=5)

    def test_sampler_cross_epoch_pending_resume(self):
        table=[dict(case_index=i,case_id=str(i)) for i in range(19)]
        a=CaseSampler(table);a.assemble();a.commit();pending=a.assemble()
        b=CaseSampler(table);b.load_state_dict(a.state_dict());self.assertEqual(b.assemble(),pending)
        a.commit();b.commit();self.assertEqual(a.assemble(),b.assemble())
        for field,value in [('pending',[-1]*16),('cursor',0),('epoch',9)]:
            bad=copy.deepcopy(a.state_dict());bad[field]=value
            with self.assertRaises(ValueError):b.load_state_dict(bad)

    def test_batch_bindings_and_teacher_identity(self):
        table=[dict(case_index=0,case_id='case',input_sha='input',label_ref_sha='label')]
        sampler=CaseSampler(table);sampler.assemble();session=object.__new__(TrainingSession);session.sampler=sampler
        identity=dict(table[0],role='train');case=build_features(self.scene(1),identity=identity)
        target=TeacherTarget(case.aircraft_ids,np.array([0]),case.aircraft_ids,('s',),identity,'label')
        session.validate_batch([case]*16,[target]*16)
        for key,value in [('case_id','other'),('role','validation'),('input_sha','other')]:
            changed=copy.deepcopy(case);changed.identity[key]=value
            with self.assertRaises(ValueError):session.validate_batch([changed]+[case]*15,[target]*16)
        wrong=copy.deepcopy(target);wrong.label_ref_sha='other'
        with self.assertRaises(ValueError):session.validate_batch([case]*16,[wrong]*16)

    def test_matching_one_call_and_sorted_teacher_columns(self):
        target=TeacherTarget(('z','a'),np.array([0,1]),('z','a'),('s0','s1'))
        out=dict(enable=torch.zeros((1,200)),owner=torch.zeros((1,2,200)),pointer=torch.zeros((1,200,2)),valid=torch.ones((1,2),dtype=torch.bool))
        def solve(cost):
            self.assertEqual(cost.dtype,np.float64);self.assertTrue(cost.flags.c_contiguous)
            self.assertEqual(cost.shape,(200,2));self.assertTrue(np.isfinite(cost).all())
            return np.array([4,9]),np.array([0,1])
        with patch('satellite_rl.coverage.ai_search_r6.learning.solver_identity',return_value={'mock':True}),patch('satellite_rl.coverage.ai_search_r6.learning.linear_sum_assignment',side_effect=solve) as mocked:
            match=match_targets(out,[target])[0]
        self.assertEqual(mocked.call_count,1);self.assertEqual(match['ordered_column_indices'],[1,0]);self.assertEqual(match['mapping'],{1:4,0:9})
        with self.assertRaises(ValueError):compute_loss_sums(out,[],[])
        bad=copy.deepcopy(out);bad['valid']=torch.tensor([[0,1]],dtype=torch.bool)
        with self.assertRaises(ValueError):compute_loss_sums(bad,[target],[match])

    def test_matching_member_mean_independent_formula(self):
        target=TeacherTarget(('a','b','c'),np.array([0,0,1]),('a','c'),('s0','s1'))
        enable=torch.linspace(-2,2,200)[None,:];owner=torch.arange(600,dtype=torch.float32).reshape(1,3,200)/100
        pointer=torch.arange(600,dtype=torch.float32).reshape(1,200,3)/200
        out=dict(enable=enable,owner=owner,pointer=pointer,valid=torch.ones((1,3),dtype=torch.bool))
        captured=[]
        def solve(matrix):captured.append(matrix.copy());return np.array([0,1]),np.array([0,1])
        with patch('satellite_rl.coverage.ai_search_r6.learning.solver_identity',return_value={'mock':True}),patch('satellite_rl.coverage.ai_search_r6.learning.linear_sum_assignment',side_effect=solve):match_targets(out,[target])
        owner_np=owner[0].numpy().astype(np.float64);pointer_np=pointer[0].numpy().astype(np.float64)
        owner_neg=np.log(np.exp(owner_np).sum(1))[:,None]-owner_np
        pointer_neg=np.log(np.exp(pointer_np).sum(1))[:,None]-pointer_np
        expected=np.logaddexp(0,-enable[0].numpy().astype(np.float64))+owner_neg[:2].mean(0)+pointer_neg[:,0]
        np.testing.assert_allclose(captured[0][:,0],expected,atol=1e-12,rtol=1e-12)

    def test_teacher_owner_integer_dtype_required(self):
        for owners in (np.array([.5,1]),np.array([False,True]),np.array([np.nan,1]),np.array([0.,1.])):
            target=TeacherTarget(('a','b'),owners,('a','b'),('s0','s1'))
            with self.assertRaises(ValueError):target.ordered()

    def test_engineering_budget_once_and_step_write_ahead_pure_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=_source_binding('frozen_commit');output=Path(tmp)/'output'
            budget=dict(source_commit='frozen_commit',source_files=source['source_files'],source_manifest_sha=record_sha(source),
                output_path=str(output),global_claim_path=str(Path(tmp)/'global_budget_claim.json'),reservation_id='once')
            _consume_budget(budget,'spec',output)
            with self.assertRaises(FileExistsError):_consume_budget(budget,'spec',output)
            other=copy.deepcopy(budget);other['output_path']=str(Path(tmp)/'other')
            with self.assertRaises(FileExistsError):_consume_budget(other,'spec',Path(tmp)/'other')
            altered=copy.deepcopy(budget);altered['source_files']['model.py']='bad'
            with self.assertRaises(ValueError):_consume_budget(altered,'spec',output)
        receipt=dict(updates=[]);snapshots=[]
        def persist():snapshots.append(copy.deepcopy(receipt))
        def failure(opt):
            self.assertEqual(snapshots[-1]['updates'][-1]['status'],'started_unknown_consumed')
            raise RuntimeError('mock failure before/within fake operation')
        with self.assertRaises(RuntimeError):_counted_optimizer_step(failure,None,(),{},receipt,persist)
        self.assertEqual(receipt['updates'][0]['status'],'unknown_consumed')
        for _ in range(3):_counted_optimizer_step(lambda opt:None,None,(),{},receipt,persist)
        with self.assertRaises(RuntimeError):_counted_optimizer_step(lambda opt:None,None,(),{},receipt,persist)
        self.assertEqual(len(receipt['updates']),4)

    def test_clock_explicit_supply_start_checkpoint_IO_and_recovery_evidence(self):
        c=build_features(self.scene(1));stats=FeatureStats.fit([c],role='train',qualified_table_sha='table')
        table=[dict(case_index=0,case_id='c',input_sha='input',label_ref_sha='label')]
        optimizer=SimpleNamespace(state_dict=lambda:{},load_state_dict=lambda state:None)
        with tempfile.TemporaryDirectory() as tmp,patch('satellite_rl.coverage.ai_search_r6.learning.solver_identity',return_value={}),patch('torch.optim.AdamW',return_value=optimizer),patch('torch.cuda.is_available',return_value=False):
            session=TrainingSession(initialize_model(),stats,CaseSampler(table),{},'cpu')
            with self.assertRaises(ValueError):session.start_activity()
            session.start_activity(clock_path=Path(tmp)/'clock.json');session.sampler.assemble()
            checkpoint=Path(tmp)/'model.pt';receipt=session.checkpoint(checkpoint)
            clock=json.loads((Path(tmp)/'clock.json').read_text())
            self.assertEqual(clock['events'][-1]['event'],'checkpoint_IO_completed')
            self.assertGreaterEqual(clock['events'][-1]['charged_active_seconds'],clock['events'][-1]['checkpoint_state_active_seconds'])
            with self.assertRaises(TypeError):session.restore(checkpoint)
            evidence=dict(verified=True,checkpoint_file_sha=receipt['checkpoint_file_sha'],unknown_tail_accounted=True,
                interval_evidence='synthetic controlled clean interval',charged_active_seconds=receipt['charged_active_seconds'],
                last_completed_checkpoint_IO_active_seconds=receipt['charged_active_seconds'])
            path=Path(tmp)/'evidence.json';path.write_text(json.dumps(evidence));binding=dict(evidence_path=str(path),evidence_sha=hashlib.sha256(path.read_bytes()).hexdigest(),clock_path=str(Path(tmp)/'restored_clock.json'))
            session.restore(checkpoint,recovery_clock=binding);self.assertEqual(session.active_seconds,evidence['charged_active_seconds'])
            evidence['unknown_tail_accounted']=False;path.write_text(json.dumps(evidence));binding['evidence_sha']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):session.restore(checkpoint,recovery_clock=binding)

    def test_ai_outer_exact_inner_no_teacher(self):
        scene=self.scene();p=SimpleNamespace(scene=scene,identity=dict(identity_sha256='role',role='development'),source_sha256='prepared',Ncap=2)
        output=dict(enable=np.full(200,-1,np.float32),pointer=np.zeros((200,3),np.float32),owner=np.arange(600,dtype=np.float32).reshape(3,200))
        output['enable'][8]=2;output['enable'][4]=1;output['enable'][10]=3
        env=build_ai_proposal(p,output,{k:'source' for k in REQUIRED_PROVENANCE},inference_cost=dict(seconds=1,model_forward_calls=1))
        self.assertEqual(env['selected_query_mapping'],[8,10]);self.assertEqual(env['actual_method'],'AI')
        self.assertEqual(env['inner']['provenance']['source_method'],'EXACT');self.assertEqual(env['inner']['anchor_aircraft_ids'],['0','0'])
        changed=copy.deepcopy(env);changed['inner']['scores_n_by_K'][0][0]+=1
        with self.assertRaises(ValueError):audit_ai_proposal(p,changed)

    def test_adapter_alloff_ties_external_asset_and_rehashed_tampering(self):
        scene=self.scene();p=SimpleNamespace(scene=scene,identity=dict(identity_sha256='role',role='development'),source_sha256='prepared',Ncap=2)
        output=dict(enable=np.full(200,-1,np.float32),pointer=np.zeros((200,3),np.float32),owner=np.zeros((3,200),np.float32))
        provenance={k:'source' for k in REQUIRED_PROVENANCE}
        with tempfile.TemporaryDirectory() as tmp:
            env=build_ai_proposal(p,output,provenance,inference_cost=dict(model_forward_calls=1),tensor_asset_path=tmp+'/arrays.npz')
            self.assertEqual(env['selected_query_mapping'],[0]);self.assertNotIn('values',env['valid_aircraft_pointer_owner_tensor_refs_and_SHAs'])
            audit_ai_proposal(p,env)
            output['enable'][:4]=1;cap=build_ai_proposal(p,output,provenance,inference_cost=dict(model_forward_calls=1))
            self.assertEqual(cap['selected_query_mapping'],[0,1])
            for field in ('case_identity','selected_query_mapping'):
                changed=copy.deepcopy(cap)
                if field=='case_identity':changed[field]['role']='test'
                else:changed[field]=[2,3]
                changed.pop('outer_JSON_SHA');changed['outer_JSON_SHA']=record_sha(changed)
                with self.assertRaises(ValueError):audit_ai_proposal(p,changed)
            missing=copy.deepcopy(provenance);missing.pop('checkpoint_file_SHA')
            with self.assertRaises(ValueError):build_ai_proposal(p,output,missing,inference_cost=dict(model_forward_calls=1))


if __name__=='__main__':unittest.main()
