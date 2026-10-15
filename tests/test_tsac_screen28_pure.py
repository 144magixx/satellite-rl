"""All tests use synthetic probabilities and a fake integer environment: 0 PHY."""
import tempfile
import os
import pickle
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from satellite_rl.tsac_screen28.proposals import draw,seed_record,uniform_probabilities
from satellite_rl.tsac_screen28.transaction import Journal,decision,snapshot,state_sha,UnknownCost,fork
from satellite_rl.tsac_screen28.worker import one_decision,candidate_log
from satellite_rl.tsac_screen28.plan import matrix,development_gate
from satellite_rl.tsac_screen28.episode import run_episode,publish_committed

class FakeEnv:
    calls=[]
    def __init__(self,n=1):
        self.N=n;self.beam_idx=0;self.starts=np.zeros(n,np.int64);self.counts=self.starts.copy()
        self.powers=np.zeros(n);self.occupied=np.zeros((8,100),bool);self.rates=np.zeros(n)
        self.requested=[];self.total_satisfaction=0.;self.schema={'fake':True};self.layout={'content_sha256':'fake'}
        self.kernel={'gain':np.ones((1,1))};self.kernel['gain'].setflags(write=False)
        self.fail=False
    def action_mask(self):
        fl=np.zeros((100,10),bool);fl[0,0]=True;fl[1,0]=True
        p=np.zeros(10,bool);p[:2]=True
        return dict(frequency_slots=fl,power=p)
    def _observation(self):return np.array([self.beam_idx],np.float32)
    def metrics(self):return dict(N=self.N,processed=self.beam_idx,complete=self.beam_idx==self.N)
    def beam_results(self):return [dict(beam_id=str(i),rate_bps=float(self.rates[i])) for i in range(self.N)]
    def step(self,action):
        FakeEnv.calls.append(tuple(action))
        if self.fail:raise RuntimeError('synthetic failure')
        index=self.beam_idx;self.starts[index]=action[0];self.counts[index]=action[1]+1
        self.powers[index]=(action[2]+1)*5;self.rates[index]=action[0]+1
        self.requested.append(dict(action=list(action)));self.total_satisfaction+=1
        self.beam_idx+=1
        info=dict(beam_service_step={'beam_id':str(index),'rate_bps':float(self.rates[index])},satisfaction_increment=1.,timing={'environment_step_seconds':123.})
        return np.array([self.beam_idx],np.float32),float(action[0]),self.beam_idx==self.N,info

class PureTests(unittest.TestCase):
    def setUp(self):FakeEnv.calls=[]
    def journal(self,folder):return Journal(Path(folder)/'current.pkl',{'source':'fake','run':'same'},dict(query=8,commit=2))
    def test_seed_first_proposal_is_coupled_but_indices_are_separate(self):
        env=FakeEnv();mask=env.action_mask();fl,p=uniform_probabilities(mask)
        first=draw(fl,p,mask,'mother','P',0,0,0)
        self.assertEqual(first,draw(fl,p,mask,'mother','P',0,0,0))
        self.assertNotEqual(seed_record('mother','P',0,0,0)['seed'],seed_record('mother','P',0,0,1)['seed'])
        self.assertAlmostEqual(first['probability'],.25)
        self.assertEqual(len(FakeEnv.calls),0)
    def test_dedup_no_refill_earliest_tie_and_one_real_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();journal=self.journal(temp)
            proposals=[dict(action=a) for a in ([0,0,0],[1,0,0],[1,0,0],[1,0,1])]
            result=decision(env,proposals,journal,{'step':0})
            self.assertEqual(journal.value['transaction']['selected'],1)
            self.assertEqual(FakeEnv.calls,[(0,0,0),(1,0,0),(1,0,1),(1,0,0)])
            self.assertEqual(journal.value['attempts']['query'],3)
            self.assertEqual(journal.value['attempts']['commit'],1)
            self.assertEqual(env.beam_idx,1)
            self.assertEqual(len(env.requested),1)
    def test_query_failure_preserves_original_and_blocks_free_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();env.fail=True;before=state_sha(snapshot(env));j=self.journal(temp)
            proposals=[dict(action=[0,0,0]),dict(action=[1,0,0])]
            with self.assertRaises(RuntimeError):decision(env,proposals,j,{'step':0})
            self.assertEqual(before,state_sha(snapshot(env)))
            reloaded=self.journal(temp)
            with self.assertRaises(UnknownCost):decision(env,proposals,reloaded,{'step':0})
            self.assertEqual(len(FakeEnv.calls),1)
            self.assertEqual(reloaded.value['attempts']['query'],1)
            self.assertEqual(reloaded.value['completed']['query'],0)
    def test_safe_pause_resumes_cached_queries_without_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();j=self.journal(temp);proposals=[dict(action=[0,0,0]),dict(action=[1,0,0])]
            self.assertIsNone(decision(env,proposals,j,{'step':0},lambda:len(FakeEnv.calls)==1))
            self.assertEqual(env.beam_idx,0)
            decision(env,proposals,self.journal(temp),{'step':0})
            self.assertEqual(FakeEnv.calls,[(0,0,0),(1,0,0),(1,0,0)])
    def test_fork_readonly_kernel_and_no_state_replay(self):
        env=FakeEnv(200);child=fork(env,snapshot(env))
        with self.assertRaises(ValueError):child.kernel['gain'][0,0]=2
        child.starts[0]=3
        self.assertEqual(env.starts[0],0)
        self.assertTrue(np.shares_memory(child.kernel['gain'],env.kernel['gain']))
        self.assertEqual(FakeEnv.calls,[])
    def test_p1_has_one_forward_no_queries_and_no_repeat_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();j=self.journal(temp)
            def fake_probs(obs,mask):return uniform_probabilities(mask)
            with patch('satellite_rl.tsac_screen28.worker.uniform_probabilities',wraps=uniform_probabilities):
                one_decision(env,np.zeros(1),'P1','case',0,j,fake_probs,lambda:False)
                one_decision(env,np.zeros(1),'P1','case',0,self.journal(temp),fake_probs,lambda:False)
            self.assertEqual(len(FakeEnv.calls),1)
            self.assertEqual(j.value['attempts']['forward'],1)
            self.assertEqual(j.value['attempts']['query'],0)
            self.assertEqual(candidate_log(j)['selected'],0)
    def test_metadata_only_exact_budgets_and_gate(self):
        dev=matrix([dict(case_id=str(i),N=47+i) for i in range(20)],'development')
        self.assertEqual(dev['episodes'],240)
        self.assertEqual(dev['commits'],12*sum(range(47,67)))
        self.assertEqual(dev['queries'],24*sum(range(47,67)))
        known=matrix([dict(case_id=str(i),N=42+i) for i in range(40)],'known')
        self.assertEqual(known['episodes'],560)
        self.assertEqual(known['jobs'][12*14]['method'],'fixed')
        self.assertEqual(known['jobs'][13*14]['method'],'greedy')
        means=dict(P1=dict(SI=.9,SGM=.5,power_w=100),P4=dict(SI=.904,SGM=.5,power_w=105),R4=dict(SI=.9,SGM=.5,power_w=100))
        self.assertTrue(development_gate(means,True)['passed'])
        self.assertFalse(development_gate(means,False)['passed'])
    def test_injected_episode_resume_has_no_reset_replay_or_duplicate_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);identity=dict(case_id='case',method='R4',repeat=0,N=3)
            env=FakeEnv(3);journal=Journal(path/'current.pkl',identity,dict(query=12,commit=3))
            args=dict(method='R4',case_id='case',repeat=0,journal=journal,policy_probabilities=None,
                evidence_path=path/'steps.jsonl',admission=dict(root_approved=True,qualification_passed=True))
            first=run_episode(env,np.zeros(1),stop_requested=lambda:env.beam_idx==1,**args)
            self.assertEqual(first['stage'],'safe_stopped')
            calls=len(FakeEnv.calls)
            restored=FakeEnv(3);args['journal']=Journal(path/'current.pkl',identity,dict(query=12,commit=3))
            done=run_episode(restored,np.zeros(1),stop_requested=lambda:False,**args)
            self.assertEqual(done['processed'],3)
            self.assertEqual(len((path/'steps.jsonl').read_text().splitlines()),3)
            self.assertGreater(len(FakeEnv.calls),calls)
            self.assertEqual(done['costs']['commit'],3)
            final_calls=len(FakeEnv.calls)
            run_episode(FakeEnv(3),np.zeros(1),stop_requested=lambda:False,**args)
            self.assertEqual(len(FakeEnv.calls),final_calls)
    def test_partial_evidence_suffix_preserved_and_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);env=FakeEnv();j=self.journal(temp)
            decision(env,[dict(action=[0,0,0])],j,{'step':0})
            evidence=path/'steps.jsonl';evidence.write_bytes(b'{partial')
            with self.assertRaises(UnknownCost):publish_committed(j,evidence)
            self.assertEqual(evidence.read_bytes(),b'{partial')
            self.assertEqual(j.value['attempts']['commit'],1)
    def test_forward_failure_is_durable_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();j=self.journal(temp)
            def fail(obs,mask):raise RuntimeError('fake forward failure')
            with self.assertRaises(RuntimeError):one_decision(env,np.zeros(1),'P4','case',0,j,fail,lambda:False)
            with self.assertRaises(UnknownCost):one_decision(env,np.zeros(1),'P4','case',0,self.journal(temp),fail,lambda:False)
            self.assertEqual(j.value['attempts']['forward'],1)
            self.assertEqual(FakeEnv.calls,[])
    def test_historical_current_to_previous_gap_never_recreates_zero_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();j=self.journal(temp)
            decision(env,[dict(action=[0,0,0])],j,{'step':0})
            committed_calls=len(FakeEnv.calls)
            # Materialize the actual old two-rename crash window on disk after
            # a committed fake step; do not mock exists() or constructor logic.
            current=Path(temp)/'current.pkl';previous=current.with_suffix('.previous')
            os.replace(current,previous)
            previous_bytes=previous.read_bytes()
            current.with_suffix('.tmp').write_bytes(pickle.dumps(j.value,protocol=5))
            with self.assertRaises(UnknownCost):self.journal(temp)
            self.assertFalse(current.exists());self.assertEqual(previous.read_bytes(),previous_bytes)
            self.assertEqual(len(FakeEnv.calls),committed_calls)
    def test_save_failure_before_current_replace_keeps_nonzero_trusted_current(self):
        with tempfile.TemporaryDirectory() as temp:
            env=FakeEnv();j=self.journal(temp);decision(env,[dict(action=[0,0,0])],j,{'step':0})
            current=Path(temp)/'current.pkl';old=current.read_bytes();original_replace=os.replace
            def crash_at_current(source,target):
                if Path(target)==current:raise OSError('simulated interruption before final atomic replace')
                return original_replace(source,target)
            with patch('satellite_rl.tsac_screen28.transaction.os.replace',side_effect=crash_at_current):
                with self.assertRaises(OSError):j.save()
            self.assertEqual(current.read_bytes(),old)
            with self.assertRaises(UnknownCost):self.journal(temp)
            self.assertEqual(pickle.loads(current.read_bytes())['attempts']['commit'],1)
            self.assertEqual(len(FakeEnv.calls),1)
    def test_full_matrix_injected_loader_and_cursor_finishes_once(self):
        from satellite_rl.tsac_screen28.runtime import run_matrix
        from satellite_rl.t0_width.data import file_sha
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);manifest=path/'manifest.json';manifest.write_text(json.dumps(dict(rows=[dict(case_id=str(i)) for i in range(20)])))
            jobs=[dict(case_id='0',mother=0,method='R4',repeat=0,N=2),dict(case_id='0',mother=0,method='R1',repeat=0,N=2)]
            plan=dict(runtime_approved=True,phase='development',source_commit='fake',policy_ref={'sha256':'fake'},
                data_manifest_ref=dict(path=str(manifest),sha256=file_sha(manifest)),wall_budget_seconds=100,
                matrix=dict(jobs=jobs,queries=8,commits=4,model_forwards=0))
            def fake_loader(plan,row,n):return FakeEnv(n),dict(fake_loader=True,kernel_preparation_calls=0)
            def forbidden_policy(plan):raise AssertionError('R methods must not load a model')
            first=run_matrix(plan,path/'run',lambda:False,policy_loader=forbidden_policy,environment_loader=fake_loader)
            self.assertEqual(first['stage'],'completed');self.assertEqual(first['costs']['commit'],4)
            calls=len(FakeEnv.calls)
            again=run_matrix(plan,path/'run',lambda:False,policy_loader=forbidden_policy,environment_loader=fake_loader)
            self.assertEqual(again['stage'],'completed');self.assertEqual(len(FakeEnv.calls),calls)
            self.assertEqual(len(list((path/'run/episodes').glob('*/terminal.json'))),2)
            self.assertFalse(list((path/'run/episodes').glob('*/binding.pkl')))
            (path/'run/episodes/0000/episode_timing.json').unlink()
            missing=run_matrix(plan,path/'run',lambda:False,policy_loader=forbidden_policy,environment_loader=fake_loader)
            self.assertEqual(missing['timing_unknown'],[0]);self.assertEqual(len(FakeEnv.calls),calls)
            self.assertTrue((path/'run/episodes/0000/timing_gap.json').exists())
    def test_engineering_injected_checks_are_bounded_and_do_not_repeat(self):
        from satellite_rl.tsac_screen28.engineering import run_engineering
        from satellite_rl.t0_width.data import file_sha
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);manifest=path/'manifest.json';manifest.write_text(json.dumps(dict(rows=[dict(case_id='case')])))
            plan=dict(runtime_approved=True,run_id='engineering',data_manifest_ref=dict(path=str(manifest),sha256=file_sha(manifest)),
                limits=dict(query=256,commit=64,forward=64,proposal=64),
                checks=[dict(check_index=i,case_id='case',mother=0,method='R4',N=3,step=i) for i in range(3)])
            def fake_loader(plan,row,n):return FakeEnv(n),dict(fake_loader=True)
            result=run_engineering(plan,path/'run',lambda:False,environment_loader=fake_loader)
            self.assertEqual(result['real_commit_attempts'],3);self.assertFalse(result['passed'])
            calls=len(FakeEnv.calls)
            run_engineering(plan,path/'run',lambda:False,environment_loader=fake_loader)
            self.assertEqual(len(FakeEnv.calls),calls)
    def test_backend_draft_rejects_before_any_child_or_resource_probe(self):
        from satellite_rl.tsac_screen28.backend import Workbench
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json'
            plan.write_text(json.dumps(dict(schema='W28WorkbenchPlanV1',experiment_id='S2-W28',max_workers=1,
                executor_port=20418,runtime_approved=False,tasks={})))
            class FakeGate:
                spec={'experiment_id':'S2-W28'}
            with patch('satellite_rl.tsac_screen28.backend.Gate',return_value=FakeGate()),patch('satellite_rl.tsac_screen28.backend.subprocess.Popen') as spawn,patch('satellite_rl.tsac_screen28.backend.subprocess.check_output') as probe:
                backend=Workbench(root/'run',plan,root/'control')
                with self.assertRaises(ValueError):backend.start('science')
                spawn.assert_not_called();probe.assert_not_called()
    def test_concrete_engineering_catalog_and_prefix_are_metadata_only(self):
        from satellite_rl.tsac_screen28.checks import catalog,fixture_manifest,validate_catalog,prefix_recipe
        bounds=validate_catalog(catalog(),fixture_manifest())
        self.assertEqual(len(catalog()),16)
        self.assertEqual(bounds['worst_case'],dict(commit=12,query=35,forward=8,kernel_preparation=16,restore_PHY=0))
        self.assertEqual(bounds['capacity_reward_calls'],47)
        large=next(s for s in fixture_manifest()['rows'] if s['N']==200)
        self.assertAlmostEqual(sum(w*w for w in large['widths']),19.23)
        prefix=prefix_recipe();self.assertEqual(prefix['total_power_w'],2960)
        occupied=set()
        for action in prefix['actions']:
            slots={(action['group'],slot) for slot in range(action['start'],action['start']+action['count'])}
            self.assertFalse(occupied & slots);occupied.update(slots)
        self.assertEqual(len([x for x in occupied if x[0]==4]),96)
        self.assertEqual(FakeEnv.calls,[])
    def test_empty_negative_nan_receipts_cannot_be_qualified_by_boolean(self):
        from satellite_rl.tsac_screen28.receipt import validate_qualification
        for invalid in (-1,float('nan'),0):
            receipt=dict(root_approved=True,passed=True,physical_calls_attempted=invalid,
                real_commit_attempts=0,candidate_restore_attempts=0,model_forward_attempts=0)
            with self.assertRaises(ValueError):validate_qualification(receipt,{})
    def test_value_only_binding_pack_removes_authority_mappingproxy_without_env_calls(self):
        from satellite_rl.tsac_screen28.binding import pack_environment
        from types import SimpleNamespace,MappingProxyType
        env=FakeEnv();authority=SimpleNamespace(case_by_id=MappingProxyType({'id':'sha'}),layout_shas=frozenset({'layout'}),allocation_shas=frozenset(),manifest_sha='manifest')
        env.stage_input=SimpleNamespace(authority=authority,case={'case_id':'id'},layout=env.layout,physics={})
        binding=pack_environment(env);raw=pickle.dumps(binding,protocol=5)
        self.assertTrue(raw);self.assertEqual(binding['authority_values']['case_by_id'],{'id':'sha'})
        self.assertEqual(FakeEnv.calls,[])
    def test_preparation_transport_scope_and_remote_original_refs_are_isolated(self):
        from scripts.tsac_screen28 import prepare_remote as helper
        old={20418:dict(user='user',python='/pinned/python',base='/old/s2_w27',api=9070)}
        copied=helper.isolated_node_configuration(old)
        self.assertEqual(old[20418]['base'],'/old/s2_w27');self.assertEqual(copied[20418]['base'],helper.REMOTE)
        self.assertEqual(copied[20418]['api'],9080)
        self.assertEqual(copied[20418]['user'],old[20418]['user']);self.assertEqual(copied[20418]['python'],old[20418]['python'])
        authority={key:dict(path='D:/canonical/'+key+'.json',sha256='a'*64) for key in ('w27_closure_ref','explicit_continue_ref','gate_contract_ref')}
        def fake_publish(port,path,destination):
            self.assertEqual(port,20418);self.assertTrue(destination.startswith(copied[20418]['base']+'/'))
            return dict(path=destination,sha256='a'*64)
        with patch.object(helper,'read_json_ref',return_value={}),patch.object(helper.transport,'publish',side_effect=fake_publish) as publish:
            refs=helper.publish_originals(authority)
        self.assertEqual(publish.call_count,3)
        self.assertTrue(all(ref['path'].startswith(helper.REMOTE+'/admissions/originals/') for ref in refs.values()))
        self.assertIn("issued_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()",helper.INSTALL)
        self.assertIn("user_continue_event_UTC='2026-10-15T00:57:33+00:00'",helper.INSTALL)
    def test_real_schema_physics_tree_representation_roundtrip_without_real_env(self):
        import copy
        from satellite_rl.t0_width.schema import observation_schema
        from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
        from satellite_rl.tsac_screen28.transaction import restore_memory
        env=FakeEnv();env.schema=observation_schema('S10');env.physics=copy.deepcopy(PHYSICS)
        before=snapshot(env);child=fork(env,before)
        self.assertIsInstance(env.physics['geometry']['latitude_range_deg'],list)
        self.assertIsInstance(child.physics['geometry']['latitude_range_deg'],tuple)
        self.assertEqual(state_sha(snapshot(child)),state_sha(before))
        # Actual nested FrozenDict copy is exercised; no T0WidthEnv constructor
        # or shared link/PHY function is called by this integer fixture.
        result=child.step([0,0,0]);self.assertEqual(env.beam_idx,0)
        self.assertEqual(snapshot(child)['values']['beam_idx'],1)
        invalid=copy.deepcopy(before);invalid['physics']['power_budget_w']=2999.
        with self.assertRaises(ValueError):restore_memory(env,invalid)
        invalid=copy.deepcopy(before);invalid['schema']['state_width']+=1
        with self.assertRaises(ValueError):restore_memory(env,invalid)
        invalid=copy.deepcopy(before);invalid['layout_SHA']='different-layout'
        with self.assertRaises(ValueError):restore_memory(env,invalid)

if __name__=='__main__':unittest.main()
