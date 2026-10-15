"""All tests use synthetic probabilities and a fake integer environment: 0 PHY."""
import tempfile
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

if __name__=='__main__':unittest.main()
