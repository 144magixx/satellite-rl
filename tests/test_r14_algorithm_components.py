"""R14 pure components only. Never calls formal fit/baseline or real backend."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import unittest
from satellite_rl.coverage.ai_search_r14 import features as f,policies as p,tree as t,stop_adapter as s

def fake_policy(kind):
    return dict(schema='R14FrozenCausalPolicyV1',policy_type=kind,actions=list(p.ACTIONS),feature_names=list(f.FEATURE_NAMES),calibration_applied=False,
        root=dict(kind='split',id='root',feature=0,threshold=.5,left=dict(kind='leaf',id='rootL',action='cap4'),right=dict(kind='leaf',id='rootR',action='window4')))

class FeaturesAndRules(unittest.TestCase):
    def test_exact_causal_feature_order_and_nearest_rank(self):
        ids=['p'+str(i) for i in range(6)];layout=dict(beam_ids=['b0','b1','b2','b3'],assignment=[0,0,0,1,2,3],widths_deg=[.3,.4,.5,.6])
        x=f.extract_features(ids,layout,Ncap=6,Omega_cap=1.)
        self.assertEqual(len(x),12);self.assertEqual(x[:3],[6,4,4/6]);self.assertEqual(x[4:6],[3,3.])
        self.assertAlmostEqual(x[6],math.sqrt(.75)/1.5);self.assertEqual(x[7:9],[.25,1]);self.assertAlmostEqual(x[9],.45);self.assertEqual(x[10:],[.6,.6])
        self.assertEqual(f.nearest_rank([1,2,3,4],.5),2);self.assertEqual(f.nearest_rank([1,2,3,4],.9),4)
    def test_no_empty_or_nonfinite_imputation(self):
        with self.assertRaises(ValueError):f.validate_vector([0]*11+[float('nan')])
        with self.assertRaises(ValueError):f.extract_features(['a'],dict(beam_ids=['b'],assignment=[0],widths_deg=[.3]),Ncap=1,Omega_cap=0)
        with self.assertRaises(ValueError):f.extract_features(['a','b'],dict(beam_ids=['x','y'],assignment=[0,0],widths_deg=[.3,.3]),Ncap=2,Omega_cap=1)
    def test_fixed_rule_window_resets_and_skips_not_counted(self):
        rule=p.StopRule('window2');self.assertFalse(rule.after('rejected'));self.assertFalse(rule.after('admissible'));self.assertFalse(rule.after('no_improvement'));self.assertTrue(rule.after('rejected'))
        self.assertEqual(rule.completed,4)
        with self.assertRaises(ValueError):rule.after('rejected')
        self.assertEqual(p.stop_index('cap0',[]),0);self.assertEqual(p.stop_index('cap32',['rejected']*32),32)
        with self.assertRaises(ValueError):p.StopRule('window2').after('unknown')
    def test_strict_quality_and_all_resource_conjuncts(self):
        teacher=dict(maxload=100.,top10_mean=80.,CV=.5,N=2,actual_Omega=.01)
        self.assertTrue(p.quality_preserving(dict(teacher,maxload=100.+1e-6),teacher))
        for delta in (dict(maxload=100.1),dict(top10_mean=80.1),dict(CV=.51),dict(N=3),dict(actual_Omega=.01001)):
            self.assertFalse(p.quality_preserving(dict(teacher,**delta),teacher))
        with self.assertRaises(ValueError):p.quality_preserving(dict(teacher,maxload=None),teacher)

class LearningComponents(unittest.TestCase):
    def tables(self,n=32):
        X=[[i]+[0]*11 for i in range(n)];safe=[];paid=[]
        for i in range(n):
            row=[False]*8;row[0]=i<n/2;row[1]=i>=n/2;row[4]=True;safe.append(row);paid.append([0,4,8,16,32,2,4,8])
        return X,safe,paid
    def test_leaf_all_mothers_safe_and_fixed_action_ties(self):
        X,safe,paid=self.tables();self.assertEqual(t.leaf_choice(list(range(32)),safe,paid)['action'],'cap32')
        self.assertEqual(t.leaf_choice(list(range(16)),safe,paid)['action'],'cap0')
        safe=[[True]*8]*8;paid=[[4]*8]*8;self.assertEqual(t.leaf_choice(list(range(8)),safe,paid)['action'],'cap0')
    def test_one_split_component_gain_and_minimum_leaf(self):
        X,safe,paid=self.tables();split=t.best_split(X,safe,paid,list(range(32)))
        self.assertEqual((split['feature'],split['threshold']),(0,15));self.assertEqual(split['gain'],960)
        self.assertEqual(split['left_choice']['action'],'cap0');self.assertEqual(split['right_choice']['action'],'cap4')
        self.assertIsNone(t.best_split(X,safe,paid,list(range(32)),min_leaf=17))
    def test_thresholds_and_baseline_count_no_top_level_fit(self):
        X=[[i*(j+1) for j in range(12)] for i in range(96)]
        self.assertEqual(t.thresholds(X,list(range(96)),0),[23,47,71]);self.assertEqual(t.baseline_candidate_count(X),2024)
        self.assertEqual(t.baseline_candidate_count([[0]*12 for _ in range(96)]),8)
    def test_route_tie_and_forbidden_unknown_training_rows(self):
        policy=fake_policy('depth3_tree');self.assertEqual(t.route(policy,[.5]+[0]*11),dict(action='cap4',node_id='rootL'))
        self.assertEqual(t.route(policy,[.50001]+[0]*11)['node_id'],'rootR')
        X,safe,paid=self.tables();safe[0][0]=None
        with self.assertRaises(ValueError):t.validate_tables(X,safe,paid,list(range(32)))
    def test_union_calibration_veto_original_route_only_and_once(self):
        tree=fake_policy('depth3_tree');baseline=fake_policy('strong_single_split');original=copy.deepcopy((tree,baseline));mothers=['cal'+str(i) for i in range(24)];records=[]
        for i,mother in enumerate(mothers):
            x=[i%2]+[0]*11
            for kind,model in (('AI',tree),('baseline',baseline)):
                route=t.route(model,x);records.append(dict(mother_id=mother,policy=kind,features=x,route_id=route['node_id'],action=route['action'],safe=not(kind=='AI' and i in (0,2)) and not(kind=='baseline' and i==1)))
        newtree,newbase,receipt=p.apply_calibration_veto(tree,baseline,records,mothers)
        self.assertEqual((tree,baseline),original);self.assertEqual(receipt['veto'],dict(AI=['rootL'],baseline=['rootR']))
        self.assertEqual(t.route(newtree,[0]*12)['action'],'cap32');self.assertEqual(t.route(newtree,[1]+[0]*11)['action'],'window4')
        self.assertEqual(t.route(newbase,[1]+[0]*11)['action'],'cap32')
        with self.assertRaises(ValueError):p.apply_calibration_veto(newtree,newbase,records,mothers)
        different_B=copy.deepcopy(records);different_B[1]['features']=[.1]+[0]*11
        # Both still route to rootL: checking route alone cannot catch this.
        self.assertEqual(t.route(baseline,different_B[1]['features'])['node_id'],'rootL')
        with self.assertRaisesRegex(ValueError,'same mother'):p.apply_calibration_veto(tree,baseline,different_B,mothers)
        records[0]['safe']=None
        with self.assertRaises(ValueError):p.apply_calibration_veto(tree,baseline,records,mothers)

class Ledger:
    def __init__(self):self.counts={}
    def tick(self,key):self.counts[key]=self.counts.get(key,0)+1
    def record(self):return dict(self.counts)

class Backend:
    ResourceUnknown=RuntimeError
    def __init__(self,thirtytwo=False):self.calls=[];self.thirtytwo=thirtytwo
    def _pool(self,scene,common,current,method,ledger):
        keys=[str(i) for i in range(32)] if self.thirtytwo else ['skip','r1','a1'] if current['metrics']['step']==0 else ['r2','r3','r4']
        return [dict(semantic_SHA=k,structurally_eligible=k!='skip') for k in keys]
    def _kernel(self,scene,common,current,descriptor,ledger):
        key=descriptor['semantic_SHA'];self.calls.append(key)
        if key=='a1':return dict(status='admissible',terminal=dict(name='next'),metrics=dict(step=1))
        return dict(status='rejected')

class StopAdapterComponents(unittest.TestCase):
    def common(self):return dict(origin=dict(layout=dict(name='B'),metrics=dict(step=0)))
    def original_control(self,backend,ledger):
        path=Path(__file__).resolve().parents[1]/'src/satellite_rl/coverage/ai_search_r10/core.py'
        raw=path.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),'440aafce1830f4a2045f1e6ac48613cdfcf873af025f9f38b8fdb1a2bd5db1f8')
        node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_sequential')
        namespace=dict(copy=copy,_pool=backend._pool,_kernel=backend._kernel,require=lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg)),ResourceUnknown=RuntimeError)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        return namespace['_sequential'](None,self.common(),'SH-C',ledger)
    def test_cap32_fake_trace_matches_byte_pinned_original_control(self):
        for thirtytwo in (False,True):
            old=Backend(thirtytwo);new=Backend(thirtytwo);oldledger=Ledger();newledger=Ledger()
            expected=self.original_control(old,oldledger);actual=s.execute_stopping_trajectory(None,self.common(),'cap32',backend=new,ledger=newledger)
            actual.pop('selected_rule');actual.pop('engineering_only')
            self.assertEqual(actual,expected);self.assertEqual(new.calls,old.calls);self.assertEqual(newledger.record(),oldledger.record())
    def test_cap0_no_pool_and_window_stops_before_next_kernel(self):
        backend=Backend();ledger=Ledger();result=s.execute_stopping_trajectory(None,self.common(),'cap0',backend=backend,ledger=ledger)
        self.assertEqual(backend.calls,[]);self.assertEqual(result['pool_generations'],0)
        backend=Backend();result=s.execute_stopping_trajectory(None,self.common(),'window2',backend=backend,ledger=Ledger())
        self.assertEqual(backend.calls,['r1','a1','r2','r3']);self.assertEqual(result['selected']['metrics'],dict(step=1))
        self.assertEqual(len(result['screenings']),5)
    def test_pause_is_unknown_and_engineering_stays_two(self):
        backend=Backend();observed={}
        with self.assertRaises(s.CooperativeStop) as caught:s.execute_stopping_trajectory(None,self.common(),'cap32',backend=backend,ledger=Ledger(),stop=lambda:True,observations=observed)
        self.assertEqual(backend.calls,[]);self.assertEqual(caught.exception.partial_result['selected']['terminal'],dict(name='B'))
        result=s.execute_engineering_prefix(None,self.common(),backend=backend,ledger=Ledger());self.assertTrue(result['engineering_only']);self.assertEqual(result['kernel_attempts'],2)

if __name__=='__main__':unittest.main()
