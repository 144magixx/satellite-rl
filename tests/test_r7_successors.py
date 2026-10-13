"""Pure phase/budget/pause/source mocks; zero model/geometry/heldout/API calls."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.stage1_ai_search_r7.contract import *
from scripts.stage1_ai_search_r7.plan_builder import build,audit
from scripts.stage1_ai_search_r7.controller import tick
from scripts.stage1_ai_search_r7.test_statistics import log_ratio
from satellite_rl.workbench.r7_successor_binding import gate_requirements,continuation
from satellite_rl.workbench.r7_node_audit import node_gate
from satellite_rl.workbench.r7_heldout_data import validate_recipe,CONTAINERS
from satellite_rl.workbench.r7_uniform_data import load_algorithm_case
from satellite_rl.workbench import r7_deadline,r4_monitor_repair

def ref(n):return dict(path='/mock/'+n,sha256=(str(len(n)%10))*64)
def inputs():
    hooks={str(p):{k:ref(str(p)+k) for k in r7_deadline.KEYS} for p in PORTS}
    return dict(design_ref=dict(path='/mock/design',sha256=DESIGN_SHA),identity_ref=ref('identity'),prerequisite_gate_ref=ref('gate'),qualification_refs={'source':ref('qualification')},model_identity=MODEL_IDENTITY,qualified_case_indices=list(range(519)),model_inputs={'authority':ref('authority')},datasets={},deadline_guard_refs_by_port=hooks,archive_roots_by_port={str(p):f'/home/u2024110224/first_stage_satellite_rl/s1_ai17/{p}/round7/learning/archive' for p in PORTS})

def rows(n,joint=0):
    return [dict(case_index=i,mother_id=str(i),methods={m:dict(valid=True,published=True,nonfallback=True,identity_closed=True,budget_closed=True,metrics=dict(maxload=98 if m=='AI' else 100,top10_mean=99 if m=='AI' else 100,CV=.1)) for m in ('AI','RULE','RANDOM')},total_request=21600000,costs=dict(AI=1.,RULE=1.,RANDOM=1.,T=1.),teacher_identity_and_cost_closed=True,actual_AI_both_N_width_capability=i<joint) for i in range(n)]

class Phases(unittest.TestCase):
    def test_successors_exact_budget_routing_and_split_steps(self):
        for phase,n in (('train6000',519),('fixed6000_development',20),('validation',20),('test',40)):
            for step in ('derive','evaluate') if phase in ('validation','test') else (None,):
                args=inputs();args['heldout_stage']=step;p=build('a'*40,phase,args)
                self.assertEqual(sum(len(t['case_indices']) for t in p['tasks'].values()),n);self.assertTrue(audit(p)['passed'])
                if phase=='train6000':self.assertEqual(p['budget']['updates'],4000);self.assertEqual(next(iter(p['tasks'].values()))['executor_port'],20474)
                if step=='derive':self.assertEqual((p['budget']['prepare'],p['budget']['T'],p['budget']['D5']),(0,0,0))
                if step=='evaluate':self.assertEqual(p['budget']['builds'],0)
        self.assertEqual(total_budget(),dict(builds=656,prepare=676,T=676,D5=876,updates=6000))
    def test_no_new_model_no_unregistered_guard_no_extension(self):
        a=inputs();a['model_identity']='other'
        with self.assertRaises(ValueError):build('a'*40,'train6000',a)
        a=inputs();a['deadline_guard_refs_by_port']={}
        with self.assertRaises(ValueError):build('a'*40,'train6000',a)
        p=build('a'*40,'train6000',inputs());next(iter(p['tasks'].values()))['config']['training']['start_update']=0
        with self.assertRaises(ValueError):audit(p)
    def test_2000_and_joint6000_gate_not_substituted(self):
        g=dict(group=GROUP,phase='node2000',passed=True,model_identity=MODEL_IDENTITY,denominator=20,hard_identity_budget_cost_closed=True,nonfallback=15,quality_successes=5,checkpoint_sha256=ref('cp')['sha256'])
        self.assertTrue(gate_requirements('train6000',g,MODEL_IDENTITY,ref('cp')))
        with self.assertRaises(ValueError):gate_requirements('train6000',dict(g,quality_successes=4),MODEL_IDENTITY,ref('cp'))
        g.update(phase='fixed6000_development',actual_AI_both_N_width_capability=4)
        with self.assertRaises(ValueError):gate_requirements('validation',g,MODEL_IDENTITY,ref('cp'))
        self.assertTrue(node_gate(rows(20,5),'fixed6000_development')['passed']);self.assertFalse(node_gate(rows(20,4),'fixed6000_development')['passed'])
    def test_controller_pause_precedes_any_POST_and_is_sticky(self):
        spec=dict(phase='train6000',tasks=[dict(port=20474,body=dict(request_id='q',mode='train6000'))]);state=dict(phase='running',intents={})
        def paused(*a):raise RuntimeError('USER_0700_PAUSE')
        tick(spec,state,lambda *a:self.fail('no API'),lambda *a:self.fail('no observe'),lambda x:None,paused)
        self.assertEqual(state['phase'],'user_paused');self.assertFalse(state['automatic_resume'])
        tick(spec,state,lambda *a:self.fail('no automatic resume'),None,lambda x:None,lambda *a:True)
        self.assertEqual(state['phase'],'user_paused')

class DataLocks(unittest.TestCase):
    def test_heldout_algorithm_blocks_before_file_read(self):
        with patch('satellite_rl.workbench.r7_uniform_data.pin',side_effect=AssertionError('no case arrays read')):
            with self.assertRaisesRegex(ValueError,'before case array read'):load_algorithm_case(ref('case'),ref('authority'),expected_role='test')
    def test_sealed_recipe_metadata_wrong_dates_roles_counts_rejected(self):
        role='validation';r=dict(schema='R7HeldoutSourceRecipeV1',role=role,design_SHA=DESIGN_SHA,fixed_request_bps=21600000,original_container_ref=dict(path='/mock/sealed',sha256=CONTAINERS[role]),rows=[])
        for i in range(20):
            utc=f'2024-02-17T00:{i:02}:00Z';r['rows'].append(dict(actual_role=role,source_role=role,source_phase_index=i,UTC=utc,original_mother_id=role+':'+utc,case_id=f'r7:{role}:{role}:{utc}:uniform21600000',source_aircraft_count=2,raw_ref=ref('raw'),metadata_ref=ref('meta'),source_asset_SHA=ref('raw')['sha256'],original_remote_ref='historical',metadata_UTC_pointer='/UTC',metadata_raw_SHA_pointer='/raw',metadata_scene_id_pointer='/scene',metadata_aircraft_count_pointer='/count',profiles=[dict(profile=str(j),scene_ref=ref('p'+str(j)),container_scene_ref_pointer='/p'+str(j)) for j in range(4)]))
        with patch('satellite_rl.workbench.r7_heldout_data.pin',side_effect=AssertionError('metadata validation must not read source files')):self.assertIs(validate_recipe(r,role),r)
        for key,value in (('UTC','2024-02-21T00:00:00Z'),('source_phase_index',1),('actual_role','test')):
            changed=copy.deepcopy(r);changed['rows'][0][key]=value
            with self.assertRaises(ValueError):validate_recipe(changed,role)
    def test_statistics_zero_cv_rules_no_epsilon(self):
        self.assertEqual(log_ratio(0,.1,'CV'),0);self.assertEqual(log_ratio(0,0,'CV'),0)
        with self.assertRaises(ValueError):log_ratio(.01,0,'CV')
        with self.assertRaises(ValueError):log_ratio(0,1,'maxload')
        with self.assertRaises(ValueError):log_ratio(float('nan'),1,'top10_mean')

class PauseAndRSS(unittest.TestCase):
    def test_shared_guard_require_uses_exact_mode_registry(self):
        refs={k:ref(k) for k in r7_deadline.KEYS};guard=SimpleNamespace(process_identity=lambda pid:dict(pid=pid,alive=True),require_dispatch_admission=lambda *a,**kw:None,registrations=lambda s:[dict(archive_root='/a',experiment_id=GROUP,modes=['fixed6000_development'])])
        with patch.object(r7_deadline,'load',return_value=(guard,{})),patch.object(r7_deadline,'ACTIVE_TRANSITION_REF',None),patch.dict('os.environ',{},clear=True):
            self.assertTrue(r7_deadline.require(refs,'/a',GROUP,'fixed6000_development'))
            with self.assertRaises(RuntimeError):r7_deadline.require(refs,'/a',GROUP,'test')
    def test_actual_service_installers_select_new_guard_after_explicit_continue(self):
        old={k:ref('old'+k) for k in r7_deadline.KEYS};new={k:ref('new'+k) for k in r7_deadline.KEYS};first=SimpleNamespace();second=SimpleNamespace()
        transition=dict(old_guard_refs_by_port={'20474':old},new_guard_refs_by_port={'20474':new})
        with patch.object(r7_deadline,'ACTIVE_TRANSITION_REF',ref('transition')),patch.object(r7_deadline,'install_transition',return_value=transition),patch.object(r7_deadline,'load',side_effect=[(first,dict(ssh_port=20474)),(second,dict(ssh_port=20474))]):
            guard,spec,actual=r7_deadline.active_service_guard(old);self.assertIs(guard,second);self.assertEqual(actual,new)
    def test_restored2000_never_enters_zero_checkpoint_branch(self):
        source=Path(__file__).resolve().parents[1]/'src/satellite_rl/workbench/r7_model_train.py';s=source.read_text();self.assertIn('if not checkpoint_index and resume_ref:',s);self.assertIn('updates=session.updates',s);self.assertIn('if not checkpoint_index and not resume_ref:',s)
    def test_trainer_pause_finally_closes_resource_clock_without_core_enum_change(self):
        from satellite_rl.workbench import r7_model_train as t
        session=SimpleNamespace(active_started=1,updates=42,elapsed=lambda:13.)
        def pause(*,reason):self.assertEqual(reason,'resource_queue');session.active_started=None
        session.pause_activity=pause
        def interrupted(j,c,d,ctx):ctx['session']=session;raise RuntimeError('cooperative stop')
        with tempfile.TemporaryDirectory() as d,patch.object(t,'_train',side_effect=interrupted):
            Path(d,'stop.request').write_text('user_restart_0700')
            with self.assertRaises(RuntimeError):t.train({}, {},d)
            self.assertEqual(json.loads(Path(d,'user_restart_activity_pause.json').read_text())['reason'],'user_restart_0700')
    def test_stop_after_durable_checkpoint_never_between_optimizer_and_checkpoint(self):
        source=Path(__file__).resolve().parents[1]/'src/satellite_rl/workbench/r7_model_train.py';s=source.read_text();step=s.index('result=session.step');checkpoint=s.index('receipt=session.checkpoint',step);boundary=s.index('stop_boundary(job,directory) # only',checkpoint)
        self.assertNotIn('stop_boundary(',s[step:checkpoint]);self.assertLess(checkpoint,boundary);self.assertIn("owner['last_completed_update']=update",s[checkpoint:boundary])
    def test_ESRCH_confirmed_only_other_errors_fatal(self):
        for state in ('Z','X','S'):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'7';p.mkdir();(p/'stat').write_text('7 (child) '+state+' '+' '.join(['0']*20));original=Path.read_text
                def read(path,*a,**k):
                    if path.name=='status':raise ProcessLookupError(3,'exit')
                    return original(path,*a,**k)
                with patch.object(Path,'read_text',read):
                    if state in ('Z','X'):self.assertEqual(r4_monitor_repair.rss_tree(7,proc=Path(d),page_size=4096),0)
                    else:
                        with self.assertRaises(ProcessLookupError):r4_monitor_repair.rss_tree(7,proc=Path(d),page_size=4096)
        with tempfile.TemporaryDirectory() as d,patch.object(Path,'read_text',side_effect=PermissionError('live permission')):
            with self.assertRaises(PermissionError):r4_monitor_repair.rss_tree(7,proc=Path(d),page_size=4096)
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'7').mkdir()
            with patch.object(Path,'read_text',side_effect=ProcessLookupError(3,'provider error while root still exists')):
                with self.assertRaises(ProcessLookupError):r4_monitor_repair.rss_tree(7,proc=Path(d),page_size=4096)

if __name__=='__main__':unittest.main()
