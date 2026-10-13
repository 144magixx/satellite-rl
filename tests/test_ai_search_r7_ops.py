"""Pure static/metadata/fault mocks: real prepare/T/D5/optimizer/data builds zero."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r7.contract import *
from scripts.stage1_ai_search_r7.plan_builder import build,audit
from scripts.stage1_ai_search_r7.controller import tick
from scripts.stage1_ai_search_r7.label_audit import gate
from satellite_rl.workbench.r7_node import method_order
from satellite_rl.workbench.r7_node_audit import node_gate
from satellite_rl.workbench.r7_uniform_data import metadata_field,validate_recipe
from satellite_rl.workbench.r7_resources import controlled_science_command,belongs_to_archive,lane_from_archive

def ref(name):return dict(path='/mock/'+name,sha256='a'*64)
def inputs():
    return dict(design_ref=dict(path='/mock/design',sha256=DESIGN_SHA),identity_ref=ref('identity'),
                prerequisite_gate_ref=ref('gate'),qualification_refs={'mock':ref('qualification')},
                model_identity='r7_seed51001_one_model',qualified_case_indices=list(range(519)),
                model_inputs={'authority':ref('authority')},datasets={})

def node_rows(quality=True):
    rows=[]
    for i in range(20):
        methods={m:dict(valid=True,published=True,nonfallback=True,identity_closed=True,budget_closed=True,
                       metrics=dict(maxload=98. if m=='AI' and quality else 100.,top10_mean=100.,CV=.1)) for m in ('AI','RULE','RANDOM')}
        rows.append(dict(case_index=i,methods=methods,total_request=21600000,costs=dict(AI=100.,RULE=1.,RANDOM=1.,T=1.),teacher_identity_and_cost_closed=True))
    return rows

class ContractTests(unittest.TestCase):
    def test_total_independent_budget(self):self.assertEqual(total_budget(),dict(builds=656,prepare=676,T=676,D5=876,updates=6000))
    def test_first_phases_exact_case_counts(self):
        for phase,count in (('data_train_dev',596),('labels',576),('node2000',20),('train_setup',519),('train2000',519)):
            p=build('b'*40,phase,inputs());self.assertEqual(sum(len(t['case_indices']) for t in p['tasks'].values()),count);self.assertTrue(audit(p)['passed'])
    def test_no_lent_ports(self):
        for port in (20418,20424):
            with self.assertRaises(ValueError):build('b'*40,'labels',inputs(),{str(i):port for i in range(576)})
    def test_model_owner_only74(self):
        with self.assertRaises(ValueError):build('b'*40,'train2000',inputs(),{str(i):20473 for i in range(519)})
    def test_duplicate_routes_and_missing_not_accepted(self):
        p=build('b'*40,'labels',inputs());task=next(iter(p['tasks'].values()));task['case_indices'].append(task['case_indices'][0])
        with self.assertRaises(ValueError):audit(p)
    def test_no_old_fourprofile_candidate_indices(self):
        with self.assertRaises(ValueError):qualified_indices(list(range(692)))
        with self.assertRaises(ValueError):qualified_indices(list(range(518)))
    def test_expanded_budget_rejected(self):
        p=build('b'*40,'node2000',inputs());p['budget']['D5']=72
        with self.assertRaises(ValueError):audit(p)
    def test_later_successors_sealed(self):
        for phase in SEALED_PHASES:
            self.assertFalse(sealed_phase_interface(phase)['dispatch_allowed'])
            with self.assertRaises(ValueError):build('b'*40,phase,inputs())
    def test_sampler_bounds(self):
        p=build('b'*40,'train2000',inputs());task=next(iter(p['tasks'].values()));task['config']['training']['end_update']=6000
        with self.assertRaises(ValueError):audit(p)
    def test_four_method_rotation(self):
        self.assertEqual(method_order(0),['AI','RULE','RANDOM','T']);self.assertEqual(method_order(1),['RULE','RANDOM','T','AI']);self.assertEqual(method_order(3),['T','AI','RULE','RANDOM'])

class GateTests(unittest.TestCase):
    def test_exact_labels_denominator_and_threshold(self):
        rows=[dict(qualified=i<519,nontrivial=i<116,capability=i<6,mother_id=str(i)) for i in range(576)]
        self.assertTrue(gate(rows)['passed']);rows[0]['nontrivial']=False;self.assertFalse(gate(rows)['passed'])
        with self.assertRaises(ValueError):gate(rows[:-1])
    def test_capability_unique_mothers(self):
        rows=[dict(qualified=True,nontrivial=True,capability=i<6,mother_id='same') for i in range(576)]
        self.assertFalse(gate(rows)['passed'])
    def test_node_quality_no_efficiency_alternative(self):
        self.assertTrue(node_gate(node_rows())['passed'])
        result=node_gate(node_rows(False));self.assertFalse(result['passed']);self.assertFalse(result['efficiency_branch'])
    def test_exact20_no_case_drop(self):
        with self.assertRaises(ValueError):node_gate(node_rows()[:-1])
    def test_unpublished_and_cost_missing_fail(self):
        rows=node_rows();rows[0]['costs']['AI']=float('nan');self.assertFalse(node_gate(rows)['passed'])
        for row in rows:row['methods']['AI'].update(published=False,nonfallback=False,metrics=None)
        self.assertEqual(node_gate(rows)['nonfallback'],0)
    def test_zero_request_no_exemption(self):
        rows=node_rows();rows[0]['total_request']=0
        with self.assertRaises(ValueError):node_gate(rows)

class DispatcherFaultTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(phase='labels',scientific_commit='b'*40,tasks=[dict(port=20474,body={'request_id':'r7_mock'},task_key='key')]);self.state=dict(phase='running',intents={});self.route=lambda port,commit:dict(ready=True,root_accepted=True,ssh_port=port,scientific_commit=commit);self.persist=lambda state:None
    def test_unknown_POST_is_never_replayed(self):
        calls=[]
        def api(port,path,body=None):calls.append(body);raise ConnectionError('mock unknown response')
        tick(self.spec,self.state,api,self.route,self.persist);self.assertEqual(self.state['phase'],'unknown_post_outcome')
        tick(self.spec,self.state,api,self.route,self.persist);self.assertEqual(len(calls),1)
    def test_crash_after_intent_before_response_is_unknown(self):
        self.state['intents']['r7_mock']=dict(phase='post_pending')
        with patch('builtins.print'):
            tick(self.spec,self.state,lambda *a:self.fail('POST forbidden'),self.route,self.persist)
        self.assertEqual(self.state['phase'],'unknown_post_outcome')
    def test_one_completed_request_same_id_no_POST(self):
        calls=[]
        def api(port,path,body=None):calls.append(body);return {'run_id':'original'} if body else {'status':{'stage':'completed'}}
        tick(self.spec,self.state,api,self.route,self.persist);self.assertEqual(self.state['phase'],'completed');self.assertEqual(sum(x is not None for x in calls),1)
    def test_existing_run_poll_does_not_require_new_dispatch_route(self):
        from scripts.stage1_ai_search_r7.controller import semantic
        self.state['intents']['r7_mock']=dict(run_id='original',phase='running',payload_sha256=semantic(self.spec['tasks'][0]['body']))
        tick(self.spec,self.state,lambda *args:{'status':{'stage':'completed'}},lambda *args:self.fail('resource only before new dispatch'),self.persist);self.assertEqual(self.state['phase'],'completed')

class SourceBoundaryTests(unittest.TestCase):
    def test_metadata_JSON_pointer_not_aircraft_timestamp(self):self.assertEqual(metadata_field({'a':[{'selected_time_utc':'original'}]},'/a/0/selected_time_utc'),'original')
    def test_only_exact_R7_recipe(self):
        with self.assertRaises(ValueError):validate_recipe(dict(schema='oldR6',fixed_request_bps=21600000))
    def test_resource_exact_archive_and_excluded_lanes(self):
        root='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round7/archive'
        self.assertEqual(lane_from_archive(root),20474)
        with self.assertRaises(ValueError):lane_from_archive(root.replace('20474','20424'))
        self.assertFalse(belongs_to_archive('python -m satellite_rl.workbench.runner --job-dir '+root+'2/job',root))
    def test_program_slot_not_argv_spoof(self):
        self.assertTrue(controlled_science_command('python -B -m satellite_rl.workbench.r7_stage --request /own/request.json'))
        self.assertFalse(controlled_science_command('python -c pass -m satellite_rl.workbench.r7_stage --request /own/request.json'))

class ParallelCaseTests(unittest.TestCase):
    def test_two_cases_unique_owner_parent_serial_merge_no_real_spawn(self):
        import satellite_rl.workbench.r7_parallel as subject
        from satellite_rl.training.run_monitor import atomic_json
        job=dict(mode='labels',case_indices=[0,1,2],experiment_id=GROUP,scientific_commit='b'*40)
        config=dict(case_parallelism=2,datasets={'train':dict(sha256='a'*64)})
        created=[]
        class Process:
            returncode=0
            def __init__(self,pid):self.pid=pid;self.polls=0
            def poll(self):self.polls+=1;return None if self.polls==1 else 0
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);atomic_json(directory/'job.json',job)
            def spawn(command,**kwargs):
                i=int(command[command.index('--case-index')+1]);target=Path(command[command.index('--directory')+1]);created.append(i)
                atomic_json(target/'case_result.json',dict(case_index=i,parent_job_sha256=subject.digest(directory/'job.json'),mode='labels',result_ref=ref(str(i))))
                return Process(100+i)
            with patch.object(subject.subprocess,'Popen',side_effect=spawn),patch.object(subject,'birth',return_value=1),patch.object(subject,'rss_tree',return_value=1024),patch.object(subject.shutil,'disk_usage',return_value=type('Disk',(),{'free':10*1024**3})()),patch.object(subject.time,'sleep'),patch.object(subject,'terminate_owned') as killed:
                result=subject.parallel_cases(job,config,directory)
            self.assertEqual(created,[0,1,2]);self.assertEqual(result['case_parallelism'],2);self.assertEqual(set(result['cases']),{'0','1','2'});killed.assert_not_called()
    def test_parent_race_rejected_before_algorithm(self):
        import satellite_rl.workbench.r7_parent as subject
        from types import SimpleNamespace
        with patch.object(subject.sys,'platform','linux'),patch.dict(subject.os.environ,{'R7_CASE_PARENT_PID':'100'}),patch.object(subject.ctypes,'CDLL',return_value=SimpleNamespace(prctl=lambda *a:0)),patch.object(subject.signal,'SIGKILL',9,create=True),patch.object(subject.os,'getppid',return_value=101):
            with self.assertRaises(RuntimeError):subject.bind_parent('R7_CASE_PARENT_PID')

class DerivationMockTests(unittest.TestCase):
    def test_transport_paths_do_not_change_content_and_UTC_is_snapshot(self):
        import gzip,hashlib,sys
        from types import SimpleNamespace
        import satellite_rl.workbench.r7_uniform_data as subject
        captured=[]
        def make_case(meta,ids,positions):
            captured.append(meta);return dict(**meta,aircraft_ids=ids,lat_lon_deg=positions,content_sha256='b'*64)
        stubs={'satellite_rl.shared_interface.stages':SimpleNamespace(make_case=make_case),
               'satellite_rl.shared_interface.content':SimpleNamespace(content_sha=lambda value:'c'*64),
               'satellite_rl.shared_interface.contracts':SimpleNamespace(Authority=lambda *a:object(),validate_case=lambda *a:None)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'one.gz'
            with gzip.open(raw,'wt',encoding='utf8') as f:f.write(json.dumps(dict(aircraft_id='keep_zero_old_request',lat=50.,lon=8.,demand_bps=0,beam_id=-1,timestamp_utc='NOT_SCENE_UTC',altitude_m=1000))+'\n')
            rawsha=hashlib.sha256(raw.read_bytes()).hexdigest();counts=dict(selected_aircraft=1)
            metadata=dict(snapshot=dict(selected_time_utc='2024-01-08T00:02:00Z',counts=counts,source_observations_ref=dict(sha256=rawsha)),file=dict(scene_id='original'))
            job=root/'job.json';job.write_text(json.dumps(metadata));jobsha=hashlib.sha256(job.read_bytes()).hexdigest()
            row=dict(source_local_path=str(raw),source_asset_SHA=rawsha,metadata_snapshot_JSON_pointer='/snapshot',metadata_file_JSON_pointer='/file',original_scene_id='original',metadata_job_SHA=jobsha,source_role='train',actual_role='train',source_phase_index=0,original_mother_id='train:2024-01-08T00:02:00Z',UTC='2024-01-08T00:02:00Z',source_aircraft_count=1,source_selection_counts=counts,source_remote_path='/immutable/historical.gz',original_mother_id_derivation='source_role:UTC')
            other=root/'replica';other.mkdir();raw2=other/'other.gz';raw2.write_bytes(raw.read_bytes());job2=other/'job2.json';job2.write_bytes(job.read_bytes())
            with patch.dict(sys.modules,stubs):
                a=subject.construct(row,dict(path=str(job),sha256=jobsha),{rawsha:str(raw),jobsha:str(job)})
                b=subject.construct(row,dict(path=str(job),sha256=jobsha),{rawsha:str(raw2),jobsha:str(job2)})
            self.assertEqual(a['source'],b['source']);self.assertEqual(a['UTC'],'2024-01-08T00:02:00Z');self.assertEqual(a['aircraft_ids'],['keep_zero_old_request'])
            self.assertNotIn('path',json.dumps(a['source']));self.assertFalse(a['source']['obsolete_request_assignment_fields_used'])

if __name__=='__main__':unittest.main()
