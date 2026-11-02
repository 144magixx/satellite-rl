"""Pure mocked contracts: no model updates, feature geometry, or real labels."""
import unittest,tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.stage1_ai_search_r6model.plan_builder import training_plan,node_plan,NODE_INDICES,DESIGN_SHA,audit
from satellite_rl.workbench.r6_model_node import method_order
from satellite_rl.workbench.r6_model_io import qualified_rows,load_target
from satellite_rl.workbench.r6_node_audit import node_gate

def ref(name):return {'path':'/frozen/'+name,'sha256':'a'*64}
def inputs():
 x={'design_ref':{'path':'/frozen/design','sha256':DESIGN_SHA},'identity_ref':ref('identity'),'qualification_refs':{'mock':ref('engineering')},'prerequisite_gate_ref':ref('gate'),'model_identity':'one_51001','dataset_ref':{'role':'train','count':2304,'sha256':'7bc2b1f8fdd583e136621e3d25bf8aed371fa72e8d4a399f21f4ed3cd27655af'},'qualified_case_indices':[4*((k*575)//191)+profile for k in range(192) for profile in range(4)][:692]}
 for k in ('qualified','initial_state','normalization','feature_cache','initialization_receipt','checkpoint','recovery_clock','node_gate','registry','supply_endpoint','source_equivalence'):x[k+'_ref']=ref(k)
 return x
class ModelOpsMocks(unittest.TestCase):
 def test_two_segments_single_model_budget(self):
  x=inputs();a=training_plan('b'*40,x,port=20418);b=training_plan('b'*40,x,port=20418,start=2000,end=6000)
  self.assertEqual(a['budget']['updates']+b['budget']['updates'],6000);self.assertEqual(a['model_identity'],b['model_identity']);self.assertIn('node_gate',next(iter(b['tasks'].values()))['model_inputs'])
 def test_cannot_skip_or_expand_updates(self):
  for bounds in ((0,6000),(2000,7000),(0,4000)):
   with self.assertRaises(ValueError):training_plan('b'*40,inputs(),port=20418,start=bounds[0],end=bounds[1])
 def test_node_exact_original24_and_rotation(self):
  x=inputs();x['dataset_ref']={'role':'development','count':80,'sha256':'9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b'};p=node_plan('b'*40,x)
  self.assertEqual(NODE_INDICES,[0,1,2,3,16,17,18,19,32,33,34,35,44,45,46,47,60,61,62,63,76,77,78,79]);self.assertEqual(p['budget'],{'prepare':0,'T':0,'D5':72,'updates':0});self.assertEqual(method_order(1),['RULE','RANDOM','AI']);self.assertEqual(method_order(2),['RANDOM','AI','RULE'])
  p['tasks']['R6_node2000_20418']['methods'].append('T')
  with self.assertRaises(ValueError):audit(p)
 def test_no_unaccepted_labels_table(self):
  gate={'passed':True,'phase':'labels','qualified':691,'nontrivial':154,'capable_mothers':6};m={'schema':'R6QualifiedTrainManifestV1','role':'train','labels_gate_sha256':'a'*64,'qualified_cases':[]}
  with patch('satellite_rl.workbench.r6_model_io.signed_json',side_effect=[m,gate]):
   with self.assertRaisesRegex(ValueError,'scientific gate'):qualified_rows(ref('m'),ref('g'))
 def test_no_mixed_role_table(self):
  gate={'passed':True,'phase':'labels','qualified':692,'nontrivial':154,'capable_mothers':6};rows=[{'case_index':i,'case_id':str(i),'input_sha':'s','role':'validation','label_ref_sha':'a'*64,'original_T_terminal_ref':ref('T')} for i in range(692)];m={'schema':'R6QualifiedTrainManifestV1','role':'train','labels_gate_sha256':'a'*64,'qualified_cases':rows}
  with patch('satellite_rl.workbench.r6_model_io.signed_json',side_effect=[m,gate]):
   with self.assertRaisesRegex(ValueError,'original T table'):qualified_rows(ref('m'),ref('g'))
 def test_efficiency_requires_original_T_closed(self):
  rows=[]
  for i in range(24):
   methods={m:{'valid':True,'identity_closed':True,'budget_closed':True,'published':True,'nonfallback':True,'metrics':{'maxload':100.,'top10_mean':100.,'CV':.1}} for m in ('AI','RULE','RANDOM')};rows.append({'case_index':i,'methods':methods,'total_request':1.,'costs':{'AI':1.,'RULE':10.,'RANDOM':10.},'preparation_seconds':0.})
  result=node_gate(rows);self.assertFalse(result['passed']);self.assertFalse(result['efficiency_branch'])
 def test_gate_fixed_missing_denominator(self):
  with self.assertRaises(ValueError):node_gate([])
 def test_target_original_identity_and_integer_owners(self):
  from types import SimpleNamespace
  ident={'role':'train','input_sha256':'s','case_id':'train:m','mother_id':'m','profile':'base','identity_sha256':'x'};row={'original_T_terminal_ref':ref('T'),'identity':ident,'input_sha':'s','label_ref_sha':'a'*64};feature=SimpleNamespace(aircraft_ids=('a','b'),identity={'role':'train'})
  value={'identity':ident,'result':{**ident,'status':'available','layout':{'assignment':[0,1],'anchor_aircraft_ids':['a','b'],'beam_ids':['q0','q1']}}}
  with patch('satellite_rl.workbench.r6_model_io.read_asset',return_value=value):self.assertEqual(load_target(row,feature).anchor_ids,('a','b'))
  value['result']['layout']['assignment']=[0.5,1.]
  with patch('satellite_rl.workbench.r6_model_io.read_asset',return_value=value):
   with self.assertRaises(ValueError):load_target(row,feature)
class PersistentWorkerMocks(unittest.TestCase):
 def test_worker_checkpoint_progress_and_no_auto_extension_without_numeric_model(self):
  from contextlib import nullcontext
  from types import SimpleNamespace
  import hashlib,sys
  import satellite_rl.workbench.r6_model_train as subject
  row={'case_index':0,'case_id':'train:synthetic','input_sha':'synthetic','role':'train','label_ref_sha':'label'}
  class Sampler:
   def __init__(self,rows):self.table=rows;self.pending=[]
   def state_dict(self):return {'pending':self.pending}
   def assemble(self):self.pending=[0]*16;return self.pending
  class Session:
   def __init__(self,model,stats,sampler,provenance,device):self.model=model;self.sampler=sampler;self.provenance=provenance;self.active_seconds=0.;self.updates=0
   def elapsed(self):return self.active_seconds+1.
   def start_activity(self,clock_path):pass
   def step(self,cases,targets,microbatch):self.updates+=1;return {'update':self.updates,'pure_mock':True}
   def checkpoint(self,path):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'pure-mock-checkpoint-not-real-model');return {'checkpoint_file_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'charged_active_seconds':self.elapsed()}
   def pause_activity(self,reason):pass
  initialization={'passed':True,'model_initial_state_SHA':'tensor','model_initial_state_file_SHA':'a'*64,'model_graph_source_SHA':'model','feature_graph_source_SHA':'feature','model_scientific_source_commit':'b'*40,'runtime_profile_SHA':'runtime','training_runtime_identity':{'runtime_profile_SHA':'runtime'}}
  stats=SimpleNamespace(qualified_table_sha='table',sha='stats');model=SimpleNamespace(load_state_dict=lambda *a,**k:None)
  refs={k:ref(k) for k in ('qualified','initialization_receipt','initial_state','normalization','feature_cache','labels_gate','supply_endpoint','source_equivalence')};job={'model_inputs':refs,'references':{'plan':ref('plan')},'case_indices':[0],'model_identity':'synthetic_single_model','executor_port':20418,'run_id':'mock','scientific_commit':'b'*40,'mode':'train2000'};config={'training':{'start_update':0,'end_update':2,'checkpoint_interval':100}}
  def read(reference):
   return {'cache_ref':refs['feature_cache'],'normalization_ref':refs['normalization'],'qualified_ref':refs['qualified'],'charged_supply_seconds_including_publication':2.} if reference['path'].endswith('supply_endpoint') else {'runtime_profile_SHA':'runtime'} if reference['path'].endswith('runtime_v3_identity.json') else initialization if reference['path'].endswith('initialization_receipt') else {'prerequisite_gate_ref':ref('gate')} if reference['path'].endswith('plan') else {}
  with tempfile.TemporaryDirectory(dir='E:/paper_archives') as d,patch.dict(sys.modules,{'satellite_rl.coverage.ai_search_r6.runtime_v3':SimpleNamespace(assert_v3_runtime=lambda identity:None)}),patch('satellite_rl.workbench.r6_model_limits.check'),patch.object(subject,'MODEL_OWNER_ROOT',Path(d)/'owners'),patch.object(subject,'exclusive_lock',side_effect=lambda *a,**k:nullcontext()),patch.object(subject,'signed_json',side_effect=read),patch.object(subject,'qualified_rows',return_value=({},[row])),patch.object(subject,'verify_model_source_equivalence',return_value={'model_graph_source_SHA':'model','feature_graph_source_SHA':'feature'}),patch.object(subject,'validate_cache',return_value=2.),patch.object(subject,'load_feature',return_value=SimpleNamespace()),patch.object(subject,'load_cached_target',return_value=SimpleNamespace()),patch.object(subject,'pin',side_effect=lambda r:Path(r['path'])),patch.object(subject.torch,'load',return_value={}),patch('satellite_rl.coverage.ai_search_r6.model.initialize_model',return_value=model),patch('satellite_rl.coverage.ai_search_r6.model.model_tensor_sha',return_value='tensor'),patch('satellite_rl.coverage.ai_search_r6.features.FeatureStats.from_record',return_value=stats),patch('satellite_rl.coverage.ai_search_r6.learning.CaseSampler',Sampler),patch('satellite_rl.coverage.ai_search_r6.learning.TrainingSession',Session):
   directory=Path(d)/'job';directory.mkdir();job['model_dir']=str(Path(d)/'models/runs/mock');(directory/'runtime_v3_identity.json').write_text('{}');result=subject.train(job,config,directory);self.assertEqual(result['updates'],2);endpoint=json.loads((directory/'training_endpoint.json').read_text());self.assertFalse(endpoint['future_phase_dispatch_allowed']);self.assertEqual(endpoint['science_updates_reserved'],2)
   self.assertTrue((directory/'step_intent.json').exists());self.assertTrue((directory/'checkpoint_index.json').exists())
class CompleteSourceFreezeTests(unittest.TestCase):
 def test_real_git_snapshot_imports_required_scripts_from_foreign_cwd(self):
  import shutil,subprocess,threading,os,sys
  from types import SimpleNamespace
  from satellite_rl.workbench.r6_model_freeze import freeze_source
  repository=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory(dir='E:/paper_archives') as d:
   root=Path(d);checkout=root/'repo';checkout.mkdir()
   for name in ('src','scripts'):shutil.copytree(repository/name,checkout/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   shutil.copy2(repository/'pyproject.toml',checkout/'pyproject.toml');shutil.copy2(repository/'.gitattributes',checkout/'.gitattributes')
   # The runtime-v3 owner works in the main repository; fold its new file only
   # into this disposable qualification checkout, never change shared source.
   runtime=repository/'src/satellite_rl/coverage/ai_search_r6/runtime_v3.py'
   if not runtime.exists():runtime=Path('D:/paper/src/satellite_rl/coverage/ai_search_r6/runtime_v3.py')
   shutil.copy2(runtime,checkout/'src/satellite_rl/coverage/ai_search_r6/runtime_v3.py')
   def git(*args):return subprocess.run(['git','-C',str(checkout),*args],check=True,capture_output=True)
   git('init','-q');git('add','.');git('-c','user.name=PureMock','-c','user.email=puremock@example.invalid','commit','-qm','pure source import qualification no scientific data')
   store=SimpleNamespace(lock=threading.RLock(),source_root=checkout,root=root/'archive');snapshot,commit=freeze_source(store)
   self.assertTrue((snapshot/'scripts/stage1_ai_search_r6model/plan_builder.py').is_file());self.assertEqual(freeze_source(store),(snapshot,commit))
   foreign=root/'foreign';foreign.mkdir();env=os.environ.copy();env.update(PYTHONPATH=str(snapshot/'src')+os.pathsep+str(snapshot),PYTHONDONTWRITEBYTECODE='1')
   code='import satellite_rl.coverage.ai_search_r6.runtime_v3;import satellite_rl.workbench.coverage_ai_search_r6_model;import satellite_rl.workbench.ai_search_r6_model_runner;import satellite_rl.workbench.r6_node_audit;import scripts.stage1_ai_search_r6model.plan_builder;print("import-only closure passed")'
   result=subprocess.run([sys.executable,'-B','-c',code],cwd=foreign,env=env,capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('import-only closure passed',result.stdout)
   (snapshot/'scripts/stage1_ai_search_r6model/plan_builder.py').write_text('tampered')
   with self.assertRaises(ValueError):freeze_source(store)
class NestedD5MockTests(unittest.TestCase):
 def test_outer_proposal_calls_separate_D5_hard_guard_once(self):
  from types import SimpleNamespace
  from satellite_rl.workbench.r6_model_eval_stage import execute
  from satellite_rl.workbench.r6_io import write_asset
  from satellite_rl.workbench.storage import digest
  import satellite_rl.coverage.ai_search_r5.core as core
  ident={'role':'development','input_sha256':'input','case_id':'development:stub','mother_id':'stub','profile':'base','identity_sha256':'identity'}
  prepared=SimpleNamespace(identity=ident,source_sha256='original_fee_preparation')
  with tempfile.TemporaryDirectory(dir='E:/paper_archives') as d:
   root=Path(d);raw=root/'raw.npz';np.savez(raw,positions=[[0.,0.]],demands=[1.],aircraft_ids=['a'],profile='europe_v1');old=root/'original.json';old.write_text(json.dumps({'result':{}}));out=root/'call';out.mkdir()
   req={'identity':ident,'method':'RULE','scene_ref':{'path':str(raw),'sha256':digest(raw)},'prepared_ref':{'path':str(old),'sha256':digest(old)},'producer_sha256':'producer','b7_decoder_source_SHA':'immutable_b7'}
   def guard(command,target,seconds,rss,**kwargs):
    self.assertEqual(seconds,15);self.assertEqual(rss,2147483648);self.assertIn('satellite_rl.workbench.r6_model_d5_stage',command)
    asset=write_asset(target/'stage_output.json.gz',{'result':{'status':'success','pure_mock':True}});(target/'stage_output.json').write_text(json.dumps({'asset_ref':asset}));(target/'guard_receipt.json').write_text('{}');return {'status':'completed','output_ref':{'path':str(target/'stage_output.json'),'sha256':digest(target/'stage_output.json')}}
   with patch('satellite_rl.coverage.ai_search.core.Scene',return_value=SimpleNamespace(input_sha256='input')),patch.object(core.PreparedCase,'from_dict',return_value=prepared),patch.object(core,'build_proposal',return_value={'schema':'ProposalV2','status':'ready'}),patch.object(core,'decode_groups') as direct,patch('satellite_rl.workbench.r5_stage_guard.run_stage',side_effect=guard) as nested:
    execute(req,out);nested.assert_called_once();direct.assert_not_called()
   request=json.loads((out/'D5_guard/request.json').read_text());self.assertEqual(request['original_prepared_source_sha256'],'original_fee_preparation');self.assertEqual(request['b7_decoder_source_SHA'],'immutable_b7');self.assertNotIn('teacher_ref',request)
 def test_parent_death_binding_prevents_orphan_algorithm(self):
  from types import SimpleNamespace
  from satellite_rl.workbench.r6_model_d5_stage import bind_parent
  with patch('satellite_rl.workbench.r6_model_d5_stage.signal.SIGKILL',9,create=True),patch('satellite_rl.workbench.r6_model_d5_stage.sys.platform','linux'),patch.dict('os.environ',{'R6_D5_PARENT_PID':'123'}),patch('satellite_rl.workbench.r6_model_d5_stage.ctypes.CDLL',return_value=SimpleNamespace(prctl=lambda *a:0)),patch('satellite_rl.workbench.r6_model_d5_stage.os.getppid',return_value=124):
   with self.assertRaisesRegex(RuntimeError,'parent changed'):bind_parent()
if __name__=='__main__':unittest.main()
