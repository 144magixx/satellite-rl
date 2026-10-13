import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch

def ref(name):return {"path":"/frozen/"+name,"sha256":"a"*64}
class PersistentWorkerMocks(unittest.TestCase):
 def test_worker_checkpoint_progress_and_no_auto_extension_without_numeric_model(self):
  from contextlib import nullcontext
  from types import SimpleNamespace
  import hashlib,sys
  import satellite_rl.workbench.r7_model_train as subject
  row={'case_index':0,'case_id':'train:synthetic','input_sha':'synthetic','role':'train','label_ref_sha':'label'}
  class Sampler:
   def __init__(self,rows):self.table=rows;self.pending=[]
   def state_dict(self):return {'pending':self.pending}
   def assemble(self):self.pending=[0]*16;return self.pending
  class Session:
   def __init__(self,model,stats,sampler,provenance,device):self.model=model;self.sampler=sampler;self.provenance=provenance;self.active_seconds=0.;self.updates=0;self.active_started=None
   def elapsed(self):return self.active_seconds+1.
   def start_activity(self,clock_path):self.active_started=1;Path(clock_path).write_text('{}')
   def step(self,cases,targets,microbatch):self.updates+=1;return {'update':self.updates,'pure_mock':True}
   def checkpoint(self,path):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'pure-mock-checkpoint-not-real-model');return {'checkpoint_file_sha':hashlib.sha256(path.read_bytes()).hexdigest(),'charged_active_seconds':self.elapsed()}
   def pause_activity(self,reason):self.active_started=None
  initialization={'passed':True,'model_initial_state_SHA':'tensor','model_initial_state_file_SHA':'a'*64,'model_graph_source_SHA':'model','feature_graph_source_SHA':'feature','model_scientific_source_commit':'b'*40,'runtime_profile_SHA':'runtime','training_runtime_identity':{'runtime_profile_SHA':'runtime'}}
  stats=SimpleNamespace(qualified_table_sha='table',sha='stats');model=SimpleNamespace(load_state_dict=lambda *a,**k:None)
  refs={k:ref(k) for k in ('qualified','initialization_receipt','initial_state','normalization','feature_cache','labels_gate','supply_endpoint','source_equivalence','train_setup_endpoint','authority')};job={'model_inputs':refs,'references':{'plan':ref('plan')},'case_indices':[0],'model_identity':'synthetic_single_model','executor_port':20474,'run_id':'mock','scientific_commit':'b'*40,'mode':'train2000'};config={'datasets':{'train':ref('dataset')},'training':{'start_update':0,'end_update':2,'checkpoint_interval':100}}
  def read(reference):
   if reference['path'].endswith('train_setup_endpoint'):return {'training_refs':{},'activity_origin':{},'runtime_identity':{'runtime_profile_SHA':'runtime'}}
   if reference['path'].endswith('gate'):return {'passed':True,'phase':'train_setup','model_identity':job['model_identity'],'train_setup_endpoint_ref':refs['train_setup_endpoint']}
   return {'cache_ref':refs['feature_cache'],'normalization_ref':refs['normalization'],'qualified_ref':refs['qualified'],'charged_supply_seconds_including_publication':2.} if reference['path'].endswith('supply_endpoint') else {'runtime_profile_SHA':'runtime'} if reference['path'].endswith('runtime_v3_identity.json') else initialization if reference['path'].endswith('initialization_receipt') else {'prerequisite_gate_ref':ref('gate')} if reference['path'].endswith('plan') else {}
  deadline=patch('satellite_rl.workbench.r7_deadline.stop_boundary',return_value=True);deadline.start();self.addCleanup(deadline.stop)
  with tempfile.TemporaryDirectory(dir='E:/paper_archives') as d,patch.dict(sys.modules,{'satellite_rl.coverage.ai_search_r6.runtime_v3':SimpleNamespace(assert_v3_runtime=lambda identity:None)}),patch('satellite_rl.workbench.r7_train_binding.validate_setup',return_value=2.),patch('satellite_rl.workbench.r7_train_binding.validate_owner',return_value=True),patch('satellite_rl.workbench.r7_model_limits.check'),patch.object(subject,'MODEL_OWNER_ROOT',Path(d)/'owners'),patch.object(subject,'exclusive_lock',side_effect=lambda *a,**k:nullcontext()),patch.object(subject,'signed_json',side_effect=read),patch.object(subject,'qualified_rows',return_value=({},[row])),patch.object(subject,'verify_model_source_equivalence',return_value={'model_graph_source_SHA':'model','feature_graph_source_SHA':'feature'}),patch.object(subject,'validate_cache',return_value=2.),patch.object(subject,'load_feature',return_value=SimpleNamespace()),patch.object(subject,'load_cached_target',return_value=SimpleNamespace()),patch.object(subject,'pin',side_effect=lambda r:Path(r['path'])),patch.object(subject.torch,'load',return_value={}),patch('satellite_rl.coverage.ai_search_r6.model.initialize_model',return_value=model),patch('satellite_rl.coverage.ai_search_r6.model.model_tensor_sha',return_value='tensor'),patch('satellite_rl.coverage.ai_search_r6.features.FeatureStats.from_record',return_value=stats),patch('satellite_rl.coverage.ai_search_r6.learning.CaseSampler',Sampler),patch('satellite_rl.coverage.ai_search_r6.learning.TrainingSession',Session):
   directory=Path(d)/'job';directory.mkdir();job['model_dir']=str(Path(d)/'models/runs/mock');(directory/'job.json').write_text(json.dumps(job));(directory/'runtime_v3_identity.json').write_text('{}');result=subject.train(job,config,directory);self.assertEqual(result['updates'],2);endpoint=json.loads((directory/'training_endpoint.json').read_text());self.assertFalse(endpoint['future_phase_dispatch_allowed']);self.assertEqual(endpoint['science_updates_reserved'],2)
   self.assertTrue((directory/'step_intent.json').exists());self.assertTrue((directory/'checkpoint_index.json').exists())

if __name__=="__main__":unittest.main()
