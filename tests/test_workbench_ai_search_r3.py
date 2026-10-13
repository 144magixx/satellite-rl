"""Self-contained R3 service/dispatch/accounting fixtures, zero native solves."""
from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from satellite_rl.workbench import coverage_ai_search_r3 as p
from satellite_rl.workbench.ai_search_r3_runner import method_order,publication_accounting
from satellite_rl.workbench.ai_search_runner import recover_rows,result_key
from satellite_rl.workbench.storage import digest
from satellite_rl.training.run_monitor import atomic_json
from scripts.stage1_ai_search_r3.plan_builder import build,audit

class R3WorkbenchTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.design=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search_r3/round3_design.json'
  self.order=[f'development:2024-01-12T{i:02}:00:00Z' for i in range(20)];mothers=[m for m in self.order for _ in range(4)];profiles=list(('base','one_hotspot','two_hotspots','edge_heavy'))*20;self.ids={'mother_ids':mothers,'case_ids':[f'case{i}'for i in range(80)],'profiles':profiles,'sha256':'9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b'}
  self.identity=self.root/'identity.json';atomic_json(self.identity,{'development':self.ids});selected=[4*m+x for m in [0,4,8,11,15,19]for x in range(4)];anchor=self.root/'anchor.json';atomic_json(anchor,{});ref={'path':str(anchor),'sha256':digest(anchor)}
  from satellite_rl.coverage.ai_search.core import Layout,layout_sha256
  layout=Layout([[0,0]],[.5],[0]);external={}
  for i in selected:
   a=self.root/f'audit{i}.json';atomic_json(a,{'passed':True,'issues':[],'case_id':self.ids['case_ids'][i],'input_sha256':'e'*64});ar={'path':str(a),'sha256':digest(a)}
   b=self.root/f'bundle{i}.json';atomic_json(b,{'schema':'r3_three_fixed_R2_external_layouts_v1','case_id':self.ids['case_ids'][i],'input_case_index':i,'input_dataset_sha256':self.ids['sha256'],'input_sha256':'e'*64,'original_case_audit_ref':ar,'layouts':{m:{'layout':layout.to_record(),'layout_sha256':layout_sha256(layout)}for m in ['parent','D0','D1']}})
   external[i]={'audit':ar,**{m:{'path':str(b),'sha256':digest(b),'json_pointer':['layouts',m,'layout']}for m in ['parent','D0','D1']}}
  self.inputs={'identities':self.ids,'identity_ref':{'path':str(self.identity),'sha256':digest(self.identity)},'dataset_ref':{'path':'/canonical/input.npz','sha256':self.ids['sha256'],'count':80,'role':'development'},'vendor_refs':{str(port):ref for port in [20418,20424,20473]},'qualification_ref':ref,'reference_inventory':{'mother_order':self.order,'selected_case_indices':selected,'selected_mothers':[self.order[m]for m in [0,4,8,11,15,19]],'cases':[{'input_case_index':i,'external_refs':external[i]}for i in selected]}}
  self.plan=build('a'*40,self.inputs);self.plan['design']={'path':str(self.design),'sha256':p.DESIGN_SHA};self.task=deepcopy(next(iter(self.plan['tasks'].values())));self.task['config']['protocol_ref']=self.plan['design'];self.plan['tasks'][next(iter(self.plan['tasks']))]=self.task
 def tearDown(self):self.tmp.cleanup()
 def validate(self,task=None,plan=None):
  with patch.object(p,'validate_vendor',return_value={'passed':True}):return p.validate_task(task or self.task,plan or self.plan,str((task or self.task)['executor_port']))
 def test_pilot_prepare_scope(self):self.validate();self.assertEqual(audit(build('a'*40,self.inputs),self.ids)['maximum_solver_calls'],0)
 def test_development_exact144(self):
  plan=build('a'*40,self.inputs,'development_gate',{str(port):{'path':'artifact','sha256':'b'*64}for port in [20418,20424,20473]});self.assertEqual(audit(plan,self.ids)['result_rows'],144);self.assertEqual(audit(plan,self.ids)['maximum_solver_calls'],144)
 def test_wrong_dataset_rejected(self):
  t=deepcopy(self.task);t['config']['datasets']['development']['sha256']='c'*64
  with self.assertRaises(ValueError):self.validate(t)
 def test_unselected_scope_rejected(self):
  q=deepcopy(self.plan);q['selected_case_indices'][0]=4
  with self.assertRaises(ValueError):self.validate(plan=q)
 def test_train_mode_rejected(self):
  t=deepcopy(self.task);t['mode']='train'
  with self.assertRaises(ValueError):self.validate(t)
 def test_extension_phase_rejected(self):
  q=deepcopy(self.plan);q['phase']='extension'
  with self.assertRaises(ValueError):self.validate(plan=q)
 def test_multiple_workers_rejected(self):
  t=deepcopy(self.task);t['config']['workers']=8
  with self.assertRaises(ValueError):self.validate(t)
 def test_changed_protocol_rejected(self):
  t=deepcopy(self.task);t['config']['protocol_ref']['sha256']='d'*64
  with self.assertRaises(ValueError):self.validate(t)
 def test_method_rotation_dependency(self):
  for i in range(80):
   order=method_order(i);self.assertEqual(set(order),set(p.METHODS));self.assertLess(order.index('REF20'),order.index('ORACLE_RESTRICT'))
 def test_full_publication_cost_sum(self):
  c=publication_accounting('RULE',29,0.6,0.5);self.assertAlmostEqual(c['comparable_publication_seconds'],30.1);self.assertFalse(c['regular_publication_eligible']);self.assertTrue(publication_accounting('REF20',35,0.5,0.5)['regular_publication_eligible'])
 def test_unknown_publication_not_zero(self):
  c=publication_accounting('RULE',None,0.1,0.2);self.assertIsNone(c['comparable_publication_seconds']);self.assertFalse(c['regular_publication_eligible'])
 def test_unknown_started_not_repeated(self):
  (self.root/'cases').mkdir();key=result_key('case0','RULE');atomic_json(self.root/'started'/(key+'.json'),{'case_id':'case0','method':'RULE'})
  with self.assertRaisesRegex(ValueError,'unknown spent'):recover_rows(self.root,{key})
 def test_prepared_other_epoch_rejected(self):
  t=deepcopy(self.task);t['mode']='development_gate';t['methods']=list(p.METHODS);f=self.root/'prepared.json';atomic_json(f,{'group':p.GROUP,'completed':True,'phase':'pilot','source_commit':'b'*40,'execution_commit':'b'*40,'design_ref':{'sha256':p.DESIGN_SHA},'cases':{str(i):{}for i in t['case_indices']}});t['prepared_ref']={'path':str(f),'sha256':digest(f)}
  with self.assertRaisesRegex(ValueError,'prepared epoch'):self.validate(t)
 def test_root_admission_absent_rejected(self):
  with self.assertRaises((ValueError,KeyError,TypeError)):p.validate_admission(self.plan,{'path':'x','sha256':'x'},None)
if __name__=='__main__':unittest.main()