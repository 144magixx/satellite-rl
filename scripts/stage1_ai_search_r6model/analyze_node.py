"""Read-only exact node24 aggregation; never runs inference/search/training."""
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.r5_io import original_records,identity
from satellite_rl.workbench.r6_io import read_asset,pin_tree
from satellite_rl.workbench.r6_node_audit import node_gate,audit_row
from scripts.stage1_ai_search_r4.audit import audit_layout,audit_ledger,scene_sha
from .plan_builder import NODE_INDICES

def analyze(manifest_refs,registry_ref,checkpoint_ref,model_identity,output):
 registry=read_json(pin(registry_ref),{});original={c['input_case_index']:c for c in registry['cases']};cases={}
 for ref in manifest_refs:
  pin_tree(ref);manifest=read_json(pin(ref),{})
  if manifest['phase']!='node2000' or manifest['checkpoint_ref']!=checkpoint_ref or manifest['model_identity']!=model_identity:raise ValueError('actual node fixed model/source chain')
  for key,value in manifest['cases'].items():
   value={**value,'expected_scientific_commit':manifest['scientific_commit']}
   if key in cases:raise ValueError('duplicate case host owner')
   cases[key]=value
 if sorted(map(int,cases))!=NODE_INDICES:raise ValueError('exact original24 denominator')
 result_cases=[]
 for key,case in cases.items():
  source=original[int(key)];prepared,teacher=original_records(source,teacher_required=True);ident=identity(source)
  if case['identity']!=ident:raise ValueError('original node identity')
  with np.load(pin(source['scene_source']['raw_scene']),allow_pickle=False) as z:scene={'positions':z['positions'].tolist(),'demands':z['demands'].tolist(),'aircraft_ids':list(map(str,z['aircraft_ids'])),'profile':str(z['profile'])}
  if scene_sha(scene)!=ident['input_sha256']:raise ValueError('original node raw SHA')
  audit=audit_layout(scene,teacher['layout'],prepared['Ncap'],prepared['Omega_cap']);old_cost=read_json(pin(source['teacher']['publication_cost']),{})
  target_closed=bool(audit['valid'] and audit_ledger(teacher['cost'],len(scene['demands']),'TEACHER')['valid'] and not old_cost.get('unknown_cost') and not old_cost.get('unknown_algorithm_spend'))
  costs={};methods={}
  for method,raw_ref in case['raw_refs'].items():
   pin_tree(raw_ref);row=read_json(pin(raw_ref),{})
   if row['identity']!=ident or row['method']!=method or row['source_commit']!=case['expected_scientific_commit']:raise ValueError('method row identity/source')
   if method=='AI' and (row['checkpoint_ref']!=checkpoint_ref or row['request']['checkpoint_ref']!=checkpoint_ref or row['request']['model_identity']!=model_identity):raise ValueError('actual AI row is not the signed fixed checkpoint/model')
   endpoint=read_json(pin(raw_ref['cost_endpoint_ref']),{})
   if endpoint['raw_ref']['sha256']!=raw_ref['sha256'] or endpoint['identity']!=ident:raise ValueError('cost endpoint raw identity')
   if row['result_ref']:
    actual=read_asset(row['result_ref']);independent=audit_row(scene,prepared,actual,row['request'])
    if independent!=row['audit']:raise ValueError('producer node audit differs from independent raw replay')
    methods[method]=independent
   else:methods[method]={'valid':True,'published':False,'nonfallback':False,'status':'unpublished','metrics':None,'identity_closed':True,'budget_closed':False}
   costs[method]=endpoint['complete_cold_seconds'] if endpoint['cost_known'] else None
  result_cases.append({'case_index':int(key),'methods':methods,'teacher_metrics':audit['metrics'] if audit['valid'] else None,'teacher_identity_and_cost_closed':target_closed,'costs':costs,'preparation_seconds':old_cost['components']['prepare'],'total_request':sum(scene['demands'])})
 gate=node_gate(result_cases);gate.update(model_identity=model_identity,checkpoint_sha256=checkpoint_ref['sha256'],manifest_refs=manifest_refs,registry_ref=registry_ref,original_T_read_only_targets=True,new_T_calls=0,new_model_updates=0);atomic_json(output,gate);return gate
