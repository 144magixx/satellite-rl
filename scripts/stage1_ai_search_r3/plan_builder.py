"""Expand only the frozen pilot24 and exact six-method matrix, without science."""
from copy import deepcopy
from pathlib import Path
import argparse,json,hashlib
from satellite_rl.training.run_monitor import atomic_json
from . import ops
from satellite_rl.workbench.coverage_ai_search_r3 import DESIGN_SHA,METHODS

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(commit,inputs,mode='prepare_candidates',prepared_refs=None):
 if len(commit)!=40 or set(commit)-set('0123456789abcdef'):raise ValueError('exact commit')
 source=inputs['reference_inventory'];order=source['mother_order'];selected=source['selected_case_indices'];mothers=source['selected_mothers'];ids=inputs['identities'];design=Path(__file__).with_name('round3_design.json')
 if digest(design)!=DESIGN_SHA or len(selected)!=24 or len(mothers)!=6:raise ValueError('frozen v2/pilot24')
 if mode not in ('prepare_candidates','development_gate'):raise ValueError('pilot-only mode')
 plan={'group':ops.GROUP,'phase':'pilot','authorized':True,'scientific_commit':commit,'design':{'path':ops.BASE+'/source/round3_design.json','sha256':DESIGN_SHA},'identity_ref':inputs['identity_ref'],'qualification_refs':{'core':inputs['qualification_ref']},'mother_order':order,'selected_mothers':mothers,'selected_case_indices':selected,'training_allowed':False,'test_allowed':False,'tasks':{},'maximum_solver_calls':0 if mode=='prepare_candidates' else 144,'source_boundary':'original inputs plus input-only candidate construction; R2 anchors for charged projection and external quality only'}
 ordinal={m:i for i,m in enumerate(mothers)};source_cases={str(r['input_case_index']):r for r in source['cases']} if isinstance(source['cases'],list) else source['cases']
 for lane,port in enumerate(ops.SSH_PORTS):
  cases=[i for i in selected if ordinal[ids['mother_ids'][i]]%3==lane];key='R3_'+mode+'_'+str(port)
  config={'schema_version':11,'research_stage':'coverage','algorithm':'ai_search_r3','name':key,'protocol_ref':plan['design'],'datasets':{'development':inputs['dataset_ref']},'workers':1,'resources':{'minimum_free_bytes':8589934592,'max_rss_bytes':4294967296},'vendor_ref':inputs['vendor_refs'][str(port)]}
  plan['tasks'][key]={'config':config,'mode':mode,'dataset_role':'development','case_indices':cases,'executor_port':port,'methods':list(METHODS) if mode=='development_gate' else [],'prepared_ref':None if mode=='prepare_candidates' else prepared_refs[str(port)],'external_refs':{str(i):deepcopy(source_cases[str(i)]['external_refs']) for i in cases}}
 audit(plan,ids);return plan

def audit(p,ids):
 mode={t['mode'] for t in p['tasks'].values()}
 if len(mode)!=1 or mode.pop()not in ('prepare_candidates','development_gate') or p['phase']!='pilot' or p['training_allowed']or p['test_allowed']:raise ValueError('pilot only')
 if p['selected_case_indices']!=sorted(p['selected_case_indices'])or len(set(p['selected_case_indices']))!=24 or len(p['tasks'])!=3:raise ValueError('exact24/3lane')
 seen=set();owners={};solves=0;rows=0
 for t in p['tasks'].values():
  if t['config']['workers']!=1 or t['config']['protocol_ref']['sha256']!=DESIGN_SHA:raise ValueError('source science settings')
  methods=t['methods']or ['POOL']
  if t['mode']=='development_gate' and methods!=list(METHODS):raise ValueError('frozen six methods')
  if t['mode']=='prepare_candidates' and (t['methods']or t['prepared_ref']is not None):raise ValueError('no solver preparation')
  for i in t['case_indices']:
   if i not in p['selected_case_indices']or i in seen:raise ValueError('duplicate/unselected case')
   seen.add(i);m=ids['mother_ids'][i]
   if owners.setdefault(m,t['executor_port'])!=t['executor_port']:raise ValueError('mother split')
   rows+=len(methods);solves+=6 if t['mode']=='development_gate' else 0
 if seen!=set(p['selected_case_indices'])or len(owners)!=6 or {t['executor_port']for t in p['tasks'].values()}!=set(ops.SSH_PORTS):raise ValueError('matrix incomplete')
 return {'passed':True,'cases':24,'mothers':6,'result_rows':rows,'maximum_solver_calls':solves,'training_updates':0,'validation_test_algorithms':0,'original_pilot_R2_solver_calls':0}

if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--commit',required=True);a.add_argument('--inputs',required=True);a.add_argument('--mode',choices=['prepare_candidates','development_gate'],default='prepare_candidates');a.add_argument('--prepared-refs');a.add_argument('--output',required=True);v=a.parse_args();inputs=json.loads(Path(v.inputs).read_text('utf-8'));prepared=json.loads(Path(v.prepared_refs).read_text('utf-8'))if v.prepared_refs else None;p=build(v.commit,inputs,v.mode,prepared);atomic_json(v.output,p);atomic_json(str(v.output)+'.expanded.json',audit(p,inputs['identities']));print(json.dumps({'path':v.output,'sha256':digest(v.output),'science_submitted':0}))