"""Mechanical B0:24 complete-case bundles, teacher24/decoder96, no science."""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_ai_search_r4 import DESIGN_SHA,IDENTITY_SHA,DATA_SHA,METHODS
from . import ops

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def configure(identity_path,dataset_ref_path,qualification_path,output):
 identity=Path(identity_path);qualification=Path(qualification_path);ids=json.loads(identity.read_text('utf-8'))['development'];q=json.loads(qualification.read_text('utf-8'));dataset=json.loads(Path(dataset_ref_path).read_text('utf-8'))
 if digest(identity)!=IDENTITY_SHA or ids['sha256']!=DATA_SHA or dataset['sha256']!=DATA_SHA or not q.get('passed'):raise ValueError('actual pinned identity/dataset/qualification needed')
 ir={'path':ops.BASE+'/source/identities/'+IDENTITY_SHA+'.json','sha256':IDENTITY_SHA};qr={'path':ops.BASE+'/source/qualifications/'+digest(qualification)+'.json','sha256':digest(qualification)}
 value={'identities':ids,'identity_ref':ir,'dataset_ref':dataset,'qualification_ref':qr,'metadata':[{'remote':ir['path'],'local':str(identity),'sha256':ir['sha256']},{'remote':qr['path'],'local':str(qualification),'sha256':qr['sha256']}]};atomic_json(output,value);return value

def build(commit,inputs):
 d=json.loads(Path(__file__).with_name('round4_design.json').read_text('utf-8'));order=sorted(set(inputs['identities']['mother_ids']));mothers=d['data']['mother_ids'];indices=d['data']['case_indices'];p={'group':ops.GROUP,'phase':'B0','authorized':True,'scientific_commit':commit,'design':{'path':ops.BASE+'/source/round4_design.json','sha256':DESIGN_SHA},'identity_ref':inputs['identity_ref'],'qualification_refs':{'geometry':inputs['qualification_ref']},'mother_order':order,'selected_mothers':mothers,'selected_case_indices':indices,'training_allowed':False,'test_allowed':False,'tasks':{},'budget':{'teacher_slots':24,'decoder_slots':96,'result_slots':120,'MILP_calls':0,'training_updates':0,'validation_test_algorithms':0}}
 ordinal={m:i for i,m in enumerate(mothers)}
 for lane,port in enumerate(ops.SSH_PORTS):
  selected=[i for i in indices if ordinal[inputs['identities']['mother_ids'][i]]%3==lane];key='R4_B0_'+str(port);cfg={'schema_version':12,'research_stage':'coverage','algorithm':'ai_search_r4','name':key,'protocol_ref':p['design'],'datasets':{'development':inputs['dataset_ref']},'workers':1,'resources':{'minimum_free_bytes':8589934592,'parent_children_rss_bytes':2147483648},'hard_watchdogs':{'preparation':20,'teacher':45,'decoder':15}};p['tasks'][key]={'config':cfg,'mode':'b0','dataset_role':'development','case_indices':selected,'executor_port':port,'methods':list(METHODS)}
 audit(p,inputs['identities']);return p

def audit(p,ids):
 seen=set();owners={};counts=[]
 for t in p['tasks'].values():
  if t['mode']!='b0'or t['methods']!=list(METHODS)or t['dataset_role']!='development'or t['config']['workers']!=1:raise ValueError('B0 exactmechanism')
  counts.append(len(t['case_indices']))
  for i in t['case_indices']:
   if i in seen or i not in p['selected_case_indices']:raise ValueError('repeat/unselectedcase')
   seen.add(i);m=ids['mother_ids'][i]
   if owners.setdefault(m,t['executor_port'])!=t['executor_port']:raise ValueError('mother split')
 if sorted(counts)!=[8,8,8]or len(seen)!=24 or len(owners)!=6 or set(p['tasks'])!={'R4_B0_'+str(port)for port in ops.SSH_PORTS}:raise ValueError('24cases/6whole mothers/3lanes required')
 return {'passed':True,'cases':24,'teacher_slots':24,'decoder_slots':96,'result_slots':120,'MILP_calls':0,'training_updates':0,'validation_test_calls':0,'geometry_calls_here':0}
if __name__=='__main__':
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='op',required=True);a=sub.add_parser('inputs')
 for k in ['identity','dataset-ref','qualification','output']:a.add_argument('--'+k,required=True)
 a=sub.add_parser('plan')
 for k in ['commit','inputs','output']:a.add_argument('--'+k,required=True)
 v=parser.parse_args()
 if v.op=='inputs':configure(v.identity,v.dataset_ref,v.qualification,v.output)
 else:
  inputs=json.loads(Path(v.inputs).read_text('utf-8'));p=build(v.commit,inputs);atomic_json(v.output,p);atomic_json(v.output+'.expanded.json',audit(p,inputs['identities']))
 print(json.dumps({'path':v.output,'sha256':digest(v.output),'science_POSTs':0}))