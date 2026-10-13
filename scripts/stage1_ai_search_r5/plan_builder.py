"""Mechanical P0 plan; no data arrays, algorithms, deployment or task POST."""
import argparse,json,hashlib
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_ai_search_r5 import DESIGN_SHA,IDENTITY_SHA,DATA_SHA,REGISTRY_SHA,METHODS
from . import ops

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def configure(identity_path,dataset_ref_path,qualification_path,process_path,registry_path,output):
 ids=json.loads(Path(identity_path).read_text('utf-8'))['development'];data=json.loads(Path(dataset_ref_path).read_text('utf-8'))
 if digest(identity_path)!=IDENTITY_SHA or ids['sha256']!=DATA_SHA or data['sha256']!=DATA_SHA or digest(registry_path)!=REGISTRY_SHA:raise ValueError('pinned development identity/data/registry')
 metadata=[]
 def ref(local,name):
  sha=digest(local);remote=ops.BASE+'/source/'+name+'/'+sha+Path(local).suffix;metadata.append({'local':str(local),'remote':remote,'sha256':sha});return {'path':remote,'sha256':sha}
 value={'identities':ids,'identity_ref':ref(identity_path,'identities'),'dataset_ref':data,'qualification_ref':ref(qualification_path,'qualifications'),'process_ref':ref(process_path,'process_qualifications'),'registry_ref':ref(registry_path,'registries'),'metadata':metadata};atomic_json(output,value);return value

def build(commit,inputs,routing=None):
 d=json.loads(Path(__file__).with_name('round5_design.json').read_text('utf-8'));indices=d['P0']['case_indices']
 if routing is None:routing={str(i):ops.SSH_PORTS[n%4] for n,i in enumerate(indices)}
 if set(routing)!=set(map(str,indices)) or any(v not in ops.SSH_PORTS for v in routing.values()):raise ValueError('exact case host routing required')
 p={'group':ops.GROUP,'phase':'P0','authorized':True,'scientific_commit':commit,'design':{'path':ops.BASE+'/source/round5_design.json','sha256':DESIGN_SHA},'identity_ref':inputs['identity_ref'],'registry_ref':inputs['registry_ref'],'qualification_refs':{'geometry':inputs['qualification_ref'],'process':inputs['process_ref']},'mother_order':sorted(set(inputs['identities']['mother_ids'])),'selected_mothers':d['data']['mother_ids'],'selected_case_indices':indices,'training_allowed':False,'test_allowed':False,'tasks':{},'budget':{'new_prepare_calls':0,'new_teacher_calls':0,'decoder_slots':168,'result_slots':168,'MILP_calls':0,'training_updates':0,'validation_test_algorithms':0}}
 for port in ops.SSH_PORTS:
  key='R5_P0_'+str(port);cfg={'schema_version':13,'research_stage':'coverage','algorithm':'ai_search_r5','name':key,'protocol_ref':p['design'],'datasets':{'development':inputs['dataset_ref']},'workers':1,'resources':{'minimum_free_bytes':8589934592,'parent_children_rss_bytes':2147483648},'hard_watchdogs':{'decoder':15}}
  p['tasks'][key]={'config':cfg,'mode':'p0','dataset_role':'development','case_indices':[i for i in indices if routing[str(i)]==port],'executor_port':port,'methods':list(METHODS)}
 audit(p,inputs['identities']);return p

def audit(p,ids):
 selected=json.loads(Path(__file__).with_name('round5_design.json').read_text('utf-8'))['P0']['case_indices'];allcases=[]
 if p.get('phase')!='P0' or p.get('training_allowed') is not False or p.get('test_allowed') is not False or p['selected_case_indices']!=selected:raise ValueError('P0 only')
 if set(p['tasks'])!={'R5_P0_'+str(port) for port in ops.SSH_PORTS}:raise ValueError('four lanes')
 for key,t in p['tasks'].items():
  if not t['case_indices'] or t['methods']!=list(METHODS) or t['mode']!='p0' or t['dataset_role']!='development' or t['executor_port'] not in ops.SSH_PORTS:raise ValueError('seven same-case same-host slots')
  allcases.extend(t['case_indices'])
 if sorted(allcases)!=selected or len(set(allcases))!=24:raise ValueError('exact 24 unique cases')
 return {'passed':True,'cases':24,'new_prepare_calls':0,'new_teacher_calls':0,'D5_decoder_slots':120,'D4_decoder_slots':48,'result_slots':168,'MILP_calls':0,'training_updates':0,'geometry_calls_here':0,'case_routes':{str(i):t['executor_port'] for t in p['tasks'].values() for i in t['case_indices']}}
if __name__=='__main__':
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='op',required=True);a=s.add_parser('inputs')
 for k in ['identity','dataset-ref','qualification','process','registry','output']:a.add_argument('--'+k,required=True)
 a=s.add_parser('plan')
 for k in ['commit','inputs','output']:a.add_argument('--'+k,required=True)
 a.add_argument('--routing');v=p.parse_args()
 if v.op=='inputs':configure(v.identity,v.dataset_ref,v.qualification,v.process,v.registry,v.output)
 else:
  inputs=json.loads(Path(v.inputs).read_text('utf-8'));routing=json.loads(Path(v.routing).read_text('utf-8')) if v.routing else None;plan=build(v.commit,inputs,routing);atomic_json(v.output,plan);atomic_json(v.output+'.expanded.json',audit(plan,inputs['identities']))
 print(json.dumps({'path':v.output,'sha256':digest(v.output),'science_POSTs':0}))
