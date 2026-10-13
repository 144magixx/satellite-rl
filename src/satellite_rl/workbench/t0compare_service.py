"""Direct-only S2-T24 workbench. Queued resource hold precedes any worker."""
from pathlib import Path
from copy import deepcopy
import argparse,json,hashlib,os,subprocess,threading,uuid,time,html
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from satellite_rl.workbench.storage import Store,read_json,digest,now
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.datasets import freeze_scene_set
from satellite_rl.workbench.t0compare_contract import validate_t0_config,reject_third_stage,training_config,ALGORITHMS
from satellite_rl.workbench.runtime import process_start_time
BASE=Path('/home/u2024110224/second_stage_satellite_rl/s2_t24');PORTS={20418:9040,20424:9041,20473:9042,20474:9043}
class Workbench:
 def __init__(self,port,source_root,reference_config,design,max_workers=1):
  if port not in PORTS:raise ValueError('authorized S2-T24 lane only')
  if max_workers not in (1,2,3):raise ValueError('finite admitted concurrency only')
  self.port=port;self.root=BASE/str(port);self.store=Store(self.root/'archive',source_root);self.reference=reference_config;self.design=design;self.max_workers=max_workers;self.children={};self.lock=threading.RLock()
 def tasks(self):
  rows=[]
  for j,s in self.store.jobs():
   row={'job':{k:j.get(k) for k in ('run_id','request_id','kind','t0_mode','algorithm','experiment_id','model_dir','git_commit','created_at')},'status':s}
   if j.get('t0_mode')=='train' and s.get('stage')=='completed':
    package=Path(j['model_dir'])/'final';metadata=read_json(package/'model.json',{})
    row['endpoint_model_ref']={'algorithm':j['algorithm'],'model_path':str(package),'env_step':metadata.get('provenance',{}).get('env_step'),'seed':metadata.get('config',{}).get('training',{}).get('seed'),'endpoint_verified':metadata.get('provenance',{}).get('env_step')==150000,'model_run_id':j['run_id'],'model_files':[{'path':str(package/'model.json'),'sha256':digest(package/'model.json')},{'path':str(package/'policy.pt'),'sha256':metadata.get('weights_sha256')}]}
   rows.append(row)
  return {'tasks':rows}
 def detail(self,run_id):
  d=self.store.job_dir(run_id);return {'job':read_json(d/'job.json'),'config':read_json(d/'config.json'),'status':read_json(d/'status.json'),'results':read_json(d/'results.json'),'logs':{p.name:p.read_text(encoding='utf8',errors='replace')[-16000:] for p in d.glob('*.log')}}
 def submit(self,body):
  reject_third_stage(body)
  data_rule_ref=body.get('uniform_request_or_explicit_historical_reproduction_authority_ref')
  if not isinstance(data_rule_ref,dict):raise ValueError('source/data hold: uniform_request_or_explicit_historical_reproduction_authority missing; old596SHAidentity is not admission')
  data_rule=read_json(data_rule_ref['path'])
  if digest(data_rule_ref['path'])!=data_rule_ref['sha256'] or data_rule.get('format')!='s2t24_data_purpose_admission_v1' or data_rule.get('uniform_request_or_explicit_historical_reproduction_authority') is not True:raise ValueError('actual uniformrequest/purpose authority notqualified')
  if data_rule.get('purpose') not in ('explicit_historical_reproduction','uniform_request_interface_qualified_n100_subdomain') or data_rule.get('final_variable_layout_pipeline_complete') is not False or not data_rule.get('actual_source_or_human_authority_ref'):raise ValueError('historical/narrow T0purpose must be explicit; cannot claim final N/width-variable pipeline')
  mode=body.get('mode');algorithm=body.get('algorithm')
  if mode not in ('prepare','engineering','train','matrix'):raise ValueError('only source prepare/train/Direct matrix allowed')
  if mode in ('train','engineering') and algorithm not in ALGORITHMS:raise ValueError('one authorized formal arm')
  expected='s2t24-'+algorithm if mode=='train' else 's2t24-engineering-'+algorithm if mode=='engineering' else 's2t24-test-prepare' if mode=='prepare' else 's2t24-direct-matrix'
  if body.get('request_id')!=expected:raise ValueError('one unique predeclared request per arm/matrix')
  roles={'train'} if mode=='engineering' else {'train','validation'} if mode=='train' else {'test'} if mode=='matrix' else set();sets=body.get('scene_sets',{})
  if set(sets)!=roles:raise ValueError('train cannot see test; matrix only frozen newtest')
  counts={'train':576,'validation':20,'test':40}
  if any(sets[k].get('count')!=counts[k] for k in roles):raise ValueError('original T0/staticN100 scene cardinality differs')
  if mode=='prepare':
   data=body.get('data_config',{})
   if data.get('experiment_id')!='S2-T24' or [row.get('date') for row in data.get('source_assets',[])]!=['2024-02-21','2024-02-22'] or not data.get('scanner_reference') or not data.get('converter_reference') or not data.get('freshness_registry'):raise ValueError('prepare source/dates/freshness/scanner/generator metadata not frozen')
  config=training_config(self.reference,algorithm if mode in ('train','engineering') else 'sac')
  if mode=='engineering':
   from satellite_rl.t0_comparison.contract import canonical_config
   config['training'].update(total_steps=512,warmup_steps=256,batch_size=256);config['evaluation'].update(validation_count=0,test_count=0,interval_steps=0,extra_steps=[]);canonical_config(config,purpose='engineering')
  else:validate_t0_config(config,self.reference)
  request_hash=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  with self.lock:
   for job,status in self.store.jobs():
    if job.get('request_id')==expected:
     if job['request_hash']!=request_hash:raise ValueError('same request used different inputs')
     return {'job':job,'status':status}
   snapshot,commit=self.store.freeze_source();run_id=('engineering_'+algorithm if mode=='engineering' else 'train_'+algorithm if mode=='train' else 'test_prepare' if mode=='prepare' else 'direct_matrix')+'_'+uuid.uuid4().hex[:12];directory=self.store.job_dir(run_id);directory.mkdir();models=self.store.models_root/run_id;models.mkdir();frozen={k:freeze_scene_set(record,directory/'scene_inputs'/k,'europe_v1') for k,record in sets.items()};endpoints={}
   if mode=='matrix':
    if set(body.get('endpoint_models',{}))!=set(ALGORITHMS):raise ValueError('three verified full endpoint references required')
    for method,ref in body['endpoint_models'].items():
     if ref.get('env_step')!=150000 or ref.get('seed')!=51001 or ref.get('algorithm')!=method or ref.get('endpoint_verified') is not True:raise ValueError('endpoint full150k/source qualification absent')
     package=directory/'model_inputs'/method;package.mkdir(parents=True);copied=[]
     for row in ref['model_files']:
      p=Path(row['path']);assert p.is_file() and digest(p)==row['sha256'];out=package/p.name
      if out.exists():raise ValueError('duplicate endpoint member')
      import shutil
      shutil.copyfile(p,out);assert digest(out)==row['sha256'];copied.append({'path':str(out),'sha256':row['sha256'],'source_path':str(p)})
     metadata=read_json(package/'model.json')
     if not metadata or metadata.get('provenance',{}).get('env_step')!=150000 or metadata.get('provenance',{}).get('git_commit')!=commit or metadata.get('config',{}).get('algorithm')!=method or metadata['config']['training']['seed']!=51001:raise ValueError('actual endpoint metadata/source differs')
     endpoints[method]={**ref,'model_path':str(package),'model_files':copied}
   job={'run_id':run_id,'kind':'engineering' if mode=='engineering' else 'train' if mode=='train' else 'prepare' if mode=='prepare' else 'evaluate','t0_mode':mode,'algorithm':algorithm if mode in ('train','engineering') else 'direct_matrix','name':'S2-T24 '+(algorithm or 'six-method Direct matrix'),'experiment_id':'S2-T24','executor_port':self.port,'created_at':now(),'request_id':expected,'request_hash':request_hash,'request_body':body,'archive_root':str(self.store.archive_root),'model_dir':str(models),'source_root':str(snapshot),'git_commit':commit,'design_contract':self.design,'design_contract_sha256':hashlib.sha256(json.dumps(self.design,sort_keys=True).encode()).hexdigest(),'scene_sets':frozen,'endpoint_models':endpoints,'from_scratch':mode in ('train','engineering'),'formal_model_reference_eligible':mode=='train','engineering_step_cap':512 if mode=='engineering' else 0,'refinement_calls':0,'data_config':deepcopy(body.get('data_config'))}
   if mode=='prepare':
    job['data_config']['owner_run_id']=run_id;job['data_config']['frozen_source_sha']=digest(Path(snapshot)/'source_manifest.json');job['data_config']['historical_purpose_authority_sha']=data_rule_ref['sha256']
    scientific_refs=[]
    for role in ('scanner_reference','converter_reference','freshness_registry','historical_authority'):
     ref=job['data_config'][role];original=Path(ref['path']);assert original.is_file() and digest(original)==ref['sha256'];target=directory/'scientific_references'/role/original.name;target.parent.mkdir(parents=True);__import__('shutil').copyfile(original,target);assert digest(target)==ref['sha256'];job['data_config'][role]={**ref,'path':str(target),'original_source_path':str(original)};scientific_refs.append({'role':role,'path':str(target),'sha256':ref['sha256'],'original_source_path':str(original)})
    for i,asset in enumerate(job['data_config']['source_assets']):
     ref=asset['metadata_reference'];original=Path(ref['path']);assert original.is_file() and digest(original)==ref['sha256'];target=directory/'scientific_references'/('official_metadata_'+str(i))/original.name;target.parent.mkdir(parents=True);__import__('shutil').copyfile(original,target);asset['metadata_reference']={**ref,'path':str(target),'original_source_path':str(original)};scientific_refs.append({'role':'official_metadata_'+str(i),'path':str(target),'sha256':ref['sha256']})
    job['data_config']['frozen_source_root']=str(directory);job['scientific_reference_files']=scientific_refs
   owner_dir=BASE/'ownership';owner_dir.mkdir(parents=True,exist_ok=True);owner_path=owner_dir/(expected+'.json')
   owner={'experiment_id':'S2-T24','request_id':expected,'run_id':run_id,'executor_port':self.port,'git_commit':commit,'request_hash':request_hash}
   with owner_path.open('x',encoding='utf8') as owner_file:json.dump(owner,owner_file)
   job['unique_owner_ref']={'path':str(owner_path),'sha256':digest(owner_path)}
   atomic_json(directory/'config.json',config);atomic_json(directory/'job.json',job);atomic_json(directory/'status.json',{'run_id':run_id,'stage':'queued_resources','env_step':0,'resource_waiting_reason':'Source/data/engineering/old-owner release/future reservations and fresh resources must pass before admission','phase':'Direct','refinement_calls':0})
   return {'job':job,'status':read_json(directory/'status.json')}
 def admission_proof(self,run_id,template_ref):
  # Only actual SHA-bound Root qualifications can authorize autonomous fresh assembly.
  with self.lock:
   d=self.store.job_dir(run_id);job=read_json(d/'job.json');template=read_json(template_ref['path'])
   if digest(template_ref['path'])!=template_ref['sha256'] or template.get('format')!='s2t24_admission_template_v1' or template.get('approved') is not True or template.get('scientific_commit')!=job['git_commit'] or template.get('executor_port')!=self.port:raise ValueError('Root typed admission template identity mismatch')
   if template.get('request_id')!=job['request_id']:raise ValueError('Root template request differs')
   proof=deepcopy(template['proof']);proof.update(format='s2t24_resource_admission_v1',run_id=run_id,executor_port=self.port,scientific_commit=job['git_commit'])
   from satellite_rl.workbench.t0compare_qualification import qualify
   qualify(job,proof,job['t0_mode'])
   from satellite_rl.workbench.t0compare_resources import observe,reasons
   known=[{'run_id':j['run_id'],'launch':read_json(self.store.job_dir(j['run_id'])/'launch.json',{})} for j,_ in self.store.jobs()]
   actual=observe(self.port,known);proof['observed_utc']=actual['observed_utc'];failed=reasons(actual,proof,self.max_workers)
   output=d/'admission_history'/str(time.time_ns());output.mkdir(parents=True);atomic_json(output/'observation.json',actual)
   if failed:return {'resource_waiting':True,'reasons':failed,'observation_ref':{'path':str(output/'observation.json'),'sha256':digest(output/'observation.json')}}
   proof['resource_observation_ref']={'path':str(output/'observation.json'),'sha256':digest(output/'observation.json')};proof['root_template_ref']=template_ref;atomic_json(output/'proof.json',proof)
   return {'proof_ref':{'path':str(output/'proof.json'),'sha256':digest(output/'proof.json')}}
 def admit(self,run_id,proof_ref,resume=False):
  with self.lock:
   d=self.store.job_dir(run_id);job=read_json(d/'job.json');status=read_json(d/'status.json')
   if not job or job['experiment_id']!='S2-T24':raise ValueError('not own exact task')
   if status['stage'] not in (('stopped','failed') if resume else ('queued_resources',)):raise ValueError('stage not eligible; no completed/attention replay')
   if resume and job['t0_mode']=='engineering':raise ValueError('engineering partial needs explicit budget audit; never restart512 blindly')
   proof=read_json(proof_ref['path'])
   if digest(proof_ref['path'])!=proof_ref['sha256']:raise ValueError('admission rawproof SHA changed')
   from satellite_rl.workbench.t0compare_qualification import qualify
   nested=qualify(job,proof,job['t0_mode']);atomic_json(d/'nested_qualification_checked.json',nested)
   # Frozen parent-coordinated release must rule out oldcontroller future work, not GPUidle.
   if proof.get('format')!='s2t24_resource_admission_v1' or proof.get('run_id')!=run_id or proof.get('executor_port')!=self.port or proof.get('scientific_commit')!=job['git_commit']:raise ValueError('admission job/source identity differs')
   required_flags=('source_qualified','data_qualified','uniform_request_or_explicit_historical_reproduction_authority','old_owner_release_verified','old_future_reservations_excluded','gpu_and_worker_actual_checked','unique_owner_verified') if job['t0_mode']=='engineering' else ('source_qualified','data_qualified','engineering_qualified','uniform_request_or_explicit_historical_reproduction_authority','old_owner_release_verified','old_future_reservations_excluded','gpu_and_worker_actual_checked','unique_owner_verified')
   for key in required_flags:
    if proof.get(key) is not True:raise ValueError('resource/source waiting: '+key)
   from datetime import datetime,timezone
   stamp=datetime.fromisoformat(proof['observed_utc'].replace('Z','+00:00'));age=(datetime.now(timezone.utc)-stamp).total_seconds()
   if not 0<=age<=60:raise ValueError('admission fresh observation required')
   released=proof['old_owner_handoff_ref'];release=read_json(released['path'])
   if digest(released['path'])!=released['sha256'] or self.port not in release.get('released_ports',[]) or release.get('persistent_exclusion_verified') is not True:raise ValueError('oldowner handoff receipt not valid')
   engineering=proof['concurrency_engineering_ref'];gate=read_json(engineering['path'])
   if job['t0_mode']=='engineering' and gate.get('format')!='s2t24_engineering_bootstrap_gate_v1':raise ValueError('actual root-approved boundedengineering bootstrap gate required')
   if digest(engineering['path'])!=engineering['sha256'] or gate.get('passed') is not True or gate.get('max_workers',0)<self.max_workers or gate.get('science_budget_changed') is not False:raise ValueError('concurrency gate missing')
   self.children={k:v for k,v in self.children.items() if v.poll() is None}
   from satellite_rl.workbench.t0compare_resources import observe,reasons
   known=[{'run_id':j['run_id'],'launch':read_json(self.store.job_dir(j['run_id'])/'launch.json',{})} for j,_ in self.store.jobs()]
   actual=observe(self.port,known);atomic_json(d/'resource_before_launch.json',actual)
   failed=reasons(actual,proof,self.max_workers)
   if failed:raise ValueError('resource_waiting '+','.join(failed))
   if len(self.children)>=self.max_workers:raise ValueError('resource_waiting dedicated lane worker slots occupied')
   launch=read_json(d/'launch.json',{})
   if launch and process_start_time(launch['pid'])==launch['process_start_time']:raise ValueError('same previous worker identity stillalive')
   atomic_json(d/'resource_admission.json',proof)
   if job['t0_mode']=='prepare':atomic_json(d/'data_preparation_admission.json',{'approved':True,'owner_run_id':run_id,'frozen_source_sha':job['data_config']['frozen_source_sha'],'actual_resource_proof_sha256':digest(d/'resource_admission.json'),'actual_qualification_refs':proof.get('qualification_refs',[]),'Direct_only':True})
   stop_name='stop.request' if job['t0_mode']=='train' else 'stop.flag'
   if (d/stop_name).exists():(d/stop_name).rename(d/(stop_name+'.history.'+str(time.time_ns())))
   latest=read_json(d/'status.json')
   if latest['stage']!=status['stage'] or latest.get('cancelled_before_worker'):raise ValueError('task state changed/cancelled before launch')
   env={**os.environ,'PYTHONPATH':str(Path(job['source_root'])/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};cmd=[__import__('sys').executable,'-B','-u','-m','satellite_rl.workbench.t0compare_runner','--job-dir',str(d)]+(['--resume'] if resume else [])
   atomic_json(d/'status.json',{**latest,'stage':'starting','admission_sha256':digest(d/'resource_admission.json')})
   try:
    with (d/'worker.stdout.log').open('ab') as out,(d/'worker.stderr.log').open('ab') as err:child=subprocess.Popen(cmd,cwd=job['source_root'],env=env,stdout=out,stderr=err,start_new_session=True)
   except Exception as error:
    atomic_json(d/'launch_failure.json',{'error_type':type(error).__name__,'error':str(error),'at':now(),'worker_created':False});atomic_json(d/'status.json',{**latest,'stage':'attention_required','error':'worker publication launch failed; explicit audit required'});raise
   birth=process_start_time(child.pid);assert birth is not None;self.children[run_id]=child;atomic_json(d/'launch.json',{'pid':child.pid,'process_start_time':birth,'command':cmd,'admission_sha256':digest(d/'resource_admission.json'),'at':now()});return {'job':job,'status':read_json(d/'status.json')}
 def stop(self,run_id):
  with self.lock:
   d=self.store.job_dir(run_id);status=read_json(d/'status.json')
   if status.get('stage') in ('completed','failed','attention_required','cancelled'):raise ValueError('terminal task not stoppable')
   job=read_json(d/'job.json')
   if status.get('stage')=='queued_resources':atomic_json(d/'status.json',{**status,'stage':'cancelled','cancelled_before_worker':True});return {'status':read_json(d/'status.json')}
   stop_name='stop.request' if job['t0_mode']=='train' else 'stop.flag';(d/stop_name).write_text(now());return {'status':status,'stop_at':'complete_episode_checkpoint' if job['t0_mode']=='train' else 'complete_rollout_control'}

class Handler(BaseHTTPRequestHandler):
 def json(self,value,status=200):
  b=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  if self.path=='/api/t0compare/schema':self.json({'phase':'Direct','algorithms':list(ALGORITHMS),'training_steps_each':150000,'seed':51001,'refinement_allowed':False});return
  if self.path=='/api/t0compare/tasks':self.json(self.server.app.tasks());return
  if self.path.startswith('/api/t0compare/task/'):
   self.json(self.server.app.detail(self.path.split('/')[-1]));return
  if self.path=='/':
   text="""<html><meta charset="utf-8"><h1>S2-T24 T0基础三算法 Direct 工作台</h1><p>150000×3，seed51001。资源等待与来源门槛未通过不会启动。第三阶段拒绝。</p><pre id="tasks"></pre><textarea id="body" rows="14" cols="90"></textarea><button onclick="submit()">发起冻结任务</button><input id="run" placeholder="实际run_id"><button onclick="control('stop')">停止</button><button onclick="control('resume')">同ID恢复</button><button onclick="control('admit')">资源准入</button><script>async function control(mode){let body=JSON.parse(document.getElementById("body").value||"{}");body.run_id=document.getElementById("run").value;let r=await fetch("/api/t0compare/"+mode,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});document.getElementById("tasks").textContent=JSON.stringify(await r.json(),null,2)}async function submit(){let r=await fetch("/api/t0compare/submit",{method:"POST",headers:{"Content-Type":"application/json"},body:document.getElementById("body").value});document.getElementById("tasks").textContent=JSON.stringify(await r.json(),null,2)}setInterval(async()=>{let r=await fetch("/api/t0compare/tasks");document.getElementById("tasks").textContent=JSON.stringify(await r.json(),null,2)},5000)</script></html>""";b=text.encode();self.send_response(200);self.send_header('Content-Type','text/html;charset=utf-8');self.end_headers();self.wfile.write(b);return
  self.json({'error':'not found'},404)
 def do_POST(self):
  try:
   body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));reject_third_stage(body);path=self.path
   if path=='/api/t0compare/submit':result=self.server.app.submit(body)
   elif path=='/api/t0compare/admission-proof':result=self.server.app.admission_proof(body['run_id'],body['template_ref'])
   elif path=='/api/t0compare/admit':result=self.server.app.admit(body['run_id'],body['proof_ref'])
   elif path=='/api/t0compare/resume':result=self.server.app.admit(body['run_id'],body['proof_ref'],True)
   elif path=='/api/t0compare/stop':result=self.server.app.stop(body['run_id'])
   else:raise ValueError('Direct-only endpoint unknown')
   self.json(result)
  except Exception as e:self.json({'error_type':type(e).__name__,'error':str(e)},400)
def main():
 p=argparse.ArgumentParser();p.add_argument('--ssh-port',type=int,choices=PORTS,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--reference-config',type=Path,required=True);p.add_argument('--design-contract',type=Path,required=True);p.add_argument('--max-workers',type=int,default=1);a=p.parse_args();app=Workbench(a.ssh_port,a.source_root,read_json(a.reference_config),read_json(a.design_contract),a.max_workers);server=ThreadingHTTPServer(('127.0.0.1',PORTS[a.ssh_port]),Handler);server.app=app;server.serve_forever()
if __name__=='__main__':main()
