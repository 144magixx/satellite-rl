"""G16 persistent prepare, reference gate, teacher/MLM, common test and archive DAG.

The signed dispatch plan supplies exact official inputs and immutable references.
This coordinator only submits workbench jobs and reads their terminal evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
from . import ops
from .archive import export_job
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_geoload import default_config
from satellite_rl.workbench.geoload_runner import exclusive_lock
from satellite_rl.workbench.storage import digest, now

SAME_N=['BH','07','09-CV','09-LEX','09+SWAP','JOINT']
LEARNING=['MLM','rule','random']

def reference(plan,key,port):
    ref=plan[key][str(port)]
    return {'path':ref['path'],'sha256':ref['sha256']}

def body(plan,key,port,kind,mode,role,datasets=None,methods=None,shard=False,**extra):
    config=default_config();config['name']='S1-G16 '+key;config['workers']=plan.get('workers_per_cpu_task',8)
    config['datasets']=datasets or {}
    value={'request_id':'g16_'+plan['scientific_commit'][:12]+'_'+key,'kind':kind,'mode':mode,
        'dataset_role':role,'methods':methods or ['BH'],'config':config,
        'plan':reference(plan,'science_plan_refs',port),'partition':reference(plan,'partition_refs',port),**extra}
    if shard:
        count=plan['mother_counts'][role];parity=0 if port==20424 else 1
        value['mother_indices']=list(range(parity,count,2))
    return value

def submit(plan,state,key,port,payload):
    tasks=state.setdefault('tasks',{})
    if key in tasks:return tasks[key]
    import hashlib
    expected=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    matches=[row['job'] for row in ops.api(port,'/api/geoload/jobs') if row['job'].get('request_id')==payload['request_id']]
    if matches:
        if len(matches)!=1 or matches[0].get('request_hash')!=expected:
            raise ValueError('original request ownership/configuration differs')
        job=matches[0]
    else:job=ops.api(port,'/api/geoload/jobs',payload)
    tasks[key]={'port':port,'run_id':job['run_id'],'request_id':payload['request_id'],'stage':'queued',
                'kind':payload['kind'],'mode':payload['mode'],'submitted_at':now()}
    return tasks[key]

def observe(state):
    from concurrent.futures import ThreadPoolExecutor
    active=[(k,t) for k,t in state.get('tasks',{}).items() if t.get('stage')!='completed']
    def one(pair):
        key,task=pair
        try:
            result=ops.api(task['port'],'/api/runs/'+task['run_id'])
            return key,{'stage':result['status']['stage'],'status':result['status'],
                        'artifacts':result['artifacts'],'observed_at':now(),'transport_error':None}
        except Exception as error:return key,{'transport_error':str(error),'failed_observation_at':now()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for key,value in pool.map(one,active):state['tasks'][key].update(value)

def done(state,*keys):return all(state.get('tasks',{}).get(k,{}).get('stage')=='completed' for k in keys)

def dataset(task,role):
    record=task['artifacts']
    if record.get('status')!='prepared':raise ValueError('prepared dataset artifact required')
    return {**record,'role':role}

def raw_rows(task):
    info=ops.instance(task['port']);root=info['archive']+'/outputs/workbench/jobs/'+task['run_id']
    code=f'''import hashlib,json,pathlib
root=pathlib.Path({root!r});index=json.loads((root/'case_index.json').read_text());rows=[]
for key,item in index.items():
 p=root/'cases'/(key+'.json');data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=item['sha256']:raise RuntimeError('gate result SHA differs')
 row=json.loads(data);rows.append({{k:row.get(k) for k in ('case_id','mother_id','demand_profile','method','outcome','metrics','reference_nodes')}})
print(json.dumps({{'rows':rows,'index_sha256':hashlib.sha256((root/'case_index.json').read_bytes()).hexdigest()}}))
'''
    result=ops.remote_python(code,task['port'],timeout=60)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1000:])
    return json.loads(result['stdout'].splitlines()[-1])

def make_gate(state,local):
    existing=Path(local)/'development_gate.json'
    if existing.is_file():
        retained=json.loads(existing.read_text(encoding='utf-8-sig'))
        if retained.get('group')!='S1-G16' or retained.get('reference_seconds')!=30:
            raise ValueError('existing gate belongs to different protocol')
        for source in retained.get('sources',[]):
            port=source['port']
            if source.get('dev_run_id')!=state['tasks']['dev_'+str(port)].get('run_id') or source.get('reference_run_id')!=state['tasks']['reference_'+str(port)].get('run_id'):
                raise ValueError('existing gate belongs to different workbench inputs')
        return retained,existing
    baselines={};refs={};sources=[]
    for port in ops.SSH_PORTS:
        dev=raw_rows(state['tasks']['dev_'+str(port)]);reference_rows=raw_rows(state['tasks']['reference_'+str(port)])
        sources.append({'port':port,'dev_index_sha256':dev['index_sha256'],'reference_index_sha256':reference_rows['index_sha256'],
            'dev_run_id':state['tasks']['dev_'+str(port)].get('run_id'),
            'reference_run_id':state['tasks']['reference_'+str(port)].get('run_id')})
        baselines.update({r['case_id']:r for r in dev['rows'] if r['method']=='09-LEX'})
        for row in reference_rows['rows']:
            nodes=row.get('reference_nodes') or []
            terminal=[n for n in nodes if n.get('seconds_limit')==30]
            if len(terminal)!=1:raise ValueError('single30s reference terminal required')
            refs[row['case_id']]={**row,'metrics':terminal[0]['metrics']}
    if len(baselines)!=80 or set(baselines)!=set(refs):raise ValueError('complete80 paired development gate required')
    mothers=sorted({r['mother_id'] for r in baselines.values()});halves=[set(mothers[:10]),set(mothers[10:])]
    if len(mothers)!=20:raise ValueError('gate20 mothers required')
    improved=[];violations=0;rows=[]
    for key,old in baselines.items():
        new=refs[key];a=old['metrics'];b=new['metrics'];valid=bool(b and b['evidence']['valid'])
        violations+=int(not valid)
        qualifies=False
        if valid and a['P'] is not None and a['P']>0:
            qualifies=(a['P']-b['P'])/a['P']>=.01 or (abs(a['P']-b['P'])<=1e-9 and a['T']>0 and (a['T']-b['T'])/a['T']>=.01)
        if qualifies:improved.append(old['mother_id'])
        rows.append({'case_id':key,'mother_id':old['mother_id'],'qualified':qualifies,'baseline':a,'reference30':b})
    fractions=[sum(x in h for x in improved)/40 for h in halves]
    gate={'group':'S1-G16','passed':violations==0 and len(improved)/80>=.2 and min(fractions)>=.1,
        'reference_seconds':30,'improved_case_fraction':len(improved)/80,'half_fractions':fractions,
        'hard_violations':violations,'sources':sources,'rows':rows,'computed_at':now()}
    path=Path(local)/'development_gate.json';atomic_json(path,gate)
    return gate,path

def tick(plan,state,local):
    observe(state);tasks=state.setdefault('tasks',{})
    # First prepare exact mother inputs; daily test files merge in the workbench.
    for key,spec in plan['preparations'].items():
        submit(plan,state,'prepare_'+key,spec['port'],body(plan,'prepare_'+key,spec['port'],'prepare','dataset',spec['role'],dataset_spec=spec['dataset_spec']))
    if not done(state,*['prepare_'+k for k in plan['preparations']]):return
    datasets=state.setdefault('datasets',{})
    for role in ('train','development','validation','test'):
        group=[('prepare_'+k,s) for k,s in plan['preparations'].items() if s['role']==role]
        if len(group)==1:datasets[role]=dataset(tasks[group[0][0]],role)
        else:
            merged_key='merge_'+role
            sources=[{k:tasks[key]['artifacts'][k] for k in ('path','sha256')} for key,_ in group]
            spec={'kind':'merge_existing','role':role,'count':plan['mother_counts'][role],'sources':sources}
            submit(plan,state,merged_key,20424,body(plan,merged_key,20424,'prepare','dataset',role,dataset_spec=spec))
            if not done(state,merged_key):return
            datasets[role]=dataset(tasks[merged_key],role)
    initials=[]
    for role,record in datasets.items():
        for port in ops.SSH_PORTS:
            key='initial_'+role+'_'+str(port);initials.append(key)
            submit(plan,state,key,port,body(plan,key,port,'prepare','initial',role,{role:record},shard=True))
    if not done(state,*initials):return
    devkeys=[]
    for port in ops.SSH_PORTS:
        for mode,methods in [('dev',SAME_N),('reference',['JOINT'])]:
            key=mode+'_'+str(port);devkeys.append(key)
            submit(plan,state,key,port,body(plan,key,port,'evaluate','evaluate' if mode=='dev' else 'reference',
                'development',{'development':datasets['development']},methods,shard=True))
    if not done(state,*devkeys):return
    if 'gate' not in state:
        gate,path=make_gate(state,local);gate_refs={}
        for port in ops.SSH_PORTS:
            target=ops.instance(port)['ops']+'/development_gate.json';receipt=ops.upload(path,target,port)
            if receipt['remote']!=digest(path):raise ValueError('gate transport SHA differs')
            gate_refs[str(port)]={'path':target,'sha256':digest(path)}
        state['gate']=gate;state['gate_refs']=gate_refs
    passed=state['gate']['passed'];model=None
    if passed:
        teacherkeys=[]
        for port in ops.SSH_PORTS:
            key='teacher_'+str(port);teacherkeys.append(key)
            submit(plan,state,key,port,body(plan,key,port,'prepare','teacher','train',{'train':datasets['train']},['JOINT'],shard=True,gate=state['gate_refs'][str(port)]))
        if not done(state,*teacherkeys):return
        teacher_sources=[tasks[k]['artifacts']['teacher'] for k in teacherkeys]
        key='teacher_merge';submit(plan,state,key,20473,body(plan,key,20473,'prepare','teacher_merge','train',
            {'train':datasets['train']},['JOINT'],gate=state['gate_refs']['20473'],teacher_sources=teacher_sources))
        if not done(state,key):return
        submit(plan,state,'train_mlm',20473,body(plan,'train_mlm',20473,'train','train','train',
            {'train':datasets['train'],'validation':datasets['validation']},['JOINT'],
            gate=state['gate_refs']['20473'],teacher=tasks[key]['artifacts']['teacher']))
        # Variable-N branch on24 overlaps73MLM; its73shard waits forMLM.
        submit(plan,state,'variable_20424',20424,body(plan,'variable_20424',20424,'evaluate','variable','test',{'test':datasets['test']},['VARIABLE'],shard=True))
        if not done(state,'train_mlm'):return
        artifact=tasks['train_mlm']['artifacts'];model={'path':artifact['model'],'sha256':artifact['model_sha256']}
    variablekeys=[]
    for port in ops.SSH_PORTS:
        key='variable_'+str(port);variablekeys.append(key)
        submit(plan,state,key,port,body(plan,key,port,'evaluate','variable','test',{'test':datasets['test']},['VARIABLE'],shard=True))
    if not done(state,*variablekeys):return
    valkeys=[]
    methods=SAME_N+(LEARNING if passed else [])
    for port in ops.SSH_PORTS:
        key='common_validation_'+str(port);valkeys.append(key)
        submit(plan,state,key,port,body(plan,key,port,'evaluate','evaluate','validation',{'validation':datasets['validation']},methods,shard=True,**({'model':model} if model else {})))
    if not done(state,*valkeys):return
    testkeys=[]
    for port in ops.SSH_PORTS:
        key='common_test_'+str(port);testkeys.append(key)
        submit(plan,state,key,port,body(plan,key,port,'evaluate','evaluate','test',{'test':datasets['test']},methods,shard=True,**({'model':model} if model else {})))
    if not done(state,*testkeys):return
    state['all_computation_completed']=True
    archives=state.setdefault('archives',{})
    for key,task in tasks.items():
        if key not in archives:
            archive_root=Path(plan.get('canonical_archive_root','E:/paper_archives/outputs/experiments/first_stage_geometry_load_20261012/job_archives'))
            archives[key]=export_job(task['port'],task['run_id'],archive_root/(task['run_id']+'.zip'))
            return
    state.update(compute_and_archives_complete=True,delivery_requires_root_review=True,
                 completed_at=now(),report_status='raw evidence archived; production statistics and human report pending')

def run(plan_path,state_path):
    plan_path=Path(plan_path).resolve();state_path=Path(state_path).resolve()
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    if plan.get('group')!='S1-G16' or plan.get('authorized') is not True:raise ValueError('signed G16 pipeline plan required')
    sha=digest(plan_path)
    state=json.loads(state_path.read_text(encoding='utf-8-sig')) if state_path.exists() else {'group':'S1-G16','plan_sha256':sha,'tasks':{}}
    if state['plan_sha256']!=sha:raise ValueError('persistent pipeline plan changed')
    state.update(scientific_commit=plan['scientific_commit'],execution_ports=list(ops.SSH_PORTS))
    with exclusive_lock(state_path.with_suffix('.lock')):
        while not state.get('compute_and_archives_complete'):
            try:
                tick(plan,state,state_path.parent);state['controller_error']=None
            except Exception as error:
                state['controller_error']={'error':str(error),'at':now(),'scientific_failure':False}
            state['updated_at']=now();atomic_json(state_path,state)
            if state_path.with_suffix('.stop').exists():break
            time.sleep(20)
    return state
