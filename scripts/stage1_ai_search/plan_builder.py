"""Expand a pinned science design into bounded contracts; never submits tasks."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
from satellite_rl.coverage.ai_search.config import GeometryConfig,SearchConfig
from satellite_rl.training.run_monitor import atomic_json
from . import ops


def identity_inventory(source_inventory):
    """Bounded read-only identities from three SHA-bound packed NPZs."""
    refs={r:{k:v[k] for k in ('path','sha256','case_count','mother_count')} for r,v in source_inventory['sources'].items()}
    code=f'''import hashlib,json,numpy as np
refs={refs!r};rows={{}}
for role,ref in refs.items():
 h=hashlib.sha256()
 with open(ref['path'],'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 if h.hexdigest()!=ref['sha256']:raise ValueError('original packed SHA differs')
 with np.load(ref['path'],allow_pickle=False) as z:
  rows[role]={{'case_ids':z['case_ids'].astype(str).tolist(),'mother_ids':z['mother_ids'].astype(str).tolist(),'profiles':z['demand_profiles'].astype(str).tolist(),'sha256':ref['sha256']}}
print(json.dumps(rows))
'''
    result=ops.remote_python(code,20424,timeout=45)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1000:])
    rows=json.loads(result['stdout'].splitlines()[-1])
    for role,record in rows.items():
        if len(record['case_ids'])!=refs[role]['case_count'] or len(set(record['mother_ids']))!=refs[role]['mother_count']:
            raise ValueError('actual packed case/mother counts differ')
        for mother in set(record['mother_ids']):
            profiles=[p for m,p in zip(record['mother_ids'],record['profiles']) if m==mother]
            if set(profiles)!={'base','one_hotspot','two_hotspots','edge_heavy'} or len(profiles)!=4:
                raise ValueError('mother does not have exactly four common profiles')
    return rows


def base_config(name,role,ref,geometry,search,workers):
    return json.loads(json.dumps({'schema_version':9,'research_stage':'coverage','algorithm':'ai_search','name':name,
        'geometry':geometry,'search':search,'datasets':{role:ref},'workers':workers,
        'resources':{'min_free_disk_bytes':8*(1<<30),'max_rss_bytes':4*(1<<30)}}))


def audit_plan(plan,identities):
    rows=[];keys=set();mother_lanes={};counts={};preparations=[]
    for task_key,t in plan['tasks'].items():
        role=t['dataset_role'];mode=t['mode']
        if mode=='dataset':
            spec=t['dataset_spec']
            if spec['kind']=='official_day':
                preparations.append({'task_key':task_key,'date':spec['date'],'role':role,
                    'mother_count':spec['count'],'profile_case_count':spec['count']*4,
                    'port':t['executor_port'],'optimization_calls':0})
            continue
        identity=identities[role]
        for index in t['case_indices']:
            mother=identity['mother_ids'][index];p=identity['profiles'][index]
            lane_key=(role,mode,mother)
            previous=mother_lanes.setdefault(lane_key,t['executor_port'])
            if previous!=t['executor_port']:raise ValueError('mother profiles split across physical lanes')
            for method in t['methods']:
                key=(role,mode,method,index)
                if key in keys:raise ValueError('duplicate semantic trajectory')
                keys.add(key);rows.append({'task_key':task_key,'role':role,'mode':mode,'method':method,
                    'case_index':index,'case_id':identity['case_ids'][index],'mother_id':mother,
                    'profile':p,'port':t['executor_port'],'seconds_cap':t['config']['search']['seconds'],
                    'candidate_cap':t['config']['search']['max_attempts']})
                counts[mode]=counts.get(mode,0)+1
    expected={'development':480,'bridge':320,'reference':80,'teacher':768}
    for mode,count in counts.items():
        if mode in expected and count!=expected[mode]:raise ValueError('incomplete frozen '+mode+' matrix')
    for mode in ('development','bridge','reference'):
        if mode in counts:
            if {r['case_index'] for r in rows if r['mode']==mode}!=set(range(80)):
                raise ValueError('development mode has missing/extra cases')
    if 'teacher' in counts:
        selected={r['mother_id'] for r in rows if r['mode']=='teacher'}
        if selected!=set(plan['teacher_mother_order']) or len(selected)!=192:
            raise ValueError('teacher selected mother/ordinal coverage differs')
    if preparations:
        expected_dates={'2024-02-17':('validation',10),'2024-02-18':('validation',10),
                        '2024-02-19':('test',20),'2024-02-20':('test',20)}
        found={r['date']:(r['role'],r['mother_count']) for r in preparations}
        if found!=expected_dates or len(preparations)!=4:raise ValueError('four formal date preparations differ from R1')
    return {'passed':True,'trajectory_counts':counts,'expanded_rows':rows,'dataset_preparations':preparations,
            'unique_trajectories':len(keys),'mother_profiles_stay_together':True,
            'AI_development_80_is_separate_gated_contract':True}


def build(commit,design_ref,source_inventory,identities,modes=('development','bridge','reference'),ports=(20418,20424,20473)):
    if len(commit)!=40 or set(commit)-set('0123456789abcdef'):raise ValueError('exact scientific commit required')
    design_path=Path(design_ref['path']);data=design_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=design_ref['sha256']:raise ValueError('science design SHA differs')
    design=json.loads(data)
    if design.get('group')!='S1-AI17':raise ValueError('wrong scientific design')
    if set(ports)-{20418,20424,20473}:raise ValueError('first execution epoch uses only three initially free lanes')
    geometry=asdict(GeometryConfig());search=asdict(SearchConfig())
    expected={'seconds':design['search']['online_seconds'],'max_attempts':design['search']['max_complete_candidates'],
        'pool_size':design['search']['tasks_per_pool'],'task_candidate_limit':design['search']['complete_candidates_per_task'],
        'initial_seconds':design['search']['initialization_seconds'],'initial_max_attempts':design['search']['initialization_candidates'],
        'max_geometry_calls':design['search']['max_geometry_calls'],'archive_limit':design['archive']['max_layouts']}
    if any(search[k]!=v for k,v in expected.items()) or list(search['region_sizes'])!=design['search']['region_sizes']:
        raise ValueError('implemented SearchConfig differs from frozen science design')
    if list(geometry['width_menu_deg'])!=design['geometry']['full_width_menu_deg']:
        raise ValueError('implemented width menu differs')
    refs={role:{'path':row['path'],'sha256':row['sha256'],'role':role,'count':row['case_count'],
                'mother_count':row['mother_count'],'format_version':row['format_version']}
          for role,row in source_inventory['sources'].items()}
    plan={'group':'S1-AI17','round':'R1','authorized':True,'scientific_commit':commit,
          'design':{'path':ops.BASE+'/source/round1_design.json','sha256':design_ref['sha256']},
          'created_utc':datetime.now(timezone.utc).isoformat(),'tasks':{},
          'source_boundary':'three pinned position/request/identity packed inputs; prior teacher/results/peer paths not recursively used'}
    selected=[]
    if 'teacher' in modes:
        all_mothers=sorted(set(identities['train']['mother_ids']))
        if len(all_mothers)!=576:raise ValueError('training reserve must be original 576 mothers')
        selected=[all_mothers[int((i+.5)*len(all_mothers)/192)] for i in range(192)]
        plan['teacher_mother_order']=selected
        plan['teacher_selection_formula']='floor((i+0.5)*576/192) on sorted frozen mother IDs before outcomes'
    methods={'development':['INIT','RULE','RANDOM','N_ONLY','W_ONLY','POST_SHRINK'],
             'bridge':['07-CV','07-LEX','JOINT-CV','JOINT-LEX'],'reference':['RULE'],'teacher':['RULE']}
    for mode in modes:
        if mode not in methods:raise ValueError('mode requires a separately frozen candidate/gate contract')
        role='train' if mode=='teacher' else 'development';identity=identities[role]
        mothers=selected if mode=='teacher' else sorted(set(identity['mother_ids']))
        if mode!='teacher' and len(mothers)!=20:raise ValueError('Jan12 development must retain 20 whole mothers')
        position={m:i for i,m in enumerate(mothers)}
        cfg_search=dict(search)
        if mode=='bridge':cfg_search.update(seconds=design['budgets']['bridge_seconds'],max_attempts=design['budgets']['bridge_complete_candidates'])
        if mode=='reference':cfg_search.update(seconds=design['budgets']['reference_seconds'],max_attempts=design['budgets']['reference_complete_candidates'])
        if mode=='teacher':cfg_search.update(seconds=design['teacher']['seconds_per_case'],
            max_attempts=design['teacher']['states_per_case_max']*design['teacher']['tasks_per_state']*design['teacher']['complete_candidates_per_task'])
        for lane_index,port in enumerate(ports):
            indices=[i for i,m in enumerate(identity['mother_ids']) if m in position and position[m]%len(ports)==lane_index]
            key=f'r1_{mode}_{port}'
            plan['tasks'][key]={'config':base_config(key,role,refs[role],geometry,cfg_search,1 if mode=='bridge' else 4),
                'mode':mode,'dataset_role':role,'case_indices':indices,'methods':methods[mode],'executor_port':port}
    audit=audit_plan(plan,identities)
    return plan,audit


def add_official_preparations(plan,metadata,ports=(20418,20424,20473)):
    """Separate workbench prepare contracts; no test optimizer/score access."""
    geometry=asdict(GeometryConfig());search=asdict(SearchConfig())
    for i,date in enumerate(('2024-02-17','2024-02-18','2024-02-19','2024-02-20')):
        role='validation' if i<2 else 'test';count=10 if i<2 else 20;port=ports[i%len(ports)]
        record=metadata['sources'][date];shared=metadata['shared_pins']
        pins={'source_preflight':{'path':ops.BASE+'/source/'+date+'/source_preflight.json',
            'sha256':hashlib.sha256((ops.LOCAL/'source'/date/'source_preflight.json').read_bytes()).hexdigest()},
            'release_api':{'path':ops.BASE+'/source/'+date+'/release_api.json','sha256':record['release_api_sha256']},
            'preferred_releases':{'path':ops.BASE+'/source/PREFERRED_RELEASES.txt','sha256':shared['preferred_releases']['sha256']},
            'license':{'path':ops.BASE+'/source/LICENSE-ODbL.txt','sha256':shared['license']['sha256']}}
        spec={'kind':'official_day','role':role,'date':date,'count':count,'seed':51001,
              'source_instance':record['source_instance'],'raw_shared_root':ops.BASE+'/source/raw/'+date,
              'parser':{'path':ops.BASE+'/source/preparation_reference.py','sha256':shared['parser']['sha256']},
              'archive':{'name':record['archive_name'],'url':record['archive_url'],'asset_id':record['asset_id'],
                         'expected_bytes':record['expected_bytes']},'metadata_pins':pins}
        key='r1_dataset_'+date.replace('-','')
        config=base_config(key,role,{},geometry,search,1);config['datasets']={}
        plan['tasks'][key]={'config':config,'mode':'dataset','dataset_role':role,'case_indices':[],
                           'methods':['INIT'],'executor_port':port,'dataset_spec':spec}
    return plan


def add_training_contract(plan,teacher_manifest,gate,validation_ref,validation_identity,port=20418,ports=(20418,20424,20473)):
    """Bind the one gated model and pre-budget all asynchronous node templates."""
    train_tasks=[task for task in plan['tasks'].values() if task['mode']=='teacher']
    if not train_tasks:raise ValueError('training contract needs original teacher input/source configuration')
    config=json.loads(json.dumps(train_tasks[0]['config']));config['name']='r1_train_ranker';config['workers']=1
    config['search']=json.loads(json.dumps(asdict(SearchConfig())))
    config['datasets']['validation']=validation_ref
    # A later training input/model binding plan does not re-publish already
    # completed teacher/development tasks under a different plan hash.
    plan['tasks']={}
    methods=['INIT','RULE','RANDOM','AI','N_ONLY','W_ONLY','POST_SHRINK']
    plan['tasks']['r1_train_ranker']={'config':config,'mode':'train','dataset_role':'train','case_indices':[],
        'methods':['AI'],'executor_port':port,'training_spec':{'teacher_manifest':teacher_manifest,'gate':gate,
            'protocol':{'seed':51001,'updates':10000,'batch':64,'learning_rate':.0003,'weight_decay':.0001,
                        'gradient_clip_norm':1.},'validation_case_indices':list(range(validation_ref['count'])),
            'validation_dispatch':'workbench_async'}}
    if validation_ref['count']!=80 or len(set(validation_identity['mother_ids']))!=20:
        raise ValueError('new validation must retain20 mothers/80 profiles')
    mothers=sorted(set(validation_identity['mother_ids']));ordinal={mother:i for i,mother in enumerate(mothers)}
    templates={};counts={}
    for node in (2500,5000,7500,10000):
        node_tasks={};seen=set()
        # Intermediate nodes avoid the training lane and overlap actual training.
        node_ports=tuple(p for p in ports if p!=port) if node<10000 else ports
        for i,physical in enumerate(node_ports):
            indices=[k for k,m in enumerate(validation_identity['mother_ids']) if ordinal[m]%len(node_ports)==i]
            if seen.intersection(indices):raise ValueError('node repeats validation case')
            seen.update(indices);key=f'r1_node{node}_{physical}'
            cfg=base_config(key,'validation',validation_ref,config['geometry'],config['search'],1)
            node_tasks[key]={'config':cfg,'mode':'diagnostic_validation' if node<10000 else 'validation',
                'dataset_role':'validation','case_indices':indices,'methods':['AI'] if node<10000 else methods,
                'executor_port':physical}
        if seen!=set(range(80)):raise ValueError('node validation case coverage differs')
        templates[str(node)]=node_tasks
        counts[str(node)]=sum(len(t['case_indices'])*len(t['methods']) for t in node_tasks.values())
    if counts!={'2500':80,'5000':80,'7500':80,'10000':560}:raise ValueError('asynchronous node budgets differ')
    plan['node_templates']=templates
    plan['node_budget_audit']={'counts':counts,'extra_intermediate_total':240,'formal_seven_methods':560,
        'diagnostic_four_nodes_total_AI':320,'node10000_AI_is_same_formal80':True,'additional_node10000_forbidden':True}
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    parser.add_argument('--design',required=True);parser.add_argument('--sources',required=True)
    parser.add_argument('--identities');parser.add_argument('--modes',nargs='+',default=['development','bridge','reference'])
    parser.add_argument('--official-metadata')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    source_inventory=json.loads(Path(args.sources).read_text(encoding='utf-8'))
    identities=json.loads(Path(args.identities).read_text(encoding='utf-8')) if args.identities else identity_inventory(source_inventory)
    design=Path(args.design).resolve();ref={'path':str(design),'sha256':hashlib.sha256(design.read_bytes()).hexdigest()}
    plan,audit=build(args.commit,ref,source_inventory,identities,tuple(args.modes))
    if args.official_metadata:
        plan=add_official_preparations(plan,json.loads(Path(args.official_metadata).read_text(encoding='utf-8')))
        audit=audit_plan(plan,identities)
    output=Path(args.output);atomic_json(output,plan);atomic_json(output.with_suffix('.audit.json'),audit)
    atomic_json(output.with_suffix('.identities.json'),identities)
    print(json.dumps({'plan_path':str(output),'audit':{k:v for k,v in audit.items() if k!='expanded_rows'},'science_submitted':0},ensure_ascii=False))
