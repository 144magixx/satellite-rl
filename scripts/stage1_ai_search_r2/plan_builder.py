"""Mechanical R2 pilot expansion from exact design and existing R1 signatures."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,hashlib,json
from pathlib import Path
from satellite_rl.coverage.ai_search.config import GeometryConfig
from satellite_rl.coverage.ai_search_r2.config import TaskConfig
from satellite_rl.training.run_monitor import atomic_json
from . import ops


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def source_inventory(design_path,identities_path,audit_path,catalog_path,phase='pilot'):
    design=json.loads(Path(design_path).read_text(encoding='utf-8'));identities=json.loads(Path(identities_path).read_text(encoding='utf-8'))['development']
    audited=json.loads(Path(audit_path).read_text(encoding='utf-8'));catalog=json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    if not audited.get('passed') or not audited.get('formal_ready'):raise ValueError('accepted closed R1 audit required')
    reverse={str(Path(v['path']).resolve()):(original,v) for original,v in catalog.items()}
    mothers=sorted(set(identities['mother_ids']),key=lambda m:(datetime.fromisoformat(m.split(':',1)[1].replace('Z','+00:00')),m))
    pilot=design['input']['pilot_mother_indices_zero_based']
    if phase not in ('pilot','extension'):raise ValueError('finite pilot or extension only')
    selected_mother_indices=pilot if phase=='pilot' else [i for i in range(20) if i not in pilot]
    chosen=[mothers[i] for i in selected_mother_indices]
    selected=[i for i,m in enumerate(identities['mother_ids']) if m in set(chosen)]
    expected_mothers=6 if phase=='pilot' else 14
    if len(chosen)!=expected_mothers or len(selected)!=expected_mothers*4:raise ValueError('exact whole mother/profile scope required')
    wanted=set(design['common_start']['source_methods']);refs={str(i):[] for i in selected}
    case_indices={identities['case_ids'][i]:i for i in selected}
    for job in audited['jobs']:
        for entry in job.get('case_audit_index',[]):
            if entry['case_id'] not in case_indices or entry['method'] not in wanted:continue
            if (entry['method']=='POST_SHRINK' and entry['mode']!='development') or (entry['method']!='POST_SHRINK' and entry['mode']!='bridge'):continue
            if digest(entry['path'])!=entry['sha256']:raise ValueError('individual source audit changed')
            report=json.loads(Path(entry['path']).read_text(encoding='utf-8'))
            if not report['passed'] or report['issues']:raise ValueError('source case audit not passed')
            original,stored=reverse[str(Path(report['source_case_path']).resolve())]
            if stored['sha256']!=report['source_case_sha256']:raise ValueError('canonical raw/audit signature differs')
            finals=[p for p in report['points'] if p['kind']=='final']
            if len(finals)!=1:raise ValueError('one actual final source point is required')
            index=case_indices[entry['case_id']]
            refs[str(index)].append({'path':original,'sha256':stored['sha256'],'method':entry['method'],
                'mode':entry['mode'],'case_id':entry['case_id'],'input_sha256':report['input_sha256'],
                'source_layout_sha256':finals[0]['sha256'],'audit_path':ops.BASE+'/source/R1_case_audits/'+entry['sha256']+'.json',
                'audit_sha256':entry['sha256'],'audit_local_canonical':entry['path'],
                'construction_cost_ref':{'path':original,'sha256':stored['sha256'],'field':'end_to_end_seconds',
                    'scope':'actual historical R1 acquisition cost; not free and not rerun'},
                'prior_audit_manifest':{'path':str(audit_path),'sha256':digest(audit_path)}})
    order={method:i for i,method in enumerate(design['common_start']['source_methods'])}
    for key,items in refs.items():
        if len(items)!=5 or {r['method'] for r in items}!=wanted:raise ValueError('pilot strong source set incomplete/extra')
        items.sort(key=lambda r:order[r['method']])
    return {'group':ops.GROUP,'phase':phase,'development_identities':identities,'mother_order':mothers,'selected_mothers':chosen,'selected_case_indices':selected,
        'source_refs':refs,'raw_R1_solver_calls':0,'new_layouts_constructed':0,
        'design_ref':{'path':str(design_path),'sha256':digest(design_path)},
        'prior_audit_ref':{'path':str(audit_path),'sha256':digest(audit_path)}}


def prepare_plan(commit,design_path,inventory,dataset_ref,extension_admission=None):
    design=json.loads(Path(design_path).read_text(encoding='utf-8'))
    if design['group']!=ops.GROUP or design['execution']['case_workers_per_physical_lane']!=1:raise ValueError('R2 design/group/timing differs')
    config={'schema_version':10,'research_stage':'coverage','algorithm':'ai_search_r2','name':'R2 common start source preparation',
        'geometry':asdict(GeometryConfig()),'task':asdict(TaskConfig()),'datasets':{'development':dataset_ref},
        'workers':1,'resources':{'minimum_free_bytes':design['outputs']['minimum_free_bytes'],'max_rss_bytes':4*(1<<30)}}
    config=json.loads(json.dumps(config))
    phase=inventory.get('phase','pilot')
    plan={'group':ops.GROUP,'round':'R2-'+phase,'authorized':True,'scientific_commit':commit,
        'training_allowed':False,'test_allowed':False,'design':{'path':ops.BASE+'/source/round2_design.json','sha256':digest(design_path)},
        'phase':phase,'mother_order':inventory.get('mother_order',inventory['selected_mothers']),
        'selected_mothers':inventory['selected_mothers'],'selected_case_indices':inventory['selected_case_indices'],
        'tasks':{},'source_boundary':'selected R1 source raw+case audits read only; common prep no solver; no val/test'}
    ordinal={m:i for i,m in enumerate(inventory['selected_mothers'])};identity=inventory['development_identities']
    if phase=='extension':
        if not extension_admission:raise ValueError('remaining56 requires root extension admission bundle')
        plan['extension_admission']=deepcopy(extension_admission)
        plan['scientific_core_sha256']=extension_admission['scientific_core_sha256']
        plan['scientific_dependency_sha256']=extension_admission['scientific_dependency_sha256']
        from satellite_rl.workbench.ai_search_r2_extension import PILOT_COMMIT
        config['admission']={'phase':'extension','algorithm_core_commit':PILOT_COMMIT,'root_gate':extension_admission['root_gate']}
    for lane,port in enumerate(ops.SSH_PORTS):
        indices=[i for i in inventory['selected_case_indices'] if ordinal[identity['mother_ids'][i]]%3==lane]
        key=f'R2_'+('extension_' if phase=='extension' else '')+f'prepare_common_{port}'
        plan['tasks'][key]={'config':{**deepcopy(config),'name':key},'mode':'prepare_common','dataset_role':'development',
            'case_indices':indices,'executor_port':port,'variants':[],
            'source_refs':{str(i):[{k:v for k,v in r.items() if k not in ('audit_local_canonical','prior_audit_manifest')}
                for r in inventory['source_refs'][str(i)]] for i in indices},'common_ref':None}
    audit(plan,inventory)
    return plan


def evaluation_plan(prepared_plan,common_refs):
    plan=deepcopy(prepared_plan);plan['tasks']={}
    for key,task in prepared_plan['tasks'].items():
        port=task['executor_port'];new=deepcopy(task);new['mode']='evaluate_closed_pool';new['variants']=['D0','D1']
        name=f'R2_'+('extension_' if plan['phase']=='extension' else '')+f'closed_pool_{port}'
        new['source_refs']={};new['common_ref']=common_refs[str(port)];new['config']['name']=name
        plan['tasks'][name]=new
    return plan


def audit(plan,inventory):
    seen=set();mothers={};tasks=0;mode_set=set()
    identity=inventory['development_identities']
    selected=set(inventory['selected_case_indices'])
    expected_cases=24 if plan.get('phase')=='pilot' else 56 if plan.get('phase')=='extension' else 0
    if len(selected)!=expected_cases or len(plan['tasks'])!=3 or {t['executor_port'] for t in plan['tasks'].values()}!=set(ops.SSH_PORTS):
        raise ValueError('exact finite phase three-lane scope required')
    for task in plan['tasks'].values():
        mode_set.add(task['mode'])
        if task['config']['workers']!=1:raise ValueError('one scientific case worker per lane')
        if task['mode']=='prepare_common' and (task['variants'] or set(task['source_refs'])!={str(i) for i in task['case_indices']}):
            raise ValueError('exact signed common-preparation sources required')
        if task['mode']=='evaluate_closed_pool' and task['variants']!=['D0','D1']:raise ValueError('both exact decoder variants required')
        for index in task['case_indices']:
            if index not in selected:raise ValueError('unselected pilot case')
            mother=identity['mother_ids'][index]
            if task['mode']=='prepare_common':
                refs=task['source_refs'][str(index)]
                if len(refs)!=5 or {r['method'] for r in refs}!={'07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK'}:
                    raise ValueError('each case requires exactly five actual final sources')
            if mothers.setdefault(mother,task['executor_port'])!=task['executor_port']:raise ValueError('mother profiles split across lanes')
            for variant in task['variants'] or ['COMMON']:
                semantic=(task['mode'],index,variant)
                if semantic in seen:raise ValueError('duplicate R2 semantic result')
                seen.add(semantic);tasks+=8 if variant!='COMMON' else 0
    expected=expected_cases if mode_set=={'prepare_common'} else expected_cases*2
    if len(seen)!=expected:raise ValueError('missing pilot result')
    if {index for _,index,_ in seen}!=selected or len(mothers)!=expected_cases//4:raise ValueError('phase identities incomplete')
    if mode_set=={'evaluate_closed_pool'} and tasks!=expected_cases*16:raise ValueError('finite phase task matrix differs')
    return {'passed':True,'cases':expected_cases,'mother_count':expected_cases//4,'result_rows':len(seen),'decoder_task_calls':tasks,
        'training_updates':0,'validation_test_algorithms':0,'same_mother_same_lane':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    inventory=sub.add_parser('inventory');inventory.add_argument('--design',default=str(Path(__file__).with_name('round2_design.json')))
    inventory.add_argument('--phase',choices=('pilot','extension'),default='pilot')
    for name in ('identities','audit','catalog','output'):inventory.add_argument('--'+name,required=True)
    build=sub.add_parser('prepare');build.add_argument('--commit',required=True);build.add_argument('--inventory',required=True)
    build.add_argument('--dataset-ref',required=True);build.add_argument('--output',required=True)
    build.add_argument('--qualification-refs')
    build.add_argument('--extension-admission')
    build.add_argument('--design',default=str(Path(__file__).with_name('round2_design.json')))
    evaluate=sub.add_parser('evaluate')
    for name in ('prepared-plan','common-refs','inventory','output'):evaluate.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.operation=='inventory':result=source_inventory(args.design,args.identities,args.audit,args.catalog,args.phase)
    elif args.operation=='prepare':result=prepare_plan(args.commit,args.design,json.loads(Path(args.inventory).read_text(encoding='utf-8')),json.loads(Path(args.dataset_ref).read_text(encoding='utf-8')),json.loads(Path(args.extension_admission).read_text(encoding='utf-8')) if args.extension_admission else None)
    else:
        result=evaluation_plan(json.loads(Path(args.prepared_plan).read_text(encoding='utf-8')),json.loads(Path(args.common_refs).read_text(encoding='utf-8')))
        audit(result,json.loads(Path(args.inventory).read_text(encoding='utf-8')))
    if getattr(args,'qualification_refs',None):
        refs=json.loads(Path(args.qualification_refs).read_text(encoding='utf-8'))
        result['qualification_refs']={k:{'path':r['path'],'sha256':r['sha256']} for k,r in refs.items()}
    atomic_json(args.output,result);print(json.dumps({'path':args.output,'sha256':digest(args.output),'scientific_tasks_submitted':0}))
