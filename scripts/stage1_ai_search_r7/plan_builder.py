"""Exact first-wave phase plans, with every external identity supplied by root."""
from copy import deepcopy
import re
from .contract import *

FIELDS = ('config','mode','dataset_role','case_indices','executor_port','methods','model_inputs','model_identity')

def build(commit, phase, inputs, routing=None):
    if not re.fullmatch('[0-9a-f]{40}', commit) or phase not in FIRST_PHASES:
        raise ValueError('frozen commit and implemented first-wave phase required')
    design = require_ref(inputs['design_ref'])
    if design['sha256'] != DESIGN_SHA: raise ValueError('R7 v2 design required')
    model = inputs.get('model_identity')
    if phase in ('train_setup','train2000','node2000') and (not isinstance(model,str) or not model):
        raise ValueError('one nonempty immutable model identity required')
    p = dict(group=GROUP, phase=phase, authorized=True, scientific_commit=commit,
             design=design, identity_ref=require_ref(inputs['identity_ref']),
             prerequisite_gate_ref=require_ref(inputs['prerequisite_gate_ref']),
             qualification_refs={k:require_ref(v) for k,v in inputs['qualification_refs'].items()},
             model_identity=model, training_seed=51001, active_seconds_max=28800,
             global_updates_max=6000, fixed_request_bps=REQUEST_BPS, tasks={},
             budget={k:PHASES[phase][k] for k in ('builds','prepare','T','D5','updates')},
             future_phase_dispatch_allowed=False, automatic_extension=False)
    if not p['qualification_refs']: raise ValueError('actual shared/source qualification required')
    indices = list(range(596)) if phase == 'data_train_dev' else TRAIN_INDICES if phase == 'labels' else DEV_INDICES if phase == 'node2000' else qualified_indices(inputs['qualified_case_indices'])
    route = routing or {str(i):(TRAIN_PORT if phase in ('train_setup','train2000') else PORTS[j%2]) for j,i in enumerate(indices)}
    if set(route)!=set(map(str,indices)) or any(v not in PORTS or (phase in ('train_setup','train2000') and v!=TRAIN_PORT) for v in route.values()):
        raise ValueError('exact73/74 unique routing; model owner74')
    methods = [] if phase in ('data_train_dev','train_setup','train2000') else ['T','D5_EXACT'] if phase=='labels' else ['AI','RULE','RANDOM','T']
    model_inputs={k:require_ref(v) for k,v in inputs.get('model_inputs',{}).items()}
    for port in sorted(set(route.values())):
        name=f'R7_{phase}_{port}'
        cfg=dict(schema_version=16,research_stage='coverage',algorithm='ai_search_r7',name=name,
                 protocol_ref=design,datasets=deepcopy(inputs.get('datasets',{})),workers=1,
                 resources=dict(minimum_free_bytes=8589934592,parent_children_rss_bytes=4294967296),
                 hard_watchdogs=dict(preparation=20,teacher=45,decoder=15,training_active_seconds=28800),case_parallelism=2 if phase in ('data_train_dev','labels') else 1)
        if phase=='train2000': cfg['training']=dict(seed=51001,batch=16,microbatch=4,start_update=0,end_update=2000,checkpoint_interval=100)
        task=dict(config=cfg,mode=phase,dataset_role=PHASES[phase]['role'],
                  case_indices=[i for i in indices if route[str(i)]==port],executor_port=port,
                  methods=methods,model_inputs=deepcopy(model_inputs),model_identity=model)
        p['tasks'][name]=task
    audit(p)
    return p

def audit(p):
    phase=p.get('phase')
    if phase not in FIRST_PHASES or p.get('group')!=GROUP or p.get('fixed_request_bps')!=REQUEST_BPS or p.get('design',{}).get('sha256')!=DESIGN_SHA or p.get('automatic_extension')is not False or p.get('future_phase_dispatch_allowed')is not False:
        raise ValueError('exact implemented R7 v2 phase')
    if (p.get('training_seed'),p.get('global_updates_max'),p.get('active_seconds_max'))!=(51001,6000,28800): raise ValueError('fixed training budget')
    if p['budget']!={k:PHASES[phase][k] for k in ('builds','prepare','T','D5','updates')}: raise ValueError('nontransferable phase cap')
    if not p.get('tasks'): raise ValueError('no tasks')
    indices=[]
    expected_methods=[] if phase in ('data_train_dev','train_setup','train2000') else ['T','D5_EXACT'] if phase=='labels' else ['AI','RULE','RANDOM','T']
    for task in p['tasks'].values():
        cfg=task['config']
        if set(task)!=set(FIELDS) or task['mode']!=phase or task['dataset_role']!=PHASES[phase]['role'] or task['executor_port'] not in PORTS or task['methods']!=expected_methods or task['model_identity']!=p['model_identity']:raise ValueError('task phase/role/owner changed')
        if cfg.get('schema_version')!=16 or cfg.get('workers')!=1 or cfg.get('algorithm')!='ai_search_r7' or cfg.get('research_stage')!='coverage' or cfg.get('protocol_ref')!=p['design']: raise ValueError('single frozen R7 worker')
        if cfg.get('case_parallelism')!=(2 if phase in ('data_train_dev','labels') else 1):raise ValueError('root frozen CPU2 cases; node and learner1')
        if cfg.get('resources')!=dict(minimum_free_bytes=8589934592,parent_children_rss_bytes=4294967296) or cfg.get('hard_watchdogs')!=dict(preparation=20,teacher=45,decoder=15,training_active_seconds=28800):raise ValueError('guard changed')
        if phase in ('train_setup','train2000') and task['executor_port']!=TRAIN_PORT:raise ValueError('one74 model owner')
        if phase=='train2000' and cfg.get('training')!=dict(seed=51001,batch=16,microbatch=4,start_update=0,end_update=2000,checkpoint_interval=100):raise ValueError('fixed2000 updates')
        indices.extend(task['case_indices'])
    target=list(range(596)) if phase=='data_train_dev' else TRAIN_INDICES if phase=='labels' else DEV_INDICES if phase=='node2000' else qualified_indices(sorted(indices))
    if sorted(indices)!=target or len(indices)!=len(set(indices)):raise ValueError('each original case exactly once')
    if phase in ('train_setup','train2000') and len(p['tasks'])!=1:raise ValueError('one learner')
    return dict(passed=True,phase=phase,budget=deepcopy(p['budget']),science_calls_here=0)
