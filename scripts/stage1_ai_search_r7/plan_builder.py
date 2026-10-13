"""Exact first-wave phase plans, with every external identity supplied by root."""
from copy import deepcopy
import re
from .contract import *

FIELDS = ('config','mode','dataset_role','case_indices','executor_port','methods','model_inputs','model_identity')

def build(commit, phase, inputs, routing=None):
    if not re.fullmatch('[0-9a-f]{40}', commit) or phase not in IMPLEMENTED_PHASES:
        raise ValueError('frozen commit and original implemented phase required')
    design = require_ref(inputs['design_ref'])
    if design['sha256'] != DESIGN_SHA: raise ValueError('R7 v2 design required')
    model = inputs.get('model_identity')
    if phase not in ('data_train_dev','labels') and (not isinstance(model,str) or not model):
        raise ValueError('one nonempty immutable model identity required')
    step=inputs.get('heldout_stage') if phase in ('validation','test') else None
    hooks=inputs.get('deadline_guard_refs_by_port',{})
    if phase in SEALED_PHASES and (model!=MODEL_IDENTITY or set(hooks)!=set(map(str,PORTS))):raise ValueError('reserved sole model and root scheduled-pause hooks for successors')
    if phase in SEALED_PHASES and set(inputs.get('archive_roots_by_port',{}))!=set(map(str,PORTS)):raise ValueError('explicit deadline-registered73/74 archive roots required')
    p = dict(group=GROUP, phase=phase, authorized=True, scientific_commit=commit,
             design=design, identity_ref=require_ref(inputs['identity_ref']),
             prerequisite_gate_ref=require_ref(inputs['prerequisite_gate_ref']),
             qualification_refs={k:require_ref(v) for k,v in inputs['qualification_refs'].items()},
             model_identity=model, training_seed=51001, active_seconds_max=28800,
             global_updates_max=6000, fixed_request_bps=REQUEST_BPS, tasks={},
             budget=phase_budget(phase,step),heldout_stage=step,
             deadline_guard_refs_by_port=deepcopy(hooks),archive_roots_by_port=deepcopy(inputs.get('archive_roots_by_port',{})),
             future_phase_dispatch_allowed=False, automatic_extension=False)
    if not p['qualification_refs']: raise ValueError('actual shared/source qualification required')
    indices = list(range(596)) if phase == 'data_train_dev' else TRAIN_INDICES if phase == 'labels' else list(range(PHASES[phase]['cases'])) if phase in EVALUATION_PHASES else qualified_indices(inputs['qualified_case_indices'])
    route = routing or {str(i):(TRAIN_PORT if phase in ('train_setup','train2000','train6000') else PORTS[j%2]) for j,i in enumerate(indices)}
    if set(route)!=set(map(str,indices)) or any(v not in PORTS or (phase in ('train_setup','train2000','train6000') and v!=TRAIN_PORT) for v in route.values()):
        raise ValueError('exact73/74 unique routing; model owner74')
    methods = [] if phase in ('data_train_dev','train_setup','train2000','train6000') or step=='derive' else ['T','D5_EXACT'] if phase=='labels' else ['AI','RULE','RANDOM','T']
    model_inputs={k:require_ref(v) for k,v in inputs.get('model_inputs',{}).items()}
    for port in sorted(set(route.values())):
        name=f'R7_{phase}_{port}'
        cfg=dict(schema_version=16,research_stage='coverage',algorithm='ai_search_r7',name=name,
                 protocol_ref=design,datasets=deepcopy(inputs.get('datasets',{})),workers=1,
                 resources=dict(minimum_free_bytes=8589934592,parent_children_rss_bytes=4294967296),
                 hard_watchdogs=dict(preparation=20,teacher=45,decoder=15,training_active_seconds=28800),case_parallelism=2 if phase in ('data_train_dev','labels') else 1)
        if phase in ('train2000','train6000'): cfg['training']=dict(seed=51001,batch=16,microbatch=4,start_update=0 if phase=='train2000' else 2000,end_update=2000 if phase=='train2000' else 6000,checkpoint_interval=100)
        if step is not None:cfg['heldout_stage']=step
        task=dict(config=cfg,mode=phase,dataset_role=PHASES[phase]['role'],
                  case_indices=[i for i in indices if route[str(i)]==port],executor_port=port,
                  methods=methods,model_inputs=deepcopy(model_inputs),model_identity=model)
        if phase=='data_train_dev' and 'transport_map_refs_by_port' in inputs:
            task['model_inputs']['transport_map']=require_ref(inputs['transport_map_refs_by_port'][str(port)])
        if hooks:
            from satellite_rl.workbench.r7_deadline import KEYS
            if set(hooks)!=set(map(str,PORTS)) or set(hooks[str(port)])!=set(KEYS):raise ValueError('complete per-lane root deadline hooks')
            task['model_inputs'].update({k:require_ref(v) for k,v in hooks[str(port)].items()})
        p['tasks'][name]=task
    audit(p)
    return p

def audit(p):
    phase=p.get('phase')
    if phase not in IMPLEMENTED_PHASES or p.get('group')!=GROUP or p.get('fixed_request_bps')!=REQUEST_BPS or p.get('design',{}).get('sha256')!=DESIGN_SHA or p.get('automatic_extension')is not False or p.get('future_phase_dispatch_allowed')is not False:
        raise ValueError('exact implemented R7 v2 phase')
    if (p.get('training_seed'),p.get('global_updates_max'),p.get('active_seconds_max'))!=(51001,6000,28800): raise ValueError('fixed training budget')
    step=p.get('heldout_stage')
    if p['budget']!=phase_budget(phase,step): raise ValueError('nontransferable phase/substep cap')
    if phase in SEALED_PHASES and (p.get('model_identity')!=MODEL_IDENTITY or set(p.get('deadline_guard_refs_by_port',{}))!=set(map(str,PORTS))):raise ValueError('same reserved model and scheduled-pause hooks')
    if not p.get('tasks'): raise ValueError('no tasks')
    indices=[]
    expected_methods=[] if phase in ('data_train_dev','train_setup','train2000','train6000') or step=='derive' else ['T','D5_EXACT'] if phase=='labels' else ['AI','RULE','RANDOM','T']
    for task in p['tasks'].values():
        cfg=task['config']
        if set(task)!=set(FIELDS) or task['mode']!=phase or task['dataset_role']!=PHASES[phase]['role'] or task['executor_port'] not in PORTS or task['methods']!=expected_methods or task['model_identity']!=p['model_identity']:raise ValueError('task phase/role/owner changed')
        if cfg.get('schema_version')!=16 or cfg.get('workers')!=1 or cfg.get('algorithm')!='ai_search_r7' or cfg.get('research_stage')!='coverage' or cfg.get('protocol_ref')!=p['design']: raise ValueError('single frozen R7 worker')
        if cfg.get('case_parallelism')!=(2 if phase in ('data_train_dev','labels') else 1):raise ValueError('root frozen CPU2 cases; node and learner1')
        if cfg.get('resources')!=dict(minimum_free_bytes=8589934592,parent_children_rss_bytes=4294967296) or cfg.get('hard_watchdogs')!=dict(preparation=20,teacher=45,decoder=15,training_active_seconds=28800):raise ValueError('guard changed')
        if phase in ('train_setup','train2000','train6000') and task['executor_port']!=TRAIN_PORT:raise ValueError('one74 model owner')
        if phase=='train2000' and cfg.get('training')!=dict(seed=51001,batch=16,microbatch=4,start_update=0,end_update=2000,checkpoint_interval=100):raise ValueError('fixed2000 updates')
        if phase=='train6000' and cfg.get('training')!=dict(seed=51001,batch=16,microbatch=4,start_update=2000,end_update=6000,checkpoint_interval=100):raise ValueError('original2000 to cumulative6000')
        if phase in ('validation','test') and cfg.get('heldout_stage')!=step:raise ValueError('heldout build/evaluation substep identity')
        if p.get('deadline_guard_refs_by_port'):
            from satellite_rl.workbench.r7_deadline import KEYS
            hooks=p['deadline_guard_refs_by_port'][str(task['executor_port'])]
            if set(hooks)!=set(KEYS) or any(task['model_inputs'].get(k)!=v for k,v in hooks.items()):raise ValueError('each actual phase task carries exact lane deadline hooks')
        indices.extend(task['case_indices'])
    target=list(range(596)) if phase=='data_train_dev' else TRAIN_INDICES if phase=='labels' else list(range(PHASES[phase]['cases'])) if phase in EVALUATION_PHASES else qualified_indices(sorted(indices))
    if sorted(indices)!=target or len(indices)!=len(set(indices)):raise ValueError('each original case exactly once')
    if phase in ('train_setup','train2000','train6000') and len(p['tasks'])!=1:raise ValueError('one learner')
    return dict(passed=True,phase=phase,budget=deepcopy(p['budget']),science_calls_here=0)

def data_plan(commit,inputs,routing=None):
    """The production data phase requires the exact root recipe and transport."""
    if inputs.get('model_inputs',{}).get('recipe',{}).get('sha256')!=UNIFORM_RECIPE_SHA:raise ValueError('frozen596 root recipe349370 required')
    maps=inputs.get('transport_map_refs_by_port',{})
    if set(maps)!=set(map(str,PORTS)):raise ValueError('root pinned independent73/74 source transport maps required')
    for reference in maps.values():require_ref(reference)
    return build(commit,'data_train_dev',inputs,routing)
