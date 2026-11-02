"""Pure original-phase gate/checkpoint/one-learner-root bindings."""
from .coverage_ai_search import pin
from .storage import read_json

MODEL='S1-AI17-R7-GNNSet-D5-seed51001'
EVALUATIONS={'node2000':('development',20,2000),'fixed6000_development':('development',20,6000),'validation':('validation',20,6000),'test':('test',40,6000)}

def gate_requirements(phase,gate,model_identity,checkpoint_ref=None):
    from scripts.stage1_ai_search_r7.contract import PRIOR
    if gate.get('passed')is not True or gate.get('group')!='S1-AI17-R7' or gate.get('phase')!=PRIOR[phase] or gate.get('model_identity')!=model_identity:raise ValueError('original prior scientific gate/model required')
    if phase in ('train6000','validation','test'):
        if gate.get('denominator')!=20 or gate.get('hard_identity_budget_cost_closed')is not True or gate.get('nonfallback',0)<15 or gate.get('quality_successes',0)<5:raise ValueError('frozen20 scientific mechanism/quality gate')
        if phase in ('validation','test') and gate.get('actual_AI_both_N_width_capability',0)<5:raise ValueError('original >=5 actual joint N/nonfloor-width capability')
        if not checkpoint_ref or gate.get('checkpoint_sha256')!=checkpoint_ref['sha256']:raise ValueError('exact gate fixed checkpoint')
    if phase in ('node2000','fixed6000_development'):
        updates=2000 if phase=='node2000' else 6000
        if gate.get('updates')!=updates or not checkpoint_ref or gate.get('checkpoint_ref')!=checkpoint_ref:raise ValueError('fixed original model endpoint')
    return True

def continuation(endpoint,refs,model_identity,indices,dataset_ref):
    if endpoint.get('phase')!='train2000' or endpoint.get('passed')is not True or endpoint.get('updates')!=2000 or endpoint.get('model_identity')!=model_identity or model_identity!=MODEL:raise ValueError('only original fixed2000 sole model continuation')
    if endpoint.get('checkpoint_ref')!=refs.get('checkpoint') or endpoint.get('train_setup_endpoint_ref')!=refs.get('train_setup_endpoint') or endpoint.get('dataset_ref')!=dataset_ref or endpoint.get('case_indices')!=indices:raise ValueError('same original2000/setup/training table/normalization source')
    if endpoint.get('training_refs')!={k:refs.get(k) for k in endpoint.get('training_refs',{})}:raise ValueError('all original setup refs frozen through continuation')
    from .r7_train_binding import TRAIN_KEYS
    if set(endpoint.get('training_refs',{}))!=set(TRAIN_KEYS):raise ValueError('complete original2000 nine input refs')
    if not endpoint.get('model_dir') or not endpoint.get('run_id') or endpoint.get('original_learner_run_id')!=endpoint['run_id']:raise ValueError('original unique learner root/run identity')
    if endpoint.get('science_updates_reserved')!=2000 or not 0<=endpoint.get('active_seconds',-1)<=28800:raise ValueError('original completed2000 ledger/8h charge')
    original_job=read_json(pin(endpoint['original_job_ref']),{})
    if original_job.get('run_id')!=endpoint['run_id'] or original_job.get('model_dir')!=endpoint['model_dir'] or original_job.get('executor_port')!=20474 or original_job.get('model_identity')!=model_identity or original_job.get('mode')!='train2000':raise ValueError('actual original2000 job root/sole learner identity')
    return endpoint['model_dir']

def evaluation_binding(phase,refs,model_identity):
    if phase not in EVALUATIONS or model_identity!=MODEL:raise ValueError('reserved one model/evaluation phase')
    updates=EVALUATIONS[phase][2]
    endpoint=read_json(pin(refs['training_endpoint']),{});setup=read_json(pin(refs['train_setup_endpoint']),{})
    if endpoint.get('phase')!=('train2000' if updates==2000 else 'train6000') or endpoint.get('passed')is not True or endpoint.get('updates')!=updates or endpoint.get('checkpoint_ref')!=refs['checkpoint'] or endpoint.get('model_identity')!=model_identity or endpoint.get('train_setup_endpoint_ref')!=refs['train_setup_endpoint']:raise ValueError('fixed model endpoint/setup/checkpoint chain')
    if endpoint.get('active_seconds',28801)>28800 or endpoint.get('science_updates_reserved',6001)>6000:raise ValueError('fixed model actual cumulative activity/update budget')
    if setup.get('passed')is not True or setup.get('model_identity')!=model_identity:raise ValueError('original single train setup')
    for key in ('normalization','initialization_receipt','source_equivalence'):
        if refs.get(key)!=setup['training_refs'][key] or endpoint.get('training_refs',{}).get(key)!=refs[key]:raise ValueError('frozen train-only normalization/qualified model source')
    return updates
