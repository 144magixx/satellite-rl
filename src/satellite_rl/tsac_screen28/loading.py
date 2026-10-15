"""Production pinned-input and actor-only loading. Nothing executes on import."""
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from satellite_rl.t0_width.data import read_json_ref,file_sha,load_published_layout
from satellite_rl.t0_width.logging import content_sha

POLICY_SHA='5e056ee244a79c05aea8e26e4b7800fd0395e9bac45302fe4ba35d0e5f7bd547'

def load_policy(plan):
    """Build only the frozen actor, no critic/replay/optimizer or model update."""
    import torch
    from satellite_rl.agents.workbench_networks import MultiHeadNetwork
    from satellite_rl.t0_width.agent import WidthEncoder,schema_identity
    from satellite_rl.tsac_search27.known_evaluation import validate_policy_metadata
    started=perf_counter();ref=plan['policy_ref']
    if ref['sha256']!=POLICY_SHA or file_sha(ref['path'])!=POLICY_SHA:raise ValueError('exact protected W26 S10/150000 original required')
    config=read_json_ref(plan['producer_config_ref']);job=read_json_ref(plan['producer_job_ref']);original=read_json_ref(plan['producer_plan_ref'])
    configs=[t for t in original['tasks'].values() if t.get('mode')=='train' and t['config'].get('group')=='S10']
    if len(configs)!=1:raise ValueError('original unique S10 producer required')
    effective=dict(configs[0]['config'],run_id=config['run_id'],model_dir=config['model_dir'],executor_port=configs[0]['executor_port'])
    if (config!=effective or job['config_sha256']!=content_sha(config)
            or job['status']['plan_sha256']!=plan['producer_plan_ref']['sha256']
            or job['status']['run_id']!=config['run_id'] or job['source_manifest']!=original['source_manifest']
            or job['status']['model_dir']!=config['model_dir'] or job['source_root']!=original['source_root']):raise ValueError('actual original W26 producer derivation differs')
    raw=torch.load(ref['path'],map_location='cpu',weights_only=False)
    provenance=dict(schema=schema_identity('S10'),env_step=150000,config_SHA=content_sha(config),source_SHA=config['source_sha'],data_manifest_SHA=config['data_manifest_sha'])
    validate_policy_metadata(raw,provenance,schema_identity('S10'))
    device=torch.device(plan.get('device','cuda'))
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        actor=MultiHeadNetwork(config,207,(100,10,10))
        actor.encoder=WidthEncoder(actor.encoder,'S10',config['training']['seed']+100003)
        actor.load_state_dict(raw['actor'],strict=True);actor=actor.to(device).eval();actor.requires_grad_(False)
    return SimpleNamespace(actor=actor,device=device,schema=schema_identity('S10'),sac_policy_structure='factorized'),dict(policy_load_seconds=perf_counter()-started,parameter_updates=0,optimizers_created=0,model_forwards=0)

def load_environment(plan,row,expected_n):
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
    from satellite_rl.t0_width.environment import T0WidthEnv
    started=perf_counter();role='development' if plan['phase']=='development' else 'test'
    case,layout,authority=load_published_layout(row,role=role,confirmation_unlock=plan.get('w26_original_unlock'))
    if len(layout['beam_ids'])!=expected_n:raise ValueError('metadata actual N differs from real frozen input')
    stage=UniformStage2Input(case,layout,authority,PHYSICS)
    input_seconds=perf_counter()-started;started=perf_counter();env=T0WidthEnv(stage,'S10')
    return env,dict(input_load_seconds=input_seconds,input_reset_kernel_seconds=perf_counter()-started,
        kernel_preparation_calls=1,capacity_reward_queries=0,restore_PHY_calls=0)
