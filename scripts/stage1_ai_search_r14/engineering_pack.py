"""Prepare/finalize the sole73 sequential original8 engineering plan; never launch.

Source transport is one small archive. Private psutil is referenced in place.
Stable stage admission precedes request hashing; final plan acceptance is last.
"""
import argparse,hashlib,json,posixpath,tarfile,io
from pathlib import Path
from .runtime import json_bytes,atomic_bytes
from .lane import pinned
from .bindings import read_json,same_ref
from .source_closure import relocate_refs
from .pipeline import stage_plan
from .dispatcher import make_request

BASE='/home/u2024110224/first_stage_satellite_rl/s1_ai17'
PRIVATE=BASE+'/20473/round12/v2_01/runtime_dependencies'
def ref(path):
    raw=Path(path).read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def remote(refvalue,path):return {**refvalue,'path':path}

def prepare(source_manifest_ref,repo_root,output):
    from .audit import engineering_fixture_definitions
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=read_json(source_manifest_ref);tag=source_manifest_ref['sha256'][:12]
    target=BASE+'/round14/engineering_'+tag;shared=BASE+'/round14/shared'
    source_root=target+'/source';metadata_root=target+'/config'
    archive=output/'source.tar';temporary=archive.with_suffix('.part')
    with tarfile.open(temporary,'w') as tar:
        for item in manifest['source_refs']:
            raw=pinned({'path':item['snapshot_path'],'sha256':item['sha256'],'bytes':item['bytes']})
            name=item['relative_path']
            if name.startswith('/') or '..' in name.split('/'):raise ValueError('source member path')
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(raw))
        raw=pinned(source_manifest_ref);info=tarfile.TarInfo('source_manifest.json');info.size=len(raw);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(raw))
    temporary.replace(archive)
    fixtures=[]
    for f in engineering_fixture_definitions():
        p=output/('fixture_'+str(f['fixture_index'])+'.json');atomic_bytes(p,json_bytes(f))
        fixtures.append({'fixture_index':f['fixture_index'],'local_ref':ref(p),'remote_ref':remote(ref(p),metadata_root+'/'+p.name)})
    input_manifest={'schema':'R14Original8EngineeringInputsV1','fixtures':fixtures,'real_aircraft_payloads':0,
        'W_RULE_D5_T_fit_calls':0,'rays_max':240,'kernels_max':16,'flows_max':128,'offline_layouts_max':24}
    p=output/'input_manifest.json';atomic_bytes(p,json_bytes(input_manifest));input_ref=remote(ref(p),metadata_root+'/input_manifest.json')
    history=Path(repo_root)/'outputs/experiments/first_stage_ai_search_20261013/round12/resume_20261015_0858'
    refs={name:ref(history/name) for name in ('root_explicit_human_continue_authority.json','root_exact_historical_pause_bindings.json','minimal_continuation_contract.json')}
    if refs['root_explicit_human_continue_authority.json']['sha256']!='aa63719a267d0442a7f9849635d4f774184d9195020d4fc074215eb9b562427d':raise ValueError('original08:58 bytes')
    if refs['root_exact_historical_pause_bindings.json']['sha256']!='4731a8158665ef7cd5f11f9ca276a3b091cc19647595a0044ae2117ca4dc8e0c':raise ValueError('original exact pause generations')
    if refs['minimal_continuation_contract.json']['sha256']!='30363c5fd69a4d7a16b12044e996ea3ece9a7016ef5353334af57b33e949b91d':raise ValueError('original full old marker watch contract')
    old=read_json(refs['root_exact_historical_pause_bindings.json']);contract=read_json(refs['minimal_continuation_contract.json'])['pause_contract']
    for name,r in refs.items():atomic_bytes(output/name,pinned(r))
    historical=[{'path':r['absolute_path'],'sha256':r['sha256'],'bytes':r['bytes']} for r in old['historical_pause_refs']]
    own=target+'/20473/control/stop.request';global_marker=BASE+'/control/r14_global.stop.request'
    epoch={'schema':'R14FiniteRuntimeEpochV1','group':'S1-AI17-R14','human_resume_ref':remote(refs['root_explicit_human_continue_authority.json'],metadata_root+'/root_explicit_human_continue_authority.json'),
        'historical_exact_refs':historical,'own_paths':[contract['old_own_marker'],contract['new_own_stop_marker'],own],
        'global_paths':[r['path'] for r in historical]+[contract['new_global_stop_marker'],global_marker],
        'boot_id':None,'expires_unix':None,'fresh_owner_resource_ref':None,'not_new_user_pause':True,'maximum_operational_seconds':1080}
    tasks=stage_plan()['engineering_tasks']
    stage={'schema':'R14StableStageAdmissionV1','accepted':False,'group':'S1-AI17-R14','engineering_only':True,
        'permitted_ports':[20473],'allowed_tasks':tasks,'source_manifest_ref':remote(source_manifest_ref,source_root+'/source_manifest.json'),
        'input_manifest_ref':input_ref,'runtime_epoch_ref':None,
        'input_bindings_by_task':{f'engineering:{j}':{'fixture_definition_ref':fixtures[j]['remote_ref']} for j in range(8)}}
    plan={'schema':'R14Original8EngineeringLanePlanV1','group':'S1-AI17-R14','port':20473,'retry':0,
        'scientific_execution_authorized':False,'root_exact_source_input_admitted':False,'source_specific_pure_checks_accepted':False,
        'actual_runtime_owner_stop_space_admitted':False,'run_id':'r14_engineering_'+tag+'_20473_once',
        'root_admission_ref':None,'child_stage_admission_ref':None,'runtime_epoch_ref':None,
        'source_root':source_root,'source_manifest_ref':stage['source_manifest_ref'],'source_refs':relocate_refs(manifest,source_root),
        'interpreter':'/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python3.11',
        'interpreter_ref':{'path':'/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python3.11','sha256':'d02403f91c9db2cc25b42ff32fa1007801d3e7a6383b21400c54e4b1c4d8f823','bytes':24630752},
        'private_dependency_paths':[PRIVATE],'private_dependency_manifest_ref':{'path':BASE+'/20473/round12/v2_01/config/dependency_manifest_v1.json','bytes':5507,'sha256':'02c29d7ee923a9d6b15cb265bc8132b82fcbad74dcdb87190293492484b4bd04'},'private_dependency_refs':[],
        'control_root':target+'/20473/control','output_root':target+'/20473/output','shared_dispatch_root':shared+'/dispatch',
        'shared_budget_root':shared+'/budget','shared_object_root':shared+'/objects','shared_audit_ledger':shared+'/audit.json',
        'shared_input_ledger_root':shared+'/input','shared_fit_intent_root':shared+'/fit','own_stop_marker':own,'global_stop_marker':global_marker,
        'outer_seconds':1080,'storage_domains':[],'tasks':[],'allowed_engineering_tasks':tasks,'data_B_fit_evaluation_authorized':False}
    for name,value in [('epoch_FALSE.json',epoch),('stage_admission_FALSE.json',stage),('lane_FALSE.json',plan)]:atomic_bytes(output/name,json_bytes(value))
    result={'schema':'R14Original8TransportPreparationV1','source_archive_ref':ref(archive),'source_manifest_ref':source_manifest_ref,
        'input_manifest_ref':input_ref,'remote_metadata_root':metadata_root,'fixture_refs':fixtures,
        'history_refs':refs,'target':target,'only_executor':20473,'only_tasks':[t['task_id'] for t in tasks],
        'actual_SSH_Popen_payload_GEO_fit':0,'requires_fresh_current_owner_boot_space_pause_and_two_root_approvals':True}
    atomic_bytes(output/'pack_manifest.json',json_bytes(result));return result

def finalize_requests(pack,stage_local_ref,stage_remote_ref,epoch_remote_ref,fresh_storage_domains):
    """After real root stage acceptance; still writes files only and never launches."""
    pack=Path(pack);meta=read_json(ref(pack/'pack_manifest.json'));stage=read_json(stage_local_ref)
    if not same_ref(stage_local_ref,stage_remote_ref) or stage.get('accepted') is not True or stage.get('engineering_only') is not True:
        raise ValueError('actual root stage bytes, explicit transport alias and original8 scope')
    if (stage['permitted_ports']!=[20473] or stage['allowed_tasks']!=stage_plan()['engineering_tasks']
        or any(t['kind']!='engineering' for t in stage['allowed_tasks']) or not same_ref(stage['input_manifest_ref'],meta['input_manifest_ref'])
        or not same_ref(stage['runtime_epoch_ref'],epoch_remote_ref)):raise ValueError('root exact8/source/input/runtime scope')
    plan=read_json(ref(pack/'lane_FALSE.json'))
    if not same_ref(plan['source_manifest_ref'],stage['source_manifest_ref']):raise ValueError('accepted source bytes')
    for flag in ('scientific_execution_authorized','root_exact_source_input_admitted','source_specific_pure_checks_accepted','actual_runtime_owner_stop_space_admitted'):plan[flag]=True
    plan.update(child_stage_admission_ref=stage_remote_ref,runtime_epoch_ref=epoch_remote_ref,storage_domains=fresh_storage_domains)
    tasks=[]
    for task in stage['allowed_tasks']:
        j=task['fixture_index'];request=make_request(plan,task,inputs={'fixture_definition_ref':meta['fixture_refs'][j]['remote_ref']},dependencies={})
        p=pack/('request_'+str(j)+'.json');atomic_bytes(p,json_bytes(request))
        tasks.append({**task,'request_ref':remote(ref(p),meta['remote_metadata_root']+'/'+p.name)})
    plan['tasks']=tasks;p=pack/'exact_plan_awaiting_final_root_review.json';atomic_bytes(p,json_bytes(plan))
    binding=hashlib.sha256(json_bytes({k:v for k,v in plan.items() if k!='root_admission_ref'})).hexdigest()
    atomic_bytes(pack/'final_plan_review_FALSE.json',json_bytes({'schema':'R14ExactPlanAcceptanceV1','accepted':False,
        'plan_binding_sha256':binding,'child_stage_admission_ref':stage_remote_ref,'only_engineering8':True}))
    return ref(p),binding

def attach_final_review(plan_path,review_local_ref,review_remote_ref):
    plan=read_json(ref(plan_path));review=read_json(review_local_ref)
    binding=hashlib.sha256(json_bytes({k:v for k,v in plan.items() if k!='root_admission_ref'})).hexdigest()
    if not same_ref(review_local_ref,review_remote_ref) or review.get('accepted') is not True or review['plan_binding_sha256']!=binding:
        raise ValueError('last exact plan acceptance, no request/admission reference cycle')
    plan['root_admission_ref']=review_remote_ref;atomic_bytes(plan_path,json_bytes(plan));return ref(plan_path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(ref(a.source_manifest),a.repo,a.output)
