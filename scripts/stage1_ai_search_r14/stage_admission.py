"""Stable child scope approval, separate from final exact-plan byte approval."""
from .bindings import read_json,same_ref

def verify_stage(request):
    stage=read_json(request['root_admission_ref'])
    if (stage.get('schema')!='R14StableStageAdmissionV1' or stage.get('accepted') is not True
        or stage.get('group')!='S1-AI17-R14' or request['port'] not in stage['permitted_ports']
        or not same_ref(stage['source_manifest_ref'],request['source_manifest_ref'])
        or not same_ref(stage['runtime_epoch_ref'],request['runtime_epoch_ref'])):
        raise ValueError('stable independently accepted source/input/runtime stage')
    allowed={t['task_id']:t for t in stage['allowed_tasks']}
    actual=allowed.get(request['task_id'])
    if actual is None:raise ValueError('task outside finite stage allowlist')
    for key in ('kind','phase','fixture_index','recipe_index','arm','repeat'):
        if actual.get(key)!=request.get(key):raise ValueError('stage task scope '+key)
    for name,ref in stage['input_bindings_by_task'][request['task_id']].items():
        if not same_ref(request.get(name),ref):raise ValueError('exact admitted task input '+name)
    if stage.get('engineering_only') is True:
        from .pipeline import stage_plan
        if (stage['allowed_tasks']!=stage_plan()['engineering_tasks']
            or any(t['kind']!='engineering' for t in stage['allowed_tasks']) or request['kind']!='engineering'):
            raise ValueError('only original8 engineering; data/B/fit cannot unlock')
    return stage
