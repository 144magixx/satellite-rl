"""Build exact metadata deployment plan from saved original74 evidence."""
import hashlib,json,subprocess
from pathlib import Path

def main():
    repo=Path(__file__).resolve().parents[2]
    base=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013')
    root=base/'round7/learning/recovery'
    contract_path=root/'labels74_recovery_contract_bb7b84d_final.json'
    review_path=root/'root_labels74_recovery_acceptance_bb7.json'
    c=json.loads(contract_path.read_text(encoding='utf8'))
    e=json.loads(Path('E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/round7/learning/recovery/deploy_readonly_bb7_03/preflight.json').read_text(encoding='utf8'))
    old=next(row for row in e['processes'] if row['pid']==60770)
    opsdir=Path(c['operations_source_refs']['module']['path']).parents[3].as_posix()
    def ref(path,data):return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())
    cref=ref(opsdir+'/contract.json',contract_path.read_bytes());rref=ref(opsdir+'/root_recovery_acceptance.json',review_path.read_bytes())
    a=dict(contract=dict(local_path=contract_path.as_posix(),ref=cref),recovery_review=dict(local_path=review_path.as_posix(),ref=rref),accounting_addendum=dict(local_path=(base/'design/recovery_science_accounting_addendum.json').as_posix(),ref=c['accounting_addendum_ref']))
    paths=dict(module='src/satellite_rl/workbench/r7_recovery_ops.py',server='scripts/stage1_ai_search_r7/recovery_server.py',bootstrap='scripts/stage1_ai_search_r7/recovery_bootstrap.py')
    for key,path in paths.items():
        raw=subprocess.check_output(['git','show',c['operational_commit']+':'+path],cwd=repo)
        if hashlib.sha256(raw).hexdigest()!=c['operations_source_refs'][key]['sha256']:raise ValueError('bb7 Git source mismatch')
        a[key]=dict(git_path=path,ref=c['operations_source_refs'][key])
    unchanged={k:c[k] for k in ('original_job_ref','original_case_index_ref','original_owner_ref')}
    unchanged.update({k:dict(path=e['records'][k]['path'],sha256=e['records'][k]['sha256']) for k in ('config.json','status.json','launch.json')})
    script=repo/'scripts/stage1_ai_search_r7/recovery_deploy_once.py'
    plan=dict(schema='R7Labels74EndpointDeploymentPlanV1',run_id=c['run_id'],ssh_port=20474,api_port=9015,
              archive_root=c['original_job']['archive_root'],python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python',
              scientific_commit=c['scientific_commit'],operational_commit=c['operational_commit'],operations_dir=opsdir,
              operations_source_refs=c['operations_source_refs'],contract_ref=cref,recovery_review_ref=rref,old_service={k:old[k] for k in ('pid','birth','birth_kind','boot_id','cmdline','cmdline_sha256','pythonpath')},
              unchanged_refs=unchanged,gpu_uuid='GPU-060b7663-4715-0619-f1f5-e0d39551bebf',resource_limits=dict(minimum_free_bytes=8589934592,minimum_available_memory_bytes=4294967296),
              assets=a,deployment_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
              scientific_calls_authorized=0,actual_POST_authorized=False,only_stop_exact_original_service_pid=True,require_pidfd=True,
              no_SIGKILL=True,health_GET='/api/ai-search-r7/tasks/'+c['run_id'],original_job_config_request_owner_source_immutable=True,
              subsequent_same_ID_resume_separately_authorized=True,maximum_endpoint_exit_wait_seconds=10,maximum_readonly_health_wait_seconds=8)
    target=root/'labels74_endpoint_deployment_plan_bb7.json'
    if target.exists():raise ValueError('immutable plan exists')
    target.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(plan_path=target.as_posix(),plan_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),deployment_script_sha256=plan['deployment_script_sha256'],operations_dir=opsdir)))

if __name__=='__main__':main()
