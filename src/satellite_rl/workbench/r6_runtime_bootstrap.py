"""Public same-ID bootstrap: original source validator and original runner."""
import argparse,hashlib,importlib.util,json,os,runpy,sys
from pathlib import Path

def bootstrap(directory,resume=False,verify_only=False):
    directory=Path(directory);job=json.loads((directory/'job.json').read_text(encoding='utf8'))
    operational_root=Path(__file__).resolve().parents[3]
    snapshot=Path(job['source_root']).resolve();manifest=json.loads((snapshot/'source_manifest.json').read_text(encoding='utf8'))
    sys.path[:0]=[str(snapshot/'src'),manifest['source_root']]
    name='satellite_rl.workbench.r6_runtime_dependency'
    path=operational_root/'src/satellite_rl/workbench/r6_runtime_dependency.py'
    spec=importlib.util.spec_from_file_location(name,path);dependency=importlib.util.module_from_spec(spec);sys.modules[name]=dependency;spec.loader.exec_module(dependency)
    from satellite_rl.workbench.storage import Store
    from satellite_rl.workbench import coverage_ai_search_r6 as policy,frozen_round_tasks
    from satellite_rl.training.run_monitor import atomic_json
    store=Store(job['archive_root'],source_root=operational_root)
    receipt=dependency.prepare(store,job,directory,sys.executable)
    original,validation=dependency.original_validation_policy(operational_root,job['archive_root'])
    policy.validate_admission=original.validate_admission
    frozen_round_tasks.verify(store,job,policy)
    receipt.update(validation,bootstrap_file=str(Path(__file__).resolve()),bootstrap_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),complete_frozen_task_verify_passed=True)
    if verify_only:return receipt
    atomic_json(directory/'runtime_bootstrap_validation.json',receipt)
    sys.argv=['satellite_rl.workbench.runner','--job-dir',str(directory)]+(['--resume']if resume else [])
    runpy.run_module('satellite_rl.workbench.runner',run_name='__main__')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-dir',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    value=bootstrap(a.job_dir,a.resume,a.verify_only)
    if a.verify_only:print(json.dumps(value))
