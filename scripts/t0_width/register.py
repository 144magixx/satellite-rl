"""Register the human-authorized W26 scope and bounded storage before dispatch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('existing immutable registration differs')
    else:path.write_bytes(raw)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def main():
    root=Path('D:/paper/outputs/experiments/s2_width26_20261014')
    proposal=next(Path(__file__).resolve().parents[2].glob('docs/*/*/26_*.md'))
    authorization=dict(schema='W26HumanAuthorizationV1',experiment_id='S2-W26',
        user_message='20418、20424上开始训练吧',client_date='2026-10-14',
        authorized=True,executor_ports=[20418,20424],proposal_path=str(proposal),
        proposal_sha256=hashlib.sha256(proposal.read_bytes()).hexdigest(),
        groups=['S00','S10','S01','S11'],seed=51001,scientific_steps_per_group=150000,
        scientific_steps_total=600000,engineering_steps_per_group_max=2048,
        engineering_total_max=8192,layout_prepare_new_max=636,
        development_episodes_max=4320,confirmation_episodes_max=680,stage3_calls=0,
        preserve='T0 legacy_repaired reward, factorized 100/10/10 policy, additive twin Q, gamma0.99 fixed alpha0.01',
        prerequisites=['fresh owner/resource check','exact frozen source and input bindings',
                       'complete 576/20/40 layout admission','CUDA true-update and recovery qualification'],
        no_additional_seed_or_algorithm=True)
    write_new(root/'authorization.json',authorization)
    storage=dict(schema='W26StoragePlanV1',experiment_id='S2-W26',
        local_control_root=str(root),local_models_root='D:/paper/models/runs/s2_width26_20261014',
        remote_root='/home/u2024110224/second_stage_satellite_rl/s2_w26',
        shared_home_single_copy=True,estimated_peak_new_bytes=20*1024**3,
        required_free_margin_bytes=15*1024**3,
        official_archives_bytes=1937121280+1811374080,
        model_checkpoint_policy='per run current plus previous trusted checkpoint; required policy nodes and final; rejected files retained for diagnosis',
        source_policy='one immutable commit snapshot per shared filesystem; per-lane plans and task owners',
        data_policy='one content-addressed active dataset; no per-model copies; selective member extraction only',
        log_policy='compressed step deltas and indexed all-beam terminal observations, no static arrays per step',
        retention=['actual data/identity/source/phase records','final policies and planned development policies',
                   'recovery checkpoints until full delivery','failures and unknown costs','raw numerical logs and figures'],
        cleanup='after verified archive and no active dependency, remove only reproducible transfer/temp copies with receipt',
        backup='local and original remote canonical assets; same shared home is not two independent backups')
    write_new(root/'storage_plan.json',storage)
    manifest=dict(schema='W26RunManifestV1',experiment_id='S2-W26',
        last_checked_utc=datetime.now(timezone.utc).isoformat(),phase='authorized_implementation_and_data_qualification',
        authorized_ports=[20418,20424],scientific_steps=0,scientific_steps_budget=600000,
        formal_tasks=[],source_frozen=False,data_admitted=False,gpu_engineering_admitted=False,
        authorization_ref=str(root/'authorization.json'),storage_ref=str(root/'storage_plan.json'),
        resource_probe_ref='D:/paper_worktrees/stage2-t0-width-v26/outputs/experiments/s2_width26_20261014/preflight/resources_01.json',
        limitations=['proposal accepted; new state/code/data qualifications still in progress',
                     'chat scheduling tools not exposed in this tool session; do not claim automated chat notifications configured'])
    write_new(root/'run_manifest.json',manifest)
    print(json.dumps(dict(experiment='S2-W26',authorized_ports=[20418,20424],scientific_steps=0,
                         status='registered_implementation_qualification'),ensure_ascii=False))


if __name__=='__main__':main()
