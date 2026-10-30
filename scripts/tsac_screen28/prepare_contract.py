"""Offline metadata-only W28 machine draft. No model/scene/PHY/network calls."""
import csv
import json
from pathlib import Path
from satellite_rl.t0_width.data import file_sha
from satellite_rl.t0_width.logging import atomic_json
from satellite_rl.tsac_screen28.plan import matrix
from satellite_rl.tsac_screen28.proposals import SEED_CONTRACT
from satellite_rl.tsac_screen28.checks import catalog,fixture_manifest,validate_catalog

ROOT=Path('D:/paper')
BASE=ROOT/'outputs/experiments/s2_tsac_screen28_20261015'
W26=ROOT/'outputs/experiments/s2_width26_20261014'
W27=ROOT/'outputs/experiments/s2_tsac_search27_20261015'
REMOTE='/home/u2024110224/second_stage_satellite_rl/s2_w28'
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(path),sha256=file_sha(path))

def main():
    out=BASE/'contract_draft_01';out.mkdir(parents=True,exist_ok=True)
    devpath=W27/'data/training_reuse_01/20418_development.json';testpath=W26/'collected/inputs/test_published.json'
    confirm=read(W26/'deployment/confirmation_01/plan_20418.json')['tasks']['confirmation_direct']['config']
    scientific=read(W27/'deployment/science_04/plan_20418.json');devref=list(scientific['tasks'].values())[0]['config']['development_manifest_ref']
    if file_sha(devpath)!=devref['sha256'] or file_sha(testpath)!=confirm['confirmation_manifest_ref']['sha256']:raise ValueError('canonical metadata provenance differs')
    with (W26/'report_analysis/dataset_cases.csv').open(encoding='utf-8-sig',newline='') as stream:
        cases={(row['role'],row['case_id']):row for row in csv.DictReader(stream)}
    producer_dir=W26/'collected/runs/65d381d7d092de5b12c52c37';job=read(producer_dir/'job.json')
    unlock=dict(confirm['confirmation_unlock'],manifest_ref=confirm['confirmation_manifest_ref'])
    tasks={}
    for phase,path,manifest_ref,expected_sum in [('development',devpath,devref,1605),('known',testpath,confirm['confirmation_manifest_ref'],3209)]:
        role='development' if phase=='development' else 'test'
        metadata=[dict(case_id=row['case_id'],N=int(cases[(role,row['case_id'])]['N']),manifest_index=i,
            layout_SHA=row['layout_content_SHA'],aircraft_SHA=row['case_content_SHA']) for i,row in enumerate(read(path)['rows'])]
        if sum(row['N'] for row in metadata)!=expected_sum:raise ValueError('real metadata sumN differs')
        matrix_value=matrix(metadata,phase);atomic_json(out/(phase+'_metadata.json'),dict(rows=metadata,source_manifest_ref=manifest_ref,dataset_cases_ref=ref(W26/'report_analysis/dataset_cases.csv')))
        plan=dict(schema='W28FinitePlanV1',experiment_id='S2-W28',root_approved=False,runtime_approved=False,
            phase=phase,max_workers=1,device='cuda',executor_port=20418,rootpauseepoch='REQUIRE_ROOT_W28_RESUMED_EPOCH',
            source_commit='REQUIRE_FROZEN_COMMIT',source_root=REMOTE+'/source/REQUIRE_FROZEN_COMMIT',source_manifest={},
            matrix=matrix_value,proposal_seed_contract=SEED_CONTRACT,new_training=0,new_parameter_updates=0,new_data=0,S3_calls=0,
            policy_group='S10',policy_node=150000,policy_ref=confirm['terminal_policies']['S10'],
            producer_config_ref=dict(path=job['status']['run_dir']+'/config.json',sha256=file_sha(producer_dir/'config.json')),
            producer_job_ref=dict(path=job['status']['run_dir']+'/job.json',sha256=file_sha(producer_dir/'job.json')),
            producer_plan_ref=dict(path='/home/u2024110224/second_stage_satellite_rl/s2_w26/deployment/scientific_01/plan_20418.json',sha256=file_sha(W26/'deployment/scientific_01/plan_20418.json')),
            data_manifest_ref=manifest_ref,metadata_ref=dict(path=REMOTE+'/contracts/'+phase+'_metadata.json',sha256=file_sha(out/(phase+'_metadata.json'))),
            w26_original_unlock=unlock if phase=='known' else None,
            w27_closure_ref=dict(path=REMOTE+'/admissions/w27_closure.json',sha256='REQUIRE_ROOT_CLOSURE_SHA'),
            qualification_ref=dict(path=REMOTE+'/admissions/qualification.json',sha256='REQUIRE_ROOT_QUALIFICATION_SHA'),
            wall_budget_seconds=21600,wall_budget_semantics='group total active scientific runtime across both phases; parent backend additionally enforces common ledger',
            independent_test=False,physical_budget_semantics='capacity/reward queries plus actual commits; initial kernel preparation separately counted, no free PHY reconstruction')
        if phase=='known':
            plan['development_decision_ref']=dict(path=REMOTE+'/control/locked_development_decision.json',sha256='REQUIRE_ROOT_LOCKED_DECISION_SHA')
            plan['development_plan_SHA']='REQUIRE_EXACT_COMPLETED_DEVELOPMENT_PLAN_SHA'
            plan['development_manifest_SHA']=devref['sha256']
        tasks[phase]=plan
    expected=[(19260,38520),(44926,77016)]
    for phase,(commits,queries) in zip(('development','known'),expected):
        if (tasks[phase]['matrix']['commits'],tasks[phase]['matrix']['queries'])!=(commits,queries):raise ValueError('exact finite capacities differ')
    fixtures=fixture_manifest();checks=catalog();bounds=validate_catalog(checks,fixtures)
    atomic_json(out/'engineering_fixtures.json',fixtures);atomic_json(out/'engineering_catalog.json',dict(checks=checks,bounds=bounds))
    fixture_ref=dict(path=REMOTE+'/contracts/engineering_fixtures.json',sha256=file_sha(out/'engineering_fixtures.json'))
    engineering=dict(tasks['development'],schema='W28EngineeringPlanV1',phase='engineering',run_id='engineering_01',
        parameter_updates=0,limits=dict(query=256,commit=64,forward=64,proposal=64),checks=checks,
        check_contract_version='W28QualificationChecksV1',fixture_manifest_ref=fixture_ref,metadata_ref=fixture_ref,
        data_manifest_ref=fixture_ref,catalog_bounds=bounds,
        engineering_checklist_status='CONCRETE_BUILTIN_16_CHECKS; root must freeze exact source/plan/admissions before runtime')
    engineering.pop('matrix');engineering.pop('qualification_ref')
    tasks['engineering']=engineering
    contract=dict(schema='W28WorkbenchPlanV1',experiment_id='S2-W28',root_approved=False,runtime_approved=False,
        executor_port=20418,http_port=9080,max_workers=1,source_root=REMOTE+'/source/REQUIRE_FROZEN_COMMIT',
        stop_url_base='http://127.0.0.1:9080',science_wall_budget_seconds=21600,engineering_wall_budget_seconds=1800,tasks=tasks,
        admission_refs={name:dict(path=REMOTE+'/admissions/'+name+'.json',sha256='REQUIRE_ROOT_'+name.upper()+'_SHA') for name in ('source','data','resources','cost')},
        engineering_budget=dict(physical_calls_attempted=320,real_commit_attempts=64,candidate_restore_attempts=256,model_forward_attempts=64,parameter_updates=0),
        recovery_rule=dict(same_ID=True,source_config_policy_data_SHA_pinned=True,unknown_calls_replayed=0,
            cursor_SHA_required=True,previous_dispatch_SHA_required=True,unrecorded_wall_upper_bound_seconds='REQUIRED_NONNEGATIVE_FINITE; zero needs explicit proof; deduct across both science phases',
            current_missing_with_any_previous_tmp_intent_evidence='BLOCK_PRESERVE_UNKNOWN',current_present_with_tmp='BLOCK_PRESERVE_UNKNOWN'),
        kernel_preparation_caps=dict(development=240,known=560,engineering=64,restore_PHY=0),
        total_science_capacity_reward_calls=179722,new_user_pause_requirement=False,
        gate_spec_template=dict(experiment_id='S2-W28',node_id='20418',scope_root=REMOTE,uid='REQUIRE_LIVE_UID',
            soft_UTC='2026-10-14T22:58:00+00:00',hard_UTC='2026-10-14T23:00:00+00:00',contract_SHA='REQUIRE_FROZEN_GATE_CONTRACT_SHA'),
        human_continue_reference=ref(W27/'control/resume_20261015_0857/root_user_continue.json'),
        gate_resume_rule='Initialize distinct W28 Gate with preserved prior cutoff; root signs new W28 explicit-continue receipt referencing 2026-10-15T00:57:33Z; never reuse W27 worker experiment ID.',
        storage=dict(scientific_peak_GiB=8,transfer_peak_GiB=1,recovery_reserve_GiB=2,scope='each canonical local and remote store; check live mount/quota before admission'))
    atomic_json(out/'workbench_plan_draft.json',contract)
    print(json.dumps(dict(runtime_approved=False,development_N_sum=1605,known_N_sum=3209,science_capacity_reward_calls=179722,model_or_PHY_calls=0)))

if __name__=='__main__':main()
