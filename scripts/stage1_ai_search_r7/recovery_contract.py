"""Metadata-only exact recovery contract; caller supplies actual root-pinned refs."""
from copy import deepcopy
from satellite_rl.workbench.r7_recovery_ops import RUN,SCIENCE,FAILED,REMAINING

def build(operational_commit,original_job,refs,failed_cases,original_pid_births):
    if len(operational_commit)!=40 or original_job['run_id']!=RUN or original_job['scientific_commit']!=SCIENCE:raise ValueError('actual new ops source and original137 job')
    required={'original_job_ref','original_case_index_ref','original_owner_ref','accounting_addendum_ref','module','bootstrap','server'}
    if set(refs)!=required or any(not ref.get('path') or len(ref.get('sha256',''))!=64 for ref in refs.values()):raise ValueError('actual root-pinned original/source/addendum refs')
    if set(failed_cases)!=set(map(str,FAILED)):raise ValueError('exact original2 failed dispositions')
    if any(o.get('birth_kind') not in ('proc_start_ticks','unix_boot_seconds') or not o.get('boot_id') for o in original_pid_births):raise ValueError('PID birth units and originalboot_id required')
    return dict(schema='R7Labels74OperationalRecoveryV1',repair_id='labels74_proc_exit_esrch_recovery01',run_id=RUN,
                request_id=original_job['request_id'],scientific_commit=SCIENCE,operational_commit=operational_commit,
                original_job=deepcopy(original_job),original_job_ref=refs['original_job_ref'],original_case_index_ref=refs['original_case_index_ref'],original_owner_ref=refs['original_owner_ref'],
                accounting_addendum_ref=refs['accounting_addendum_ref'],operations_source_refs={k:refs[k] for k in ('module','bootstrap','server')},
                original_pid_births=original_pid_births,preserved_case_indices=list(range(1,295,2)),failed_case_indices=FAILED,remaining_case_indices=REMAINING,
                failed_cases=failed_cases,new_stage_attempts_max=dict(prepare=139,T=139,D5=139),repeat_original_attempts=False,label_denominator=576,
                final_label_attempts_upper=dict(prepare=576,T=575,D5=574),thresholds_unchanged=dict(qualified=519,nontrivial=116,capability_mothers=6),
                original_job_config_request_owner_bytes_unchanged=True,operations_metadata_sidecar_only=True,source698_immutable=True,
                no_science_dispatch_without_root_review=True,unsupported_other_groups_or_ports=True)
