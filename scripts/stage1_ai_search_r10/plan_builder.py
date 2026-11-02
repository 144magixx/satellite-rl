"""Pure finite R10 plan assembly; no source/input fabrication or resource admission."""
from .contracts import GROUP,DESIGN_SHA,ARMS,CAPS,MOTHERS,METHOD_HARD_SECONDS,METHOD_RSS_BYTES
from .schedule import partition

def assemble(*,run_id,request_id,executor_port,archive_root,cases,mother_host_assignment,**refs):
 partition(cases,mother_host_assignment,executor_port)
 if not run_id.startswith('r10_') or '/round10/' not in archive_root:raise ValueError('new independent R10 namespace')
 required=('design_ref','root_design_acceptance_ref','numeric_origin_protocol_ref','source_files','original_source_manifest_ref','human_resume_authority_ref','human_epoch_ref','input_qualification_ref','storage_inventory_ref','actual_resource_route_ref','host_compute_lock','execution_epoch_id')
 if any(k not in refs for k in required) or refs['design_ref']['sha256']!=DESIGN_SHA:raise ValueError('all real frozen reference candidates required')
 return dict(schema='R10ExactMechanismPlanV2',group=GROUP,run_id=run_id,request_id=request_id,executor_port=executor_port,archive_root=archive_root,cases=cases,mother_host_assignment=mother_host_assignment,arms=list(ARMS),caps=dict(CAPS),common_hard_seconds=30,method_hard_seconds=METHOD_HARD_SECONDS,method_rss_bytes=METHOD_RSS_BYTES,source_input_budget_frozen=False,**refs)
