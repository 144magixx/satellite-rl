"""Local candidate writer. Never accepts, connects, stages, or launches anything."""
import argparse,copy,json
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from .contracts import GROUP,CAPS,PARTITION

def build(config):
 cases=copy.deepcopy(config['cases'])
 if [v['original_index'] for v in cases]!=list(c.MOTHERS):raise ValueError('complete exact12 cases')
 for case in cases:
  c.check_identity(case['identity'])
  if not case.get('historical_common_SHA') or set(case['historical_controls'])!={'Gr','S2','SH-C'}:raise ValueError('saved common and paid historical controls')
 result={}
 for port in PARTITION:
  h=config['hosts'][str(port)];root=Path(h['epoch_root'])
  plan=dict(schema='R11ExactSavedCommonGhPlanV1',group=GROUP,run_id=h['run_id'],request_id=h['request_id'],execution_epoch_id=h['execution_epoch_id'],executor_port=port,workbench_port={20473:9050,20474:9051}[port],source_repo=h['source_repo'],archive_root=str(root/'archive'),outer_control_root=str(root/'control/outer'),worker_control_root=str(root/'control/worker'),host_compute_lock=h['host_compute_lock'],source_files=copy.deepcopy(config['source_files']),lifecycle_dependencies=copy.deepcopy(config['lifecycle_dependencies']),original_source_manifest_ref=config['original_source_manifest_ref'],design_ref=config['design_ref'],root_design_acceptance_ref=config['root_design_acceptance_ref'],human_resume_authority_ref=config['human_resume_authority_ref'],cases=cases,arms=['Gh'],caps=CAPS.copy(),method_hard_seconds=300,method_rss_bytes=2147483648,outer_hard_seconds=2400,human_epoch_ref=None,input_qualification_ref=None,storage_inventory_ref=None,engineering_acceptance_ref=None,actual_resource_route_ref=None,source_input_budget_frozen=False)
  epoch=dict(schema='R11NewGhHumanEpochV1',group=GROUP,run_id=plan['run_id'],request_id=plan['request_id'],execution_epoch_id=plan['execution_epoch_id'],archive_root=plan['archive_root'],authority_ref=plan['human_resume_authority_ref'],new_deadline_unix_seconds=None,watched_pause_paths=[],historical_hold_exemptions=[],root_accepted=False)
  review=dict(schema='R11RootExactGhExecutionAdmissionV1',exact_plan_ref=None,passed=False,execution_authorized=False,scientific_execution_authorized=False,source_accepted=False,input_accepted=False,runtime_accepted=False,engineering4_accepted=False,human_epoch_ref=None,input_qualification_ref=None,storage_inventory_ref=None,engineering_acceptance_ref=None)
  storage=dict(schema='R11OwnedStorageInventoryV1',archive_root=plan['archive_root'],owned_growth_root=str(root.parent),owned_growth_roots=[str(root.parent)],owned_temporary_root=str(root/'temporary'),owned_temporary_roots=[str(root/'temporary')],new_logical_bytes_max=536870912,temporary_bytes_max=268435456,free_reserve_bytes=8589934592,personal_quota='unknown',root_accepted=False)
  result[str(port)]=dict(plan=plan,epoch=epoch,review=review,storage=storage)
 return dict(schema='R11LocalCandidateGraphV1',scientific_execution_authorized=False,hosts=result,source_epoch_note='R11 wrappers explicitly reference historical R10 common/run; no wrapper relabelling or common regeneration',outer_scope='2400s includes six 300s native stages plus at most600s parent/input/IO/start/retirement allowance; no strict speed comparison to historic controls')

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--config-sha256',required=True);a.add_argument('--output',required=True);v=a.parse_args();config=c.read(dict(path=v.config,sha256=v.config_sha256));c.immutable(v.output,build(config))
if __name__=='__main__':main()
