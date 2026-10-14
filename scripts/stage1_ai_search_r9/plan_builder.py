"""Metadata-only prospective M plan builder. Never posts or imports geometry."""
import argparse,json
from pathlib import Path
from .contracts import GROUP,CAPS,M_INDICES,OPS_FILES,digest,immutable

def source_refs(repo):
 repo=Path(repo);paths=[repo/'scripts/stage1_ai_search_r9'/name for name in OPS_FILES]+sorted((repo/'src/satellite_rl/coverage/ai_search_r9').glob('*.py'))
 return {p.relative_to(repo).as_posix():dict(path=str(p.resolve()),sha256=digest(p)) for p in paths}
def build(mapping,config,repo):
 cases=[c for c in mapping['cases'] if c['original_index'] in M_INDICES]
 if [c['original_index'] for c in cases]!=M_INDICES or any(c['identity']['role']!='train' or c['uniform_request_bps']!=21600000 for c in cases):raise ValueError('original12mechanism train mothers')
 if config['executor_port'] not in (20473,20474):raise ValueError('only73/74 after fresh independent admission')
 if not config['run_id'].startswith('r9_M_') or '/round9/' not in config['archive_root']:raise ValueError('new independent R9 task path')
 return dict(config,schema='R9MechanismMExactPlanV1',group=GROUP,stage='M',case_indices=M_INDICES,cases=cases,caps=CAPS,source_files=source_refs(repo),scientific_execution_authorized_by_plan=False,automatic_L_dispatch=False)
def main():
 a=argparse.ArgumentParser();a.add_argument('--mapping',required=True);a.add_argument('--config',required=True);a.add_argument('--output',required=True);v=a.parse_args();repo=Path(__file__).resolve().parents[2];plan=build(json.loads(Path(v.mapping).read_bytes()),json.loads(Path(v.config).read_bytes()),repo);print(json.dumps(immutable(v.output,plan)))
if __name__=='__main__':main()
