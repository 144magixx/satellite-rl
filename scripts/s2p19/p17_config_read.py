"""Read-only P17 V1/greedy SCA config for the P18 route comparison."""

import json
from pathlib import Path

from scripts.variable_beams import ops18

CODE = r'''
import hashlib,json,pathlib
root=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/archive/outputs/workbench/jobs')
ids={'V1':'refine_sac_20261005_014002_932212d0','B1':'refine_sac_20261005_075851_e6741d48'}
out={}
for role,run_id in ids.items():
 p=root/run_id
 config_path=p/'config.json';job_path=p/'job.json'
 config=json.loads(config_path.read_text(encoding='utf-8'))
 job=json.loads(job_path.read_text(encoding='utf-8'))
 out[role]={'run_id':run_id,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
  'job_sha256':hashlib.sha256(job_path.read_bytes()).hexdigest(),
  'config_power_refinement':config['power_refinement'],
  'job_power_refinement':job['power_refinement']}
print(json.dumps(out,ensure_ascii=False))
'''


def main():
    result = ops18.remote_python(CODE, 20424)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    output = Path("D:/paper/outputs/verification/s2p19_research_20261007/p17_sca_configs.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(result["stdout"].splitlines()[-1])
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
