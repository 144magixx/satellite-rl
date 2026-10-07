"""Read compact per-scene final B metrics from already saved S2-P18 jobs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.variable_beams import ops18

OUT = Path("D:/paper/outputs/verification/s2p19_research_20261007/p18_existing_B_compact.json")
RUNS = {"V2": (20418, "refine_sac_20261006_224531_316265cd"),
        "V3": (20424, "refine_sac_20261006_173515_4b10636d"),
        "G0": (20418, "refine_sac_20261007_024934_39422e19")}
CODE = r'''
import datetime,hashlib,json,pathlib,statistics
run_id=__RUN_ID__
port=__PORT__
root=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/variable_beams_s2p18')/str(port)/'archive/outputs/workbench/jobs'/run_id
cases=sorted((root/'power_cases').glob('*_sca_si_lp.json'))
if len(cases)!=1320:raise RuntimeError('Expected 1320 saved final B cases, got '+str(len(cases)))
rows=[]
for path in cases:
 raw=path.read_bytes(); x=json.loads(raw)
 if x['method']!='sca_si_lp' or x['run_id']!=run_id:raise RuntimeError('case identity differs')
 m=x['metrics']; meta=x['scene_metadata']
 rows.append({'case_name':path.name,'case_sha256':hashlib.sha256(raw).hexdigest(),
  'source_scene_sha256':x['source_scene_sha256'],'beam_count':int(x['beam_count']),
  'mother_scene_id':meta['mother_scene_id'],'load_multiplier':float(meta['load_multiplier']),
  'repeat':int(x['repeat']),'si':float(m['final_si_mean']),'sgm':float(m['final_sgm_mean']),
  'power_w':float(m['power_w']),'basic_met_count':int(m['basic_met_count'])})
by_n={}
for n in sorted({x['beam_count'] for x in rows}):
 r=[x for x in rows if x['beam_count']==n]
 if len(r)!=120:raise RuntimeError('per N case count differs')
 by_n[str(n)]={'actions':len(r),'si':statistics.mean(x['si'] for x in r),
  'sgm':statistics.mean(x['sgm'] for x in r),
  'power_w':statistics.mean(x['power_w'] for x in r)}
job_path=root/'job.json';config_path=root/'config.json';status_path=root/'status.json'
job=json.loads(job_path.read_text(encoding='utf-8'))
config=json.loads(config_path.read_text(encoding='utf-8'))
status=json.loads(status_path.read_text(encoding='utf-8'))
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'run_id':run_id,'port':port,'status_stage':status['stage'],
 'job_sha256':hashlib.sha256(job_path.read_bytes()).hexdigest(),
 'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
 'source_model_sha256':rows and job.get('model_files'),
 'job_power_refinement':job.get('power_refinement'),
 'config_power_refinement':config.get('power_refinement'),
 'source_evaluation_run_id':rows and job.get('initial_source'),
 'unique_scene_count':len({x['source_scene_sha256'] for x in rows}),
 'unique_mother_count':len({x['mother_scene_id'] for x in rows}),
 'by_n':by_n,'rows':rows}
print(json.dumps(result,ensure_ascii=False,separators=(',',':')))
'''


def main() -> None:
    output = {"scope": "read-only saved remote S2-P18 final B files; no optimization",
              "runs": {}}
    for role, (port, run_id) in RUNS.items():
        code = CODE.replace("__RUN_ID__", repr(run_id)).replace("__PORT__", repr(port))
        result = ops18.remote_python(code, port, timeout=300)
        if result["exit_code"]:
            raise RuntimeError(f"{role}: " + result["stderr"][-1200:])
        parsed = json.loads(result["stdout"].splitlines()[-1])
        output["runs"][role] = parsed
        print(role, parsed["observed_utc"], parsed["status_stage"], parsed["unique_scene_count"],
              parsed["unique_mother_count"], {n: v["si"] for n, v in parsed["by_n"].items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
