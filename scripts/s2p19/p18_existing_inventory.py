"""Read-only inventory of already completed S2-P18 refinement assets."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.variable_beams import ops18

OUT = Path("D:/paper/outputs/experiments/sac_routes_s2p19_20261007/preflight/p18_existing_inventory.json")
RUNS = {"V2": (20418, "refine_sac_20261006_224531_316265cd"),
        "V3": (20424, "refine_sac_20261006_173515_4b10636d"),
        "G0": (20418, "refine_sac_20261007_024934_39422e19")}
CODE = r'''
import datetime,json,pathlib,subprocess
run_id=__RUN_ID__
port=__PORT__
root=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/variable_beams_s2p18')/str(port)/'archive/outputs/workbench/jobs'/run_id
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'run_id':run_id,'root':str(root),'exists':root.exists()}
if root.is_dir():
 result['directory_bytes']=int(subprocess.run(['du','-sb',str(root)],capture_output=True,text=True,check=True).stdout.split()[0])
 result['top_files']=[{'name':p.name,'bytes':p.stat().st_size if p.is_file() else None}
  for p in root.iterdir() if p.name in ('job.json','config.json','results.json','status.json','power_cases','refinements')]
 cases=root/'power_cases'
 if cases.is_dir():
  paths=sorted(cases.glob('*.json'))
  result['power_case_count']=len(paths)
  if paths:
   first=json.loads(paths[0].read_text(encoding='utf-8'))
   result['first_case']={'name':paths[0].name,'keys':list(first),'values':{k:str(v)[:180] for k,v in first.items() if k in ('method','scene_id','metrics','result','scene_metadata','route')}}
 for filename in ('job.json','config.json','status.json'):
  path=root/filename
  if path.is_file():
   item=json.loads(path.read_text(encoding='utf-8'))
   result[filename]={'keys':list(item),'refinement':{k:str(v)[:180] for k,v in item.items() if 'sca' in k.lower() or 'method' in k.lower()}}
print(json.dumps(result,ensure_ascii=False))
'''


def main() -> None:
    inventory = {}
    for role, (port, run_id) in RUNS.items():
        code = CODE.replace("__RUN_ID__", repr(run_id)).replace("__PORT__", repr(port))
        try:
            result = ops18.remote_python(code, port)
            if result["exit_code"]:
                inventory[role] = {"error": result["stderr"][-500:]}
            else:
                inventory[role] = json.loads(result["stdout"].splitlines()[-1])
        except Exception as exc:
            inventory[role] = {"error": type(exc).__name__ + ": " + str(exc)[:200]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
