"""Read-only source/date inventory for S2-P19 preparation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.variable_beams import ops18

OUT = Path("D:/paper/outputs/experiments/sac_routes_s2p19_20261007/preflight/data_inventory.json")
CODE = r'''
import datetime,json,pathlib,subprocess
root=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424')
records={}
for name,path in {
 'train':root/'data/prepared/train/record.json',
 'validation':root/'data/prepared/validation_eligible12_v2/record.json',
 'test':root/'data/prepared/test/record.json',
 'validation_mothers':root/'data/prepared/validation_eligible12_v2/mother_record.json',
}.items():
 if path.is_file():
  try:
   x=json.loads(path.read_text(encoding='utf-8'))
   records[name]={'path':str(path),'count':x.get('count'),'record_id':x.get('id'),
    'directory_bytes':int(subprocess.run(['du','-sb',str(path.parent)],capture_output=True,text=True,check=True).stdout.split()[0]),
    'source_dates':x.get('source',{}).get('source_dates'),
    'mother_record_path':x.get('source',{}).get('mother_record_path'),
    'files_count':len(x.get('files',[])),
    'bytes':path.stat().st_size,
    'source_seed':x.get('source',{}).get('seed'),
    'top_profile':x.get('profile'),
    'environment_profile':x.get('source',{}).get('environment_profile'),
    'max_record_age_seconds':x.get('source',{}).get('max_record_age_seconds'),
    'demand_window_seconds':(x.get('source',{}).get('demand_model') or {}).get('window_seconds'),
    'mother_input_count':len(x.get('source',{}).get('mother_scene_inputs') or []),
    'first_mother_input':(x.get('source',{}).get('mother_scene_inputs') or [None])[0]}
  except Exception as exc:records[name]={'path':str(path),'error':str(exc)[:300]}
 else:records[name]={'path':str(path),'exists':False}
raw=root/'data/raw'
files=[]
if raw.is_dir():
 files=[{'name':p.name,'size_bytes':p.stat().st_size} for p in raw.iterdir() if p.is_file() and
  any(d in p.name for d in ('2024.02.01','2024.02.02','2024.01.12','2024.01.29'))]
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'records':records,'matching_raw_files':files},ensure_ascii=False))
'''


def main() -> None:
    result = ops18.remote_python(CODE, 20424)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    payload = json.loads(result["stdout"].splitlines()[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
