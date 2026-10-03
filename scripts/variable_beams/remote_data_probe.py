"""Bounded 20424 aviation input check before S2-P17 bulk preparation."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


TRANSPORT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(TRANSPORT))
from remote_transport import connect, execute  # noqa: E402

DEV = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/dev/20261003T185801Z"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
ROOT = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data"
ORIGINAL = "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes/scenes_20261001_161621_17ceaa35"
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")


REMOTE = f'''
import datetime,hashlib,json,os,pathlib,sys
import numpy as np
dev=pathlib.Path({DEV!r});root=pathlib.Path({ROOT!r});original=pathlib.Path({ORIGINAL!r})
sys.path.insert(0,str(dev/'src'))
import satellite_rl
from satellite_rl.workbench.aviation import build_aviation_scenes,CSV_UNITS
if pathlib.Path(satellite_rl.__file__).resolve()!=dev/'src/satellite_rl/__init__.py':
 raise RuntimeError('Dev source import path mismatch')
old=json.loads((original/'record.json').read_text())
if old['count']!=576 or old['source']['partition_role']!='train':raise RuntimeError('G11 mother record mismatch')
item=old['source']['files'][0];snapshot=old['source']['snapshots'][0]
csv=root/'source_csv/train'/snapshot['selected_time_utc'][:10]/pathlib.Path(item['path']).name
if hashlib.sha256(csv.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Mother CSV SHA mismatch')
model=old['source']['demand_model'];time=snapshot['selected_time_utc']
results=[]
for n in (50,100,200):
 for load in (.75,1.25):
  arrays,prov=build_aviation_scenes([csv],[time],source_format='csv_snapshot',
   units={{**CSV_UNITS,'persons':'count'}},demand_model=model,seed=51001,
   max_record_age_seconds=60,environment_profile='europe_v1',partition_role='train',
   window_seconds=60,beam_count=n,load_multiplier=load,min_positive_aircraft=200)
  scene=arrays[0];snap=prov['snapshots'][0]
  if len(scene['rate'])!=n or not np.isfinite(scene['rate']).all() or (scene['rate']<=0).any():
   raise RuntimeError('Variable-N arrays invalid')
  if not np.isclose(scene['rate'].sum(),snap['covered_demand_bps'],rtol=1e-12):
   raise RuntimeError('Beam demand differs from coverage sum')
  results.append({{'N':n,'load':load,'total_demand_bps':snap['total_demand_bps'],
   'covered_demand_bps':snap['covered_demand_bps'],'uncovered_demand_bps':snap['uncovered_demand_bps'],
   'covered_aircraft_fraction':snap['covered_aircraft_fraction'],
   'positive_aircraft':snap['counts']['positive_demand_aircraft']}})
legacy_arrays,legacy_prov=build_aviation_scenes([csv],[time],source_format='csv_snapshot',
 units={{**CSV_UNITS,'persons':'count'}},demand_model=model,seed=51001,
 max_record_age_seconds=60,environment_profile='europe_v1',partition_role='train',
 window_seconds=60,beam_count=100,load_multiplier=1.0,min_positive_aircraft=200)
with np.load(original/'scene_00000.npz',allow_pickle=False) as frozen:
 equal={{key:np.array_equal(legacy_arrays[0][key],frozen[key]) for key in legacy_arrays[0]}}
 differences={{key:{{'new_dtype':str(np.asarray(legacy_arrays[0][key]).dtype),
   'old_dtype':str(frozen[key].dtype),
   'max_abs':float(np.max(np.abs(np.asarray(legacy_arrays[0][key],dtype=float)-np.asarray(frozen[key],dtype=float))))}}
   for key in legacy_arrays[0] if not equal[key]}}
if any(x['max_abs']>1e-5 for x in differences.values()):
 raise RuntimeError('Old 100-beam numerical regression failed: '+str(differences))
print(json.dumps({{'status':'development_inputs_verified_no_training',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424',
 'source_root':str(dev),'package_file':satellite_rl.__file__,
 'mother_csv':str(csv),'mother_sha256':item['sha256'],
 'old_100_arrays_equal':equal,'old_100_differences':differences,'results':results}},ensure_ascii=False))
'''


def main() -> None:
    encoded = base64.b64encode(REMOTE.encode()).decode()
    command = f"PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {DQN} -B -c 'exec(__import__(\"base64\").b64decode(\"{encoded}\"))'"
    transport = connect(20424)
    try:
        result = execute(transport, command, timeout=180)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-2000:])
    report = json.loads(result["stdout"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / ("remote_data_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
