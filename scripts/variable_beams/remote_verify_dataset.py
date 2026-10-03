"""Verify every prepared S2-P17 input SHA, shape, pairing and split on 20424."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


TRANSPORT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(TRANSPORT))
from remote_transport import connect, execute  # noqa: E402

BASE = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data/prepared"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")
DIRECTORY = {"validation": "validation_eligible12_v2", "test": "test", "train": "train"}
EXPECTED = {"validation": (12, 144), "test": (20, 440), "train": (576, 8064)}
HELD_OUT = {60, 90, 140, 180}


def verify(role: str) -> dict:
    target = BASE + "/" + DIRECTORY[role]
    mothers, count = EXPECTED[role]
    code = f'''
import collections,datetime,hashlib,json,pathlib,numpy as np
root=pathlib.Path({target!r});role={role!r}
record_path=root/'record.json';record=json.loads(record_path.read_text())
if record.get('count')!={count} or record['source'].get('experiment_id')!='S2-P17' or record['source'].get('partition_role')!=role:
 raise RuntimeError('Record count or experiment/partition metadata differs')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
if len(record['files'])!={count}:raise RuntimeError('Record file count differs')
by_mother=collections.defaultdict(lambda:collections.defaultdict(set));counts=collections.Counter()
seen=set();details={{}}
for index,item in enumerate(record['files']):
 p=pathlib.Path(item['path'])
 if p.parent!=root or not p.is_file() or sha(p)!=item['sha256']:
  raise RuntimeError(f'Input scene SHA/path failed at {{index}}')
 if item['scene_id'] in seen:raise RuntimeError('Duplicate scene ID')
 seen.add(item['scene_id'])
 n=item['beam_count'];load=item['load_multiplier']
 if not 1<=n<=200 or load not in (.75,1.25):raise RuntimeError('N or load out of range')
 if role in ('train','validation') and n in {sorted(HELD_OUT)!r}:raise RuntimeError('Heldout N in train/validation')
 if role=='validation' and n not in (50,80,100,120,160,200):raise RuntimeError('Validation N differs')
 if role=='test' and n not in (50,60,75,90,100,125,140,150,175,180,200):raise RuntimeError('Test N differs')
 by_mother[item['mother_scene_id']][n].add(load);counts[n]+=1
 detail=item['observation_detail'];dp=pathlib.Path(detail['path'])
 if dp.parent!=root:raise RuntimeError('Observation detail outside input root')
 if str(dp) not in details:details[str(dp)]=detail['sha256']
 elif details[str(dp)]!=detail['sha256']:raise RuntimeError('Detail SHA declared inconsistently')
 with np.load(p,allow_pickle=False) as arrays:
  actual_n=arrays['rate'].shape[0]
  if actual_n!=n or arrays['beam_location'].shape!=(n,2) or arrays['rate'].shape!=(n,1):
   raise RuntimeError('NPZ array shape/N differs')
  if not np.isfinite(arrays['rate']).all() or (arrays['rate']<=0).any():
   raise RuntimeError('Invalid beam demand')
  if not np.isclose(arrays['rate'].sum(),item['covered_demand_bps'],rtol=1e-12):
   raise RuntimeError('Covered demand does not equal beam sum')
  if not np.isclose(item['total_demand_bps'],item['covered_demand_bps']+item['uncovered_demand_bps'],rtol=1e-12):
   raise RuntimeError('Demand accounting not conserved')
for path,digest in details.items():
 if sha(pathlib.Path(path))!=digest:raise RuntimeError('Observation detail SHA differs: '+path)
if len(by_mother)!={mothers}:raise RuntimeError('Mother count differs')
expected_n=7 if role=='train' else 6 if role=='validation' else 11
for mother,values in by_mother.items():
 if len(values)!=expected_n or any(loads!={{.75,1.25}} for loads in values.values()):
  raise RuntimeError('Mother/N/load pairing incomplete: '+mother)
report={{'status':'verified_prepared_inputs_only_no_policy_calls',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','role':role,
 'record_path':str(record_path),'record_sha256':sha(record_path),
 'mother_count':len(by_mother),'scene_count':len(seen),'detail_count':len(details),
 'beam_count_distribution':dict(sorted(counts.items())),
 'source_record_sha256':record['source'].get('mother_record_sha256')}}
print(json.dumps(report,ensure_ascii=False))
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = f"PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {DQN} -B -c 'exec(__import__(\"base64\").b64decode(\"{encoded}\"))'"
    transport = connect(20424)
    try:
        result = execute(transport, command, timeout=1800)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-2000:])
    report = json.loads(result["stdout"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"verify_{role}_inputs_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=tuple(DIRECTORY), required=True)
    print(json.dumps(verify(parser.parse_args().role), ensure_ascii=False))


if __name__ == "__main__":
    main()
