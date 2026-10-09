"""Read-only SHA and date probe for approved historical train/development inputs."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.stage1_variable import ops


SOURCE = "/home/u2024110224/first_stage_satellite_rl/s1_03_r1/20418/outputs/workbench/jobs/train_ppo_20261005_093803_4570222a/scene_inputs"
EXPECTED = {"train": "61be543ac64bf4d0b9e14fd39c7a88d1346d5540cdd3b023433e2ab57ac02ec6",
            "validation": "f73525eaab0e41bc208d4aa6fb14ad174b49c71947c837e2c04751fdec876219"}


def main():
    code = f'''
import hashlib,json,pathlib
root=pathlib.Path({SOURCE!r});out={{}}
for role in ('train','validation'):
 path=root/role/'inputs.npz';record=path.parent/'record.json'
 out[role]={{'exists':path.exists(),'record_exists':record.exists(),
  'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
  'record_source_dates':json.loads(record.read_text()).get('source',{{}}).get('source_dates') if record.exists() else None}}
print(json.dumps(out))
'''
    response = ops.remote_python(code, 20418, timeout=90)
    if response["exit_code"]:
        raise RuntimeError(response["stderr"][-1000:])
    result = json.loads(response["stdout"].splitlines()[-1])
    if any(result[role]["sha256"] != sha or not result[role]["record_exists"]
           for role, sha in EXPECTED.items()):
        raise RuntimeError("historical remote aircraft source SHA or record differs")
    target = ops.LOCAL / "preflight" / "historical_aircraft_source.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
