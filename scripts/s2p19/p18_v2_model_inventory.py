"""Read-only identity receipt for the frozen historical S2-P18 V2 model."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.variable_beams import ops18


OUT = Path("D:/paper/outputs/experiments/sac_routes_s2p19_20261007/preflight/p18_v2_model.json")
CODE = r'''
import hashlib,json,pathlib
root=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/variable_beams_s2p18/20418/archive/models/runs/workbench/train_sac_20261006_084454_365eff71/final')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
model=json.loads((root/'model.json').read_text(encoding='utf-8'))
print(json.dumps({'source_dir':str(root),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
 for p in (root/'policy.pt',root/'model.json')},
 'provenance':model.get('provenance'),'environment_profile':model.get('config',{}).get('environment',{}).get('profile')},
 ensure_ascii=False))
'''


def main():
    result = ops18.remote_python(CODE, 20418)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    payload = json.loads(result["stdout"].splitlines()[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_dir": payload["source_dir"], "files": payload["files"],
                      "provenance_run_id": payload["provenance"].get("run_id"),
                      "provenance_env_step": payload["provenance"].get("env_step")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
