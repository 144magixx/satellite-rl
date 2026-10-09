"""Read-only S1-D12A source provenance inventory, without downloading assets."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from scripts.stage1_drl_r04.ops import LOCAL, remote_python


REMOTE = r'''
import hashlib,json,pathlib
base=pathlib.Path('/home/u2024110224/first_stage_satellite_rl/s1_v10')
rows=[]
for p in base.glob('*/outputs/workbench/scenes/*/record.json'):
 try:
  d=json.loads(p.read_text()); role=d.get('partition_role')
  if role not in ('train','development'): continue
  path=pathlib.Path(d.get('path',''))
  rows.append({'port':p.parts[-6] if len(p.parts)>6 else None,'id':d.get('id'),
   'role':role,'record_path':str(p),'path':str(path),'exists':path.is_file(),
   'sha256':d.get('sha256'),'count':d.get('count'),'dates':d.get('dates'),
   'source_frozen_npz_sha256':d.get('source',{}).get('source_frozen_npz_sha256')})
 except Exception as e: rows.append({'path':str(p),'error':str(e)})
sources=[]
for date in ('2024-02-07','2024-02-08','2024-02-09'):
 for p in (base/'source'/date/'source_preflight.json',
           base.parent/'s1_d12a'/'source'/date/'source_preflight.json'):
  if p.is_file(): sources.append({'date':date,'path':str(p)})
print(json.dumps({'historical_records':rows,'candidate_source_pins':sources},ensure_ascii=False))
'''


def main():
    result = remote_python(REMOTE, 20418, timeout=90)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    value = {"observed_utc": datetime.now(timezone.utc).isoformat(),
             "read_only": True, **json.loads(result["stdout"].splitlines()[-1])}
    path = LOCAL / "preflight" / "source_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "historical_records": len(value["historical_records"]),
                      "candidate_source_pins": len(value["candidate_source_pins"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
