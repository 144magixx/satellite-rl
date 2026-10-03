"""Read-only mother eligibility inventory from G11 records on 20424."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys

TRANSPORT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(TRANSPORT))
from remote_transport import connect, execute  # noqa: E402

BASE = "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes"
RECORDS = {"train": BASE + "/scenes_20261001_161621_17ceaa35/record.json",
           "validation": BASE + "/scenes_20261001_161628_819160d7/record.json"}
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight/mother_eligibility.json")

REMOTE = f'''
import datetime,hashlib,json,pathlib
records={RECORDS!r}
out={{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','roles':{{}}}}
for role,name in records.items():
 record=json.loads(pathlib.Path(name).read_text())
 rows=[]
 for index,item in enumerate(record['source']['snapshots']):
  counts=item['counts'];selected=counts.get('selected_aircraft');zero=counts.get('zero_demand_aircraft')
  positive=selected-zero if selected is not None and zero is not None else None
  rows.append({{'index':index,'selected_time_utc':item['selected_time_utc'],
   'selected_aircraft':selected,'positive_demand_aircraft':positive,
   'eligible_200':positive is not None and positive>=200}})
 eligible=[x['index'] for x in rows if x['eligible_200']]
 selected12=[eligible[int((i+0.5)*len(eligible)/12)] for i in range(12)] if role=='validation' and len(eligible)>=12 else None
 out['roles'][role]={{'record':name,'record_sha256':hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest(),
 'mother_count':len(rows),'eligible_200_count':sum(x['eligible_200'] for x in rows),
 'min_positive':min(x['positive_demand_aircraft'] for x in rows if x['positive_demand_aircraft'] is not None),
 'max_positive':max(x['positive_demand_aircraft'] for x in rows if x['positive_demand_aircraft'] is not None),
 'ineligible':[x for x in rows if not x['eligible_200']],
 'eligible_indices':eligible,'selected_validation_12_indices':selected12,
 'selection_rule':'fixed time ranks among eligible Jan12 mothers' if role=='validation' else 'all original Jan08/09 mothers subject to >=200 qualification'}}
print(json.dumps(out,ensure_ascii=False))
'''


def main():
    encoded = base64.b64encode(REMOTE.encode()).decode()
    transport = connect(20424)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=120)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    report = json.loads(result["stdout"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(OUTPUT), "roles": {key: {k: value[k] for k in (
        "mother_count", "eligible_200_count", "min_positive", "max_positive", "ineligible", "eligible_indices", "selected_validation_12_indices")}
        for key, value in report["roles"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
