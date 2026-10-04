"""Independent read-only audit of the 520+56 mother input replacement."""
from __future__ import annotations

import json
from pathlib import Path

from ops_common import BASE, remote_python, write_json


OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/replenishment")
G11 = "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes/scenes_20261001_161621_17ceaa35/record.json"


def audit() -> dict:
    code = f'''
import collections,datetime,hashlib,json,pathlib
base=pathlib.Path({BASE!r});data=base/'data';old_path=pathlib.Path({G11!r})
record_path=data/'prepared/train_mothers_replaced576/record.json'
mapping_path=data/'replenishment/replacement_mapping.json'
rule_path=data/'replenishment/selection_contract.json'
predeclared=data/'replenishment/predeclared_rule.json'
auth=data/'replenishment/authorization.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
old=json.loads(old_path.read_text());record=json.loads(record_path.read_text())
mapping=json.loads(mapping_path.read_text())['mapping'];rule=json.loads(rule_path.read_text());fixed=json.loads(predeclared.read_text())
authorization=json.loads(auth.read_text())
if (record['count']!=576 or len(record['source']['files'])!=576 or len(record['source']['snapshots'])!=576
 or record['source']['experiment_id']!='S2-P17' or record['source']['partition_role']!='train'
 or record['source']['original_g11_record_sha256']!=sha(old_path)
 or record['source']['authorization_sha256']!=sha(auth)
 or record['source']['selection_contract_sha256']!=sha(rule_path)
 or record['source']['replacement_mapping_sha256']!=sha(mapping_path)):
 raise RuntimeError('Replenished mother record identity/SHA differs')
if (authorization['decision']!='replace576' or authorization['status']!='user_confirmed_decision'
 or fixed['status']!='predeclared_before_halfgrid_scan'
 or fixed['date_quota']!={{'2024-01-08':32,'2024-01-09':24}}
 or len(mapping)!=56 or len({{x['old_index'] for x in mapping}})!=56):
 raise RuntimeError('User decision or frozen pre-scan quotas differ')
mapped={{x['old_index']:x for x in mapping}}
old_times={{s['selected_time_utc'] for s in old['source']['snapshots']}}
new_times=set();counts=collections.Counter();retained=0
for index,(reference,snapshot) in enumerate(zip(record['source']['files'],record['source']['snapshots'])):
 path=pathlib.Path(reference['path'])
 if not path.is_file() or sha(path)!=reference['sha256']:
  raise RuntimeError(f'Mother CSV SHA mismatch at {{index}}')
 if snapshot['selected_time_utc']!=reference['selected_time_utc']:
  raise RuntimeError('Mother file/snapshot time mismatch')
 day=snapshot['selected_time_utc'][:10]
 if day not in ('2024-01-08','2024-01-09') or snapshot['counts']['selected_aircraft']-snapshot['counts']['zero_demand_aircraft']<200:
  raise RuntimeError('Mother date or >=200 qualification fails')
 if index in mapped:
  item=mapped[index]
  if (reference['sha256']!=item['new_csv_sha256'] or reference['path']!=item['new_csv_path']
   or snapshot['selected_time_utc']!=item['new_time_utc']
   or item['positive_demand_aircraft']<200 or item['unique_positive_positions']<200
   or item['old_source_sha256']!=old['source']['files'][index]['sha256']):
   raise RuntimeError('Mapped replacement differs')
  if item['new_time_utc'] in old_times or item['new_time_utc'] in new_times:
   raise RuntimeError('Replacement time duplicates original or another replacement')
  new_times.add(item['new_time_utc']);counts[day]+=1
 else:
  retained+=1
  if (reference['sha256']!=old['source']['files'][index]['sha256']
   or snapshot['selected_time_utc']!=old['source']['snapshots'][index]['selected_time_utc']):
   raise RuntimeError('Original qualified mother changed')
if retained!=520 or dict(counts)!={{'2024-01-08':32,'2024-01-09':24}}:
 raise RuntimeError('Original/replacement counts differ')
for day in ('2024-01-08','2024-01-09'):
 selected=rule['selected'][day]
 if selected['times_utc']!=[item['new_time_utc'] for item in mapping if item['new_time_utc'][:10]==day]:
  raise RuntimeError('Selected contract times differ from replacement mapping')
report={{'status':'verified_576_mother_inputs_only_no_model_calls',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424',
 'original_retained':retained,'replaced':len(mapping),'replaced_by_day':dict(counts),
 'mother_record_path':str(record_path),'mother_record_sha256':sha(record_path),
 'original_g11_record_sha256':sha(old_path),'authorization_sha256':sha(auth),
 'predeclared_rule_sha256':sha(predeclared),'selection_contract_sha256':sha(rule_path),
 'replacement_mapping_sha256':sha(mapping_path)}}
print(json.dumps(report,ensure_ascii=False))
'''
    result = remote_python(code, timeout=180)
    path = OUTPUT / "mother_audit.json"
    write_json(path, result)
    return {"receipt": str(path), **result}


def main() -> None:
    print(json.dumps(audit(), ensure_ascii=False))


if __name__ == "__main__":
    main()
