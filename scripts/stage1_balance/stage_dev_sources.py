"""Copy S1-01 train/validation aircraft sources into S1-02 isolated storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from remote_sources import LOCAL, REMOTE, connect, run_code


def main() -> None:
    code = f'''
from datetime import datetime,timezone
import hashlib,json,pathlib,shutil
old=pathlib.Path('/home/u2024110224/first_stage_satellite_rl/data_inputs')
new=pathlib.Path({REMOTE!r})/'data'
expected={{'train':(576,['2024-01-08','2024-01-09']),
          'validation':(20,['2024-01-12'])}}
result={{'experiment_id':'S1-02','copied_utc':datetime.now(timezone.utc).isoformat(),
        'source':'S1-01 aircraft-level immutable input','roles':{{}}}}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for role,(count,dates) in expected.items():
 source=old/role;target=new/('source_'+role);target.mkdir(parents=True,exist_ok=True)
 files=sorted(source.glob('source_observations_*.jsonl.gz'))
 if len(files)!=count:raise RuntimeError(role+' old detail count differs')
 record=json.loads((source/'record.json').read_text())
 if (record.get('count')!=count or record.get('source',{{}}).get('source_dates')!=dates
  or record.get('source',{{}}).get('partition_role')!=role):
  raise RuntimeError(role+' old record provenance differs')
 ledger=[]
 for item in [source/'record.json',*files]:
  out=target/item.name;original=sha(item)
  if out.exists():
   if sha(out)!=original:raise RuntimeError('Existing staged file SHA differs: '+str(out))
   action='verified_existing'
  else:
   temp=out.with_suffix(out.suffix+'.tmp');shutil.copyfile(item,temp)
   if sha(temp)!=original:raise RuntimeError('Copied file SHA differs: '+str(out))
   temp.replace(out);action='copied_verified'
  ledger.append({{'name':item.name,'sha256':original,'bytes':out.stat().st_size,'action':action}})
 result['roles'][role]={{'source_record':str(source/'record.json'),
  'staged_record':str(target/'record.json'),'record_sha256':ledger[0]['sha256'],
  'count':count,'source_dates':dates,'files':ledger}}
receipt=new/'source_copy_receipt.json'
temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n');temp.replace(receipt)
print(json.dumps({{'status':'train_validation_staged_sha_verified',
 'receipt':str(receipt),'roles':{{r:{{'count':v['count'],'record_sha256':v['record_sha256'],
 'staged_record':v['staged_record']}} for r,v in result['roles'].items()}}}}))
'''
    client = connect()
    try:
        result = run_code(client, code)
    finally:
        client.close()
    output = LOCAL / "preflight" / "train_validation_source_copy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"observed_utc": datetime.now(timezone.utc).isoformat(),
                                  **result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
