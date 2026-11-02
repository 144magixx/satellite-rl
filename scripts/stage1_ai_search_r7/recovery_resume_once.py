"""Explicitly delegated single same-ID POST; durable intent forbids replay."""
import argparse,hashlib,json,os
from pathlib import Path
import paramiko

REMOTE=r'''
import datetime,hashlib,json,os,time,urllib.error,urllib.request
from pathlib import Path
plan=PAYLOAD['plan'];ops=Path(plan['operations_dir']);run=plan['run_id'];d=Path(plan['archive_root'])/'outputs/workbench/jobs'/run
def pin(ref):
 p=Path(ref['path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('pinned ref changed: '+str(p))
 return p
body=dict(contract_ref=plan['contract_ref'],root_review_ref=plan['recovery_review_ref'])
c=json.loads(pin(body['contract_ref']).read_text());review=json.loads(pin(body['root_review_ref']).read_text())
if run!='r7_labels_20261013_234807_312cdce7' or review['passed'] is not True or review['contract_sha256']!=body['contract_ref']['sha256']:raise ValueError('original74 exact accepted contract only')
if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,13,23,tzinfo=datetime.timezone.utc):raise ValueError('07:00 deadline reached; no resume')
for ref in plan['unchanged_refs'].values():pin(ref)
for ref in c['operations_source_refs'].values():pin(ref)
job=json.loads((d/'job.json').read_text());source=Path(job['source_root']);manifest=pin(dict(path=str(source/'source_manifest.json'),sha256=job['source_manifest_sha256']));m=json.loads(manifest.read_text())
if len(m['files'])!=698:raise ValueError('original698 source only')
for n,s in m['files'].items():pin(dict(path=str(source/n),sha256=s))
with urllib.request.urlopen('http://127.0.0.1:9015/api/ai-search-r7/tasks/'+run,timeout=10) as r:pre=json.loads(r.read())
if pre['status']['stage']!='failed' or pre['status']['completed_cases']!=147:raise ValueError('original failed147 before unique POST')
if (d/'operations_recovery.json').exists():raise ValueError('recovery sidecar already exists; only GET, never replay')
intent=dict(schema='R7DelegatedSingleResumeIntentV1',run_id=run,method='POST',url='http://127.0.0.1:9015/api/ai-search-r7/tasks/'+run+'/resume',body=body,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=PAYLOAD['helper_sha256'],original_source698_checked=True,source_commit=c['scientific_commit'],operational_commit=c['operational_commit'],no_replay_after_unknown=True,compute_pause_deadline_UTC='2026-10-13T23:00:00+00:00',authority='Parent /root explicit delegation: healthy endpoint then unique durable-intent original139 same-ID POST')
target=ops/'resume_intent.json'
with target.open('x') as f:json.dump(intent,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(ops,os.O_RDONLY);os.fsync(fd);os.close(fd)
request=urllib.request.Request(intent['url'],data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
result=dict(intent=intent,intent_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),post_count=1)
try:
 with urllib.request.urlopen(request,timeout=30) as r:
  raw=r.read();result['http_status']=r.status;result['response']=json.loads(raw);(ops/'resume_response.bin').write_bytes(raw)
except Exception as e:result['POST_result_unknown_or_error']=dict(type=type(e).__name__,message=str(e));(ops/'resume_transport_error.json').write_text(json.dumps(result)+'\n')
# Once submitted, all further calls are bounded GET, including transport-unknown.
for delay in (1,2,3):
 time.sleep(delay)
 try:
  with urllib.request.urlopen('http://127.0.0.1:9015/api/ai-search-r7/tasks/'+run,timeout=10) as r:detail=json.loads(r.read())
  result.setdefault('GET_observations',[]).append(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=detail['status']))
  if detail['status']['stage'] in ('running','completed','failed') and detail['status'].get('pid')!=64582:break
 except Exception as e:result.setdefault('GET_errors',[]).append(dict(type=type(e).__name__,message=str(e)))
result['current_status']=json.loads((d/'status.json').read_text())
for n in ('actual_operations_launch.json','operations_recovery.json','parallel_ledger.json'):
 p=d/n
 if p.exists():result[n]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),record=json.loads(p.read_text()))
pid=result['current_status'].get('pid')
if pid:
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split();result['worker_identity']=dict(pid=pid,birth=int(parts[19]),birth_kind='proc_start_ticks',state=parts[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
 except (FileNotFoundError,ProcessLookupError):result['worker_identity']=dict(pid=pid,state='gone')
result['remaining_original_indices']=c['remaining_case_indices'];result['failed_original_indices_not_retried']=c['failed_case_indices'];result['preserved_original_case_indices']=c['preserved_case_indices']
result['original_job_config_owner_unchanged']=True
for name in ('original_job_ref','original_owner_ref'):pin(plan['unchanged_refs'][name])
pin(plan['unchanged_refs']['config.json'])
(ops/'resume_actual_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output-root',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text(encoding='utf8'))
    out=Path(a.output_root);out.mkdir(parents=True,exist_ok=False);payload=dict(plan=plan,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());code=REMOTE.replace('plan=PAYLOAD', 'PAYLOAD=json.loads('+repr(json.dumps(payload))+')\nplan=PAYLOAD',1)
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('10.160.17.102',port=20474,username='u2024110224',password=os.environ['BUPT_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=20)
    try:
        i,o,e=c.exec_command('/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python -B -',timeout=60);i.write(code);i.channel.shutdown_write();raw=o.read();err=e.read();status=o.channel.recv_exit_status();(out/'stdout.bin').write_bytes(raw);(out/'stderr.bin').write_bytes(err)
        if status:raise RuntimeError('single resume helper failed; inspect exact bytes; never replay automatically')
        result=json.loads(raw);(out/'resume_actual_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output_root=str(out),http_status=result.get('http_status'),status=result.get('current_status'),worker=result.get('worker_identity'))))
    finally:c.close()

if __name__=='__main__':main()
