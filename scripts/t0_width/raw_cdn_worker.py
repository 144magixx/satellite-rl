"""Frozen pure transport worker for exactly two original W26 official tar assets."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse
import urllib.request

CHUNK=8*1024**2
PINS=[('2024-02-23',153286680,1937121280),('2024-02-24',153417613,1811374080)]

def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def acquire(ticket,asset,deadline):
    root=Path(ticket['canonical_directory']);root.mkdir(parents=True,exist_ok=True)
    day=asset['date'];target=root/(day+'.tar');partial=root/(day+'.tar.transport.part')
    ledger=root/(day+'.transport_ledger.json');status=root/(day+'.transport_status.json')
    owner_path=root/(day+'.owner.json')
    identity=dict(request_id=ticket['request_id'],source_admission_SHA=ticket['original_download_admission_SHA'],
        source_url=asset['source_url'],release_asset_id=asset['asset_id'],release_asset_bytes=asset['expected_bytes'])
    if owner_path.exists():
        if json.loads(owner_path.read_text())!=identity:raise ValueError('original download owner differs')
    else:
        with owner_path.open('x') as stream:json.dump(identity,stream)
    if root.joinpath(day+'.tar.part').exists():raise ValueError('existing original partial needs root accounting')
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(identity=identity,transport_ticket_SHA=ticket['ticket_SHA'],chunks=[],failures=[])
    if state['identity']!=identity or state['transport_ticket_SHA']!=ticket['ticket_SHA']:raise ValueError('transport ledger owner differs')
    offset=0
    if partial.exists():
        with partial.open('rb') as stream:
            for row in state['chunks']:
                raw=stream.read(row['bytes'])
                if row['start']!=offset or len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['SHA']:raise ValueError('actual prefix differs')
                offset+=len(raw)
            if stream.read(1):raise ValueError('unledgered byte suffix preserved')
    elif state['chunks']:raise ValueError('ledgered original bytes missing')
    if target.exists():raise ValueError('canonical original already exists; separate root verification required')
    while offset<asset['expected_bytes']:
        if time.monotonic()>deadline:raise TimeoutError('four hour total transport wall expired')
        if shutil.disk_usage(root).free<15*1024**3+CHUNK:raise ValueError('fifteen GiB storage safety margin required')
        length=min(CHUNK,asset['expected_bytes']-offset);end=offset+length-1
        expected='bytes %d-%d/%d'%(offset,end,asset['expected_bytes'])
        for attempt in range(3):
            try:
                req=urllib.request.Request(asset['transport_url'],headers={'Range':'bytes=%d-%d'%(offset,end),'User-Agent':'W26-root-official-original-byte-transport'})
                with urllib.request.urlopen(req,timeout=90) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=expected or urlparse(response.url).hostname!='release-assets.githubusercontent.com':raise ValueError('exact official206 Range required')
                    body=response.read(length+1)
                if len(body)!=length:raise ValueError('short/excess original body')
                break
            except Exception as error:
                state['failures'].append(dict(observed_UTC=now(),offset=offset,end=end,attempt=attempt,error=repr(error),read_bytes_unknown_not_zero=True))
                save(ledger,state)
                save(status,dict(status='network_retry',asset_id=asset['asset_id'],bytes=offset,error=repr(error),source_url=asset['source_url'],algorithm_calls=0))
                if attempt==2:raise
                time.sleep(2)
        with partial.open('ab' if partial.exists() else 'xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        state['chunks'].append(dict(start=offset,bytes=len(body),SHA=hashlib.sha256(body).hexdigest(),observed_UTC=now()))
        offset+=len(body);save(ledger,state)
        save(status,dict(status='transporting',asset_id=asset['asset_id'],bytes=offset,expected_bytes=asset['expected_bytes'],observed_UTC=now(),algorithm_calls=0))
    whole=digest(partial)
    if partial.stat().st_size!=asset['expected_bytes']:raise ValueError('whole official size differs')
    state.update(whole_SHA=whole,completed_UTC=now());save(ledger,state)
    os.link(partial,target)
    if digest(target)!=whole:raise ValueError('published canonical original differs')
    partial.unlink()
    result=dict(status='transport_complete',date=day,asset_id=asset['asset_id'],bytes=offset,whole_SHA=whole,
                path=str(target),source_url=asset['source_url'],transport_url=asset['transport_url'],ticket_SHA=ticket['ticket_SHA'],
                redirect_receipt_SHA=ticket['redirect_receipt_SHA'],algorithm_calls=0,root_source_admitted=False,
                requires_original_workbench_sameID_ingest=True,original_failed_job=ticket['original_run_id'])
    save(status,result);return result

def main():
    path=Path(sys.argv[1]);raw=path.read_bytes();ticket=json.loads(raw)
    if len(sys.argv)!=3 or hashlib.sha256(raw).hexdigest()!=sys.argv[2]:raise ValueError('root ticket file SHA differs')
    if ticket.get('root_approved')is not True or ticket.get('schema')!='W26OfficialRawTransportTicketV1' or ticket.get('max_wall_seconds')!=14400:raise ValueError('finite delegated transport approval required')
    if ticket.get('request_id')!='w26-official-feb23-feb24-v1' or ticket.get('original_run_id')!='8ad262bcfa72690c833da979':raise ValueError('same original task ownership required')
    root=Path(ticket['canonical_directory']).resolve()
    if str(root)!='/home/u2024110224/second_stage_satellite_rl/s2_w26/shared_data/official':raise ValueError('isolated canonical directory required')
    if [(a['date'],a['asset_id'],a['expected_bytes']) for a in ticket['assets']]!=PINS:raise ValueError('exact original asset contract required')
    for asset in ticket['assets']:
        tag='v'+asset['date'].replace('-','.')+'-planes-readsb-prod-0'
        if asset['source_url']!='https://github.com/adsblol/globe_history_2024/releases/download/'+tag+'/'+tag+'.tar':raise ValueError('original source URL changed')
        p=urlparse(asset['transport_url'])
        if p.scheme!='https' or p.hostname!='release-assets.githubusercontent.com':raise ValueError('official HTTPS transport only')
    ticket['ticket_SHA']=sys.argv[2]
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'transport_recovery.lock';handle=lock.open('a+b');fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+14400
    save(root/'transport_process.json',dict(pid=os.getpid(),started_UTC=now(),ticket_SHA=ticket['ticket_SHA'],algorithm_calls=0))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda a:acquire(ticket,a,deadline),ticket['assets']))
        save(root/'transport_aggregate.json',dict(status='transport_complete',assets=results,algorithm_calls=0,root_source_admitted=False,
             original_workbench_job_still_failed_until_ingest=True))
    except BaseException as error:
        save(root/'transport_failure.json',dict(status='failed',error=repr(error),observed_UTC=now(),algorithm_calls=0));raise

if __name__=='__main__':main()
