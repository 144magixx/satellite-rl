"""Official original-byte network repair; no case, layout, model or algorithm calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import requests
from scripts.t0_width import remote_ops as remote

CONTROL=Path('D:/paper/outputs/experiments/s2_width26_20261014/deployment/network_recovery_01')
ASSETS=[dict(date='2024-02-23',asset_id=153286680,expected_bytes=1937121280),
        dict(date='2024-02-24',asset_id=153417613,expected_bytes=1811374080)]

def now():return datetime.now(timezone.utc).isoformat()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def save(name,value):
    CONTROL.mkdir(parents=True,exist_ok=True);path=CONTROL/name
    if path.exists():raise ValueError('preserve prior network evidence: '+name)
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2),encoding='utf8');return path

def redirect(asset):
    api='https://api.github.com/repos/adsblol/globe_history_2024/releases/assets/'+str(asset['asset_id'])
    tag='v'+asset['date'].replace('-','.')+'-planes-readsb-prod-0'
    origin='https://github.com/adsblol/globe_history_2024/releases/download/'+tag+'/'+tag+'.tar'
    with requests.get(api,headers={'Accept':'application/vnd.github+json'},timeout=(10,25)) as response:
        response.raise_for_status();metadata=response.json()
    if (metadata['id'],metadata['size'],metadata['browser_download_url'])!=(asset['asset_id'],asset['expected_bytes'],origin):
        raise ValueError('official asset metadata mismatch')
    request_url=api+'?w26_transport='+str(time.time_ns())
    with requests.get(request_url,headers={'Accept':'application/octet-stream','Cache-Control':'no-cache'},
                      allow_redirects=False,stream=True,timeout=(10,25)) as response:
        location=response.headers.get('Location','');parts=urlparse(location)
        if response.status_code!=302 or parts.scheme!='https' or parts.hostname!='release-assets.githubusercontent.com':
            raise ValueError('official fresh API redirect must be302 to HTTPS release assets')
        chain=dict(request_url=request_url,status=response.status_code,headers=dict(response.headers),location=location)
    return {**asset,'source_url':origin,'transport_url':location,'observed_UTC':now(),
            'metadata':{k:metadata.get(k) for k in ('id','size','name','digest','browser_download_url')},
            'redirect_chain':[chain],'TLS_verified':True,'algorithm_calls':0}

def probe_remote(row,port):
    code='''
import urllib.request,urllib.parse,json,hashlib,datetime,pathlib
url=URL;total=TOTAL
try:
 req=urllib.request.Request(url,headers={'Range':'bytes=0-4095','User-Agent':'W26-official-byte-probe'})
 with urllib.request.urlopen(req,timeout=25) as r:
  body=r.read(4097);value=dict(http_status=r.status,content_range=r.headers.get('Content-Range'),headers=dict(r.headers),body_hex=body.hex(),body_bytes=len(body),body_SHA=hashlib.sha256(body).hexdigest(),final_url=r.url)
 if value['http_status']!=206 or value['content_range']!='bytes 0-4095/'+str(total) or value['body_bytes']!=4096 or urllib.parse.urlparse(value['final_url']).hostname!='release-assets.githubusercontent.com':raise ValueError('exact official Range proof failed')
 value['reachable']=True
except Exception as e:value=dict(reachable=False,error=repr(e),body_bytes=0)
base=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/s2_w26/shared_data/official')
value.update(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),existing=[dict(name=p.name,bytes=p.stat().st_size) for p in base.glob('*') if p.is_file()],algorithm_calls=0)
print(json.dumps(value))
'''.replace('URL',repr(row['transport_url'])).replace('TOTAL',str(row['expected_bytes']))
    result=remote.remote_python(port,code,timeout=35,interpreter=remote.DQN)
    if result['exit_code']:value=dict(reachable=False,error=result['stderr'],body_bytes=0)
    else:value=json.loads(result['stdout'].splitlines()[-1])
    save('range_%s_%s.json'%(row['asset_id'],port),value)
    return value

def probe_local(row):
    with requests.get(row['transport_url'],headers={'Range':'bytes=0-4095'},timeout=(10,30)) as response:
        body=response.content
        if response.status_code!=206 or response.headers.get('Content-Range')!='bytes 0-4095/'+str(row['expected_bytes']) or len(body)!=4096:
            raise ValueError('local exact original bytes Range failed')
        value=dict(observed_UTC=now(),http_status=response.status_code,headers=dict(response.headers),
                   body_bytes=len(body),body_SHA=digest(body),final_url=response.url,reachable=True,algorithm_calls=0)
    CONTROL.joinpath('range_'+str(row['asset_id'])+'_local.bin').write_bytes(body)
    save('range_'+str(row['asset_id'])+'_local.json',value);return value

def main():
    rows=[redirect(a) for a in ASSETS]
    save('official_redirects.json',dict(assets=rows,algorithm_calls=0))
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[(row,p) for row in rows for p in remote.PORTS]
        results=list(pool.map(lambda job:probe_remote(*job),jobs))
    locals_=[probe_local(row) for row in rows]
    summary=dict(remote=[dict(asset=job[0]['asset_id'],port=job[1],reachable=result['reachable'],body_SHA=result.get('body_SHA'),error=result.get('error')) for job,result in zip(jobs,results)],
                 local=[dict(asset=row['asset_id'],reachable=v['reachable'],body_SHA=v['body_SHA']) for row,v in zip(rows,locals_)],algorithm_calls=0)
    save('network_probe_summary.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
