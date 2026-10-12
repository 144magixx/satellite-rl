"""Read-only official metadata and bounded prior-use inventory; no raw download."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request
from satellite_rl.training.run_monitor import atomic_json
from . import ops

DATES=('2024-02-17','2024-02-18','2024-02-19','2024-02-20')
SOURCE=ops.LOCAL/'source'


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'satellite-rl-source-inventory','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30) as response:
        data=response.read(8*(1<<20)+1)
        if len(data)>8*(1<<20):raise ValueError('metadata response exceeds 8MiB')
    return data


def official_metadata():
    SOURCE.mkdir(parents=True,exist_ok=True)
    urls={'preferred_releases':'https://raw.githubusercontent.com/adsblol/globe_history_2024/main/PREFERRED_RELEASES.txt',
          'license':'https://raw.githubusercontent.com/adsblol/globe_history_2024/main/LICENSE-ODbL.txt'}
    pins={}
    for key,url in urls.items():
        data=fetch(url);path=SOURCE/('PREFERRED_RELEASES.txt' if key=='preferred_releases' else 'LICENSE-ODbL.txt')
        path.write_bytes(data);pins[key]={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'url':url}
    parser_origin=Path('D:/paper/outputs/experiments/first_stage_geometry_load_20261012/source/preparation_reference.py')
    parser=SOURCE/'preparation_reference.py';shutil.copyfile(parser_origin,parser)
    pins['parser']={'path':str(parser),'sha256':hashlib.sha256(parser.read_bytes()).hexdigest(),
                    'source_path':str(parser_origin),'reuse':'SHA-pinned trace parser used by G16 pure position/request adapter'}
    preferred=(SOURCE/'PREFERRED_RELEASES.txt').read_text(encoding='utf-8')
    def day(date):
        dotted=date.replace('-','.')
        matches=re.findall(r'https://github\.com/adsblol/globe_history_2024/releases/download/(v'+re.escape(dotted)+r'-planes-readsb-(?:prod|staging)-0)/(v[^\s]+\.tar)',preferred)
        if len(matches)!=1:raise ValueError('expected one official preferred source for '+date)
        tag,name=matches[0];url='https://api.github.com/repos/adsblol/globe_history_2024/releases/tags/'+tag
        data=fetch(url);release=json.loads(data);assets=[a for a in release.get('assets',[]) if a['name']==name]
        if len(assets)!=1 or release.get('draft'):raise ValueError('official release asset unavailable for '+date)
        asset=assets[0];folder=SOURCE/date;folder.mkdir(exist_ok=True);(folder/'release_api.json').write_bytes(data)
        record={'group':'S1-AI17','date_utc':date,'release_tag':tag,'release_api':url,
            'release_api_sha256':hashlib.sha256(data).hexdigest(),'archive_name':asset['name'],
            'archive_url':asset['browser_download_url'],'expected_bytes':asset['size'],'asset_id':asset['id'],
            'asset_digest':asset.get('digest'),'source_instance':tag.split('planes-readsb-')[1],
            'preferred_sha256':pins['preferred_releases']['sha256'],'license_sha256':pins['license']['sha256'],
            'parser_sha256':pins['parser']['sha256'],'observation_utc':datetime.now(timezone.utc).isoformat(),
            'status':'metadata_verified_not_downloaded'}
        atomic_json(folder/'source_preflight.json',record)
        return date,record
    with ThreadPoolExecutor(max_workers=4) as pool:rows=dict(pool.map(day,DATES))
    result={'group':'S1-AI17','sources':rows,'shared_pins':pins,'raw_downloaded':False,
            'observed_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(SOURCE/'official_metadata_inventory.json',result)
    return result


def prior_use():
    code=r'''import datetime,hashlib,json,pathlib
base=pathlib.Path('/home/u2024110224/first_stage_satellite_rl')
dates=('2024-02-17','2024-02-18','2024-02-19','2024-02-20')
patterns=('*/*/archive/outputs/workbench/scenes/*/record.json',
 '*/*/archive/outputs/workbench/jobs/*/prepared_dataset/preparation_record.json',
 'archive/outputs/workbench/scenes/*/record.json','*/archive/outputs/workbench/scenes/*/record.json')
paths=sorted({p for pattern in patterns for p in base.glob(pattern)})
rows=[];skipped=[];matches=[]
for path in paths:
 size=path.stat().st_size
 if size>16*(1<<20):skipped.append({'path':str(path),'bytes':size,'reason':'bounded metadata exceeds16MiB'});continue
 data=path.read_bytes();hits=[d for d in dates if d.encode() in data]
 record=json.loads(data)
 row={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':size,
 'role':record.get('partition_role',record.get('role')),'dates':record.get('dates',record.get('source_dates',[])),
 'count':record.get('count'),'matched_candidate_dates':hits}
 rows.append(row)
 if hits:matches.append(row)
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'query_ssh_port':20418,'shared_nfs_root':str(base),'patterns':patterns,'metadata_count':len(rows),
 'metadata_bytes_read':sum(r['bytes'] for r in rows),'records':rows,'matched_records':matches,
 'skipped':skipped,'complete_project_prior_use_proof':False,
 'boundary':'registered first-stage scene and packed-preparation metadata only; no teacher/results/raw archives/NPZ opened'}))
'''
    result=ops.remote_python(code,20418,timeout=65)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1000:])
    value=json.loads(result['stdout'].splitlines()[-1]);atomic_json(SOURCE/'remote_prior_use_inventory.json',value)
    return value


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        official=pool.submit(official_metadata);history=pool.submit(prior_use)
        a=official.result();b=history.result()
    print(json.dumps({'official_days':list(a['sources']),'raw_downloaded':False,
        'prior_metadata_count':b['metadata_count'],'candidate_date_matches':len(b['matched_records']),
        'skipped_count':len(b['skipped']),'evidence_root':str(SOURCE)},ensure_ascii=False))
