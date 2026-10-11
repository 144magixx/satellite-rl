"""Idempotent official-source preparation and four-lane dataset registration.

Only the workbench performs scientific preparation. This client transfers pinned
metadata, submits API jobs, and registers already verified canonical inputs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from . import ops

ASSIGNMENT = {'2024-02-10': ('validation', 20473),
              '2024-02-11': ('test', 20474), '2024-02-12': ('test', 20418)}
STATE = ops.LOCAL / 'controller/data_preparation.json'
REGISTRY = ops.LOCAL / 'controller/dataset_registry.json'
PARTITION = ops.LOCAL / 'source/partition_manifest.json'
PARSER = ops.LOCAL / 'source/preparation_reference.py'
OLD_REGISTRY = Path('D:/paper/outputs/experiments/first_stage_s1_d12a_20261010/controller/dataset_registry.json')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def upload_one(path, destination, port=20418):
    expected=sha(path)
    probe=ops.remote_python(f"import pathlib,hashlib,json;p=pathlib.Path({destination!r});p.parent.mkdir(parents=True,exist_ok=True);print(json.dumps({{'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}}))",port)
    if probe['exit_code']:
        raise RuntimeError(probe['stderr'][-500:])
    old=json.loads(probe['stdout'].splitlines()[-1])['sha256']
    if old is not None and old!=expected:
        raise ValueError('refusing to replace different pinned metadata '+destination)
    if old is None:
        receipt=ops.upload(path,destination,port)
        if receipt['local_sha256']!=expected or receipt['remote']!=expected:
            raise ValueError('metadata upload SHA differs')
    return {'path':destination,'sha256':expected}

def upload_sources():
    remote={}
    for day,(role,_) in ASSIGNMENT.items():
        folder=ops.LOCAL/'source'/role/day
        pin=read(folder/'source_preflight.json')
        if pin['group']!='S1-ND12' or pin['date_utc']!=day or pin['archive_url'] not in (folder/'PREFERRED_RELEASES.txt').read_text():
            raise ValueError('official preferred source identity differs')
        for name in ('source_preflight.json','selection_contract.json','release_api.json','PREFERRED_RELEASES.txt','LICENSE-ODbL-official.txt'):
            remote[f'{day}/{name}']=upload_one(folder/name,f'{ops.BASE}/source/{role}/{day}/{name}')
    remote['partition']=upload_one(PARTITION,f'{ops.BASE}/source/partition_manifest.json')
    remote['parser']=upload_one(PARSER,f'{ops.BASE}/source/preparation_reference.py')
    save(ops.LOCAL/'preflight/source_metadata_upload.json',remote)
    return remote

def prepare_once(commit):
    if len(commit)!=40:
        raise ValueError('explicit full source commit required')
    state=read(STATE) if STATE.exists() else {'experiment_id':'S1-ND12','scientific_commit':commit,'tasks':{}}
    if state['scientific_commit']!=commit:
        raise ValueError('preparation source differs from saved receipt')
    refs=read(ops.LOCAL/'preflight/source_metadata_upload.json')
    for day,(role,port) in ASSIGNMENT.items():
        if day not in state['tasks']:
            catalog=ops.api(port,'/api/coverage/nondrl/catalog')
            body={'config':catalog['defaults']['06'],'kind':'prepare','preparation_mode':'dataset',
                  'dataset_role':role,'source_date':day,'partition_manifest_path':refs['partition']['path'],
                  'partition_manifest_sha256':refs['partition']['sha256'],'parser_path':refs['parser']['path'],
                  'parser_sha256':refs['parser']['sha256'],'request_id':'s1nd12_prepare_'+day.replace('-','')}
            job=ops.api(port,'/api/coverage/nondrl/runs',body)
            if job['git_commit']!=commit or job['experiment_id']!='S1-ND12':
                raise ValueError('submitted preparation source differs')
            state['tasks'][day]={'port':port,'run_id':job['run_id'],'role':role,'source_date':day,'resume_attempts':0}
            save(STATE,state)
        row=state['tasks'][day]
        detail=ops.api(port,'/api/runs/'+row['run_id'])
        if detail['job']['git_commit']!=commit:
            raise ValueError('saved task source changed')
        status=detail['status'];row['stage']=status['stage'];row['last_status']=status
        row['artifacts']=detail.get('artifacts',{})
    state['observed_utc']=datetime.now(timezone.utc).isoformat()
    state['all_completed']=all(x['stage']=='completed' for x in state['tasks'].values())
    save(STATE,state)
    return state

def records(port):
    value=ops.api(port,'/api/coverage/nondrl/datasets')
    return value.get('datasets',[]) if isinstance(value,dict) else value

def ensure_import(port, role, source, partition):
    matching=[r for r in records(port) if r.get('partition_role')==role and r.get('sha256')==source['sha256']]
    if matching:
        if len(matching)!=1:
            raise ValueError('ambiguous duplicate canonical dataset')
        record=matching[0]
        if record.get('partition_manifest_sha256')!=partition['sha256']:
            raise ValueError('existing dataset partition SHA differs')
    else:
        record=ops.api(port,'/api/coverage/nondrl/datasets/import',{'partition_role':role,
            'path':source['path'],'sha256':source['sha256'],'name':'S1-ND12 '+role,
            'partition_manifest_path':partition['path'],'partition_manifest_sha256':partition['sha256']})
    return {key:record[key] for key in ('id','sha256','count','path')}

def register(commit):
    state=prepare_once(commit)
    if not state['all_completed']:
        raise RuntimeError('official preparations are not all completed')
    refs=read(ops.LOCAL/'preflight/source_metadata_upload.json');sources={}
    for day,task in state['tasks'].items():
        found=[r for r in records(task['port']) if r.get('source_run_id')==task['run_id']]
        if len(found)!=1:
            raise ValueError('prepared dataset record not unique')
        sources[day]=found[0]
    existing=[r for r in records(20418) if r.get('partition_role')=='test' and r.get('count')==100 and r.get('partition_manifest_sha256')==refs['partition']['sha256']]
    if len(existing)>1:
        raise ValueError('ambiguous merged test dataset')
    merged=existing[0] if existing else ops.api(20418,'/api/coverage/nondrl/datasets/merge',{
        'source_datasets':[{'port':ASSIGNMENT[d][1],'id':sources[d]['id']} for d in ('2024-02-11','2024-02-12')]})
    old=read(OLD_REGISTRY)['lanes']['20418']
    inputs={'train':old['train'],'development':old['development'],
            'validation':sources['2024-02-10'],'test':merged}
    def lane(port):
        return str(port),{role:ensure_import(port,role,source,refs['partition']) for role,source in inputs.items()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        lanes=dict(pool.map(lane,ops.SSH_PORTS))
    value={'experiment_id':'S1-ND12','scientific_commit':commit,'partition_manifest_sha256':refs['partition']['sha256'],
           'lanes':lanes,'preparation_tasks':state['tasks'],'observed_utc':datetime.now(timezone.utc).isoformat()}
    for role in inputs:
        if len({(r[role]['sha256'],r[role]['count']) for r in lanes.values()})!=1:
            raise ValueError('canonical role differs across instances')
    save(REGISTRY,value)
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','prepare','register'))
    parser.add_argument('--commit');args=parser.parse_args()
    result=upload_sources() if args.action=='upload' else prepare_once(args.commit) if args.action=='prepare' else register(args.commit)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
