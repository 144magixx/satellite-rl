"""F20 frozen ADSB preparation, invoked exclusively by a workbench worker."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
from .s2p19_data import _scan_day, _detail_gzip
from .aviation import build_aviation_scenes, CSV_UNITS

DATES = ('2024-02-13','2024-02-14','2024-02-15')
COUNTS = {'validation':(50,100,150,200),'test':(50,60,75,90,100,125,140,150,175,180,200)}
LOADS = (0.75,1.0,1.25)


def partition_assets(role, receipts):
    """Published training inputs must not disclose raw test archive paths."""
    dates = {'validation': {DATES[0]}, 'test': set(DATES[1:])}
    if role not in dates:
        raise ValueError('unknown new-data partition')
    selected = [dict(asset) for asset in receipts if asset['date'] in dates[role]]
    if {asset['date'] for asset in selected} != dates[role]:
        raise ValueError('partition source ledger incomplete')
    return selected


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)


def select_mothers(candidates, count, exclude):
    """Earliest eligible UTC grid, minimum twenty minutes apart; no scores."""
    selected=[];last=None
    from datetime import datetime
    for i,row in enumerate(candidates):
        stamp=row['selected_time_utc'];t=datetime.fromisoformat(stamp.replace('Z','+00:00'))
        if not row['eligible_200'] or stamp in exclude:continue
        if last is not None and (t-last).total_seconds()<1200:continue
        selected.append(i);last=t
        if len(selected)==count:return selected
    raise ValueError('insufficient preregistered eligible mothers; revise common contract before freeze')


def run_sourceprep(directory, config, progress=None, stopped=None):
    directory=Path(directory).resolve();out=directory/'prepared';out.mkdir(parents=True,exist_ok=True)
    stop=stopped or (lambda:False)
    source=Path(config['train_record_path'])
    if sha(source)!=config['train_record_sha256']:raise ValueError('train record SHA mismatch')
    train=json.loads(source.read_text(encoding='utf-8'))
    model=train['source']['demand_model']
    files=train.get('files',train['source'].get('files',[]))
    if len(files)!=8064 or train.get('count')!=8064:raise ValueError('training requires frozen 8064 scenes')
    mothers={f['mother_scene_id'] for f in files}
    if len(mothers)!=576:raise ValueError('training requires frozen 576 mothers')
    exclude=set(config.get('exclude_mother_ids',[]))
    # Demand/time split cannot be checked from opaque hash IDs alone.
    for f in files:
        if 'selected_time_utc' not in f:raise ValueError('train scene lacks mother UTC identity')
        exclude.add(f['selected_time_utc'])
    target=out/'train';target.mkdir(exist_ok=True);copied=[]
    for i,f in enumerate(files):
        if stop():raise InterruptedError('source preparation stopped')
        src=Path(f['path'])
        if sha(src)!=f['sha256']:raise ValueError('train scene SHA changed')
        dst=target/(f'scene_{i:05d}.npz')
        if dst.exists():
            if sha(dst)!=f['sha256']:raise ValueError('copied train scene differs')
        else:shutil.copy2(src,dst)
        copied.append({**f,'path':str(dst),'partition_role':'train'})
    train_record={**train,'path':str(target),'files':copied,'source':{**train['source'],'experiment_id':'S2-F20','partition_role':'train','original_record_sha256':config.get('original_train_record_sha256',config['train_record_sha256'])}}
    write(target/'record.json',train_record)
    datasets={'train':{'path':str(target/'record.json'),'sha256':sha(target/'record.json'),'count':8064,'files':copied,'source':train_record['source']}}
    assets=config['source_assets']
    if sorted(a['date'] for a in assets)!=list(DATES):raise ValueError('F20 dates must be Feb13/14/15')
    selected_by_role={'validation':[],'test':[]};receipts=[]
    for asset in sorted(assets,key=lambda x:x['date']):
        if not asset.get('path'):
            from .fair20_download import acquire_asset
            asset = acquire_asset(directory, asset, progress, stop)
        day=asset['date'];archive=Path(asset['path'])
        if sha(archive)!=asset['sha256']:raise ValueError('official source SHA changed')
        if not asset.get('official_metadata_sha256') or not asset.get('asset_id'):raise ValueError('official metadata pin missing')
        rows,candidates,scan=_scan_day(archive,day,model,progress,stop)
        role='validation' if day==DATES[0] else 'test';required=12 if role=='validation' else 20
        indices=select_mothers(candidates,required,exclude)
        write(out/f'selection_{day}.json',{'candidates':candidates,'scan':scan,'selected_indices':indices,'rule':'earliest eligible 5-minute UTC grid; spacing >=20min; exclusions before scores','source_asset':asset})
        receipts.append(asset)
        for i in indices:
            stamp=candidates[i]['selected_time_utc'];exclude.add(stamp)
            csvpath=out/f'mother_{day}_{i:03d}.csv'
            with csvpath.open('w',newline='',encoding='utf-8') as s:
                w=csv.writer(s);w.writerow(('aircraft_id','timestamp','lat','lon','altitude_m'));w.writerows(sorted(rows[i]))
            selected_by_role[role].append({'path':str(csvpath),'sha256':sha(csvpath),'selected_time_utc':stamp})
    for role,motherrows in selected_by_role.items():
        folder=out/role;folder.mkdir(exist_ok=True);refs=[]
        for m,mother in enumerate(motherrows):
            for n in COUNTS[role]:
                if stop():raise InterruptedError('source preparation stopped')
                arrays,provenance=build_aviation_scenes([Path(mother['path'])],[mother['selected_time_utc']],source_format='csv_snapshot',units={**CSV_UNITS,'persons':'count'},demand_model=model,seed=51001,max_record_age_seconds=60,environment_profile='europe_v1',partition_role=role,window_seconds=60,beam_count=n,min_positive_aircraft=200)
                base=arrays[0];snapshot=provenance['snapshots'][0]
                detail=folder/f'observations_{m:03d}_{n:03d}.jsonl.gz';detailsha=_detail_gzip(detail,snapshot.pop('source_observations'))
                for load in LOADS:
                    data=dict(base);data['rate']=np.asarray(base['rate'],dtype=np.float64)*load
                    # Builder sorts descending demand and freezes processing-index group.
                    dst=folder/f'scene_{m:03d}_{n:03d}_{int(load*100):03d}.npz';np.savez_compressed(dst,**data)
                    refs.append({'path':str(dst),'sha256':sha(dst),'mother_scene_id':mother['selected_time_utc'],'selected_time_utc':mother['selected_time_utc'],'beam_count':n,'load_multiplier':load,'partition_role':role,'source_sha256':mother['sha256'],'observation_detail':{'path':str(detail),'sha256':detailsha},'total_demand_bps':snapshot['total_demand_bps']*load,'covered_demand_bps':snapshot['covered_demand_bps']*load,'uncovered_demand_bps':snapshot['uncovered_demand_bps']*load})
            if progress:progress({'partition':role,'completed_mothers':m+1,'total_mothers':len(motherrows),'completed_scenes':len(refs)})
        expected=144 if role=='validation' else 1320
        if len(refs)!=expected:raise ValueError('prepared count mismatch')
        record={'id':'s2_f20_'+role,'path':str(folder),'count':expected,'files':refs,'profile':'europe_v1','source':{'experiment_id':'S2-F20','partition_role':role,'demand_model':model,'upstream_assets':partition_assets(role,receipts),'mother_count':len(motherrows),'excluded_mothers':sorted(config.get('exclude_mother_ids',[])),'selection_rule':'earliest eligible with >=20 minute spacing'}}
        write(folder/'record.json',record)
        datasets[role]={'path':str(folder/'record.json'),'sha256':sha(folder/'record.json'),'count':expected,'files':refs,'source':record['source']}
    write(out/'dataset_manifest.json',{'experiment_id':'S2-F20','source_assets':receipts,'datasets':datasets})
    return {'datasets':datasets,'manifest_path':str(out/'dataset_manifest.json'),'manifest_sha256':sha(out/'dataset_manifest.json')}
