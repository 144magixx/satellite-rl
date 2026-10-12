"""Workbench-only, pinned position/request preparation for S1-G16.

Old layouts and capacity metadata are never inputs. Official trace parsing uses
the SHA-pinned parser; demand synthesis reproduces the aviation adapter's
timestamp stream directly, including zero-demand aircraft, without its cover.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from contextlib import contextmanager
import numpy as np
from ..geometry import satellite_rays_numpy, ray_angle_numpy, visible_numpy, squared_ray_limit
from .core import Scene

PROFILES = ('base', 'one_hotspot', 'two_hotspots', 'edge_heavy')
DEMAND_MODEL = {'persons_per_aircraft':180, 'satellite_access_fraction':.2,
    'active_fraction':.3, 'rate_mbps_per_active_person':2., 'aircraft_base_mbps':0,
    'sampling':'binomial', 'lognormal_sigma':.35, 'window_seconds':60,
    'parameter_source':'engineering assumption, not measured occupants or requests'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def _pin(item):
    path = Path(item['path'])
    if digest(path) != item['sha256']: raise ValueError('frozen source SHA differs: ' + str(path))
    return path

def _json(path, record):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

@contextmanager
def _source_lock(path):
    """Kernel ownership releases on crash; retained source markers are evidence."""
    with Path(path).open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0); stream.write(b'\0'); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            if os.name == 'nt': stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream,fcntl.LOCK_UN)

def demand_profiles(positions, demands, ids):
    """Input-only satellite-angle profiles; each preserves original total D."""
    scene = Scene(positions, demands); p = scene.positions; d = scene.demands
    n = len(d); total = float(d.sum())
    if len(ids) != n or len(set(map(str, ids))) != n: raise ValueError('aircraft IDs must be unique')
    if not n: return {name: d.copy() for name in PROFILES}
    rays = satellite_rays_numpy(p)
    # Count neighborhoods one at a time, avoiding a dense aircraft-square array.
    scores = np.empty(n,dtype=np.int64)
    limit = squared_ray_limit(.5 + 1e-12)
    for first_index in range(0,n,128):
        block=rays[first_index:first_index+128]
        scores[first_index:first_index+len(block)] = np.sum(np.sum((block[:,None]-rays[None])**2,axis=-1)<=limit,axis=1)
    first = sorted(range(n), key=lambda i: (-scores[i], str(ids[i])))[0]
    distance = ray_angle_numpy(rays, rays[first])
    second = sorted(range(n), key=lambda i: (-distance[i], str(ids[i])))[0]
    near1 = distance <= .5 + 1e-12
    near2 = ray_angle_numpy(rays, rays[second]) <= .5 + 1e-12
    mean = rays.mean(axis=0)
    if np.linalg.norm(mean) < 1e-12: mean = rays[min(range(n), key=lambda i: str(ids[i]))]
    mean = mean / np.linalg.norm(mean)
    edge_distance = ray_angle_numpy(rays, mean)
    edge = sorted(range(n), key=lambda i: (-edge_distance[i], str(ids[i])))[:max(1, int(np.ceil(.05*n)))]
    weights = {'base': np.ones(n), 'one_hotspot': np.where(near1, 4., 1.),
               'two_hotspots': np.where(near1 | near2, 4., 1.), 'edge_heavy': np.ones(n)}
    weights['edge_heavy'][edge] = 8.
    results = {}
    for name, weight in weights.items():
        result = d * weight
        if total: result *= total / result.sum()
        if not np.isclose(result.sum(), total, rtol=1e-12, atol=1e-6): raise ValueError('profile demand not conserved')
        results[name] = result
    return results

def _requests(rows, stamp):
    rows = sorted(rows, key=lambda row: str(row[0]))
    ids = [str(row[0]).strip().lower() for row in rows]
    if len(set(ids)) != len(ids): raise ValueError('duplicate aircraft IDs')
    positions = np.asarray([[r[2], r[3]] for r in rows], dtype=float).reshape(-1, 2)
    canonical = datetime.fromisoformat(stamp.replace('Z', '+00:00')).isoformat().replace('+00:00', 'Z')
    stream = hashlib.sha256(canonical.encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence([51001, *np.frombuffer(stream, dtype='<u4').tolist()]))
    active = rng.binomial(np.full(len(rows), 180), .2 * .3)
    demand = active * 2e6
    demand = demand * np.exp(rng.normal(-.35**2/2, .35, len(rows)))
    Scene(positions, demand)
    return positions, demand, ids

def _download(spec, output, progress, stopped):
    item = spec['archive']; root = Path(spec['raw_shared_root']); root.mkdir(parents=True, exist_ok=True)
    archive = root / item['name']
    if Path(item['name']).name != item['name']: raise ValueError('archive name must be basename')
    expected = int(item['expected_bytes'])
    if expected <= 0: raise ValueError('positive archive size required')
    if archive.exists():
        if archive.stat().st_size != expected: raise ValueError('existing official archive size mismatch')
        return archive
    lock = archive.with_name(archive.name + '.download.lock')
    owner = str(output.resolve())
    guard = _source_lock(lock); guard.__enter__()
    partial = archive.with_name(archive.name + '.part')
    marker = archive.with_name(archive.name + '.part.owner.json')
    try:
        if marker.exists() and json.loads(marker.read_text())['owner'] != owner:
            raise ValueError('partial official download belongs to a different ID')
        _json(marker, {'owner':owner, 'asset_id':item['asset_id'], 'url':item['url'], 'expected_bytes':expected})
        for attempt in range(4):
            if stopped(): return None
            if partial.exists() and partial.stat().st_size == expected:
                partial.replace(archive); return archive
            with (output / 'download.log').open('ab') as log:
                child = subprocess.Popen(['curl', '--location', '--fail', '--silent', '--show-error',
                    '--connect-timeout', '20', '--max-time', '1800', '--continue-at', '-',
                    '--output', str(partial), item['url']], stdout=log, stderr=log)
                try:
                    while child.poll() is None:
                        progress(stage='download', attempt=attempt+1,
                                 downloaded_bytes=partial.stat().st_size if partial.exists() else 0,
                                 expected_bytes=expected)
                        if stopped(): return None
                        time.sleep(2)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
            if child.returncode == 0 and partial.exists() and partial.stat().st_size == expected:
                partial.replace(archive); return archive
        raise RuntimeError('official download unavailable; same-ID .part retained')
    finally: guard.__exit__(None,None,None)

def _official(spec, output, source_root, progress, stopped):
    pins = spec.get('metadata_pins', {})
    if not pins: raise ValueError('official metadata pins required')
    for item in pins.values(): _pin(item)
    parser = _pin(spec['parser']); day = spec['date']; instance = spec['source_instance'].removeprefix('planes-readsb-')
    item = spec['archive']
    expected_name = f"v{day.replace('-', '.')}-planes-readsb-{instance}.tar"
    if instance not in ('prod-0', 'staging-0') or item['name'] != expected_name:
        raise ValueError('preferred source instance/date/archive differ')
    releases = [p for name, p in pins.items() if name in ('release_api', 'release_api.json')]
    if releases:
        release = json.loads(Path(releases[0]['path']).read_text(encoding='utf-8'))
        assets = [a for a in release.get('assets', []) if a.get('name') == item['name']]
        if len(assets) != 1 or any(assets[0].get(key) != item[value] for key, value in
                    (('id','asset_id'),('size','expected_bytes'),('browser_download_url','url'))):
            raise ValueError('official release asset differs')
    preferred=[p for name,p in pins.items() if name in ('preferred_releases','PREFERRED_RELEASES.txt')]
    if preferred and item['url'] not in Path(preferred[0]['path']).read_text(encoding='utf-8'):
        raise ValueError('archive URL differs from pinned preferred releases')
    archive = _download(spec, output, progress, stopped)
    if archive is None: return None, {'status':'stopped'}
    archive_sha = digest(archive)
    if item.get('sha256') and archive_sha != item['sha256']: raise ValueError('official archive SHA mismatch')
    work = output / 'parser'; raw = work / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    linked = raw / archive.name
    if not linked.exists():
        try: linked.hardlink_to(archive)
        except OSError: linked.symlink_to(archive)
    if not linked.samefile(archive): raise ValueError('raw archive link does not bind pinned shared source')
    os.environ['STAGE1_FROZEN_SRC'] = str(Path(source_root) / 'src')
    loader = importlib.util.spec_from_file_location('g16_pinned_trace', parser)
    reference = importlib.util.module_from_spec(loader); loader.loader.exec_module(reference)
    reference.ROOT = work; reference.STATUS = work / 'trace_progress.json'
    reference.DATES = (day,); reference.ROLES = (spec['role'],); reference.MAX_AGE = 60.
    reference.SOURCE_INSTANCE = instance
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(minutes=2)
    times = [start + timedelta(minutes=5*i) for i in range(288)]
    reference.candidate_times = lambda unused: times
    original_write = reference.write_json
    def write_status(path, value):
        if stopped(): raise InterruptedError('dataset stopped')
        original_write(path, value)
        progress(stage='parse', **{key:value[key] for key in ('files_processed','source_records') if key in value})
    reference.write_json = write_status
    rows, candidates = reference.process_day(day, {'status':'processing'})
    if len(rows) != 288 or len(candidates) != 288: raise ValueError('parser changed frozen 288 grid')
    audit = []; eligible = []; zero_demand_nonempty = []
    for index, (rowset, candidate) in enumerate(zip(rows, candidates)):
        stamp = times[index].isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        if candidate['selected_time_utc'] != stamp: raise ValueError('parser selected time differs')
        positions = np.asarray([[r[2], r[3]] for r in rowset], dtype=float).reshape(-1,2)
        legal = bool(np.isfinite(positions).all() and visible_numpy(positions).all())
        _, request, ids = _requests(rowset, stamp)
        total = float(request.sum())
        reason = ('empty_after_trace_filters' if not rowset else 'invalid_input_geometry' if not legal
                  else 'zero_total_requested_demand' if total == 0 else None)
        audit.append({**candidate, 'eligible': reason is None, 'rejection_reason':reason,
                      'total_requested_demand':total, 'zero_demand_aircraft':int(np.sum(request==0))})
        if reason is None: eligible.append(index)
        if rowset and total == 0:
            zero_demand_nonempty.append({'selected_time_utc':stamp, 'aircraft_ids':ids,
                                        'positions':positions.tolist(), 'demands':request.tolist()})
    _json(output / 'all_288_candidate_audit.json', audit)
    _json(output / 'zero_demand_nonempty.json', zero_demand_nonempty)
    count = int(spec.get('count', spec.get('expected_mothers', 0)))
    provenance = {'archive_path':str(archive), 'archive_sha256':archive_sha,
                  'parser_sha256':spec['parser']['sha256'], 'metadata_pins':pins,
                  'grid':'00:02 UTC + 5*i minutes, i=0..287', 'max_age_seconds':60,
                  'candidate_audit_path':str(output/'all_288_candidate_audit.json'),
                  'candidate_audit_sha256':digest(output/'all_288_candidate_audit.json'),
                  'zero_demand_nonempty_path':str(output/'zero_demand_nonempty.json'),
                  'zero_demand_nonempty_count':len(zero_demand_nonempty),
                  'eligible_count':len(eligible),
                  'selection_rule':'time-ascending nonempty legal positive-total inputs; floor((i+.5)*eligible_count/count)',
                  'no_outcome_based_replacement':True,
                  'parser_summary_path':str(work/'candidates'/day/'summary.json'),
                  'parser_summary_sha256':digest(work/'candidates'/day/'summary.json')}
    if len(eligible) < count:
        return None, {'status':'data_unavailable', 'expected_mothers':count, 'available_mothers':len(eligible), **provenance}
    chosen = [eligible[int((i+.5)*len(eligible)/count)] for i in range(count)]
    provenance['selected_candidate_indices']=chosen
    provenance['selected_times_utc']=[candidates[i]['selected_time_utc'] for i in chosen]
    mothers = []
    for index in chosen:
        if stopped(): return None, {'status':'stopped', **provenance}
        stamp = candidates[index]['selected_time_utc']; p, d, ids = _requests(rows[index], stamp)
        mothers.append((p,d,ids, f"{spec['role']}:{stamp}", {**provenance,
            'selected_time_utc':stamp, 'candidate_index':index, 'counts':audit[index]['counts'],
            'aircraft_observation_times':[row[1] for row in sorted(rows[index])], 'demand_model':DEMAND_MODEL}))
        progress(stage='convert', prepared_mothers=len(mothers), expected_mothers=count)
    return mothers, {'status':'prepared', **provenance}

def _merge_packed(spec, output, progress, stopped):
    chunks=[]; arrays={name:[] for name in ('positions','demands','case_ids','mother_ids','demand_profiles','aircraft_ids')}
    offsets=[0]; mother_metadata=[]
    for reference in spec['sources']:
        if stopped(): return {'status':'stopped'}
        path = _pin(reference)
        with np.load(path, allow_pickle=False) as values:
            meta=json.loads(str(values['metadata_json']))
            if meta.get('format_version') != 'geoload_aircraft_v1' or meta.get('role') != spec['role']:
                raise ValueError('packed merge role/schema differs')
            raw={name:values[name].copy() for name in arrays}
            local=values['offsets'].copy()
        if local.shape != (len(raw['case_ids'])+1,) or local[0] != 0 or local[-1] != len(raw['demands']) or np.any(np.diff(local)<0):
            raise ValueError('packed source offsets invalid')
        if raw['positions'].shape != (len(raw['demands']),2) or len(raw['aircraft_ids']) != len(raw['demands']):
            raise ValueError('packed source aircraft shapes invalid')
        if len(raw['mother_ids']) != len(raw['case_ids']) or len(raw['demand_profiles']) != len(raw['case_ids']):
            raise ValueError('packed source case shapes invalid')
        for name in arrays: arrays[name].append(raw[name])
        offsets.extend((local[1:]+offsets[-1]).tolist()); mother_metadata.extend(meta['mothers']); chunks.append(reference)
    joined={name:np.concatenate(items) for name,items in arrays.items()}
    if len(set(joined['case_ids'].tolist())) != len(joined['case_ids']): raise ValueError('duplicate merged case IDs')
    unique=set(joined['mother_ids'].tolist()); expected=int(spec.get('count',spec.get('expected_mothers',0)))
    if len(unique)!=expected or len(mother_metadata)!=expected or len({m['mother_id'] for m in mother_metadata})!=expected:
        raise ValueError('merged mother identity/count differs')
    for mother in unique:
        profiles=joined['demand_profiles'][joined['mother_ids']==mother].tolist()
        if sorted(profiles)!=sorted(PROFILES): raise ValueError('merged mother profile set differs')
    metadata={'format_version':'geoload_aircraft_v1','role':spec['role'],'kind':'merge_existing',
              'mothers':mother_metadata,'provenance':{'packed_sources':chunks,'numerical_transform':False},
              'profile_names':PROFILES,'geometry_only':True,'request_demand_only':True}
    temp=output/'inputs.part.npz'; target=output/'inputs.npz'
    np.savez_compressed(temp,**joined,offsets=np.asarray(offsets,dtype=np.int64),
                        metadata_json=np.asarray(json.dumps(metadata,ensure_ascii=False,allow_nan=False)))
    temp.replace(target)
    record={'status':'prepared','path':str(target),'sha256':digest(target),'mother_count':expected,
            'count':len(joined['case_ids']),'role':spec['role'],'format_version':'geoload_aircraft_v1','source':metadata}
    _json(output/'preparation_record.json',record);progress(stage='prepared',mother_count=expected,count=record['count'])
    return record

def prepare_dataset(spec, output_dir, source_root, progress=lambda **kw: None, stopped=lambda:False):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if spec.get('role') not in ('train','development','dev','validation','test'): raise ValueError('dataset role required')
    if spec.get('seed',51001) != 51001: raise ValueError('G16 demand seed is frozen at 51001 for every role')
    count = int(spec.get('count', spec.get('expected_mothers', 0)))
    if count <= 0: raise ValueError('positive mother count required')
    if spec['kind'] == 'merge_existing': return _merge_packed(spec,output,progress,stopped)
    if spec['kind'] == 'import_existing':
        from ..variable_dataset import _loaded_arrays
        sources = spec.get('sources', [{'path':spec.get('path'), 'sha256':spec.get('sha256')}])
        mothers = []
        for item in sources:
            if stopped(): return {'status':'stopped'}
            dataset = _loaded_arrays(_pin(item))
            source_role=dataset.metadata.get('partition_role')
            expected_role=item.get('expected_role')
            if expected_role and source_role != expected_role: raise ValueError('import source role differs from pinned role')
            allowed={'train':{'train'},'development':{'development','validation','dev'},'dev':{'development','validation','dev'},
                     'validation':{'validation'},'test':{'test'}}[spec['role']]
            if source_role and source_role not in allowed: raise ValueError('import source role would leak a different partition')
            for scene in dataset.scenes:
                mother = str(scene.metadata.get('selected_time_utc') or scene.scene_id)
                meta = {key:value for key,value in scene.metadata.items() if not any(
                    term in key.lower() for term in ('capacity','initial','beam','center','carry','qos','satisfaction'))}
                original_ids=list(map(str,scene.aircraft_ids))
                order=np.argsort(np.asarray(original_ids,dtype=str),kind='stable')
                canonical_ids=[original_ids[i] for i in order]
                mothers.append((scene.positions[order].copy(), scene.demands[order].copy(), canonical_ids,
                    f"{spec['role']}:{mother}", {**meta, 'import_path':item['path'], 'import_sha256':item['sha256'],
                        'source_hash':scene.source_hash, 'source_partition_role':source_role,
                        'aircraft_order':'stable_lexical_aircraft_ID', 'source_aircraft_ids':original_ids,
                        'canonical_to_source_index':order.tolist(),
                        'legacy_layout_and_capacity_discarded':True}))
        if len(mothers) != count: raise ValueError('import mother count differs from frozen expected count')
        provenance = {'status':'prepared', 'import_sources':sources}
    elif spec['kind'] == 'official_day':
        try: mothers, provenance = _official(spec, output, source_root, progress, stopped)
        except InterruptedError: return {'status':'stopped'}
        if mothers is None:
            _json(output/'preparation_record.json', provenance); return provenance
    else: raise ValueError('unknown dataset preparation kind')
    if len({m[3] for m in mothers}) != len(mothers): raise ValueError('duplicate mother IDs')
    positions=[]; demands=[]; aircraft=[]; offsets=[0]; case_ids=[]; mother_ids=[]; profiles=[]; metadata=[]
    for p,d,ids,mother,meta in mothers:
        if stopped(): return {'status':'stopped'}
        if not visible_numpy(p).all(): raise ValueError('import includes invisible aircraft; no silent deletion')
        variants = demand_profiles(p,d,ids)
        metadata.append({'mother_id':mother, **meta, 'aircraft_count':len(d),
            'zero_demand_aircraft':int(np.sum(d==0)), 'total_demand':float(np.sum(d)),
            'profiles_equal_base':[name for name,value in variants.items() if np.array_equal(value,d)]})
        for name, variant in variants.items():
            positions.append(p); demands.append(variant); aircraft.extend(map(str,ids)); offsets.append(offsets[-1]+len(d))
            case_ids.append(mother+':'+name); mother_ids.append(mother); profiles.append(name)
    metadata_record = {'format_version':'geoload_aircraft_v1', 'role':spec['role'], 'kind':spec['kind'],
                       'mothers':metadata, 'provenance':provenance, 'profile_names':PROFILES,
                       'geometry_only':True, 'request_demand_only':True}
    temp = output/'inputs.part.npz'; target=output/'inputs.npz'
    np.savez_compressed(temp, positions=np.concatenate(positions) if positions else np.empty((0,2)),
        demands=np.concatenate(demands) if demands else np.empty(0), offsets=np.asarray(offsets,dtype=np.int64),
        case_ids=np.asarray(case_ids,dtype=str), mother_ids=np.asarray(mother_ids,dtype=str),
        demand_profiles=np.asarray(profiles,dtype=str), aircraft_ids=np.asarray(aircraft,dtype=str),
        metadata_json=np.asarray(json.dumps(metadata_record, ensure_ascii=False, allow_nan=False)))
    temp.replace(target)
    record={'status':'prepared', 'path':str(target), 'sha256':digest(target), 'mother_count':len(mothers),
            'count':len(case_ids), 'role':spec['role'], 'format_version':'geoload_aircraft_v1', 'source':metadata_record}
    _json(output/'preparation_record.json',record)
    progress(stage='prepared', mother_count=len(mothers), count=len(case_ids))
    return record
