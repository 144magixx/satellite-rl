"""Unlocked role-only full raw ancestry derivation; profiles provide geometry proof."""
import gzip,json,os
from pathlib import Path
import numpy as np
from .coverage_ai_search import pin
from .storage import digest,read_json
from .r7_uniform_data import metadata_field,transported
from satellite_rl.training.run_monitor import atomic_json
from scripts.stage1_ai_search_r7.contract import DESIGN_SHA,REQUEST_BPS,ROLE_DATES,ROLE_COUNTS,require_ref

CONTAINERS={'validation':'f64258ba9dbb257c0107537e2839a6f28eaf27f8a75851106c1be7d61f8802e5','test':'1c289338eebd3b6e9d7d26fe307d6001381d667d9fa12092b453e2500e791b98'}

def validate_recipe(recipe,role):
    if role not in CONTAINERS or recipe.get('schema')!='R7HeldoutSourceRecipeV1' or recipe.get('role')!=role or recipe.get('design_SHA')!=DESIGN_SHA or recipe.get('fixed_request_bps')!=REQUEST_BPS or recipe.get('original_container_ref',{}).get('sha256')!=CONTAINERS[role]:raise ValueError('original sealed role/container/design recipe')
    require_ref(recipe['original_container_ref']);rows=recipe['rows']
    if len(rows)!=ROLE_COUNTS[role]:raise ValueError('one20/40 frozen mother; no four-profile expansion')
    mothers=[]
    for i,row in enumerate(rows):
        if row.get('actual_role')!=role or row.get('source_role')!=role or row.get('source_phase_index')!=i or row.get('UTC','')[:10] not in ROLE_DATES[role] or row.get('original_mother_id')!=role+':'+row['UTC'] or row.get('case_id')!=f"r7:{role}:{row['original_mother_id']}:uniform21600000":raise ValueError('frozen source actual role/date/index/UTC/mother')
        if type(row.get('source_aircraft_count'))is not int or row['source_aircraft_count']<1:raise ValueError('complete positive raw ancestor count')
        require_ref(row['raw_ref']);require_ref(row['metadata_ref'])
        if row.get('source_asset_SHA')!=row['raw_ref']['sha256'] or not isinstance(row.get('original_remote_ref'),str) or not row['original_remote_ref']:raise ValueError('immutable raw source SHA and historical source ID')
        if not all(isinstance(row.get(k),str) and row[k].startswith('/') for k in ('metadata_UTC_pointer','metadata_raw_SHA_pointer','metadata_scene_id_pointer','metadata_aircraft_count_pointer')):raise ValueError('authoritative metadata pointers required')
        profiles=row.get('profiles',[])
        if len(profiles)!=4 or len({p['profile'] for p in profiles})!=4:raise ValueError('four ancestor profiles only for equal ID/order/position check')
        for p in profiles:
            require_ref(p['scene_ref'])
            if not p.get('container_scene_ref_pointer','').startswith('/'):raise ValueError('profile asset must bind original container pointer')
        mothers.append(row['original_mother_id'])
    if len(set(mothers))!=len(mothers):raise ValueError('duplicate original mother forbidden')
    return recipe

def unlock(recipe,gate_ref,role):
    gate=read_json(pin(gate_ref),{})
    from .r7_successor_binding import gate_requirements,MODEL
    gate_requirements(role,gate,MODEL,recipe['fixed6000_checkpoint_ref'])
    if recipe.get('unlock_gate_ref')!=gate_ref:raise ValueError('root source recipe bound to exact original role-unlock gate')
    return gate

def source_arrays(recipe,row,transport=None,stop=None):
    """Called only after unlock(); never called by plan/recipe metadata audit."""
    if stop:stop()
    container_ref=transported(recipe['original_container_ref'],transport);container=read_json(pin(container_ref),{})
    metadata_ref=transported(row['metadata_ref'],transport);metadata=read_json(pin(metadata_ref),{})
    raw_ref=transported(row['raw_ref'],transport)
    if metadata_field(metadata,row['metadata_UTC_pointer'])!=row['UTC'] or metadata_field(metadata,row['metadata_raw_SHA_pointer'])!=raw_ref['sha256'] or metadata_field(metadata,row['metadata_scene_id_pointer'])!=row['original_scene_id'] or metadata_field(metadata,row['metadata_aircraft_count_pointer'])!=row['source_aircraft_count']:raise ValueError('authoritative original UTC/scene/raw/full selected count ancestry')
    with gzip.open(pin(raw_ref),'rt',encoding='utf8') as f:planes=[json.loads(line) for line in f]
    ids=[p['aircraft_id'] for p in planes];positions=np.asarray([[p['lat'],p['lon']] for p in planes],dtype='<f8')
    if len(ids)!=row['source_aircraft_count'] or len(set(ids))!=len(ids) or any(not isinstance(x,str) or not x for x in ids) or not np.isfinite(positions).all():raise ValueError('full raw plane count/ID/order/position; no filter or resample')
    for profile in row['profiles']:
        if stop:stop()
        historical=metadata_field(container,profile['container_scene_ref_pointer'])
        if historical!=profile['scene_ref']:raise ValueError('ancestor profile asset not in original container')
        ref=transported(profile['scene_ref'],transport)
        with np.load(pin(ref),allow_pickle=False) as z:
            if list(map(str,z['aircraft_ids']))!=ids or z['positions'].dtype!=positions.dtype or z['positions'].shape!=positions.shape or z['positions'].tobytes()!=positions.tobytes():raise ValueError('four profiles/full raw geometry mismatch; do not select a profile')
    return ids,positions,planes,raw_ref,metadata_ref,container_ref

def derive(recipe_ref,gate_ref,role,indices,directory,stop=None,transport_ref=None):
    recipe=validate_recipe(read_json(pin(recipe_ref),{}),role);unlock(recipe,gate_ref,role)
    from satellite_rl.shared_interface.stages import make_case
    from satellite_rl.shared_interface.content import wire_bytes
    transport=read_json(pin(transport_ref),{})['sha256_to_path'] if transport_ref else None
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);index=read_json(directory/'build_index.json',{})
    for i in indices:
        if type(i)is not int or not 0<=i<ROLE_COUNTS[role]:raise ValueError('original heldout index only')
        if stop:stop()
        row=recipe['rows'][i];target=directory/'cases'/str(i);target.mkdir(parents=True,exist_ok=True)
        if str(i) in index:
            prior=index[str(i)]
            if prior['recipe_sha256']!=recipe_ref['sha256'] or prior['unlock_gate_ref']!=gate_ref:raise ValueError('immutable original heldout derivation')
            pin(prior['case_ref']);continue
        if (target/'started.json').exists():raise ValueError('unknown heldout build retained; no repeat')
        atomic_json(target/'started.json',dict(recipe_ref=recipe_ref,unlock_gate_ref=gate_ref,case_index=i,one_build_reserved=True))
        ids,positions,planes,raw_ref,metadata_ref,container_ref=source_arrays(recipe,row,transport,stop)
        case=make_case(dict(case_id=row['case_id'],actual_role=role,source_role=row['source_role'],original_mother_id=row['original_mother_id'],UTC=row['UTC'],source_asset_SHA=raw_ref['sha256'],
                            source=dict(raw_perplane_SHA=raw_ref['sha256'],metadata_job_SHA=metadata_ref['sha256'],original_container_SHA=container_ref['sha256'],original_case_index=i,original_remote_ref=row['original_remote_ref'],original_scene_id=row['original_scene_id'],
                                        historical_profile_geometry_SHAs=[dict(profile=p['profile'],sha256=p['scene_ref']['sha256']) for p in row['profiles']],ancestry_fields=[{k:p[k] for k in ('timestamp_utc','source_date','source_sha256','altitude_m') if k in p} for p in planes],obsolete_request_assignment_fields_used=False)),ids,positions)
        path=target/'case.json';temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:f.write(wire_bytes(case));f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        result=dict(build_index=i,role=role,case_index=i,case_id=case['case_id'],original_mother_id=case['original_mother_id'],case_content_sha256=case['content_sha256'],aircraft_count=len(ids),total_request_bps_int64=len(ids)*REQUEST_BPS,
                    recipe_sha256=recipe_ref['sha256'],unlock_gate_ref=gate_ref,case_ref=dict(path=str(path),sha256=digest(path)),raw_ref=raw_ref,metadata_ref=metadata_ref,container_ref=container_ref,transport_map_ref=transport_ref)
        from .r7_case_binding import validate_binding
        validate_binding(case,role,i,result,row);index[str(i)]=result;atomic_json(directory/'build_index.json',index)
    return dict(group='S1-AI17-R7',phase=role+'_data',role=role,cases=index,recipe_ref=recipe_ref,unlock_gate_ref=gate_ref,builds=len(index),algorithm_calls=0,requires_independent_root_dataset_authority=True)
