"""Lossless train/dev raw-ancestor derivation. No profile input is ever read.

Recipe metadata is root supplied and file-pinned. UTC and mother identity are
read from that authoritative metadata, never inferred from aircraft timestamps.
Heldout roles are deliberately unavailable in this first-wave builder.
"""
import gzip
import json
import os
from pathlib import Path
import numpy as np
from .coverage_ai_search import pin
from .storage import digest
from satellite_rl.training.run_monitor import atomic_json
from scripts.stage1_ai_search_r7.contract import REQUEST_BPS, GROUP

def metadata_field(metadata, path):
    if not isinstance(path,str) or not path.startswith('/'):raise ValueError('explicit metadata JSON pointer required')
    value=metadata
    for key in path.split('/')[1:]:
        key=key.replace('~1','/').replace('~0','~')
        value=value[int(key) if isinstance(value,list) else key]
    return value

def validate_recipe(recipe):
    from scripts.stage1_ai_search_r7.contract import DESIGN_SHA,require_ref
    if recipe.get('schema')!='R7UniformSourceRecipeV1' or recipe.get('design_SHA')!=DESIGN_SHA or recipe.get('fixed_request_bps')!=REQUEST_BPS:raise ValueError('root uniform derivation recipe')
    require_ref(recipe['source_job_ref'])
    rows=recipe['rows']
    if len(rows)!=596 or recipe.get('actual_roles')!={'train':576,'development':20}:raise ValueError('complete596 original order')
    for i,row in enumerate(rows):
        role='train' if i<576 else 'development'
        source_role='train' if i<576 else 'validation'
        index=i if i<576 else i-576
        if row['actual_role']!=role or row['source_role']!=source_role or row['source_phase_index']!=index:raise ValueError('original source/actual role and index differ')
        if type(row.get('source_aircraft_count'))is not int or row['source_aircraft_count']<1:raise ValueError('original all-aircraft count required')
        require_ref({'path':row['source_local_path'],'sha256':row['source_asset_SHA']})
        if row['metadata_job_SHA']!=recipe['source_job_ref']['sha256'] or row['original_mother_id']!=source_role+':'+row['UTC'] or row['case_id']!=f"r7:{role}:{row['original_mother_id']}:uniform21600000":raise ValueError('root source/mother identity differs')
        for key in ('metadata_snapshot_JSON_pointer','metadata_file_JSON_pointer'):
            if not isinstance(row.get(key),str) or not row[key].startswith('/'):raise ValueError('authoritative metadata pointer')
    return recipe

def transported(ref, transport=None):
    if transport is not None:
        value=transport.get(ref['sha256'])
        if not isinstance(value,str) or not value:raise ValueError('root frozen transport mapping missing')
        ref={**ref,'path':value}
    pin(ref)
    return ref

def construct(row, metadata_ref, transport=None):
    """Build one new typed object; caller freezes result before algorithm reads."""
    from satellite_rl.shared_interface.stages import make_case
    from satellite_rl.shared_interface.content import content_sha
    from satellite_rl.shared_interface.contracts import Authority,validate_case
    raw_ref=transported({'path':row['source_local_path'],'sha256':row['source_asset_SHA']},transport)
    metadata_ref=transported(metadata_ref,transport)
    raw_path=pin(raw_ref);metadata=json.loads(pin(metadata_ref).read_text(encoding='utf-8'))
    snapshot=metadata_field(metadata,row['metadata_snapshot_JSON_pointer'])
    file_record=metadata_field(metadata,row['metadata_file_JSON_pointer'])
    utc=snapshot['selected_time_utc'];mother=row['source_role']+':'+utc
    if file_record['scene_id']!=row['original_scene_id'] or row['metadata_job_SHA']!=metadata_ref['sha256']:raise ValueError('metadata scene/source mismatch')
    if snapshot['source_observations_ref']['sha256']!=raw_ref['sha256'] or snapshot['counts']!=row['source_selection_counts']:raise ValueError('original snapshot raw/count chain differs')
    if mother!=row['original_mother_id'] or utc!=row['UTC']:raise ValueError('root metadata identity mismatch')
    role=row['actual_role']
    if role not in ('train','development'):raise ValueError('heldout role locked')
    planes=[]
    with gzip.open(raw_path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():raise ValueError('empty aircraft row')
            planes.append(json.loads(line))
    if len(planes)!=row['source_aircraft_count'] or snapshot['counts']['selected_aircraft']!=len(planes):raise ValueError('original full aircraft count differs; no delete/dedup')
    ids=[p['aircraft_id'] for p in planes]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):raise ValueError('stable IDs missing/duplicate')
    positions=np.asarray([[p['lat'],p['lon']] for p in planes],dtype='<f8')
    if not np.isfinite(positions).all() or positions.shape!=(len(ids),2):raise ValueError('invalid raw positions')
    source=dict(raw_perplane_SHA=raw_ref['sha256'],metadata_job_SHA=metadata_ref['sha256'],
                original_remote_ref=row['source_remote_path'],original_case_index=row['source_phase_index'],
                original_scene_id=row['original_scene_id'],original_mother_id_derivation=row['original_mother_id_derivation'],
                metadata_snapshot_JSON_pointer=row['metadata_snapshot_JSON_pointer'],metadata_file_JSON_pointer=row['metadata_file_JSON_pointer'],
                ancestry_fields=[{k:p[k] for k in ('timestamp_utc','source_date','source_sha256','altitude_m') if k in p} for p in planes],
                obsolete_request_assignment_fields_used=False)
    case_id=f'r7:{role}:{mother}:uniform21600000'
    record=make_case(dict(case_id=case_id,actual_role=role,source_role=row['source_role'],original_mother_id=mother,
                         UTC=utc,source_asset_SHA=raw_ref['sha256'],source=source),ids,positions)
    # This provisional authority validates a derivation, but is never an
    # algorithm admission. The finished manifest requires independent root pin.
    provisional=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={case_id:record['content_sha256']},layout_shas=[],allocation_shas=[])
    validate_case(record,Authority(provisional,content_sha(provisional)))
    return record

def derive(recipe_ref, indices, directory, transport_ref=None):
    from satellite_rl.shared_interface.content import wire_bytes
    recipe=json.loads(pin(recipe_ref).read_text(encoding='utf-8'));validate_recipe(recipe)
    transport=json.loads(pin(transport_ref).read_text(encoding='utf-8'))['sha256_to_path'] if transport_ref else None
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    index_path=directory/'build_index.json';index=json.loads(index_path.read_text()) if index_path.exists() else {}
    for i in indices:
        if type(i)is not int or i not in range(596):raise ValueError('first wave train/dev only')
        row=recipe['rows'][i];target=directory/'cases'/str(i);target.mkdir(parents=True,exist_ok=True)
        if str(i) in index:
            old=index[str(i)]
            if old['recipe_sha256']!=recipe_ref['sha256'] or old['build_index']!=i:raise ValueError('old derivation identity changed')
            pin(old['case_ref']);continue
        if (target/'started.json').exists():raise ValueError('unknown data build preserved; root reconciliation required')
        atomic_json(target/'started.json',dict(recipe_ref=recipe_ref,build_index=i,one_build_reserved=True))
        case=construct(row,recipe['source_job_ref'],transport);path=target/'case.json';temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:stream.write(wire_bytes(case));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        record=dict(build_index=i,role=case['actual_role'],case_index=row['source_phase_index'],case_id=case['case_id'],
                    original_mother_id=case['original_mother_id'],case_content_sha256=case['content_sha256'],
                    aircraft_count=len(case['aircraft_ids']),total_request_bps_int64=case['total_request_bps_int64'],
                    recipe_sha256=recipe_ref['sha256'],case_ref=dict(path=str(path),sha256=digest(path)),
                    raw_ref=transported({'path':row['source_local_path'],'sha256':row['source_asset_SHA']},transport),
                    metadata_ref=transported(recipe['source_job_ref'],transport),transport_map_ref=transport_ref)
        index[str(i)]=record;atomic_json(index_path,index)
    return dict(group=GROUP,phase='data_train_dev',cases=index,builds=len(index),algorithm_reads=0,
                requires_independent_root_manifest=True,validation_test_reads=0)

def load_algorithm_case(case_ref, authority_ref):
    """Only independent root-frozen authority enables raw-bps algorithm input."""
    from satellite_rl.shared_interface.content import read_wire,content_sha
    from satellite_rl.shared_interface.contracts import Authority
    from satellite_rl.shared_interface.stages import algorithm_scene
    authority=json.loads(pin(authority_ref).read_text(encoding='utf-8'))
    record=read_wire(pin(case_ref).read_bytes())
    if record['actual_role'] not in ('train','development'):raise ValueError('heldout algorithm role locked')
    return record,algorithm_scene(record,Authority(authority,content_sha(authority)))
