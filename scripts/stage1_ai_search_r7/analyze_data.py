"""Independent derivation audit and manifest publication; never builds new cases."""
import gzip,json
from pathlib import Path
import numpy as np
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.r7_uniform_data import validate_recipe,metadata_field
from satellite_rl.training.run_monitor import atomic_json
from .contract import GROUP,REQUEST_BPS

def analyze(manifest_refs,recipe_ref,output):
    from satellite_rl.shared_interface.content import read_wire,verify_self,content_sha
    from satellite_rl.shared_interface.contracts import Authority,validate_case
    recipe=read_json(pin(recipe_ref),{});validate_recipe(recipe);rows={}
    for ref in manifest_refs:
        manifest=read_json(pin(ref),{})
        if manifest.get('group')!=GROUP or manifest.get('phase')!='data_train_dev':raise ValueError('actual derivation manifest')
        for i,row_ref in manifest['cases'].items():
            if i in rows:raise ValueError('duplicate lane owner')
            rows[i]=read_json(pin(row_ref),{})
    if sorted(map(int,rows))!=list(range(596)):raise ValueError('all596 new train/dev source identities required')
    records={}
    for i in range(596):
        row=rows[str(i)];expected=recipe['rows'][i]
        if row['recipe_sha256']!=recipe_ref['sha256'] or row['build_index']!=i:raise ValueError('root frozen recipe identity')
        record=read_wire(pin(row['case_ref']).read_bytes());verify_self(record)
        for key in ('case_id','original_mother_id','UTC','source_asset_SHA','actual_role','source_role'):
            if record[key]!=expected[key]:raise ValueError('original scene identity changed')
        with gzip.open(pin(row['raw_ref']),'rt',encoding='utf8') as stream:planes=[json.loads(line) for line in stream]
        metadata=read_json(pin(row['metadata_ref']),{});snapshot=metadata_field(metadata,expected['metadata_snapshot_JSON_pointer'])
        if snapshot['selected_time_utc']!=record['UTC'] or row['metadata_ref']['sha256']!=recipe['source_job_ref']['sha256'] or row['raw_ref']['sha256']!=expected['source_asset_SHA']:raise ValueError('actual full original ancestry')
        ids=[plane['aircraft_id'] for plane in planes];positions=np.asarray([[p['lat'],p['lon']] for p in planes],dtype='<f8')
        if ids!=record['aircraft_ids'] or len(ids)!=expected['source_aircraft_count'] or positions.tobytes()!=record['lat_lon_deg'].tobytes():raise ValueError('all original IDs/order/position bytes required')
        if record['source'].get('raw_perplane_SHA')!=row['raw_ref']['sha256'] or record['source'].get('metadata_job_SHA')!=row['metadata_ref']['sha256']:raise ValueError('content raw/source SHA')
        if not np.array_equal(record['request_bps_int64'],np.full(len(ids),REQUEST_BPS,dtype='<i8')) or record['total_request_bps_int64']!=len(ids)*REQUEST_BPS:raise ValueError('integer uniform all-plane conservation')
        records[record['case_id']]=record
    if len(records)!=596:raise ValueError('duplicate mother/case identity')
    registry=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={k:r['content_sha256'] for k,r in records.items()},layout_shas=[],allocation_shas=[])
    authority=Authority(registry,content_sha(registry))
    for record in records.values():validate_case(record,authority,geometry_check=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);authority_path=output/'authority.json';atomic_json(authority_path,registry)
    authority_ref=dict(path=str(authority_path),sha256=digest(authority_path));datasets={}
    for role,count in (('train',576),('development',20)):
        selected=[rows[str(i)] for i in range(596) if rows[str(i)]['role']==role]
        if len(selected)!=count or [r['case_index'] for r in selected]!=list(range(count)):raise ValueError('exact original role order')
        path=output/(role+'_dataset.json');atomic_json(path,dict(schema='R7UniformDatasetV1',role=role,count=count,cases=selected,authority_ref=authority_ref,fixed_request_bps=REQUEST_BPS,recipe_ref=recipe_ref))
        datasets[role]=dict(path=str(path),sha256=digest(path),role=role,count=count)
    result=dict(group=GROUP,phase='data_train_dev',passed=True,cases=596,source_records=sum(r['aircraft_count'] for r in rows.values()),
                recipe_ref=recipe_ref,authority_ref=authority_ref,datasets=datasets,manifest_refs=manifest_refs,
                new_builds_in_audit=0,algorithm_calls=0,geometry_calls=0,validation_test_reads=0)
    atomic_json(output/'data_gate.json',result);return result
