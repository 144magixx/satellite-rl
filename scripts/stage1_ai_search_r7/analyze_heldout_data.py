"""Independent unlocked raw/profile identity audit; immutable role authority."""
from pathlib import Path
import numpy as np
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.r7_heldout_data import validate_recipe,unlock,source_arrays
from satellite_rl.workbench.r7_case_binding import validate_binding
from satellite_rl.workbench.r7_postprocess_source import new_output,execution_source
from satellite_rl.training.run_monitor import atomic_json

def analyze(manifest_refs,recipe_ref,unlock_gate_ref,role,output):
    recipe=validate_recipe(read_json(pin(recipe_ref),{}),role);unlock(recipe,unlock_gate_ref,role)
    from satellite_rl.shared_interface.content import read_wire,verify_self,content_sha
    from satellite_rl.shared_interface.contracts import Authority,validate_case
    rows={}
    for reference in manifest_refs:
        manifest=read_json(pin(reference),{})
        if manifest.get('phase')!=role+'_data' or manifest.get('recipe_ref')!=recipe_ref or manifest.get('unlock_gate_ref')!=unlock_gate_ref or manifest.get('algorithm_calls')!=0:raise ValueError('original unlocked build source only')
        for i,row in manifest['cases'].items():
            if i in rows:raise ValueError('duplicate heldout lane ownership')
            rows[i]=row
    if sorted(map(int,rows))!=list(range(len(recipe['rows']))):raise ValueError('complete original20/40 mother denominator')
    records={};ordered=[]
    for i,expected in enumerate(recipe['rows']):
        row=rows[str(i)];case=read_wire(pin(row['case_ref']).read_bytes());verify_self(case);validate_binding(case,role,i,row,expected)
        if row['recipe_sha256']!=recipe_ref['sha256'] or row['unlock_gate_ref']!=unlock_gate_ref or row['raw_ref']['sha256']!=expected['raw_ref']['sha256'] or row['metadata_ref']['sha256']!=expected['metadata_ref']['sha256']:raise ValueError('original immutable heldout recipe and full ancestor')
        transport={r['sha256']:r['path'] for r in (row['raw_ref'],row['metadata_ref'],row['container_ref'])}
        if row.get('transport_map_ref'):transport.update(read_json(pin(row['transport_map_ref']),{})['sha256_to_path'])
        else:transport=None
        ids,positions,planes,raw_ref,metadata_ref,container_ref=source_arrays(recipe,expected,transport)
        if case['aircraft_ids']!=ids or case['lat_lon_deg'].tobytes()!=positions.tobytes() or not np.array_equal(case['request_bps_int64'],np.full(len(ids),21600000,dtype='<i8')):raise ValueError('all full raw planes/order/positions/integer demand')
        if case['source'].get('raw_perplane_SHA')!=raw_ref['sha256'] or case['source'].get('original_container_SHA')!=container_ref['sha256']:raise ValueError('immutable content provenance differs')
        records[case['case_id']]=case;ordered.append(row)
    if len(records)!=len(ordered):raise ValueError('duplicate original mother case')
    output=new_output(Path(output)/'data_gate.json').parent
    registry=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={k:v['content_sha256'] for k,v in records.items()},layout_shas=[],allocation_shas=[])
    authority=Authority(registry,content_sha(registry))
    for case in records.values():validate_case(case,authority,geometry_check=False)
    authority_path=output/'authority.json';atomic_json(authority_path,registry);authority_ref=dict(path=str(authority_path),sha256=digest(authority_path))
    dataset_path=output/(role+'_dataset.json');atomic_json(dataset_path,dict(schema='R7UniformDatasetV1',role=role,count=len(ordered),cases=ordered,authority_ref=authority_ref,fixed_request_bps=21600000,recipe_ref=recipe_ref,unlock_gate_ref=unlock_gate_ref))
    dataset_ref=dict(path=str(dataset_path),sha256=digest(dataset_path),role=role,count=len(ordered))
    gate=dict(group='S1-AI17-R7',phase=role+'_data',passed=True,recipe_ref=recipe_ref,unlock_gate_ref=unlock_gate_ref,dataset_ref=dataset_ref,authority_ref=authority_ref,case_content_shas=registry['case_sha_by_id'],manifest_refs=manifest_refs,algorithm_calls=0,new_builds_in_audit=0,**execution_source())
    atomic_json(output/'data_gate.json',gate);return gate
