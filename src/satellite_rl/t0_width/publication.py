"""Independent full636 Wenv geometry qualification; never signs root authority.

Confirmation details and candidate registry/test manifest stay in sealed output.
Public progress contains counts only. Primitive preparation and all model/PHY
calls are zero; at most636 independently validated candidate layouts are read.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from satellite_rl.shared_interface.content import read_wire, content_sha
from satellite_rl.shared_interface.contracts import validate_layout
from .data import (COUNTS, read_ref, read_json_ref, load_case, validate_manifest,
                   file_sha, _json_once)

CORE_SHA='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5'


def _same_ref(first, second):
    if first.get('path')!=second.get('path') or first.get('sha256')!=second.get('sha256'):
        raise ValueError('exact pinned publication path/SHA differs')


def validate_audit_inputs(config):
    manifest=validate_manifest(read_json_ref(config['prepare_manifest_ref']))
    admission=read_json_ref(config['audit_admission_ref'])
    if (admission.get('schema')!='W26DataAuditExecutionAdmissionV1' or admission.get('root_approved')is not True
            or admission.get('source_commit')!=config['source_commit']
            or admission.get('manifest_SHA')!=config['prepare_manifest_ref']['sha256']
            or admission.get('core_SHA')!=CORE_SHA
            or manifest['source_SHA']!=CORE_SHA or admission.get('expected_counts')!=COUNTS
            or admission.get('max_layout_checks')!=636):
        raise ValueError('exact independent root audit execution admission required')
    ports=[admission[k] for k in ('remote_port','executor_port') if k in admission]
    if not ports or any(p!=config['executor_port'] or p not in (20418,20424) for p in ports):
        raise ValueError('audit execution lane differs')
    if len(config['shard_index_refs'])!=4 or admission.get('shard_index_SHAs')!=[r['sha256']for r in config['shard_index_refs']]:
        raise ValueError('four exact root-pinned shard indices required')
    expected_native=sum(bool(r.get('native_wenv_admission_ref'))for r in manifest['rows'])
    if expected_native!=594 or admission.get('expected_native_imports')!=594 or admission.get('expected_new_prepare')!=42:
        raise ValueError('native594/new42 frozen accounting differs')
    for ref in manifest['source_file_refs']:
        if file_sha(ref['path'])!=ref['sha256']:
            raise ValueError('original preparation source member differs')
    if not any(ref['sha256']==CORE_SHA and Path(ref['path']).as_posix().endswith('/coverage/ai_search_r5/core.py')
               for ref in manifest['source_file_refs']):
        raise ValueError('exact original Linux/Git Wenv core bytes required in source closure')
    rows={};native=new=0
    for ref in config['shard_index_refs']:
        shard=read_json_ref(ref)
        if shard.get('schema')!='W26PreparationShardIndexV1' or shard.get('manifest_SHA')!=admission['manifest_SHA']:
            raise ValueError('shard preparation parent differs')
        indices=shard['case_indices'];aggregate=read_json_ref(shard['aggregate_ref'])
        if not isinstance(indices,list) or any(type(i)is not int or not 0<=i<636 for i in indices):
            raise ValueError('invalid root shard indices')
        if (aggregate.get('schema')!='W26DataPreparationAggregateV1' or aggregate.get('manifest_SHA')!=admission['manifest_SHA']
                or aggregate.get('case_indices')!=indices or aggregate.get('completed')!=len(indices)
                or aggregate.get('total')!=len(indices) or aggregate.get('candidate_ready')is not True
                or aggregate.get('all_dataset_total')!=636 or aggregate.get('reused_published')!=0
                or any(aggregate.get(k)!=0 for k in ('T_calls','D5_calls','MILP_calls'))):
            raise ValueError('shard complete result/call ledger differs')
        if [r['case_index']for r in shard['rows']]!=indices or len(set(indices))!=len(indices):
            raise ValueError('exact original ordered shard rows required')
        expected=sum(bool(manifest['rows'][i].get('native_wenv_admission_ref'))for i in indices)
        if aggregate.get('native_Wenv_imports')!=expected or aggregate.get('new_prepare_started')!=len(indices)-expected:
            raise ValueError('actual shard native/new accounting differs')
        native+=aggregate['native_Wenv_imports'];new+=aggregate['new_prepare_started']
        for row in shard['rows']:
            i=row['case_index']
            if type(i)is not int or not 0<=i<636 or i in rows:
                raise ValueError('overlap or invalid shard index')
            rows[i]=row
    if set(rows)!=set(range(636)) or native!=594 or new!=42:
        raise ValueError('shard union must exactly cover636 once')
    return manifest,admission,rows


def _case_input_sha(case):
    value={'positions':case['lat_lon_deg'].tolist(),'demands':case['request_bps_int64'].astype('float64').tolist(),
           'aircraft_ids':case['aircraft_ids'],'profile':'europe_v1'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _native_prepared(row, case, stage_output):
    admission=read_json_ref(row['native_wenv_admission_ref'])
    if (admission.get('schema')!='W26NativeWenvAdmissionV1' or admission.get('root_approved')is not True
            or admission.get('source_SHA')!=CORE_SHA or admission.get('case_content_SHA')!=case['content_sha256']
            or admission.get('prepared_file_SHA')!=row['native_prepared_ref']['sha256']
            or admission.get('original_guard_file_SHA')!=row['native_guard_ref']['sha256']):
        raise ValueError('native original/root admission differs')
    guard=read_json_ref(row['native_guard_ref'])
    if guard.get('status')!='completed' or guard.get('exit_code')!=0:
        raise ValueError('native original guard not complete')
    deps=row['dependencies']
    for dep in deps.values():
        read_ref(dep['local_ref'])
        if dep['local_ref']['sha256']!=dep['original_ref']['sha256']:
            raise ValueError('original dependency transport differs')
    wrapper_ref=deps['guard_output']['local_ref']
    if guard['output_ref']['sha256']!=wrapper_ref['sha256']:
        raise ValueError('native actual guard output bytes differ')
    wrapper=read_json_ref(wrapper_ref)
    raw=read_ref(row['native_prepared_ref']);outer=json.loads(raw)
    compressed_ref=dict(row['native_prepared_ref']);compressed_ref.pop('compression',None)
    compressed=read_ref(compressed_ref);asset=wrapper['asset_ref']
    if (asset['sha256']!=hashlib.sha256(compressed).hexdigest()
            or asset['canonical_json_sha256']!=hashlib.sha256(raw).hexdigest()
            or asset['compressed_bytes']!=len(compressed) or asset['uncompressed_bytes']!=len(raw)
            or outer.get('stage')!='preparation' or outer.get('source_sha256')!=CORE_SHA):
        raise ValueError('native actual compressed/canonical publication chain differs')
    _same_ref(stage_output['original_prepared_ref'],row['native_prepared_ref'])
    _same_ref(stage_output['original_guard_ref'],row['native_guard_ref'])
    if any(stage_output.get(k)!=0 for k in ('new_W_calls','new_T_calls','new_D5_calls')):
        raise ValueError('native import cannot contain new scientific calls')
    request=read_json_ref(deps['request_ref']['local_ref'])
    if request['identity']!=outer['identity'] or request['stage']!='preparation' or request['source_sha256']!=CORE_SHA:
        raise ValueError('original native prepare request differs')
    return outer['result'],row['native_prepared_ref']['sha256']


def audit_one_candidate(row, indexed, *, validate_geometry=True, check_started=lambda:None):
    """One independent layout audit. Default performs real shared ray coverage."""
    case,authority=load_case(row,allow_confirmation=True,geometry_check=False)
    guard=read_json_ref(indexed['guard_ref'])
    if guard.get('status')!='completed' or guard.get('exit_code')!=0:
        raise ValueError('new candidate execution guard not complete')
    _same_ref(guard['output_ref'],indexed['stage_output_ref'])
    stage=read_json_ref(indexed['stage_output_ref']);accepted=read_json_ref(indexed['accepted_candidate_ref'])
    if stage!=accepted or stage['case_content_SHA']!=case['content_sha256']:
        raise ValueError('accepted child output/case binding differs')
    _same_ref(stage['candidate_ref'],indexed['candidate_ref'])
    request=read_json_ref(indexed['request_ref'])
    if (request['row']!=row or request['source_SHA']!=CORE_SHA or request['identity']!=row['prepare_identity']
            or request['phase_manifest']!=row['prepare_phase_manifest']):
        raise ValueError('actual candidate request/manifest differs')
    native=bool(row.get('native_wenv_admission_ref'))
    if native:
        if stage['status']!='native_candidate_only':raise ValueError('native method changed')
        prepared,prepared_sha=_native_prepared(row,case,stage)
    else:
        if stage['status']!='candidate_only':raise ValueError('new prepare method changed')
        prepared=read_json_ref(stage['prepared_ref']);prepared_sha=stage['prepared_ref']['sha256']
        if prepared['identity']!=row['prepare_identity']:
            raise ValueError('new prepared identity differs')
    identity=prepared['identity']
    if (prepared.get('schema')!='PreparedCaseV1' or prepared.get('status')!='available'
            or prepared.get('source_sha256')!=CORE_SHA or identity['case_id']!=case['case_id']
            or identity['role']!=case['actual_role'] or identity['shared_case_content_sha256']!=case['content_sha256']
            or identity['source_asset_SHA']!=case['source_asset_SHA']
            or prepared['input_sha256']!=_case_input_sha(case) or identity['input_sha256']!=prepared['input_sha256']):
        raise ValueError('actual complete prepared input/source identity differs')
    ledger=prepared['ledger']
    if ledger['seconds_nominal']!=10 or ledger['raypairs_max']!=1024*len(case['aircraft_ids']):
        raise ValueError('frozen primitive budget differs')
    if ledger['raypairs']>ledger['raypairs_max'] or ledger['audit'].get('valid')is not True:
        raise ValueError('invalid original primitive budget/audit')
    for key,layout_key in [('Wenv_sha256','Wenv'),('raw_W_sha256','raw_w')]:
        raw=json.dumps(prepared[layout_key],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest()!=ledger[key]:
            raise ValueError('actual original Wenv/witness differs from recorded ledger')
    layout=read_wire(read_ref(indexed['candidate_ref']))
    wenv=prepared['Wenv'];owners=[layout['beam_ids'].index(p[1])for p in layout['owner_aircraft_to_beam_id']]
    if (layout['beam_ids']!=wenv['beam_ids'] or layout['anchor_aircraft_ids']!=prepared['anchors']
            or layout['Ncap']!=prepared['Ncap'] or layout['geometry_Omega_cap_sr']!=prepared['Omega_cap']
            or not np.array_equal(layout['centres_lat_lon_deg'],np.asarray(wenv['centers'],dtype='<f8'))
            or not np.array_equal(layout['full_width_deg'],np.asarray(wenv['widths_deg'],dtype='<f8'))
            or owners!=wenv['assignment'] or layout['source_and_cost'].get('source_SHA')!=CORE_SHA):
        raise ValueError('published candidate is not exact original Wenv')
    source_key='original_prepared_SHA'if native else'prepared_SHA'
    if layout['source_and_cost'].get(source_key)!=prepared_sha:
        raise ValueError('candidate prepare provenance SHA differs')
    if layout['geometry_audit']['audit_provenance_SHA']!=stage['audit_ref']['sha256']:
        raise ValueError('candidate original audit ref differs')
    original_audit=read_json_ref(stage['audit_ref'])
    if original_audit.get('parent_case_SHA')!=case['content_sha256']:
        raise ValueError('original audit is not bound to this case')
    if native:
        if (original_audit.get('original_prepared_SHA')!=prepared_sha
                or original_audit.get('original_guard_SHA')!=row['native_guard_ref']['sha256']
                or original_audit.get('native_admission_SHA')!=row['native_wenv_admission_ref']['sha256']):
            raise ValueError('native source-audit provenance differs')
    else:
        _same_ref(original_audit['prepared_ref'],stage['prepared_ref'])
    check_started()
    validate_layout(case,layout,authority,published=False,geometry_check=validate_geometry)
    return case,layout


def audit_prepared_dataset(config,run_dir,progress=lambda value:None,stop=lambda:False):
    began=time.monotonic();run_dir=Path(run_dir)
    manifest,admission,indexed=validate_audit_inputs(config)
    counts={role:0 for role in COUNTS};rows={role:[]for role in COUNTS};failures=[];layout_checks=0
    def check_started():
        nonlocal layout_checks
        layout_checks+=1
    registry={'schema':'R7PublishedParentRegistryV1','case_sha_by_id':{},'layout_shas':[],'allocation_shas':[]}
    visible_stats={role:{'N':[],'widths':[],'requests':0}for role in ('train','development')}
    for i,row in enumerate(manifest['rows']):
        if stop():raise InterruptedError('independent audit stopped; no partial publication')
        try:
            case,layout=audit_one_candidate(row,indexed[i],check_started=check_started)
            role=row['role'];counts[role]+=1
            registry['case_sha_by_id'][case['case_id']]=case['content_sha256'];registry['layout_shas'].append(layout['content_sha256'])
            published={'role':role,'case_id':case['case_id'],'UTC':case['UTC'],'case_content_SHA':case['content_sha256'],
                       'case_ref':row['case_ref'],'case_authority_ref':row['case_authority_ref'],
                       'case_authority_content_SHA':row['case_authority_content_SHA'],
                       'layout_ref':indexed[i]['candidate_ref'],'layout_content_SHA':layout['content_sha256'],
                       'publication_authority_ref':None,'publication_authority_content_SHA':None}
            rows[role].append(published)
            if role!='test':
                visible_stats[role]['N'].append(layout['N']);visible_stats[role]['widths'].extend(layout['full_width_deg'].tolist())
                visible_stats[role]['requests']+=int(layout['request_bps_by_beam_int64'].sum())
        except Exception as error:
            # Every failure, including identity or N/width fields, stays sealed.
            failures.append({'case_index':i,'role':row['role'],'case_id':row['case_id'],'error':repr(error)})
        progress({'phase':'independent_layout_audit','checked':i+1,'total':636})
    sealed=run_dir/'sealed_confirmation'
    failures_ref=_json_once(sealed/'audit_failures.json',{'failures':failures})
    passed=not failures and counts==COUNTS
    registry_ref=None;partition_refs={}
    if passed:
        registry_ref=_json_once(sealed/'authority_candidate.json',registry)
        registry_sha=content_sha(registry)
        for role in COUNTS:
            for row in rows[role]:
                row['publication_authority_ref']=registry_ref;row['publication_authority_content_SHA']=registry_sha
            destination=sealed/'test_candidate.json'if role=='test'else run_dir/(role+'_candidate.json')
            partition_refs[role]=_json_once(destination,{'schema':'W26PublishedDatasetV1','root_approved':False,
                'role':role,'method':'Wenv','source_SHA':CORE_SHA,'rows':rows[role]})
    stats={}
    for role,value in visible_stats.items():
        widths=np.asarray(value['widths'],dtype=np.float32)
        stats[role]={'N_min':min(value['N'])if value['N']else None,'N_max':max(value['N'])if value['N']else None,
                     'width_FP32_distinct':int(len(np.unique(widths))),'exact_total_request_bps_int64':value['requests']}
    # The public receipt contains references/hashes, not confirmation arrays or distribution.
    result={'schema':'W26Independent636PublicationCandidateV1','passed':passed,'qualified_counts':counts,
        'root_approved':False,'method':'Wenv','source_SHA':CORE_SHA,'execution_source_commit':config['source_commit'],
        'prepare_manifest_SHA':config['prepare_manifest_ref']['sha256'],'audit_admission_ref':config['audit_admission_ref'],
        'registry_candidate_ref':registry_ref,'partition_candidate_refs':partition_refs,
        'confirmation_sealed':True,'failure_count':len(failures),'sealed_failures_ref':failures_ref,
        'train_development_descriptive':stats,'candidate_rows_read':636,'independent_layout_checks':layout_checks,'native_imports':594,'new_preparation_calls':0,
        'prior_preparation_actual_new_calls':42,'T_calls':0,'D5_calls':0,'MILP_calls':0,'model_calls':0,
        'audit_wall_seconds':time.monotonic()-began}
    _json_once(run_dir/'audit_candidate_summary.json',result)
    return result
