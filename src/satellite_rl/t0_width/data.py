"""Root-authorized Wenv data, with preparation separated from publication.

No teacher, decoder, model or physical-link calculation is called here. A
candidate's self hash is never publication authority. Confirmation artifacts
remain outside the training loader and public progress messages.
"""
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

from satellite_rl.shared_interface.content import content_sha, read_wire, wire_bytes
from satellite_rl.shared_interface.contracts import Authority, validate_case, validate_layout

COUNTS = {'train': 576, 'development': 20, 'test': 40}
MAX_MEMBER_BYTES = 64 * 1024 * 1024


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_ref(ref, *, limit=MAX_MEMBER_BYTES):
    """Read a pinned file or one pinned archive member, never extract a tree."""
    if 'archive_path' in ref:
        name = ref['member']
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts:
            raise ValueError('unsafe archive member')
        if file_sha(ref['archive_path']) != ref['archive_sha256']:
            raise ValueError('archive SHA differs')
        with tarfile.open(ref['archive_path'], 'r:*') as archive:
            matches = [m for m in archive if m.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > limit:
                raise ValueError('missing, duplicate or oversized regular member')
            with archive.extractfile(matches[0]) as stream:
                raw = stream.read(limit + 1)
    else:
        path = Path(ref['path'])
        if path.stat().st_size > limit:
            raise ValueError('oversized input')
        raw = path.read_bytes()
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('pinned input bytes differ')
    if ref.get('compression') == 'gzip':
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('oversized uncompressed input')
    return raw


def read_json_ref(ref):
    return json.loads(read_ref(ref))


def load_authority(ref, expected_content_sha):
    return Authority(read_json_ref(ref), expected_content_sha)


def load_case(row, *, allow_confirmation=False, geometry_check=False):
    if row['role'] == 'test' and not allow_confirmation:
        raise ValueError('confirmation data is sealed')
    case = read_wire(read_ref(row['case_ref']))
    authority = load_authority(row['case_authority_ref'], row['case_authority_content_SHA'])
    validate_case(case, authority, geometry_check=geometry_check)
    if (case['actual_role'] != row['role'] or case['case_id'] != row['case_id']
            or case['content_sha256'] != row['case_content_SHA']):
        raise ValueError('root row/case binding differs')
    return case, authority


def prepare_binding(row, dataset_sha, identity_sha, *, allow_confirmation=False):
    """Pure byte/identity conversion for a subsequently root-signed prepare plan.

    Reproduces Scene's input hash without instantiating Scene or tracing rays.
    The caller still signs/freezes this row and phase before any prepare call.
    """
    case,_=load_case(row,allow_confirmation=allow_confirmation,geometry_check=False)
    identity_tree={'positions':case['lat_lon_deg'].tolist(),
                   'demands':case['request_bps_int64'].astype('float64').tolist(),
                   'aircraft_ids':case['aircraft_ids'],'profile':'europe_v1'}
    input_sha=hashlib.sha256(json.dumps(identity_tree,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    identity={'case_id':case['case_id'],'mother_id':case['original_mother_id'],'profile':'uniform21600000',
              'role':case['actual_role'],'input_sha256':input_sha,'identity_sha256':identity_sha,
              'shared_case_content_sha256':case['content_sha256'],'source_asset_SHA':case['source_asset_SHA']}
    stage={'train':'labels','development':'development','test':'test'}[case['actual_role']]
    phase={'stage':stage,'role':case['actual_role'],'admitted':True,'identity_sha256':identity_sha,
           'dataset_sha256':dataset_sha,'cases':{identity['case_id']:identity}}
    return {'prepare_identity':identity,'prepare_phase_manifest':phase}


def load_published_layout(row, *, role, confirmation_unlock=None, geometry_check=True):
    """Training cannot open test, even when its objects are root authorized."""
    if row['role'] != role:
        raise ValueError('dataset role differs')
    if role == 'test':
        if not isinstance(confirmation_unlock, dict):
            raise ValueError('terminal-lock confirmation admission required')
        manifest_ref=confirmation_unlock.get('manifest_ref')
        if not isinstance(manifest_ref,dict) or manifest_ref.get('sha256')!=confirmation_unlock.get('manifest_SHA'):
            raise ValueError('unlock binds exact confirmation manifest file SHA')
        confirmation_manifest=read_json_ref(manifest_ref)
        if (confirmation_manifest.get('schema')!='W26PublishedDatasetV1'
                or confirmation_manifest.get('role')!='test'
                or confirmation_manifest.get('root_approved')is not True
                or len(confirmation_manifest.get('rows',[]))!=40 or row not in confirmation_manifest['rows']):
            raise ValueError('unlock cannot be moved to another confirmation partition/row')
        unlock = read_json_ref(confirmation_unlock['receipt_ref'])
        if (unlock.get('schema') != 'W26ConfirmationUnlockV1'
                or unlock.get('root_approved') is not True
                or unlock.get('manifest_SHA') != confirmation_unlock.get('manifest_SHA')
                or set(unlock.get('terminal_model_SHAs', {})) != {'S00','S10','S01','S11'}):
            raise ValueError('all four terminal weights must be root locked')
    case, _ = load_case(row, allow_confirmation=role == 'test', geometry_check=False)
    layout = read_wire(read_ref(row['layout_ref']))
    authority = load_authority(row['publication_authority_ref'], row['publication_authority_content_SHA'])
    validate_layout(case, layout, authority, geometry_check=geometry_check)
    if layout['content_sha256'] != row['layout_content_SHA']:
        raise ValueError('published layout identity differs')
    return case, layout, authority


def load_dataset(manifest_ref, role, *, confirmation_unlock=None, geometry_check=True):
    """Load only one root-pinned published partition, preserving mother order."""
    manifest = read_json_ref(manifest_ref)
    if manifest.get('schema') != 'W26PublishedDatasetV1' or manifest.get('root_approved') is not True:
        raise ValueError('independent root dataset publication required')
    if manifest.get('role') != role or role not in COUNTS or len(manifest['rows']) != COUNTS[role]:
        raise ValueError('complete expected partition required')
    if len({r['case_id'] for r in manifest['rows']}) != COUNTS[role]:
        raise ValueError('duplicate published mothers')
    if role=='test' and (not isinstance(confirmation_unlock,dict) or confirmation_unlock.get('manifest_SHA')!=manifest_ref['sha256']):
        raise ValueError('confirmation unlock differs from loaded manifest')
    return [load_published_layout(row, role=role, confirmation_unlock=confirmation_unlock,
                                  geometry_check=geometry_check) for row in manifest['rows']]


def validate_training_admission(config):
    """Bind train/dev to the independent complete636 gate without opening test."""
    gate = read_json_ref(config['data_admission_ref'])
    if (gate.get('schema') != 'W26Complete636DataAdmissionV1' or gate.get('root_approved') is not True
            or gate.get('qualified_counts') != COUNTS or gate.get('method') != 'Wenv'
            or gate.get('source_SHA') != config['source_sha']
            or gate.get('confirmation_sealed') is not True):
        raise ValueError('root complete636 geometry admission required before training')
    for role,key in [('train','train_cases'),('development','dev_cases')]:
        ref=config[role+'_manifest_ref']
        if gate['partition_manifest_SHAs'].get(role)!=ref['sha256']:
            raise ValueError('training partition differs from complete636 gate')
        manifest=read_json_ref(ref)
        if (manifest.get('schema')!='W26PublishedDatasetV1' or manifest.get('root_approved') is not True
                or manifest.get('role')!=role or len(manifest['rows'])!=COUNTS[role]
                or manifest['rows']!=config[key]):
            raise ValueError('exact complete ordered training partition required')
    return {'passed':True,'train_count':576,'development_count':20,'confirmation_qualified':True}


def materialize_selected_members(archive_ref, members, destination, *, max_total_bytes=512*1024*1024):
    """One streaming pass, only explicitly SHA-pinned canonical inputs retained.

    members maps original member names to SHA256, with no array interpretation.
    Source archives remain canonical; output is a registered active transport
    cache, whose returned SHA-to-path map can be discarded after input use.
    """
    if file_sha(archive_ref['path']) != archive_ref['sha256']:
        raise ValueError('canonical archive differs')
    if not members or len(set(members.values())) != len(members):
        raise ValueError('explicit unique pinned members required')
    target=Path(destination);target.mkdir(parents=True,exist_ok=True)
    found=set();total=0;transport={}
    with tarfile.open(archive_ref['path'],'r|*') as archive:
        for member in archive:
            if member.name not in members:
                continue
            if member.name in found or not member.isfile() or member.size>MAX_MEMBER_BYTES:
                raise ValueError('duplicate/nonregular/oversized canonical member')
            if total+member.size>max_total_bytes:
                raise ValueError('active transport cache exceeds frozen storage budget')
            with archive.extractfile(member) as stream:
                raw=stream.read(MAX_MEMBER_BYTES+1)
            sha=hashlib.sha256(raw).hexdigest()
            if sha!=members[member.name]:
                raise ValueError('canonical member SHA differs')
            path=target/sha
            if path.exists():
                if file_sha(path)!=sha:
                    raise ValueError('existing content-addressed input differs; refuse overwrite')
                ref={'path':str(path.resolve()),'sha256':sha}
            else:
                ref=_write_once(path,raw)
            transport[sha]=ref['path'];found.add(member.name);total+=len(raw)
    if found!=set(members):
        raise ValueError('canonical inputs missing; no replacement')
    return {'transport_by_SHA':transport,'retained_members':len(found),'retained_bytes':total,
            'canonical_archive_SHA':archive_ref['sha256']}


stage_archive_members = materialize_selected_members


def r7_case_transport_index(root_acceptance_ref):
    """Read only accepted596 dataset metadata, not case arrays or heldout data."""
    acceptance=read_json_ref(root_acceptance_ref)
    if (acceptance.get('schema')!='R7RootUniformDataAcceptanceV1' or acceptance.get('passed')is not True
            or acceptance.get('cases')!=596 or acceptance.get('fixed_request_bps')!=21600000):
        raise ValueError('original root complete596 case acceptance required')
    archive=acceptance['original_archive_ref']
    def archived(ref):
        return {'archive_path':archive['path'],'archive_sha256':archive['sha256'],
                'member':ref['path'].lstrip('/'),'sha256':ref['sha256']}
    authority_ref=archived(acceptance['authority_ref'])
    registry=read_json_ref(authority_ref)
    authority_content_sha=content_sha(registry)
    # Computing registry content identity does not authorize any new cases.
    Authority(registry,authority_content_sha)
    members={authority_ref['member']:authority_ref['sha256']};rows=[]
    for role in ('train','development'):
        ref=archived(acceptance['datasets'][role]);dataset=read_json_ref(ref)
        if dataset['count']!=COUNTS[role] or dataset['role']!=role:
            raise ValueError('original596 partition differs')
        for i,item in enumerate(dataset['cases']):
            if item['case_index']!=i or item['role']!=role:
                raise ValueError('original case order/role differs')
            cref=archived(item['case_ref']);members[cref['member']]=cref['sha256']
            if registry['case_sha_by_id'].get(item['case_id'])!=item['case_content_sha256']:
                raise ValueError('original case authority differs')
            rows.append({'role':role,'case_id':item['case_id'],'case_content_SHA':item['case_content_sha256'],
                'UTC':item['original_mother_id'].split(':',1)[1],'case_ref':cref,
                'case_authority_ref':authority_ref,'case_authority_content_SHA':authority_content_sha})
    return {'archive_ref':archive,'members':members,'rows':rows,'root_acceptance_ref':root_acceptance_ref,
            'heldout_reads':0,'layout_publication_ready':False}


def materialize_r7_cases(root_acceptance_ref, destination):
    """Retain exactly596 active case objects plus original authority, no raw mirror."""
    index=r7_case_transport_index(root_acceptance_ref)
    receipt=stage_archive_members(index['archive_ref'],index['members'],destination)
    mapping=receipt['transport_by_SHA'];rows=[]
    for row in index['rows']:
        row=dict(row)
        for key in ('case_ref','case_authority_ref'):
            sha=row[key]['sha256'];row[key]={'path':mapping[sha],'sha256':sha}
        rows.append(row)
    return {'schema':'W26Accepted596ActiveCaseTransportV1','rows':rows,'transport_receipt':receipt,
            'root_acceptance_ref':root_acceptance_ref,'layout_publication_ready':False,'heldout_reads':0}


def validate_manifest(manifest):
    if manifest.get('schema') != 'W26WenvPreparationManifestV1' or manifest.get('method') != 'Wenv':
        raise ValueError('root Wenv manifest required')
    rows = manifest['rows']
    counts = {role: sum(row['role'] == role for row in rows) for role in COUNTS}
    if len(rows) != 636 or counts != COUNTS:
        raise ValueError('complete 576/20/40 mothers required; no teacher filtering')
    if len({r['case_id'] for r in rows}) != len(rows) or len({r['UTC'] for r in rows}) != len(rows):
        raise ValueError('duplicate or cross-role mother UTC')
    for row in rows:
        if row['role'] not in COUNTS:
            raise ValueError('unexpected role')
        if row['role'] == 'test' and not row.get('confirmation_source_admission_ref'):
            raise ValueError('new official confirmation sources need root cross-use admission')
    if manifest.get('nominal_seconds') != 10 or manifest.get('hard_seconds') != 20 or manifest.get('raypairs_per_aircraft') != 1024:
        raise ValueError('frozen Wenv preparation budget differs')
    return manifest


def validate_source_files(manifest):
    import importlib
    required = {'satellite_rl.coverage.ai_search_r5.core', 'satellite_rl.coverage.ai_search.core',
                'satellite_rl.shared_interface.contracts', 'satellite_rl.shared_interface.geometry',
                'satellite_rl.t0_width.guard', __name__}
    refs = {str(Path(ref['path']).resolve()):ref['sha256'] for ref in manifest['source_file_refs']}
    for name in required:
        path = str(Path(importlib.import_module(name).__file__).resolve())
        if path not in refs:
            raise ValueError('actual loaded provider/core/audit source absent from closure')
    for ref in manifest['source_file_refs']:
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('actual Wenv/provider/audit source bytes differ')
    if not manifest['source_file_refs']:
        raise ValueError('complete source closure is required')


def _write_once(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def _json_once(path, value):
    return _write_once(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())


def reuse_wenv(row, source_sha):
    """Root receipt must identify Wenv, not a T or AI published terminal."""
    receipt = read_json_ref(row['reuse_admission_ref'])
    if (receipt.get('schema') != 'W26WenvReuseAdmissionV1' or receipt.get('root_approved') is not True
            or receipt.get('method') != 'Wenv' or receipt.get('source_SHA') != source_sha
            or receipt.get('case_content_SHA') != row['case_content_SHA']
            or receipt.get('layout_content_SHA') != row['layout_content_SHA']):
        raise ValueError('same-source Wenv reuse not root admitted')
    return load_published_layout(row, role=row['role'], geometry_check=True)


def prepare_candidate(request, directory):
    """Child-only bounded primitive. Caller supplies a root-pinned phase."""
    from satellite_rl.coverage.ai_search_r5.core import prepare_case
    from satellite_rl.shared_interface.stages import algorithm_scene, make_layout
    from satellite_rl.shared_interface.geometry import cone_omega
    directory = Path(directory)
    row = request['row']
    case, authority = load_case(row, allow_confirmation=True, geometry_check=True)
    scene = algorithm_scene(case, authority)
    identity = request['identity']
    if identity['case_id'] != case['case_id'] or identity['role'] != case['actual_role']:
        raise ValueError('prepare role/case differs')
    prepared = prepare_case(scene, identity, request['source_SHA'], phase_manifest=request['phase_manifest'], time_limit=10.)
    prepared_ref = _json_once(directory / 'prepared.json', prepared.to_dict())
    if prepared.status != 'available' or prepared.Wenv is None:
        raise ValueError('Wenv unavailable; no retry or scene replacement')
    wenv = prepared.Wenv
    audit_ref = _json_once(directory / 'primitive_audit.json', {'parent_case_SHA': case['content_sha256'],
        'method': 'Wenv', 'audit': prepared.ledger['audit'], 'prepared_ref': prepared_ref})
    layout = make_layout(case, beam_ids=wenv.beam_ids, source_slot_ids=wenv.beam_ids,
        anchor_aircraft_ids=prepared.anchors, centres_lat_lon_deg=wenv.centers,
        full_width_deg=wenv.widths,
        owner_aircraft_to_beam_id=[[aid, wenv.beam_ids[int(owner)]] for aid,owner in zip(case['aircraft_ids'],wenv.assignment)],
        Ncap=prepared.Ncap, geometry_Omega_cap_sr=prepared.Omega_cap, geometry_Omega_sr=cone_omega(wenv.widths),
        geometry_audit={'independently_valid':True,'audit_provenance_SHA':audit_ref['sha256']},
        source_and_cost={'source_SHA':request['source_SHA'],'prepared_SHA':prepared_ref['sha256'],'audit_SHA':audit_ref['sha256']})
    # No layout authority exists yet. This checks a candidate, not publication.
    validate_layout(case, layout, authority, published=False)
    candidate_ref = _write_once(directory / 'candidate.json', wire_bytes(layout))
    _json_once(directory / 'stage_output.json', {'status':'candidate_only', 'candidate_ref':candidate_ref,
        'prepared_ref':prepared_ref,'audit_ref':audit_ref,'case_content_SHA':case['content_sha256']})


def prepare_native_candidate(request, directory):
    """Audit/publish-candidate from original prepare.Wenv; new W/T/D5 calls=0.

    Root pins original complete guard, original compressed prepared output and
    source/core lineage first. The teacher/qualified label is never consumed.
    Independent root publication still follows this transport/audit operation.
    """
    from satellite_rl.coverage.ai_search_r5.core import PreparedCase
    from satellite_rl.shared_interface.stages import algorithm_scene,make_layout
    from satellite_rl.shared_interface.geometry import cone_omega
    row=request['row'];directory=Path(directory)
    admission=read_json_ref(row['native_wenv_admission_ref'])
    if (admission.get('schema')!='W26NativeWenvAdmissionV1' or admission.get('root_approved')is not True
            or admission.get('source_SHA')!=request['source_SHA']
            or admission.get('case_content_SHA')!=row['case_content_SHA']
            or admission.get('prepared_file_SHA')!=row['native_prepared_ref']['sha256']
            or admission.get('original_guard_file_SHA')!=row['native_guard_ref']['sha256']):
        raise ValueError('original complete same-source Wenv native import not root admitted')
    guard=read_json_ref(row['native_guard_ref'])
    if guard.get('status')!='completed' or guard.get('exit_code')!=0:
        raise ValueError('native original preparation guard incomplete/failed')
    outer=read_json_ref(row['native_prepared_ref'])
    original=outer['result']
    if outer.get('stage')!='preparation' or outer.get('source_sha256')!=request['source_SHA']:
        raise ValueError('native must be original preparation, not teacher or AI terminal')
    case,authority=load_case(row,allow_confirmation=False,geometry_check=True)
    scene=algorithm_scene(case,authority)
    prepared=PreparedCase.from_dict(scene,original)
    if (prepared.status!='available' or prepared.Wenv is None or prepared.source_sha256!=request['source_SHA']
            or prepared.identity.get('shared_case_content_sha256')!=case['content_sha256']
            or prepared.identity['role']!=case['actual_role'] or prepared.identity['case_id']!=case['case_id']
            or prepared.ledger.get('audit',{}).get('valid')is not True):
        raise ValueError('native Wenv identity or original audit differs')
    wenv=prepared.Wenv
    audit={'method':'Wenv','parent_case_SHA':case['content_sha256'],'original_prepared_SHA':row['native_prepared_ref']['sha256'],
           'original_guard_SHA':row['native_guard_ref']['sha256'],'native_admission_SHA':row['native_wenv_admission_ref']['sha256']}
    audit_ref=_json_once(directory/'native_source_audit.json',audit)
    layout=make_layout(case,beam_ids=wenv.beam_ids,source_slot_ids=wenv.beam_ids,anchor_aircraft_ids=prepared.anchors,
        centres_lat_lon_deg=wenv.centers,full_width_deg=wenv.widths,
        owner_aircraft_to_beam_id=[[aid,wenv.beam_ids[int(owner)]] for aid,owner in zip(case['aircraft_ids'],wenv.assignment)],
        Ncap=prepared.Ncap,geometry_Omega_cap_sr=prepared.Omega_cap,geometry_Omega_sr=cone_omega(wenv.widths),
        geometry_audit={'independently_valid':True,'audit_provenance_SHA':audit_ref['sha256']},
        source_and_cost={'source_SHA':request['source_SHA'],'original_prepared_SHA':row['native_prepared_ref']['sha256'],
                         'original_guard_SHA':row['native_guard_ref']['sha256'],'audit_SHA':audit_ref['sha256']})
    validate_layout(case,layout,authority,published=False)
    candidate_ref=_write_once(directory/'candidate.json',wire_bytes(layout))
    _json_once(directory/'stage_output.json',{'status':'native_candidate_only','candidate_ref':candidate_ref,
        'case_content_SHA':case['content_sha256'],'new_W_calls':0,'new_T_calls':0,'new_D5_calls':0,
        'original_prepared_ref':row['native_prepared_ref'],'original_guard_ref':row['native_guard_ref'],'audit_ref':audit_ref})


def prepare_dataset(config, run_dir, progress=lambda value: None, stop=lambda: False):
    """636-row guarded preparation; confirmation details stay in sealed output.

    Config includes externally root-pinned manifest_ref and execution_admission_ref.
    Returns only aggregate readiness. Publication requires a separate root registry.
    Existing starts are not replayed automatically after interruption.
    """
    from satellite_rl.t0_width.guard import run_stage
    manifest = validate_manifest(read_json_ref(config['manifest_ref']))
    admission = read_json_ref(config['execution_admission_ref'])
    if (admission.get('schema') != 'W26DataExecutionAdmissionV1' or admission.get('root_approved') is not True
            or admission.get('manifest_SHA') != config['manifest_ref']['sha256']
            or admission.get('remote_port') not in (20418,20424)):
        raise ValueError('fresh remote 18/24 data execution admission required')
    validate_source_files(manifest)
    run_dir = Path(run_dir)
    indices=config.get('case_indices',list(range(636)))
    if (not isinstance(indices,list) or not indices or any(type(i)is not int or not 0<=i<636 for i in indices)
            or len(set(indices))!=len(indices) or indices!=sorted(indices)):
        raise ValueError('explicit unique ascending shard indices required')
    if admission.get('case_indices')!=indices:
        raise ValueError('exact shard differs from root execution admission')
    if admission.get('total_new_prepare_budget')!=636:
        raise ValueError('all-shard cumulative new prepare budget must remain636')
    completed = reused = started = native_imports = 0
    for i in indices:
        row=manifest['rows'][i]
        if stop():
            raise RuntimeError('data preparation stopped; preserve unknown spend')
        target = run_dir / ('sealed_confirmation' if row['role']=='test' else row['role']) / str(i)
        target.mkdir(parents=True, exist_ok=True)
        if (target / 'accepted_candidate.json').exists():
            raise ValueError('existing per-row result requires root same-ID recovery; no automatic replay')
        if row.get('reuse_admission_ref'):
            if row['role']=='test':
                raise ValueError('new confirmation cannot reuse already used layouts')
            reuse_wenv(row,manifest['source_SHA'])
            result={'status':'reused_published','case_content_SHA':row['case_content_SHA'],'layout_ref':row['layout_ref']}
            reused += 1
        else:
            if row['role']=='test':
                source = read_json_ref(row['confirmation_source_admission_ref'])
                if (source.get('root_approved') is not True or source.get('case_content_SHA') != row['case_content_SHA']
                        or source.get('not_used_for_method_or_model_selection') is not True):
                    raise ValueError('confirmation official source/use admission differs')
            request={'row':row,'source_SHA':manifest['source_SHA'],'identity':row['prepare_identity'],
                     'phase_manifest':row['prepare_phase_manifest']}
            request_ref=_json_once(target/'request.json',request)
            native=bool(row.get('native_wenv_admission_ref'))
            if native:native_imports+=1
            else:started+=1
            command=[sys.executable,'-B','-m',__name__,'--request',request_ref['path'],'--request-sha',request_ref['sha256'],
                     '--directory',str(target)]
            if native:command.append('--native')
            guard=run_stage(command,target,20,4*1024**3,stop)
            if guard.get('status')!='completed':
                raise RuntimeError('prepare guard failed/unknown; preserve result, no retry')
            result=read_json_ref(guard['output_ref'])
            # Outer guard success is not independent root publication acceptance.
        _json_once(target/'accepted_candidate.json',result)
        completed+=1
        progress({'completed':completed,'total':len(indices),'status':'preparing'})
    summary={'schema':'W26DataPreparationAggregateV1','completed':completed,'total':len(indices),
             'manifest_SHA':config['manifest_ref']['sha256'],'case_indices':indices,'all_dataset_total':636,
             'new_prepare_started':started,'reused_published':reused,'candidate_ready':completed==len(indices),
             'native_Wenv_imports':native_imports,
             'root_publication_admitted':False,'test_details_sealed':True,'T_calls':0,'D5_calls':0,'MILP_calls':0}
    _json_once(run_dir/'aggregate.json',summary)
    return summary


def prepare_confirmation_sources(config, run_dir, progress=lambda value: None, stop=lambda: False):
    """Stream two root-admitted official daily archives into forty sealed cases.

    No archive extraction, interpolation, activity demand, old coverage filter,
    layout planning, fallback time selection or policy access. A root-supplied
    source admission pins URLs, complete asset SHA, twenty UTCs/day and source
    cross-use checks before this function reads any trace. Result is candidate
    case wire only; root must authorize it separately before Wenv preparation.
    """
    from bisect import bisect_right
    from datetime import datetime, timezone
    from collections import Counter
    import math
    import io
    from satellite_rl.workbench.s2p19_data import _trace_position
    from satellite_rl.envs.regions import get_profile
    from satellite_rl.shared_interface.stages import make_case
    admission = read_json_ref(config['source_admission_ref'])
    if (admission.get('schema') != 'W26OfficialConfirmationSourceAdmissionV1'
            or admission.get('root_approved') is not True
            or admission.get('not_used_for_method_or_model_selection') is not True
            or admission.get('remote_port') not in (20418,20424)):
        raise ValueError('root official source/use/remote admission required')
    assets = admission['assets']
    if len(assets) != 2 or len({a['date'] for a in assets}) != 2:
        raise ValueError('two distinct official days required')
    profile = get_profile('europe_v1')
    directory = Path(run_dir) / 'sealed_confirmation_sources'
    output_rows = []
    for asset in assets:
        if not asset['source_url'].startswith('https://github.com/adsblol/globe_history_2024/releases/download/'):
            raise ValueError('official ADSB.lol release URL required')
        path = Path(asset['path'])
        if path.stat().st_size != asset['release_asset_bytes'] or file_sha(path) != asset['release_asset_SHA']:
            raise ValueError('complete official release asset bytes/SHA differ')
        utcs = asset['selected_UTCs']
        if len(utcs) != 20 or len(set(utcs)) != 20:
            raise ValueError('twenty preselected distinct UTCs per date required')
        dates = [datetime.fromisoformat(t.replace('Z','+00:00')) for t in utcs]
        if any(t.date().isoformat()!=asset['date'] or t.utcoffset().total_seconds()!=0 for t in dates):
            raise ValueError('explicit same-day UTC sources required')
        stamps = [t.timestamp() for t in dates]
        planes = [[] for _ in stamps]; tallies = [Counter() for _ in stamps]
        seen = set(); trace_files = 0
        with tarfile.open(path, 'r|*') as archive:
            for member in archive:
                if stop():
                    raise InterruptedError('sealed source preparation stopped')
                if not member.isfile() or not member.name.lstrip('./').startswith('traces/') or not member.name.endswith('.json'):
                    continue
                if member.size > MAX_MEMBER_BYTES:
                    raise ValueError('oversized official trace member')
                with archive.extractfile(member) as stream:
                    raw = stream.read(MAX_MEMBER_BYTES+1)
                if raw[:2] == b'\x1f\x8b':
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                        raw = stream.read(MAX_MEMBER_BYTES+1)
                if len(raw)>MAX_MEMBER_BYTES:
                    raise ValueError('oversized uncompressed official trace')
                data = json.loads(raw)
                aid = data['icao']
                if not isinstance(aid,str) or not aid.strip():
                    raise ValueError('missing stable ICAO')
                aid = aid.strip().lower()
                if aid in seen:
                    raise ValueError('duplicate official daily ICAO; no dedup/delete')
                seen.add(aid)
                base = float(data['timestamp']);trace=data['trace']
                offsets = [float(r[0]) for r in trace]
                if not math.isfinite(base) or not all(math.isfinite(t) for t in offsets):
                    raise ValueError('nonfinite official time')
                if offsets != sorted(offsets):
                    ordered = sorted(zip(offsets,trace),key=lambda item:item[0])
                    offsets=[item[0] for item in ordered];trace=[item[1] for item in ordered]
                for i,stamp in enumerate(stamps):
                    index=bisect_right(offsets,stamp-base)-1
                    if index<0:
                        tallies[i]['no_prior_record']+=1;continue
                    age=stamp-(base+offsets[index])
                    if age>60:
                        tallies[i]['expired']+=1;continue
                    reason, position = _trace_position(trace[index],profile)
                    tallies[i][reason]+=1
                    if reason=='selected':
                        lat,lon,altitude=position
                        planes[i].append({'aircraft_id':aid,'lat':lat,'lon':lon,'altitude_m':altitude,
                            'timestamp_utc':datetime.fromtimestamp(base+offsets[index],timezone.utc).isoformat(),
                            'source_date':asset['date'],'source_sha256':asset['release_asset_SHA']})
                trace_files+=1
                if trace_files%5000==0:
                    progress({'phase':'sealed_source_scan','date':asset['date'],'trace_files_processed':trace_files})
        for i,(utc,rows) in enumerate(zip(utcs,planes)):
            if not rows:
                raise ValueError('preselected confirmation mother is empty; no replacement')
            rows.sort(key=lambda r:r['aircraft_id'])
            mother='official_adsblol:'+utc
            target=directory/asset['date']/str(i)
            raw_ref=_write_once(target/'aircraft.jsonl',b''.join((json.dumps(r,sort_keys=True)+'\n').encode() for r in rows))
            metadata={'case_id':'w26:test:'+mother+':uniform21600000','actual_role':'test','source_role':'test',
                'original_mother_id':mother,'UTC':utc,'source_asset_SHA':raw_ref['sha256'],
                'source':{'release_asset_SHA':asset['release_asset_SHA'],'source_url':asset['source_url'],
                    'source_admission_SHA':config['source_admission_ref']['sha256'],
                    'observation_policy':'latest_at_or_before_60s_invalid_blocks_prior_no_interpolation',
                    'position_filter':'airborne_valid_altitude_europe_rectangle_elevation10',
                    'request_policy':'all_selected_aircraft_21600000_no_activity_or_coverage_filter',
                    'counts':dict(tallies[i]),'raw_perplane_SHA':raw_ref['sha256']}}
            case=make_case(metadata,[r['aircraft_id'] for r in rows],[[r['lat'],r['lon']] for r in rows])
            case_ref=_write_once(target/'case.json',wire_bytes(case))
            output_rows.append({'role':'test','case_id':case['case_id'],'UTC':utc,'case_content_SHA':case['content_sha256'],
                                'case_ref':case_ref,'raw_ref':raw_ref})
    _json_once(directory/'sealed_candidate_index.json',{'schema':'W26SealedConfirmationCaseCandidatesV1','rows':output_rows,
        'source_admission_SHA':config['source_admission_ref']['sha256'],'root_publication_admitted':False})
    return {'status':'candidate_sources_ready','case_count':40,'root_publication_admitted':False,'sealed':True}


def download_official_sources(config, run_dir, progress=lambda value: None, stop=lambda: False):
    """Two bounded official downloads into one canonical shared-home directory.

    Resume is permitted only for the exact original task owner and a verified
    HTTP206 Content-Range starting at the actual .part length. No trace is read.
    SHA receipts record downloaded bytes; root source admission is subsequent.
    """
    import urllib.request
    import urllib.parse
    import shutil
    import os
    import time
    admission=read_json_ref(config['download_admission_ref'])
    if (admission.get('schema')!='W26OfficialDownloadAdmissionV1' or admission.get('root_approved') is not True
            or admission.get('remote_port') not in (20418,20424) or len(admission.get('assets',[]))!=2):
        raise ValueError('root exact official download/resource admission required')
    directory=Path(admission['canonical_directory']).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    owner=config['request_id']
    if not isinstance(owner,str) or not owner:
        raise ValueError('exact workbench download task owner required')
    if not 0<admission.get('download_wall_seconds',0)<=14400:
        raise ValueError('finite root download wall budget required')
    began=time.monotonic();results=[]
    for asset in admission['assets']:
        url=asset['source_url'];parts=urllib.parse.urlparse(url)
        if parts.scheme!='https' or parts.hostname not in ('github.com','api.github.com'):
            raise ValueError('TLS verified official GitHub source required')
        if (type(asset['release_asset_id'])is not int or asset['release_asset_id']<=0
                or type(asset['release_asset_bytes'])is not int or not 0<asset['release_asset_bytes']<=3*1024**3):
            raise ValueError('bounded exact release asset identity/size required')
        if parts.hostname=='api.github.com' and parts.path!=f"/repos/adsblol/globe_history_2024/releases/assets/{asset['release_asset_id']}":
            raise ValueError('official API asset ID differs')
        if parts.hostname=='github.com' and not parts.path.startswith('/adsblol/globe_history_2024/releases/download/'):
            raise ValueError('official release repository differs')
        day=asset['date']
        from datetime import date
        if date.fromisoformat(day).isoformat()!=day:
            raise ValueError('canonical ISO date required')
        final=directory/(day+'.tar');partial=directory/(day+'.tar.part');owner_path=directory/(day+'.owner.json')
        identity={'request_id':owner,'source_admission_SHA':config['download_admission_ref']['sha256'],
                  'source_url':url,'release_asset_id':asset['release_asset_id'],'release_asset_bytes':asset['release_asset_bytes']}
        if final.exists() and asset.get('release_asset_SHA'):
            if final.stat().st_size!=asset['release_asset_bytes'] or file_sha(final)!=asset['release_asset_SHA']:
                raise ValueError('root-pinned existing canonical asset differs')
            results.append({**asset,'path':str(final),'reused_canonical':True})
            continue
        if owner_path.exists():
            if json.loads(owner_path.read_bytes())!=identity:
                raise ValueError('canonical asset belongs to another task/admission; no takeover')
        else:
            if partial.exists() or final.exists():
                raise ValueError('unowned canonical bytes need root provenance verification')
            _json_once(owner_path,identity)
        if final.exists():
            if final.stat().st_size!=asset['release_asset_bytes']:
                raise ValueError('existing canonical asset size differs')
            digest=file_sha(final)
            if asset.get('release_asset_SHA') and digest!=asset['release_asset_SHA']:
                raise ValueError('existing canonical asset SHA differs')
            results.append({**asset,'path':str(final),'release_asset_SHA':digest,'reused_canonical':True})
            continue
        if shutil.disk_usage(directory).free<5*1024**3:
            raise ValueError('download requires five GiB free storage gate')
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>asset['release_asset_bytes']:
            raise ValueError('partial larger than official asset')
        if offset<asset['release_asset_bytes']:
            headers={'User-Agent':'satellite-rl-W26-source-provider','Accept':'application/octet-stream'}
            if offset:headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(request,timeout=30) as response:
                status=response.status
                if offset:
                    import re
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if status!=206 or not match or int(match[1])!=offset or int(match[3])!=asset['release_asset_bytes']:
                        raise ValueError('resume requires exact HTTP206 offset/total; never append HTTP200')
                elif status!=200:
                    raise ValueError('fresh complete download requires HTTP200')
                remaining=asset['release_asset_bytes']-offset
                length=response.headers.get('Content-Length')
                if length is not None and int(length)!=remaining:
                    raise ValueError('official response remaining size differs')
                with partial.open('ab' if offset else 'xb') as stream:
                    while offset<asset['release_asset_bytes']:
                        if stop():raise InterruptedError('download stopped; preserve owned .part')
                        if time.monotonic()-began>admission['download_wall_seconds']:
                            raise TimeoutError('download wall budget exhausted; preserve owned .part')
                        chunk=response.read(min(4*1024*1024,asset['release_asset_bytes']-offset))
                        if not chunk:raise EOFError('official download truncated; preserve owned .part')
                        stream.write(chunk);offset+=len(chunk)
                        progress({'phase':'official_download','date':day,'bytes':offset,'expected_bytes':asset['release_asset_bytes']})
                    if response.read(1):raise ValueError('official download exceeds frozen asset bytes')
                    stream.flush();os.fsync(stream.fileno())
        digest=file_sha(partial)
        if asset.get('release_asset_SHA') and digest!=asset['release_asset_SHA']:
            raise ValueError('complete downloaded SHA differs; retain failure bytes')
        if final.exists():raise ValueError('canonical destination unexpectedly appeared')
        partial.rename(final)
        result={**asset,'path':str(final),'release_asset_SHA':digest,'reused_canonical':False}
        _json_once(directory/(day+'.download_receipt.json'),{**result,'owner':identity,
            'size_verified':True,'TLS_verification':True,'root_source_admitted':False})
        results.append(result)
    return {'schema':'W26OfficialDownloadsV1','assets':results,'total_bytes':sum(a['release_asset_bytes'] for a in results),
            'root_source_admitted':False,'trace_reads':0,'canonical_directory':str(directory)}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha',required=True)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    function=prepare_native_candidate if args.native else prepare_candidate
    function(read_json_ref({'path':args.request,'sha256':args.request_sha}),args.directory)
