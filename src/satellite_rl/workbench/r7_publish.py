"""Convert independently audited S1 arrays to the shared variable-layout object.

No source path is part of immutable content. Geometry is already independently
audited in the originating full label/node receipt; this conversion checks the
typed structure and conservation without adding unregistered ray work.
"""
import numpy as np
import os,time
from pathlib import Path
from .coverage_ai_search import pin
from .storage import digest,read_json
from satellite_rl.training.run_monitor import atomic_json

def shared_layout(case,authority,layout,prepared,audit,source_and_cost_shas):
    from satellite_rl.shared_interface.stages import make_layout
    from satellite_rl.shared_interface.contracts import validate_layout
    from scripts.stage1_ai_search_r4.audit import canonical_sha
    if audit.get('valid')is not True:raise ValueError('independent geometry audit required')
    if audit.get('parent_case_SHA')!=case['content_sha256'] or audit.get('layout_sha256')!=canonical_sha(layout):raise ValueError('independent audit does not bind these exact layout arrays and case')
    if any(not isinstance(v,str) or len(v)!=64 for v in source_and_cost_shas.values()):raise ValueError('SHA-only provenance and cost identities required')
    owners=np.asarray(layout['assignment'])
    if owners.dtype.kind not in 'iu' or owners.shape!=(len(case['aircraft_ids']),):raise ValueError('exact whole-aircraft original ownership')
    beam_ids=list(layout['beam_ids'])
    record=make_layout(case,beam_ids=beam_ids,source_slot_ids=beam_ids,anchor_aircraft_ids=layout['anchor_aircraft_ids'],
                       centres_lat_lon_deg=layout['centers'],full_width_deg=layout['widths_deg'],
                       owner_aircraft_to_beam_id=[[aircraft,beam_ids[int(owner)]] for aircraft,owner in zip(case['aircraft_ids'],owners)],
                       Ncap=prepared['Ncap'],geometry_Omega_cap_sr=prepared['Omega_cap'],geometry_Omega_sr=audit['metrics']['actual_Omega'],
                       geometry_audit={'independently_valid':True,'audit_provenance_SHA':source_and_cost_shas['audit_SHA']},source_and_cost=source_and_cost_shas)
    validate_layout(case,record,authority,geometry_check=False,published=False)
    return record

def publish_terminal(case,case_ref,authority_ref,layout,prepared,geometry_audit,source_and_cost_refs,directory,method,*,postprocess_source=None,original_algorithm_source_epoch=None):
    """Actual independently audited S1 terminal -> immutable wire + registry.

    Uses the already executed geometry audit, with no additional geometry/PHY
    evaluation. Every method pays its own conversion/validation/publication.
    The returned descriptor is consumed by the real new S2 input constructor.
    """
    from satellite_rl.shared_interface.content import content_sha,wire_bytes,read_wire
    from satellite_rl.shared_interface.contracts import Authority
    from scripts.stage1_ai_search_r4.audit import canonical_sha
    began=time.monotonic();directory=Path(directory)
    if postprocess_source is None:
        from .r7_postprocess_source import execution_source
        postprocess_source=execution_source()
    if original_algorithm_source_epoch is None:raise ValueError('original algorithm source epoch must remain explicit')
    if directory.exists():raise ValueError('immutable publication target already exists; preserve original attempt')
    directory.mkdir(parents=True)
    original_case=read_wire(pin(case_ref).read_bytes())
    if original_case.get('content_sha256')!=case.get('content_sha256') or wire_bytes(original_case)!=wire_bytes(case):raise ValueError('actual original case wire differs from publisher input')
    registry=read_json(pin(authority_ref),{});authority=Authority(registry,content_sha(registry));authority.case(case)
    if geometry_audit.get('valid')is not True:raise ValueError('invalid/unavailable terminal cannot publish a layout')
    if geometry_audit.get('parent_case_SHA')!=case['content_sha256'] or geometry_audit.get('layout_sha256')!=canonical_sha(layout):raise ValueError('already executed geometry audit does not bind this exact case/layout')
    for reference in source_and_cost_refs.values():pin(reference)
    audit={**geometry_audit,'schema':'R7BoundIndependentLayoutAuditV1','method':method,
           'original_algorithm_source_epoch':original_algorithm_source_epoch,**postprocess_source}
    audit_path=directory/'independent_layout_audit.json';atomic_json(audit_path,audit)
    shas={key:reference['sha256'] for key,reference in source_and_cost_refs.items()};shas['audit_SHA']=digest(audit_path)
    record=shared_layout(case,authority,layout,prepared,audit,shas)
    wire=directory/'layout.json';temporary=wire.with_suffix('.tmp')
    with temporary.open('wb') as stream:stream.write(wire_bytes(record));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,wire)
    published_registry=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id=dict(authority.case_by_id),
                            layout_shas=sorted(set(authority.layout_shas)|{record['content_sha256']}),allocation_shas=sorted(authority.allocation_shas))
    registry_path=directory/'registry.json';atomic_json(registry_path,published_registry)
    registry_ref=dict(path=str(registry_path),sha256=digest(registry_path));layout_ref=dict(path=str(wire),sha256=digest(wire))
    descriptor=dict(schema='R7PublishedS1TerminalV1',method=method,case_ref=case_ref,parent_case_SHA=case['content_sha256'],
                    layout_ref=layout_ref,layout_content_SHA=record['content_sha256'],authority_ref=registry_ref,
                    authority_content_SHA=content_sha(published_registry),audit_ref=dict(path=str(audit_path),sha256=digest(audit_path)),
                    source_and_cost_refs=source_and_cost_refs,reader='satellite_rl.shared_interface.stages.UniformStage2Input.from_wire',
                    N=record['N'],beam_ids=record['beam_ids'],total_request_bps_int64=case['total_request_bps_int64'],
                    original_layout_used=True,new_decoder_calls=0,new_geometry_calls=0,new_PHY_calls=0)
    descriptor.update(original_algorithm_source_epoch=original_algorithm_source_epoch,**postprocess_source)
    descriptor_path=directory/'s1_terminal.json';atomic_json(descriptor_path,descriptor)
    cost_path=directory/'publication_cost.json';atomic_json(cost_path,dict(method=method,publication_ref=dict(path=str(descriptor_path),sha256=digest(descriptor_path)),
        conversion_validation_wire_registry_descriptor_seconds=time.monotonic()-began,requires_caller_whole_function_elapsed_for_complete_cost=True,
        same_cost_rule_all_methods=True,inner_measurement_ends_before_cost_receipt_write=True,
        original_algorithm_source_epoch=original_algorithm_source_epoch,**postprocess_source))
    return dict(publication_ref=dict(path=str(descriptor_path),sha256=digest(descriptor_path)),cost_ref=dict(path=str(cost_path),sha256=digest(cost_path)),
                layout_ref=layout_ref,layout_content_SHA=record['content_sha256'],authority_ref=registry_ref,
                inner_descriptor_through_seconds=read_json(cost_path,{})['conversion_validation_wire_registry_descriptor_seconds'])

def read_for_stage2(publication_ref,physics_config):
    """Real new S2 consumption path; coefficient/capacity methods are not called."""
    from satellite_rl.shared_interface.content import content_sha
    from satellite_rl.shared_interface.contracts import Authority
    from satellite_rl.shared_interface.stages import UniformStage2Input
    descriptor=read_json(pin(publication_ref),{})
    if descriptor.get('schema')!='R7PublishedS1TerminalV1':raise ValueError('actual published S1 terminal descriptor required')
    registry=read_json(pin(descriptor['authority_ref']),{})
    authority=Authority(registry,descriptor['authority_content_SHA'])
    stage2=UniformStage2Input.from_wire(pin(descriptor['case_ref']).read_bytes(),pin(descriptor['layout_ref']).read_bytes(),authority,physics_config)
    if stage2.case['content_sha256']!=descriptor['parent_case_SHA'] or stage2.layout['content_sha256']!=descriptor['layout_content_SHA']:raise ValueError('new S2 parent/layout identities differ from actual published S1')
    return stage2
