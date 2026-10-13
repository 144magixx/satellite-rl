"""Convert independently audited S1 arrays to the shared variable-layout object.

No source path is part of immutable content. Geometry is already independently
audited in the originating full label/node receipt; this conversion checks the
typed structure and conservation without adding unregistered ray work.
"""
import numpy as np

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
