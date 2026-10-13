"""Actual AI outputs wrapped for immutable D5 EXACT compatibility only."""
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
from .features import array_sha,record_sha

REQUIRED_PROVENANCE=('checkpoint_file_SHA','model_tensor_SHA','model_graph_source_SHA','feature_graph_source_SHA',
    'normalization_SHA','model_initial_state_SHA','adapter_source_SHA','b7_decoder_source_SHA','source_epoch')


def build_ai_proposal(prepared,output,provenance,*,inference_cost,tensor_asset_path=None):
    started=time.monotonic()
    if any(not provenance.get(k) for k in REQUIRED_PROVENANCE):raise ValueError('missing AI provenance')
    if inference_cost.get('model_forward_calls')!=1:raise ValueError('exactly one actual model forward required')
    n=len(prepared.scene.aircraft_ids)
    def convert(name,shape):
        value=output[name]
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        value=np.asarray(value,dtype=np.float32)
        if value.shape== (1,*shape):value=value[0]
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid actual model '+name)
        return np.array(value,copy=True,order='C')
    enable=convert('enable',(200,));pointer=convert('pointer',(200,n));owner=convert('owner',(n,200))
    cap=min(200,n,prepared.Ncap)
    if cap<1:raise ValueError('unavailable prepared case')
    selected=np.flatnonzero(enable>=0).tolist()
    if not selected:selected=[min(range(200),key=lambda q:(-float(enable[q]),q))]
    if len(selected)>cap:selected=sorted(selected,key=lambda q:(-float(enable[q]),q))[:cap]
    selected=sorted(selected)
    ids=tuple(prepared.scene.aircraft_ids)
    anchors=[ids[min(range(n),key=lambda i:(-float(pointer[q,i]),ids[i]))] for q in selected]
    scores=np.ascontiguousarray(owner[:,selected]);mask=np.ones(scores.shape,bool)
    prov=dict(case_identity=copy.deepcopy(prepared.identity),input_sha=prepared.scene.input_sha256,
        role_sha=prepared.identity['identity_sha256'],prepared_source_sha256=prepared.source_sha256,
        producer_sha256=provenance['adapter_source_SHA'],source_method='EXACT')
    inner=dict(schema='ProposalV2',status='ready',case_identity=copy.deepcopy(prepared.identity),
        input_sha=prepared.scene.input_sha256,role_sha=prepared.identity['identity_sha256'],provenance=prov,
        slot_ids=[f'q{q:03d}' for q in selected],anchor_aircraft_ids=anchors,scores_n_by_K=scores.tolist(),
        mask_n_by_K=mask.tolist(),construction_cost=dict(seconds=time.monotonic()-started,raypairs=0,
        hash_calls=0,sort_input_count=200+n*len(selected),scan_input_count=n))
    hashes=dict(enable=array_sha(enable),pointer=array_sha(pointer),owner=array_sha(owner),selected_owner=array_sha(scores),mask=array_sha(mask))
    if tensor_asset_path is None:
        tensor_refs=dict(values=dict(enable=enable.tolist(),pointer=pointer.tolist(),owner=owner.tolist()),sha256=hashes)
    else:
        path=Path(tensor_asset_path)
        if path.exists():raise ValueError('never overwrite inference tensor assets')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:np.savez_compressed(stream,enable=enable,pointer=pointer,owner=owner)
        tensor_refs=dict(npz_path=str(path.resolve()),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sha256=hashes)
    result=dict(schema='AIProposalEnvelopeV1',actual_method='AI',**copy.deepcopy(provenance),
        case_identity=copy.deepcopy(prepared.identity),input_SHA=prepared.scene.input_sha256,
        role_SHA=prepared.identity['identity_sha256'],b7_prepared_source_SHA=prepared.source_sha256,
        aircraft_ids=list(ids),full200_enable_logits=enable.tolist(),selected_query_mapping=selected,
        valid_aircraft_pointer_owner_tensor_refs_and_SHAs=tensor_refs,
        inference_cost=copy.deepcopy(inference_cost),adapter_cost=dict(seconds=time.monotonic()-started),
        inner=inner,inner_JSON_SHA=record_sha(inner),canonical_before_after_SHA=dict(before=hashes['selected_owner'],after=array_sha(np.asarray(inner['scores_n_by_K'],np.float32))))
    result['outer_JSON_SHA']=record_sha(result)
    audit_ai_proposal(prepared,result)
    return result


def audit_ai_proposal(prepared,envelope):
    if envelope.get('actual_method')!='AI' or envelope.get('schema')!='AIProposalEnvelopeV1':raise ValueError('outer AI schema')
    if any(not envelope.get(k) for k in REQUIRED_PROVENANCE):raise ValueError('missing AI provenance')
    if envelope['case_identity']!=prepared.identity or envelope['input_SHA']!=prepared.scene.input_sha256 or envelope['role_SHA']!=prepared.identity['identity_sha256'] or envelope['b7_prepared_source_SHA']!=prepared.source_sha256:raise ValueError('AI role/input/source mismatch')
    if any('offset' in str(k).lower() for k in envelope) or any('offset' in str(k).lower() for k in envelope['inner']):raise ValueError('offset forbidden')
    outer=copy.deepcopy(envelope);sha=outer.pop('outer_JSON_SHA')
    if record_sha(outer)!=sha or record_sha(envelope['inner'])!=envelope['inner_JSON_SHA']:raise ValueError('altered AI envelope')
    inner=envelope['inner'];refs=envelope['valid_aircraft_pointer_owner_tensor_refs_and_SHAs'];selected=envelope['selected_query_mapping']
    if 'npz_path' in refs:
        path=Path(refs['npz_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=refs['file_sha256']:raise ValueError('altered tensor asset')
        with np.load(path,allow_pickle=False) as arrays:values={k:arrays[k].copy() for k in ('enable','pointer','owner')}
    else:values=refs['values']
    owner=np.asarray(values['owner'],np.float32);pointer=np.asarray(values['pointer'],np.float32);enable=np.asarray(values['enable'],np.float32)
    for name,array in [('enable',enable),('pointer',pointer),('owner',owner)]:
        if not np.isfinite(array).all() or array_sha(array)!=refs['sha256'][name]:raise ValueError('altered model arrays')
    if envelope['aircraft_ids']!=list(prepared.scene.aircraft_ids) or enable.tolist()!=envelope['full200_enable_logits']:raise ValueError('aircraft/enable changed')
    n=len(prepared.scene.aircraft_ids)
    if enable.shape!=(200,) or pointer.shape!=(200,n) or owner.shape!=(n,200):raise ValueError('actual model array shape')
    cap=min(200,n,prepared.Ncap)
    expected_selected=np.flatnonzero(enable>=0).tolist()
    if not expected_selected:expected_selected=[min(range(200),key=lambda q:(-float(enable[q]),q))]
    if len(expected_selected)>cap:expected_selected=sorted(expected_selected,key=lambda q:(-float(enable[q]),q))[:cap]
    if sorted(expected_selected)!=selected or envelope['inference_cost'].get('model_forward_calls')!=1:raise ValueError('selection/forward count changed')
    scores=np.asarray(inner['scores_n_by_K'],np.float32);mask=np.asarray(inner['mask_n_by_K'])
    if not np.array_equal(scores,owner[:,selected]) or mask.dtype!=bool or not mask.all() or mask.shape!=scores.shape:raise ValueError('scores/mask changed')
    if inner['slot_ids']!=[f'q{q:03d}' for q in selected] or inner['provenance']['source_method']!='EXACT':raise ValueError('compatibility mapping')
    prov=inner['provenance']
    if inner['case_identity']!=prepared.identity or prov['case_identity']!=prepared.identity or prov['input_sha']!=prepared.scene.input_sha256 or prov['role_sha']!=prepared.identity['identity_sha256'] or prov['prepared_source_sha256']!=prepared.source_sha256 or prov['producer_sha256']!=envelope['adapter_source_SHA']:raise ValueError('inner source/role mismatch')
    if array_sha(scores)!=refs['sha256']['selected_owner'] or array_sha(mask)!=refs['sha256']['mask'] or envelope['canonical_before_after_SHA']!={'before':array_sha(scores),'after':array_sha(scores)}:raise ValueError('conversion changed bytes')
    ids=tuple(prepared.scene.aircraft_ids)
    expected=[ids[min(range(len(ids)),key=lambda i:(-float(pointer[q,i]),ids[i]))] for q in selected]
    if expected!=inner['anchor_aircraft_ids']:raise ValueError('anchors changed')
    return True
