"""Independent R5 P0 replay. No search, decoder, producer, or trainer imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.stage1_ai_search_r4.audit import audit_layout, audit_ledger as old_ledger, scene_sha, IDENTITY_FIELDS
from scripts.stage1_ai_search_r3.audit import rays_visible
from scripts.stage1_ai_search_r4.gates import finite
from .gates import evaluate_gates

CONTRACT_SHA='3555fdc7c64142f7af32fb6b32177db97e931aad52877270a97d1fce6ed74e90'
REGISTRY_SHA='df568fc4bcf51ea6b22c0ebd18ef7e23ed95ef50dcdf8830e77030fdff8afc37'
METHODS=('D5_RULE','D5_RANDOM','D5_EXACT','D5_OWNER10','D5_JOINT10','D4_OWNER10','D4_JOINT10')

def pinned_refs(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value: yield value
        for child in value.values(): yield from pinned_refs(child)
    elif isinstance(value,list):
        for child in value: yield from pinned_refs(child)

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest()

def audit_accepted(scene,cost,ncap,ocap):
    """Replay stored states only; partial EVICT may exceed Omega, never other hard bounds."""
    errors=[];q0=None
    for event in cost.get('accepted',[]):
        op=event.get('operator');layout=dict(event.get('layout',{}));layout['anchor_aircraft_ids']=event.get('anchor_aircraft_ids')
        try:
            original=np.asarray(layout['assignment'])
            if original.shape!=(len(scene['demands']),) or not np.isfinite(original).all() or not np.equal(original,np.floor(original)).all() or (original < -1).any(): raise ValueError('partial_owners')
            owners=original.astype(int);active=owners>=0
            partial=op in ('EVICT','INSERT','INSERT_SINGLETON')
            if partial:
                sub={**scene,'positions':np.asarray(scene['positions'])[active].tolist(),'demands':np.asarray(scene['demands'])[active].tolist(),'aircraft_ids':np.asarray(scene['aircraft_ids'])[active].tolist()}
                # Anchors may be orphaned; validate original anchor coordinates separately.
                centers=np.asarray(layout['centers'],float);anchors=layout['anchor_aircraft_ids'];ids=scene['aircraft_ids']
                if len(anchors)!=len(centers) or any(a not in ids for a in anchors) or not np.array_equal(centers,np.asarray(scene['positions'])[[ids.index(a) for a in anchors]]): errors.append('accepted_anchor')
                layout['assignment']=owners[active].tolist()
                if not active.any():
                    if len(centers): errors.append('partial_empty_ownership')
                    continue
                check=audit_layout(sub,layout,ncap,float('inf') if op=='EVICT' else ocap,False)
            else: check=audit_layout(scene,layout,ncap,ocap)
            if not check['valid']: errors.extend('accepted_'+x for x in check['failures'])
            key=event.get('optimization_key')
            if op=='Q0':
                if q0 is not None: errors.append('Q0_mutated')
                q0=key
            elif op in ('REFIT','SPLIT','MOVE'):
                if not q0 or any(key[j]>q0[j]+tol+1e-12*max(abs(key[j]),abs(q0[j])) for j,tol in ((1,1e-6),(2,1e-9))): errors.append('Q0_guard')
            if check.get('metrics'):
                independent=[check['metrics'][k] for k in ('maxload','top10_mean','CV','N','actual_Omega')]
                if any(v is None for v in independent): continue
                if not isinstance(key,list) or len(key)!=5 or not np.allclose(key,independent,rtol=1e-12,atol=1e-9): errors.append('accepted_key')
        except (KeyError,ValueError,TypeError,IndexError): errors.append('accepted_malformed')
    if cost.get('q0') and cost['q0'].get('optimization_key')!=q0: errors.append('Q0_fixed_record')
    return {'valid':not errors,'failures':errors}

def audit_derivation(envelope,proposal,reg,method,source_sha):
    errors=[]
    if (envelope.get('identity')!={k:reg[k] for k in IDENTITY_FIELDS} or envelope.get('actual_method')!=method
        or envelope.get('new_source_sha256')!=source_sha or envelope.get('contract_sha256')!=CONTRACT_SHA
        or envelope.get('original_prepared_ref')!=reg['prepared']['stage_output']
        or envelope.get('original_teacher_ref')!=reg['teacher']['stage_output']): errors.append('derivation_lineage')
    if method.startswith('D4_'):
        d=envelope.get('result_derivation') or {};inner=d.get('proposal_after',{})
        if d.get('diagnostic_method')!=method.split('_',1)[1] or d.get('proposal_before')!=proposal: errors.append('D4_actual_outer_method')
        if d.get('proposal_before_sha256')!=canonical_sha(proposal) or d.get('proposal_after_sha256')!=canonical_sha(inner): errors.append('D4_conversion_bytes_SHA')
        if inner.get('source_method')!='TEACHER_EXACT' or inner.get('source_sha256')!=reg['prepared']['source_sha256']: errors.append('D4_compatibility_source')
        if any(inner.get(k)!=reg[k] for k in IDENTITY_FIELDS): errors.append('D4_inner_identity')
        for new,old in (('slot_ids','slot_ids'),('anchor_aircraft_ids','anchor_aircraft_ids'),('scores_n_by_K','scores_nxK_float32_finite'),('mask_n_by_K','score_mask_nxK_bool')):
            if proposal.get(new)!=inner.get(old): errors.append('D4_conversion_arrays_'+new)
        offset=np.asarray(inner.get('direction_offset_Kx3'),float)
        if offset.shape!=(len(proposal.get('slot_ids',[])),3) or not np.equal(offset,0).all(): errors.append('D4_zero_offset')
        conversion=d.get('conversion_cost',{})
        if not finite(conversion.get('seconds')) or conversion.get('raypairs')!=0 or conversion.get('hash_calls')!=2: errors.append('D4_conversion_cost')
    return {'valid':not errors,'failures':errors}

class Reader:
    def __init__(self,path_map=None): self.path_map=path_map or {};self.refs=[]
    def path(self,ref):
        if not isinstance(ref,dict) or not isinstance(ref.get('sha256'),str): raise ValueError('unpinned_reference')
        original=ref['path'];p=Path(self.path_map.get(original,ref.get('local_path',original)))
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=ref['sha256']: raise ValueError('reference_SHA_mismatch:'+original)
        self.refs.append({'path':str(p),'original_path':original,'sha256':actual,'bytes':p.stat().st_size})
        return p
    def json(self,ref): return json.loads(self.path(ref).read_text(encoding='utf-8-sig'))

def load_scene(reader,ref):
    with np.load(reader.path(ref),allow_pickle=False) as z:
        return {'positions':z['positions'].tolist(),'demands':z['demands'].tolist(),'aircraft_ids':[str(x) for x in z['aircraft_ids']],'profile':str(z['profile'].item())}

def digest(*args): return hashlib.sha256(json.dumps([51001,*args],ensure_ascii=False,separators=(',',':')).encode('utf8')).digest()

def audit_proposal(scene,proposal,expected,prepared,teacher,method,source_sha):
    errors=[]
    try:
        identity={k:expected[k] for k in IDENTITY_FIELDS}
        if proposal.get('case_identity')!=identity or proposal.get('input_sha')!=expected['input_sha256'] or proposal.get('role_sha')!=expected['identity_sha256']: errors.append('proposal_identity')
        provenance=proposal.get('provenance',{})
        if provenance.get('source_method')!=method or provenance.get('producer_sha256')!=source_sha or provenance.get('prepared_source_sha256')!=prepared.get('source_sha256'): errors.append('proposal_actual_producer_semantics')
        ids=scene['aircraft_ids'];n=len(ids);cid=expected['case_id'];rays,_=rays_visible(scene['positions'],scene['profile'])
        if method=='RULE':
            anchors=prepared['anchors'];slots=prepared['raw_w']['beam_ids'];scores=(-np.sum((rays[:,None]-rays[[ids.index(a) for a in anchors]][None])**2,axis=-1)).astype(np.float32);mask=np.ones(scores.shape,bool)
        elif method=='RANDOM':
            anchors=sorted(ids,key=lambda a:(digest(cid,'anchor',a),a))[:len(prepared['anchors'])];slots=['random:'+a for a in anchors]
            scores=np.asarray([[int.from_bytes(digest(cid,'score',a,b)[:8],'big')/2**64 for b in slots] for a in ids],np.float32);mask=np.ones(scores.shape,bool)
        else:
            layout=teacher['layout'];anchors=list(layout['anchor_aircraft_ids']);slots=list(layout['beam_ids']);owner=np.asarray(layout['assignment'],int);scores=np.zeros((n,len(slots)),np.float32);scores[np.arange(n),owner]=1;mask=np.ones(scores.shape,bool)
            if method in ('OWNER10','JOINT10'):
                selected=sorted(range(n),key=lambda i:(digest(cid,'mask',ids[i]),ids[i]))[:math.floor(.1*n)];mask[selected,owner[selected]]=False
            if method=='JOINT10' and n>1:
                selected=sorted(range(len(slots)),key=lambda j:(digest(cid,'r5_anchor_corrupt',slots[j]),slots[j]))[:math.floor(.1*len(slots))]
                for j in selected:
                    a=ids.index(anchors[j]);candidates=[i for i in range(n) if i!=a];b=min(candidates,key=lambda i:(float(np.sum((rays[i]-rays[a])**2)),ids[i]));anchors[j]=ids[b]
        actual=np.asarray(proposal['scores_n_by_K'],np.float32);actual_mask=np.asarray(proposal['mask_n_by_K'])
        if proposal['slot_ids']!=slots or proposal['anchor_aircraft_ids']!=anchors: errors.append('fixed_slot_anchor_definition')
        if actual.shape!=scores.shape or not np.array_equal(actual,scores): errors.append('fixed_score_definition')
        if actual_mask.dtype!=np.bool_ or not np.array_equal(actual_mask,mask): errors.append('fixed_mask_definition')
        if 'direction_offset_Kx3' in proposal or 'offset' in proposal: errors.append('V2_offset_forbidden')
        return {'valid':not errors,'failures':errors,'ready':True,'shape':list(scores.shape)}
    except (ValueError,KeyError,TypeError,IndexError) as e: return {'valid':False,'failures':['proposal_malformed:'+str(e)]}

def audit_decoder_ledger(cost,n,decoder,expected_nominal=5):
    if decoder=='D4': return old_ledger(cost,n,'decoder')
    errors=[]
    if not isinstance(cost,dict): return {'valid':False,'failures':['ledger_missing']}
    for key,cap in (('raypairs',2048*n),('proposals',512),('refits',8)):
        v=cost.get(key)
        if not finite(v) or int(v)!=v or v>cap: errors.append('budget_'+key)
    if not finite(cost.get('seconds')): errors.append('seconds_missing')
    for key in ('geometry_calls','center_constructions','sort_input_count','scan_input_count','hash_calls','cache_hits','rank_geometry_pairs'):
        v=cost.get(key)
        if not finite(v) or int(v)!=v: errors.append('operation_ledger_'+key)
    if cost.get('terminal_reserve_raypairs')!=3*n: errors.append('terminal_reservation')
    if cost.get('seconds_nominal')!=expected_nominal or not 0<=expected_nominal<=5 or cost.get('raypairs_max')!=2048*n or cost.get('proposals_max')!=512 or cost.get('refits_max')!=8: errors.append('declared_caps')
    phasecaps={'INITIAL':0,'EVICT':128,'INSERT':256,'REFIT':8,'SPLIT':64,'MOVE':56,'AUDIT':0};order=list(phasecaps);phases=cost.get('phases',[]);seen=[]
    for phase in phases:
        op=phase.get('operator',phase.get('name'));v=phase.get('proposals')
        if op not in phasecaps or not finite(v) or int(v)!=v or v>phasecaps.get(op,0): errors.append('phase_caps')
        seen.append(op)
    if len(seen)!=len(set(seen)) or any(op not in order for op in seen) or [order.index(op) for op in seen if op in order]!=sorted(order.index(op) for op in seen if op in order): errors.append('phase_order')
    if sum(p.get('proposals',0) for p in phases)!=cost.get('proposals'): errors.append('phase_sum')
    return {'valid':not errors,'failures':errors}

def mechanisms(metrics,wenv):
    widths=np.asarray(metrics['widths'],float);old=np.asarray(wenv['widths'],float)
    return {'N_changed':metrics['N']!=wenv['N'],'width_distribution_changed':len(widths)!=len(old) or not np.allclose(np.sort(widths),np.sort(old),rtol=0,atol=1e-12),'nonfloor_heterogeneous_widths':bool((widths>.3+1e-12).any() and len(widths)>1 and np.ptp(widths)>1e-12)}

def audit_cost(cost,raw_ref,method,registry_case,reader):
    """Full cold reconstruction pays pinned old input/prepare/T exactly once."""
    errors=[]
    if cost.get('raw_ref')!=raw_ref or cost.get('case_id')!=registry_case['case_id'] or cost.get('method')!=method: errors.append('cost_backlink_identity')
    components=cost.get('components',{})
    keys=('input','prepare','teacher','new_registry_read','new_guarded_stage_wall','row_index_status_publication')
    if not all(finite(components.get(k)) for k in keys): errors.append('cost_components_unknown')
    elif not finite(cost.get('complete_method_seconds')) or not math.isclose(sum(components[k] for k in keys),cost['complete_method_seconds'],rel_tol=1e-12,abs_tol=1e-9): errors.append('cost_sum')
    if cost.get('unknown_algorithm_spend') is not False: errors.append('unknown_spend')
    if cost.get('old_cold_cost_ref')!=registry_case['teacher']['publication_cost'] or cost.get('old_prepare_ref')!=registry_case['prepared']['stage_output']: errors.append('cold_reference_lineage')
    old=reader.json(registry_case['teacher']['publication_cost'])
    if old.get('unknown_cost') or old.get('unknown_algorithm_spend'): errors.append('old_unknown_not_known')
    oldcomponents=old.get('components',{})
    input_receipt=reader.json(registry_case['scene_source']['input_read_receipt'])
    teacher_row=reader.json(registry_case['teacher']['row'])
    guards=[reader.json(ref) for ref in registry_case['prepared']['guard_candidates']]
    matching=[g for g in guards if g.get('output_ref')=={k:registry_case['prepared']['stage_output'][k] for k in ('path','sha256')} and g.get('status')=='completed' and g.get('unknown_algorithm_spend') is False]
    actual={'input':input_receipt.get('actual_seconds'),'prepare':matching[0].get('actual_stage_wall_seconds') if matching else None,'teacher':teacher_row.get('stage_guard',{}).get('actual_stage_wall_seconds')}
    for key in actual:
        if not finite(actual[key]) or not finite(oldcomponents.get(key)) or not math.isclose(actual[key],oldcomponents[key],rel_tol=1e-12,abs_tol=1e-9): errors.append('old_actual_'+key)
    if teacher_row.get('stage_guard',{}).get('unknown_algorithm_spend') is not False: errors.append('old_teacher_unknown')
    for key in ('input','prepare'):
        if not finite(components.get(key)) or not finite(oldcomponents.get(key)) or not math.isclose(components[key],oldcomponents[key],rel_tol=1e-12,abs_tol=1e-9): errors.append('cold_'+key+'_not_restored')
    teacher_expected=oldcomponents.get('teacher') if method.split('_',1)[1] in ('EXACT','OWNER10','JOINT10') else 0
    if not finite(components.get('teacher')) or not finite(teacher_expected) or not math.isclose(components['teacher'],teacher_expected,rel_tol=1e-12,abs_tol=1e-9): errors.append('cold_teacher_not_restored')
    return {'valid':not errors,'failures':errors,'seconds':cost.get('complete_method_seconds')}

def audit_request(request):
    reader=Reader(request.get('path_map'));registry=reader.json(request['registry_ref']);contract=reader.json(request['contract_ref'])
    if request['registry_ref']['sha256']!=REGISTRY_SHA or request['contract_ref']['sha256']!=CONTRACT_SHA: raise ValueError('signed_contract_or_registry_SHA')
    # Every original reference is retained, including historical failures/D4 rows;
    # SHA replay reads bytes only and never turns them into new calls.
    for ref in pinned_refs(registry):
        if ref.get('local_path'): reader.path_map.setdefault(ref['path'],ref['local_path'])
    for ref in pinned_refs(registry): reader.path(ref)
    data=contract['data'];rc=registry['cases']
    if len(rc)!=24 or [c['input_case_index'] for c in rc]!=data['case_indices']: raise ValueError('fixed_24_case_indices')
    if set((c['mother_id'],c['profile']) for c in rc)!=set((m,p) for m in data['mother_ids'] for p in data['profiles']): raise ValueError('fixed_mother_profile_crossproduct')
    if any(c['identity_sha256']!=data['identity_sha256'] or c['scene_source']['packed']['sha256']!=data['original_dataset_sha256'] for c in rc): raise ValueError('fixed_identity_dataset')
    expected_ids=[c['case_id'] for c in registry['cases']];registered={c['case_id']:c for c in registry['cases']};entries={};errors=[]
    for ref in request.get('manifest_refs',[]):
        m=reader.json(ref)
        for nested in pinned_refs(m): reader.path(nested)
        if m.get('schema')!='R5P0ResultManifestV1': raise ValueError('manifest_schema')
        if any(m.get(k)!=0 for k in ('MILP_calls','training_updates','new_prepare_calls','new_teacher_calls')): raise ValueError('forbidden_new_science')
        if m.get('design_ref',{}).get('sha256')!=CONTRACT_SHA or m.get('registry_ref',{}).get('sha256')!=REGISTRY_SHA: raise ValueError('manifest_contract_registry')
        for key in ('source_manifest_ref','job_ref','config_ref','identity_ref','design_ref','registry_ref'): reader.json(m[key])
        source=reader.json(m['source_manifest_ref'])
        source_root=request.get('source_roots_map',{}).get(source['source_root'],source['source_root'])
        for rel,sha in source['files'].items(): reader.path({'path':str(Path(source_root)/rel),'sha256':sha})
        qualifications={k:reader.json(v) for k,v in m.get('qualification_refs',{}).items()}
        if set(qualifications)!={'geometry','process'}: raise ValueError('qualification_pair_missing')
        geo=qualifications['geometry'];proc=qualifications['process']
        if any(q.get('passed') is not True or q.get('scientific_commit')!=m['source_commit'] for q in qualifications.values()): raise ValueError('qualification_source_pass')
        if geo.get('design_sha256')!=CONTRACT_SHA or geo.get('geometry_fixture_cases')!=8 or proc.get('process_stub_calls')!=3 or proc.get('geometry_calls')!=0 or proc.get('CUDA_calls')!=0: raise ValueError('qualification_bounded_real_calls')
        for e in m.get('cases',[]):
            if e['case_id'] in entries or e['case_id'] not in registered: raise ValueError('duplicate_or_unregistered_case')
            entries[e['case_id']]=(e,m)
    cases=[]
    for cid in expected_ids:
        reg=registered[cid];case={'case_id':cid,'mother_id':reg['mother_id'],'D5':{},'D4':{},'complete_slots':False,'audit_closed':False,'witness_valid':False,'failures':[]}
        cases.append(case)
        if cid not in entries: case['failures'].append('missing_case');continue
        e,m=entries[cid]
        try:
            if e.get('registry_case')!=reg: raise ValueError('registry_case_mutation')
            if any(e.get(k)!=reg[k] for k in IDENTITY_FIELDS) or reg['role']!='development': raise ValueError('case_identity_role')
            scene=load_scene(reader,reg['scene_source']['raw_scene']);prep=reader.json(reg['prepared']['stage_output'])['result'];teacher=reader.json(reg['teacher']['stage_output'])['result']
            if scene_sha(scene)!=reg['input_sha256'] or prep['identity']!={k:reg[k] for k in IDENTITY_FIELDS}: raise ValueError('input_or_prepared_identity')
            if prep.get('source_sha256')!=reg['prepared']['source_sha256'] or teacher.get('source_sha256')!=reg['teacher']['source_sha256']: raise ValueError('original_source_rewritten')
            n=len(scene['demands']);nc=prep['Ncap'];oc=prep['Omega_cap'];w=audit_layout(scene,prep['raw_w'],200,float('inf'),False,False);we=audit_layout(scene,prep['Wenv'],nc,oc,False);ta=audit_layout(scene,teacher['layout'],nc,oc)
            case['witness_valid']=bool(w['valid'] and we['valid'] and nc==min(200,n,math.ceil(1.25*w['metrics']['N'])) and abs(oc-w['metrics']['actual_Omega'])<=1e-12)
            teacher_identity=all(teacher.get(k)==reg[k] for k in IDENTITY_FIELDS)
            teacher_budget=old_ledger(teacher['cost'],n,'TEACHER')['valid']
            case['teacher']={'valid':ta['valid'] and teacher_identity and teacher_budget,'published':teacher.get('status')=='available','identity_closed':teacher_identity,'budget_closed':teacher_budget,'status':teacher['status'],'metrics':ta['metrics']}
            for method in METHODS:
                if method not in e.get('raw_refs',{}) or method not in e.get('publication_cost_refs',{}): case['failures'].append(method+':missing_slot');continue
                try:
                    raw=reader.json(e['raw_refs'][method]);cost=reader.json(e['publication_cost_refs'][method]);r=raw['result'];decoder,actual_method=method.split('_',1);fails=[];published=r.get('status') in ('success','fallback')
                    for nested in pinned_refs(raw): reader.path(nested)
                    if raw.get('method')!=method or raw.get('decoder')!=decoder or raw.get('proposal_method')!=actual_method or any(raw.get(k)!=reg[k] for k in IDENTITY_FIELDS): fails.append('outer_actual_identity_semantics')
                    if raw.get('original_prepared_ref')!=reg['prepared']['stage_output'] or raw.get('original_teacher_ref')!=reg['teacher']['stage_output']: fails.append('original_refs')
                    if raw.get('source_sha256')!=m['source_manifest_ref']['sha256'] or raw.get('source_commit')!=m['source_commit']: fails.append('execution_source')
                    guard=raw.get('stage_guard',{})
                    if guard.get('output_ref'): reader.json(guard['output_ref'])
                    if published and (guard.get('status')!='completed' or raw.get('decoder_invoked') is not True): fails.append('uncompleted_or_uncalled_published')
                    if not finite(guard.get('peak_parent_children_rss_bytes')) or guard.get('peak_parent_children_rss_bytes',0)>2147483648 or guard.get('unknown_algorithm_spend') is not False: fails.append('guard_resource_or_unknown')
                    proposal=reader.json(raw['proposal_ref'])
                    pa=audit_proposal(scene,proposal,reg,prep,teacher,actual_method,m['source_manifest_ref']['sha256'])
                    if not pa['valid']: fails.extend(pa['failures'])
                    derivation=audit_derivation(reader.json(raw['derived_envelope_ref']),proposal,reg,method,m['source_manifest_ref']['sha256'])
                    if not derivation['valid']: fails.extend(derivation['failures'])
                    childcost=raw.get('child_measured_costs',{})
                    childkeys=('actual_provider_read_seconds','actual_proposal_generation_seconds','actual_proposal_publication_seconds','actual_decoder_algorithm_seconds')
                    if not all(finite(childcost.get(k)) for k in childkeys): fails.append('actual_child_cost_missing')
                    elif finite(guard.get('actual_stage_wall_seconds')) and sum(childcost[k] for k in childkeys)>guard['actual_stage_wall_seconds']+1e-6: fails.append('child_cost_exceeds_stage')
                    if cost.get('child_measured_costs')!=childcost: fails.append('child_cost_backlink')
                    la=audit_layout(scene,r.get('layout',{}),nc,oc) if published else {'valid':False,'metrics':None,'failures':['unpublished']}
                    budget=audit_decoder_ledger(r.get('cost'),n,decoder)
                    if decoder=='D5' and isinstance(r.get('cost'),dict):
                        accepted=audit_accepted(scene,r['cost'],nc,oc)
                        if not accepted['valid']: fails.extend(accepted['failures'])
                    if not la['valid']: fails.extend(la['failures'])
                    if not budget['valid']: fails.extend(budget['failures'])
                    ca=audit_cost(cost,e['raw_refs'][method],method,reg,reader)
                    wall=guard.get('actual_stage_wall_seconds');paid=cost.get('components',{}).get('new_guarded_stage_wall')
                    if not finite(wall) or not finite(paid) or not math.isclose(wall,paid,rel_tol=1e-12,abs_tol=1e-9): ca['valid']=False;ca['failures'].append('stage_wall_not_paid')
                    endpoint_ref=e.get('publication_endpoint_refs',{}).get(method)
                    if not endpoint_ref: ca['valid']=False;ca['failures'].append('publication_endpoint_missing')
                    else:
                        endpoint=reader.json(endpoint_ref);endpoint_seconds=endpoint.get('actual_cost_index_status_finalization_seconds');total=endpoint.get('complete_method_seconds_with_endpoint')
                        if endpoint.get('raw_ref')!=e['raw_refs'][method] or endpoint.get('cost_ref')!=e['publication_cost_refs'][method]: ca['valid']=False;ca['failures'].append('endpoint_backlink')
                        if not finite(endpoint_seconds) or not finite(ca['seconds']) or not finite(total) or not math.isclose(total,ca['seconds']+endpoint_seconds,rel_tol=1e-12,abs_tol=1e-9): ca['valid']=False;ca['failures'].append('publication_endpoint_cost')
                        else: ca['seconds']=total
                    if la['valid']:
                        for k in ('N','maxload','top10_mean','CV','actual_Omega','total_request'):
                            x=r.get('metrics',{}).get({'actual_Omega':'solid_angle_sum_sr','total_request':'total_demand'}.get(k,k));y=la['metrics'][k];tol=1e-9 if k=='CV' else 1e-6 if k in ('maxload','top10_mean','total_request') else 1e-12
                            if y is None:
                                if k=='CV' and x is not None: fails.append('producer_zero_quality_'+k)
                            elif not finite(x) or not math.isclose(x,y,rel_tol=1e-12,abs_tol=tol): fails.append('producer_metric_'+k)
                    if not ca['valid']: fails.extend(ca['failures'])
                    row={'status':r.get('status'),'published':published,'valid':not fails,'identity_closed':not any('identity' in x for x in fails),'budget_closed':budget['valid'],'cost_closed':ca['valid'],'publication_seconds':ca['seconds'],'metrics':la['metrics'],'proposal_audit':pa,'failures':fails}
                    if la['valid']: row['mechanisms']=mechanisms(la['metrics'],we['metrics'])
                    case[decoder][actual_method]=row;case['failures'].extend(method+':'+x for x in fails)
                except (ValueError,KeyError,OSError,TypeError,IndexError) as exc:
                    decoder,actual_method=method.split('_',1)
                    failure='malformed_or_unpublished_slot:'+str(exc)
                    case[decoder][actual_method]={'status':'missing_or_malformed','published':False,'valid':False,'identity_closed':False,'budget_closed':False,'cost_closed':False,'metrics':None,'failures':[failure]}
                    case['failures'].append(method+':'+failure)
            case['complete_slots']=all(m in e.get('raw_refs',{}) and m in e.get('publication_cost_refs',{}) for m in METHODS)
            case['audit_closed']=case['witness_valid'] and case['teacher']['valid'] and not case['failures']
        except (ValueError,KeyError,OSError,TypeError,IndexError) as exc: case['failures'].append(str(exc));errors.append({'case_id':cid,'error':str(exc)})
    return {'schema':'R5P0IndependentAuditV1','contract_sha256':CONTRACT_SHA,'registry_sha256':REGISTRY_SHA,'algorithm_calls':0,'cases':cases,'errors':errors,'gates':evaluate_gates(cases,expected_ids,request.get('engineering_g0',False)),'raw_frozen_refs':reader.refs,'status':'complete' if len(entries)==24 and all(c['complete_slots'] for c in cases) else 'partial'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=audit_request(json.loads(Path(a.request).read_text(encoding='utf-8-sig')));out=Path(a.output)
    if out.exists(): raise ValueError('immutable_output_exists')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
if __name__=='__main__':main()
