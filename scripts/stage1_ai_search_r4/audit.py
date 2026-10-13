"""Independent read-only raw R4 audit. Never runs geometry search or training.

Every read is pinned by original file SHA. Only owner-pair geometry is replayed;
producer audits/metrics are compared against independent values, never trusted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.stage1_ai_search_r3.audit import rays_visible,solid_angle
from .gates import METHODS,evaluate_gates,finite

CONTRACT_SHA='512482fa8f350df120b0bb501224a9b433b4ffd1ad05099a90f75c8bd38891c7'
IDENTITY_FIELDS=('case_id','mother_id','profile','input_sha256','identity_sha256','role')

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def scene_sha(scene):
    return canonical_sha({'positions':np.asarray(scene['positions'],float).tolist(),'demands':np.asarray(scene['demands'],float).tolist(),'aircraft_ids':list(scene['aircraft_ids']),'profile':scene.get('profile','europe_v1')})

class Reader:
    def __init__(self,path_map=None):self.path_map=path_map or {};self.refs=[]
    def path(self,ref):
        if not isinstance(ref,dict) or not isinstance(ref.get('sha256'),str):raise ValueError('unpinned_reference')
        original=ref['path'];path=Path(self.path_map.get(original,original))
        actual=sha256(path)
        if actual!=ref['sha256']:raise ValueError('raw_reference_SHA_mismatch:'+original)
        self.refs.append({'path':str(path),'original_path':original,'sha256':actual,'bytes':path.stat().st_size});return path
    def json(self,ref):return json.loads(self.path(ref).read_text(encoding='utf-8-sig'))

def audit_layout(scene,layout,ncap,omega_cap,anchor_required=True,minimum_envelope=True):
    failures=[]
    try:
        pos=np.asarray(scene['positions'],float);dem=np.asarray(scene['demands'],float);ids=list(scene['aircraft_ids'])
        centers=np.asarray(layout['centers'],float);widths=np.asarray(layout['widths_deg'],float);owners=np.asarray(layout['assignment']);b=len(centers);n=len(dem)
        if not n or pos.shape!=(n,2) or dem.shape!=(n,) or len(ids)!=n or len(set(ids))!=n or not np.isfinite(pos).all() or not np.isfinite(dem).all() or (dem<0).any():raise ValueError('scene_shape_finite_identity')
        if centers.shape!=(b,2) or widths.shape!=(b,) or owners.shape!=(n,) or not np.isfinite(centers).all() or not np.isfinite(widths).all() or not np.isfinite(owners).all():raise ValueError('layout_shape_finite')
        if not 1<=b<=min(200,n,ncap):failures.append('N_cap')
        if (widths<.3).any() or (widths>1).any():failures.append('width_bounds')
        if not np.equal(owners,np.floor(owners)).all() or (owners<0).any() or (owners>=b).any():raise ValueError('unique_integral_complete_ownership')
        owners=owners.astype(int);counts=np.bincount(owners,minlength=b)
        if (counts==0).any():failures.append('empty_beam')
        bid=layout.get('beam_ids')
        if not bid or len(bid)!=b or len(set(bid))!=b:failures.append('beam_IDs')
        anchors=layout.get('anchor_aircraft_ids')
        if anchor_required:
            if not isinstance(anchors,list) or len(anchors)!=b or any(a not in ids for a in anchors):failures.append('anchor_IDs')
            elif not np.array_equal(centers,pos[[ids.index(a) for a in anchors]]):failures.append('final_anchor_projection')
        ur,uv=rays_visible(pos,scene.get('profile','europe_v1'));cr,cv=rays_visible(centers,scene.get('profile','europe_v1'))
        if not uv.all() or not cv.all():failures.append('visibility')
        chords=np.sum((ur-cr[owners])**2,axis=1);limits=4*np.sin(np.deg2rad(widths[owners]/2)/2)**2
        if (chords>limits+1e-14).any():failures.append('coverage')
        if minimum_envelope:
            for j in range(b):
                owned=chords[owners==j]
                if len(owned):
                    required=max(.3,float(np.nextafter(np.rad2deg(4*np.arcsin(min(math.sqrt(float(owned.max()))/2,1.))),np.inf)))
                    if not math.isclose(float(widths[j]),required,rel_tol=0,abs_tol=1e-12):failures.append('not_actual_minimum_envelope')
        loads=np.bincount(owners,weights=dem,minlength=b);total=float(dem.sum());area=solid_angle(widths)
        if not np.isclose(loads.sum(),total,rtol=1e-12,atol=1e-6):failures.append('request_conservation')
        if area>omega_cap+1e-12:failures.append('Omega_cap')
        tail=float(np.sort(loads)[-max(1,math.ceil(.1*b)):].mean())
        quality={'N':b,'maxload':float(loads.max()) if total else None,'top10_mean':tail if total else None,'CV':float(loads.std()/loads.mean()) if total else None,'actual_Omega':area,'total_request':total,'loads':loads.tolist(),'widths':widths.tolist()}
        return {'valid':not failures,'failures':failures,'metrics':quality,'owner_raypairs_replayed':n,'zero_request_quality_undefined':total==0}
    except (KeyError,ValueError,TypeError,IndexError,OverflowError) as exc:return {'valid':False,'failures':['malformed_raw_layout:'+str(exc)],'metrics':None}

def audit_ledger(cost,n,kind):
    limits={'prepare':(1024,0,0,10),'TEACHER':(4096,2048,16,30),'decoder':(2048,512,8,5)}[kind];failures=[]
    if not isinstance(cost,dict):return {'valid':False,'failures':['ledger_missing']}
    for key,cap in zip(('raypairs','proposals','refits'),(limits[0]*n,limits[1],limits[2])):
        value=cost.get(key)
        if not finite(value) or int(value)!=value or value>cap:failures.append('budget_'+key)
    nominal=cost.get('seconds_nominal')
    if not finite(nominal) or nominal>limits[3]:failures.append('nominal_seconds')
    if not finite(cost.get('seconds')):failures.append('seconds_missing')
    if kind!='prepare':
        phasecaps=(('MOVE',512),('SPLIT',512),('MOVE',512),('MERGE',240),('REFIT',16),('MOVE',256)) if kind=='TEACHER' else (('RESOURCE_MERGE',64),('INSERT',256),('REFIT',8),('SPLIT',64),('MOVE',120))
        phases=cost.get('phases',[])
        if len(phases)>len(phasecaps):failures.append('phase_count')
        for phase,(op,cap) in zip(phases,phasecaps):
            if phase.get('operator')!=op or not finite(phase.get('proposals')) or phase['proposals']>cap:failures.append('phase_budget_or_order')
        if sum(p.get('proposals',0) for p in phases)!=cost.get('proposals'):failures.append('phase_proposal_sum')
    return {'valid':not failures,'failures':failures,'nominal_overrun_allowed_for_terminal_audit':True}

def audit_cost(cost,method):
    failures=[]
    if not isinstance(cost,dict):return {'valid':False,'failures':['publication_cost_missing'],'seconds':None}
    comp=cost.get('components',{});keys=('input','prepare','teacher','proposal','decoder','wrapper')
    if not all(finite(comp.get(k)) for k in keys):failures.append('cost_components_unknown')
    elif not finite(cost.get('complete_method_seconds')) or not math.isclose(sum(comp[k] for k in keys),cost['complete_method_seconds'],rel_tol=1e-12,abs_tol=1e-9):failures.append('cost_sum')
    if cost.get('unknown_algorithm_spend') is not False:failures.append('unknown_spend')
    if method in ('RULE','RANDOM') and comp.get('teacher')!=0:failures.append('forbidden_teacher_cost')
    if method.startswith('TEACHER_') and comp.get('teacher')!=cost.get('teacher_once_seconds'):failures.append('actual_teacher_cost_missing')
    return {'valid':not failures,'failures':failures,'seconds':cost.get('complete_method_seconds')}

def audit_case(scene,prepared,raw_rows,costs,expected,source_sha256):
    failures=[];n=len(scene['demands']);expected={k:expected[k] for k in IDENTITY_FIELDS}
    if scene_sha(scene)!=expected['input_sha256']:failures.append('scene_input_SHA')
    if expected['role']!='development':failures.append('input_role')
    prep_identity=prepared.get('identity',{})
    if prep_identity!=expected or prepared.get('source_sha256')!=source_sha256:failures.append('prepared_identity_source')
    pc=prepared.get('ledger');pl=audit_ledger(pc,n,'prepare')
    if not pl['valid']:failures.extend(pl['failures'])
    witness=None;wenv=None;ncap=prepared.get('Ncap',0);cap=prepared.get('Omega_cap',0)
    if prepared.get('status')=='available':
        witness=audit_layout(scene,prepared.get('raw_w',{}),200,float('inf'),False,False)
        wenv=audit_layout(scene,prepared.get('Wenv',{}),ncap,cap,False)
        if not witness['valid'] or not wenv['valid']:failures.append('invalid_W_or_Wenv')
        elif ncap!=min(200,n,math.ceil(1.25*witness['metrics']['N'])) or abs(cap-witness['metrics']['actual_Omega'])>1e-12:failures.append('common_caps')
    methods={}
    for method in ('TEACHER',*METHODS):
        raw=raw_rows.get(method);cost=costs.get(method)
        if raw is None:methods[method]={'status':'missing','valid':False,'published':False,'identity_closed':False,'budget_closed':False,'cost_closed':False};continue
        errors=[];r=raw.get('result',{});status=r.get('status');published=status in ('available','success','fallback')
        identity_closed=all(raw.get(k)==v for k,v in expected.items()) and raw.get('method')==method
        if not identity_closed:errors.append('raw_identity_or_method')
        if raw.get('source_manifest_sha256',source_sha256)!=source_sha256:errors.append('raw_source')
        if status not in (('available','unavailable') if method=='TEACHER' else ('success','fallback','unavailable','unpublished')):errors.append('invalid_status_tag')
        if not published and any(k in r for k in ('layout','metrics','anchor_aircraft_ids')):errors.append('unavailable_has_layout')
        if published:
            if any(r.get(k)!=v for k,v in expected.items()):errors.append('published_identity')
            if method=='TEACHER' and r.get('source_sha256')!=source_sha256:errors.append('teacher_source')
        la=audit_layout(scene,r.get('layout',{}),ncap,cap) if published else None
        if la and not la['valid']:errors.extend(la['failures'])
        quality=la['metrics'] if la else None
        if quality:
            for k,rawkey in (('N','N'),('maxload','maxload'),('top10_mean','top10_mean'),('CV','CV'),('actual_Omega','solid_angle_sum_sr')):
                x,y=quality[k],r.get('metrics',{}).get(rawkey)
                if x is None:
                    if k=='CV' and y is not None:errors.append('zero_quality_not_undefined')
                elif not finite(y) or not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9 if k=='CV' else 1e-6 if k in ('maxload','top10_mean') else 1e-12):errors.append('producer_metric_'+k)
        ledger=r.get('cost');no_call=raw.get('stage_guard',{}).get('status')=='not_invoked'
        budget={'valid':True,'failures':[]} if no_call else audit_ledger(ledger,n,'TEACHER' if method=='TEACHER' else 'decoder')
        if not budget['valid']:errors.extend(budget['failures'])
        ca=audit_cost(cost,method)
        if ca['valid']:
            if cost.get('case_id')!=expected['case_id'] or cost.get('method')!=method:ca['valid']=False;ca['failures'].append('cost_identity')
        if method!='TEACHER' and published and (raw.get('decoder_invoked') is not True or not raw.get('proposal_ref')):errors.append('published_decoder_proposal_evidence_missing')
        if method.startswith('TEACHER_') and r.get('reason')=='unavailable_teacher' and raw.get('decoder_invoked') is not False:errors.append('missing_teacher_decoder_invoked')
        guard=raw.get('stage_guard',{})
        if guard.get('unknown_algorithm_spend') is True:errors.append('unknown_stage_spend')
        if finite(guard.get('peak_parent_children_rss_bytes')) and guard['peak_parent_children_rss_bytes']>2147483648:errors.append('RSS_limit')
        if not no_call:
            if not finite(guard.get('peak_parent_children_rss_bytes')):errors.append('RSS_evidence_missing')
            if guard.get('status')!='completed' and published:errors.append('uncompleted_stage_published')
        if method in ('RULE','RANDOM') and raw.get('teacher_stage_ref'):errors.append('teacher_access_forbidden')
        if isinstance(cost,dict):
            actual=guard.get('actual_stage_wall_seconds')
            key='teacher_once_seconds' if method=='TEACHER' else 'proposal_decoder_wall_seconds'
            if not finite(actual) or not finite(cost.get(key)) or not math.isclose(actual,cost[key],rel_tol=1e-12,abs_tol=1e-9):errors.append('cost_stage_actual_mismatch')
            if method.startswith('TEACHER_'):
                teacher_actual=raw_rows.get('TEACHER',{}).get('stage_guard',{}).get('actual_stage_wall_seconds')
                if not finite(teacher_actual) or not finite(cost.get('teacher_once_seconds')) or not math.isclose(teacher_actual,cost['teacher_once_seconds'],rel_tol=1e-12,abs_tol=1e-9):errors.append('teacher_actual_cost_not_paid')
        projection=r.get('final_anchor_projection')
        if status=='success' and (not projection or projection.get('status')!='valid' or projection.get('proposals_added')!=0 or projection.get('refits_added')!=0):errors.append('projection_evidence')
        mechanisms=None
        if quality and wenv and wenv['valid']:
            ww=wenv['metrics']['widths'];widths=quality['widths']
            mechanisms={'N':quality['N'],'N_changed_vs_Wenv':quality['N']!=wenv['metrics']['N'],'widths_actual':widths,'heterogeneous_widths':max(widths)-min(widths)>1e-12,'width_distribution_changed_vs_Wenv':len(widths)!=len(ww) or not np.allclose(sorted(widths),sorted(ww),rtol=0,atol=1e-12),'capability_is_not_effect_evidence':True}
        methods[method]={'status':status,'published':published,'valid':bool(published and la and la['valid'] and not errors),'identity_closed':identity_closed and not any('source' in x or 'identity' in x for x in errors),'budget_closed':budget['valid'],'cost_closed':ca['valid'],'publication_seconds':ca['seconds'],'metrics':quality,'failures':errors,'cost_audit':ca,'mechanisms':mechanisms}
        failures.extend(method+':'+e for e in errors)
        if not ca['valid']:failures.extend(method+':'+e for e in ca['failures'])
    common=None
    for c in costs.values():
        if isinstance(c,dict) and finite(c.get('input_read_seconds')) and finite(c.get('common_preparation_once_seconds')):
            value=c['input_read_seconds']+c['common_preparation_once_seconds']
            if common is None:common=value
            elif not math.isclose(common,value,abs_tol=1e-9,rel_tol=1e-12):failures.append('common_cost_inconsistent')
    return {'case_id':expected['case_id'],'mother_id':expected['mother_id'],'profile':expected['profile'],'audit_closed':not failures and len(raw_rows)==5,'complete_slots':all(m in raw_rows for m in ('TEACHER',*METHODS)),'witness_valid':bool(witness and witness['valid'] and wenv and wenv['valid']),'common_input_preparation_seconds':common,'methods':methods,'failures':failures,'Ncap':ncap,'Omega_cap':cap}

def audit_proposal(scene,proposal,expected,prepared,teacher,method,source):
    failures=[];n=len(scene['demands'])
    try:
        if proposal.get('status')!='ready':
            if proposal.get('status')!='unavailable':raise ValueError('proposal_tag')
            if any(k in proposal for k in ('layout','slot_ids','anchor_aircraft_ids','scores_nxK_float32_finite','score_mask_nxK_bool','direction_offset_Kx3')):raise ValueError('unavailable_arrays')
            provenance=proposal.get('provenance',{})
            if any(provenance.get(k)!=expected[k] for k in IDENTITY_FIELDS) or provenance.get('source_sha256')!=source or provenance.get('source_method')!=method:raise ValueError('unavailable_proposal_provenance')
            return {'valid':True,'failures':[],'ready':False}
        if any(proposal.get(k)!=expected[k] for k in IDENTITY_FIELDS) or proposal.get('source_sha256')!=source or proposal.get('source_method')!=method:failures.append('proposal_identity')
        slots=proposal['slot_ids'];anchors=proposal['anchor_aircraft_ids'];ids=scene['aircraft_ids'];k=len(slots)
        scores=np.asarray(proposal['scores_nxK_float32_finite'],np.float32);mask=np.asarray(proposal['score_mask_nxK_bool']);offset=np.asarray(proposal['direction_offset_Kx3'],float)
        if len(set(slots))!=k or len(anchors)!=k or any(a not in ids for a in anchors) or scores.shape!=(n,k) or mask.shape!=(n,k) or mask.dtype!=np.bool_ or offset.shape!=(k,3) or not np.isfinite(scores).all() or not np.isfinite(offset).all():raise ValueError('proposal_shapes_finite_ID')
        def digest(*items):return hashlib.sha256(json.dumps([51001,*items],ensure_ascii=False,separators=(',',':')).encode('utf8')).digest()
        expected_scores=np.zeros((n,k),np.float32);expected_mask=np.ones((n,k),bool);expected_offset=np.zeros((k,3));rays,_=rays_visible(scene['positions'],scene.get('profile','europe_v1'));cid=expected['case_id']
        if method=='RULE':
            if anchors!=prepared['anchors'] or slots!=prepared['raw_w']['beam_ids']:failures.append('RULE_anchor_slots')
            expected_scores=(-np.sum((rays[:,None]-rays[[ids.index(a) for a in anchors]][None])**2,axis=-1)).astype(np.float32)
        elif method=='RANDOM':
            wanted=sorted(ids,key=lambda a:(digest(cid,'anchor',a),a))[:len(prepared['anchors'])]
            if anchors!=wanted or slots!=['random:'+a for a in wanted]:failures.append('RANDOM_fixed_anchor_hash')
            expected_scores=np.asarray([[int.from_bytes(digest(cid,'score',a,b)[:8],'big')/2**64 for b in slots] for a in ids],np.float32)
        else:
            if not teacher or teacher.get('status')!='available':raise ValueError('proposal_without_teacher')
            layout=teacher['layout'];owner=np.asarray(layout['assignment'],int)
            if anchors!=layout['anchor_aircraft_ids'] or slots!=layout['beam_ids']:failures.append('teacher_discrete_labels')
            expected_scores[np.arange(n),owner]=1.
            if method=='TEACHER_PERTURBED':
                selected=sorted(range(n),key=lambda i:(digest(cid,'mask',ids[i]),ids[i]))[:math.floor(.1*n)]
                expected_mask[selected,owner[selected]]=False
                for b,a in enumerate(anchors):
                    ray=rays[ids.index(a)];axis=np.eye(3)[int(np.argmin(np.abs(ray)))];u=axis-float(axis@ray)*ray;u/=np.linalg.norm(u);v=np.cross(ray,u)
                    expected_offset[b]=(u,v,-u,-v)[digest(cid,'direction',slots[b])[0]%4]
        if not np.array_equal(scores,expected_scores):failures.append('fixed_score_definition')
        if not np.array_equal(mask,expected_mask):failures.append('fixed_mask_definition')
        if not np.allclose(offset,expected_offset,rtol=0,atol=1e-12):failures.append('fixed_direction_definition')
        cost=proposal.get('proposal_cost_ledger',{})
        if cost.get('raypairs')!=(n*k if method=='RULE' else 0):failures.append('proposal_pair_accounting')
        if cost.get('hash_calls')!=(n+n*k if method=='RANDOM' else n+k if method=='TEACHER_PERTURBED' else 0):failures.append('proposal_hash_accounting')
        return {'valid':not failures,'failures':failures,'ready':True,'score_shape':[n,k]}
    except (ValueError,KeyError,TypeError,IndexError) as exc:return {'valid':False,'failures':['malformed_proposal:'+str(exc)]}

def load_scene(reader,ref):
    path=reader.path(ref['packed']);i=int(ref['case_index'])
    with np.load(path,allow_pickle=False) as z:
        offsets=z['offsets'];a,b=int(offsets[i]),int(offsets[i+1])
        scene={'positions':z['positions'][a:b].tolist(),'demands':z['demands'][a:b].tolist(),'aircraft_ids':[str(x) for x in z['aircraft_ids'][a:b]],'profile':'europe_v1'}
        identity={'case_id':str(z['case_ids'][i]),'mother_id':str(z['mother_ids'][i]),'profile':str(z['demand_profiles'][i])}
    return scene,identity

def audit_manifests(manifest_refs,reader,expected_case_ids,engineering_g0=False):
    cases=[];errors=[];seen=set()
    for ref in manifest_refs:
        manifest=reader.json(ref)
        if manifest.get('schema')!='R4B0ResultManifestV1':raise ValueError('manifest_schema')
        if manifest.get('MILP_calls')!=0 or manifest.get('training_updates')!=0:raise ValueError('forbidden_science')
        if manifest['design_ref']['sha256']!=CONTRACT_SHA:raise ValueError('signed_contract_SHA')
        design=reader.json(manifest['design_ref'])
        if manifest['identity_ref']['sha256']!=design['data']['identity_sha256']:raise ValueError('identity_contract_SHA')
        reader.json(manifest['job_ref']);reader.json(manifest['config_ref']);reader.json(manifest['qualification_ref'])
        source=manifest['source_manifest_ref']['sha256'];reader.json(manifest['source_manifest_ref']);pinned_identity=reader.json(manifest['identity_ref'])
        pinned_ids=pinned_identity['development']['case_ids']
        if list(expected_case_ids)!=[pinned_ids[i] for i in design['data']['case_indices']]:raise ValueError('prespecified_expected_ids_not_contract')
        for entry in manifest['cases']:
            cid=entry['case_id']
            if cid in seen:raise ValueError('duplicate_case_across_manifests')
            if entry.get('input_case_index') not in design['data']['case_indices'] or entry.get('selected_case_ordinal')!=design['data']['case_indices'].index(entry['input_case_index']):raise ValueError('unregistered_case_index_or_ordinal')
            seen.add(cid)
            try:
                scene,identity=load_scene(reader,entry['scene_ref'])
                if any(entry.get(k)!=v for k,v in identity.items()):raise ValueError('packed_identity_mismatch')
                raw={m:reader.json(r) for m,r in entry['raw_refs'].items()};cost={m:reader.json(r) for m,r in entry['publication_cost_refs'].items()}
                first=next(iter(raw.values()));prep=reader.json(first['prepared_stage_ref'])['result']
                # Independently pin every referenced stage/proposal and raw cost backlink.
                for m,row in raw.items():
                    if row.get('source_commit')!=manifest.get('source_commit'):raise ValueError('execution_source_commit')
                    if row.get('prepared_stage_ref')!=first['prepared_stage_ref']:raise ValueError('common_preparation_ref')
                    for key in ('teacher_stage_ref','proposal_ref'):
                        if row.get(key):reader.json(row[key])
                    if row.get('stage_guard',{}).get('output_ref'):reader.json(row['stage_guard']['output_ref'])
                    if cost[m].get('raw_ref')!=entry['raw_refs'][m]:raise ValueError('cost_raw_backlink')
                case_audit=audit_case(scene,prep,raw,cost,entry,source)
                for method,row in raw.items():
                    if method!='TEACHER' and row.get('proposal_ref'):
                        pa=audit_proposal(scene,reader.json(row['proposal_ref']),entry,prep,raw.get('TEACHER',{}).get('result'),method,source)
                        case_audit['methods'][method]['proposal_audit']=pa
                        if not pa['valid']:
                            case_audit['audit_closed']=False;case_audit['methods'][method]['valid']=False;case_audit['failures'].extend(method+':'+x for x in pa['failures'])
                cases.append(case_audit)
            except (ValueError,KeyError,OSError,TypeError) as exc:
                errors.append({'case_id':cid,'error':str(exc)});cases.append({'case_id':cid,'mother_id':entry.get('mother_id'),'audit_closed':False,'complete_slots':False,'methods':{},'failures':[str(exc)]})
    gates=evaluate_gates(cases,expected_case_ids,engineering_g0)
    return {'schema':'R4B0IndependentAuditV1','contract_sha256':CONTRACT_SHA,'algorithm_calls':0,'cases':cases,'errors':errors,'gates':gates,'raw_frozen_refs':reader.refs,'numeric_entry':{'teacher_count':gates['teacher_count'],'exact_count':gates['exact_count'],'perturbed_count':gates['perturbed_count'],'quality_count':gates['quality_count'],'restricted_complete_means':gates['restricted_complete_means']}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);args=p.parse_args();request=json.loads(Path(args.request).read_text(encoding='utf-8-sig'))
    result=audit_manifests(request['manifest_refs'],Reader(request.get('path_map')),request['expected_case_ids'],request.get('engineering_g0',False));output=Path(args.output)
    if output.exists():raise ValueError('immutable_audit_output_exists')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
if __name__=='__main__':main()
