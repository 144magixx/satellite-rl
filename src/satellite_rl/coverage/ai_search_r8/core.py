"""One fixed pool, exact integer load acceptance, arm-specific tie ordering only."""
import base64,hashlib,heapq,json,math,os,time
from pathlib import Path
import numpy as np
from .ledger import Ledger,Limit
from .geometry import FrozenGeometry,omega

DESIGN_SHA='e0e24cf00e873965ee585d77f5f1e38c7455196a19a5357de7c5732fcfa08bb4'
CHECKPOINT_SHA='10ff42e745ee0b1639cab9781d53ca43f5baa61b7e9e75aa12e49eec15bc9a40'
IDENTITY_FIELDS=('case_id','mother_id','profile','input_sha256','identity_sha256','role','shared_case_content_sha256','source_asset_SHA')
REF_FIELDS=('shared_case','original_scene','original_prepared_Wenv','AI_envelope','actual_model_outputs_NPZ','R7_RULE_final','R7_T_final')
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(array):
    a=np.ascontiguousarray(array);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def random_key(case_id,candidate):return hashlib.sha256(canonical([51001,case_id,'r8_transfer_order',*candidate])).hexdigest()
def integer_key(counts):
    c=sorted(map(int,counts),reverse=True);return (c[0],sum(c[:(len(c)+9)//10]),sum(x*x for x in c))
def prospective_key(counts,source,destination,current_key,ledger):
    """Exactly one <=200-count scan; top-k heap, squared-sum delta exact."""
    length=len(counts)
    if not 1<=length<=200:raise ValueError('at most200 integer counts per rank')
    ledger.charge('integer_scan_items',length);ledger.charge('integer_heap_input_items',length)
    top=(length+9)//10;heap=[];top_sum=maximum=0
    for beam,count in enumerate(counts):
        value=int(count)+(beam==destination)-(beam==source);maximum=max(maximum,value)
        if len(heap)<top:heapq.heappush(heap,value);top_sum+=value
        elif value>heap[0]:top_sum+=value-heapq.heapreplace(heap,value)
    squared=current_key[2]-2*int(counts[source])+2*int(counts[destination])+2
    return (maximum,top_sum,squared)
def softmax_owner(logits,ledger):
    a=np.asarray(logits)
    if a.dtype!=np.float32 or a.shape!=(ledger.n,200) or not np.isfinite(a).all():raise ValueError('original finite full200 float32 owner cache required')
    ledger.charge('owner_cache_conversion_elements',a.size);a=a.astype(np.float64)
    ledger.charge('softmax_scan_items',a.size);a-=a.max(axis=1,keepdims=True)
    ledger.charge('softmax_exp',a.size);a=np.exp(a)
    ledger.charge('softmax_reduce_items',a.size);a/=a.sum(axis=1,keepdims=True)
    return a
def affinity(probabilities,member_sums,counts,aircraft,source,destination,ledger):
    if counts[source]<2 or counts[destination]<1:raise ValueError('dynamic source exclude-self/destination nonempty')
    ledger.charge('affinity_multiply_add_items',400);ledger.charge('affinity_mean_elements',400);ledger.charge('affinity_exclude_self_elements',200)
    p=probabilities[aircraft]
    value=float(np.dot(p,member_sums[destination]/counts[destination])-np.dot(p,(member_sums[source]-p)/(counts[source]-1)))
    if not math.isfinite(value):raise ValueError('finite float64 affinity')
    return value
def encode(array):
    a=np.ascontiguousarray(array,dtype=np.float64);return dict(shape=list(a.shape),dtype='float64',bytes_b64=base64.b64encode(a.tobytes()).decode('ascii'))
def decode(record):return np.frombuffer(base64.b64decode(record['bytes_b64']),dtype=np.float64).reshape(record['shape']).copy()
def atomic(path,value):
    temporary=path.with_name(path.name+'.tmp');raw=canonical(value)
    with temporary.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
def layout_record(centers,widths,owners,beam_ids):return dict(centers=np.asarray(centers).tolist(),widths_deg=np.asarray(widths).tolist(),assignment=np.asarray(owners).tolist(),beam_ids=list(beam_ids))
def validate_inputs(prepared,arm,provenance):
    scene=prepared.scene;identity=prepared.identity;n=len(scene.demands);w=prepared.Wenv
    if arm not in ('G','A','R') or prepared.status!='available' or w is None or n<1:raise ValueError('available original Wenv and one fixed R8 arm')
    if set(identity)!=set(IDENTITY_FIELDS) or identity['role']!='development' or identity['profile']!='uniform21600000' or identity['input_sha256']!=scene.input_sha256:
        raise ValueError('original complete uniform development identity')
    if scene.profile!='europe_v1' or np.asarray(scene.demands).dtype!=np.float64 or not np.all(scene.demands==21600000):raise ValueError('original geometry and exact uniform bps representation')
    if len(set(scene.aircraft_ids))!=n or len(w.assignment)!=n or len(w.beam_ids)!=len(w.widths) or len(prepared.anchors)!=len(w.widths):raise ValueError('original full plane/order/beam/anchor lengths')
    if len(set(w.beam_ids))!=len(w.beam_ids) or any(a not in scene.aircraft_ids for a in prepared.anchors):raise ValueError('original unique stable beams and aircraft anchor IDs')
    if not 1<=len(w.widths)<=min(200,n,prepared.Ncap):raise ValueError('actual fixed nonempty N cap')
    if provenance.get('design_sha256')!=DESIGN_SHA or provenance.get('checkpoint_sha256')!=CHECKPOINT_SHA or provenance.get('case_identity')!=identity or provenance.get('actual_arm')!=arm:
        raise ValueError('original root design/model/identity/actual outer arm')
    refs=provenance.get('source_refs',{})
    if set(refs)!=set(REF_FIELDS) or any(not isinstance(refs[k].get('path'),str) or not isinstance(refs[k].get('sha256'),str) or len(refs[k]['sha256'])!=64 for k in REF_FIELDS):raise ValueError('closed root-qualified per-case original refs')
    if not provenance.get('source_epoch') or not provenance.get('adapter_source_SHA'):raise ValueError('actual new source epoch required')
    return scene,w
def fixed_pool(ids,beam_ids,owners,distances,counts,ledger):
    n=len(ids);pool=[]
    for i in range(n):
        source=int(owners[i]);ledger.charge('destination_sort_items',len(beam_ids)-1)
        dest=sorted((b for b in range(len(beam_ids)) if b!=source),key=lambda b:(float(distances[i,b]),beam_ids[b]))[:4]
        if counts[source]<2:continue
        for b in dest:
            if counts[b]<counts[source]-1:pool.append((ids[i],beam_ids[source],beam_ids[b]))
    if len(pool)>4*n or len(set(pool))!=len(pool):raise ValueError('fixed original tuple pool bounds')
    ledger.charge('pool_generated',len(pool));ledger.charge('pool_sort_items',len(pool))
    ii={aid:i for i,aid in enumerate(ids)};bb={bid:b for b,bid in enumerate(beam_ids)}
    pool.sort(key=lambda c:(-int(counts[bb[c[1]]]),int(counts[bb[c[2]]]),float(distances[ii[c[0]],bb[c[2]]]),*c))
    return [list(c) for c in pool[:512]]

def local_transfer(prepared,owner_logits=None,*,arm,provenance,state_path,stop_callback,resume_admission=None,geometry_backend=None,clock=time.monotonic):
    """Caller pins refs, owns single worker/guard and charges complete return/publication IO.

    Partial state is atomic and cannot resume a pending extraction. Root must
    verify charged active-clock tail; global deadline admission belongs to ops.
    """
    entered=clock();scene,w=validate_inputs(prepared,arm,provenance);path=Path(state_path);path.parent.mkdir(parents=True,exist_ok=True)
    ids=tuple(scene.aircraft_ids);beam_ids=tuple(w.beam_ids);n=len(ids);B=len(beam_ids);geometry=geometry_backend or FrozenGeometry(scene)
    if path.exists():
        prior_sha=digest(path);state=json.loads(path.read_bytes())
        if not resume_admission or resume_admission.get('root_accepted')is not True or resume_admission.get('prior_state_sha256')!=prior_sha or resume_admission.get('input_provenance')!=provenance:raise ValueError('root explicit same-source state/clock recovery required')
        if state.get('pending')is not None:raise ValueError('unknown extracted tuple preserved; no automatic replay')
        if state['provenance']!=provenance or state['arm']!=arm or state['status']!='paused':raise ValueError('same original partial method only; completed immutable')
        verified=resume_admission['verified_charged_active_seconds']
        if not math.isfinite(verified) or verified<state['ledger']['active_seconds']:raise ValueError('never reset previous quotas or active clock')
        ledger=Ledger(n,clock,state['ledger'],verified);static_ref=state['static_ref']
        if digest(static_ref['path'])!=static_ref['sha256']:raise ValueError('immutable static arrays changed')
        with np.load(static_ref['path'],allow_pickle=False) as z:distances=z['distances'].copy();probabilities=z['probabilities'].copy() if arm=='A' else None
        owners=np.asarray(state['owners'],dtype=np.int64);widths=np.asarray(state['widths'],dtype=np.float64);member_sums=decode(state['member_sums']) if arm=='A' else None
        if owners.shape!=(n,) or (owners<0).any() or (owners>=B).any() or widths.shape!=(B,) or distances.shape!=(n,B) or not np.isfinite(distances).all():raise ValueError('same full valid partial state shapes')
        if arm=='A' and (probabilities.shape!=(n,200) or member_sums.shape!=(B,200) or not np.isfinite(probabilities).all() or not np.isfinite(member_sums).all()):raise ValueError('fixed exact saved probability/sum arrays')
    else:
        if resume_admission is not None:raise ValueError('resume requires existing original state')
        ledger=Ledger(n,clock);ledger.started=entered
        owners=np.array(w.assignment,dtype=np.int64,copy=True);widths=np.array(w.widths,dtype=np.float64,copy=True)
        if owners.min()<0 or owners.max()>=B:raise ValueError('Wenv full unique ownership')
        counts=np.bincount(owners,minlength=B)
        if (counts<1).any():raise ValueError('Wenv nonempty beams')
        initial_audit=geometry.audit(w.centers,widths,owners,beam_ids,ledger)
        if not initial_audit['valid'] or omega(widths)>prepared.Omega_cap+1e-12:raise ValueError('independent original Wenv audit')
        distances=geometry.distances(w.centers,ledger)
        if distances.shape!=(n,B) or distances.dtype!=np.float64 or not np.isfinite(distances).all():raise ValueError('fixed float64 original-anchor angle table')
        probabilities=softmax_owner(owner_logits,ledger) if arm=='A' else None
        if arm=='A':
            ledger.charge('hash_calls',1);ledger.charge('owner_array_hash_elements',200*n)
            if array_sha(owner_logits)!=provenance.get('owner_array_sha256'):raise ValueError('actual saved full owner array SHA differs')
        member_sums=np.zeros((B,200),dtype=np.float64) if arm=='A' else None
        if arm=='A':ledger.charge('member_sum_allocation_elements',200*B);ledger.charge('initial_member_sum_elements',200*n);np.add.at(member_sums,owners,probabilities)
        pool=fixed_pool(ids,beam_ids,owners,distances,counts,ledger)
        static=path.with_name('static.npz')
        if static.exists():raise ValueError('unindexed initial static asset; preserve original partial instead of regeneration')
        with static.open('xb') as stream:np.savez_compressed(stream,distances=distances,**({'probabilities':probabilities} if arm=='A' else {}));stream.flush();os.fsync(stream.fileno())
        ledger.charge('hash_calls',1);static_ref=dict(path=str(static),sha256=digest(static))
        state=dict(schema='R8LocalStateV1',arm=arm,provenance=provenance,initial_layout=w.to_record(),initial_key=list(integer_key(counts)),initial_audit=initial_audit,
            pool=pool,remaining=list(range(len(pool))),cursor=0,trace=[],pending=None,static_ref=static_ref,status='running')
    ii={aid:i for i,aid in enumerate(ids)};bb={bid:b for b,bid in enumerate(beam_ids)}
    def persist():
        began=clock();state.update(owners=owners.tolist(),widths=widths.tolist(),member_sums=encode(member_sums) if arm=='A' else None,ledger=ledger.record());atomic(path,state)
        ledger.data['persistence_seconds']=ledger.data.get('persistence_seconds',0.)+clock()-began
        ledger.data['state_publications']=ledger.data.get('state_publications',0)+1
    persist();reason=None;paused=False
    try:
        while state['remaining'] and ledger.data.get('candidate_extractions',0)<128:
            ledger.check()
            if stop_callback():paused=True;reason='caller_pause';break
            ledger.charge('current_owner_count_scan_items',n);ledger.charge('current_count_sort_items',B)
            counts=np.bincount(owners,minlength=B);current=integer_key(counts);rank=[]
            for position in state['remaining']:
                ledger.charge('scalar_rank_items',1)
                c=state['pool'][position];i,s,d=ii[c[0]],bb[c[1]],bb[c[2]]
                key=prospective_key(counts,s,d,current,ledger)
                invalid='stale_source' if owners[i]!=s else 'singleton_source' if counts[s]<2 else 'integer_not_strict_improvement' if key>=current else 'initial_top10_guard' if key[1]>state['initial_key'][1] else 'initial_sumsq_guard' if key[2]>state['initial_key'][2] else None
                if invalid:order=(1,*c)
                else:
                    g=float(distances[i,d]-distances[i,s])
                    secondary=(g,) if arm=='G' else (-affinity(probabilities,member_sums,counts,i,s,d,ledger),g) if arm=='A' else (random_key(prepared.identity['case_id'],c),g)
                    if arm=='R':ledger.charge('hash_calls',1)
                    order=(0,*key,*secondary,*c)
                rank.append((order,position,invalid,key))
            ledger.charge('rank_sort_items',len(rank));rank.sort(key=lambda x:x[0]);_,position,invalid,key=rank[0];c=state['pool'][position];i,s,d=ii[c[0]],bb[c[1]],bb[c[2]]
            ledger.charge('candidate_extractions',1);state['remaining'].remove(position);state['cursor']+=1;state['pending']=dict(pool_position=position,candidate=c);persist()
            rejection=invalid
            if not invalid:
                ledger.charge('geometric_attempts',1);candidate=owners.copy();candidate[i]=d
                newwidths=geometry.widths_for_move(candidate,s,d,widths,distances,ledger)
                if newwidths is None:rejection='envelope_width'
                elif omega(newwidths)>prepared.Omega_cap+1e-12:rejection='Omega_cap'
                elif not geometry.audit(w.centers,newwidths,candidate,beam_ids,ledger)['valid']:rejection='independent_candidate_geometry'
                else:
                    if arm=='A':ledger.charge('accepted_member_sum_update_elements',400);member_sums[s]-=probabilities[i];member_sums[d]+=probabilities[i]
                    previous_key=list(current);owners=candidate;widths=newwidths
                    state['trace'].append(dict(pool_position=position,candidate=c,status='accepted',old_key=previous_key,new_key=list(key),source_width=float(widths[s]),destination_width=float(widths[d])))
            if rejection:state['trace'].append(dict(pool_position=position,candidate=c,status='rejected',reason=rejection,current_key=list(current),prospective_key=list(key)))
            state['pending']=None;persist()
        if not reason:reason='candidate_quota' if ledger.data.get('candidate_extractions',0)>=128 else 'pool_exhausted'
    except Limit as error:
        reason=str(error)
        # A quota/time interruption within an uncommitted tuple cannot replay
        # its already charged extraction; retain last legal owners and reject it.
        if state.get('pending')is not None:
            state['trace'].append({**state['pending'],'status':'rejected','reason':reason});state['pending']=None
    ledger.terminal=True;final_audit=geometry.audit(w.centers,widths,owners,beam_ids,ledger)
    if not final_audit['valid'] or omega(widths)>prepared.Omega_cap+1e-12:raise ValueError('independent final point audit failed; unpublished')
    counts=np.bincount(owners,minlength=B);final_key=integer_key(counts)
    if (counts<1).any() or counts.sum()!=n or final_key[1]>state['initial_key'][1] or final_key[2]>state['initial_key'][2]:raise ValueError('whole-plane exact conservation and initial integer guards')
    state.update(status='paused' if paused else 'completed',truncation_reason=reason,final_audit=final_audit);persist()
    final=layout_record(w.centers,widths,owners,beam_ids);proposal=None
    if not paused:
        ordered=sorted(range(B),key=lambda b:beam_ids[b]);inverse={b:j for j,b in enumerate(ordered)};scores=np.zeros((n,B),dtype=np.float32);scores[np.arange(n),[inverse[int(b)] for b in owners]]=1
        ledger.charge('proposal_score_elements',n*B);ledger.charge('proposal_mask_elements',n*B)
        proposal=dict(schema='ProposalV2',status='ready',case_identity=dict(prepared.identity),input_sha=scene.input_sha256,role_sha=prepared.identity['identity_sha256'],
            provenance=dict(case_identity=dict(prepared.identity),input_sha=scene.input_sha256,role_sha=prepared.identity['identity_sha256'],prepared_source_sha256=prepared.source_sha256,
                producer_sha256=provenance['adapter_source_SHA'],source_method='EXACT',actual_outer_arm=arm,EXACT_compatibility_only_not_teacher=True,R8_input_provenance=provenance),
            slot_ids=[beam_ids[b] for b in ordered],anchor_aircraft_ids=[prepared.anchors[b] for b in ordered],scores_n_by_K=scores.tolist(),mask_n_by_K=np.ones((n,B),bool).tolist(),
            construction_cost=dict(seconds=ledger.elapsed(),raypairs=ledger.data.get('raypairs',0),hash_calls=ledger.data.get('hash_calls',0),sort_input_count=ledger.data.get('rank_sort_items',0),scan_input_count=ledger.data.get('integer_scan_items',0)))
    ledger.charge('hash_calls',1);state_ref=dict(path=str(path),sha256=digest(path))
    output=dict(status=state['status'],proposal=proposal,local_result=dict(initial_layout=state['initial_layout'],final_layout=final,anchor_aircraft_ids=list(prepared.anchors),
        initial_integer_key=state['initial_key'],final_integer_key=list(final_key),beam_counts=counts.tolist(),beam_request_bps_int64=(counts*21600000).tolist(),total_request_bps_int64=n*21600000),
        ledger=ledger.record(),trace=state['trace'],state_ref=state_ref,static_ref=static_ref,
        D5_compatibility_conversion=dict(schema='R8CurrentLegalOwnerOneHotCompatibilityV1',actual_outer_arm=arm,inner_label='EXACT',not_a_teacher_proposer=True,local_state_ref=state_ref),
        caller_must_charge_return_and_publication_IO=True,complete_exit_clock_requires_external_receipt=True)
    output['ledger']=ledger.record()
    return output
