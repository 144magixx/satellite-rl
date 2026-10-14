"""R10 v2 core. Pure caller-owned scene inputs, no file/network side effects."""
import copy
import json
import math
import time
import numpy as np
from .flow import Ledger, ResourceUnknown, solve
from .costs import matrix as cost_matrix
from ..ai_search_r9.core import (REQUEST, ACTIONS, IDENTITY_KEYS, require,
    canonical, digest, angles, audit, quality, omega, protected, objective)
from ..ai_search_r9.flow import Ledger as R9Ledger
from .local_flow import minimum_capacity
from .anchors import choose_anchors

DESIGN_SHA = '47cb6befc27519e18c8da8cf07513299d84c04b109db9aae6982d273febbb498'
RECIPE = '4a82bf0889c35e62d27d695b35c9a85e86491409'
METHODS = ('Gd', 'Gr', 'Gw', 'S2', 'SH-C')
INDICES = (0,6,12,18,24,36,42,48,54,60,72,78)


def physical_state(scene, layout, case_id, ledger=None):
    ids = layout['beam_ids']
    members = [[] for _ in ids]
    for aid, owner in zip(scene.aircraft_ids, layout['assignment']):
        members[int(owner)].append(aid)
    order=(sorted(range(len(ids)),key=lambda j:ids[j]) if ledger is None else
           ledger.ordered(list(range(len(ids))),lambda j:ids[j],'physical_state_beam_order'))
    ordered_members=[sorted(m) if ledger is None else ledger.ordered(m,lambda a:a,'physical_state_member_ID_order') for m in members]
    beams = [[ids[j], layout['anchor_aircraft_ids'][j],
              float(layout['centers'][j][0]).hex(),
              float(layout['centers'][j][1]).hex(),
              float(layout['widths_deg'][j]).hex(), ordered_members[j]] for j in order]
    return digest(dict(case_id=case_id, beams=beams))


def _scene_identity(scene, identity):
    require(all(k in identity for k in IDENTITY_KEYS) and
            all(type(identity[k]) is str and identity[k] for k in IDENTITY_KEYS),
            'full original identity strings')
    require(identity['role']=='train' and identity['profile']=='uniform21600000' and
            scene.input_sha256 == identity['input_sha256'], 'original training diagnosis identity')
    n=len(scene.aircraft_ids)
    require(n>0 and len(set(scene.aircraft_ids))==n and
            all(type(a) is str and a for a in scene.aircraft_ids), 'all stable original aircraft IDs')
    require(np.asarray(scene.positions).shape==(n,2) and
            np.isfinite(scene.positions).all() and
            np.asarray(scene.rays).shape==(n,3) and np.isfinite(scene.rays).all() and
            np.asarray(scene.demands).shape==(n,) and
            np.all(np.asarray(scene.demands)==REQUEST), 'all original positions/rays/fixed requests')


def static_digest(scene):
    return digest(dict(aircraft_ids=list(scene.aircraft_ids),positions=np.asarray(scene.positions).tolist(),
        rays=np.asarray(scene.rays).tolist(),demands=np.asarray(scene.demands).tolist(),input_sha256=scene.input_sha256))


def prepare_common(scene, origin, *, original_index=None, stop=None):
    """Qualify an already audited saved R9 best terminal, never regenerate it.

    origin contains identity, layout, metrics, Ncap, Omega_cap, layout_SHA and
    source_refs. Caller pins and verifies those source bytes before this call.
    Returned common is JSON serializable. No solver, model or R9 producer runs.
    """
    began=time.monotonic()
    require(original_index is None or original_index in INDICES, 'original twelve train mothers')
    _scene_identity(scene,origin['identity'])
    require(bool(origin.get('source_refs')), 'explicit original best-terminal source references')
    require(origin['layout_SHA']==digest(origin['layout']), 'saved best-terminal layout SHA')
    n=len(scene.aircraft_ids)
    ledger=Ledger(n,stop=stop)
    layout=copy.deepcopy(origin['layout'])
    metrics=audit(scene,layout,origin['Ncap'],origin['Omega_cap'],ledger)
    require(metrics==origin['metrics'], 'recomputed saved B metrics exactly equal')
    owners=np.asarray(layout['assignment'],np.int64)
    b=len(layout['beam_ids'])
    counts=np.bincount(owners,minlength=b)
    maximum=int(counts.max())
    ledger.scan('common_peak_source_bincount',n+b)
    pindex={a:i for i,a in enumerate(scene.aircraft_ids)}
    original_ids=set(layout['anchor_aircraft_ids'])
    used_coords={tuple(map(float,c)) for c in layout['centers']}
    ledger.scan('common_original_center_ID_sets',n+b)
    sources=[]
    unavailable=[]
    for j in range(b):
        if int(counts[j]) != maximum:
            continue
        ledger.scan('common_peak_members',n)
        members=[i for i in range(n) if owners[i]==j]
        members=ledger.ordered(members,lambda i:scene.aircraft_ids[i],'common_peak_member_ID_order')
        anchor_ray=scene.rays[pindex[layout['anchor_aircraft_ids'][j]]]
        distance=angles(scene.rays[members],anchor_ray,ledger)
        eligible=[]
        ledger.scan('common_peak_member_exclusion',len(members))
        for ii,i in enumerate(members):
            aid=scene.aircraft_ids[i]
            coord=tuple(map(float,scene.positions[i]))
            if aid not in original_ids and coord not in used_coords:
                eligible.append((float(distance[ii]),aid,i,coord))
        eligible=ledger.ordered(eligible,lambda x:(-x[0],x[1]),'common_peak_radial_order')
        if eligible:
            sources.append((eligible[0][0],layout['beam_ids'][j],j,eligible))
        else:
            unavailable.append(dict(source_beam_id=layout['beam_ids'][j],reason='no_eligible_member'))
    sources=ledger.ordered(sources,lambda x:(-x[0],x[1]),'common_peak_source_order')
    prefix=[]
    used_ids=set(original_ids)
    for _,bid,j,candidates in sources:
        chosen=None
        for distance,aid,i,coord in candidates:
            ledger.scan('common_prefix_collision_lookup',1)
            if aid not in used_ids and coord not in used_coords:
                chosen=(distance,aid,i,coord)
                break
        if chosen is None:
            unavailable.append(dict(source_beam_id=bid,reason='prefix_ID_or_coordinate_collision'))
            continue
        distance,aid,i,coord=chosen
        newid=json.dumps([origin['identity']['case_id'],'R10_added',bid,aid,len(prefix)],
                         ensure_ascii=False,separators=(',',':'))
        prefix.append(dict(beam_id=newid,anchor_aircraft_id=aid,center=list(coord),
                           source_beam_id=bid,radial_distance_deg=distance))
        used_ids.add(aid)
        used_coords.add(coord)
    allanchors=list(layout['anchor_aircraft_ids'])+[x['anchor_aircraft_id'] for x in prefix]
    ledger.scan('common_all_anchor_index_lookup',len(allanchors))
    arays=scene.rays[[pindex[a] for a in allanchors]]
    distances=angles(scene.rays[:,None,:],arays[None,:,:],ledger)
    ledger.pairs(n*len(allanchors))
    chord=np.sum((scene.rays[:,None,:]-arays[None,:,:])**2,axis=-1)
    ledger.scan('common_full_adjacency_entries',n*len(allanchors))
    eligible=(chord<=4*np.sin(np.deg2rad(.5)/2)**2+1e-14)
    slots=[]
    seen={}
    for target in (b,min(origin['Ncap'],b+1),min(origin['Ncap'],b+int(np.count_nonzero(counts==maximum)))):
        slot=dict(position=len(slots),target_N=target)
        if target in seen:
            slot.update(status='duplicate_N',duplicate_of=seen[target])
        else:
            seen[target]=len(slots)
            if maximum-1<1 or target>min(n,origin['Ncap']) or n>(maximum-1)*target:
                slot['status']='count_infeasible'
            elif target-b>len(prefix):
                slot['status']='known_structure_unavailable'
            else:
                slot['status']='available'
        slots.append(slot)
    common=dict(schema='R10CommonV2',design_sha256=DESIGN_SHA,original_index=original_index,origin_SHA=origin['layout_SHA'],static_SHA=static_digest(scene),
        identity=copy.deepcopy(origin['identity']),origin=copy.deepcopy(origin),
        n=n,q=maximum-1,M=maximum,maximum_source_count=int(np.count_nonzero(counts==maximum)),
        added_center_prefix=prefix,unavailable_peak_sources=unavailable,N_slots=slots,
        all_anchor_ids=allanchors,distances_deg=distances.tolist(),adjacency=eligible.tolist(),
        common_ledger=ledger.record(),observed_common_seconds=time.monotonic()-began,
        new_RULE_D5_T_W_model_calls=0)
    common['common_SHA']=digest(common)
    return common


def _common_valid(scene,common):
    require(common['design_sha256']==DESIGN_SHA, 'unique v2 design')
    require(digest({k:v for k,v in common.items() if k!='common_SHA'})==common['common_SHA'], 'pinned common bytes')
    _scene_identity(scene,common['identity'])
    require(common['n']==len(scene.aircraft_ids), 'full original denominator')
    require(common['static_SHA']==static_digest(scene), 'same full original static arrays')


def _layout_global(scene,common,slot,owners,ledger):
    base=common['origin']['layout']
    extra=slot['target_N']-len(base['beam_ids'])
    prefix=common['added_center_prefix'][:extra]
    layout=dict(beam_ids=list(base['beam_ids'])+[x['beam_id'] for x in prefix],
        anchor_aircraft_ids=list(base['anchor_aircraft_ids'])+[x['anchor_aircraft_id'] for x in prefix],
        centers=copy.deepcopy(base['centers'])+[x['center'] for x in prefix],assignment=list(owners))
    distances=np.asarray(common['distances_deg'],np.float64)
    owners=np.asarray(owners,np.int64)
    widths=[]
    for j in range(slot['target_N']):
        ledger.scan('global_final_member_masks',common['n'])
        members=np.flatnonzero(owners==j)
        require(len(members)>0,'flow lower one')
        ledger.scan('global_final_distance_max',len(members))
        required=2*float(distances[members,j].max())
        width=max(.3,float(np.nextafter(required,np.inf)))
        if width>1.:
            layout.update(observed_widths_deg=widths,overwidth_beam_index=j,
                          overwidth_required_fullwidth_deg=required,overwidth_outward_fullwidth_deg=width)
            return layout,'envelope_over1_conservative_kernel'
        widths.append(width)
    layout['widths_deg']=widths
    if omega(widths)>common['origin']['Omega_cap']+1e-12:
        return layout,'whole_Omega_cap'
    return layout,None


def _global(scene,common,method,ledger,observations=None):
    origin=common['origin']
    results=[]
    best=dict(terminal=copy.deepcopy(origin['layout']),metrics=copy.deepcopy(origin['metrics']),position=-1,status='incumbent')
    if observations is not None:observations.update(results=results,selected=best)
    b=len(origin['layout']['beam_ids'])
    ownerids=[origin['layout']['beam_ids'][i] for i in origin['layout']['assignment']]
    for slot in common['N_slots']:
        record=copy.deepcopy(slot)
        results.append(record)
        ledger.tick('global_structure_slots')
        if slot['status']!='available':
            ledger.tick('global_'+slot['status'])
            continue
        record['status']='entered_unknown'
        target=slot['target_N']
        prefix=common['added_center_prefix'][:target-b]
        bids=list(origin['layout']['beam_ids'])+[x['beam_id'] for x in prefix]
        refs=list(origin['layout']['widths_deg'])+[.3]*len(prefix)
        ledger.scan('global_structure_slice',common['n']*target)
        distances=[row[:target] for row in common['distances_deg']]
        adjacency=[row[:target] for row in common['adjacency']]
        costs=cost_matrix(distances,adjacency,bids,ownerids,refs,method,ledger)
        flow=solve(tuple(scene.aircraft_ids),bids,adjacency,costs,common['q'],ledger)
        record['flow']=flow
        if not flow['feasible']:
            record['status']='graph_infeasible'
            continue
        layout,reason=_layout_global(scene,common,slot,flow['assignment'],ledger)
        record['raw_layout']=layout
        if reason:
            record.update(status='rejected',reason=reason)
            continue
        metrics=audit(scene,layout,origin['Ncap'],origin['Omega_cap'],ledger)
        record['raw_metrics']=metrics
        if not protected(metrics,origin['metrics']):
            record.update(status='rejected',reason='common_B_quality_protection')
            continue
        record.update(status='protected',terminal=layout,metrics=metrics)
        if objective(metrics)+(slot['position'],)<objective(best['metrics'])+(best['position'],):
            best=record
            if observations is not None:observations['selected']=best
    return dict(results=results,selected=best,global_matching_slots=len(results))


class _KernelLedger(R9Ledger):
    """Enforce the frozen per-kernel R9 bounds AND the shared R10 ceiling."""
    def __init__(self,n,outer):
        super().__init__(n)
        self.outer=outer

    def tick(self,key,count=1):
        super().tick(key,count)
        if key not in ('scan_input_count','sort_input_count'):
            self.outer.tick(key,count)

    def scan(self,label,count):
        R9Ledger._pass(self,'scan',label,count)
        self.outer.scan(label,count)

    def sort(self,label,count):
        R9Ledger._pass(self,'sort',label,count)
        self.outer.sort(label,count)

    def ordered(self,values,key,label):
        R9Ledger._pass(self,'sort',label,len(values))
        return self.outer.ordered(values,key,label)


def _pool(scene,common,current,method,ledger):
    layout=current['layout']
    ids=layout['beam_ids']
    b=len(ids)
    owners=np.asarray(layout['assignment'],np.int64)
    counts=np.bincount(owners,minlength=b)
    n=common['n']
    q=common['q']
    ledger.scan('sequential_current_counts_and_plane_index',n+b)
    index={a:i for i,a in enumerate(scene.aircraft_ids)}
    rays=scene.rays[[index[a] for a in layout['anchor_aircraft_ids']]]
    distance=np.zeros((b,b),np.float64)
    for j in range(b-1):
        values=angles(rays[j],rays[j+1:],ledger)
        distance[j,j+1:]=values
        distance[j+1:,j]=values
    ledger.scan('sequential_physical_state',n+b)
    state=physical_state(scene,layout,common['identity']['case_id'],ledger)
    descriptors=[]
    actions=ACTIONS if method=='SH-C' else (ACTIONS[1],)
    for source in ledger.ordered(list(range(b)),lambda j:ids[j],'sequential_source_ID_order'):
        ledger.scan('sequential_nearest_partner_domain',b)
        other=[j for j in range(b) if j!=source]
        partner=min(other,key=lambda j:(float(distance[source,j]),ids[j])) if other else None
        for action,oldk,k in actions:
            region=[source] if oldk==1 else [source]+([] if partner is None else [partner])
            ledger.scan('sequential_descriptor_membership',n)
            memberindex=np.flatnonzero(np.isin(owners,region)).tolist()
            ledger.scan('sequential_descriptor_outside',b)
            outside=[j for j in range(b) if j not in region]
            m=len(memberindex)
            peak=max([int(counts[j]) for j in outside]+[(m+k-1)//k])
            delta=sum(max(int(counts[j])-q,0) for j in region)-max(0,m-k*q)
            a,r=divmod(m,k)
            square=sum(int(counts[j])**2 for j in outside)+(k-r)*a*a+r*(a+1)**2
            lower=omega([layout['widths_deg'][j] for j in outside])+k*omega([.3])
            ledger.scan('sequential_integer_bounds_and_Omega',2*len(outside)+len(region))
            reasons=[]
            if oldk==2 and partner is None:reasons.append('missing_partner')
            if m<k:reasons.append('members_below_target')
            if m>128:reasons.append('region_above128')
            if b-oldk+k>common['origin']['Ncap']:reasons.append('target_Ncap')
            if lower>common['origin']['Omega_cap']+1e-12:reasons.append('floor_Omega_cap')
            regionids=ledger.ordered([ids[j] for j in region],lambda a:a,'sequential_region_ID_order')
            semantic=digest([state,regionids,k,RECIPE])
            memberids=ledger.ordered([scene.aircraft_ids[i] for i in memberindex],lambda a:a,'sequential_region_member_ID_order')
            descriptor=dict(position=len(descriptors),action=action,region_beam_ids=regionids,
                source_beam_id=ids[source],partner_beam_id=ids[partner] if oldk==2 and partner is not None else None,
                member_ids=memberids,target_k=k,
                target_N=b-oldk+k,structurally_eligible=not reasons,unavailable_reasons=reasons,
                state_SHA=state,semantic_SHA=semantic,
                rank=[peak,-delta,square,lower,ids[source],ids[partner] if oldk==2 and partner is not None else '',action])
            descriptor['descriptor_SHA']=digest(descriptor)
            descriptors.append(descriptor)
            ledger.tick('descriptors_generated')
    if method=='S2':
        counts_by_id=dict(zip(ids,map(int,counts)))
        descriptors=ledger.ordered(descriptors,lambda d:(-counts_by_id[d['source_beam_id']],d['source_beam_id']),'sequential_descriptor_order')
    else:
        descriptors=ledger.ordered(descriptors,lambda d:tuple(d['rank']),'sequential_descriptor_order')
    return descriptors


def _kernel_body(scene,common,current,descriptor,outer,observations):
    """Frozen R9 numerical recipe with only v2 semantic beam-ID adaptation."""
    n=common['n']
    ledger=_KernelLedger(n,outer)
    observations['ledger']=ledger
    layout=copy.deepcopy(current['layout'])
    members=descriptor['member_ids']
    k=descriptor['target_k']
    anchors=choose_anchors(scene,members,k,ledger)
    result=dict(status='entered_unknown',semantic_SHA=descriptor['semantic_SHA'],selected_anchor_ids=anchors,kernel_entered=True)
    observations['result']=result
    def finish(reason):
        if result['status']=='entered_unknown':result['status']='rejected'
        result.update(reason=reason,ledger=ledger.record())
        return result
    if anchors is None:return finish('degenerate_mean_ray')
    ledger.scan('candidate_plane_index',n)
    ledger.scan('candidate_member_anchor_index_gather',len(members)+k)
    pindex={a:i for i,a in enumerate(scene.aircraft_ids)}
    mi=[pindex[a] for a in members]
    ai=[pindex[a] for a in anchors]
    ledger.pairs(len(mi)*k)
    d2=np.sum((scene.rays[mi,None,:]-scene.rays[None,ai,:])**2,axis=-1)
    ledger.scan('candidate_adjacency_entries',len(mi)*k)
    adjacency=(d2<=4*np.sin(np.deg2rad(.5)/2)**2+1e-14).tolist()
    match=minimum_capacity(tuple(members),tuple(anchors),adjacency,ledger)
    result['flow_capacity']=None if match is None else match['capacity']
    if match is None:return finish('fixed_anchor_capacity_infeasible')
    result['raw_local_matching']=dict(member_ids=list(members),anchor_ids=list(anchors),assignment=list(match['assignment']))
    local=np.asarray(match['assignment'],np.int64)
    widths=[]
    for j in range(k):
        ledger.scan('candidate_group_owner_mask',len(mi))
        ledger.scan('candidate_group_width_max',int(np.count_nonzero(local==j)))
        required=float(np.rad2deg(4*np.arcsin(min(math.sqrt(float(d2[local==j,j].max()))/2,1.))))
        width=max(.3,float(np.nextafter(required,np.inf)))
        ledger.tick('cached_member_scans',int(np.count_nonzero(local==j)))
        if width>1.:return finish('envelope_over1_conservative_kernel')
        widths.append(width)
    oldids=layout['beam_ids']
    region=set(descriptor['region_beam_ids'])
    ledger.scan('candidate_outside_beam_ID_domain',len(oldids))
    outside=[j for j,bid in enumerate(oldids) if bid not in region]
    newids=[json.dumps([common['identity']['case_id'],'R10_seq',descriptor['state_SHA'],
                       descriptor['semantic_SHA'],aid],ensure_ascii=False,separators=(',',':')) for aid in anchors]
    ids=[oldids[j] for j in outside]+newids
    require(len(ids)==len(set(ids)),'semantic new beam ID collision')
    ledger.scan('candidate_outside_index_map',len(outside))
    ledger.scan('candidate_full_owner_reindex',n)
    mapping={old:j for j,old in enumerate(outside)}
    owners=[mapping.get(int(owner),-1) for owner in layout['assignment']]
    for i,owner in zip(mi,local):owners[i]=len(outside)+int(owner)
    candidate=dict(beam_ids=ids,anchor_aircraft_ids=[layout['anchor_aircraft_ids'][j] for j in outside]+anchors,
        centers=[layout['centers'][j] for j in outside]+scene.positions[ai].tolist(),
        widths_deg=[layout['widths_deg'][j] for j in outside]+widths,assignment=owners)
    result['raw_layout']=candidate
    if omega(candidate['widths_deg'])>common['origin']['Omega_cap']+1e-12:return finish('whole_Omega_cap')
    metrics=audit(scene,candidate,common['origin']['Ncap'],common['origin']['Omega_cap'],ledger,audit_reserved=True)
    result['raw_metrics']=metrics
    if not protected(metrics,current['metrics']) or not protected(metrics,common['origin']['metrics']):
        return finish('dual_current_common_B_quality_protection')
    if objective(metrics)>=objective(current['metrics']):
        result['status']='no_improvement'
        return finish('exact_current_retained_on_tie_or_worse')
    result.update(status='admissible',terminal=candidate,metrics=metrics,
        new_beam_lineage=dict(removed_region_beam_ids=descriptor['region_beam_ids'],new_ids=newids))
    return finish('strict_raw_objective_improvement')


def _kernel(scene,common,current,descriptor,outer):
    observations={}
    try:return _kernel_body(scene,common,current,descriptor,outer,observations)
    except Exception as error:
        error.partial_kernel=copy.deepcopy(observations.get('result',{}))
        error.partial_kernel.pop('status',None)
        error.partial_kernel.update(complete=False,partial_status='unknown',kernel_entered=True,
            semantic_SHA=descriptor['semantic_SHA'])
        if 'ledger' in observations:error.partial_kernel['ledger']=observations['ledger'].record()
        raise


def _sequential(scene,common,method,ledger,observations=None):
    current=dict(layout=copy.deepcopy(common['origin']['layout']),metrics=copy.deepcopy(common['origin']['metrics']))
    seen=set()
    results=[]
    screenings=[]
    generations=0
    exhausted=False
    if observations is not None:observations.update(results=results,screenings=screenings,
        selected=dict(terminal=current['layout'],metrics=current['metrics'],status='protected_current'))
    while len(results)<32:
        ledger.tick('pool_generations')
        pool=_pool(scene,common,current,method,ledger)
        generations+=1
        require(generations<=33,'finite pool generation bound')
        rebuild=False
        poolkeys=set()
        for descriptor in pool:
            if len(screenings)>=26400:raise ResourceUnknown('descriptor_screenings_cap',ledger)
            key=descriptor['semantic_SHA']
            reason=('known_unavailable' if not descriptor['structurally_eligible'] else
                    'duplicate_semantics' if key in poolkeys else 'already_seen' if key in seen else 'kernel_attempted')
            screenings.append(dict(descriptor=descriptor,screening=reason))
            ledger.tick('screenings_'+reason)
            poolkeys.add(key)
            if reason!='kernel_attempted':continue
            seen.add(key)
            result=_kernel(scene,common,current,descriptor,ledger)
            result['attempt']=len(results)
            results.append(result)
            if result['status']=='admissible':
                current=dict(layout=result['terminal'],metrics=result['metrics'])
                if observations is not None:observations['selected']=dict(terminal=current['layout'],metrics=current['metrics'],status='protected_current')
                rebuild=True
                break
            if len(results)>=32:break
        if len(results)>=32:break
        if not rebuild:
            exhausted=True
            break
    return dict(results=results,screenings=screenings,kernel_attempts=len(results),
        pool_generations=generations,termination='no_progress_pool_exhausted' if exhausted else 'kernel_attempt_limit',
        selected=dict(terminal=current['layout'],metrics=current['metrics'],status='protected_current'))


def run_method(scene,common,method,*,stop=None):
    """Execute one admitted method; caller supplies native time/RSS guard.

    Resource exceptions carry completed legal output and precise partial work.
    This function never retries, changes q, accesses another case, or spawns.
    """
    require(method in METHODS,'frozen five methods')
    _common_valid(scene,common)
    began=time.monotonic()
    ledger=Ledger(common['n'],stop=stop)
    observations=dict(complete=False,partial_status='unknown',method=method,
        selected=dict(terminal=copy.deepcopy(common['origin']['layout']),metrics=copy.deepcopy(common['origin']['metrics']),status='incumbent'))
    try:
        result=(_global(scene,common,method,ledger,observations) if method in ('Gd','Gr','Gw')
                else _sequential(scene,common,method,ledger,observations))
    except Exception as error:
        error.partial_ledger=ledger.record()
        error.partial_result=copy.deepcopy(observations)
        raise
    result.update(schema='R10MethodV2',method=method,complete=True,identity=common['identity'],
        original_index=common['original_index'],common_SHA=common['common_SHA'],origin_SHA=common['origin_SHA'],q=common['q'],partial_status=None,
        ledger=ledger.record(),common_ledger=copy.deepcopy(common['common_ledger']),
        common_ledger_added_exactly_once_in_cold_view=True,
        observed_method_seconds=time.monotonic()-began,new_RULE_D5_T_W_model_calls=0)
    return result
