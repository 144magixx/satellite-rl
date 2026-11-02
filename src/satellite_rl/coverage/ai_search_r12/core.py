"""R12 finite two-step joint region tree. No work on import.

All geometry/flow entry points are callable only by a separately admitted worker.
The persist callback is mandatory and must durably commit before returning.
"""
import copy
import json
import math
import time
import numpy as np
from ..ai_search_r9.core import (REQUEST, require, digest, angles, audit, omega,
                                protected, hit)
from ..ai_search_r10.core import physical_state, _scene_identity, qualify_metrics
from .anchors import choose_anchors
from .exact_order import exact_key, layout_evidence, exact_evidence as counts_widths_evidence
from .local_flow import minimum_capacity
from .ledger import Ledger, MotherLedger, ResourceUnknown
from .schema import SCHEMA, NodeState, qualified_complete

RECIPE = 'R12-contact-slack-exact-order-v2'
INDICES = (0,6,12,18,24,36,42,48,54,60,72,78)


def support_region(counts, beam_ids, contact_pairs):
    """Pure supplied squared-chord / angle selector; never calls geometry.

    contact_pairs maps (peak index, nonpeak index) to (chord2, angle).
    All eligible-domain pairs must be supplied, even noncontacting ones.
    """
    require(len(counts)==len(beam_ids)>0 and len(beam_ids)<=200 and len(set(beam_ids))==len(beam_ids)
            and all(type(x) is int and x>0 for x in counts),'nonempty current counts')
    p=max(counts);peaks=[i for i,c in enumerate(counts) if c==p];s=len(peaks)
    candidates=[j for j,c in enumerate(counts) if c<p and s<=30 and s*p+c<=128]
    if not candidates:return dict(region=[],peaks=peaks,support=None,unsupported_reason='unsupported_no_slack_support')
    threshold=4*math.sin(math.radians(.5)/2)**2+1e-14
    contacting=[]
    for j in candidates:
        values=[contact_pairs[i,j] for i in peaks]
        require(all(len(v)==2 and all(math.isfinite(float(x)) and x>=0 for x in v) for v in values),'complete finite given contact cache')
        if any(v[0]<=threshold for v in values):
            contacting.append(( ((s*p+counts[j]+s)//(s+1),counts[j],min(v[1] for v in values),beam_ids[j]),j))
    if not contacting:return dict(region=[],peaks=peaks,support=None,unsupported_reason='unsupported_no_contacting_support')
    _,j=min(contacting)
    return dict(region=sorted(peaks+[j],key=lambda i:beam_ids[i]),peaks=peaks,support=j,unsupported_reason=None,
                immediate_peak_reduction_necessary=counts[j]<=p-s-1)


def region_indices(counts, beam_ids, contact_pairs=None):
    return support_region(counts,beam_ids,{} if contact_pairs is None else contact_pairs)['region']


def describe(scene,current,origin,ledger,*,fixed_region=None,fixed_target=None,physical_SHA=None):
    """Two current-region actions, or an explicitly labelled engineering region."""
    layout=current['layout'];ids=layout['beam_ids'];n=len(scene.aircraft_ids)
    counts=np.bincount(np.asarray(layout['assignment'],np.int64),minlength=len(ids)).tolist()
    ledger.scan('region_current_owner_counts',n+len(ids))
    selection=dict(peaks=[],support=None,unsupported_reason=None)
    contact_cache=dict(pairs=[],charged_raypairs=0)
    if fixed_region is None:
        peakcount=max(counts);peaks=[j for j,c in enumerate(counts) if c==peakcount]
        candidates=[j for j,c in enumerate(counts) if c<peakcount and len(peaks)<=30 and len(peaks)*peakcount+c<=128]
        pairs={}
        if candidates:
            index={a:i for i,a in enumerate(scene.aircraft_ids)}
            arays=scene.rays[[index[a] for a in layout['anchor_aircraft_ids']]]
            # Exactly one center-pair construction per distinct parent, shared J0/J1.
            ledger.pairs(len(peaks)*len(candidates))
            ledger.scan('support_contact_cache',len(peaks)*len(candidates)*3+len(ids)+n)
            for i in peaks:
                for j in candidates:
                    chord2=float(np.sum((arays[i]-arays[j])**2))
                    angle=float(np.rad2deg(2*np.arcsin(min(math.sqrt(chord2)/2,1.))))
                    pairs[i,j]=(chord2,angle)
                    contact_cache['pairs'].append(dict(peak_beam_id=ids[i],support_beam_id=ids[j],chord2=chord2,angle=angle))
            contact_cache['charged_raypairs']=len(peaks)*len(candidates)
        ledger.tick('sort_key_comparisons',len(ids)*len(ids))
        selection=support_region(counts,ids,pairs)
        region=selection['region'] or sorted(selection['peaks'],key=lambda j:ids[j]);mode='production_contact_slack_v2'
    else:
        require(fixed_target is not None,'explicit fixed target only for engineering')
        region=sorted(list(fixed_region),key=lambda j:ids[j]);mode='engineering_fixed_region'
        require(len(region)==len(set(region)) and all(0<=j<len(ids) for j in region),'fixed region indices')
    # Pure selector contains an opaque <=200 item stable-ID sort.
    ledger.tick('sort_key_comparisons',len(region)*len(region))
    regionids=[ids[j] for j in region];region_set=set(region)
    memberids=ledger.ordered([a for a,o in zip(scene.aircraft_ids,layout['assignment']) if o in region_set],lambda a:a,'region_member_ID_order')
    outside=[j for j in range(len(ids)) if j not in region_set]
    r,m=len(region),len(memberids)
    state=physical_state(scene,layout,origin['identity']['case_id'],ledger) if physical_SHA is None else physical_SHA
    ledger.scan('region_full_members_outside',6*n+8*len(ids))
    qB=max(layout_evidence(origin['layout'],ledger)['counts'])-1
    descriptors=[]
    targets=[fixed_target] if fixed_region is not None else ([r,r+1] if selection['unsupported_reason'] is None else [r+1,r+2])
    for pos,k in enumerate(targets):
        ledger.scan('region_action_bounds_identity',2*len(memberids)+12*len(ids))
        action='E'+str(k) if fixed_region is not None else 'J'+str(pos)
        newN=len(outside)+k
        lower=omega([layout['widths_deg'][j] for j in outside])+k*omega([.3])
        reasons=[] if selection['unsupported_reason'] is None else [selection['unsupported_reason']]
        if not 1<=r<=31:reasons.append('region_above31')
        if m>128:reasons.append('region_above128')
        if not 1<=k<=32 or k>m:reasons.append('members_below_target_or_k_bound')
        if newN>min(200,origin['Ncap']):reasons.append('target_Ncap')
        if lower>origin['Omega_cap']+1e-12:reasons.append('floor_Omega_cap')
        a,rem=divmod(m,k)
        peak=max([counts[j] for j in outside]+[(m+k-1)//k])
        excess=sum(max(counts[j]-qB,0) for j in region)-max(0,m-k*qB)
        square=sum(counts[j]**2 for j in outside)+(k-rem)*a*a+rem*(a+1)**2
        semantic=digest([state,memberids,k,RECIPE])
        d=dict(action=action,mode=mode,region_beam_ids=regionids,member_ids=memberids,
               target_k=k,target_N=newN,state_SHA=state,
               unsupported_reason=selection['unsupported_reason'],support_beam_id=None if selection['support'] is None else ids[selection['support']],
               peak_beam_ids=sorted(ids[j] for j in selection['peaks']),contact_cache=contact_cache,
               immediate_peak_reduction_necessary=selection.get('immediate_peak_reduction_necessary'),current_layout_SHA=digest(layout),semantic_SHA=semantic,
               structurally_eligible=not reasons,unavailable_reasons=reasons,
               rank=[peak,-excess,square,lower,newN,action])
        d['descriptor_SHA']=digest(d);descriptors.append(d)
    return descriptors


def _body(scene,origin,current,descriptor,ledger,result):
    layout=copy.deepcopy(current['layout']);members=descriptor['member_ids'];k=descriptor['target_k']
    result['kernel_entered']=True
    anchors=choose_anchors(scene,members,k,ledger);result['selected_anchor_ids']=anchors
    if anchors is None:return NodeState.REJECTED,'degenerate_mean_ray','observed_degenerate_mean'
    index={a:i for i,a in enumerate(scene.aircraft_ids)}
    mi=[index[a] for a in members];ai=[index[a] for a in anchors]
    ledger.scan('candidate_indices',len(scene.aircraft_ids)+len(mi)+k)
    ledger.pairs(len(mi)*k)
    d2=np.sum((scene.rays[mi,None,:]-scene.rays[None,ai,:])**2,axis=-1)
    ledger.scan('candidate_adjacency',len(mi)*k)
    adjacency=(d2<=4*np.sin(np.deg2rad(.5)/2)**2+1e-14).tolist()
    result['raw_adjacency']=adjacency
    match=minimum_capacity(tuple(members),tuple(anchors),adjacency,ledger)
    result['flow_capacity']=None if match is None else match['capacity']
    if match is None:
        return NodeState.GRAPH_INFEASIBLE,'fixed_anchor_capacity_infeasible','complete_pinned_graph_hi_m_infeasible'
    result['raw_local_matching']=dict(member_ids=members,anchor_ids=anchors,assignment=match['assignment'])
    local=np.asarray(match['assignment'],np.int64);widths=[]
    for j in range(k):
        ledger.scan('candidate_group_owner_mask',4*len(mi))
        ledger.scan('candidate_group_max',int(np.count_nonzero(local==j)))
        required=float(np.rad2deg(4*np.arcsin(min(math.sqrt(float(d2[local==j,j].max()))/2,1.))))
        width=max(.3,float(np.nextafter(required,np.inf)))
        result.setdefault('observed_widths_deg',[]).append(width)
        if width>1.:return NodeState.REJECTED,'envelope_over1_conservative_kernel','observed_outward_envelope'
        widths.append(width)
    region=set(descriptor['region_beam_ids']);oldids=layout['beam_ids']
    ledger.scan('candidate_outside_ID_domain',len(oldids))
    outside=[j for j,b in enumerate(oldids) if b not in region]
    newids=[json.dumps([origin['identity']['case_id'],'R12',descriptor['semantic_SHA'],a],
                      ensure_ascii=False,separators=(',',':')) for a in anchors]
    ids=[oldids[j] for j in outside]+newids;require(len(ids)==len(set(ids)),'semantic beam collision')
    mapping={old:j for j,old in enumerate(outside)}
    owners=[mapping.get(int(o),-1) for o in layout['assignment']]
    for i,o in zip(mi,local):owners[i]=len(outside)+int(o)
    ledger.scan('candidate_full_owner_reindex',len(owners)+len(oldids))
    candidate=dict(beam_ids=ids,anchor_aircraft_ids=[layout['anchor_aircraft_ids'][j] for j in outside]+anchors,
        centers=[layout['centers'][j] for j in outside]+scene.positions[ai].tolist(),
        widths_deg=[layout['widths_deg'][j] for j in outside]+widths,assignment=owners)
    result['raw_layout']=candidate
    if omega(candidate['widths_deg'])>origin['Omega_cap']+1e-12:
        return NodeState.REJECTED,'whole_Omega_cap','actual_whole_Omega'
    metrics=audit(scene,candidate,origin['Ncap'],origin['Omega_cap'],ledger)
    result['raw_metrics']=metrics
    result['raw_exact_evidence']=layout_evidence(candidate,ledger)
    outside_set=set(outside)
    ledger.scan('candidate_outside_immutability',len(layout['assignment'])+len(outside))
    for i,old in enumerate(layout['assignment']):
        if old in outside_set:
            new=owners[i]
            require(ids[new]==oldids[old] and candidate['centers'][new]==layout['centers'][old]
                and candidate['widths_deg'][new]==layout['widths_deg'][old]
                and candidate['anchor_aircraft_ids'][new]==layout['anchor_aircraft_ids'][old],
                'outside owner geometry immutable')
    if not protected(metrics,current['metrics']) or not protected(metrics,origin['metrics']):
        return NodeState.REJECTED,'dual_current_B_quality_protection','complete_raw_metrics'
    if exact_key(result['raw_exact_evidence'],ledger)>=exact_key(current.get('exact_evidence') or layout_evidence(current['layout'],ledger),ledger):
        return NodeState.NO_IMPROVEMENT,'exact_current_tie_or_worse','complete_raw_metrics'
    result.update(terminal=candidate,metrics=metrics,exact_evidence=result['raw_exact_evidence'])
    return NodeState.ADMISSIBLE,'strict_raw_objective_improvement','complete_hard_legal_dual_protected'


def evaluate(scene,origin,current,descriptor,*,stop=None):
    """One attempt; caller must journal entered before calling. Never retries."""
    began=time.monotonic()
    user_stop=stop
    stop=lambda: time.monotonic()-began>=30 or (user_stop is not None and user_stop())
    ledger=Ledger(len(scene.aircraft_ids),stop=stop)
    layout=current['layout']
    require(descriptor['descriptor_SHA']==digest({k:v for k,v in descriptor.items() if k!='descriptor_SHA'}),'descriptor content SHA')
    require(descriptor['current_layout_SHA']==digest(layout),'descriptor exact current layout content')
    ledger.scan('descriptor_current_layout_binding',len(scene.aircraft_ids)+len(layout['beam_ids']))
    regions=set(descriptor['region_beam_ids'])
    members=[a for a,o in zip(scene.aircraft_ids,layout['assignment']) if layout['beam_ids'][int(o)] in regions]
    ledger.scan('descriptor_member_semantic_validation',3*len(scene.aircraft_ids)+3*len(descriptor['member_ids'])+len(layout['beam_ids']))
    require(set(members)==set(descriptor['member_ids']) and len(members)==len(descriptor['member_ids']) and len(regions)==len(descriptor['region_beam_ids']) and regions<=set(layout['beam_ids']),'full region members without truncation')
    require(descriptor['semantic_SHA']==digest([descriptor['state_SHA'],descriptor['member_ids'],descriptor['target_k'],RECIPE]),'semantic content')
    result=dict(schema=SCHEMA,complete=False,state=NodeState.UNKNOWN.value,
        semantic_SHA=descriptor['semantic_SHA'],descriptor=copy.deepcopy(descriptor),
        identity=copy.deepcopy(origin['identity']),kernel_entered=False,
        terminal=copy.deepcopy(current['layout']),metrics=copy.deepcopy(current['metrics']),
        exact_evidence=copy.deepcopy(current.get('exact_evidence') or layout_evidence(current['layout'],ledger)),
        raw_exact_evidence=None,raw_layout=None,raw_metrics=None,raw_local_matching=None,
        raw_producer_label='entered_unknown',proof_qualification='unknown_not_a_negative')
    try:
        if not descriptor['structurally_eligible']:
            state,label,proof=NodeState.KNOWN_UNAVAILABLE,'known_unavailable','analytic_structure_bound'
            result['unavailable_reasons']=descriptor['unavailable_reasons']
        else:state,label,proof=_body(scene,origin,current,descriptor,ledger,result)
        result.update(observed_native_seconds=time.monotonic()-began,state=state.value,raw_producer_label=label,proof_qualification=proof,
                      ledger=ledger.record(),complete=True)
        result['source_content_id']=digest(result)
        return result
    except Exception as error:
        result.update(ledger=ledger.record(),error_type=type(error).__name__,error=str(error))
        error.partial_node=result
        raise


def explore_tree(scene,origin,persist,*,stop=None):
    """Shared prefix six-attempt tree; callback must save entered/completed/alias.

    No recovery replay API: a persisted unknown/inflight task closes incomplete.
    All rejected and known-unavailable semantics stay cached, including fallback.
    """
    began=time.monotonic()
    user_stop=stop
    stop=lambda: time.monotonic()-began>=180 or (user_stop is not None and user_stop())
    _scene_identity(scene,origin['identity'])
    require(origin['layout_SHA']==digest(origin['layout']),'original B content')
    mother=MotherLedger(len(scene.aircraft_ids),stop=stop)
    metrics=audit(scene,origin['layout'],origin['Ncap'],origin['Omega_cap'],mother)
    qualify_metrics(metrics,origin['metrics'])
    base=dict(layout=copy.deepcopy(origin['layout']),metrics=copy.deepcopy(origin['metrics']),exact_evidence=layout_evidence(origin['layout'],mother))
    cache={};nodes=[];descriptors={};states={'':base};paths={}
    for prefix in ('','0','1'):
        if prefix not in states:raise AssertionError('missing shared prefix')
        current=states[prefix]
        state=physical_state(scene,current['layout'],origin['identity']['case_id'],mother)
        if state not in descriptors:descriptors[state]=describe(scene,current,origin,mother,physical_SHA=state)
        for action,d in enumerate(descriptors[state]):
            path=prefix+str(action);key=d['semantic_SHA']
            if key in cache:
                node=cache[key]
                require(qualified_complete(node),'unknown semantic cannot replay')
                persist('alias',dict(path=path,semantic_SHA=key,source_content_id=node['source_content_id']))
            else:
                persist('entered',dict(path=path,semantic_SHA=key,descriptor=d,
                    state=NodeState.UNKNOWN.value,complete=False))
                try:node=evaluate(scene,origin,current,d,stop=stop)
                except Exception as error:
                    partial=getattr(error,'partial_node',dict(state='unknown',complete=False))
                    # Save observed raw counters even when stop forbids further
                    # budget ticks. Never reinterpret interrupted work as zero.
                    error.partial_mother_ledger=dict(completed=mother.record(),inflight=partial.get('ledger'))
                    persist('unknown',dict(path=path,node=partial,mother_ledger=error.partial_mother_ledger))
                    raise
                persist('completed',dict(path=path,node=node))
                mother.absorb(node['ledger'])
                if node['kernel_entered']:mother.tick('kernels')
                cache[key]=node;nodes.append(node)
            states[path]=dict(layout=copy.deepcopy(node['terminal']),metrics=copy.deepcopy(node['metrics']),exact_evidence=copy.deepcopy(node['exact_evidence']))
            paths[path]=dict(semantic_SHA=key,terminal=states[path]['layout'],metrics=states[path]['metrics'],exact_evidence=states[path]['exact_evidence'])
    return dict(schema='R12TreeV2',complete=True,identity=origin['identity'],origin_SHA=origin['layout_SHA'],
                nodes=nodes,paths=paths,ledger=mother.record(),new_model_calls=0)


def select_controls(origin,tree,saved_S):
    """Saved controls only. No legacy solver or historical producer replay."""
    require(tree.get('complete') is True and len(tree['paths'])==6,'complete finite tree')
    order_ledger=MotherLedger(len(origin['layout']['assignment']))
    order_ledger.absorb(tree['ledger'])
    saved_S=copy.deepcopy(saved_S)
    if 'counts' in saved_S and 'widths_deg' in saved_S:
        e=counts_widths_evidence(saved_S['counts'],saved_S['widths_deg'],order_ledger)
        if 'exact_evidence' in saved_S:require(exact_key(e,order_ledger)==exact_key(saved_S['exact_evidence'],order_ledger),'historical saved exact evidence agrees')
        saved_S['exact_evidence']=e
    else:
        require('exact_evidence' in saved_S,'historical full counts and widths or audited exact evidence')
        exact_key(saved_S['exact_evidence'],order_ledger)
    expected_n=len(origin['layout']['assignment'])
    require(saved_S['exact_evidence']['n']==expected_n and all(p['exact_evidence']['n']==expected_n for p in tree['paths'].values()),'same complete aircraft count for exact comparisons')
    paths=tree['paths'];bykey={n['semantic_SHA']:n for n in tree['nodes']}
    def best(keys):return min(keys,key=lambda key:exact_key(paths[key]['exact_evidence'],order_ledger)+(key,))
    g1=best(['0','1']);g=best([g1+'0',g1+'1'])
    h=''
    for _ in range(2):
        children=[h+'0',h+'1']
        eligible=[key for key in children if bykey[paths[key]['semantic_SHA']]['descriptor']['structurally_eligible']]
        h=min(eligible,key=lambda key:tuple(bykey[paths[key]['semantic_SHA']]['descriptor']['rank'])) if eligible else children[0]
    order_ledger.scan('control_H_fixed_comparisons',64)
    order_ledger.tick('sort_key_comparisons',32)
    controls={'S':saved_S,'H':paths[h],'G':paths[g]}
    q=min(('S','H','G'),key=lambda key:exact_key(controls[key]['exact_evidence'],order_ledger)+(('S','H','G').index(key),))
    o1=min([dict(metrics=origin['metrics'],terminal=origin['layout'],exact_evidence=layout_evidence(origin['layout'],order_ledger))]+[paths[k] for k in ('0','1')],key=lambda x:exact_key(x['exact_evidence'],order_ledger))
    o2=paths[best(['00','01','10','11'])]
    def logical_cost(keys):
        semantics=sorted(set(paths[k]['semantic_SHA'] for k in keys))
        return dict(path_references=keys,unique_semantic_SHAs=semantics,
            kernels=sum(int(bykey[s]['kernel_entered']) for s in semantics),
            flows=sum(bykey[s]['ledger']['flow_queries'] for s in semantics),
            scope='logical path unique semantics; historical S costs separate; physical tree charged once')
    costs=dict(H=logical_cost([h[:1],h]),G=logical_cost(['0','1',g1+'0',g1+'1']))
    costs.update({'F'+seq:logical_cost([seq[:1],seq]) for seq in ('00','01','10','11')})
    return dict(**controls,Q=controls[q],O1=o1,O2=o2,F={k:paths[k] for k in ('00','01','10','11')},logical_costs=costs,ordering_ledger=order_ledger.record())


def mechanism_gate(rows):
    require([r['original_index'] for r in rows]==list(INDICES),'complete fixed original12')
    totals=dict(S=0,Q=0,O1=0);fixed={s:0 for s in ('00','01','10','11')}
    for row in rows:
        require(row.get('complete') is True and row.get('reader_validated') is True,'full independent constraint/budget qualification')
        c=row['controls']
        for ref in totals:totals[ref]+=int(hit(c['O2']['metrics'],c[ref]['metrics']))
        for seq in fixed:fixed[seq]+=int(hit(c['F'][seq]['metrics'],c['Q']['metrics']))
    best=min(fixed,key=lambda s:(-fixed[s],s));gap=totals['Q']-fixed[best]
    return dict(complete=True,hits=totals,fixed_hits=fixed,best_fixed=best,gap=gap,
                passed=totals['S']>=3 and totals['Q']>=3 and gap>=3 and totals['O1']>=2,
                training_auto_start=False)
