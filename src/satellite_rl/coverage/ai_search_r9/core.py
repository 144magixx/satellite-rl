"""R9 M-stage deterministic protected reconstruction. No work on import.

Original RULE/D5 acquisition is a separate, explicitly charged caller. This
module never calls RULE, D5, teacher, model, optimizer or a continuous solver.
"""
import copy,hashlib,json,math,time
import numpy as np
from .flow import Ledger,ResourceUnknown,minimum_capacity

REQUEST=21600000
ACTIONS=(('1_to_2_split_source',1,2),('2_to_2_repartition_source_partner',2,2),('2_to_3_expand_source_partner',2,3),('2_to_1_merge_source_partner',2,1))
IDENTITY_KEYS=('case_id','mother_id','profile','role','input_sha256','identity_sha256','shared_case_content_sha256','source_asset_SHA')

class InterfaceError(ValueError):pass

def require(ok,message):
    if not ok:raise InterfaceError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def omega(widths):return float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(np.asarray(widths,dtype=np.float64)/2)))))
def finite(x):return isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,(bool,np.bool_)) and math.isfinite(x) and x>=0
def within(x,y,f=1.,key='maxload'):return finite(x) and finite(y) and x<=f*y+(1e-9 if key=='CV' else 1e-6)+1e-12*abs(y)
def protected(a,b):return all(within(a[k],b[k],key=k) for k in ('maxload','top10_mean','CV'))
def objective(m):return tuple(m[k] for k in ('maxload','top10_mean','CV','N','actual_Omega'))
def hit(a,b):return b['maxload']>0 and within(a['maxload'],b['maxload'],.99) and all(within(a[k],b[k],key=k) for k in ('top10_mean','CV'))

class CommonLedger(Ledger):
    def __init__(self,n,prior=None):
        super().__init__(n)
        if prior is not None:
            require(prior.get('common_raypairs_bound')==3*n+19900,'same cumulative common ledger bound')
            self.data.update(copy.deepcopy({k:v for k,v in prior.items() if k in self.data}))
            require(self.data['raypairs']<=3*n+19900,'prior common raypairs')
    def pairs(self,count,audit=False):
        # Three source layout audits + full upper triangle of<=200anchors.
        if self.data['raypairs']+count>3*self.n+19900:raise ResourceUnknown('common_raypairs_bound',self.record())
        self.tick('raypairs',count)
    def record(self):
        r=dict(self.data);r['common_raypairs_bound']=3*self.n+19900;r['scope']='shared origin qualification, pool and features; no candidate work';return r

def angles(first,second,ledger):
    # Calls the unchanged original primitive only after debit.
    count=int(np.prod(np.broadcast_shapes(np.shape(first)[:-1],np.shape(second)[:-1])))
    ledger.pairs(count)
    from ..geometry import ray_angle_numpy
    return ray_angle_numpy(first,second)

def quality(counts,widths):
    counts=np.asarray(counts,dtype=np.int64);loads=counts.astype(np.float64)*REQUEST;b=len(counts)
    return dict(N=b,maxload=int(counts.max())*REQUEST,top10_mean=float(np.sort(loads)[-max(1,math.ceil(.1*b)):].mean()),CV=float(loads.std()/loads.mean()),actual_Omega=omega(widths),total_request=int(counts.sum())*REQUEST)

def audit(scene,layout,ncap,omegacap,ledger,audit_reserved=False):
    """Full scene stable IDs/anchors, exact coverage/envelope/request audit."""
    n=len(scene.aircraft_ids);ids=layout['beam_ids'];anchors=layout['anchor_aircraft_ids'];b=len(ids)
    centers=np.asarray(layout['centers'],np.float64);widths=np.asarray(layout['widths_deg'],np.float64);owners=np.asarray(layout['assignment'])
    require(all(isinstance(x,str) and x for x in list(scene.aircraft_ids)+list(ids)+list(anchors)) and len(set(scene.aircraft_ids))==n,'original stable string IDs without coercion')
    require(1<=b<=min(200,n,ncap) and len(set(ids))==b and len(anchors)==b,'actual nonempty N/unique IDs')
    require(centers.shape==(b,2) and widths.shape==(b,) and owners.shape==(n,) and np.isfinite(centers).all() and np.isfinite(widths).all() and np.isfinite(owners).all(),'full shapes finite')
    require(np.all((widths>=.3)&(widths<=1)) and np.equal(owners,np.floor(owners)).all() and np.all((owners>=0)&(owners<b)),'fullwidth/integer complete ownership')
    planeindex={aid:i for i,aid in enumerate(scene.aircraft_ids)};require(all(a in planeindex for a in anchors),'actual aircraft anchor IDs')
    anchorindex=[planeindex[a] for a in anchors];require(np.array_equal(centers,scene.positions[anchorindex]),'actual center to original anchor projection')
    owners=owners.astype(np.int64);counts=np.bincount(owners,minlength=b);require(np.all(counts>0),'nonempty groups')
    require(np.asarray(scene.demands).shape==(n,) and np.all(np.asarray(scene.demands)==REQUEST),'all original uniform requests')
    ledger.pairs(n,audit=audit_reserved);chord=np.sum((scene.rays-scene.rays[anchorindex][owners])**2,axis=1)
    require(np.all(chord<=4*np.sin(np.deg2rad(widths[owners]/2)/2)**2+1e-14),'all original aircraft geometric coverage')
    for j in range(b):
        required=max(.3,float(np.nextafter(np.rad2deg(4*np.arcsin(min(math.sqrt(float(chord[owners==j].max()))/2,1.))),np.inf)))
        require(math.isclose(float(widths[j]),required,rel_tol=0,abs_tol=1e-12),'original minimum fullwidth envelope')
    ledger.tick('cached_member_scans',n);ledger.tick('scan_input_count',n+b+n*b);ledger.tick('sort_input_count',b)
    metrics=quality(counts,widths);require(metrics['actual_Omega']<=omegacap+1e-12,'whole scene original Omega cap')
    require(metrics['total_request']==n*REQUEST,'integer whole-plane request conservation')
    return metrics

def _layout(record,anchors=None):
    result=copy.deepcopy(record)
    if 'anchor_aircraft_ids' not in result:
        require(anchors is not None,'explicit original Wenv anchors');result['anchor_aircraft_ids']=list(anchors)
    return result

def select_origin(scene,prepared,rule,teacher=None):
    """RULE is mandatory floor; choose once before any descriptor generation."""
    n=len(scene.aircraft_ids);ledger=CommonLedger(n);began=time.monotonic();identity=prepared.identity
    require(all(k in identity for k in IDENTITY_KEYS) and identity['profile']=='uniform21600000' and identity['role'] in ('train','development'),'full original role/identity')
    require(prepared.status=='available' and scene.input_sha256==identity['input_sha256'] and isinstance(prepared.Ncap,int) and 1<=prepared.Ncap<=200 and finite(prepared.Omega_cap) and prepared.Omega_cap>0,'original qualified prepared/scene/caps')
    rule=rule.get('result',rule);require(rule.get('status')=='success','mandatory successful RULE floor; fallback is unqualified, never replaced')
    require(all(rule.get(k)==identity[k] for k in IDENTITY_KEYS),'RULE full identity')
    rulelayout=_layout(rule['layout']);rulemetrics=audit(scene,rulelayout,prepared.Ncap,prepared.Omega_cap,ledger)
    candidates=[('RULE',rulelayout,rulemetrics)];eligibility=[dict(source='RULE',eligible=True,status=rule['status'],metrics=rulemetrics,layout_SHA=digest(rulelayout))]
    wenv=_layout(prepared.Wenv.to_record(),prepared.anchors);wm=audit(scene,wenv,prepared.Ncap,prepared.Omega_cap,ledger)
    for tag,layout,metrics in [('Wenv',wenv,wm)]:
        eligible=protected(metrics,rulemetrics);eligibility.append(dict(source=tag,eligible=eligible,metrics=metrics,layout_SHA=digest(layout),reason=None if eligible else 'not_componentwise_RULE_protected'))
        if eligible:candidates.append((tag,layout,metrics))
    if teacher is not None:
        teacher=teacher.get('result',teacher);tid=teacher.get('provenance',teacher)
        require(all(tid.get(k)==identity[k] for k in IDENTITY_KEYS),'T original full identity')
        if teacher.get('status')=='available':
            layout=_layout(teacher['layout']);metrics=audit(scene,layout,prepared.Ncap,prepared.Omega_cap,ledger);eligible=protected(metrics,rulemetrics)
            eligibility.append(dict(source='T',eligible=eligible,metrics=metrics,layout_SHA=digest(layout),reason=None if eligible else 'not_componentwise_RULE_protected'))
            if eligible:candidates.append(('T',layout,metrics))
        else:eligibility.append(dict(source='T',eligible=False,reason='original_T_unavailable'))
    else:eligibility.append(dict(source='T',eligible=False,reason='original_T_absent'))
    tag,layout,metrics=min(candidates,key=lambda x:objective(x[2])+(x[0],))
    return dict(identity=copy.deepcopy(identity),layout=layout,metrics=metrics,RULE_layout=rulelayout,RULE_metrics=rulemetrics,source=tag,eligibility=eligibility,Ncap=prepared.Ncap,Omega_cap=prepared.Omega_cap,layout_SHA=digest(layout),RULE_layout_SHA=digest(rulelayout),common_ledger=ledger.record(),observed_selection_seconds=time.monotonic()-began)

def build_pool(scene,origin):
    began=time.monotonic();n=len(scene.aircraft_ids);ledger=CommonLedger(n,origin['common_ledger']);layout=origin['layout'];owners=np.asarray(layout['assignment'],np.int64);widths=np.asarray(layout['widths_deg']);ids=layout['beam_ids'];b=len(ids);counts=np.bincount(owners,minlength=b);pindex={aid:i for i,aid in enumerate(scene.aircraft_ids)};rays=scene.rays[[pindex[aid] for aid in layout['anchor_aircraft_ids']]]
    matrix=np.zeros((b,b),np.float64)
    for i in range(b-1):values=angles(rays[i],rays[i+1:],ledger);matrix[i,i+1:]=values;matrix[i+1:,i]=values
    ranks=sorted(range(b),key=lambda j:(-int(counts[j]),ids[j]))[:8];ledger.tick('sort_input_count',2*b);ledger.tick('scan_input_count',n);descriptors=[];allcounts=sorted(map(int,counts),reverse=True);rankcount=allcounts[math.ceil(.1*b)-1]
    for sb in ranks:
        other=[j for j in range(b) if j!=sb];partner=min(other,key=lambda j:(matrix[sb,j],ids[j])) if other else None;ledger.tick('scan_input_count',len(other))
        for action,oldk,k in ACTIONS:
            region=[sb] if oldk==1 else [sb]+([partner] if partner is not None else [])
            members=np.flatnonzero(np.isin(owners,region));outside=[j for j in range(b) if j not in region];m=len(members);newN=b-oldk+k;outsidepeak=max((int(counts[j]) for j in outside),default=0);outsideomega=omega(widths[outside]);loweromega=outsideomega+k*omega([.3]);lowerpeak=max(outsidepeak,(m+k-1)//k)
            ledger.tick('scan_input_count',n+3*b);ledger.tick('sort_input_count',m)
            reasons=[]
            if oldk==2 and partner is None:reasons.append('missing_partner')
            if m<k:reasons.append('members_below_target')
            if m>128:reasons.append('region_above128')
            if newN>origin['Ncap']:reasons.append('target_Ncap')
            if loweromega>origin['Omega_cap']+1e-12:reasons.append('floor_Omega_cap')
            # Split partner features are all zero, even if a neighbor exists.
            pb=partner if oldk==2 else None;pc=int(counts[pb]) if pb is not None else 0;pw=float(widths[pb]) if pb is not None else 0.;pa=float(matrix[sb,pb]) if pb is not None else 0.
            features=[n/2000,b/200,origin['Ncap']/200,origin['metrics']['actual_Omega']/origin['Omega_cap'],origin['metrics']['maxload']/REQUEST/n,origin['metrics']['top10_mean']/REQUEST/n,origin['metrics']['CV'],int(counts[sb])/n,pc/n,m/n,oldk/3,k/3,float(widths[sb]),pw,pa,omega(widths[region])/origin['Omega_cap'],(origin['Omega_cap']-origin['metrics']['actual_Omega'])/origin['Omega_cap'],(origin['Ncap']-b)/200,outsidepeak/n,rankcount/n,m/k/n,lowerpeak/n,int(oldk==1),(k-oldk)/2]
            require(len(features)==24 and np.isfinite(features).all(),'exact24 pre-solve features')
            descriptor=dict(position=len(descriptors),action=action,region_beam_ids=[ids[j] for j in region],source_beam_id=ids[sb],partner_beam_id=ids[partner] if oldk==2 and partner is not None else None,member_ids=sorted(scene.aircraft_ids[i] for i in members),old_region_beams=oldk,target_k=k,target_N=newN,global_peak_lower_count=lowerpeak,Omega_lower_bound=loweromega,structurally_eligible=not reasons,unavailable_reasons=reasons,features24=features)
            descriptor['descriptor_SHA']=digest(descriptor);descriptors.append(descriptor)
    return dict(case_id=origin['identity']['case_id'],origin_SHA=origin['layout_SHA'],descriptors=descriptors,pool_SHA=digest(descriptors),common_ledger=ledger.record(),common_ledger_is_cumulative_includes_origin=True,origin_ledger_SHA=digest(origin['common_ledger']),observed_pool_feature_seconds=time.monotonic()-began)

def method_order(pool,method,scores=None):
    descriptors=pool['descriptors'];caseid=pool['case_id']
    require(method in ('Hq','HOmega','R','A'),'known method')
    if method=='A':require(scores is not None and len(scores)==len(descriptors) and np.isfinite(scores).all(),'finite actual model scores')
    def key(d):
        q=(d['global_peak_lower_count'],d['Omega_lower_bound'],d['position']);eligible=(not d['structurally_eligible'],)
        if method=='Hq':return eligible+q
        if method=='HOmega':return eligible+(q[1],q[0],q[2])
        if method=='A':return eligible+(-float(scores[d['position']]),)+q
        raw=[51001,caseid,'R9_candidate_order',d['region_beam_ids'],d['action']]
        return eligible+(hashlib.sha256(json.dumps(raw,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).digest(),d['position'])
    return [d['position'] for d in sorted(descriptors,key=key)]

def choose_anchors(scene,member_ids,k,ledger):
    pindex={aid:i for i,aid in enumerate(scene.aircraft_ids)};ids=tuple(sorted(member_ids));rays=scene.rays[[pindex[aid] for aid in ids]];mean=np.sum(rays,axis=0,dtype=np.float64);norm=float(np.linalg.norm(mean));ledger.tick('mean_input_count',len(ids))
    if norm<1e-12:return None
    ledger.tick('anchor_selection_passes');v=angles(rays,mean/norm,ledger);meanidx=min(range(len(ids)),key=lambda i:(float(v[i]),ids[i]))
    if k==1:return [ids[meanidx]]
    ledger.tick('anchor_selection_passes');v=angles(rays,rays[meanidx],ledger);first=min(range(len(ids)),key=lambda i:(-float(v[i]),ids[i]))
    ledger.tick('anchor_selection_passes');remaining=[i for i in range(len(ids)) if i!=first];v=angles(rays[remaining],rays[first],ledger);second=min(range(len(remaining)),key=lambda j:(-float(v[j]),ids[remaining[j]]));second=remaining[second];selected=[first,second]
    if k==3:
        ledger.tick('anchor_selection_passes');remaining=[i for i in range(len(ids)) if i not in selected];v=angles(rays[remaining,None,:],rays[None,selected,:],ledger);third=min(range(len(remaining)),key=lambda j:(-float(v[j].min()),ids[remaining[j]]));selected.append(remaining[third])
    return sorted(ids[i] for i in selected)

def evaluate_candidate(scene,origin,descriptor):
    """Exactly one independent descriptor attempt; missing resources propagate."""
    began=time.monotonic();n=len(scene.aircraft_ids);ledger=Ledger(n);position=descriptor['position'];base=copy.deepcopy(origin['layout']);original_SHA=digest(origin['layout']);result=dict(position=position,descriptor_SHA=descriptor['descriptor_SHA'],extracted=True,kernel_entered=False,status='known_unavailable',reason=descriptor['unavailable_reasons'],terminal=base,metrics=copy.deepcopy(origin['metrics']),raw_layout=None,raw_metrics=None,origin_SHA=original_SHA)
    def finish(status,reason):
        require(digest(origin['layout'])==original_SHA,'independent origin was modified')
        result.update(status=status,reason=reason,ledger=ledger.record(),observed_candidate_seconds=time.monotonic()-began);return result
    if not descriptor['structurally_eligible']:return finish('known_unavailable',descriptor['unavailable_reasons'])
    result['kernel_entered']=True;member_ids=descriptor['member_ids'];k=descriptor['target_k'];anchors=choose_anchors(scene,member_ids,k,ledger)
    if anchors is None:return finish('rejected','degenerate_mean_ray')
    pindex={aid:i for i,aid in enumerate(scene.aircraft_ids)};members=[pindex[aid] for aid in member_ids];anchorindex=[pindex[aid] for aid in anchors];ledger.pairs(len(members)*k);d2=np.sum((scene.rays[members,None,:]-scene.rays[None,anchorindex,:])**2,axis=-1);adjacency=(d2<=4*np.sin(np.deg2rad(.5)/2)**2+1e-14).tolist();matching=minimum_capacity(tuple(member_ids),tuple(anchors),adjacency,ledger)
    result.update(selected_anchor_ids=anchors,adjacency_edges=int(np.count_nonzero(adjacency)),flow_capacity=matching['capacity'] if matching else None)
    if matching is None:return finish('rejected','fixed_anchor_capacity_infeasible')
    localowners=np.asarray(matching['assignment'],np.int64);newwidths=[]
    for j in range(k):
        required=float(np.rad2deg(4*np.arcsin(min(math.sqrt(float(d2[localowners==j,j].max()))/2,1.))));width=max(.3,float(np.nextafter(required,np.inf)));ledger.tick('cached_member_scans',int(np.count_nonzero(localowners==j)))
        if width>1.:return finish('rejected','envelope_over1_conservative_kernel')
        newwidths.append(width)
    region=set(descriptor['region_beam_ids']);oldids=base['beam_ids'];outside=[j for j,bid in enumerate(oldids) if bid not in region];newids=[json.dumps([origin['identity']['case_id'],'R9',position,aid],ensure_ascii=False,separators=(',',':')) for aid in anchors];ids=[oldids[j] for j in outside]+newids;require(len(set(ids))==len(ids),'new stable beam IDs collision')
    old_to_new={old:j for j,old in enumerate(outside)};owners=np.asarray(base['assignment'],np.int64);newowners=[old_to_new.get(int(owner),-1) for owner in owners]
    for mi,owner in zip(members,localowners):newowners[mi]=len(outside)+int(owner)
    layout=dict(beam_ids=ids,anchor_aircraft_ids=[base['anchor_aircraft_ids'][j] for j in outside]+anchors,centers=[base['centers'][j] for j in outside]+scene.positions[anchorindex].tolist(),widths_deg=[base['widths_deg'][j] for j in outside]+newwidths,assignment=newowners)
    result['raw_layout']=layout
    if omega(layout['widths_deg'])>origin['Omega_cap']+1e-12:return finish('rejected','whole_Omega_cap')
    metrics=audit(scene,layout,origin['Ncap'],origin['Omega_cap'],ledger,audit_reserved=True);result['raw_metrics']=metrics;ledger.tick('scan_input_count',n+len(members)*k)
    # Stable identity rather than compacted integer ordinal defines outside owner.
    for i,oldowner in enumerate(owners):
        if oldids[int(oldowner)] not in region:
            old=int(oldowner);new=newowners[i]
            require(ids[new]==oldids[old] and layout['anchor_aircraft_ids'][new]==base['anchor_aircraft_ids'][old] and layout['centers'][new]==base['centers'][old] and layout['widths_deg'][new]==base['widths_deg'][old],'outside stable owner/anchor/center/width changed')
    if not protected(metrics,origin['metrics']) or not protected(metrics,origin['RULE_metrics']):return finish('rejected','dual_origin_RULE_quality_protection')
    if objective(metrics)>=objective(origin['metrics']):return finish('no_improvement','exact_incumbent_retained_on_tie_or_worse')
    result.update(terminal=layout,metrics=metrics,new_beam_lineage=dict(removed_region_beam_ids=list(descriptor['region_beam_ids']),new_ids=newids))
    return finish('admissible','strict_raw_objective_improvement')

def best_terminal(origin,results):
    best=dict(position=-1,terminal=copy.deepcopy(origin['layout']),metrics=copy.deepcopy(origin['metrics']),status='incumbent')
    for result in results:
        if result['status']=='admissible' and objective(result['metrics'])+(result['position'],)<objective(best['metrics'])+(best['position'],):best=result
    return best

def evaluate_pool(scene,origin,pool,positions=None,stop=None):
    """Common artifact completes in separate30s guard; this batch has60s/2GiB.

    Pool.common_ledger already includes origin.common_ledger; never add them.
    """
    require(pool['origin_SHA']==origin['layout_SHA'] and pool['pool_SHA']==digest(pool['descriptors']),'one frozen common origin/pool')
    positions=list(range(len(pool['descriptors']))) if positions is None else list(positions)
    require(len(positions)==len(set(positions)) and all(isinstance(p,int) and 0<=p<len(pool['descriptors']) for p in positions),'unique original fixed slots')
    results=[]
    for position in positions:
        try:
            if stop is not None and stop():raise ResourceUnknown('native_or_user_pause_boundary')
            results.append(evaluate_candidate(scene,origin,pool['descriptors'][position]))
        except ResourceUnknown as error:
            error.completed_results=results;error.incomplete_position=position;error.remaining_positions=positions[len(results):];raise
    return dict(complete=True,results=results,selected=best_terminal(origin,results),descriptor_extractions=len(results),kernel_entries=sum(x['kernel_entered'] for x in results),new_RULE_D5_T_W_model_calls=0)

MECHANISM_INDICES=(0,6,12,18,24,36,42,48,54,60,72,78)

def mechanism_gate(rows):
    """M from saved completed pool labels only; no re-execution or model."""
    require([r['original_index'] for r in rows]==list(MECHANISM_INDICES),'fixed12 mechanism mothers/order')
    cases=[];fixed=[0]*32;oracle_hits=0;beyond_hits=0
    for row in rows:
        origin=row['origin'];pool=row['pool'];batch=row['batch'];ds=pool['descriptors'];results=batch['results']
        require(batch.get('complete') is True and pool['origin_SHA']==origin['layout_SHA'] and pool['pool_SHA']==digest(ds),'complete pinned shared pool')
        require(len(ds)<=32 and [d['position'] for d in ds]==list(range(len(ds))) and [r['position'] for r in results]==list(range(len(ds))) and batch['descriptor_extractions']==len(ds),'all original descriptor positions, no gap/subset')
        require(all(r['status'] in ('known_unavailable','rejected','no_improvement','admissible') and protected(r['metrics'],origin['metrics']) and protected(r['metrics'],origin['RULE_metrics']) for r in results),'all known protected actual terminal outcomes')
        oracle=best_terminal(origin,results);controls={}
        for method in ('Hq','HOmega'):
            order=method_order(pool,method)[:8];controls[method]=best_terminal(origin,[results[i] for i in order])
        name,reference=min(controls.items(),key=lambda item:objective(item[1]['metrics'])+(item[0],))
        first=hit(oracle['metrics'],origin['metrics']);beyond=hit(oracle['metrics'],reference['metrics']);oracle_hits+=int(first);beyond_hits+=int(beyond)
        for position in range(32):
            if position<len(results):fixed[position]+=int(hit(results[position]['metrics'],origin['metrics']))
        cases.append(dict(original_index=row['original_index'],oracle_position=oracle['position'],oracle_hit_incumbent=first,oracle_hit_better_H=beyond,reference_method=name,H_prefixes={a:method_order(pool,a)[:8] for a in ('Hq','HOmega')},absent_fixed_positions=list(range(len(ds),32))))
    difference=oracle_hits-max(fixed)
    return dict(schema='R9MechanismGateV3',complete_mothers=12,oracle_hit_incumbent=oracle_hits,oracle_hit_better_H=beyond_hits,fixed_position_hits=fixed,oracle_minus_best_fixed_position=difference,passed=oracle_hits>=6 and beyond_hits>=3 and difference>=3,cases=cases,new_kernel_calls=0,new_model_calls=0,automatic_next_stage_allowed=False)
