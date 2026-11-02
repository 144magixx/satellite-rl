"""Independent linear full-owner geometry checks. No flow/producer audit calls.

Scene rays must be independently constructed and source-qualified by admitted
input loader. This module never trusts producer coverage or producer metrics.
"""
import math
import json
import numpy as np
from .persistence import content_sha
from .reader import Incomplete
from .exact_order import layout_evidence,key as exact_key
from .region_v2 import domain as support_domain,select as support_select

def need(condition, message):
    if not condition: raise Incomplete(message)

def verify_flow_proofs(members,anchors,adjacency,queries,capacity=None,assignment=None):
    """Pure original-graph certificate checker; never searches for a matching."""
    need(1<=len(queries)<=8,'bounded complete flow proof ledger')
    free=[a for a in members if a not in anchors];f=len(free);failures=set();successes={}
    lo,hi=math.ceil(len(members)/len(anchors)),len(members)
    for iteration,query in enumerate(queries):
        q=query['q'];need(type(q) is int and 1<=q<=len(members),'query capacity')
        expected=len(members) if iteration==0 else (lo+hi)//2
        need(q==expected and (iteration==0 or lo<hi),'exact bounded binary query order')
        need(type(query.get('feasible')) is bool,'explicit query semantic boolean')
        if query['feasible']:
            observed=query.get('assignment')
            need(isinstance(observed,list) and len(observed)==len(members),'full query assignment proof')
            need(all(type(a) is int and 0<=a<len(anchors) for a in observed),'query owner range')
            need(max(observed.count(j) for j in range(len(anchors)))<=q,'query true capacity')
            need(all(adjacency[i][j] for i,j in enumerate(observed)),'query matching edges')
            need(all(observed[members.index(a)]==j for j,a in enumerate(anchors)),'query pinned owners')
            successes[q]=observed
            if iteration>0:hi=q
        else:
            proof=query.get('proof',{});need(proof.get('kind')=='source_sink_cut_v1','cut proof kind')
            F=proof.get('reachable_free_member_ids',[]);T=proof.get('reachable_anchor_ids',[])
            need(len(F)==len(set(F)) and set(F)<=set(free) and len(T)==len(set(T)) and set(T)<=set(anchors),'cut vertices')
            crossing=sum(adjacency[members.index(a)][j] for a in F for j,b in enumerate(anchors) if b not in T)
            cut=f-len(F)+crossing+len(T)*(q-1)
            need(proof.get('crossing_free_anchor_arcs')==crossing and proof.get('cut_capacity')==cut and proof.get('required_free')==f and cut<f,'strict infeasible cut')
            failures.add(q)
            if iteration==0:need(len(queries)==1,'hi=m infeasible stops immediately')
            else:lo=q+1
    if capacity is None:
        need(len(queries)==1 and len(members) in failures,'complete graph infeasible certificate')
        return True
    need(lo==hi==capacity,'complete capacity search exhausted bounds')
    need(capacity in successes and successes[capacity]==assignment,'final matching certificate')
    need(capacity==math.ceil(len(members)/len(anchors)) or capacity-1 in failures,'minimum capacity certificate')
    return True

def metrics(counts, widths):
    loads = np.asarray(counts, dtype=np.float64) * 21600000
    return dict(N=len(counts), maxload=int(max(counts))*21600000,
                top10_mean=float(np.sort(loads)[-max(1, math.ceil(.1*len(counts))):].mean()),
                CV=float(loads.std()/loads.mean()),
                actual_Omega=float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(np.asarray(widths)/2))))),
                total_request=int(sum(counts))*21600000)

class GeometryAuditor:
    def __init__(self, scene, origin, parent_cache_SHA, current=None,stop=None):
        self.scene, self.origin, self.parent_cache_SHA = scene, origin, parent_cache_SHA
        self.current = current or dict(layout=origin['layout'],metrics=origin['metrics'])
        self.stop=stop
        self.index = {a:i for i,a in enumerate(scene.aircraft_ids)}
        need(len(self.index)==len(scene.aircraft_ids), 'full unique aircraft')
        need(np.all(np.asarray(scene.demands)==21600000), 'all fixed requests')

    def boundary(self):
        if self.stop is not None and self.stop():raise Incomplete('cooperative independent geometry pause')

    def __call__(self, node, cache, allowance):
        self.boundary()
        cache = cache or dict(schema='R12ReaderGeometryCacheV2',parent_cache_SHA=self.parent_cache_SHA, layouts={},exact={},pairs={},regions={},contacts={})
        need(cache.get('schema')=='R12ReaderGeometryCacheV2','versioned independent geometry cache')
        need(cache.get('parent_cache_SHA')==self.parent_cache_SHA, 'geometry parent binding')
        cost=dict(rays=0,raypairs=0,layouts=0)
        region_lookup={};region_matrix=None;examined_layouts=set()
        def chord2(a,b):
            if a in region_lookup and b in region_lookup:
                return region_matrix[region_lookup[a]][region_lookup[b]]
            key=content_sha(sorted([a,b]))
            if key not in cache['pairs']:
                self.boundary()
                cost['raypairs']+=1
                need(cost['raypairs']<=allowance['raypairs'],'pair reservation before geometry')
                cache['pairs'][key]=float(np.sum((self.scene.rays[self.index[a]]-self.scene.rays[self.index[b]])**2))
            return cache['pairs'][key]
        def angle(a,b):
            return float(np.rad2deg(2*np.arcsin(min(math.sqrt(chord2(a,b))/2,1))))
        def layout_audit(layout):
            key=content_sha(layout)
            examined_layouts.add(key)
            if key in cache['layouts']:
                need(key in cache['exact'],'committed exact key accompanies cached geometry')
                return cache['layouts'][key]
            ids=layout['beam_ids']; anchors=layout['anchor_aircraft_ids']; b=len(ids)
            owners=np.asarray(layout['assignment']); widths=np.asarray(layout['widths_deg'],dtype=float)
            n=len(self.index)
            need(1<=b<=min(200,self.origin['Ncap'],n) and len(set(ids))==b, 'actual N')
            need(len(anchors)==b and all(a in self.index for a in anchors), 'anchor IDs')
            need(owners.shape==(n,) and np.isfinite(owners).all() and np.equal(owners,np.floor(owners)).all(), 'full integer owners')
            owners=owners.astype(np.int64)
            need(np.all((owners>=0)&(owners<b)), 'owner range')
            need(widths.shape==(b,) and np.isfinite(widths).all() and np.all((widths>=.3)&(widths<=1)), 'width envelope domain')
            ai=[self.index[a] for a in anchors]
            need(np.array_equal(np.asarray(layout['centers']),np.asarray(self.scene.positions)[ai]), 'unchanged anchor positions')
            counts=np.bincount(owners,minlength=b);need(np.all(counts>0),'nonempty')
            self.boundary()
            cost['raypairs']+=n;cost['layouts']+=1
            need(all(cost[k]<=allowance[k] for k in cost), 'prospective allowance before geometry')
            chord=np.sum((self.scene.rays-self.scene.rays[ai][owners])**2,axis=1)
            need(np.all(chord<=4*np.sin(np.deg2rad(widths[owners]/2)/2)**2+1e-14),'all aircraft coverage')
            for j in range(b):
                expected=max(.3,float(np.nextafter(np.rad2deg(4*np.arcsin(min(math.sqrt(float(chord[owners==j].max()))/2,1.))),np.inf)))
                need(abs(widths[j]-expected)<=1e-12,'actual member envelope')
            m=metrics(counts,widths)
            need(m['actual_Omega']<=self.origin['Omega_cap']+1e-12,'whole Omega')
            need(m['total_request']==n*21600000,'request conservation')
            cache['layouts'][key]=m
            self.boundary()
            cache['exact'][key]=layout_evidence(layout)
            return m
        current=self.current['layout'];ids=current['beam_ids'];owners=current['assignment']
        parent_metrics=layout_audit(current)
        for key,value in parent_metrics.items():
            tolerance=0 if key in ('N','maxload','total_request') else (1e-12 if key=='actual_Omega' else 4*abs(float(np.spacing(value))))
            need(abs(value-self.current['metrics'][key])<=tolerance,'independent parent/B metric '+key)
        counts=np.bincount(owners,minlength=len(ids)).tolist()
        d=node['descriptor'];k=d['target_k']
        need(d.get('current_layout_SHA')==content_sha(current),'parent layout binding')
        beams=[]
        for j in sorted(range(len(ids)),key=lambda j:ids[j]):
            beams.append([ids[j],current['anchor_aircraft_ids'][j],float(current['centers'][j][0]).hex(),
                          float(current['centers'][j][1]).hex(),float(current['widths_deg'][j]).hex(),
                          sorted(a for a,o in zip(self.scene.aircraft_ids,owners) if o==j)])
        state=content_sha(dict(case_id=self.origin['identity']['case_id'],beams=beams))
        need(d['state_SHA']==state,'physical parent state')
        peaks,candidates=support_domain(counts)
        contact_key=state
        contact=cache['contacts'].get(contact_key)
        if contact is None:
            self.boundary()
            total=len(peaks)*len(candidates)
            cost['raypairs']+=total
            need(total<=5100 and cost['raypairs']<=allowance['raypairs'],'shared parent contact allowance before geometry')
            pairs=[]
            for i in peaks:
                for j in candidates:
                    a=self.scene.rays[self.index[current['anchor_aircraft_ids'][i]]]
                    b=self.scene.rays[self.index[current['anchor_aircraft_ids'][j]]]
                    squared=float(np.sum((a-b)**2))
                    angle_value=float(np.rad2deg(2*np.arcsin(min(math.sqrt(squared)/2,1.))))
                    pairs.append(dict(peak_beam_id=ids[i],support_beam_id=ids[j],chord2=squared,angle=angle_value))
            contact=dict(pairs=pairs,charged_raypairs=total)
            cache['contacts'][contact_key]=contact
        expected_pairs={(v['peak_beam_id'],v['support_beam_id']):(v['chord2'],v['angle']) for v in contact['pairs']}
        need(set(expected_pairs)=={(ids[i],ids[j]) for i in peaks for j in candidates},'complete same-parent contact cache IDs')
        rows={j:[expected_pairs[ids[i],ids[j]] for i in peaks] for j in candidates}
        threshold=float(4*np.sin(np.deg2rad(.5)/2)**2+1e-14)
        selection=support_select(counts,ids,rows,threshold)
        region=selection['region'] or sorted(peaks,key=lambda j:ids[j])
        regionids=[ids[j] for j in region]
        members=sorted(a for a,o in zip(self.scene.aircraft_ids,owners) if o in region)
        r,m=len(region),len(members)
        need(d.get('mode')=='production_contact_slack_v2','versioned production descriptor mode')
        need(d.get('unsupported_reason')==selection['reason'] and d.get('support_beam_id')==(None if selection['support'] is None else ids[selection['support']]),'exact contacting lower-load support')
        need(d.get('peak_beam_ids')==sorted(ids[j] for j in peaks),'complete peak set never truncated')
        observed=d.get('contact_cache',{})
        reported={(v['peak_beam_id'],v['support_beam_id']):(v['chord2'],v['angle']) for v in observed.get('pairs',[])}
        need(reported==expected_pairs and len(observed.get('pairs',[]))==len(expected_pairs) and observed.get('charged_raypairs')==len(expected_pairs),'independent full original-center contact proof')
        necessary=None if selection['support'] is None else counts[selection['support']]<=max(counts)-len(peaks)-1
        need(d.get('immediate_peak_reduction_necessary')==necessary,'diagnostic necessary count condition only')
        need(d['region_beam_ids']==regionids and d['member_ids']==members,'all exact current region members')
        need(k==len(peaks)+1+(d['action']=='J1') and d['action'] in ('J0','J1'),'two exact support-region actions')
        semantic=content_sha([state,members,k,'R12-contact-slack-exact-order-v2'])
        need(node['semantic_SHA']==d['semantic_SHA']==semantic,'complete v2 semantic identity')
        need(d['descriptor_SHA']==content_sha({a:b for a,b in d.items() if a!='descriptor_SHA'}),'descriptor content')
        outside=[j for j in range(len(ids)) if j not in region]
        lower=metrics([1]*max(1,len(outside)),[current['widths_deg'][j] for j in outside])['actual_Omega'] if outside else 0.
        floor=metrics([1],[.3])['actual_Omega'];lower+=k*floor
        reasons=[] if selection['reason'] is None else [selection['reason']]
        if not 1<=r<=31:reasons.append('region_above31')
        if m>128:reasons.append('region_above128')
        if not 1<=k<=32 or k>m:reasons.append('members_below_target_or_k_bound')
        if len(outside)+k>min(200,self.origin['Ncap']):reasons.append('target_Ncap')
        if lower>self.origin['Omega_cap']+1e-12:reasons.append('floor_Omega_cap')
        need(d['unavailable_reasons']==reasons and d['structurally_eligible']==(not reasons),'analytic availability')
        qB=self.origin['metrics']['maxload']//21600000-1
        a,rem=divmod(m,k)
        expected_rank=[max([int(counts[j]) for j in outside]+[(m+k-1)//k]),
            -(sum(max(int(counts[j])-qB,0) for j in region)-max(0,m-k*qB)),
            sum(int(counts[j])**2 for j in outside)+(k-rem)*a*a+rem*(a+1)**2,
            lower,len(outside)+k,d['action']]
        need(d['rank']==expected_rank and d['target_N']==len(outside)+k,'independent preoutcome H rank')
        if reasons:
            need(node['state']=='known_unavailable' and node['terminal']==current and node.get('kernel_entered') is False and node['ledger']['flow_queries']==0,'known unavailable true fallback without solve')
        else:
            # Independently reproduce the finite deterministic anchor recipe.
            rays=self.scene.rays[[self.index[a] for a in members]]
            region_key=content_sha([d['state_SHA'],members])
            cache.setdefault('regions',{})
            if region_key not in cache['regions']:
                self.boundary()
                cost['raypairs']+=m*m
                need(cost['raypairs']<=allowance['raypairs'],'full regional matrix allowance before geometry')
                matrix=np.sum((rays[:,None,:]-rays[None,:,:])**2,axis=-1)
                cache['regions'][region_key]=dict(member_ids=members,chord2=matrix.tolist())
            region_record=cache['regions'][region_key]
            need(region_record['member_ids']==members,'same physical parent full regional cache')
            region_matrix=region_record['chord2'];region_lookup={a:i for i,a in enumerate(members)}
            mean=np.sum(rays,axis=0,dtype=np.float64);norm=float(np.linalg.norm(mean))
            if norm<1e-12:
                need(node['state']=='rejected' and node['raw_producer_label']=='degenerate_mean_ray' and node['terminal']==current,'degenerate fallback')
            else:
                self.boundary()
                cost['raypairs']+=m
                need(cost['raypairs']<=allowance['raypairs'],'mean-angle allowance')
                v=np.rad2deg(2*np.arcsin(np.minimum(np.linalg.norm(rays-mean/norm,axis=1)/2,1)))
                meanid=min(range(m),key=lambda i:(float(v[i]),members[i]))
                if k==1:selected=[members[meanid]]
                else:
                    first=min(members,key=lambda a:(-angle(a,members[meanid]),a))
                    second=min((a for a in members if a!=first),key=lambda a:(-angle(a,first),a))
                    selected=[first,second]
                    nearest={a:min(angle(a,first),angle(a,second)) for a in members if a not in selected} if k>=3 else {}
                    while len(selected)<k:
                        chosen=min(nearest,key=lambda a:(-nearest[a],a));selected.append(chosen);del nearest[chosen]
                        if len(selected)<k:
                            nearest={a:min(v,angle(a,chosen)) for a,v in nearest.items()}
                selected=sorted(selected)
                need(node.get('selected_anchor_ids')==selected,'independent farthest-first anchors')
                adjacency=[[chord2(a,b)<=4*np.sin(np.deg2rad(.5)/2)**2+1e-14 for b in selected] for a in members]
                adjacency=[[bool(x) for x in row] for row in adjacency]
                need(node.get('raw_adjacency')==adjacency,'full independent raw adjacency')
                queries=node['ledger'].get('flow_results',[]);graphs=node['ledger'].get('flow_graphs',[])
                need(node['ledger'].get('flow_queries')==len(queries)==len(graphs),'exact flow query/result/graph count')
                free=[a for a in members if a not in selected]
                edges=sum(sum(adjacency[members.index(a)]) for a in free)
                for query,graph in zip(queries,graphs):
                    need(graph.get('q')==query['q'] and graph.get('nodes')==m+2 and graph.get('residual_arcs')==2*(len(free)+edges+len(selected)),'full original graph size ledger')
                if node['state']=='graph_infeasible':
                    need(any(not any(adjacency[i]) for i,a in enumerate(members) if a not in selected),'hi=m actual infeasibility witness')
                    verify_flow_proofs(members,selected,adjacency,queries)
        result=layout_audit(node['terminal'])
        terminal_exact=cache['exact'][content_sha(node['terminal'])]
        need(node.get('exact_evidence')==terminal_exact,'independent exact terminal count/rationalTail/NU/Omega key')
        for k in result:
            tolerance=0 if k in ('N','maxload','total_request') else (1e-12 if k=='actual_Omega' else 4*abs(float(np.spacing(result[k]))))
            need(abs(result[k]-node['metrics'][k])<=tolerance,'independent metric '+k)
        if node.get('raw_layout') is not None and node.get('raw_metrics') is not None:
            raw=layout_audit(node['raw_layout'])
            need(node.get('raw_exact_evidence')==cache['exact'][content_sha(node['raw_layout'])],'independent exact raw evidence')
            need(all(abs(raw[k]-node['raw_metrics'][k]) <= (1e-12+4*abs(float(np.spacing(raw[k])))) for k in raw),'raw independent metrics')
        match=node.get('raw_local_matching')
        if match is not None:
            members=match['member_ids'];anchors=match['anchor_ids'];assignment=match['assignment']
            need(members==node['descriptor']['member_ids'] and len(assignment)==len(members),'full raw matching')
            need(anchors==selected and len(anchors)==node['descriptor']['target_k'],'exact independently selected matching anchors')
            need(len(set(anchors))==len(anchors) and all(a in members for a in anchors),'raw anchor members')
            need(all(type(a) is int and 0<=a<len(anchors) for a in assignment),'raw owner range')
            need(set(assignment)==set(range(len(anchors))),'raw nonempty')
            for j,a in enumerate(anchors):need(assignment[members.index(a)]==j,'anchor self ownership')
            need(all(node['raw_adjacency'][i][j] for i,j in enumerate(assignment)),'all matched edges valid')
            # Every failed capacity is certified by a strict original-graph cut;
            # reader does not run BFS/flow or trust a producer feasibility label.
            verify_flow_proofs(members,anchors,node['raw_adjacency'],node['ledger'].get('flow_results',[]),node['flow_capacity'],assignment)
            widths=[]
            for j,b in enumerate(anchors):
                largest=max(chord2(a,b) for a,o in zip(members,assignment) if o==j)
                widths.append(max(.3,float(np.nextafter(np.rad2deg(4*np.arcsin(min(math.sqrt(largest)/2,1))),np.inf))))
            if max(widths)>1:
                need(node['raw_producer_label']=='envelope_over1_conservative_kernel' and node['raw_layout'] is None,'true overwidth rejection')
            else:
                newids=[json.dumps([self.origin['identity']['case_id'],'R12',semantic,a],ensure_ascii=False,separators=(',',':')) for a in anchors]
                mapping={old:j for j,old in enumerate(outside)}
                expected_owners=[mapping.get(int(o),-1) for o in owners]
                for a,o in zip(members,assignment):expected_owners[self.index[a]]=len(outside)+o
                expected=dict(beam_ids=[ids[j] for j in outside]+newids,
                    anchor_aircraft_ids=[current['anchor_aircraft_ids'][j] for j in outside]+anchors,
                    centers=[current['centers'][j] for j in outside]+np.asarray(self.scene.positions)[[self.index[a] for a in anchors]].tolist(),
                    widths_deg=[current['widths_deg'][j] for j in outside]+widths,assignment=expected_owners)
                need(node['raw_layout']==expected,'complete raw layout construction and unchanged outside')
                raw_counts=np.bincount(expected_owners,minlength=len(expected['beam_ids']))
                raw=metrics(raw_counts,expected['widths_deg'])
                def protect(a,b):return all(a[x]<=b[x]+(1e-9 if x=='CV' else 1e-6)+1e-12*abs(b[x]) for x in ('maxload','top10_mean','CV'))
                if raw['actual_Omega']>self.origin['Omega_cap']+1e-12:
                    need(node['raw_producer_label']=='whole_Omega_cap' and node['state']=='rejected','actual wholeOmega rejection')
                elif not protect(raw,self.current['metrics']) or not protect(raw,self.origin['metrics']):
                    need(node['state']=='rejected' and node['raw_producer_label']=='dual_current_B_quality_protection','exact dual protection rejection')
                elif exact_key(cache['exact'][content_sha(expected)])>=exact_key(cache['exact'][content_sha(current)]):
                    need(node['state']=='no_improvement','exact no-improvement state')
                else:need(node['state']=='admissible','exact admissible state')
        if node['state']!='admissible':need(node['terminal']==current,'rejected fallback must equal parent')
        if node['state']=='admissible':
            def protected(a,b):
                return all(a[k]<=b[k]+(1e-9 if k=='CV' else 1e-6)+1e-12*abs(b[k]) for k in ('maxload','top10_mean','CV'))
            need(node['terminal']==node['raw_layout'],'accepted raw layout')
            need(protected(result,self.current['metrics']) and protected(result,self.origin['metrics']),'dual full quality protection')
            need(exact_key(terminal_exact)<exact_key(cache['exact'][content_sha(current)]),'strict exact current improvement')
        return dict(passed=True,geometry_verified=True,identity_verified=node.get('identity')==self.origin['identity'],
                    state_acceptance_verified=True,
                    independent_graph_proof=('analytic_unavailable' if reasons else
                        'hi_m_isolated_free_member' if node['state']=='graph_infeasible' else
                        'matching_and_minimum_capacity_cut' if match is not None else 'degenerate_mean'),
                    producer_raw_label=node['raw_producer_label'],producer_proof_qualification=node['proof_qualification'],
                    validated_layout_SHA_metrics={k:cache['layouts'][k] for k in examined_layouts},
                    validated_layout_SHA_exact={k:cache['exact'][k] for k in examined_layouts},terminal_exact=terminal_exact,
                    actual_cost=cost,terminal_metrics=result),cache
