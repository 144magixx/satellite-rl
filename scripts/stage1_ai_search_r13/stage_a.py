"""Scalar reconstruction of already independently audited R10 SH-C prefixes.

No geometry, feasibility solver, metric derivation, model or training calls.
"""
import hashlib
import json
import math

INDICES=(0,6,12,18,24,36,42,48,54,60,72,78)
CAPS=(0,4,8,16,32)
WINDOWS=(2,4,8)
RECIPE='4a82bf0889c35e62d27d695b35c9a85e86491409'
DESIGN='47cb6befc27519e18c8da8cf07513299d84c04b109db9aae6982d273febbb498'
FIELDS=('maxload','top10_mean','CV','N','actual_Omega','total_request')

class Unknown(ValueError):pass

def need(value,reason):
    if not value:raise Unknown(reason)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()

def parse(raw):
    def pairs(values):
        result={}
        for key,value in values:
            need(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(Unknown('nonfinite JSON')))

def metrics(value):
    need(isinstance(value,dict) and all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) for k in FIELDS),'complete finite saved scalar metrics')
    need(type(value['N']) is int and 1<=value['N']<=200 and all(value[k]>=0 for k in FIELDS),'saved scalar domains')
    return {k:value[k] for k in FIELDS}

def objective(m):return tuple(m[k] for k in ('maxload','top10_mean','CV','N','actual_Omega'))

def protected(a,b):
    return all(a[k]<=b[k]+(1e-9 if k=='CV' else 1e-6)+1e-12*abs(b[k]) for k in ('maxload','top10_mean','CV'))

def quality_preserving(a,b):
    """Own full SH-C teacher; frozen strict tolerances, no1% relaxation."""
    a=metrics(a);b=metrics(b)
    return protected(a,b) and a['N']<=b['N'] and a['actual_Omega']<=b['actual_Omega']+1e-12

def state_sha(identity,aircraft_ids,layout):
    # Exact source identity serialization only, not metric/coverage arithmetic.
    groups=[[] for _ in layout['beam_ids']]
    for aid,owner in zip(aircraft_ids,layout['assignment']):groups[owner].append(aid)
    return sha(dict(case_id=identity['case_id'],beams=[[layout['beam_ids'][j],layout['anchor_aircraft_ids'][j],
        float(layout['centers'][j][0]).hex(),float(layout['centers'][j][1]).hex(),float(layout['widths_deg'][j]).hex(),sorted(groups[j])]
        for j in sorted(range(len(groups)),key=lambda j:layout['beam_ids'][j])]))

def structural_layout(layout,static,origin,saved):
    """Structural identity and saved bounds only; old certificate supplies GEO."""
    need(isinstance(layout,dict) and all(k in layout for k in ('beam_ids','anchor_aircraft_ids','centers','widths_deg','assignment')),'full stored physical layout')
    ids=layout['beam_ids'];n=len(static['aircraft_ids']);b=len(ids)
    need(b==saved['N'] and 1<=b<=min(n,origin['Ncap'],200) and len(set(ids))==b,'stored actual N and stable beam identities')
    need(all(type(x) is str and x for x in ids),'stable beam IDs')
    need(all(len(layout[k])==b for k in ('anchor_aircraft_ids','centers','widths_deg')),'stored full beam arrays')
    pindex={a:i for i,a in enumerate(static['aircraft_ids'])}
    need(all(a in pindex for a in layout['anchor_aircraft_ids']),'original aircraft anchors')
    need(layout['centers']==[static['positions'][pindex[a]] for a in layout['anchor_aircraft_ids']],'stored center/source identity')
    owners=layout['assignment'];need(len(owners)==n and all(type(j) is int and 0<=j<b for j in owners) and set(owners)==set(range(b)),'all original aircraft unique nonempty ownership')
    need(all(type(w) in (int,float) and math.isfinite(w) and .3<=w<=1. for w in layout['widths_deg']),'saved width envelope')
    need(saved['actual_Omega']<=origin['Omega_cap']+1e-12 and saved['total_request']==n*21600000,'saved Omega/request scalar constraints')

def policy_stops(statuses):
    total=len(statuses);result={'full':total}
    for cap in CAPS:result['cap_'+str(cap)]=min(cap,total)
    for window in WINDOWS:
        streak=0;position=total
        for t,status in enumerate(statuses,1):
            streak=0 if status=='admissible' else streak+1
            if streak>=window:position=t;break
        result['window_'+str(window)]=position
    return result

def analyze_case(common_wrapper,method_wrapper,common_ref,method_ref,certificate_ref,stop=lambda:False):
    need(not stop(),'stop before mother')
    c=common_wrapper.get('common',{});result=method_wrapper.get('result',{});static=common_wrapper.get('static',{});origin=c.get('origin',{});identity=origin.get('identity',{})
    index=common_wrapper.get('original_index')
    need(common_wrapper.get('actual_stage')=='common' and method_wrapper.get('actual_stage')=='SH-C','original wrapper stage slots')
    need(common_wrapper.get('run_id')==method_wrapper.get('run_id') and type(common_wrapper.get('run_id')) is str,'same original run identity')
    need(common_wrapper.get('source_input_ref')==method_wrapper.get('source_input_ref'),'same original source input reference')
    need(index in INDICES and method_wrapper.get('original_index')==index,'same exact original mother')
    need(c.get('schema')=='R10CommonV2' and c.get('original_index')==index and c.get('design_sha256')==DESIGN,'exact existing common design')
    need(c.get('identity')==identity==common_wrapper.get('identity')==method_wrapper.get('identity')==result.get('identity'),'common/method original identity chain')
    need(identity.get('role')=='train' and identity.get('profile')=='uniform21600000' and static.get('profile')=='europe_v1','examined uniform-request development role')
    aids=static.get('aircraft_ids',[]);need(len(aids)>0 and len(aids)==len(set(aids)) and all(type(a) is str and a for a in aids),'all original aircraft stable IDs')
    need(len(static.get('positions',[]))==len(aids) and static.get('demands')==[21600000]*len(aids) and static.get('input_sha256')==identity.get('input_sha256'),'exact full original input/request scale')
    need(c.get('common_SHA')==sha({k:v for k,v in c.items() if k!='common_SHA'}) and c.get('origin_SHA')==origin.get('layout_SHA')==sha(origin['layout']),'common/origin content identity')
    need(method_wrapper.get('common_ref')==common_ref,'method exact selected common reference')
    need(result.get('schema')=='R10MethodV2' and result.get('method')=='SH-C' and result.get('complete') is True and result.get('partial_status') is None,'own complete SH-C teacher only')
    need(result.get('original_index')==index and result.get('common_SHA')==c['common_SHA'] and result.get('origin_SHA')==c['origin_SHA'] and result.get('q')==c['q'] and result.get('common_ledger')==c['common_ledger'],'same complete accepted source/common contract')
    need(result.get('new_RULE_D5_T_W_model_calls')==0,'no substituted producer route')
    trials=result.get('results');screens=result.get('screenings')
    need(isinstance(trials,list) and len(trials)<=32 and isinstance(screens,list) and len(screens)<=26400,'finite stored records')
    need(result.get('kernel_attempts')==len(trials) and result.get('termination') in ('no_progress_pool_exhausted','kernel_attempt_limit'),'complete natural or32 termination')
    need((result['termination']=='kernel_attempt_limit')==(len(trials)==32),'exact full endpoint termination')
    current=origin['layout'];m=metrics(origin['metrics']);structural_layout(current,static,origin,m)
    teacher=metrics(result['selected']['metrics']);prefixes=[];statuses=[];attempt=0;poolkeys=set();seen=set();flows=0;accepted_trial=None;screen_paid=0
    def append_prefix(t,status,semantic=None,descriptor=None):
        need(not stop(),'stop before prefix publication')
        prefixes.append(dict(original_index=index,prefix_index=t,completed_kernel_attempts=t,last_attempt_status=status,
            last_accepted_trial_zero_based=accepted_trial,layout_SHA=sha(current),metrics=dict(m),quality_preserving_own_teacher=quality_preserving(m,teacher),
            input_sha256=identity['input_sha256'],identity=identity,common_ref=common_ref,SHC_ref=method_ref,
            existing_full12x5_certificate_ref=certificate_ref,qualification='source-covered existing independent full12x5 audit; no new GEO',
            lineage=dict(previous_prefix=None if t==0 else t-1,semantic_SHA=semantic,descriptor_SHA=descriptor,retains_previous=t>0 and status!='admissible'),
            screens_already_paid=screen_paid,inner_flow_queries_proxy_paid=flows,pool_cost_attribution='not recorded',wall_time_savings_seconds=None))
    append_prefix(0,'original_B')
    for screen in screens:
        need(not stop(),'stop before stored descriptor')
        need(attempt<32,'no stored screening after32th completed kernel')
        d=screen['descriptor'];screen_paid+=1
        need(d.get('descriptor_SHA')==sha({k:v for k,v in d.items() if k!='descriptor_SHA'}),'saved descriptor exact content')
        state=state_sha(identity,aids,current)
        need(d.get('state_SHA')==state and d.get('semantic_SHA')==sha([state,d['region_beam_ids'],d['target_k'],RECIPE]),'descriptor/current semantic lineage')
        region=d['region_beam_ids'];need(region==sorted(set(region)) and set(region)<=set(current['beam_ids']),'descriptor exact current region IDs')
        memberids=sorted(aids[i] for i,j in enumerate(current['assignment']) if current['beam_ids'][j] in region)
        need(d.get('member_ids')==memberids,'descriptor current ownership region identity')
        key=d['semantic_SHA'];eligible=d.get('structurally_eligible')
        need(type(eligible) is bool,'typed eligibility')
        expected='known_unavailable' if not eligible else 'duplicate_semantics' if key in poolkeys else 'already_seen' if key in seen else 'kernel_attempted'
        need(screen.get('screening')==expected,'original skip/attempt chronology')
        poolkeys.add(key)
        if expected!='kernel_attempted':continue
        need(attempt<len(trials),'missing completed trial makes whole mother unknown')
        seen.add(key);trial=trials[attempt];status=trial.get('status')
        need(trial.get('attempt')==attempt and trial.get('semantic_SHA')==key and trial.get('kernel_entered') is True,'original trial ordinal/semantic entry')
        need(status in ('admissible','rejected','no_improvement'),'unknown/unclosed trial never a target')
        charge=trial.get('ledger',{}).get('flow_queries');need(type(charge) is int and 0<=charge<=8,'saved per-kernel proxy cost')
        flows+=charge
        if status=='admissible':
            candidate=trial.get('terminal');next_m=metrics(trial.get('metrics'))
            need(candidate==trial.get('raw_layout') and trial['metrics']==trial.get('raw_metrics'),'accepted exact saved terminal/raw binding')
            structural_layout(candidate,static,origin,next_m)
            need(protected(next_m,m) and protected(next_m,origin['metrics']) and objective(next_m)<objective(m),'frozen scalar acceptance protection and strict improvement')
            anchors=trial['selected_anchor_ids']
            newids=[json.dumps([identity['case_id'],'R10_seq',state,key,a],ensure_ascii=False,separators=(',',':')) for a in anchors]
            need(trial.get('new_beam_lineage')==dict(removed_region_beam_ids=region,new_ids=newids),'saved accepted exact semantic beam lineage')
            need(set(candidate['beam_ids'])==(set(current['beam_ids'])-set(region))|set(newids),'saved accepted beam set inheritance')
            current=candidate;m=next_m;accepted_trial=attempt;poolkeys=set()
        statuses.append(status);attempt+=1;append_prefix(attempt,status,key,d['descriptor_SHA'])
    need(attempt==len(trials),'every saved completed trial consumed')
    need(result['selected'].get('terminal')==current and result['selected'].get('metrics')==m,'own full SH-C endpoint equals exact incumbent history')
    need(prefixes[-1]['quality_preserving_own_teacher'] is True,'full teacher self qualification')
    stops=policy_stops(statuses)
    policies=[dict(policy=name,stop_prefix=t,safe=prefixes[t]['quality_preserving_own_teacher'],saved_attempts=len(trials)-t) for name,t in stops.items()]
    first=next(p['prefix_index'] for p in prefixes if p['quality_preserving_own_teacher'])
    envelope=min(p['stop_prefix'] for p in policies if p['policy']!='full' and p['safe'])
    return dict(original_index=index,complete=True,aircraft_count=len(aids),full_attempts=len(trials),first_safe_prefix=first,strong_fixed_envelope=envelope,
        prefix_rows=prefixes,policies=policies,screens=len(screens),instrumentation_missing=['per-attempt/pool times','cold B historical generation','prefix feature/model/exit costs'],actual_acceleration_measured=False)

def opportunity_gate(rows,instrumentable):
    need([r['original_index'] for r in rows]==list(INDICES) and all(r.get('complete') is True for r in rows),'all12 fully qualified, unknown cannot become a gate negative')
    safe_early=sum(r['full_attempts']-r['first_safe_prefix']>=8 for r in rows)
    unsafe_mothers=sum(any(not p['quality_preserving_own_teacher'] for p in r['prefix_rows']) for r in rows)
    safe_mothers=sum(any(p['quality_preserving_own_teacher'] for p in r['prefix_rows']) for r in rows)
    oracle=sum(r['first_safe_prefix'] for r in rows);envelope=sum(r['strong_fixed_envelope'] for r in rows)
    additional=sum(r['strong_fixed_envelope']-r['first_safe_prefix']>=4 for r in rows)
    conditions=[True,safe_early>=3,len({r['first_safe_prefix'] for r in rows})>=2 and unsafe_mothers>=3 and safe_mothers>=3,
                envelope>0 and oracle<=.8*envelope and additional>=3,instrumentable is True]
    return dict(schema='R13FrozenStageAGateV1',conditions=conditions,passed=all(conditions),denominator=12,
        safe_at_least8_earlier_mothers=safe_early,unsafe_label_mothers=unsafe_mothers,safe_label_mothers=safe_mothers,
        aggregate_oracle_attempts=oracle,aggregate_strong_fixed_envelope_attempts=envelope,
        aggregate_proxy_saving_fraction=None if envelope==0 else 1-oracle/envelope,
        at_least4_beyond_envelope_mothers=additional,wall_acceleration_claim=False,training_authorized=False)
