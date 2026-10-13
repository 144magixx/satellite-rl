"""R5 P0 fixed-denominator gates; consumes independent audit rows only."""
from functools import cmp_to_key
from scripts.stage1_ai_search_r4.gates import finite, legal, within, target_hit, compare

D5_METHODS=('RULE','RANDOM','EXACT','OWNER10','JOINT10')
D4_METHODS=('OWNER10','JOINT10')

def evaluate_gates(cases,expected_case_ids,engineering_g0=False):
    ids=list(expected_case_ids)
    if len(ids)!=24 or len(set(ids))!=24: raise ValueError('exactly24_prespecified_unique_cases_required')
    by={c['case_id']:c for c in cases}
    if len(by)!=len(cases) or set(by)-set(ids): raise ValueError('duplicate_or_unregistered_case')
    complete=set(by)==set(ids) and all(c.get('complete_slots') for c in cases)
    g0=bool(engineering_g0 and complete)
    target={m:0 for m in ('EXACT','OWNER10','JOINT10')}
    quality={m:0 for m in D4_METHODS};gains=losses=0;changed=set();details=[]
    for cid in ids:
        c=by.get(cid,{});d5=c.get('D5',{});d4=c.get('D4',{});teacher=c.get('teacher')
        g0=g0 and bool(c.get('audit_closed') and c.get('witness_valid'))
        hits={m:target_hit(d5.get(m),teacher,True) for m in target}
        for m,h in hits.items(): target[m]+=int(h)
        qs={m:False for m in D4_METHODS};best_name=None
        if all(legal(d5.get(m)) for m in ('RULE','RANDOM')):
            best_name,best=sorted([(m,d5[m]) for m in ('RULE','RANDOM')],key=cmp_to_key(compare))[0]
            bm=best['metrics']
            for m in D4_METHODS:
                row=d5.get(m);cm=(row or {}).get('metrics',{})
                qs[m]=bool(legal(row) and row.get('status')=='success' and bm['maxload']>0 and cm['maxload']<=.99*bm['maxload']+1e-6+1e-12*abs(bm['maxload']) and all(within(cm[k],bm[k],1.02,k) for k in ('top10_mean','CV')))
                quality[m]+=int(qs[m])
        paired={}
        for m in D4_METHODS:
            old=target_hit(d4.get(m),teacher,True);new=hits[m]
            # Missing/nonpublished rows cannot become artificial repair gains.
            pair_legal=legal(d4.get(m)) and legal(d5.get(m))
            gain=bool(pair_legal and new and not old);loss=bool(pair_legal and old and not new)
            gains+=int(gain);losses+=int(loss);paired[m]={'D4_target':old,'D5_target':new,'pair_complete':pair_legal,'gain':gain,'loss':loss}
        for m in target:
            row=d5.get(m,{})
            mechanism=row.get('mechanisms',{})
            if legal(row) and row.get('status')=='success' and mechanism.get('N_changed') and mechanism.get('width_distribution_changed') and mechanism.get('nonfloor_heterogeneous_widths'): changed.add(cid)
        details.append({'case_id':cid,'target':hits,'quality':qs,'best_baseline':best_name,'paired':paired})
    conditions={'exact':target['EXACT']>=22,'owner':target['OWNER10']>=18,'joint':target['JOINT10']>=18,'quality':all(quality[m]>=6 for m in quality),'repair_effect':gains-losses>=6 and losses<=2,'capability':len(changed)>=6}
    return {'schema':'R5P0IndependentGatesV1','status':'complete' if complete else 'partial','denominator':24,'paired_denominator':48,'G0':bool(g0),'conditions':conditions,'pass':bool(complete and g0 and all(conditions.values())),'target_counts':target,'quality_counts':quality,'paired_gains':gains,'paired_losses':losses,'paired_net_gain':gains-losses,'capability_cases':sorted(changed),'cases':details,'efficiency_gate':False,'limitations':['Repeated development; no independent test or AI contribution claim','Missing or unpublished rows retained; no dropped cases or zero-filled metrics','Historical R4 D4 rows excluded from strict timing comparison']}
