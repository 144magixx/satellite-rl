"""Frozen v2 pretraining gates over independently audited, prespecified cases."""
import math
from functools import cmp_to_key
import numpy as np

ORDER=('RULE','RANDOM','FULL')
TOL={'maxload':1e-6,'top10_mean':1e-6,'CV':1e-9,'N':0,'actual_Omega':1e-12}

def _compare(a,b,keys):
    for k in keys:
        x,y=a[k],b[k];tol=TOL[k]+1e-12*max(abs(x),abs(y)) if k!='N' else 0
        if x<y-tol:return -1
        if x>y+tol:return 1
    return 0

def choose_external(entries,ncap,omega_cap):
    allowed=('R2_common_parent','R2_D0_final_primary','R2_D1_final_primary')
    candidates=[e for e in entries if e.get('source') in allowed and e.get('valid') and e.get('metrics') and all(e['metrics'].get(k) is not None for k in ('maxload','top10_mean','CV')) and e['metrics']['N']<=ncap and e['metrics']['actual_Omega']<=omega_cap+1e-12]
    def compare(a,b):
        c=_compare(a['metrics'],b['metrics'],('maxload','top10_mean','CV','N','actual_Omega'))
        return c or ((allowed.index(a['source'])>allowed.index(b['source']))-(allowed.index(a['source'])<allowed.index(b['source']))) or ((a.get('layout_sha256','')>b.get('layout_sha256',''))-(a.get('layout_sha256','')<b.get('layout_sha256','')))
    return sorted(candidates,key=cmp_to_key(compare))[0] if candidates else None

def _legal(row):
    return bool(row and row.get('valid') and row.get('published') and row.get('publication_audited') and row.get('metrics') and all(row['metrics'].get(k) is not None for k in ('maxload','top10_mean','CV')))

def _within(value,reference,factor,key):
    return value<=factor*reference+TOL[key]+1e-12*abs(reference)

def target_hit(row,reference):
    return _legal(row) and all(_within(row['metrics'][k],reference['metrics'][k],f,k) for k,f in (('maxload',1.01),('top10_mean',1.02),('CV',1.02)))

def evaluate_gates(cases,expected_case_ids,engineering_g0=False):
    """Missing rows stay in denominator. REF unavailable makes efficiency fail."""
    ids=list(expected_case_ids);n=len(ids)
    if n not in (24,80) or len(set(ids))!=n:raise ValueError('prespecified24_or80_unique_case_ids_required')
    by={c['case_id']:c for c in cases}
    if len(by)!=len(cases) or set(by)-set(ids):raise ValueError('duplicate_or_unregistered_case')
    detail=[];eligible=0;domain=0;quality=0;quality_complete=True;efficiency_complete=True
    successes={m:0 for m in (*ORDER,'ORACLE_RESTRICT')};times={m:[] for m in successes}
    g0=engineering_g0 and set(by)==set(ids)
    for cid in ids:
        c=by.get(cid,{});rows=c.get('methods',{});ref=rows.get('REF20');oracle=rows.get('ORACLE_RESTRICT')
        witness=rows.get('WITNESS');g0=g0 and bool(c.get('audit_closed')) and bool(witness and witness.get('valid') and witness.get('published') and witness.get('publication_audited'))
        ext=choose_external(c.get('external',[]),c.get('Ncap',0),c.get('Omega_cap',0))
        eligible+=int(ext is not None)
        native_ref=_legal(ref) and bool(ref.get('native_reference_valid'))
        domain_hit=bool(ext and native_ref and all(_within(ref['metrics'][k],ext['metrics'][k],f,k) for k,f in (('maxload',1.02),('top10_mean',1.05),('CV',1.05))))
        domain+=int(domain_hit)
        required=[rows.get(m) for m in (*ORDER,'ORACLE_RESTRICT')]
        complete=all(_legal(r) for r in required);quality_complete=quality_complete and complete
        q=False
        if complete:
            def compare(a,b):return _compare(a[1]['metrics'],b[1]['metrics'],('maxload','top10_mean','CV')) or ORDER.index(a[0])-ORDER.index(b[0])
            best=sorted([(m,rows[m]) for m in ORDER],key=cmp_to_key(compare))[0][1]['metrics'];om=oracle['metrics']
            q=om['maxload']<=.99*best['maxload']+TOL['maxload'] and all(_within(om[k],best[k],1.02,k) for k in ('top10_mean','CV'))
        quality+=int(q);efficiency_complete=efficiency_complete and native_ref
        for m in successes:
            r=rows.get(m);hit=bool(native_ref and target_hit(r,ref));successes[m]+=int(hit)
            actual=r.get('publication_seconds') if r else None
            known=actual is not None and np.isfinite(actual) and actual>=0
            if not known:efficiency_complete=False
            times[m].append(float(actual) if hit and known else max(30.,float(actual)) if known else None)
        detail.append({'case_id':cid,'external_eligible':ext is not None,'external_source':ext['source'] if ext else None,'domain_quality':domain_hit,'quality_win':q,'REF_target_defined':native_ref})
    mean={m:float(np.mean(v)) if all(t is not None for t in v) else None for m,v in times.items()}
    efficiency=efficiency_complete and all(successes['ORACLE_RESTRICT']>=successes[m] and mean['ORACLE_RESTRICT']<=.8*mean[m] for m in ORDER)
    g1=eligible>=math.ceil(.9*n) and domain>=math.ceil(.9*eligible)
    quality_pass=quality_complete and quality>=(6 if n==24 else 20)
    return {'schema':'s1_ai17_r3_independent_gates_v2','denominator':n,'G0':bool(g0),'G1':bool(g1),'G2':bool(quality_pass or efficiency),
            'pass':bool(g0 and g1 and (quality_pass or efficiency)),'eligible_count':eligible,'eligible_required':math.ceil(.9*n),
            'domain_count':domain,'domain_required':math.ceil(.9*eligible),'quality_count':quality,'quality_required':6 if n==24 else 20,
            'quality_complete':quality_complete,'quality_pass':bool(quality_pass),'efficiency_complete':bool(efficiency_complete),'efficiency_pass':bool(efficiency),
            'target_successes':successes,'restricted_publication_means':mean,'cases':detail,'resource_claim':'R2 anchors only; no crossround same-resource superiority'}
