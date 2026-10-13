"""Read-only R4 gates. No producer/optimizer imports or algorithm invocation."""
import math
from functools import cmp_to_key
METHODS=('RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED')
TOL={'maxload':1e-6,'top10_mean':1e-6,'CV':1e-9}

def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0

def legal(row):
    return bool(row and row.get('valid') and row.get('published') and row.get('identity_closed') and row.get('budget_closed') and row.get('metrics') and all(finite(row['metrics'].get(k)) for k in TOL))

def within(x,y,factor,key):
    # Zero references use absolute nonworsening; they cannot create relative gain.
    return finite(x) and finite(y) and x<=factor*y+TOL[key]+1e-12*abs(y)

def target_hit(row,teacher,nonfallback=False):
    return legal(teacher) and teacher.get('status')=='available' and legal(row) and (not nonfallback or row.get('status')=='success') and all(within(row['metrics'][k],teacher['metrics'][k],f,k) for k,f in (('maxload',1.02),('top10_mean',1.05),('CV',1.05)))

def compare(a,b):
    for k in TOL:
        x,y=a[1]['metrics'][k],b[1]['metrics'][k];t=TOL[k]+1e-12*max(abs(x),abs(y))
        if x<y-t:return -1
        if x>y+t:return 1
    return ('RULE','RANDOM').index(a[0])-('RULE','RANDOM').index(b[0])

def evaluate_gates(cases,expected_case_ids,engineering_g0=False):
    ids=list(expected_case_ids)
    if len(ids)!=24 or len(set(ids))!=24:raise ValueError('exactly24_prespecified_unique_cases_required')
    by={c['case_id']:c for c in cases}
    if len(by)!=len(cases) or set(by)-set(ids):raise ValueError('duplicate_or_unregistered_case')
    successes={m:0 for m in METHODS};times={m:[] for m in METHODS};details=[]
    teacher_count=exact=pert=quality=0;cost_complete=True
    complete=set(by)==set(ids) and all(c.get('complete_slots') for c in cases)
    g0=bool(engineering_g0 and complete)
    for cid in ids:
        c=by.get(cid,{});rows=c.get('methods',{});teacher=rows.get('TEACHER')
        g0=g0 and bool(c.get('audit_closed') and c.get('witness_valid'))
        teacher_good=legal(teacher) and teacher.get('status')=='available'
        teacher_count+=int(teacher_good)
        eh=target_hit(rows.get('TEACHER_EXACT'),teacher,True);ph=target_hit(rows.get('TEACHER_PERTURBED'),teacher,True)
        exact+=int(eh);pert+=int(ph);q=False;best_name=None
        if all(legal(rows.get(m)) for m in ('RULE','RANDOM','TEACHER_PERTURBED')):
            best_name,best=sorted([(m,rows[m]) for m in ('RULE','RANDOM')],key=cmp_to_key(compare))[0]
            candidate=rows['TEACHER_PERTURBED'];bm=best['metrics'];cm=candidate['metrics']
            q=bm['maxload']>0 and cm['maxload']<=.99*bm['maxload']+TOL['maxload'] and all(within(cm[k],bm[k],1.02,k) for k in ('top10_mean','CV'))
        quality+=int(q)
        common=c.get('common_input_preparation_seconds');floor=common+35 if finite(common) else None
        case_hits={};case_times={}
        for m in METHODS:
            r=rows.get(m);hit=target_hit(r,teacher);successes[m]+=int(hit);case_hits[m]=hit
            actual=r.get('publication_seconds') if r else None
            known=finite(actual) and finite(floor) and bool(r.get('cost_closed'))
            if not known:cost_complete=False
            value=actual if known and hit else max(actual,floor) if known else None
            times[m].append(value);case_times[m]=value
        details.append({'case_id':cid,'mother_id':c.get('mother_id'),'teacher_qualified':bool(teacher_good),'exact_qualified':bool(eh),'perturbed_qualified':bool(ph),'quality_success':bool(q),'best_baseline':best_name,'target_success':case_hits,'restricted_complete_costs':case_times,'censor_floor':floor})
    means={m:sum(t)/24 if all(x is not None for x in t) else None for m,t in times.items()}
    efficiency=cost_complete and all(successes['TEACHER_PERTURBED']>=successes[m] and means['TEACHER_PERTURBED']<=.8*means[m] for m in ('RULE','RANDOM'))
    g1=teacher_count>=22 and exact>=22 and pert>=18;g2=quality>=6 or efficiency
    mothers={}
    for d in details:
        if d['mother_id'] is not None:mothers.setdefault(d['mother_id'],[]).append(d)
    mother_summary={m:{'present_profiles':len(ds),'quality_fraction_of4':sum(d['quality_success'] for d in ds)/4,'teacher_fraction_of4':sum(d['teacher_qualified'] for d in ds)/4} for m,ds in mothers.items()}
    return {'schema':'R4B0IndependentGatesV1','status':'complete' if complete else 'partial','denominator':24,'G0':bool(g0),'G1':bool(g1),'G2':bool(g2),'pass':bool(complete and g0 and g1 and g2),'teacher_count':teacher_count,'teacher_required':22,'exact_count':exact,'exact_required':22,'perturbed_count':pert,'perturbed_required':18,'quality_count':quality,'quality_required':6,'quality_pass':quality>=6,'efficiency_complete':cost_complete,'efficiency_pass':bool(efficiency),'target_successes':successes,'restricted_complete_means':means,'cases':details,'mother_aggregation':mother_summary,'limitations':['development repeatedly reused; no independent test or AI evidence','24 slots are 6 mothers x 4 profiles; no independent24 significance claim','PERT cost includes its actual teacher cost; no R3 30second censor']}
