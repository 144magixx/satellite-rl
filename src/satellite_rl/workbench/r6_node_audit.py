"""Read-only node24 hard constraints, actual AI semantics, fixed quality gate."""
import math
import numpy as np
from scripts.stage1_ai_search_r4.audit import audit_layout,scene_sha
from scripts.stage1_ai_search_r5.audit import audit_decoder_ledger,audit_proposal
from scripts.stage1_ai_search_r4.gates import within,finite
from .r6_io import read_asset

def audit_row(scene,prepared,value,request):
 identity=request['identity'];result=value['result'];method=request['method']
 if value.get('identity')!=identity or value.get('actual_method')!=method or scene_sha(scene)!=identity['input_sha256']:raise ValueError('node input/role/method mismatch')
 if any(result.get(k)!=v for k,v in identity.items()):raise ValueError('terminal actual identity')
 if result.get('status') not in ('success','fallback','unavailable','unpublished'):raise ValueError('decoder result union')
 if not audit_decoder_ledger(result['cost'],len(scene['demands']),'D5')['valid']:raise ValueError('D5 capped ledger')
 proposal=read_asset(value['proposal_ref'])
 if method=='AI':
  from satellite_rl.coverage.ai_search.core import Scene
  from satellite_rl.coverage.ai_search_r5.core import PreparedCase
  from satellite_rl.coverage.ai_search_r6.ai_adapter import audit_ai_proposal
  p=PreparedCase.from_dict(Scene(np.asarray(scene['positions']),np.asarray(scene['demands']),tuple(scene['aircraft_ids']),scene['profile']),prepared)
  envelope=read_asset(value['AI_envelope_ref']);audit_ai_proposal(p,envelope)
  if envelope['inner']!=proposal or any(envelope.get(k)!=v for k,v in request['AI_provenance'].items()) or envelope['checkpoint_file_SHA']!=request['checkpoint_ref']['sha256']:raise ValueError('signed actual model source/checkpoint')
 else:
  checked=audit_proposal(scene,proposal,identity,prepared,None,method,request['producer_sha256'])
  if not checked['valid']:raise ValueError('independent baseline proposal')
 published=result['status'] in ('success','fallback');checked=audit_layout(scene,result['layout'],prepared['Ncap'],prepared['Omega_cap']) if published else None
 if published and not checked['valid']:raise ValueError('terminal hard constraints')
 if published:
  for key in ('maxload','top10_mean','CV','N'):
   x=result['metrics'][key];y=checked['metrics'][key]
   if not finite(x) or abs(x-y)>(1e-9 if key=='CV' else 1e-6)+1e-12*abs(y):raise ValueError('independent metrics')
 return {'valid':True,'published':published,'nonfallback':result['status']=='success','status':result['status'],'metrics':checked['metrics'] if checked else None,'identity_closed':True,'budget_closed':True}

def node_gate(cases):
 if len(cases)!=24 or len({c['case_index'] for c in cases})!=24:raise ValueError('fixed24 denominator')
 nonfallback=quality=0;costs_known=True;teacher_targets_closed=True;hard_closed=True;teacher_hits={'AI':0,'RULE':0,'RANDOM':0};costs={k:[] for k in teacher_hits}
 for case in cases:
  methods=case['methods']
  if set(methods)!=set(teacher_hits) or any(not r.get('valid') for r in methods.values()):raise ValueError('hard error/missing method stops dependency')
  hard_closed=hard_closed and all(r.get('identity_closed') and r.get('budget_closed') for r in methods.values())
  ai=methods['AI'];nonfallback+=ai['nonfallback'];base=min(((m,methods[m]) for m in ('RULE','RANDOM') if methods[m]['published']),key=lambda p:(p[1]['metrics']['maxload'],p[1]['metrics']['top10_mean'],p[1]['metrics']['CV'],p[0]),default=None)
  if base and ai['nonfallback'] and case['total_request']>0:
   am=ai['metrics'];bm=base[1]['metrics'];quality+=bool(bm['maxload']>0 and within(am['maxload'],bm['maxload'],.99,'maxload') and within(am['top10_mean'],bm['top10_mean'],1.02,'top10_mean') and within(am['CV'],bm['CV'],1.02,'CV'))
  t=case.get('teacher_metrics');teacher_targets_closed=teacher_targets_closed and bool(t and case.get('teacher_identity_and_cost_closed') and all(finite(t.get(k)) for k in ('maxload','top10_mean','CV')))
  for method,row in methods.items():
   if t and row['published']:teacher_hits[method]+=all(within(row['metrics'][k],t[k],ratio,k) for k,ratio in (('maxload',1.02),('top10_mean',1.05),('CV',1.05)))
   c=case['costs'].get(method)
   if c is None or not finite(c):costs_known=False
   else:costs[method].append(c if row['nonfallback'] else max(c,case['preparation_seconds']+35.))
 efficiency=bool(teacher_targets_closed and costs_known and all(len(v)==24 for v in costs.values()) and teacher_hits['AI']>=teacher_hits['RULE'] and teacher_hits['AI']>=teacher_hits['RANDOM'] and all(np.mean(costs['AI'])<=.8*np.mean(costs[m]) for m in ('RULE','RANDOM')))
 return {'phase':'node2000','passed':hard_closed and costs_known and nonfallback>=18 and (quality>=6 or efficiency),'hard_identity_budget_cost_closed':bool(hard_closed and costs_known),'nonfallback':nonfallback,'quality_successes':quality,'efficiency_branch':efficiency,'denominator':24,'future_dispatch_allowed':False}
