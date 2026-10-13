import copy
import unittest
from scripts.stage1_ai_search_r4.gates import evaluate_gates
from scripts.stage1_ai_search_r4.audit import audit_layout,audit_cost,audit_ledger,audit_proposal,scene_sha

IDS=[f'c{i}' for i in range(24)]
def row(peak=100.,cost=10.,status='success'):
 return {'valid':True,'published':True,'identity_closed':True,'budget_closed':True,'cost_closed':True,'status':status,'publication_seconds':cost,'metrics':{'maxload':peak,'top10_mean':peak,'CV':0.}}
def cases():
 return [{'case_id':cid,'mother_id':f'm{i//4}','audit_closed':True,'complete_slots':True,'witness_valid':True,'common_input_preparation_seconds':2.,'methods':{'TEACHER':row(90.,20.,'available'),'RULE':row(100.,50.),'RANDOM':row(100.,50.),'TEACHER_EXACT':row(90.,30.),'TEACHER_PERTURBED':row(90.,30.)}} for i,cid in enumerate(IDS)]

class R4AuditTests(unittest.TestCase):
 def test_signed_denominator_and_complete(self):
  g=evaluate_gates(cases(),IDS,True);self.assertTrue(g['pass']);self.assertEqual(g['denominator'],24);self.assertEqual(g['quality_count'],24);self.assertEqual(len(g['mother_aggregation']),6)
 def test_partial_never_fullpass(self):
  g=evaluate_gates(cases()[:6],IDS,True);self.assertEqual(g['status'],'partial');self.assertFalse(g['pass']);self.assertEqual(g['quality_count'],6)
 def test_missing_cost_not_zero(self):
  c=cases();c[0]['methods']['TEACHER_PERTURBED']['publication_seconds']=None
  g=evaluate_gates(c,IDS,True);self.assertFalse(g['efficiency_complete']);self.assertIsNone(g['restricted_complete_means']['TEACHER_PERTURBED'])
 def test_teacher_cost_cannot_be_omitted(self):
  cost={'complete_method_seconds':5.,'components':dict(input=1.,prepare=1.,teacher=0.,proposal=1.,decoder=1.,wrapper=1.),'teacher_once_seconds':20.,'unknown_algorithm_spend':False}
  self.assertFalse(audit_cost(cost,'TEACHER_PERTURBED')['valid'])
 def test_missing_teacher_target_all_fail_newfloor(self):
  c=cases();c[0]['methods']['TEACHER']={'status':'unavailable'}
  g=evaluate_gates(c,IDS,True);d=g['cases'][0];self.assertEqual(d['censor_floor'],37.);self.assertFalse(any(d['target_success'].values()));self.assertEqual(d['restricted_complete_costs']['TEACHER_PERTURBED'],37.)
 def test_baseline_missing_not_quality_success(self):
  c=cases();c[0]['methods'].pop('RANDOM');g=evaluate_gates(c,IDS,True);self.assertEqual(g['quality_count'],23)
 def test_fallback_does_not_pass_g1(self):
  c=cases();[r['methods']['TEACHER_PERTURBED'].update(status='fallback') for r in c]
  g=evaluate_gates(c,IDS,True);self.assertEqual(g['perturbed_count'],0);self.assertFalse(g['G1'])
 def test_zero_reference_no_relative_advantage(self):
  c=cases()
  for r in c:
   for m in r['methods'].values():m['metrics'].update(maxload=0.,top10_mean=0.)
  self.assertEqual(evaluate_gates(c,IDS,True)['quality_count'],0)
 def test_prespecified_uniqueness(self):
  with self.assertRaises(ValueError):evaluate_gates(cases(),IDS[:-1],True)
  with self.assertRaises(ValueError):evaluate_gates(cases()+[cases()[0]],IDS,True)
 def test_ledger_cap_not_wallclock_censor(self):
  c=dict(raypairs=20,proposals=2,refits=0,seconds=6.,seconds_nominal=5.,phases=[dict(operator='RESOURCE_MERGE',proposals=0),dict(operator='INSERT',proposals=2)])
  self.assertTrue(audit_ledger(c,2,'decoder')['valid']);c['raypairs']=4097;self.assertFalse(audit_ledger(c,2,'decoder')['valid'])
 def test_static_raw_replay_duplicate_anchor_allowed(self):
  scene={'positions':[[48.,8.],[48.,8.]],'demands':[2.,0.],'aircraft_ids':['a','b'],'profile':'europe_v1'}
  layout={'centers':[[48.,8.],[48.,8.]],'widths_deg':[.3,.3],'assignment':[0,1],'beam_ids':['x','y'],'anchor_aircraft_ids':['a','a']}
  a=audit_layout(scene,layout,2,1.);self.assertTrue(a['valid']);self.assertEqual(a['metrics']['total_request'],2.)
  layout['assignment']=[0,0];self.assertFalse(audit_layout(scene,layout,2,1.)['valid'])
 def test_strict_width_bound(self):
  scene={'positions':[[48.,8.]],'demands':[0.],'aircraft_ids':['a'],'profile':'europe_v1'}
  layout={'centers':[[48.,8.]],'widths_deg':[1.000000000000001],'assignment':[0],'beam_ids':['x'],'anchor_aircraft_ids':['a']}
  self.assertFalse(audit_layout(scene,layout,1,1.)['valid'])

class R4ProposalAuditTests(unittest.TestCase):
 def fixture(self):
  scene={'positions':[[48.,8.],[48.,8.]],'demands':[2.,1.],'aircraft_ids':['a','b'],'profile':'europe_v1'}
  identity={'case_id':'c','mother_id':'m','profile':'base','input_sha256':scene_sha(scene),'identity_sha256':'id','role':'development'}
  prepared={'anchors':['a','a'],'raw_w':{'beam_ids':['x','y']}}
  teacher={'status':'available','layout':{'beam_ids':['x','y'],'anchor_aircraft_ids':['a','a'],'assignment':[0,1]}}
  proposal={**identity,'status':'ready','source_method':'TEACHER_EXACT','source_sha256':'s','slot_ids':['x','y'],'anchor_aircraft_ids':['a','a'],'scores_nxK_float32_finite':[[1.,0.],[0.,1.]],'score_mask_nxK_bool':[[True,True],[True,True]],'direction_offset_Kx3':[[0.,0.,0.],[0.,0.,0.]],'proposal_cost_ledger':{'raypairs':0,'hash_calls':0}}
  return scene,identity,prepared,teacher,proposal
 def test_exact_mock_preserves_duplicate_anchor(self):
  scene,i,p,t,r=self.fixture();self.assertTrue(audit_proposal(scene,r,i,p,t,'TEACHER_EXACT','s')['valid'])
 def test_exact_changed_score_rejected(self):
  scene,i,p,t,r=self.fixture();r['scores_nxK_float32_finite'][0][0]=.5;self.assertFalse(audit_proposal(scene,r,i,p,t,'TEACHER_EXACT','s')['valid'])
 def test_unavailable_identity_cannot_bypass(self):
  scene,i,p,t,r=self.fixture();r={'status':'unavailable','provenance':{**i,'source_sha256':'foreign','source_method':'TEACHER_EXACT'}};self.assertFalse(audit_proposal(scene,r,i,p,t,'TEACHER_EXACT','s')['valid'])
 def test_identity_float64_encoding(self):
  scene,i,p,t,r=self.fixture();self.assertEqual(i['input_sha256'],scene_sha({**scene,'positions':[[48,8],[48,8]],'demands':[2,1]}))
if __name__=='__main__':unittest.main()
