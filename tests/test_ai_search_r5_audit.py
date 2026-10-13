import copy
import unittest
from scripts.stage1_ai_search_r5.gates import evaluate_gates
from scripts.stage1_ai_search_r5.audit import Reader,audit_decoder_ledger,audit_derivation,canonical_sha,CONTRACT_SHA

def row(peak=100,status='success'):
    return {'valid':True,'published':True,'identity_closed':True,'budget_closed':True,'status':status,'metrics':{'maxload':peak,'top10_mean':peak,'CV':1},'mechanisms':{'N_changed':True,'width_distribution_changed':True,'nonfloor_heterogeneous_widths':True}}

def cases():
    result=[]
    for i in range(24):
        result.append({'case_id':str(i),'mother_id':str(i//4),'complete_slots':True,'audit_closed':True,'witness_valid':True,'teacher':row(100,'available'),'D5':{m:row(100 if m in ('RULE','RANDOM') else 98) for m in ('RULE','RANDOM','EXACT','OWNER10','JOINT10')},'D4':{m:row(130) for m in ('OWNER10','JOINT10')}})
    return result

class AuditTests(unittest.TestCase):
    def test_fixed_gates_pass(self):
        g=evaluate_gates(cases(),[str(i) for i in range(24)],True)
        self.assertTrue(g['pass']);self.assertEqual(g['paired_net_gain'],48)
    def test_missing_case_partial_not_drop(self):
        g=evaluate_gates(cases()[:-1],[str(i) for i in range(24)],True)
        self.assertEqual(g['denominator'],24);self.assertEqual(g['status'],'partial');self.assertFalse(g['pass'])
    def test_missing_d4_cannot_fake_gain(self):
        c=cases()
        for x in c:x['D4']={}
        g=evaluate_gates(c,[str(i) for i in range(24)],True)
        self.assertEqual(g['paired_net_gain'],0);self.assertFalse(g['pass'])
    def test_quality_each_method_no_efficiency_or(self):
        c=cases()
        for x in c:x['D5']['JOINT10']=row(100)
        g=evaluate_gates(c,[str(i) for i in range(24)],True)
        self.assertFalse(g['conditions']['quality']);self.assertFalse(g['pass'])
    def test_fallback_no_capability_credit(self):
        c=cases()
        for x in c:
            for m in ('EXACT','OWNER10','JOINT10'):x['D5'][m]['status']='fallback'
        g=evaluate_gates(c,[str(i) for i in range(24)],True)
        self.assertEqual(g['capability_cases'],[]);self.assertFalse(g['pass'])
    def test_duplicate_rejected(self):
        c=cases();c[-1]['case_id']='0'
        with self.assertRaises(ValueError):evaluate_gates(c,[str(i) for i in range(24)],True)
    def test_unpinned_read_rejected(self):
        with self.assertRaises(ValueError):Reader().path({'path':'unused'})
    def test_d5_accounting_requires_reserve(self):
        c={k:0 for k in ('raypairs','proposals','refits','geometry_calls','center_constructions','sort_input_count','scan_input_count','hash_calls','cache_hits','rank_geometry_pairs')};c.update(seconds=0,terminal_reserve_raypairs=9,seconds_nominal=5,raypairs_max=6144,proposals_max=512,refits_max=8)
        self.assertTrue(audit_decoder_ledger(c,3,'D5')['valid']);c['raypairs']=6145
        self.assertFalse(audit_decoder_ledger(c,3,'D5')['valid'])

    def test_negative_spend_rejected(self):
        self.assertFalse(audit_decoder_ledger({'raypairs':-1},3,'D5')['valid'])

    def test_actual_zero_nominal_engineering_cutoff(self):
        c={k:0 for k in ('raypairs','proposals','refits','geometry_calls','center_constructions','sort_input_count','scan_input_count','hash_calls','cache_hits','rank_geometry_pairs')};c.update(seconds=0,terminal_reserve_raypairs=9,seconds_nominal=0,raypairs_max=6144,proposals_max=512,refits_max=8)
        self.assertTrue(audit_decoder_ledger(c,3,'D5',0)['valid'])
        self.assertFalse(audit_decoder_ledger(c,3,'D5')['valid'])

    def test_d4_conversion_zero_offset_and_actual_method(self):
        reg={k:k for k in ('case_id','mother_id','profile','input_sha256','identity_sha256','role')};reg.update(prepared={'stage_output':{'path':'p','sha256':'p'},'source_sha256':'old'},teacher={'stage_output':{'path':'t','sha256':'t'}})
        proposal={'slot_ids':['s'],'anchor_aircraft_ids':['a'],'scores_n_by_K':[[1]],'mask_n_by_K':[[False]]}
        inner={**{k:reg[k] for k in ('case_id','mother_id','profile','input_sha256','identity_sha256','role')},'source_method':'TEACHER_EXACT','source_sha256':'old','slot_ids':['s'],'anchor_aircraft_ids':['a'],'scores_nxK_float32_finite':[[1]],'score_mask_nxK_bool':[[False]],'direction_offset_Kx3':[[0,0,0]]}
        d={'diagnostic_method':'OWNER10','proposal_before':proposal,'proposal_after':inner,'proposal_before_sha256':canonical_sha(proposal),'proposal_after_sha256':canonical_sha(inner),'conversion_cost':{'seconds':0,'raypairs':0,'hash_calls':2}}
        envelope={'identity':{k:reg[k] for k in ('case_id','mother_id','profile','input_sha256','identity_sha256','role')},'actual_method':'D4_OWNER10','new_source_sha256':'new','contract_sha256':CONTRACT_SHA,'original_prepared_ref':reg['prepared']['stage_output'],'original_teacher_ref':reg['teacher']['stage_output'],'result_derivation':d}
        self.assertTrue(audit_derivation(envelope,proposal,reg,'D4_OWNER10','new')['valid'])
        inner['direction_offset_Kx3']=[[1,0,0]];d['proposal_after_sha256']=canonical_sha(inner)
        self.assertIn('D4_zero_offset',audit_derivation(envelope,proposal,reg,'D4_OWNER10','new')['failures'])

if __name__=='__main__':unittest.main()


