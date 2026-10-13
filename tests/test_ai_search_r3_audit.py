"""No solver or CUDA: independently hand constructed adversarial records."""
import copy
import unittest
import json
import tempfile
from pathlib import Path
from scripts.stage1_ai_search_r3.audit import audit_layout,audit_problem_vector,audit_integral_semantics,solid_angle,apply_publication_cost,audit_method
from scripts.stage1_ai_search_r3.gates import choose_external,evaluate_gates

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.scene={'positions':[[45.,10.],[45.,10.]],'demands':[3.,7.],'aircraft_ids':['a','b'],'profile':'europe_v1'}
        self.layout={'centers':[[45.,10.]],'widths_deg':[.3],'assignment':[0,0]}
    def test_owned_geometry_metrics(self):
        a=audit_layout(self.scene,self.layout,[.5],1,solid_angle([.5]))
        self.assertTrue(a['valid']);self.assertEqual(a['metrics']['maxload'],10.)
        self.assertEqual(a['metrics']['CV'],0.);self.assertLess(a['metrics']['actual_Omega'],a['metrics']['menu_Omega'])
    def test_fractional_owner_not_coerced(self):
        self.layout['assignment']=[0,.1]
        self.assertFalse(audit_layout(self.scene,self.layout,[.5])['valid'])
    def test_unassigned_and_empty_beam_fail(self):
        self.layout['assignment']=[0,-1];self.assertFalse(audit_layout(self.scene,self.layout,[.5])['valid'])
        self.layout.update(assignment=[0,0],centers=[[45,10],[45,10]],widths_deg=[.3,.3])
        self.assertIn('empty_beam',[f['code'] for f in audit_layout(self.scene,self.layout,[.5,.5])['failures']])
    def test_actual_geometry_menu_bounds(self):
        self.layout['centers']=[[55.,10.]]
        self.assertFalse(audit_layout(self.scene,self.layout,[.5])['valid'])
        self.layout['centers']=[[45.,10.]];self.layout['widths_deg']=[.6]
        self.assertFalse(audit_layout(self.scene,self.layout,[.5])['valid'])
    def test_resource_not_free_after_envelope(self):
        self.assertFalse(audit_layout(self.scene,self.layout,[1.],omega_cap=solid_angle([.5]))['valid'])
    def test_original_integer_values(self):
        p={'costs':[0],'integer':[1],'lower':[0],'upper':[1],'starts':[0,1],'indices':[0],'values':[1],'row_lower':[1],'row_upper':[1]}
        self.assertTrue(audit_problem_vector(p,[1])['valid'])
        self.assertFalse(audit_problem_vector(p,[.999])['valid'])
    def test_semantic_full_start_and_missing_edge(self):
        p={'centers':[[45.,10.]],'ncap':1,'omega_cap':solid_angle([.5])}
        self.assertTrue(audit_integral_semantics(self.scene,p,[2],[0,1],[0,0],[1,1,1,1])['valid'])
        self.assertFalse(audit_integral_semantics(self.scene,p,[2],[0],[0],[1,1,1])['valid'])
    def test_semantic_copy_prefix_and_peak(self):
        p={'centers':[[45.,10.]],'ncap':1,'omega_cap':solid_angle([.5])}
        self.assertFalse(audit_integral_semantics(self.scene,p,[3],[0,1],[0,0],[1,1,1,1])['valid'])
        self.assertFalse(audit_integral_semantics(self.scene,p,[2],[0,1],[0,0],[1,1,1,.7])['valid'])
    def test_uncertified_W_cannot_be_REF_target(self):
        witness={**self.layout,'widths_deg':[.5]};pool={'centers':[[45.,10.]],'ncap':2,'omega_cap':solid_angle([.5]),'witness':witness}
        # Six slots, twelve legal ownership edges, z. Native splits 7/3 at0.3 degrees.
        native=[1.,1.,0.,0.,0.,0.]+[0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]+[.7]
        seed=[0.,0.,1.,0.,0.,0.]+[0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.]+[1.]
        call={'vector':native,'seed_vector':seed,'options':{'threads':1,'random_seed':51001,'time_limit':12,'mip_max_nodes':20000,'mip_rel_gap':.001,'presolve':'on'},
              'status':'HighsModelStatus.kOptimal','value_valid':True,'backend':'highspy1.15.1','start_set_status':'HighsStatus.kOk','highspy_version':'1.15.1','native_library_version':'1.15.1',
              'fun':.7,'mip_dual_bound':.7,'mip_gap':0.,'start_native_log_ref':'fixture'}
        metrics={'N':1,'maxload':10.,'top10_mean':10.,'CV':0.,'solid_angle_sum_sr':solid_angle([.3])}
        result={'case_id':'x','method':'REF20','status':'published','layout':self.layout,'postprocessing':{'menu_layout':witness},'metrics':metrics,'selected_slot_ids':list(range(6)),
                'solver_calls':[call],'chosen_source':'external_W','comparable_seconds':1.}
        audit=audit_method(self.scene,pool,result,lambda q:'MIP start solution is feasible')
        self.assertTrue(audit['valid']);self.assertTrue(audit['native_audits'][0]['native_reference_valid'])
        self.assertFalse(audit['native_reference_valid'])

def metric(peak=100,tail=100,cv=1):return {'maxload':peak,'top10_mean':tail,'CV':cv,'N':2,'actual_Omega':.001}
def case(i):
    row={'valid':True,'published':True,'publication_audited':True,'metrics':metric(),'publication_seconds':10,'native_reference_valid':True}
    methods={m:copy.deepcopy(row) for m in ('WITNESS','RULE','RANDOM','FULL','REF20','ORACLE_RESTRICT')}
    methods['ORACLE_RESTRICT']['metrics']=metric(98,100,1)
    return {'case_id':str(i),'audit_closed':True,'Ncap':3,'Omega_cap':.002,'methods':methods,
            'external':[{'source':'R2_common_parent','valid':True,'metrics':metric(),'layout_sha256':'a'}]}

class GateTests(unittest.TestCase):
    def test_ops_order_and_publication_boundary(self):
        from satellite_rl.workbench.ai_search_r3_runner import method_order,publication_accounting
        for i in range(80):
            order=method_order(i)
            self.assertEqual(len(set(order)),6);self.assertEqual(order.index('ORACLE_RESTRICT'),order.index('REF20')+1)
        self.assertFalse(publication_accounting('RULE',29,1,.01)['regular_publication_eligible'])
        self.assertTrue(publication_accounting('REF20',29,1,.01)['regular_publication_eligible'])
        self.assertIsNone(publication_accounting('RULE',None,1,.01)['comparable_publication_seconds'])
    def test_ops_exact_development_SHA(self):
        from satellite_rl.workbench.coverage_ai_search_r3 import validate_config,DESIGN_SHA
        c={'schema_version':11,'research_stage':'coverage','algorithm':'ai_search_r3','name':'fixture','protocol_ref':{'sha256':DESIGN_SHA},
           'datasets':{'development':{'role':'development','count':80,'sha256':'9929e31c2ecd7adca912deeb16ad58abd6bd97d7cb2c631fc53cb4a2c57e6b5b'}},'workers':1,
           'resources':{'minimum_free_bytes':8589934592,'max_rss_bytes':4294967296},'vendor_ref':{}}
        self.assertEqual(validate_config(c),c)
        c['datasets']['development']['sha256']='0'*64
        with self.assertRaises(ValueError):validate_config(c)
    def test_unknown_started_case_cannot_repeat(self):
        from satellite_rl.workbench.ai_search_runner import recover_rows
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'cases').mkdir();(p/'started').mkdir();(p/'started/x.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unknown spent budget'):recover_rows(p,{'x'})
            self.assertTrue(json.loads((p/'unknown_budget.json').read_text())['automatic_repeat_forbidden'])
    def test_complete_publication_guard_and_unknown(self):
        r={'case_id':'x','method':'RULE','status':'published','comparable_seconds':29.}
        row={'valid':True,'inner_published':True,'metrics':metric()}
        receipt={'case_id':'x','method':'RULE','endpoint':'row_case_index_status_visible','core_comparable_seconds':29.,'required_provider_seconds':1.,'wrapper_publication_seconds':.1,'comparable_publication_seconds':30.1}
        audited=apply_publication_cost(row,r,receipt)
        self.assertFalse(audited['published']);self.assertTrue(audited['valid']);self.assertEqual(audited['publication_seconds'],30.1)
        unknown=apply_publication_cost(row,r,None)
        self.assertFalse(unknown['published']);self.assertIsNone(unknown['publication_seconds'])
        r['method']='REF20';receipt['method']='REF20'
        self.assertTrue(apply_publication_cost(row,r,receipt)['published'])
    def test_publication_sum_and_provider_tampering(self):
        r={'case_id':'x','method':'RULE','status':'published','comparable_seconds':1.}
        receipt={'case_id':'x','method':'RULE','endpoint':'row_case_index_status_visible','core_comparable_seconds':1.,'required_provider_seconds':1.,'wrapper_publication_seconds':1.,'comparable_publication_seconds':2.}
        self.assertFalse(apply_publication_cost({'inner_published':True},r,receipt)['publication_audited'])
    def test_fixed_external_candidates_only(self):
        e=[{'source':'arbitrary_archive','valid':True,'metrics':metric(1)}, {'source':'R2_D1_final_primary','valid':True,'metrics':metric(),'layout_sha256':'0'}, {'source':'R2_common_parent','valid':True,'metrics':metric(),'layout_sha256':'z'}]
        self.assertEqual(choose_external(e,3,.002)['source'],'R2_common_parent')
    def test_exact_denominator_quality(self):
        rows=[case(i) for i in range(24)];r=evaluate_gates(rows,[str(i) for i in range(24)],True)
        self.assertTrue(r['pass']);self.assertEqual(r['denominator'],24);self.assertEqual(r['eligible_required'],22)
    def test_missing_method_fails_quality_even_many_wins(self):
        rows=[case(i) for i in range(24)];del rows[0]['methods']['FULL']
        r=evaluate_gates(rows,[str(i) for i in range(24)],True)
        self.assertEqual(r['quality_count'],23);self.assertFalse(r['quality_pass']);self.assertFalse(r['pass'])
    def test_ref_unavailable_cannot_create_target(self):
        rows=[case(i) for i in range(24)];rows[0]['methods']['REF20']['native_reference_valid']=False
        r=evaluate_gates(rows,[str(i) for i in range(24)],True)
        self.assertFalse(r['efficiency_complete']);self.assertFalse(r['efficiency_pass']);self.assertEqual(r['target_successes']['RULE'],23)
    def test_all_denominator_failed_target_penalty(self):
        rows=[case(i) for i in range(24)];rows[0]['methods']['ORACLE_RESTRICT']['metrics']=metric(200)
        r=evaluate_gates(rows,[str(i) for i in range(24)],True)
        self.assertAlmostEqual(r['restricted_publication_means']['ORACLE_RESTRICT'],(23*10+30)/24)
    def test_unknown_time_not_zero_filled(self):
        rows=[case(i) for i in range(24)];rows[0]['methods']['FULL']['publication_seconds']=None
        r=evaluate_gates(rows,[str(i) for i in range(24)],True)
        self.assertIsNone(r['restricted_publication_means']['FULL']);self.assertFalse(r['efficiency_complete'])

if __name__=='__main__':unittest.main()
