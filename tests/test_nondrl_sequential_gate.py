"""Mechanism follows an audited pre-screen, not an unchecked false gate flag."""
from copy import deepcopy
import unittest

from scripts.stage1_nondrl12 import report


def fixture(improved=7):
    threshold={'relative_cv_improvement':0.,'minimum_improved_mothers':8,'count_mothers':20,
               'primary_method':'rule','require_mechanism_audit':True,
               'required_mechanism':'oracle_vs_uniform_region_mean_gain_at_least_0p01_with_raw_trials'}
    recorded=[];actual=[]
    for index in range(20):
        for multiplier in (.75,1.,1.25):
            cv=.9 if index<improved else 1.
            row={'index':index,'multiplier':multiplier,'mother_id':f'm{index}','cv':cv,'initial_cv':1.,
                 'carry':1.,'initial_carry':1.,'feasible':True,'initial_sha256':f'init{index}'}
            recorded.append(row)
            actual.append({'dataset_index':index,'multiplier':multiplier,'mother_scene_id':f'm{index}',
                           'load_cv_all':cv,'initial_cv':1.,'aircraft_carry':1.,'initial_carry':1.,
                           'hard_feasible':True,'initial_sha256':f'init{index}'})
    gate={'experiment_id':'S1-ND12','route':'05','threshold':threshold,'passed':False,'status':'science_stopped',
          'actual_row_count':60,'expected_row_count':60,'mother_count':20,
          'improved_mothers':list(range(improved)),'geometry_pass':True,'mechanism':None,
          'sources':[{'run_id':'rule_source','manifest_sha256':'a'*64,'rows':recorded}]}
    audits={'rule_source':{'state':'completed','raw_manifest_sha256':'a'*64,'rows':actual}}
    return gate,threshold,audits


class SequentialGateTests(unittest.TestCase):
    def test_complete60_audited_prefilter_failure_does_not_require_unreached_oracle(self):
        gate,threshold,audits=fixture()
        outcome=report.verify_gate(gate,threshold,audits,state_gate=deepcopy(gate))
        self.assertFalse(outcome['actual_passed']);self.assertEqual(outcome['improved_mothers'],7)
        self.assertEqual(outcome['mechanism_stage'],'not_reached_after_audited_preliminary_failure')

    def test_prefilter_passing_but_false_outcome_cannot_hide_missing_mechanism(self):
        gate,threshold,audits=fixture(8)
        with self.assertRaisesRegex(ValueError,'dynamic mechanism|engineering gap'):
            report.verify_gate(gate,threshold,audits,state_gate=deepcopy(gate))

    def test_forged_false_count_raw_rows_sha_or_missing_row_do_not_skip_mechanism(self):
        for corruption in ('gate_count','raw_quality','manifest_sha','missing_row','baseline','initial_identity'):
            with self.subTest(corruption=corruption):
                gate,threshold,audits=fixture()
                if corruption=='gate_count':gate['improved_mothers']=list(range(6))
                elif corruption=='raw_quality':audits['rule_source']['rows'][0]['load_cv_all']=.5
                elif corruption=='manifest_sha':audits['rule_source']['raw_manifest_sha256']='b'*64
                elif corruption=='missing_row':audits['rule_source']['rows'].pop()
                elif corruption=='baseline':gate['sources'][0]['rows'][0]['initial_cv']=2.
                else:gate['sources'][0]['rows'][0]['initial_sha256']='forged'
                with self.assertRaises(ValueError):report.verify_gate(gate,threshold,audits,state_gate=deepcopy(gate))

    def test_gate_state_status_threshold_or_claimed_mechanism_must_match(self):
        gate,threshold,audits=fixture();state=deepcopy(gate);state['status']='passed'
        with self.assertRaisesRegex(ValueError,'gate/state'):
            report.verify_gate(gate,threshold,audits,state_gate=state)
        gate['status']='pending'
        with self.assertRaisesRegex(ValueError,'terminal gate'):
            report.verify_gate(gate,threshold,audits)
        gate['status']='science_stopped';gate['mechanism']={'passed':False}
        with self.assertRaisesRegex(ValueError,'unexpectedly claims'):
            report.verify_gate(gate,threshold,audits)
        gate['mechanism']=None;wrong={**threshold,'minimum_improved_mothers':7}
        with self.assertRaisesRegex(ValueError,'threshold'):
            report.verify_gate(gate,wrong,audits)


if __name__=='__main__':unittest.main()
