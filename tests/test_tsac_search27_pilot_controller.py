import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime,timezone
from scripts.tsac_search27.pilot_controller import decide,gate_statistics,continuation,SOFT,guard

class PilotControllerTests(unittest.TestCase):
    def stats(self,si=.95,sgm=.6,power=2000):return {'means':dict(SI_mean=si,SGM=sgm,total_power_w=power)}
    def test_registered_gate_and_baseline_not_independently_extended(self):
        result=decide(dict(B=self.stats(),E=self.stats(si=.954),LP=self.stats(sgm=.606,power=2300)))
        self.assertEqual(result['passed_variants'],['E']);self.assertFalse(result['known40_opened'])
        result=decide(dict(B=self.stats(),E=self.stats(si=.951),LP=self.stats(sgm=.59)))
        self.assertEqual(result['passed_variants'],[])
    def test_exact_mother_repeat_complete_and_legal(self):
        cases=[str(i) for i in range(20)];rows=[dict(node=50000,phase='development',identity={'case_id':c},repeat=k) for c in cases for k in range(3)]
        good=dict(SI_mean=.95,SGM=.6,total_power_w=1000,complete=1,constraint_violations=0,unserved_beams=0)
        with patch('scripts.tsac_search27.pilot_controller.audit_terminal',return_value=(good,None,None)):
            self.assertEqual(gate_statistics(rows,cases)['episodes'],60)
            for bad in [rows[:-1],rows+[rows[0]]]:
                with self.assertRaises(ValueError):gate_statistics(bad,cases)
        with patch('scripts.tsac_search27.pilot_controller.audit_terminal',return_value=(dict(good,constraint_violations=1),None,None)):
            with self.assertRaises(ValueError):gate_statistics(rows,cases)
    def test_unknown_post_only_adopts_actual_same_task_and_never_reposts(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'continue_E.intent.json').write_text('{}')
            with patch('scripts.tsac_search27.pilot_controller.now',return_value=SOFT.replace(hour=21)),patch('scripts.tsac_search27.pilot_controller.filtered_status',return_value=dict(stage='running',env_step=50001)),patch('scripts.tsac_search27.pilot_controller.r.api') as post:
                answer=continuation('E',{'port':20424},{},{},out,{}, {'previous_worker':{}},None)
                self.assertEqual(answer['stage'],'adopted_existing_continuation');post.assert_not_called()
    def test_soft_deadline_prohibits_all_dispatch(self):
        with patch('scripts.tsac_search27.pilot_controller.now',return_value=SOFT):
            with self.assertRaises(RuntimeError):guard()

if __name__=='__main__':unittest.main()
