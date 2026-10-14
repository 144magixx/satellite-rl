import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json

import numpy as np

path = Path(__file__).resolve().parents[1] / 'scripts/t0_width/report_analysis.py'
spec = importlib.util.spec_from_file_location('w26_report', path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def terminal():
    return dict(phase='confirmation_S00', node=150000, mother=0, repeat=0, action_seed=275000,
                identity=dict(case_id='c', layout_SHA='l', aircraft_SHA='a', physics_SHA='p'),
                beams=[dict(beam_id='b0', demand_bps=21600000, capacity_bps=21600000, SI=1., SGM=1.),
                       dict(beam_id='b1', demand_bps=21600000, capacity_bps=0., SI=0., SGM=0.)],
                metrics=dict(SI_mean=.5, SI_sum=1., SGM=.5, throughput_bps=21600000.,
                             total_power_w=5., N=2, zero_slot_rate=.5, processed=1),
                timing=dict(total_s=.1),
                allocation_wire=dict(beam_ids=['b0', 'b1'], parent_layout_SHA='l', parent_case_SHA='a',
                                     actual_start_slot=[0, 0], actual_slot_count=[1, 0],
                                     actual_whole_beam_power_w=[5., 0.], group_by_beam_id=dict(b0=0, b1=1)))


class ReportTests(unittest.TestCase):
    def test_failure_denominator_and_zero_capacity(self):
        metrics, _, _ = report.audit_terminal(terminal())
        self.assertEqual(metrics['SI_mean'], .5)
        self.assertEqual(metrics['SGM'], .5)
        self.assertEqual(metrics['complete'], 0)
        self.assertEqual(metrics['unserved_beams'], 1)

    def test_constraint_count_does_not_discard_episode(self):
        row = terminal()
        row['allocation_wire']['actual_whole_beam_power_w'][0] = 6.
        row['metrics']['total_power_w'] = 6.
        self.assertEqual(report.audit_terminal(row)[0]['constraint_violations'], 1)

    def test_metric_and_identity_corruption_rejected(self):
        row = terminal()
        row['metrics']['SI_mean'] = 1.
        with self.assertRaisesRegex(ValueError, 'metric mismatch'):
            report.audit_terminal(row)
        row = terminal()
        row['allocation_wire']['parent_case_SHA'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'parent identity'):
            report.audit_terminal(row)

    def test_repeats_average_within_mother(self):
        rows = []
        for m in range(40):
            for g in report.METHODS:
                for r in range(3 if g in report.GROUPS + ('random',) else 1):
                    row = terminal()
                    row.update(phase='confirmation_' + g, mother=m, repeat=r)
                    row['action_seed'] = 275000 + m + r * 1000000
                    row['timing']['total_s'] = float(r + 1)
                    rows.append(row)
        raw, mothers, contrasts, differences, summary = report.summarize_confirmation(rows)
        self.assertEqual(len(raw), 680)
        self.assertEqual(len(mothers), 280)
        self.assertEqual(len(contrasts), 32)
        self.assertEqual(summary['S00']['SI_mean']['mothers'], 40)
        self.assertEqual(next(r for r in mothers if r['method'] == 'S00')['time_total_s'], 2.)
        self.assertEqual(sum(c['preregistered'] for c in contrasts), 8)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            report.summarize_confirmation(rows[:-1])

    def test_bootstrap_reproducible(self):
        self.assertEqual(report.bootstrap([1., 2., 3.]), report.bootstrap([1., 2., 3.]))
        self.assertEqual(report.bootstrap(np.zeros(40))['high'], 0.)

    def test_confirmation_rule_mapping_and_duplicate_steps(self):
        row = terminal()
        row['phase'] = 'confirmation_fixed'
        row['allocation_wire']['processing_index_to_beam_id'] = ['b0', 'b1']
        row['allocation_wire']['actual_slot_count'] = [1, 1]
        row['allocation_wire']['actual_whole_beam_power_w'] = [5., 5.]
        steps = [dict(phase='confirmation_fixed', mother=0, repeat=0, beam_step=i, beam_id='b' + str(i),
                      identity=row['identity'], action_seed=275000, requested_action=[0, 0, 0],
                      actual_action=dict(start=0, count=1, power_w=5.), source_preference=dict(mapping_changed=bool(i)))
                 for i in range(2)]
        with tempfile.TemporaryDirectory(prefix='w26_behavior_test_') as tmp:
            p = Path(tmp) / 'steps.jsonl'
            p.write_text('\n'.join(json.dumps(s) for s in steps), encoding='utf8')
            # Only fixed has terminals in this deliberately small correctness fixture.
            expected = [row]
            original_methods = report.METHODS
            try:
                report.METHODS = ('fixed',)
                result = report.confirmation_behavior(p, expected)[0]
                self.assertEqual(result['mapping_changed_rate'], .5)
                self.assertEqual(result['request_execution_mismatch'], 0)
                p.write_text('\n'.join(json.dumps(s) for s in steps + [steps[0]]), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    report.confirmation_behavior(p, expected)
            finally:
                report.METHODS = original_methods


if __name__ == '__main__':
    unittest.main()
