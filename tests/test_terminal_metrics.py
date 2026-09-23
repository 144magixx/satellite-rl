import unittest
from types import SimpleNamespace
import numpy as np
from satellite_rl.evaluation.terminal_metrics import terminal_beam_arrays, terminal_metrics


class TerminalMetricsTests(unittest.TestCase):
    def test_final_rates_and_unserved_beams_define_denominator(self):
        env = SimpleNamespace(beam_info={'rate': np.array([100., 100., 100.])},
                              beam_bps=[50., 10.], beta=100., history_beam=[(0, 0, 0, 1, 10), (1, 1, 1, 0, 10)],
                              total_power=20., overload=True,
                              satellite_freq_state=SimpleNamespace(freq_pool_real=np.zeros((8, 100))))
        values = terminal_beam_arrays(env)
        np.testing.assert_array_equal(values['rates_bps'], [50., 10., 0.])
        np.testing.assert_array_equal(values['satisfaction'], [.5, .1, 0.])
        self.assertEqual(values['sgm'][-1], 0.)
        summary = terminal_metrics(env)
        self.assertAlmostEqual(summary['si_mean'], .2)
        self.assertEqual(summary['throughput_bps'], 60.)
        self.assertFalse(summary['completed'])
        self.assertTrue(summary['power_exceeded'])
        self.assertEqual(summary['zero_slot_rate'], .5)


if __name__ == '__main__': unittest.main()
