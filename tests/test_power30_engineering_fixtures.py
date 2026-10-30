"""Pure fixture arithmetic/admission checks, with no real environment or model."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from satellite_rl.power30.engineering_fixtures import expected_summary,fixture_admission
from satellite_rl.power30.logging import content_sha


class FixturePreparationTest(unittest.TestCase):
    def test_slot_first_summary_uses_only_existing_prefix_arrays(self):
        gain=np.zeros((3,3));gain[0,2]=2.;gain[1,2]=4.
        env=SimpleNamespace(N=3,beam_idx=2,indices=np.arange(3),groups=np.array([0,2,4]),
            starts=np.array([3,3,0]),counts=np.array([1,1,0]),powers=np.array([1.,1.,0.]),
            rates=np.ones(3),demands=np.ones(3),physics={'rx_gain_dBi':0.},
            kernel={'gain':gain,'path':np.ones((3,100))},
            last_coefficients=SimpleNamespace(slot_owner=np.array([0,1]),occupied_slots=np.array([3,3]),
                noise=np.array([2.,2.]),interference=np.zeros((2,3))))
        value=expected_summary(env)
        self.assertAlmostEqual(value[3],np.log1p(3.))
        self.assertEqual(np.count_nonzero(value),1)
        env.groups[0]=4
        self.assertAlmostEqual(expected_summary(env)[3],np.log1p(2.))
        env.beam_idx=3
        self.assertFalse(expected_summary(env).any())

    def test_fixture_admission_is_exactly_bound_to_root_source_case_and_caps(self):
        config=dict(fixture_admission_ref={},source_sha='candidate',case_record={'role':'train','case_id':'fixed'})
        receipt=dict(schema='W30PhysicalFixtureAdmissionV1',experiment_id='S2-W30',root_approved=True,passed=True,
            source_sha='candidate',case_record_sha=content_sha(config['case_record']),independent_fixture_review_passed=True,
            max_phy=1800,max_kernel=9,max_coeff=1800,max_forward=0,max_updates=0)
        with patch('satellite_rl.power30.engineering_fixtures.read_ref',return_value=receipt):
            self.assertEqual(fixture_admission(config),receipt)
            receipt['max_updates']=1
            with self.assertRaises(ValueError):fixture_admission(config)


if __name__=='__main__':unittest.main()
