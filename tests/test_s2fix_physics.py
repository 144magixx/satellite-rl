import unittest
from types import SimpleNamespace
import numpy as np
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS,PHYSICS_V2
from satellite_rl.shared_interface.link import prepare_slot_kernel,build_link_coefficients,capacity_bps
from satellite_rl.s2fix.evaluate import fixed_problem,metrics,run_sca
from satellite_rl.s2fix.greedy import greedy_allocation,proportional_power


def scene():
    case=dict(aircraft_ids=['a','b','c','d'],lat_lon_deg=np.array([[45.,10.],[45.1,10.1],[46.,11.],[46.1,11.1]]))
    layout=dict(beam_ids=['x','y'],centres_lat_lon_deg=np.array([[45.,10.],[46.,11.]]),
        full_width_deg=np.array([.5,.5]),owner_aircraft_to_beam_id=[['a','x'],['b','x'],['c','y'],['d','y']],
        aircraft_count_by_beam=np.array([2,2]),request_bps_by_beam_int64=np.array([172800000,172800000]),
        content_sha256='layout')
    return SimpleNamespace(case=case,layout=layout,physics=PHYSICS_V2)


class PhysicsV2Test(unittest.TestCase):
    def test_shared_parity_and_derivative(self):
        s=scene();g=np.array([0,2]);starts=np.array([0,0]);counts=np.array([4,4]);p=np.array([31.,43.])
        kernel=prepare_slot_kernel(s.layout,s.physics,s.case)
        allocation=dict(beam_ids=['x','y'],group_by_beam_id={'x':0,'y':2},actual_start_slot=starts,
            actual_slot_count=counts,actual_whole_beam_power_w=p)
        c=build_link_coefficients(s.layout,allocation,s.physics,kernel)
        problem=fixed_problem(s,g,starts,counts,p)
        ev=problem.evaluate(p,True)
        np.testing.assert_allclose(capacity_bps(c),ev.capacity_bps,rtol=1e-14)
        for i in range(2):
            step=np.zeros(2);step[i]=1e-3
            derivative=(problem.evaluate(p+step).capacity_bps-problem.evaluate(p-step).capacity_bps)/2e-3
            np.testing.assert_allclose(derivative,ev.capacity_jacobian[:,i],rtol=1e-5,atol=.1)
        self.assertTrue(np.all(np.diag(kernel['gain'])<1e5))

    def test_spectral_cap_and_zero_jacobian(self):
        s=scene();g=np.array([0,1]);counts=np.array([1,1]);p=np.array([50.,50.])
        problem=fixed_problem(s,g,np.zeros(2,int),counts,p)
        problem.signal=np.ones(2)
        ev=problem.evaluate(p,True)
        np.testing.assert_array_equal(ev.capacity_bps,np.full(2,25e6*4.5))
        np.testing.assert_array_equal(ev.capacity_jacobian,np.zeros((2,2)))

    def test_greedy_and_sca_cap(self):
        s=scene();g,st,c,p=greedy_allocation(s)
        self.assertTrue(np.all(p<=50))
        result=run_sca(s,g,st,c,p,dict(maxiter=3,time_limit_seconds=2.))
        self.assertTrue(np.all(np.asarray(result['power_w'])<=50+1e-7))
        row=metrics(s,g,st,c,p)
        self.assertAlmostEqual(row['W'],row['SI'])
        np.testing.assert_allclose(proportional_power([1,3,100]),[50,50,50])


if __name__=='__main__':unittest.main()
