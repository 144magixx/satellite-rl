"""Small synthetic S3 integration/derivative checks, not scientific SCA results."""
from copy import deepcopy
import unittest
import numpy as np
from test_layout_resource_environment import synthetic_input
from satellite_rl.layout_resource.environment import LayoutResourceEnv, encode_action
from satellite_rl.layout_resource.power import LayoutPowerProblem, refine_layout_power
from satellite_rl.shared_interface.content import content_sha, seal, wire_bytes, read_wire
from satellite_rl.shared_interface.contracts import Authority
from satellite_rl.shared_interface.stages import UniformStage3FixedInput


def stage3(n=3, reverse=False, zero=False, reorder=False):
    env = LayoutResourceEnv(synthetic_input(n, reverse))
    for i in range(n): env.step(0 if zero else encode_action(0, 2))
    allocation = env.export_allocation()
    if reorder:
        allocation = deepcopy(allocation)
        order = list(range(n))[::-1]
        allocation['beam_ids'] = [allocation['beam_ids'][i] for i in order]
        for name in ('actual_start_slot', 'actual_slot_count', 'actual_whole_beam_power_w'):
            allocation[name] = allocation[name][order]
        allocation = seal(allocation)
    registry = dict(schema='R7PublishedParentRegistryV1',
        case_sha_by_id={'engineering':env.stage.case['content_sha256']},
        layout_shas=[env.stage.layout['content_sha256']], allocation_shas=[allocation['content_sha256']])
    authority = Authority(registry, content_sha(registry))
    stage = UniformStage3FixedInput.from_layout_and_allocation(env.stage.case,
        env.stage.layout, allocation, authority, env.stage.physics)
    return env, stage


class LayoutPowerChecks(unittest.TestCase):
    def test_shared_initial_and_parent_preservation_wire(self):
        env, stage = stage3(reorder=True)
        problem = LayoutPowerProblem(stage)
        result = problem.evaluate(problem.initial_power_w)
        np.testing.assert_array_equal(result.capacity_bps, env.capacities)
        initial = problem.initial_power_w.copy()
        record = problem.refined_record(initial, method='engineering_fixed', diagnostics=dict(engineering_only=True))
        restored = read_wire(wire_bytes(record))
        self.assertEqual(restored['content_sha256'], record['content_sha256'])
        self.assertEqual(record['parent_case_SHA'], stage.case['content_sha256'])
        self.assertEqual(record['parent_layout_SHA'], stage.layout['content_sha256'])
        self.assertEqual(record['parent_allocation_SHA'], stage.allocation['content_sha256'])
        self.assertEqual(record['beam_ids'], stage.layout['beam_ids'])
        np.testing.assert_array_equal(record['initial_whole_beam_power_w'], initial)
        np.testing.assert_array_equal(record['request_bps_by_beam_int64'], stage.layout['request_bps_by_beam_int64'])
        self.assertEqual(record['request_bps_by_beam_int64'].dtype, np.dtype('int64'))
        np.testing.assert_array_equal(record['actual_start_slot'], problem.starts)
        np.testing.assert_array_equal(record['actual_slot_count'], problem.slots)
        np.testing.assert_array_equal(stage.layout['full_width_deg'], env.stage.layout['full_width_deg'])

    def test_reordered_layout_and_allocation_id_binding(self):
        _, normal = stage3()
        _, reverse = stage3(reverse=True, reorder=True)
        a, b = LayoutPowerProblem(normal), LayoutPowerProblem(reverse)
        ca, cb = a.evaluate(a.initial_power_w).capacity_bps, b.evaluate(b.initial_power_w).capacity_bps
        by_id = dict(zip(b.beam_ids, cb))
        np.testing.assert_array_equal(ca, [by_id[bid] for bid in a.beam_ids])
        self.assertEqual(dict(zip(a.beam_ids, a.groups)), dict(zip(b.beam_ids, b.groups)))

    def test_capacity_and_service_jacobians_finite_difference(self):
        _, stage = stage3()
        problem = LayoutPowerProblem(stage)
        power = np.array([1.5, 2.5, 3.5])
        exact = problem.evaluate(power, with_jacobian=True)
        numerical = np.zeros((3, 3))
        si, sgm = [], []
        eps = 1e-5
        for j in range(3):
            delta = np.zeros(3); delta[j] = eps
            plus, minus = problem.evaluate(power+delta), problem.evaluate(power-delta)
            numerical[:, j] = (plus.capacity_bps-minus.capacity_bps)/(2*eps)
            si.append((plus.si-minus.si)/(2*eps))
            sgm.append((plus.sgm-minus.sgm)/(2*eps))
        np.testing.assert_allclose(exact.capacity_jacobian, numerical, rtol=2e-5, atol=.02)
        np.testing.assert_allclose(exact.si_jacobian, si, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(exact.sgm_jacobian, sgm, rtol=2e-5, atol=1e-8)

    def test_all_noop_and_strict_power_constraints(self):
        _, empty = stage3(zero=True)
        problem = LayoutPowerProblem(empty)
        result = problem.evaluate(np.zeros(3), with_jacobian=True)
        self.assertEqual(result.si, 0)
        self.assertEqual(result.sgm, 0)
        np.testing.assert_array_equal(result.capacity_jacobian, np.zeros((3, 3)))
        with self.assertRaises(ValueError): problem.evaluate([0, 1, 0])
        _, active = stage3()
        problem = LayoutPowerProblem(active)
        for vector in ([-1, 0, 0], [3000.0001, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0], [0, 0]):
            with self.assertRaises(ValueError): problem.evaluate(vector)

    def test_bounded_small_solver_produces_serializable_actual_record(self):
        _, stage = stage3(n=1)
        record = refine_layout_power(stage, options=dict(maxiter=2, maxeval=8, max_time_seconds=2))
        self.assertEqual(record['schema'], 'ContinuousLayoutPowerV1')
        np.testing.assert_array_equal(record['initial_whole_beam_power_w'], [7.5])
        self.assertLessEqual(float(record['refined_whole_beam_power_w'].sum()), 3000)
        self.assertEqual(read_wire(wire_bytes(record))['content_sha256'], record['content_sha256'])
        self.assertEqual(record['diagnostics']['options']['maxeval'], 8)
        np.testing.assert_array_equal(record['actual_slot_count'], [2])

    def test_solver_budget_alias_and_partial_allocation_are_rejected(self):
        _, stage = stage3(n=1)
        with self.assertRaisesRegex(ValueError, 'aliases'):
            refine_layout_power(stage, options=dict(maxiter=2, maxeval=8,
                max_time_seconds=2, max_evaluations=800))
        allocation = deepcopy(stage.allocation)
        allocation['requested_actions_separate']['terminal'] = False
        allocation = seal(allocation)
        registry = dict(schema='R7PublishedParentRegistryV1',
            case_sha_by_id={'engineering': stage.case['content_sha256']},
            layout_shas=[stage.layout['content_sha256']], allocation_shas=[allocation['content_sha256']])
        partial = UniformStage3FixedInput.from_layout_and_allocation(stage.case,
            stage.layout, allocation, Authority(registry, content_sha(registry)), stage.physics)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            LayoutPowerProblem(partial)


if __name__ == '__main__': unittest.main()
