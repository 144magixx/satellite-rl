"""Pure stub qualification: zero shared kernel/PHY/model calls or updates."""
import unittest
from unittest.mock import patch
import numpy as np

from satellite_rl.bottleneck29 import core


def context(n=4, groups=None):
    ids = tuple('b'+str(i) for i in range(n))
    return core.Context(dict(content_sha256='synthetic-not-scientific'), {}, ids,
                        np.arange(n, dtype=np.int64),
                        np.asarray(groups if groups is not None else range(n), dtype=np.int64),
                        np.full(n, 21_600_000., dtype=np.float64), {},
                        dict(layout_SHA='synthetic-not-scientific', physics_SHA='stub',
                             order=list(ids), groups=list(groups if groups is not None else range(n))))


class PureCoreTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        # Replace both physical entry points before any test can call them.
        self.kernel_patch = patch.object(core, 'prepare_slot_kernel', side_effect=AssertionError('real kernel forbidden'))
        self.build_patch = patch.object(core, 'build_link_coefficients', side_effect=lambda layout, allocation, physics, kernel: allocation)
        self.capacity_patch = patch.object(core, 'capacity_bps', side_effect=self.stub_capacity)
        self.kernel_patch.start(); self.build_patch.start(); self.capacity_patch.start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def stub_capacity(allocation):
        # Deliberately simple nonphysical arithmetic used only to check plumbing.
        power = allocation['actual_whole_beam_power_w']
        count = allocation['actual_slot_count']
        return np.asarray(power*count*1e6, dtype=np.float64)

    def meter(self, kind, call_id, operation):
        self.calls.append((kind, call_id))
        return operation()

    def test_positions(self):
        self.assertEqual(core.prefix_positions(80), [10, 40, 70])
        self.assertEqual(core.prefix_positions(1), [0])

    def test_empty_mask_955_and_grid_domains(self):
        ctx = context(); state = core.initial_snapshot(ctx)
        scans = core.candidate_specs(ctx, state, [0, 0, 0])
        self.assertEqual(len(scans['minimum5']), 955)
        self.assertEqual(len(scans['fine_power']), 100)
        self.assertEqual(sum(s['domain']=='original' for s in scans['fine_power']), 10)
        self.assertEqual(sum(s['domain']=='lower_bound_relaxation' for s in scans['fine_power']), 9)
        self.assertEqual(sum(s['domain']=='granularity_relaxation' for s in scans['fine_power']), 81)
        self.assertEqual(self.calls, [])

    def test_snapshot_replay_no_mutation_and_reward(self):
        ctx = context(); state = core.initial_snapshot(ctx); digest = core.content_sha(state)
        result = core.replay_original_action(ctx, state, [4, 1, 2], self.meter, 'rebuild0')
        self.assertEqual(self.calls, [('phy', 'rebuild0')])
        self.assertEqual(core.content_sha(state), digest)
        self.assertEqual(result['current']['capacity_bps'], 30e6)
        self.assertEqual(result['requested_action'], result['actual_action'])
        self.assertAlmostEqual(result['reward'], sum([result['reward_parts']['SI_threshold'], result['reward_parts']['current_SGM'], -result['reward_parts']['deviation_penalty']]))
        restored = core.restore_snapshot(ctx, result['next_snapshot'])
        self.assertEqual(core.content_sha(restored), core.content_sha(result['next_snapshot']))
        self.assertEqual(len(self.calls), 1)

    def test_occupied_and_future_slot_reserve(self):
        ctx = context(3, [0, 0, 0]); state = core.initial_snapshot(ctx)
        state['occupied'][0, :98] = True
        # Constraint checker itself needs no physical call; two remaining free
        # slots cannot allocate even one while reserving two future beams.
        spec = core.candidate_spec(ctx, state, 98, 1, 5., 'minimum5')
        self.assertIn('current_FL_mask', spec['invalid_reasons'])
        self.assertEqual(self.calls, [])

    def test_invalid_points_are_not_clipped_or_called(self):
        ctx = context(); state = core.initial_snapshot(ctx)
        spec = core.candidate_spec(ctx, state, 99, 2, .5, 'fine_power')
        result = core.evaluate_candidate(ctx, state, spec, self.meter, 'must-not-call')
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['spec']['power_w'], .5)
        self.assertEqual(result['spec']['count'], 2)
        self.assertEqual(self.calls, [])

    def test_relaxed_result_not_original_resume(self):
        ctx = context(); state = core.initial_snapshot(ctx)
        spec = core.candidate_spec(ctx, state, 0, 1, .5, 'fine_power')
        result = core.evaluate_candidate(ctx, state, spec, self.meter, 'relaxed0')
        self.assertEqual(result['spec']['domain'], 'lower_bound_relaxation')
        with self.assertRaisesRegex(ValueError, 'original-prefix'):
            core.restore_snapshot(ctx, result['next_snapshot'])

    def test_exact_identity_and_dtype(self):
        ctx = context(); state = core.initial_snapshot(ctx)
        state['rates'] = state['rates'].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'rates'):
            core.restore_snapshot(ctx, state)

    def test_power_reserve_rejection_and_bool(self):
        ctx = context(); state = core.initial_snapshot(ctx)
        state['powers'][0] = 2990.
        spec = core.candidate_spec(ctx, state, 0, 1, 5., 'fine_power')
        self.assertIn('future_5W_reserve', spec['invalid_reasons'])
        self.assertNotIn('total_power_budget', spec['invalid_reasons'])
        with self.assertRaises(ValueError):
            core.candidate_spec(ctx, state, 0, 1, True, 'fine_power')

    def test_frozen_mask_equivalence_without_environment_constructor(self):
        from satellite_rl.t0_width.environment import T0WidthEnv
        ctx = context(4, [0, 0, 2, 2]); state = core.initial_snapshot(ctx)
        for step, action in enumerate(([0, 1, 1], [2, 2, 0], [0, 3, 2])):
            frozen = T0WidthEnv.__new__(T0WidthEnv)  # constructor would build a real kernel
            frozen.N = ctx.N; frozen.indices = ctx.indices; frozen.groups = ctx.groups
            frozen.beam_idx = state['beam_idx']; frozen.occupied = state['occupied']; frozen.powers = state['powers']
            reference = frozen.action_mask()
            np.testing.assert_array_equal(core.legal_frequency_slots(ctx, state), reference['frequency_slots'])
            for pi in range(10):
                spec = core.candidate_spec(ctx, state, action[0], action[1]+1, (pi+1)*5., 'original')
                self.assertEqual(spec['valid'], bool(reference['frequency_slots'][action[0], action[1]] and reference['power'][pi]))
            state = core.replay_original_action(ctx, state, action, self.meter, 'stub'+str(step))['next_snapshot']

    def test_intent_rejection_prevents_physical_operation(self):
        ctx = context(); state = core.initial_snapshot(ctx)
        spec = core.candidate_spec(ctx, state, 0, 1, 5., 'minimum5')
        def rejected(kind, call_id, operation):
            raise RuntimeError('budget refused before operation')
        with self.assertRaisesRegex(RuntimeError, 'budget refused'):
            core.evaluate_candidate(ctx, state, spec, rejected, 'denied')
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
