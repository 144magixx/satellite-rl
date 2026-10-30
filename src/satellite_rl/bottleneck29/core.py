"""A0 arithmetic with explicit metered physical calls and replay-free snapshots.

``meter(kind, call_id, operation)`` must durably reserve an intent before invoking
operation and record completion afterwards. An interrupted intent is not replayable.
This module does not implement persistence, retry, dispatch, or budget ownership.
All physical/service floating arrays are explicitly float64, as in frozen W26.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from satellite_rl.shared_interface.link import prepare_slot_kernel, build_link_coefficients, capacity_bps
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
from satellite_rl.t0_width.environment import sgm_values, legacy_repaired_reward
from satellite_rl.t0_width.logging import content_sha


@dataclass(frozen=True)
class Context:
    layout: dict
    physics: dict
    ids: tuple
    indices: np.ndarray
    groups: np.ndarray
    demands: np.ndarray
    kernel: dict
    identity: dict

    @property
    def N(self):
        return len(self.ids)


def prepare_context(stage_input, meter, call_id):
    """One charged static kernel build; input must already pass the shared importer."""
    from satellite_rl.shared_interface.stages import UniformStage2Input
    if not isinstance(stage_input, UniformStage2Input):
        raise TypeError('validated UniformStage2Input required')
    ids = tuple(stage_input.layout['beam_ids'])
    order = tuple(stage_input.processing_index_to_beam_id)
    indices = np.asarray([ids.index(b) for b in order], dtype=np.int64)
    groups = np.asarray([stage_input.group_by_beam_id[b] for b in ids], dtype=np.int64)
    demands = np.asarray(stage_input.layout['request_bps_by_beam_int64'], dtype=np.float64).copy()
    if not 0 < len(ids) <= 200 or len(set(order)) != len(ids) or set(order) != set(ids):
        raise ValueError('complete variable-N identity required')
    identity = dict(layout_SHA=stage_input.layout['content_sha256'], physics_SHA=PHYSICS_SHA,
                    order=list(order), groups=groups.tolist())
    kernel = meter('kernel', call_id, lambda: prepare_slot_kernel(stage_input.layout, stage_input.physics))
    for value in (indices, groups, demands):
        value.setflags(write=False)
    return Context(stage_input.layout, stage_input.physics, ids, indices, groups, demands, kernel, identity)


def initial_snapshot(context):
    n = context.N
    return dict(schema='W29A0PrefixV1', identity=deepcopy(context.identity), beam_idx=0,
                starts=np.zeros(n, dtype=np.int64), counts=np.zeros(n, dtype=np.int64),
                powers=np.zeros(n, dtype=np.float64), occupied=np.zeros((8, 100), dtype=bool),
                rates=np.zeros(n, dtype=np.float64), total_satisfaction=0., requested=[])


def restore_snapshot(context, snapshot):
    """Validate and copy; no kernel build, environment step, or PHY replay."""
    if snapshot.get('schema') != 'W29A0PrefixV1' or snapshot['identity'] != context.identity:
        raise ValueError('prefix source identity mismatch')
    result = deepcopy(snapshot)
    k = result['beam_idx']
    if type(k) is not int or not 0 <= k <= context.N or len(result['requested']) != k:
        raise ValueError('prefix processing identity mismatch')
    for key, dtype in (('starts', np.int64), ('counts', np.int64), ('powers', np.float64), ('rates', np.float64)):
        array = result[key]
        if not isinstance(array, np.ndarray) or array.shape != (context.N,) or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError('invalid prefix array: ' + key)
    if result['occupied'].shape != (8, 100) or result['occupied'].dtype != bool:
        raise ValueError('invalid prefix occupancy')
    if (result['rates'] < 0).any() or (result['powers'] < 0).any():
        raise ValueError('negative prefix service or power')
    processed = context.indices[:k]
    future = context.indices[k:]
    if (result['counts'][processed] < 1).any() or (result['counts'][processed] > 10).any():
        raise ValueError('invalid prefix slot count')
    if any((result[key][future] != 0).any() for key in ('starts', 'counts', 'powers', 'rates')):
        raise ValueError('unprocessed beam has allocation or service')
    occupied = np.zeros((8, 100), dtype=bool)
    for position, index in enumerate(processed):
        start, count = int(result['starts'][index]), int(result['counts'][index])
        power = float(result['powers'][index])
        if not 0 <= start <= 100-count or power not in tuple(range(5, 51, 5)):
            raise ValueError('invalid original-prefix action')
        group = context.groups[index]
        if occupied[group, start:start+count].any():
            raise ValueError('prefix slot conflict')
        occupied[group, start:start+count] = True
        expected_request = dict(beam_id=context.ids[index], start=start, count=count, power_w=power)
        if result['requested'][position] != expected_request:
            raise ValueError('prefix saved action/identity mismatch')
    if not np.array_equal(occupied, result['occupied']) or float(result['powers'].sum()) > 3000:
        raise ValueError('prefix resource mismatch')
    if float(result['powers'].sum())+5*(context.N-k) > 3000:
        raise ValueError('prefix violates original future power reserve')
    expected = float(np.minimum(result['rates'][processed]/context.demands[processed], 1).sum())
    if expected != result['total_satisfaction']:
        raise ValueError('prefix satisfaction mismatch')
    return result


def prefix_positions(n):
    if type(n) is not int or not 1 <= n <= 200:
        raise ValueError('N outside actual domain')
    return sorted(set(min(n-1, math.floor(fraction*n)) for fraction in (.125, .5, .875)))


def legal_frequency_slots(context, snapshot):
    """Exact frozen current-state FL mask, including future one-slot reserve."""
    k = snapshot['beam_idx']
    mask = np.zeros((100, 10), dtype=bool)
    if k == context.N:
        return mask
    current = context.indices[k]
    group = context.groups[current]
    remaining = context.indices[k:]
    reserve = int(np.count_nonzero(context.groups[remaining] == group))-1
    free = ~snapshot['occupied'][group]
    for count in range(1, min(10, int(free.sum())-reserve)+1):
        for start in range(101-count):
            mask[start, count-1] = bool(free[start:start+count].all())
    return mask


def candidate_spec(context, snapshot, start, count, power_w, family):
    """Keep requested continuous watts; invalid points are explicit, never clipped."""
    reasons = []
    if family not in ('original', 'minimum5', 'fine_power'):
        raise ValueError('unknown candidate family')
    if type(start) is not int or type(count) is not int or not 0 <= start < 100 or not 1 <= count <= 10 or start+count > 100:
        reasons.append('slot_range_or_type')
    if isinstance(power_w, (bool, np.bool_)) or not isinstance(power_w, (int, float, np.integer, np.floating)) or not math.isfinite(float(power_w)):
        raise ValueError('finite nonboolean power required')
    power_w = float(power_w)
    original = power_w in tuple(range(5, 51, 5))
    domain = 'original' if original else ('lower_bound_relaxation' if power_w < 5 else 'granularity_relaxation')
    if family in ('original', 'minimum5') and not original:
        reasons.append('not_original_power')
    if family == 'minimum5' and power_w != 5:
        reasons.append('not_minimum5')
    if not .5 <= power_w <= 50 or power_w*2 != int(power_w*2):
        reasons.append('outside_diagnostic_half_watt_grid')
    k = snapshot['beam_idx']
    if k >= context.N:
        reasons.append('terminal_prefix')
    else:
        if 'slot_range_or_type' not in reasons:
            group = context.groups[context.indices[k]]
            free = ~snapshot['occupied'][group]
            reserve = int(np.count_nonzero(context.groups[context.indices[k:]] == group))-1
            if not bool(free[start:start+count].all()) or count > int(free.sum())-reserve:
                reasons.append('current_FL_mask')
        total = float(snapshot['powers'].sum())+power_w
        if total > 3000:
            reasons.append('total_power_budget')
        if total+5*(context.N-k-1) > 3000:
            reasons.append('future_5W_reserve')
    return dict(start=start, count=count, power_w=power_w, family=family,
                domain=domain, valid=not reasons, invalid_reasons=reasons)


def candidate_specs(context, snapshot, original_action):
    """Return two fixed scan lists; duplicate physical points remain identifiable."""
    state = restore_snapshot(context, snapshot)
    start, count, power = decode_action(original_action)
    original_spec = candidate_spec(context, state, start, count, power, 'original')
    if not original_spec['valid']:
        raise ValueError('original saved decision is not legal in exact prefix')
    minimum = [candidate_spec(context, state, f, count, 5., 'minimum5')
               for f in range(100) for count in range(1, 11) if f+count <= 100]
    fine = [candidate_spec(context, state, start, count, half/2., 'fine_power') for half in range(1, 101)]
    return dict(minimum5=minimum, fine_power=fine)


def decode_action(action):
    """Frozen saved action indices (f,l,p) -> actual start/count/whole-beam watts."""
    if len(action) != 3 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in action):
        raise ValueError('three integer original action indices required')
    f, l, p = map(int, action)
    if not 0 <= f < 100 or not 0 <= l < 10 or not 0 <= p < 10:
        raise ValueError('original action index outside domain')
    return f, l+1, float((p+1)*5)


def evaluate_candidate(context, snapshot, spec, meter, call_id):
    """One metered shared-PHY call. Input snapshot remains byte-for-byte intact.

    Returned next_snapshot is only resumable as an original trajectory when the
    candidate is an original action. Counterfactual relaxed snapshots are outputs,
    not inputs for continuing the original-prefix API.
    """
    state = restore_snapshot(context, snapshot)
    checked = candidate_spec(context, state, spec['start'], spec['count'], spec['power_w'], spec['family'])
    if checked != spec:
        raise ValueError('candidate metadata or constraint result changed')
    if not spec['valid']:
        return dict(status='invalid', spec=deepcopy(spec), physical_calls=0)
    k = state['beam_idx']; current = int(context.indices[k]); group = int(context.groups[current])
    start, count, power = spec['start'], spec['count'], float(spec['power_w'])
    before_rates = state['rates'].copy()
    before_satisfaction = state['total_satisfaction']
    state['starts'][current] = start; state['counts'][current] = count; state['powers'][current] = power
    state['occupied'][group, start:start+count] = True
    allocation = dict(beam_ids=list(context.ids), actual_start_slot=state['starts'],
                      actual_slot_count=state['counts'], actual_whole_beam_power_w=state['powers'],
                      group_by_beam_id={b:int(context.groups[i]) for i,b in enumerate(context.ids)},
                      content_sha256='W29_internal_partial_not_published')
    def physical():
        coeff = build_link_coefficients(context.layout, allocation, context.physics, kernel=context.kernel)
        return np.asarray(capacity_bps(coeff), dtype=np.float64)
    rates = np.asarray(meter('phy', call_id, physical), dtype=np.float64)
    if rates.shape != (context.N,) or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('invalid shared physical output')
    beta = float(context.demands.mean())
    si = np.minimum(rates/context.demands, 1)
    sgm = sgm_values(rates, context.demands, beta)
    processed = context.indices[:k+1]; previous = context.indices[:k]
    satisfaction = float(si[processed].sum())
    increment = satisfaction-before_satisfaction
    threshold = 1. if increment >= 1 else -1.
    deviation = max(abs(float(rates[current])-float(context.demands[current]))-5e7, 0.)/1e9
    reward = legacy_repaired_reward(float(rates[current]), float(context.demands[current]), beta, increment, 0, count)
    action = dict(beam_id=context.ids[current], start=start, count=count, power_w=power)
    state.update(beam_idx=k+1, rates=rates.copy(), total_satisfaction=satisfaction)
    state['requested'].append(deepcopy(action))
    before_si = np.minimum(before_rates/context.demands, 1)
    before_sgm = sgm_values(before_rates, context.demands, beta)
    return dict(status='completed', spec=deepcopy(spec), physical_calls=1, beam_id=context.ids[current],
                requested_action=action, actual_action=deepcopy(action), next_snapshot=state,
                capacity_bps=rates.copy(), demand_bps=context.demands.copy(), SI=si, SGM=sgm,
                current=dict(capacity_bps=float(rates[current]), demand_bps=float(context.demands[current]),
                             SI=float(si[current]), SGM=float(sgm[current])),
                previous_prefix=dict(indices=previous.copy(), SI_delta=(si-before_si)[previous],
                                     SGM_delta=(sgm-before_sgm)[previous]),
                prefix=dict(indices=processed.copy(), SI_sum=satisfaction, SGM_sum=float(sgm[processed].sum())),
                reward=float(reward), reward_parts=dict(SI_threshold=threshold, current_SGM=float(sgm[current]),
                    deviation_penalty=deviation, occupancy_penalty=0., satisfaction_increment=increment),
                constraints=dict(total_power_w=float(state['powers'].sum()),
                    future_reserved_power_w=5*(context.N-k-1), slot_conflicts=0,
                    requested_execution_equal=True, domain=spec['domain']),
                base_snapshot_SHA=content_sha(snapshot))


def replay_original_action(context, snapshot, action, meter, call_id):
    start, count, power = decode_action(action)
    spec = candidate_spec(context, snapshot, start, count, power, 'original')
    result = evaluate_candidate(context, snapshot, spec, meter, call_id)
    if result['status'] != 'completed':
        raise ValueError('saved original action is not legal in exact prefix: '+repr(spec['invalid_reasons']))
    return result
