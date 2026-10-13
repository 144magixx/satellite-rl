"""Independent variable-layout discrete resource environment using the shared PHY.

Actions never contain power: the declared per-slot rule supplies whole-beam W.
An exported allocation is validated but must be independently published before S3.
"""
from copy import deepcopy
import numpy as np
from satellite_rl.shared_interface.stages import UniformStage2Input
from satellite_rl.shared_interface.link import build_link_coefficients, capacity_bps
from satellite_rl.shared_interface.geometry import directions_and_distance
from satellite_rl.shared_interface.spherical_gaussian import linear_gain
from satellite_rl.power_refinement.physics import _sgm
from .schema import (TOKEN_DIM, RELATION_DIM, GLOBAL_DIM, ACTION_DIM, STATE_SCHEMA,
    ACTION_SCHEMA, PHYSICS_SCHEMA, TOKEN_FEATURES, RELATION_FEATURES,
    POWER_RULE_W_PER_SLOT, REWARD_SCHEMA, observation_semantics)


def encode_action(start, count):
    if isinstance(start, (bool, np.bool_)) or isinstance(count, (bool, np.bool_)) or not isinstance(start, (int, np.integer)) or not isinstance(count, (int, np.integer)):
        raise ValueError('integer start/count required')
    if count == 0 and start == 0:
        return 0
    if not 1 <= count <= 10 or not 0 <= start <= 100-count:
        raise ValueError('invalid frequency action')
    return 1 + int(start)*10 + int(count)-1


def decode_action(action):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < ACTION_DIM:
        raise ValueError('joint action integer outside range')
    if action == 0:
        return 0, 0
    start, index = divmod(int(action)-1, 10)
    return start, index+1


class LayoutResourceEnv:
    def __init__(self, stage2_input, config=None):
        if not isinstance(stage2_input, UniformStage2Input):
            raise ValueError('validated UniformStage2Input required')
        self.stage = UniformStage2Input(stage2_input.case, stage2_input.layout,
            stage2_input.authority, stage2_input.physics)
        self.config = dict(power_w_per_slot=POWER_RULE_W_PER_SLOT, reward=REWARD_SCHEMA)
        if config is not None:
            if set(config)-set(self.config):
                raise ValueError('unknown environment config')
            self.config.update(config)
        power = self.config['power_w_per_slot']
        if isinstance(power, (bool, np.bool_)) or not isinstance(power, (int, float)) or power != POWER_RULE_W_PER_SLOT:
            raise ValueError('frozen 3.75 W per-slot rule required')
        if self.config['reward'] != REWARD_SCHEMA:
            raise ValueError('unsupported reward')
        self.beam_ids = tuple(self.stage.processing_index_to_beam_id)
        self.N = len(self.beam_ids)
        self._layout_indices = [self.stage.layout['beam_ids'].index(b) for b in self.beam_ids]
        self.demands = self.stage.layout['request_bps_by_beam_int64'][self._layout_indices].copy()
        self.groups = np.array([self.stage.group_by_beam_id[b] for b in self.beam_ids])
        centres = self.stage.layout['centres_lat_lon_deg'][self._layout_indices]
        self.widths = self.stage.layout['full_width_deg'][self._layout_indices].copy()
        rays, self.distances = directions_and_distance(centres)
        chord = np.linalg.norm(rays[:, None]-rays[None, :], axis=-1)
        angles = 2*np.arcsin(np.minimum(chord/2, 1))
        gain, _ = linear_gain(angles, np.deg2rad(self.widths)[None, :])
        self.peak_gain, _ = linear_gain(np.zeros(self.N), np.deg2rad(self.widths))
        self.relations = np.stack((angles/np.pi, gain/1e5,
            ((self.groups[:, None]%2 == self.groups[None, :]%2) &
             (self.groups[:, None] != self.groups[None, :])),
            np.zeros((self.N, self.N))), axis=-1).astype(np.float32)
        self.reset()

    def reset(self, seed=None):
        # Deterministic fixed input; seed is accepted without touching global RNG.
        self.current_index = 0
        self.starts = np.zeros(self.N, np.int64)
        self.counts = np.zeros(self.N, np.int64)
        self.powers = np.zeros(self.N, np.float64)
        self.occupancy = np.zeros((8, 100), bool)
        self.requested_actions = []
        self.capacities = np.zeros(self.N, np.float64)
        self.done = False
        return self.observation()

    def action_mask(self):
        mask = np.zeros(ACTION_DIM, bool)
        mask[0] = True  # Absorbing terminal observation still has a safe no-op.
        if self.done:
            return mask
        occupied = self.occupancy[self.groups[self.current_index]]
        remaining = 3000.-float(self.powers.sum())
        for count in range(1, 11):
            if count*self.config['power_w_per_slot'] > remaining:
                continue
            for start in range(101-count):
                if not occupied[start:start+count].any():
                    mask[encode_action(start, count)] = True
        return mask

    def export_allocation(self):
        rows = {bid: dict(group=int(self.groups[i]), start=int(self.starts[i]),
            count=int(self.counts[i]), power_w=float(self.powers[i]))
            for i, bid in enumerate(self.beam_ids)}
        return self.stage.export_actual_allocation(rows, requested_actions_separate=dict(
            actions=deepcopy(self.requested_actions), rule='whole_beam_power=count*power_w_per_slot',
            power_w_per_slot=float(self.config['power_w_per_slot']), terminal=bool(self.done)))

    def metrics(self):
        ratio = self.capacities/self.demands
        si = np.minimum(ratio, 1.)
        sgm, _ = _sgm(self.capacities, self.demands)
        return dict(beam_ids=self.beam_ids, capacity_bps=self.capacities.copy(),
            demand_bps=self.demands.copy(), si_by_beam=si, sgm_by_beam=sgm,
            required_bps=self.demands.copy(), effective_served_bps=np.minimum(self.capacities, self.demands),
            capacity_demand_ratio=ratio, beam_si=si.copy(), beam_sgm=sgm.copy(),
            SI=float(si.mean()), SGM=float(sgm.mean()), power_w=float(self.powers.sum()),
            occupied_slots=int(self.counts.sum()), allocation_fraction=float(np.mean(self.counts>0)),
            processed_fraction=self.current_index/self.N, terminal=self.done)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after terminal')
        start, count = decode_action(action)
        if not self.action_mask()[int(action)]:
            raise ValueError('masked frequency action; no clipping or repair')
        previous = self.metrics()['SI']
        i = self.current_index
        self.starts[i], self.counts[i] = start, count
        self.powers[i] = count*self.config['power_w_per_slot']
        self.occupancy[self.groups[i], start:start+count] = True
        self.requested_actions.append(dict(beam_id=self.beam_ids[i], action=int(action), start=start, count=count))
        self.current_index += 1
        self.done = self.current_index == self.N
        allocation = self.export_allocation()
        coefficients = build_link_coefficients(self.stage.layout, allocation, self.stage.physics)
        physical = capacity_bps(coefficients)
        # NumPy bincount with empty weights can return int64 for all-no-op.
        self.capacities = np.asarray(physical[self._layout_indices], dtype=np.float64).copy()
        metrics = self.metrics()
        info = dict(metrics=metrics, actual_allocation=allocation if self.done else None,
            actual_action=dict(beam_id=self.beam_ids[i], start=start, count=count, power_w=float(self.powers[i])))
        return self.observation(), metrics['SI']-previous, self.done, info

    def observation(self):
        metrics = self.metrics()
        centres = self.stage.layout['centres_lat_lon_deg'][self._layout_indices]
        indices = np.arange(self.N)
        tokens = np.column_stack((centres[:, 0]/90, centres[:, 1]/180, self.widths,
            self.peak_gain/1e5, self.distances/4e7, self.demands/1e9,
            self.demands/self.demands.sum(), self.groups/7, indices<self.current_index,
            indices==self.current_index, self.starts/100, self.counts/10, self.powers/3000,
            self.capacities/1e9, metrics['si_by_beam'], metrics['sgm_by_beam'])).astype(np.float32)
        global_features = np.array([self.N/200, self.current_index/self.N,
            self.powers.sum()/3000, self.counts.sum()/800, metrics['SI'], metrics['SGM'],
            self.demands.sum()/1e9, self.config['power_w_per_slot']/3000,
            *self.occupancy.reshape(4, 200).mean(axis=1)], np.float32)
        relations = self.relations.copy()
        overlap = np.maximum(0, np.minimum((self.starts+self.counts)[:, None],
            (self.starts+self.counts)[None, :])-np.maximum(self.starts[:, None], self.starts[None, :]))
        relations[:, :, 3] = overlap/10
        return dict(tokens=tokens, relations=relations, global_features=global_features,
            padding_mask=np.zeros(self.N, bool), current_index=self.current_index,
            action_mask=self.action_mask(), beam_ids=self.beam_ids, feature_names=TOKEN_FEATURES,
            relation_names=RELATION_FEATURES, state_schema=STATE_SCHEMA,
            action_schema=ACTION_SCHEMA, physics_schema=PHYSICS_SCHEMA,
            semantics=observation_semantics())
