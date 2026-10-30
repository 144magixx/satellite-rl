"""Random legal construction and the unchanged S2-FIX tabu neighbourhood."""
from collections import deque
from time import perf_counter
import numpy as np
from satellite_rl.shared_interface.link import prepare_slot_kernel
from satellite_rl.shared_interface.geometry import directions_and_distance
from satellite_rl.s2fix.greedy import proportional_power, greedy_allocation
from satellite_rl.s2fix.search_env import SearchEnv, layout_key, rule_priority


def isolated_slot_counts(stage_input):
    """Size each beam independently; no allocation candidate is ranked."""
    layout, config = stage_input.layout, stage_input.physics
    kernel = prepare_slot_kernel(layout, config, stage_input.case)
    stage_input._s2fix_slot_kernel = kernel
    demand = np.asarray(layout['request_bps_by_beam_int64'])
    estimated_power = np.minimum(config['power_budget_w'] * demand / demand.sum(), 50.)
    _, distance = directions_and_distance(layout['centres_lat_lon_deg'], validate=False)
    coefficient = (10**(-config['signal_loss_dB']/10) * np.diag(kernel['gain']) *
                   10**(config['rx_gain_dBi']/10) * (config['c_m_per_s'] /
                   (4*np.pi*distance*config['f0_Hz']))**2)
    noise = config['k_J_per_K'] * config['noise_temperature_K'] * config['slot_Hz']
    slots = np.full(len(demand), 10, int)
    for k in range(1, 11):
        snr = coefficient * estimated_power / k / noise
        capacity = k * config['slot_Hz'] * np.minimum(np.log2(1+snr *
            10**(-config['implementation_loss_dB']/10)), config['spectral_efficiency_cap'])
        slots[(slots == 10) & (capacity >= demand)] = k
    return slots


def random_allocation(stage_input, seed):
    """Random beam order, cyclic groups, then contiguous packing in that order."""
    demand = np.asarray(stage_input.layout['request_bps_by_beam_int64'])
    slots = isolated_slot_counts(stage_input)
    order = np.random.default_rng(seed).permutation(len(demand))
    groups, starts = np.empty(len(demand), int), np.zeros(len(demand), int)
    groups[order] = np.arange(len(order)) % 8
    for group in range(8):
        members = order[groups[order] == group]
        # Same proportional integer quota as G1 if this group exceeds 100 slots.
        if slots[members].sum() > 100:
            quota = (100-len(members)) * demand[members] / demand[members].sum()
            count = np.minimum(slots[members], 1+np.floor(quota).astype(int))
            while count.sum() < 100 and np.any(count < slots[members]):
                score = np.where(count < slots[members], quota-(count-1), -np.inf)
                count[np.argmax(score)] += 1
            slots[members] = count
        starts[members] = np.cumsum(slots[members])-slots[members]
    return groups, starts, slots, proportional_power(demand)


class InitializedSearchEnv(SearchEnv):
    def reset_allocation(self, stage_input, allocation):
        """Same SearchEnv reset, with an explicit independently built allocation."""
        self.stage_input = stage_input
        self.groups, self.current_starts, self.current_slots, self.power = [
            np.asarray(x).copy() for x in allocation]
        self.demand = np.asarray(stage_input.layout['request_bps_by_beam_int64'], float)
        self.locations = np.asarray(stage_input.layout['centres_lat_lon_deg'], float)
        self.scene = {'beam_location': self.locations}
        self.current_eval = self._evaluate(self.groups, self.current_starts, self.current_slots)
        self.base_eval = self.best_eval = self.current_eval
        self.base_basic = self.current_eval.capacity_bps >= .1*self.demand
        self.best_groups = self.groups.copy()
        self.best_starts, self.best_slots = self.current_starts.copy(), self.current_slots.copy()
        self.decision_count = self.probe_count = self.invalid_count = self.repeat_count = 0
        self.proposals_generated = 0
        self.done = False
        self.trace = []
        self._candidate_cache = None
        self.candidate_generation_seconds = self.probe_seconds = 0.


def initialize(stage_input, kind, seed, budget):
    allocation = (random_allocation(stage_input, seed) if kind == 'random'
                  else greedy_allocation(stage_input))
    env = InitializedSearchEnv(dict(max_probes=max(1,budget), max_decisions=max(1,budget)))
    env.reset_allocation(stage_input, allocation)
    return env, allocation


def tabu_search(env, budget):
    """Original run_search's tabu path; record every probe and selected move."""
    trace = []
    tabu = deque([layout_key(env.groups, env.current_starts, env.current_slots)], maxlen=8)
    while not env.done and env.probe_count < budget:
        choices = [c for c in env.candidates()
                   if layout_key(c.groups,c.starts,c.slots) not in tabu]
        choices = sorted(choices,key=lambda c:rule_priority(env,c),reverse=True)[:min(4,budget-env.probe_count)]
        if not choices:
            break
        outcomes = []
        for candidate in choices:
            evaluation, gain = env.probe(candidate)
            outcomes.append((evaluation,candidate))
            trace.append(dict(event='probe',probe=env.probe_count,decision=env.decision_count,
                kind=candidate.kind,changes=[dict(beam=int(i),group=int(candidate.groups[i]),
                start=int(start),count=int(count)) for i,start,count in candidate.changes],
                W=evaluation.W,best_W=env.best_eval.W,protected=bool(env._protected(evaluation))))
        evaluation, chosen = max(outcomes,key=lambda pair:pair[0].W)
        env.move(chosen,evaluation)
        tabu.append(layout_key(chosen.groups,chosen.starts,chosen.slots))
        env.decision_count += 1
        trace.append(dict(event='move',decision=env.decision_count,candidate_id=chosen.id,W=evaluation.W))
        env.done = env.probe_count >= budget or env.decision_count >= (budget+3)//4
    return env.allocation(), trace
