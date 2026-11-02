"""Literature methods with the same metrics and beam output as S2-FIX."""
from pathlib import Path
from time import perf_counter
import numpy as np

from satellite_rl.s2fix.data import load_input
from satellite_rl.s2fix.evaluate import metrics, run_sca
from satellite_rl.s2fix.greedy import greedy_allocation
from .data import records
from .moves import Alloc, legal, uniform_start
from .objective import Probe, POINTS, j_gap, j_w
from .anneal import anneal
from .genetic import genetic
from .convex_relax import convex_relax
from .power_lit import C1, C2

METHODS = ('B1_32', 'B1_320', 'B1_3200', 'B1W_320', 'B1W_3200',
           'B2_32', 'B2_320', 'B2_3200', 'B2W_320', 'B2W_3200',
           'B3', 'P0', 'C1', 'C2', 'E1_320', 'E1_3200')


def evaluate_scene(job):
    index, method, seed = job['index'], job['method'], job['seed']
    record = records(job['set_dir'])[index]
    stage = load_input(record)
    return evaluate_input(stage, record, index, method, seed, job.get('output_dir'))


def evaluate_input(stage, record, index, method, seed, output_dir=None):
    started = perf_counter()
    allocation = uniform_start(stage) if method.startswith('E1') else Alloc(*greedy_allocation(stage))
    t_init = perf_counter() - started
    probe = Probe(stage)
    search_started = perf_counter()
    if method.startswith(('B1', 'B2', 'E1')):
        objective = j_w if 'W' in method else j_gap
        search = genetic if method.startswith('B2') else anneal
        allocation = search(stage, allocation, probe, int(method.split('_')[1]), seed, objective)
    elif method == 'B3':
        allocation = convex_relax(stage, allocation)
    t_search = perf_counter() - search_started
    before = metrics(stage, *allocation)
    refine_started = perf_counter()
    if method == 'P0':
        power = allocation.powers
    elif method == 'C1' or method.startswith('E1'):
        power = C1(stage, allocation)
    elif method == 'C2':
        power = C2(stage, allocation, seed, probe)
    else:
        power = np.asarray(run_sca(stage, *allocation)['power_w'])
    t_refine = perf_counter() - refine_started
    after = metrics(stage, allocation.groups, allocation.starts, allocation.counts, power)
    total = perf_counter() - started
    row = dict(set=record['set_name'], scene_id=record['case_id'], date=stage.case['UTC'][:10],
        method=method, seed=seed, repeat=0, action_seed=seed, N=len(power),
        probes=probe.calls, best_at=probe.best_at, decisions=probe.calls, model_loading_seconds=0.,
        initial_seconds=t_init, search_seconds=t_search, sca_seconds=t_refine, total_seconds=total,
        t_init=t_init, t_search=t_search, t_refine=t_refine, t_total=total, beams='')
    for prefix, values in (('before', before), ('after', after)):
        row.update({f'{prefix}_{key}': value for key, value in values.items() if np.isscalar(value)})
    row.update({f'w_at_{point}': probe.curve.get(point, '') for point in POINTS})
    row['constraints_ok'] = bool(legal(allocation) and np.all(power >= 0) and
        np.all(power <= 50 + 1e-7) and power.sum() <= 3000 + 1e-7 and
        np.all(allocation.powers >= 0) and np.all(allocation.powers <= 50 + 1e-7) and
        allocation.powers.sum() <= 3000 + 1e-7)
    if output_dir is not None:
        path = Path(output_dir) / 'beams' / f'{index:04d}_{method}_s{seed}_r0.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        demand = stage.layout['request_bps_by_beam_int64']
        np.savez_compressed(path, beam_ids=stage.layout['beam_ids'], groups=allocation.groups,
            starts=allocation.starts, counts=allocation.counts, power_before=allocation.powers,
            power_after=power, demand_bps=demand, aircraft_count=stage.layout['aircraft_count_by_beam'],
            capacity_before=before['capacity_bps'], capacity_after=after['capacity_bps'],
            served_before=np.minimum(before['capacity_bps'], demand),
            served_after=np.minimum(after['capacity_bps'], demand))
        row['beams'] = str(path.resolve())
    return row
