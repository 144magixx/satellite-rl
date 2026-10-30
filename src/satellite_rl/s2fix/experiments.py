"""Per-scene evaluation shared by the command and the two-scene smoke check."""
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .data import load_input
from .evaluate import metrics, run_sca
from .greedy import greedy_allocation


@lru_cache(maxsize=None)
def tsac_policy(path, device):
    from .tsac import load_policy
    return load_policy(path, device)


def evaluate_scene(job):
    index, record, methods, models, directory, device, threads = job
    torch.set_num_threads(threads)
    loading_seconds = {}
    for method in methods:
        if method not in ('L1', 'L2', 'L3'):
            continue
        if method == 'L1':
            from .i0 import load_policy as loader
        else:
            loader = tsac_policy
        for _, path in models[method]:
            misses = loader.cache_info().misses
            loading_started = perf_counter()
            loader(path, device)
            loading_seconds[method, path] = (perf_counter() - loading_started
                if loader.cache_info().misses > misses else 0.)
    stage_input = load_input(record)
    started = perf_counter()
    initial = greedy_allocation(stage_input)
    initial_seconds = perf_counter() - started
    rows = []
    for method in methods:
        if method in ('L1', 'L2', 'L3'):
            runs = [(seed, repeat, path) for seed, path in models[method] for repeat in range(3)]
        elif method == 'R':
            runs = [(seed, 0, None) for seed in (51001, 51002, 51003)]
        else:
            runs = [(0, 0, None)]
        for seed, repeat, path in runs:
            action_seed = seed if method == 'R' else 274000 + index + repeat*1000000
            search_started = perf_counter()
            probes, decisions = 0, 0
            if method == 'G1':
                allocation = initial
            elif method in ('R', 'S1', 'S2'):
                from .search_env import run_search
                allocation, info = run_search(stage_input, method, action_seed, return_info=True)
                probes, decisions = info['probes'], info['decisions']
            elif method == 'L1':
                from .i0 import policy
                allocation, info = policy(stage_input, path, action_seed, return_info=True, device=device)
                probes, decisions = info['probes'], info['decisions']
            else:
                from .tsac import rollout
                allocation = rollout(stage_input, tsac_policy(path, device), action_seed)
            search_seconds = perf_counter() - search_started
            groups, starts, counts, powers = allocation
            before = metrics(stage_input, *allocation)
            optimized = run_sca(stage_input, *allocation)
            after_power = np.asarray(optimized['power_w'], dtype=float)
            after = metrics(stage_input, groups, starts, counts, after_power)
            model_loading_seconds = loading_seconds.get((method, path), 0.) if repeat == 0 else 0.
            total_seconds = model_loading_seconds + initial_seconds + perf_counter() - search_started
            basename = f'{index:04d}_{method}_s{seed}_r{repeat}.npz'
            np.savez_compressed(Path(directory) / 'beams' / basename,
                beam_ids=np.asarray(stage_input.layout['beam_ids']), groups=groups, starts=starts,
                counts=counts, power_before=powers, power_after=after_power,
                demand_bps=stage_input.layout['request_bps_by_beam_int64'],
                aircraft_count=stage_input.layout['aircraft_count_by_beam'],
                capacity_before=before['capacity_bps'], capacity_after=after['capacity_bps'],
                served_before=np.minimum(before['capacity_bps'],stage_input.layout['request_bps_by_beam_int64']),
                served_after=np.minimum(after['capacity_bps'],stage_input.layout['request_bps_by_beam_int64']))
            row = dict(set=record['set_name'], scene_id=record['case_id'], date=stage_input.case['UTC'][:10],
                method=method, seed=seed, repeat=repeat, action_seed=action_seed,
                N=len(groups), probes=probes, decisions=decisions,
                model_loading_seconds=model_loading_seconds,
                initial_seconds=initial_seconds, search_seconds=search_seconds,
                sca_seconds=optimized['sca_seconds'], total_seconds=total_seconds,
                beams=str((Path(directory)/'beams'/basename).resolve()))
            for prefix, value in (('before', before), ('after', after)):
                row.update({f'{prefix}_{k}':v for k,v in value.items() if np.isscalar(v)})
            same_group_overlap = any(groups[i] == groups[j] and
                max(starts[i],starts[j]) < min(starts[i]+counts[i],starts[j]+counts[j])
                for i in range(len(groups)) for j in range(i))
            row['constraints_ok'] = bool(not same_group_overlap and np.all(counts>=1) and
                np.all(counts<=10) and np.all(starts>=0) and np.all(starts+counts<=100) and
                np.all((groups>=0)&(groups<8)) and after_power.max()<=50+1e-7 and
                after_power.sum()<=3000+1e-7 and powers.max()<=50+1e-7 and powers.sum()<=3000+1e-7)
            rows.append(row)
    return rows
