"""Common v2 service metrics and fixed-allocation continuous power refinement."""
from time import perf_counter
import numpy as np
from satellite_rl.shared_interface.link import prepare_slot_kernel,build_link_coefficients
from satellite_rl.power_refinement import FixedAllocationPhysics,optimize_power


def fixed_problem(stage_input, groups, starts, counts, powers):
    layout=stage_input.layout
    ids=list(layout['beam_ids'])
    allocation=dict(beam_ids=ids,group_by_beam_id=dict(zip(ids,map(int,groups))),
        actual_start_slot=np.asarray(starts),actual_slot_count=np.asarray(counts),
        actual_whole_beam_power_w=np.asarray(powers,float))
    kernel=getattr(stage_input,'_s2fix_slot_kernel',None)
    if kernel is None:
        kernel=prepare_slot_kernel(layout,stage_input.physics,stage_input.case)
        stage_input._s2fix_slot_kernel=kernel
    coefficients=build_link_coefficients(layout,allocation,stage_input.physics,kernel)
    config=stage_input.physics
    return FixedAllocationPhysics(coefficients.demand_bps,np.asarray(groups),np.asarray(starts),
        np.asarray(counts),coefficients.signal,coefficients.interference,coefficients.noise,
        coefficients.slot_owner,config['power_budget_w'],config['spectral_efficiency_cap'],
        config['implementation_loss_dB'])


def capacities(stage_input, groups, starts, counts, powers):
    return fixed_problem(stage_input,groups,starts,counts,powers).evaluate(powers).capacity_bps


def metrics(stage_input, groups, starts, counts, powers):
    problem=fixed_problem(stage_input,groups,starts,counts,powers)
    evaluation=problem.evaluate(powers)
    demand=problem.demand_bps
    aircraft=np.asarray(stage_input.layout['aircraft_count_by_beam'])
    service=np.repeat(evaluation.beam_si,aircraft)
    worst=max(1,int(np.ceil(.1*len(service))))
    overlap=0
    for i in range(len(groups)):
        for j in range(i):
            if groups[i]%2==groups[j]%2 and groups[i]!=groups[j]:
                overlap+=max(0,min(starts[i]+counts[i],starts[j]+counts[j])-max(starts[i],starts[j]))
    return dict(W=float(np.minimum(evaluation.capacity_bps,demand).sum()/demand.sum()),
        SI=evaluation.si,SGM=evaluation.sgm,
        aircraft_worst10=float(np.partition(service,worst-1)[:worst].mean()),
        aircraft_ge90=float(np.mean(service>=.9)),total_power_w=float(np.sum(powers)),
        total_slots=int(np.sum(counts)),overlap_slots=int(overlap),
        capacity_bps=evaluation.capacity_bps,beam_si=evaluation.beam_si,
        beam_sgm=evaluation.beam_sgm)


def run_sca(stage_input, groups, starts, counts, powers, options=None):
    problem=fixed_problem(stage_input,groups,starts,counts,powers)
    opts=dict(maxiter=200,time_limit_seconds=30.,beam_power_cap_w=50.)
    opts.update(options or {})
    started=perf_counter()
    result=optimize_power(problem,powers,'sca',opts)
    result['sca_seconds']=perf_counter()-started
    return result
