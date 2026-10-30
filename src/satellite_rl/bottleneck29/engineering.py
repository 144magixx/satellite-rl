"""Bounded qualification only; results never self-authorize science."""
import copy
import time
from pathlib import Path
import numpy as np
from . import core
from .inputs import input_loader
from .ledger import Halt, Ledger
from .worker import ENGINEERING_LIMITS


def select_engineering_mother(mothers,lane):
    """Pure pre-outcome workload proxy over all 15 registered prefix states."""
    workload=[]
    for index,row in enumerate(mothers):
        if index%4!=lane:continue
        for position in core.prefix_positions(row['N']):
            slots=sum(action[1]+1 for action in row['actions_indices_by_processing_order'][:position])
            score=row['N']**2+(slots+10)*(position+1)
            workload.append(dict(mother=index,prefix=position,N=row['N'],prefix_slots=slots,score=score))
    chosen=max(workload,key=lambda r:(r['score'],-r['mother']))['mother']
    return mothers[chosen],workload


def run_engineering(config,active_started=None):
    from satellite_rl.t0_width.environment import T0WidthEnv
    from satellite_rl.t0_width.schema import observation_schema
    from satellite_rl.t0_width.logging import content_sha
    root=Path(config['root'])/'engineering'
    ledger=Ledger(root,ENGINEERING_LIMITS,config['binding'],active_started=active_started)
    def meter(kind,call_id,operation):
        if any(Path(p).exists() for p in config['stop_paths']):raise Halt('pause/stop')
        return ledger.meter('phy_prefix' if kind=='phy' else kind,call_id,operation)
    ledger.begin();started=time.monotonic()
    try:
        row,workload=select_engineering_mother(config['mothers'],config['lane'])
        before=time.monotonic();stage_input,actions=input_loader(config)(row);input_seconds=time.monotonic()-before
        before=time.monotonic();ctx=core.prepare_context(stage_input,meter,'engineering/kernel');kernel_seconds=time.monotonic()-before
        old=T0WidthEnv.__new__(T0WidthEnv)
        old.stage_input=stage_input;old.group='S10';old.schema=observation_schema('S10')
        old.layout=ctx.layout;old.physics=ctx.physics;old.ids=list(ctx.ids)
        old.order=list(stage_input.processing_index_to_beam_id);old.indices=ctx.indices;old.groups=ctx.groups
        old.N=ctx.N;old.demands=ctx.demands;old.beta=float(ctx.demands.mean());old.kernel=ctx.kernel
        old.noise=ctx.physics['k_J_per_K']*ctx.physics['noise_temperature_K']*ctx.physics['slot_Hz']
        old.reset();snapshot=core.initial_snapshot(ctx);positions=core.prefix_positions(ctx.N)
        prefix_seconds=[];candidate_seconds=[];probe_records=[]
        for k in range(positions[-1]+1):
            if k in positions:
                restored=core.restore_snapshot(ctx,snapshot)
                if restored['beam_idx']!=k:raise Halt('restore cursor')
                snapshot_sha=content_sha(snapshot)
                ledger.artifact('snapshot/'+str(k),root/'snapshots'/(str(k)+'.pkl'),snapshot)
                tables=core.candidate_specs(ctx,snapshot,actions[k])
                watts=[.5,1.,5.,10.,20.,30.,40.,50.]
                original_power=core.decode_action(actions[k])[2]
                if original_power not in watts:watts[5]=original_power
                probes=[s for s in tables['fine_power'] if s['power_w'] in watts]
                probes += sorted((s for s in tables['minimum5'] if s['valid']),key=lambda s:(-s['count'],s['start']))[:8]
                for i,spec in enumerate(probes):
                    before=time.monotonic();call_id='engineering/durable/'+str(k)+'/'+str(i)
                    result=core.evaluate_candidate(ctx,snapshot,spec,meter,call_id)
                    result.pop('next_snapshot',None)
                    ledger.artifact(call_id,root/'rows'/(str(k)+'_'+str(i)+'.pkl'),
                        dict(row_id=call_id,snapshot_sha=snapshot_sha,spec=spec,binding=config['binding'],result=result))
                    duration=time.monotonic()-before;candidate_seconds.append(duration)
                    probe_records.append(dict(prefix=k,spec=spec,seconds=duration))
                    if content_sha(snapshot)!=snapshot_sha:raise Halt('candidate polluted prefix')
                invalid=core.candidate_spec(ctx,snapshot,99,10,.5,'fine_power');counts=ledger.counts()
                if core.evaluate_candidate(ctx,snapshot,invalid,meter,'engineering/illegal/'+str(k))['physical_calls']!=0 or ledger.counts()!=counts:raise Halt('illegal PHY')
            if k==positions[-1]:break
            before=time.monotonic()
            result=core.replay_original_action(ctx,snapshot,actions[k],meter,'engineering/new/'+str(k))
            if k<=positions[0]:
                def reference():
                    outcome=old.step(actions[k])
                    state={name:copy.deepcopy(getattr(old,name)) for name in
                        ('beam_idx','starts','counts','powers','occupied','rates','requested','total_satisfaction')}
                    return dict(rates=old.rates.copy(),powers=old.powers.copy(),reward=outcome[1],state=state)
                ref=meter('phy','engineering/old/'+str(k),reference)
                for name,value in ref['state'].items():setattr(old,name,copy.deepcopy(value))
                np.testing.assert_allclose(result['capacity_bps'],ref['rates'],rtol=1e-12,atol=1e-6)
                np.testing.assert_array_equal(result['next_snapshot']['powers'],ref['powers'])
                np.testing.assert_allclose(result['reward'],ref['reward'],rtol=1e-12,atol=1e-12)
            snapshot=result['next_snapshot'];prefix_seconds.append(time.monotonic()-before)
        empty=core.initial_snapshot(ctx);rates=[]
        for watts in (.5,1.):
            spec=core.candidate_spec(ctx,empty,0,1,watts,'fine_power')
            result=core.evaluate_candidate(ctx,empty,spec,meter,'engineering/half/'+str(watts))
            if result['actual_action']['power_w']!=watts:raise Halt('half-watt clipping')
            index=int(ctx.indices[0]);p=ctx.physics
            signal=10**(-p['signal_loss_dB']/10)*ctx.kernel['gain'][index,index]*10**(p['rx_gain_dBi']/10)*ctx.kernel['path'][index,0]
            noise=p['k_J_per_K']*p['noise_temperature_K']*p['slot_Hz']
            hand=p['slot_Hz']/np.log(2)*np.log1p(signal*watts/noise)
            np.testing.assert_allclose(result['current']['capacity_bps'],hand,rtol=1e-12,atol=1e-6)
            rates.append(result['current']['capacity_bps'])
        if not 0<rates[0]<rates[1]:raise Halt('isolated power monotonicity')
        counts=ledger.counts();real_phy=sum(v for k,v in counts.items() if k.startswith('phy_'))
        if real_phy>256:raise Halt('engineering cap')
        ledger.artifact('evidence',root/'evidence.pkl',dict(status='checks_passed_requires_root_review',counts=counts,
            elapsed_seconds=time.monotonic()-started,input_load_seconds=input_seconds,kernel_seconds=kernel_seconds,
            prefix_step_seconds=prefix_seconds,candidate_row_seconds=candidate_seconds,probe_records=probe_records,
            selected_mother=row['mother'],workload_selection=workload,selection_is_runtime_bound=False,
            half_watt_capacities=rates,durable_probe_rows=len(candidate_seconds),maximum_durable_candidate_seconds=max(candidate_seconds),
            candidate_only_extrapolation_seconds=max(candidate_seconds)*15*1055,
            throughput_is_science_admission=False,real_PHY=real_phy,
            theoretical_N200_max_PHY=175+26+48+2))
    finally:ledger.finish()
