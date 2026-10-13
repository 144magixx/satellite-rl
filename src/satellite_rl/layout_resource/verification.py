"""Bounded synthetic engineering entry point, never a scientific trainer.

No dataset or step-budget arguments are accepted. Four synthetic episodes and
forward passes exercise the production-size network. No optimizer is run.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from satellite_rl.paths import PROJECT_ROOT, project_path
from satellite_rl.shared_interface.content import content_sha, seal, wire_bytes
from satellite_rl.shared_interface.contracts import Authority, validate_allocation
from satellite_rl.shared_interface.geometry import cone_omega
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS, PHYSICS_SHA
from satellite_rl.shared_interface.stages import make_case, make_layout, UniformStage2Input, UniformStage3FixedInput
from .agent import AgentConfig, LayoutSAC
from .environment import LayoutResourceEnv, encode_action
from .power import LayoutPowerProblem
from .schema import observation_semantics


def synthetic_input(n):
    """Hand-specified one-aircraft beams; not an S1 planner or scientific data."""
    recipe = dict(purpose="S2-L25 synthetic engineering", N=n,
                  positions="latitude=52+0.15*(i%3), longitude=8",
                  full_width_deg="(0.3,0.5,1.0)[i%3]", aircraft_per_beam=1)
    ident = f"l25-engineering-N{n}"
    aircraft = [f"{ident}:a{i:03}" for i in range(n)]
    beams = [f"{ident}:b{i:03}" for i in range(n)]
    positions = np.array([[52 + .15 * (i % 3), 8] for i in range(n)], np.float64)
    widths = np.array([(.3, .5, 1.)[i % 3] for i in range(n)])
    case = make_case(dict(case_id=ident, actual_role="synthetic_engineering",
        source_role="synthetic_engineering", original_mother_id=ident,
        UTC="2026-10-13T00:00:00Z", source_asset_SHA=content_sha(recipe), source=recipe),
        aircraft, positions)
    layout = make_layout(case, beam_ids=beams, source_slot_ids=beams,
        anchor_aircraft_ids=aircraft, centres_lat_lon_deg=positions, full_width_deg=widths,
        owner_aircraft_to_beam_id=list(zip(aircraft, beams)), Ncap=n,
        geometry_Omega_cap_sr=cone_omega(np.ones(n)), geometry_Omega_sr=cone_omega(widths),
        geometry_audit=dict(purpose="synthetic hand recipe, verified on import"),
        source_and_cost=dict(purpose="synthetic", planner_calls=0, teacher_calls=0))
    registry = dict(schema="R7PublishedParentRegistryV1",
        case_sha_by_id={ident: case['content_sha256']},
        layout_shas=[layout['content_sha256']], allocation_shas=[])
    authority = Authority(registry, content_sha(registry))
    return UniformStage2Input(case, layout, authority, PHYSICS), registry


def _json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_verification(output, device):
    if device not in ('cpu', 'cuda'):
        raise ValueError('explicit cpu/cuda engineering device required')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = 73025
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = AgentConfig()
    source_files = list((PROJECT_ROOT/'src/satellite_rl/layout_resource').glob('*.py'))
    source_files += list((PROJECT_ROOT/'src/satellite_rl/shared_interface').glob('*.py'))
    source_files += [PROJECT_ROOT/'src/satellite_rl'/relative for relative in (
        'agents/action_masking.py', 'power_refinement/physics.py', 'power_refinement/priority.py')]
    baseline = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    manifest = dict(purpose='synthetic correctness only; not scientific training',
        started_utc=datetime.now(timezone.utc).isoformat(), seed=seed, device=device,
        executable=sys.executable, package_root=str(PROJECT_ROOT), torch=str(torch.__version__),
        numpy=np.__version__, cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
        source_base_commit=baseline,
        source_sha256={str(p.relative_to(PROJECT_ROOT)).replace('\\', '/'): _sha(p) for p in source_files},
        model_config=asdict(config), semantics=observation_semantics(),
        planned=dict(synthetic_N=[1, 3, 9, 200], environment_steps=213,
                     optimizer_updates=0, new_scientific_steps=0, power_solver_calls=0),
        scope='local DQN engineering, not remote formal workbench training')
    _json(output/'pre_execution.json', manifest)
    started = perf_counter()
    agent = LayoutSAC(config, device)
    transitions, scenes, steps = [], [], []
    for n in (1, 3, 9, 200):
        stage, registry = synthetic_input(n)
        env = LayoutResourceEnv(stage)
        obs = env.reset()
        total_reward = 0.
        folder = output/f'N{n}'
        folder.mkdir()
        initial_action = agent.act(obs, deterministic=True)
        if not obs['action_mask'][initial_action]:
            raise AssertionError('network selected a masked action')
        for i in range(n):
            before = obs
            # Recorded common legal sequence; learning is not used to claim quality.
            action = encode_action(i//8, 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if i == 0 and n in (1, 3, 200):
                transitions.append((before, action, reward, None if done else obs, done))
                np.savez_compressed(folder/'update_observations.npz',
                    **{f'before_{k}': before[k] for k in ('tokens', 'relations', 'global_features', 'action_mask')},
                    **{f'after_{k}': obs[k] for k in ('tokens', 'relations', 'global_features', 'action_mask')})
            steps.append(dict(N=n, step=i+1, beam_id=env.beam_ids[i], action=action,
                actual=info['actual_action'], reward=reward, SI=info['metrics']['SI'],
                SGM=info['metrics']['SGM'], done=done))
        allocation = env.export_allocation()
        validate_allocation(stage.case, stage.layout, allocation, stage.authority,
                            PHYSICS_SHA, published=False)
        # Explicit separate synthetic audit registry, never changes S1 authority.
        registry['allocation_shas'] = [allocation['content_sha256']]
        authority = Authority(registry, content_sha(registry))
        s3 = UniformStage3FixedInput.from_layout_and_allocation(
            stage.case, stage.layout, allocation, authority, PHYSICS)
        fixed = s3.fixed_initial_evaluation()
        order = [stage.layout['beam_ids'].index(bid) for bid in env.beam_ids]
        np.testing.assert_allclose(env.capacities, fixed['capacity_bps'][order], rtol=1e-10, atol=1e-6)
        derivative = LayoutPowerProblem(s3).evaluate(fixed['initial_power_w'], with_jacobian=True)
        if not np.isfinite(derivative.capacity_jacobian).all():
            raise AssertionError('nonfinite shared derivative')
        if abs(total_reward - env.metrics()['SI']) > 1e-12:
            raise AssertionError('all-N reward failed telescoping identity')
        for name, record in dict(case=stage.case, layout=stage.layout,
                                 allocation=allocation, authority=seal(registry)).items():
            (folder/f'{name}.json').write_bytes(wire_bytes(record))
        np.savez_compressed(folder/'fixed_results.npz', capacity_bps=env.capacities,
            request_bps=env.demands, start_slot=env.starts, slot_count=env.counts, power_w=env.powers,
            s3_capacity_bps=fixed['capacity_bps'][order])
        scenes.append(dict(N=n, case_SHA=stage.case['content_sha256'], layout_SHA=stage.layout['content_sha256'],
            allocation_SHA=allocation['content_sha256'], exact_request_total_bps=int(env.demands.sum()),
            initial_untrained_network_action=initial_action, final_SI=env.metrics()['SI'],
            final_SGM=env.metrics()['SGM'], reward_sum=total_reward, S2_S3_max_abs_bps=float(
                np.max(np.abs(env.capacities-fixed['capacity_bps'][order])))))
    updates = []
    checkpoint = output/'engineering_only.pt'
    agent.save(checkpoint, provenance=dict(purpose='engineering_only_do_not_train_from_this', seed=seed))
    restored = LayoutSAC(config, device)
    restored.load(checkpoint, resume=True)
    if restored.act(transitions[0][0], deterministic=True) != agent.act(transitions[0][0], deterministic=True):
        raise AssertionError('saved model changed deterministic action')
    if device == 'cuda':
        torch.cuda.synchronize()
    (output/'steps.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'
                                               for row in steps), encoding='utf-8')
    _json(output/'updates.json', updates)
    summary = dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        config=asdict(config), scenes=scenes, engineering_environment_steps=len(steps),
        engineering_optimizer_updates=agent.updates,
        parameters_per_network=sum(p.numel() for p in agent.actor.parameters()),
        elapsed_seconds=perf_counter()-started, scientific_training_steps=0,
        scientific_evaluation_scenes=0, power_solver_calls=0,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        interpretation='correctness and finite update evidence only; no policy quality claim',
        full_task_resume='not implemented; model/optimizer continuation only',
        artifacts={str(p.relative_to(output)): _sha(p) for p in output.rglob('*') if p.is_file()})
    _json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    parser.add_argument('--output', default=project_path('outputs/verification/layout_resource_v25'))
    args = parser.parse_args()
    summary = run_verification(args.output, args.device)
    print(json.dumps({k: summary[k] for k in ('status', 'engineering_environment_steps',
        'engineering_optimizer_updates', 'scientific_training_steps', 'elapsed_seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
