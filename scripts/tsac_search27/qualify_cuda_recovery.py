"""Bounded W27 actual-CUDA recovery qualification: <=9 synthetic updates, 0 env.

Run before engineering on a root-qualified unused GPU. Never falls back to CPU.
The durable intent receipt preserves unknown spend if an update crashes. Failed
receipts are not overwritten or automatically retried. A separate explicitly
accounted receipt path is necessary for any subsequent qualification attempt.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from satellite_rl.tsac_search27.agent import SearchAgent, build_config
from satellite_rl.tsac_search27.pause import atomic_json
from satellite_rl.t0_width.runner import seed_all


def write_receipt(path, value):
    atomic_json(path, value)


def tensor_comparison(left, right):
    maximum = 0.
    bitwise = True
    for network in ('actor', 'critic1', 'critic2', 'target1', 'target2'):
        for name, value in getattr(left, network).state_dict().items():
            other = getattr(right, network).state_dict()[name]
            # Same source, device and optimizer replay should match exactly;
            # record a bounded floating tolerance if the GPU library differs.
            torch.testing.assert_close(value, other, rtol=3e-6, atol=2e-6)
            maximum = max(maximum, float((value - other).abs().max().item()))
            bitwise = bitwise and torch.equal(value, other)
    return dict(maximum_absolute_parameter_error=maximum, parameters_bitwise_equal=bitwise,
                parameter_tolerance=dict(rtol=3e-6, atol=2e-6))


def qualify(output, device='cuda:0'):
    output = Path(output)
    if output.exists():
        raise ValueError('qualification receipt already exists; do not overwrite or free replay')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    import satellite_rl.tsac_search27.agent as agent_module
    agent_source = Path(agent_module.__file__).resolve()
    receipt = dict(schema='W27CUDARecoveryQualificationV1', experiment_id='S2-W27',
                   started_UTC=datetime.now(timezone.utc).isoformat(), device=device,
                   update_budget=9, completed_synthetic_updates=0, pending_update=None,
                   unknown_update_spend_not_zero=False, synthetic_replay_transitions_added=0,
                   environment_interactions=0, PHY_calls=0, scientific_training_steps=0,
                   formal_engineering_interactions=0, variants={},
                   torch=str(torch.__version__), cuda_build=torch.version.cuda,
                   source_sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
                                  str(agent_source): hashlib.sha256(agent_source.read_bytes()).hexdigest()})
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        receipt.update(status='not_run_actual_CUDA_unavailable_no_CPU_fallback', completed_UTC=datetime.now(timezone.utc).isoformat())
        write_receipt(output, receipt)
        return receipt
    receipt['gpu_name'] = torch.cuda.get_device_name(device)
    receipt['gpu_device_properties'] = str(torch.cuda.get_device_properties(device))
    torch.set_num_threads(1)
    def update(agent, variant, stage):
        if receipt['completed_synthetic_updates'] >= receipt['update_budget']:
            raise RuntimeError('synthetic update budget exhausted')
        receipt['pending_update'] = dict(variant=variant, stage=stage)
        receipt['unknown_update_spend_not_zero'] = True
        write_receipt(output, receipt)
        result = agent.update()
        torch.cuda.synchronize(device)
        receipt['completed_synthetic_updates'] += 1
        receipt['pending_update'] = None
        receipt['unknown_update_spend_not_zero'] = False
        write_receipt(output, receipt)
        return result
    write_receipt(output, receipt)
    try:
        for variant in ('B', 'E', 'LP'):
            config = build_config(variant, device=device, phase='correctness')
            config['training'].update(batch_size=2, warmup_steps=2, replay_capacity=16)
            seed_all(config['training']['seed'])
            agent = SearchAgent(config, device)
            mask = dict(frequency_slots=np.ones((100, 10), bool), power=np.ones(10, bool))
            mask['frequency_slots'][7:20, 5:] = False
            mask['power'][3] = False
            for i in range(2):
                state = np.linspace(-.2, .7, 208, dtype=np.float32) + i * .01
                agent.observe(state, (i, 1, 2), float(i - .5), state + .01, i == 1, mask, None if i == 1 else mask)
                receipt['synthetic_replay_transitions_added'] += 1
            update(agent, variant, 'create_optimizer_state')
            stream = io.BytesIO()
            torch.save(agent.state_dict(), stream)
            stream.seek(0)
            # Explicit CUDA mapping reproduces the byte-RNG-tensor recovery hazard.
            saved = torch.load(stream, map_location=device, weights_only=False)
            expected_action = agent.act(state, action_mask=mask)
            expected_diagnostics = update(agent, variant, 'trusted_next_update')
            restored = SearchAgent(config, device)
            restored.load_state_dict(saved)
            actual_action = restored.act(state, action_mask=mask)
            if actual_action != expected_action:
                raise AssertionError('restored next action differs')
            actual_diagnostics = update(restored, variant, 'restored_next_update')
            max_diagnostic_error = 0.
            for key, value in expected_diagnostics.items():
                other = actual_diagnostics[key]
                if isinstance(value, (int, float)):
                    if not np.isfinite(value) or not np.isfinite(other) or not np.isclose(value, other, rtol=3e-6, atol=2e-6):
                        raise AssertionError('restored diagnostic differs: ' + key)
                    max_diagnostic_error = max(max_diagnostic_error, abs(value - other))
                elif value != other:
                    raise AssertionError('restored diagnostic label differs')
            receipt['variants'][variant] = dict(next_action=list(actual_action), next_action_equal=True,
                                                max_diagnostic_absolute_error=max_diagnostic_error,
                                                initialization_rng=agent.initialization_rng,
                                                schema_sha=agent.schema['sha256'],
                                                fixture_seed=config['training']['seed'],
                                                **tensor_comparison(agent, restored))
            write_receipt(output, receipt)
            del agent, restored, saved, stream
        receipt['status'] = 'passed_actual_CUDA_next_action_update_recovery'
    except Exception as error:
        receipt['status'] = 'failed_preserved_no_automatic_retry'
        receipt['error'] = type(error).__name__ + ': ' + str(error)[:1000]
        write_receipt(output, receipt)
        raise
    finally:
        receipt['completed_UTC'] = datetime.now(timezone.utc).isoformat()
        write_receipt(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    result = qualify(args.output, args.device)
    print(json.dumps(dict(status=result['status'], completed_synthetic_updates=result['completed_synthetic_updates'],
                         unknown_update_spend_not_zero=result['unknown_update_spend_not_zero'])))
