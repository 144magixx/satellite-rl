"""按全部目标波束计算终态满意度、供需匹配程度和资源指标。"""
import numpy as np


def terminal_beam_arrays(env):
    required = np.asarray(env.beam_info['rate'], dtype=np.float64).reshape(-1)
    if not np.isfinite(required).all() or np.any(required <= 0):
        raise ValueError('波束需求必须为有限正数')
    rates = np.zeros_like(required)
    served = np.asarray(env.beam_bps, dtype=np.float64).reshape(-1)
    if len(served) > len(rates) or not np.isfinite(served).all() or np.any(served < -1e-6):
        raise ValueError('终态速率无效')
    rates[:len(served)] = np.maximum(served, 0.0)
    satisfaction = np.minimum(rates / required, 1.0)
    beta = float(np.asarray(env.beta).reshape(-1)[0])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('当前场景需求尺度必须为有限正数')
    sgm = np.zeros_like(rates)
    positive = rates > 0
    ratio = rates[positive] / required[positive]
    real = np.where(ratio >= 1, ratio - 1, 1 - 1 / ratio)
    imag = (rates[positive] - required[positive]) / beta
    sgm[positive] = 1 - (1 - np.exp(-np.abs(real + 1j * imag))) ** 3
    return dict(rates_bps=rates, required_bps=required,
                satisfaction=satisfaction, sgm=sgm,
                basic_threshold_bps=0.1 * required,
                basic_met=rates >= 0.1 * required)


def terminal_metrics(env):
    arrays = terminal_beam_arrays(env)
    total = len(arrays['required_bps'])
    history = env.history_beam
    served = len(env.beam_bps)
    allocated_slots = int(np.count_nonzero(env.satellite_freq_state.freq_pool_real))
    throughput = float(arrays['rates_bps'].sum())
    return dict(
        si_sum=float(arrays['satisfaction'].sum()),
        si_mean=float(arrays['satisfaction'].mean()),
        sgm_mean=float(arrays['sgm'].mean()),
        basic_service_count=int(arrays['basic_met'].sum()),
        zero_capacity_count=int(np.count_nonzero(arrays['rates_bps'] == 0)),
        below_basic_indices=np.flatnonzero(~arrays['basic_met']).tolist(),
        lowest_10_rates_mbps=(np.sort(arrays['rates_bps'])[:10] / 1e6).tolist(),
        throughput_bps=throughput,
        total_power_w=float(env.total_power),
        served_beams=served, total_beams=total,
        power_exceeded=bool(env.overload), completed=served == total,
        occupancy_fraction=float(np.count_nonzero(env.satellite_freq_state.freq_pool_real)
                                 / env.satellite_freq_state.freq_pool_real.size),
        zero_slot_rate=float(sum(row[3] == 0 for row in history) / len(history)) if history else 0.0,
        allocated_slots=allocated_slots,
        spectral_efficiency=throughput / (allocated_slots * 2.5e7) if allocated_slots else 0.0,
        interference_beams=int(getattr(env, 'interfere_num', 0)),
        in_range_interference_beams=int(getattr(env, 'interfere_in_range_num', 0)),
    )
