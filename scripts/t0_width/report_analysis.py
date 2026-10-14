"""Read-only W26 analysis. No environment, policy, or link calculations are invoked.

Input manifest: {confirmation: path, confirmation_steps: path, development: {group: path},
training: {group: episodes_path}, steps: {group: steps_path}, provenance: {...}}.
Paths are relative to the manifest. Confirmation requires the complete 680-row matrix.
"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = ('S00', 'S10', 'S01', 'S11')
METHODS = GROUPS + ('random', 'fixed', 'greedy')
CONTRASTS = (('S10', 'S00'), ('S11', 'S01'), ('S01', 'S00'), ('S11', 'S10'))
METRICS = ('SI_mean', 'SGM', 'throughput_bps', 'total_power_w', 'zero_slot_rate', 'complete')
NODES = (0, 2000, 5000) + tuple(range(10000, 150001, 10000))


def stream_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def decode_array(value):
    if isinstance(value, dict) and set(value) == {'dtype', 'shape', 'data_hex'}:
        return np.frombuffer(bytes.fromhex(value['data_hex']), dtype=value['dtype']).reshape(value['shape']).copy()
    return np.asarray(value)


def audit_terminal(row):
    """Independent arithmetic from retained terminal capacities; never recompute PHY."""
    beams = row['beams']
    ids = [b['beam_id'] for b in beams]
    n = len(ids)
    if not 1 <= n <= 200 or len(set(ids)) != n:
        raise ValueError('invalid or duplicate terminal beam IDs')
    demand = np.array([b['demand_bps'] for b in beams], dtype=float)
    rate = np.array([b['capacity_bps'] for b in beams], dtype=float)
    if not np.isfinite(demand).all() or not np.isfinite(rate).all() or (demand <= 0).any() or (rate < 0).any():
        raise ValueError('nonfinite/invalid demand or capacity')
    if np.any(demand % 21600000):
        raise ValueError('beam demand is not an integer aircraft request sum')
    si = np.minimum(rate / demand, 1.)
    sgm = np.zeros(n)
    positive = rate > 0
    ratio = rate[positive] / demand[positive]
    deviation = np.where(ratio >= 1, ratio - 1, 1 - 1 / ratio)
    distance = np.hypot(deviation, (rate[positive] - demand[positive]) / demand.mean())
    sgm[positive] = 1 - (1 - np.exp(-distance)) ** 3
    wire = row['allocation_wire']
    if 'content_sha256' in wire:
        # Pure typed-content verification; this import invokes no environment or PHY.
        from satellite_rl.shared_interface.content import read_wire
        checked = read_wire(json.dumps(wire))
        if row.get('allocation_SHA') != checked['content_sha256']:
            raise ValueError('logged allocation SHA differs')
    allocation_ids = wire['beam_ids']
    if len(allocation_ids) != n or set(allocation_ids) != set(ids):
        raise ValueError('allocation beam set differs')
    starts = decode_array(wire['actual_start_slot'])
    counts = decode_array(wire['actual_slot_count'])
    powers = decode_array(wire['actual_whole_beam_power_w'])
    if any(a.shape != (n,) or not np.isfinite(a).all() for a in (starts, counts, powers)):
        raise ValueError('allocation arrays invalid')
    groups = wire['group_by_beam_id']
    violations = int(np.sum((starts < 0) | (starts != np.floor(starts)) | (counts < 0) |
                    (counts > 10) | (counts != np.floor(counts)) | (starts + counts > 100) |
                    (powers < 0) | ((counts == 0) & ((starts != 0) | (powers != 0))) |
                    ((counts > 0) & ((powers < 5) | (powers > 50) | (powers % 5 != 0)))))
    violations += int(powers.sum() > 3000 + 1e-9)
    for i, bid in enumerate(allocation_ids):
        if bid not in groups or type(groups[bid]) is not int or not 0 <= groups[bid] < 8:
            raise ValueError('invalid group')
        for j in range(i):
            violations += int(groups[bid] == groups[allocation_ids[j]] and counts[i] > 0 and counts[j] > 0
                              and max(starts[i], starts[j]) < min(starts[i] + counts[i], starts[j] + counts[j]))
    identity = row['identity']
    if wire['parent_layout_SHA'] != identity['layout_SHA'] or wire['parent_case_SHA'] != identity['aircraft_SHA']:
        raise ValueError('allocation parent identity differs')
    if 'physics_config_SHA' in wire and wire['physics_config_SHA'] != identity['physics_SHA']:
        raise ValueError('allocation physics differs')
    result = dict(SI_mean=float(si.mean()), SI_sum=float(si.sum()), SGM=float(sgm.mean()),
                  throughput_bps=float(rate.sum()), total_power_w=float(powers.sum()), N=n,
                  demand_bps=float(demand.sum()), zero_slot_rate=float(np.mean(counts == 0)),
                  complete=int(row['metrics']['processed'] == n), constraint_violations=violations,
                  unserved_beams=int(np.sum(counts == 0)))
    for key in ('SI_mean', 'SI_sum', 'SGM', 'throughput_bps', 'total_power_w', 'N', 'zero_slot_rate'):
        if not np.isclose(result[key], row['metrics'][key], rtol=1e-10, atol=1e-8):
            raise ValueError('logged metric mismatch: ' + key)
    for i, b in enumerate(beams):
        if not np.isclose(si[i], b['SI'], atol=1e-10) or not np.isclose(sgm[i], b['SGM'], atol=1e-10):
            raise ValueError('logged beam metric mismatch')
    return result, tuple(sorted(zip(ids, demand))), dict(zip(ids, si))


def bootstrap(values, seed=73026):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty mother sample required')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(samples, .025)),
                high=float(np.quantile(samples, .975)), mothers=len(values),
                improved=int(np.sum(values > 0)), worsened=int(np.sum(values < 0)), tied=int(np.sum(values == 0)))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_confirmation(rows):
    buckets = defaultdict(list)
    seen = set()
    signatures = {}
    layouts = {}
    raw = []
    for row in rows:
        method = row['phase'].removeprefix('confirmation_')
        key = (row['mother'], method, row['repeat'])
        if method not in METHODS or key in seen or row['node'] != 150000:
            raise ValueError('unexpected/duplicate confirmation identity')
        if row['action_seed'] != 275000 + row['mother'] + row['repeat'] * 1000000:
            raise ValueError('confirmation action seed differs')
        seen.add(key)
        audit, signature, _ = audit_terminal(row)
        mother = row['mother']
        layout = (row['identity']['case_id'], row['identity']['layout_SHA'], row['identity']['aircraft_SHA'], row['identity']['physics_SHA'])
        if mother in signatures and (signatures[mother] != signature or layouts[mother] != layout):
            raise ValueError('paired mother beam demands/layout differ')
        signatures[mother], layouts[mother] = signature, layout
        entry = dict(method=method, mother=mother, repeat=row['repeat'], case_id=layout[0], **audit,
                     **{'time_' + k: v for k, v in row['timing'].items()})
        raw.append(entry)
        buckets[mother, method].append(entry)
    expected = {(m, g, r) for m in range(40) for g in METHODS for r in range(3 if g in GROUPS + ('random',) else 1)}
    if seen != expected:
        raise ValueError(f'incomplete confirmation matrix: missing={len(expected-seen)}, extra={len(seen-expected)}')
    mother_rows = []
    for (mother, method), entries in sorted(buckets.items()):
        numeric = [k for k in entries[0] if k not in ('method', 'mother', 'repeat', 'case_id')]
        mother_rows.append(dict(mother=mother, method=method, case_id=entries[0]['case_id'],
                                repeats=len(entries), **{k: float(np.mean([e[k] for e in entries])) for k in numeric}))
    paired = {(r['mother'], r['method']): r for r in mother_rows}
    contrasts = []
    differences = []
    for a, b in CONTRASTS + tuple((g, baseline) for g in GROUPS for baseline in ('random', 'fixed', 'greedy')):
        for metric in ('SI_mean', 'SGM'):
            delta = [paired[m, a][metric] - paired[m, b][metric] for m in range(40)]
            contrasts.append(dict(a=a, b=b, metric=metric, preregistered=(a, b) in CONTRASTS, **bootstrap(delta)))
            differences.extend(dict(mother=m, a=a, b=b, metric=metric, delta=d) for m, d in enumerate(delta))
    summary = {g: {k: bootstrap([paired[m, g][k] for m in range(40)]) for k in METRICS + tuple(k for k in paired[0, g] if k.startswith('time_'))} for g in METHODS}
    return raw, mother_rows, contrasts, differences, summary


def action_mismatch(row):
    actual, requested = row['actual_action'], row['requested_action']
    return int(actual['start'] != requested[0] or actual['count'] != requested[1] + 1
               or actual['power_w'] != (requested[2] + 1) * 5)


def confirmation_behavior(path, terminals):
    expected = {(r['mother'], r['phase'].removeprefix('confirmation_'), r['repeat']): r for r in terminals}
    seen = defaultdict(set)
    counts = {g: dict(method=g, decisions=0, preference_observations=0, mapping_changed=0, request_execution_mismatch=0) for g in METHODS}
    for _, row in stream_rows(path):
        method = row['phase'].removeprefix('confirmation_')
        key = (row['mother'], method, row['repeat'])
        if key not in expected or row['beam_step'] in seen[key]:
            raise ValueError('unknown/duplicate confirmation step')
        terminal = expected[key]
        if row['action_seed'] != terminal['action_seed'] or row['identity'] != terminal['identity']:
            raise ValueError('confirmation step identity/seed differs')
        ids = terminal['allocation_wire']['processing_index_to_beam_id']
        if not 0 <= row['beam_step'] < len(ids) or row['beam_id'] != ids[row['beam_step']]:
            raise ValueError('confirmation step processing identity differs')
        wire = terminal['allocation_wire']
        i = wire['beam_ids'].index(row['beam_id'])
        if any(row['actual_action'][field] != decode_array(wire[array])[i] for field, array in
               (('start', 'actual_start_slot'), ('count', 'actual_slot_count'), ('power_w', 'actual_whole_beam_power_w'))):
            raise ValueError('confirmation step action differs from terminal allocation')
        seen[key].add(row['beam_step'])
        count = counts[method]
        count['decisions'] += 1
        count['request_execution_mismatch'] += action_mismatch(row)
        if method in ('fixed', 'greedy'):
            preference = row['source_preference']
            count['preference_observations'] += 1
            count['mapping_changed'] += int(preference['mapping_changed'])
    for key, terminal in expected.items():
        if seen[key] != set(range(terminal['metrics']['N'])):
            raise ValueError('confirmation step count differs from actual N')
    for count in counts.values():
        count['mapping_changed_rate'] = count['mapping_changed'] / count['preference_observations'] if count['preference_observations'] else None
        count['request_execution_mismatch_rate'] = count['request_execution_mismatch'] / count['decisions']
    return list(counts.values())


def figures(out, mothers, contrasts, development, training, step_bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + '.png'), dpi=180)
        fig.savefig(out / (name + '.pdf'))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ('SI_mean', 'SGM')):
        data = [[r[metric] for r in mothers if r['method'] == g] for g in METHODS]
        ax.boxplot(data, labels=METHODS, showmeans=True)
        ax.set_ylabel(metric + ' (mother mean)')
    save(fig, '01_confirmation_quality')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ('SI_mean', 'SGM')):
        rows = [r for r in contrasts if r['preregistered'] and r['metric'] == metric]
        for i, r in enumerate(rows):
            ax.plot([r['low'], r['high']], [i, i], linewidth=3)
            ax.scatter(r['mean'], i, color='black')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_yticks(range(4), [r['a'] + '-' + r['b'] for r in rows])
        ax.set_xlabel(metric + ' paired difference, descriptive 95% CI')
    save(fig, '02_preregistered_contrasts')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, scale in zip(axes, ('throughput_bps', 'total_power_w'), (1e9, 1)):
        ax.bar(METHODS, [np.mean([r[metric] for r in mothers if r['method'] == g]) / scale for g in METHODS])
        ax.set_ylabel('Throughput (Gbps)' if scale > 1 else 'Whole-beam power (W)')
    save(fig, '03_confirmation_resources')
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(7)
    for key in ('input_and_reset_s', 'mask_s', 'inference_s', 'environment_s', 'terminal_s'):
        if not all('time_' + key in r for r in mothers):
            continue
        vals = np.array([np.mean([r['time_' + key] for r in mothers if r['method'] == g]) for g in METHODS])
        ax.bar(METHODS, vals, bottom=bottom, label=key)
        bottom += vals
    ax.set_ylabel('Algorithm seconds incl. terminal (when recorded)')
    ax.legend(fontsize=8)
    save(fig, '04_accounted_time')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for g in GROUPS:
        for ax, metric in zip(axes[0], ('SI_mean', 'SGM')):
            rows = [r for r in development if r['method'] == g]
            ax.plot([r['node'] for r in rows], [r[metric] for r in rows], label=g)
            ax.set_xlabel('Training interactions (fixed endpoint 150000)')
            ax.set_ylabel('Development ' + metric)
            ax.legend()
    lookup = {(r['method'], r['node']): r for r in development}
    for a, b in CONTRASTS:
        nodes = [n for n in NODES if (a, n) in lookup and (b, n) in lookup]
        for ax, metric in zip(axes[1], ('SI_mean', 'SGM')):
            ax.plot(nodes, [lookup[a, n][metric] - lookup[b, n][metric] for n in nodes], label=a + '-' + b)
            ax.axhline(0, color='grey', linestyle='--', linewidth=.6)
            ax.set_ylabel('Development paired ' + metric + ' difference')
            ax.set_xlabel('Training interactions')
            ax.legend(fontsize=8)
    save(fig, '05_development_curve')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g in GROUPS:
        rows = [r for r in training if r['method'] == g]
        for ax, metric in zip(axes, ('SI_mean', 'SGM')):
            ax.plot([r['env_step'] for r in rows], [r[metric] for r in rows], linewidth=.6, alpha=.65, label=g)
            ax.set_xlabel('Training interactions')
            ax.set_ylabel('Training episode ' + metric)
            ax.legend()
    save(fig, '06_training_raw_curve')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g in GROUPS:
        rows = [r for r in training if r['method'] == g]
        axes[0].plot([r['env_step'] for r in rows], [r['episode_return'] for r in rows], linewidth=.6, label=g)
        bins = [r for r in step_bins if r['method'] == g]
        for key, style in (('actor_loss', '-'), ('critic_1_loss', '--'), ('critic_2_loss', ':')):
            selected = [r for r in bins if r.get(key + '_count', 0)]
            axes[1].plot([r['bin_start'] for r in selected], [r[key + '_sum'] / r[key + '_count'] for r in selected],
                         linestyle=style, linewidth=.8, label=g + ' ' + key)
    axes[0].set_ylabel('Raw episode return (diagnostic only)')
    axes[1].set_ylabel('Learning loss: mean in 1000-interaction bins')
    for ax in axes:
        ax.set_xlabel('Training interactions')
        ax.legend(fontsize=7)
    save(fig, '07_reward_learning_diagnostics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else args.manifest.parent / path
    confirmation = [r for _, r in stream_rows(resolve(manifest['confirmation']))]
    raw, mothers, contrasts, differences, summary = summarize_confirmation(confirmation)
    behavior = confirmation_behavior(resolve(manifest['confirmation_steps']), confirmation) if manifest.get('confirmation_steps') else []
    beam_rows = []
    for row in confirmation:
        wire = row['allocation_wire']
        actions = {bid: dict(start=int(decode_array(wire['actual_start_slot'])[i]), count=int(decode_array(wire['actual_slot_count'])[i]),
                            power_w=float(decode_array(wire['actual_whole_beam_power_w'])[i]), group=wire['group_by_beam_id'][bid])
                   for i, bid in enumerate(wire['beam_ids'])}
        beam_rows.extend(dict(method=row['phase'].removeprefix('confirmation_'), mother=row['mother'], repeat=row['repeat'],
                              **beam, **actions[beam['beam_id']]) for beam in row['beams'])
    development, training, step_bins = [], [], []
    feature_diagnostics = {}
    development_identities = {}
    for g, path in manifest.get('development', {}).items():
        buckets = defaultdict(list)
        seen = set()
        for _, row in stream_rows(resolve(path)):
            key = (row['node'], row['mother'], row['repeat'])
            if key in seen or row['phase'] != 'development':
                raise ValueError('duplicate/nondevelopment row')
            seen.add(key)
            if row['action_seed'] != 274000 + row['mother'] + row['repeat'] * 1000000:
                raise ValueError('development seed differs')
            audit, _, _ = audit_terminal(row)
            pair = (row['mother'], row['repeat'])
            identity = row['identity']
            if pair in development_identities and development_identities[pair] != identity:
                raise ValueError('development paired layout identity differs')
            development_identities[pair] = identity
            buckets[row['node']].append(audit)
        if seen != {(n, m, r) for n in NODES for m in range(20) for r in range(3)}:
            raise ValueError('incomplete 18x20x3 development matrix')
        for node, entries in sorted(buckets.items()):
            development.append(dict(method=g, node=node, episodes=len(entries), **{k: float(np.mean([r[k] for r in entries])) for k in METRICS}))
    for g, path in manifest.get('training', {}).items():
        for _, row in stream_rows(resolve(path)):
            audit, _, _ = audit_terminal(row)
            training.append(dict(method=g, env_step=row['env_step'], episode=row['episode'],
                                 budget_truncated=row.get('budget_truncated', False), episode_return=row['episode_return'], **audit))
    for g, path in manifest.get('steps', {}).items():
        bins = defaultdict(lambda: defaultdict(float))
        count = 0
        seen_steps = np.zeros(150001, dtype=bool)
        feature_sums = {key: dict(total=0, zero=0, nonzero=0, max_abs=0., abs_sum=0., abs_ge_1e_minus6=0)
                        for key in ('out_observation_float32', 'future_in_observation_float32')} if g in ('S01', 'S11') else {}
        for _, row in stream_rows(resolve(path)):
            if row.get('phase') != 'scientific':
                continue
            step = row['env_step']
            if type(step) is not int or not 1 <= step <= 150000 or seen_steps[step] or step != count + 1:
                raise ValueError('duplicate/out-of-range training step')
            seen_steps[step] = True
            if feature_sums:
                audit = row.get('relation_audit')
                if not isinstance(audit, dict):
                    raise ValueError('relation-enabled training step missing input audit')
                for key, totals in feature_sums.items():
                    values = np.asarray(audit[key], dtype=np.float32)
                    if values.shape != (100,) or not np.isfinite(values).all():
                        raise ValueError('relation observation audit shape/nonfinite')
                    absolute = np.abs(values.astype(np.float64))
                    zeros = int(np.count_nonzero(values == 0))
                    totals['total'] += values.size
                    totals['zero'] += zeros
                    totals['nonzero'] += values.size - zeros
                    totals['max_abs'] = max(totals['max_abs'], float(absolute.max()))
                    totals['abs_sum'] += float(absolute.sum())
                    totals['abs_ge_1e_minus6'] += int(np.count_nonzero(absolute >= 1e-6))
            count += 1
            entry = bins[(row['env_step'] - 1) // 1000]
            entry['count'] += 1
            entry['reward_sum'] += row['reward']
            entry['actual_power_w_sum'] += row['actual_action']['power_w']
            entry['actual_slot_count_sum'] += row['actual_action']['count']
            entry['request_execution_mismatch'] += action_mismatch(row)
            for key in ('actor_loss', 'critic_1_loss', 'critic_2_loss', 'alpha'):
                update = row.get('update') or {}
                if key in update:
                    if not np.isfinite(update[key]):
                        raise ValueError('nonfinite learning diagnostic')
                    entry[key + '_sum'] += update[key]
                    entry[key + '_count'] += 1
        if count != 150000 or not seen_steps[1:].all():
            raise ValueError('training step count must equal 150000')
        if feature_sums:
            for totals in feature_sums.values():
                totals['mean_abs'] = totals.pop('abs_sum') / totals['total']
            feature_diagnostics[g] = feature_sums
        step_bins.extend(dict(method=g, bin_start=b * 1000 + 1, **values) for b, values in sorted(bins.items()))
    for name, rows in (('confirmation_repeats', raw), ('confirmation_behavior', behavior), ('confirmation_beams', beam_rows), ('confirmation_mothers', mothers), ('contrasts', contrasts),
                       ('paired_differences', differences), ('development_nodes', development), ('training_episodes', training), ('training_step_bins', step_bins)):
        write_csv(out / (name + '.csv'), rows)
    inputs = [manifest['confirmation']] + [p for key in ('development', 'training', 'steps') for p in manifest.get(key, {}).values()]
    if manifest.get('confirmation_steps'):
        inputs.append(manifest['confirmation_steps'])
    hashes = {}
    for p in inputs:
        h = hashlib.sha256()
        with resolve(p).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        hashes[p] = h.hexdigest()
    feature_report = dict(schema='W26_state_feature_diagnostics_v1', groups=feature_diagnostics,
                          threshold_abs=1e-6, threshold_use='descriptive only; not a qualification or performance threshold',
                          source='pre-action relation_audit in original scientific training steps; FP32 values',
                          original_step_input_sha256={g: hashes[p] for g, p in manifest.get('steps', {}).items()},
                          width_observation='not recorded in step audit; no values imputed',
                          interpretation='Zero or tiny physical relation values do not imply hidden channels were zero-filled. Providing an input channel does not establish that the model effectively used it.')
    (out / 'state_feature_diagnostics.json').write_text(json.dumps(feature_report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    result = dict(schema='W26_readonly_analysis_v1', episodes=len(raw), mother_count=40, methods=summary,
                  contrasts=contrasts, confirmation_behavior=behavior, confirmation_step_audit_complete=bool(behavior),
                  learning_diagnostics='1000-interaction aggregates; original steps retained and SHA referenced; not individual-update raw loss curves',
                  state_feature_diagnostics='state_feature_diagnostics.json',
                  bootstrap=dict(replicates=10000, seed=73026, interval='percentile95_descriptive', unit='mother_after_repeat_average'),
                  constraint_violations=sum(r['constraint_violations'] for r in raw), incomplete_episodes=sum(not r['complete'] for r in raw),
                  input_sha256=hashes, provenance=manifest.get('provenance', {}), limitations=[
                      'Single training seed; descriptive intervals do not control multiplicity.',
                      'Adjacent mother snapshots may be correlated; no unregistered arbitrary block inference.',
                      'Capacity arithmetic audit only; no PHY recomputation.',
                      'Aircraft coverage/owner conservation relies on referenced independent data audit.',
                      'input_and_reset includes relation precomputation; environment includes state construction; these are not separately timed.',
                      'Legacy total_s excludes terminal serialization and WAL/logging; algorithm_with_terminal_s includes terminal; wall_to_terminal_s includes WAL but excludes log commit and model loading. Missing historical timing fields are not imputed.',
                      'Training curves include budget-truncated episode with explicit flag.'])
    (out / 'analysis_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    if not args.no_figures:
        figures(out, mothers, contrasts, development, training, step_bins)


if __name__ == '__main__':
    main()
