"""Read-only endpoint, validation and training terminal/best audit; no inference."""
from pathlib import Path
import argparse
import json
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from audit import sha, read, jsonl, config_digest, audit_row, metrics, edge_components, save


def check(value, message):
    if not value:
        raise ValueError(message)


def audit_best(scene, start, row, cfg):
    best = row['best_record_diagnostic']
    actual, _, covered, owner = metrics(scene, best['centers'], best['assignment'],
                                      set(start['state']['edge_beam_ids']), cfg)
    check(math.isclose(actual['quality'], best['quality'], abs_tol=1e-8), 'best physical quality')
    check(math.isclose(best['quality'], row['best_quality_diagnostic'], abs_tol=1e-8), 'best diagnostic binding')
    check(best['quality'] + 1e-8 >= row['metrics']['quality'], 'best below terminal')
    for region in start['state']['edge_regions']:
        beam = region.get('dedicated_beam_id')
        if beam is not None:
            members = region['aircraft_indices']
            check(all(owner[i] == beam for i in members) and int((owner == beam).sum()) == len(members),
                  'best fixed edge ownership')
    return best['quality'] - row['metrics']['quality']


def formal_best(jobdir, output):
    from satellite_rl.coverage.scenes import load_dataset
    directory = Path(jobdir)
    cfg = read(directory/'config.json')
    result = read(directory/'results.json')
    rows_path = Path(result['rows_path'])
    check(sha(rows_path) == result['rows_sha256'], 'formal raw SHA')
    rows = jsonl(rows_path)
    role = next(iter(cfg['datasets']))
    data, initial = cfg['datasets'][role], cfg['reconstruction_starts'][role]
    check(sha(data['path']) == data['sha256'] and sha(initial['path']) == initial['sha256'], 'formal inputs SHA')
    scenes = {s.scene_id:s for s in load_dataset(data['path'],'cluster',cfg['coverage']).scenes}
    starts = {r['scene_id']:r for r in jsonl(initial['path'])}
    check(len(rows) == 100 and {r['scene_id'] for r in rows} == set(scenes), 'formal 100 unique scenes')
    pairs = []
    for row in rows:
        gap = audit_best(scenes[row['scene_id']],starts[row['scene_id']],row,cfg['coverage']['reconstruction'])
        pairs.append({'scene_id':row['scene_id'],'terminal_quality':row['metrics']['quality'],
                      'best_quality':row['best_quality_diagnostic'],'best_minus_terminal':gap})
    result = {'passed':True,'scope':'Jan31 P1 diagnostic best only; original terminals covered by formal audit',
              'run_id':read(directory/'job.json')['run_id'],'count':len(rows),'pairs':pairs,
              'raw_sha256':sha(rows_path),'positive_gap_count':sum(p['best_minus_terminal'] > 1e-8 for p in pairs),
              'mean_gap':sum(p['best_minus_terminal'] for p in pairs)/len(pairs),
              'max_gap':max(p['best_minus_terminal'] for p in pairs)}
    save(output,result)
    return result


def contract_counts(jobdirs, output):
    import numpy as np
    records = []
    for jobdir in jobdirs:
        directory = Path(jobdir)
        cfg = read(directory/'config.json')
        result = read(directory/'results.json')
        path = Path(result['rows_path'])
        check(sha(path) == result['rows_sha256'], 'contract raw SHA')
        role = next(iter(cfg['datasets']))
        initial = cfg['reconstruction_starts'][role]
        check(sha(initial['path']) == initial['sha256'], 'contract initial SHA')
        starts = {r['scene_id']:r for r in jsonl(initial['path'])}
        for row in jsonl(path):
            ids = set(starts[row['scene_id']]['state']['edge_beam_ids'])
            users = row['users']
            assigned = [u for u in users if u['assigned_beam'] is not None]
            loads = np.bincount([u['assigned_beam'] for u in assigned],
                                weights=[u['demand_bps'] for u in assigned],minlength=100)
            check(np.allclose(loads,row['beam_load_demand_bps'],rtol=1e-12,atol=1e-5), 'contract ownership demand')
            subject = np.array([i not in ids for i in range(100)])
            mean = float(loads[subject].mean())
            count = int(np.count_nonzero(np.abs(loads/mean - 1) <= cfg['coverage']['reconstruction']['range_tolerance'] + 1e-12))
            frozen = row['metrics']['subject_in_band_beams']
            check(int(np.count_nonzero((np.abs(loads/mean-1) <= cfg['coverage']['reconstruction']['range_tolerance']+1e-12)&subject)) == frozen,
                  'frozen subject count reproduction')
            target = (row['metrics']['coverage_100'] and not row['metrics']['empty_beams']
                      and len(starts[row['scene_id']]['state']['edge_regions']) <= cfg['coverage']['reconstruction']['max_edge_beams']
                      and row['metrics']['subject_cv'] <= cfg['coverage']['reconstruction']['subject_cv_target']
                      and count >= cfg['coverage']['reconstruction']['min_subject_beams_in_range'])
            records.append({'route':row['route'],'scene_id':row['scene_id'],'frozen_subject_count':frozen,
                            'contract_all100_count':count,'count_difference':count-frozen,
                            'frozen_target_met':row['metrics']['target_met'],'contract_target_met':bool(target)})
    result = {'passed':True,'scope':'derived all100 versus frozen non-edge in-band count from original ownership demand; no inference',
              'count':len(records),'difference_count':sum(r['count_difference'] != 0 for r in records),
              'target_difference_count':sum(r['frozen_target_met'] != r['contract_target_met'] for r in records),'rows':records}
    save(output,result)
    return result


def run(jobdir, source, output):
    import torch
    from satellite_rl.coverage.reconstruction_ppo import RelationPolicy
    from satellite_rl.coverage.scenes import load_dataset
    import satellite_rl
    started = time.time()
    directory = Path(jobdir)
    cfg, job, status = [read(directory / name) for name in ('config.json','job.json','status.json')]
    check(status['stage'] == 'completed' and status['env_step'] == 262144 and status['update_step'] == 4096,
          'endpoint completion budget')
    check(job['git_commit'] == source, 'science source')
    cp = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    final = Path(job['model_dir'])/'final'
    meta = read(final/'model.json')
    weights = torch.load(final/'policy.pt', map_location='cpu', weights_only=False)
    for record in (cp,meta):
        check(record['run_id'] == job['run_id'] and record['source_commit'] == source
              and record['source_root'] == job['source_root'] and record['config_hash'] == config_digest(cfg),
              'endpoint run/source/config binding')
    check(cp['step'] == 262144 and cp['update'] == 4096 and cp['last_validation'] == 262144
          and cp['gate_passed'] and cp['episode'] == status['episode'], 'checkpoint counters/gate')
    check(meta['env_step'] == 262144 and meta['selection'] == 'fixed_budget_endpoint'
          and meta['training_seed'] == 51001 and meta['network'] == cfg['network'], 'fixed endpoint selection')
    check(sha(directory/'checkpoint.pt') == status['checkpoint_sha256'], 'checkpoint SHA')
    check(sha(final/'policy.pt') == meta['policy_sha256'], 'policy SHA')
    check(cp['dataset_hashes'] == {k:v['sha256'] for k,v in cfg['datasets'].items()}
          and cp['start_hashes'] == {k:v['sha256'] for k,v in cfg['reconstruction_starts'].items()}, 'checkpoint input hashes')
    policy = RelationPolicy()
    for state in (cp['model'], weights):
        policy.load_state_dict(state, strict=True)
        check(all(torch.isfinite(v).all().item() for v in state.values()), 'finite policy parameters')
    check(set(weights) == set(cp['model']) and all(torch.equal(v,cp['model'][k]) for k,v in weights.items()),
          'final equals checkpoint tensors')
    tensor_count = 0
    def finite_optimizer(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            check(torch.isfinite(value).all().item(), 'finite optimizer state')
        elif isinstance(value, dict):
            for item in value.values(): finite_optimizer(item)
        elif isinstance(value, (list,tuple)):
            for item in value: finite_optimizer(item)
    finite_optimizer(cp['optimizer'])
    datasets, starts, certificates = {}, {}, {}
    for role in ('train','validation'):
        data, initial = cfg['datasets'][role], cfg['reconstruction_starts'][role]
        check(sha(data['path']) == data['sha256'] and sha(initial['path']) == initial['sha256'], 'input file SHA')
        dataset = load_dataset(data['path'],'cluster',cfg['coverage'])
        datasets[role] = {s.scene_id:s for s in dataset.scenes}
        starts[role] = {r['scene_id']:r for r in jsonl(initial['path'])}
        certificates[role] = {s.scene_id:edge_components(s,cfg['coverage']['reconstruction']['edge_demand_ratio']) for s in dataset.scenes}
    counts = {'validation_terminal':0,'validation_best':0,'training_terminal':0,'training_best':0}
    distinct = set()
    max_gap = 0.0
    def physical(row, role):
        nonlocal max_gap
        scene = datasets[role][row['scene_id']]
        start = starts[role][row['scene_id']]
        if role == 'train':
            # Frozen runner records preloaded training initialization as 0,
            # unlike validation, which carries the shared preparation time.
            check(row['initialization_seconds'] == 0 and row['final_check_seconds'] == 0,
                  'training preloaded timing convention')
            start = dict(start, initialization_seconds=0.0)
        audit_row(scene,start,row,cfg['coverage'],expected_components=certificates[role][scene.scene_id])
        counts[role.replace('train','training')+'_terminal'] += 1
        max_gap = max(max_gap,audit_best(scene,start,row,cfg['coverage']['reconstruction']))
        counts[role.replace('train','training')+'_best'] += 1
    ledger = jsonl(directory/'evaluations.jsonl')
    steps = list(range(16384,262145,16384))
    check([x['scheduled_step'] for x in ledger] == steps,'16 unique ordered validation nodes')
    nodes = []
    for entry in ledger:
        step = entry['scheduled_step']
        path = directory/'validation_nodes'/f'{step}.json'
        node = read(path)
        check(entry['node_sha256'] == sha(path) and Path(entry['node']).resolve() == path.resolve()
              and entry['actual_env_step'] == node['actual_env_step'] == node['scheduled_step'] == step
              and node['config_hash'] == config_digest(cfg)
              and entry['scene_count'] == node['scene_count'] == len(node['rows']) == 20,
              'validation node ledger/hash/count binding')
        check({r['scene_id'] for r in node['rows']} == set(datasets['validation']), 'validation scene uniqueness')
        for row in node['rows']: physical(row,'validation')
        nodes.append({'step':step,'sha256':sha(path),'rows':20})
    max_step = 0
    with (directory/'episodes.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            check(row['training_seed'] == 51001 and 0 < row['terminal_env_step'] <= 262144,'training row budget/seed')
            check(row['terminal_env_step'] not in distinct,'duplicate terminal interaction step')
            distinct.add(row['terminal_env_step'])
            max_step = max(max_step,row['terminal_env_step'])
            physical(row,'train')
    check(counts['training_terminal'] == status['episode'],'training episode count')
    result = {'passed':True,'scope':'read-only endpoint strict loading, all validation and training original terminals plus diagnostic best; no inference or new evaluation',
              'run_id':job['run_id'],'source_commit':source,'config_hash':config_digest(cfg),
              'checkpoint_sha256':sha(directory/'checkpoint.pt'),'final_policy_sha256':sha(final/'policy.pt'),
              'counts':counts,'nodes':nodes,'optimizer_finite_tensors':tensor_count,'last_training_terminal_step':max_step,
              'best_terminal_max_quality_gap':max_gap,'episodes_sha256':sha(directory/'episodes.jsonl'),
              'runtime':{'python':sys.executable,'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
                         'package_file':satellite_rl.__file__},'elapsed_seconds':time.time()-started,
              'metadata_note':{'completed_status_residual_stop_reason':status.get('stop_reason'),
                               'training_timing':'preloaded initialization and final check recorded as zero by frozen runner; not full allocation timing'}}
    save(output,result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job-dir',required=True)
    p.add_argument('--science')
    p.add_argument('--formal-best-only',action='store_true')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    print(json.dumps(formal_best(a.job_dir,a.output) if a.formal_best_only else
                     run(a.job_dir,a.science,a.output),ensure_ascii=False))
