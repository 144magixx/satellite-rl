"""W27 content-pinned admissions and accounted pilot continuation (never sign)."""
from pathlib import Path
import json
from satellite_rl.t0_width.data import file_sha, read_json_ref
from satellite_rl.t0_width.logging import content_sha


def validate_training_admission(config):
    gate = read_json_ref(config['data_admission_ref'])
    if (gate.get('schema') != 'W27TrainingDataAdmissionV1' or gate.get('root_approved') is not True
            or gate.get('experiment_id') != 'S2-W27' or gate.get('source_SHA') != config['source_sha']
            or gate.get('qualified_counts') != {'train': 576, 'development': 20}
            or not gate.get('logical_dataset_identity')):
        raise ValueError('W27 exact train/development root admission required')
    if config.get('data_manifest_sha') != gate['logical_dataset_identity']:
        raise ValueError('config logical dataset identity differs from admitted ordered partitions')
    for role, key, count in [('train', 'train_cases', 576), ('development', 'dev_cases', 20)]:
        ref = config[role + '_manifest_ref']
        if gate['partition_manifest_SHAs'].get(role) != ref['sha256']:
            raise ValueError('partition file SHA differs')
        manifest = read_json_ref(ref)
        if manifest.get('role') != role or len(manifest.get('rows', [])) != count or manifest['rows'] != config[key]:
            raise ValueError('ordered exact W26 reused partition required')
        expected = gate['ordered_content_SHAs'][role]
        actual = [[r['case_content_SHA'], r['layout_content_SHA']] for r in manifest['rows']]
        if actual != expected:
            raise ValueError('logical case/layout identity or order changed by rebase')
    return gate


def validate_continuation(directory, config, receipt_ref):
    """No config rewrite or free replay; binds root decision and exact trusted current."""
    import torch
    directory = Path(directory)
    r = read_json_ref(receipt_ref)
    status = json.loads((directory/'progress.json').read_text(encoding='utf8'))
    current = Path(config.get('model_dir', directory/'models'))/'current.pt'
    if (r.get('schema') != 'W27PilotContinuationV1' or r.get('root_approved') is not True
            or r.get('experiment_id') != 'S2-W27' or r.get('run_id') != config['run_id']
            or r.get('config_SHA') != content_sha(config) or r.get('source_SHA') != config['source_sha']
            or r.get('rootpauseepoch') != config['rootpauseepoch'] or r.get('env_step') != 50000
            or r.get('gate_passed') is not True or not r.get('decision_ref')
            or status.get('stage') != 'pilot_complete' or status.get('env_step') != 50000):
        raise ValueError('root same-ID completed pilot decision required')
    decision = read_json_ref(r['decision_ref'])
    if decision.get('root_approved') is not True or decision.get('passed_variants') is None:
        raise ValueError('locked all-candidate development decision required')
    if config['variant'] != 'B' and config['variant'] not in decision['passed_variants']:
        raise ValueError('candidate failed continuation gate')
    if not decision['passed_variants']:
        raise ValueError('baseline alone cannot continue after both candidates fail')
    if file_sha(current) != r['checkpoint_SHA']:
        raise ValueError('exact current pilot checkpoint required')
    saved = torch.load(current, map_location='cpu', weights_only=False)
    if saved['config_sha'] != content_sha(config) or saved['progress']['env_step'] != 50000 or 50000 not in saved['progress']['completed_nodes'] or saved['progress'].get('completed'):
        raise ValueError('pilot checkpoint not exact closed development node')
    for name, offset in saved['log_offsets'].items():
        log = directory/'logs'/(name+'.jsonl.gz')
        if (log.stat().st_size if log.exists() else 0) != offset:
            raise ValueError('unaccounted evidence suffix; no replay')
    wal = directory/'training_decision_wal.pt'
    if wal.exists():
        w = torch.load(wal, map_location='cpu', weights_only=False)
        if w.get('config_sha') != content_sha(config) or w.get('stage') != 'committed' or w.get('env_step') != 50000:
            raise ValueError('unaccounted decision WAL')
    if (directory/'evaluation_wal.pt').exists():
        raise ValueError('pilot development not safely closed')
    return r
