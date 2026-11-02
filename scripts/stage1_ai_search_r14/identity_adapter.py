"""R14 task partitions wrap original source roles/IDs; old contracts untouched."""
import hashlib
from .runtime import json_bytes, require_execution


def task_binding(selected, case):
    row = selected['original_recipe_row']
    fields = ('case_id', 'actual_role', 'source_role', 'original_mother_id', 'UTC')
    if any(case.get(k) != row[k] for k in fields):
        raise ValueError('original mother/case/source role must be preserved')
    if case.get('source_asset_SHA') != row['source_asset_SHA']:
        raise ValueError('original raw source SHA differs')
    if (len(case['aircraft_ids']) != row['source_aircraft_count']
            or case['fixed_request_bps'] != 21600000
            or case['total_request_bps_int64'] != row['source_aircraft_count'] * 21600000):
        raise ValueError('all aircraft/count/uniform request conservation')
    return {'schema': 'R14TaskCaseBindingV1', 'recipe_index': selected['recipe_index'],
            'r14_partition': selected['r14_partition'], 'case_id': case['case_id'],
            'case_content_sha256': case['content_sha256'], 'actual_role': case['actual_role'],
            'source_role': case['source_role'], 'original_mother_id': case['original_mother_id'],
            'historically_exposed': True}


def root_authority(envelope, expected_envelope_sha, expected_registry_sha, config):
    """A root-pinned R14 envelope carries unchanged shared registry semantics.

    No locally self-created registry is accepted as independent publication.
    The name of the original shared registry schema is historical, not an R7
    experiment authorization. R14 identity/usage is external and exact.
    """
    require_execution(config)
    if hashlib.sha256(json_bytes(envelope)).hexdigest() != expected_envelope_sha:
        raise ValueError('externally pinned R14 authority bytes')
    if (envelope.get('schema') != 'R14PublishedParentAuthorityV1'
            or envelope.get('independently_qualified') is not True
            or envelope.get('group') != 'S1-AI17-R14'):
        raise ValueError('independent R14 authority required')
    from satellite_rl.shared_interface.contracts import Authority
    return Authority(envelope['shared_registry'], expected_registry_sha)


def phase_identity(scene, case, binding, *, dataset_sha256, config):
    """Original R5 labels/development capability plus explicit new usage sidecar."""
    require_execution(config)
    if binding['case_id'] != case['case_id'] or binding['case_content_sha256'] != case['content_sha256']:
        raise ValueError('typed case/usage binding differs')
    if (case['actual_role'] not in ('train', 'development')
            or binding['r14_partition'] not in ('train', 'calibration', 'locked_known_development')):
        raise ValueError('R14 has no new heldout role')
    if (binding['r14_partition'] == 'locked_known_development') != (case['actual_role'] == 'development'):
        raise ValueError('source role cannot be rewritten for calibration')
    identity_sha = hashlib.sha256(json_bytes({'binding': binding, 'dataset_sha256': dataset_sha256,
                                             'input_sha256': scene.input_sha256})).hexdigest()
    identity = {'case_id': case['case_id'], 'mother_id': case['original_mother_id'],
                'role': case['actual_role'], 'profile': 'uniform21600000',
                'input_sha256': scene.input_sha256, 'identity_sha256': identity_sha}
    phase = {'role': identity['role'], 'stage': 'labels' if identity['role'] == 'train' else 'development',
             'admitted': True, 'identity_sha256': identity_sha, 'cases': {identity['case_id']: identity},
             'dataset_sha256': dataset_sha256}
    return identity, phase
