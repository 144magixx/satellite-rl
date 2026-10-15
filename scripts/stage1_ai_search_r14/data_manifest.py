"""R14 metadata selection only. Never open/hash/decode an aircraft payload."""
import copy
import hashlib
import json
from pathlib import Path

RECIPE_SHA = '349370987e28f5ba925f8c0412d463d0408d694b865284d2db0a66ec339f39c3'
DESIGN_SHA = '9a238198053c3a39db9fe65623ea7343e224eb5ce0df0457474b985b77672d74'
PARTITIONS = {'train': tuple(range(0, 288, 3)),
              'calibration': tuple(range(288, 565, 12)),
              'locked_known_development': tuple(range(576, 596))}


def metadata_json(path, expected_sha, limit=8 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('bounded pinned metadata bytes required')
    return json.loads(raw), {'path': str(path), 'sha256': expected_sha, 'bytes': len(raw)}


def select_metadata(recipe, recipe_ref):
    if (recipe_ref['sha256'] != RECIPE_SHA or recipe.get('schema') != 'R7UniformSourceRecipeV1'
            or recipe.get('fixed_request_bps') != 21600000 or len(recipe['rows']) != 596):
        raise ValueError('original pinned596 uniform recipe required')
    selected = []
    for partition, indices in PARTITIONS.items():
        for i in indices:
            row = copy.deepcopy(recipe['rows'][i])
            actual, source = ('train', 'train') if i < 576 else ('development', 'validation')
            if (row['actual_role'] != actual or row['source_role'] != source
                    or row['source_phase_index'] != (i if i < 576 else i - 576)
                    or row['original_mother_id'] != source + ':' + row['UTC']
                    or row['case_id'] != f"r7:{actual}:{row['original_mother_id']}:uniform21600000"
                    or row['UTC'].startswith('2024-01-15')
                    or not 1 <= row['source_aircraft_count'] <= 2000):
                raise ValueError('original identity/count/date differs')
            selected.append({'recipe_index': i, 'r14_partition': partition,
                             'historically_exposed': True, 'original_recipe_row': row,
                             'raw_payload_qualification': 'not_entered'})
    if (len({r['original_recipe_row']['case_id'] for r in selected}) != 140
            or sum(r['original_recipe_row']['source_aircraft_count'] for r in selected) != 151814):
        raise ValueError('fixed140/count metadata differs')
    return {'schema': 'R14ExactRecipeReferencesV1', 'recipe_ref': recipe_ref,
            'source_job_ref': copy.deepcopy(recipe['source_job_ref']), 'design_sha256': DESIGN_SHA,
            'rows': selected, 'payload_reads': 0, 'payload_hash_reads': 0,
            'payload_decodes': 0, 'scientific_execution_authorized': False,
            'all_B_qualified': False, 'independent_confirmation_mothers': 0}


def lstat_payload_paths(manifest):
    """Optional existence/size observation; no dereference, read, hash or remote IO."""
    result = []
    for entry in manifest['rows']:
        path = Path(entry['original_recipe_row']['source_local_path'])
        try:
            st = path.lstat()
            observation = {'present': True, 'size': st.st_size, 'mode': st.st_mode,
                           'mtime_ns': st.st_mtime_ns, 'symlink': path.is_symlink()}
        except FileNotFoundError:
            observation = {'present': False}
        result.append({'recipe_index': entry['recipe_index'], 'path': str(path),
                       'lstat_only': observation, 'payload_qualified': False})
    return result


def require_all_B(manifest, receipts):
    """No substitution, omission, retry or fallback can unlock the sole fit."""
    expected = {r['recipe_index']: r for r in manifest['rows']}
    if set(receipts) != set(expected) or len(expected) != 140:
        raise ValueError('all fixed140 B receipts required before fit')
    for i, receipt in receipts.items():
        if (receipt.get('status') != 'success' or receipt.get('proposal_success') is not True
                or receipt.get('qualified') is not True or receipt.get('complete_cost_exit') is not True
                or receipt.get('recipe_index') != i
                or receipt.get('case_id') != expected[i]['original_recipe_row']['case_id']
                or receipt.get('independent_layouts') != ['raw_W', 'Wenv', 'D5terminal']):
            raise ValueError('missing/failed/unknown fixed B closes dependent science')
    return True
