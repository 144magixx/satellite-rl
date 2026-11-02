"""Explicit metadata-only R14 preparation entry point; no execution switch."""
import argparse
import hashlib
import json
from pathlib import Path
from .data_manifest import RECIPE_SHA, DESIGN_SHA, metadata_json, select_metadata, lstat_payload_paths
from .runtime import json_bytes, development_schedule, STAGE_SECONDS


def prepare(repo, worktree, output):
    repo, worktree, output = Path(repo), Path(worktree), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recipe, ref = metadata_json(repo / 'outputs/experiments/first_stage_ai_search_20261013/round7/source/uniform_train_dev_recipe.json', RECIPE_SHA)
    manifest = select_metadata(recipe, ref)
    manifest['local_lstat_observations'] = lstat_payload_paths(manifest)
    (output / 'exact140_metadata_manifest.json').write_bytes(json_bytes(manifest))
    files = [p for p in (worktree / 'scripts/stage1_ai_search_r14').glob('*.py')]
    # Candidate source snapshots only; final transitive closure awaits all owners.
    files += [worktree / p for p in ('src/satellite_rl/coverage/ai_search_r5/core.py',
              'src/satellite_rl/shared_interface/contracts.py',
              'src/satellite_rl/workbench/r7_uniform_data.py')]
    refs = []
    for p in sorted(files):
        raw = p.read_bytes()
        refs.append({'path': str(p), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    config = {'schema': 'R14FalseRuntimePreparationV1', 'group': 'S1-AI17-R14',
              'scientific_execution_authorized': False, 'root_exact_source_input_admitted': False,
              'source_specific_pure_checks_accepted': False, 'actual_runtime_owner_stop_space_admitted': False,
              'engineering_execution_authorized': False, 'data_payload_access_authorized': False,
              'root_admission_ref': None, 'runtime_epoch_ref': None,
              'design_sha256': DESIGN_SHA, 'candidate_ports': [20473, 20474], 'port': None,
              'excluded_ports': [20418, 20424, 20627], 'retry': 0, 'old_budget_transfer': 0,
              'stage_seconds': STAGE_SECONDS, 'outer_all_seconds': 10800,
              'B_mother_outer_seconds': 120, 'B_stop_new_work_seconds': 108,
              'storage_per_physical_domain': {'canonical_bytes': 2147483648, 'archive_bytes': 2281701376,
                 'temporary_bytes': 536870912, 'peak_bytes': 4966055936,
                 'reserve_bytes': 6442450944, 'minimum_free_bytes': 12884901888,
                 'aggregate_73_74_shared_NFS': True, 'aggregate_D_E_same_disk': True},
              'auth': {'password_environment_only': 'BUPT_PASSWORD', 'known_host_required': True,
                       'agent': False, 'key_lookup': False},
              'cold_request_scope': ['fresh_parent_before_launch','case_B_cache_read_hash',
                 'own_B_numeric_audit','policy_load_feature_kernel','own_terminal_numeric_audit',
                 'atomic_output','actual_reap_and_owned_tree_exit'],
              'future_teacher_online_access': False, 'OS_page_cache_cold_claim': False,
              'development_schedule': development_schedule()}
    (output / 'runtime_FALSE_config.json').write_bytes(json_bytes(config))
    preparation = {'schema': 'R14RuntimePreparationV1', 'state': 'components_only_not_dispatchable',
        'source_candidates': refs, 'exact_import_closure_qualified': False,
        'raw_payload_transport_qualification_pending': True,
        'actual': {'payload_reads': 0, 'payload_hash_reads': 0, 'decode': 0, 'GEO': 0,
                   'solver': 0, 'model_fit': 0, 'evaluation': 0, 'SSH': 0},
        'required_next': ['Independent source and import closure review including original math dependencies',
          'Freeze bounded raw derivation/transport and140 source bindings before any payload access',
          'Bind exact source/input/8fixtures and accept pure checks',
          'Fresh interpreter/dependency/boot/owned-process/global-pause/physical-space admission',
          'Freeze one persistent lane backend and direct child wait/reap source before finite single dispatch',
          'Execute only admitted8fixtures; qualify before B/scientific production',
          'All140 B independent rawW/Wenv/D5terminal and actual cost/exit before sole fit']}
    (output / 'preparation_manifest.json').write_bytes(json_bytes(preparation))
    return preparation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--worktree', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.repo, args.worktree, args.output)
