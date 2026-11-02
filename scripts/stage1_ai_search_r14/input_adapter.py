"""Bounded prospective R14 source decoder. Not invoked during preparation.

The shared-domain receipt is reserved before reading; failures retain claimed
IO and do not permit retries. Actual decompression bytes are separately saved.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
from .runtime import exclusive, atomic_bytes, json_bytes, require_execution
from .data_manifest import PARTITIONS


class InputLedger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def reserve(self, index, compressed_bytes, aircraft_count):
        if (type(index) is not int or type(compressed_bytes) is not int or compressed_bytes < 0
                or type(aircraft_count) is not int or not 1 <= aircraft_count <= 2000):
            raise ValueError('exact bounded input reservation')
        with exclusive(self.directory / '.input.lock'):
            target = self.directory / 'input_ledger.json'
            ledger = json.loads(target.read_bytes()) if target.exists() else {}
            if str(index) in ledger:
                raise ValueError('one raw qualification per fixed mother; unknown not retried')
            entry = {'compressed_bytes': compressed_bytes, 'decoded_reserved': 32 * 1024 ** 2,
                     'aircraft_count': aircraft_count, 'state': 'reserved'}
            proposed = {**ledger, str(index): entry}
            if (sum(x['compressed_bytes'] for x in proposed.values()) > 512 * 1024 ** 2
                    or sum(x.get('decoded_actual', x['decoded_reserved']) for x in proposed.values()) > 2 * 1024 ** 3
                    or sum(x['aircraft_count'] for x in proposed.values()) > 280000):
                raise ValueError('shared compressed/decoded/aircraft quota; no mother substitution')
            atomic_bytes(target, json_bytes(proposed))

    def finish(self, index, decoded_actual):
        if type(decoded_actual) is not int or not 0 <= decoded_actual <= 32 * 1024 ** 2:
            raise ValueError('bounded actual decompression cost')
        with exclusive(self.directory / '.input.lock'):
            target = self.directory / 'input_ledger.json'
            ledger = json.loads(target.read_bytes())
            entry = ledger[str(index)]
            if entry['state'] != 'reserved':
                raise ValueError('raw qualification terminal immutable')
            entry.update(state='decoded', decoded_actual=decoded_actual)
            atomic_bytes(target, json_bytes(ledger))


def read_source(selected, exact_transport_ref, *, config, ledger, checkpoint):
    """One compressed acquire/hash and bounded streaming decode; no geometry.

    This returns every original row unchanged. Typed case sealing and full-ID
    metadata-source binding are separate admitted steps, not silently inferred
    from the recipe. Jan15 is rejected before any open.
    """
    require_execution(config)
    if config.get('data_payload_access_authorized') is not True:
        raise ValueError('raw payload qualification separately admitted')
    row = selected['original_recipe_row']
    if (selected['recipe_index'] not in PARTITIONS.get(selected.get('r14_partition'), ())
            or row['UTC'].startswith('2024-01-15')
            or exact_transport_ref.get('sha256') != row['source_asset_SHA']
            or type(exact_transport_ref.get('bytes')) is not int
            or not 0 < exact_transport_ref['bytes'] <= 32 * 1024 ** 2):
        raise ValueError('exact approved original source transport required')
    ledger.reserve(selected['recipe_index'], exact_transport_ref['bytes'], row['source_aircraft_count'])
    checkpoint('before_raw_acquire', {'recipe_index': selected['recipe_index']})
    with Path(exact_transport_ref['path']).open('rb') as stream:
        raw = stream.read(exact_transport_ref['bytes'] + 1)
    if len(raw) != exact_transport_ref['bytes'] or hashlib.sha256(raw).hexdigest() != exact_transport_ref['sha256']:
        raise ValueError('source bytes/SHA differ; preserve reserved unknown cost')
    checkpoint('raw_hash_complete', {'compressed_bytes': len(raw)})
    decoded_bytes, planes = 0, []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        while True:
            # Cumulative bound applies even to a single adversarial long line.
            line = stream.readline(32 * 1024 ** 2 - decoded_bytes + 1)
            if not line:
                break
            decoded_bytes += len(line)
            if decoded_bytes > 32 * 1024 ** 2 or not line.strip():
                raise ValueError('decoded file/line bound or empty aircraft row')
            planes.append(json.loads(line))
            if len(planes) > row['source_aircraft_count']:
                raise ValueError('extra aircraft rows; do not truncate or deduplicate')
    if len(planes) != row['source_aircraft_count']:
        raise ValueError('all original aircraft count required')
    ids = [p['aircraft_id'] for p in planes]
    if any(type(x) is not str or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('full stable unique original aircraft IDs')
    ledger.finish(selected['recipe_index'], decoded_bytes)
    checkpoint('raw_decode_complete', {'decoded_bytes': decoded_bytes, 'aircraft_count': len(planes)})
    return planes


def seal_source(selected, planes, metadata, metadata_ref, raw_ref):
    """Exact original job pointer/count provenance to typed case; no GEO.

    This is a derivation-only provisional authority. Independent publication
    and algorithm admission must later use the root-pinned R14 envelope.
    """
    import numpy as np
    from satellite_rl.shared_interface.stages import make_case
    from satellite_rl.shared_interface.content import content_sha
    from satellite_rl.shared_interface.contracts import Authority,validate_case
    from .identity_adapter import task_binding
    row=selected['original_recipe_row']
    def pointer(path):
        if not path.startswith('/'):raise ValueError('explicit original metadata pointer')
        value=metadata
        for key in path[1:].split('/'):
            key=key.replace('~1','/').replace('~0','~')
            value=value[int(key) if isinstance(value,list) else key]
        return value
    snapshot=pointer(row['metadata_snapshot_JSON_pointer']);file=pointer(row['metadata_file_JSON_pointer'])
    if (metadata_ref['sha256']!=row['metadata_job_SHA'] or raw_ref['sha256']!=row['source_asset_SHA']
            or snapshot['source_observations_ref']['sha256']!=raw_ref['sha256']
            or snapshot['selected_time_utc']!=row['UTC'] or file['scene_id']!=row['original_scene_id']
            or snapshot['counts']!=row['source_selection_counts']
            or len(planes)!=row['source_aircraft_count']
            or snapshot['counts']['selected_aircraft']!=len(planes)):
        raise ValueError('complete exact source metadata chain differs')
    ids=[p['aircraft_id'] for p in planes]
    positions=np.asarray([[p['lat'],p['lon']] for p in planes],dtype='<f8')
    source={'raw_perplane_SHA':raw_ref['sha256'],'metadata_job_SHA':metadata_ref['sha256'],
        'original_remote_ref':row['source_remote_path'],'original_case_index':row['source_phase_index'],
        'original_scene_id':row['original_scene_id'],'original_mother_id_derivation':row['original_mother_id_derivation'],
        'metadata_snapshot_JSON_pointer':row['metadata_snapshot_JSON_pointer'],
        'metadata_file_JSON_pointer':row['metadata_file_JSON_pointer'],
        'ancestry_fields':[{k:p[k] for k in ('timestamp_utc','source_date','source_sha256','altitude_m') if k in p} for p in planes],
        'obsolete_request_assignment_fields_used':False}
    case=make_case({k:row[k] for k in ('case_id','actual_role','source_role','original_mother_id','UTC')} | {
        'source_asset_SHA':raw_ref['sha256'],'source':source},ids,positions)
    registry={'schema':'R7PublishedParentRegistryV1','case_sha_by_id':{case['case_id']:case['content_sha256']},
              'layout_shas':[],'allocation_shas':[]}
    validate_case(case,Authority(registry,content_sha(registry)),geometry_check=False)
    return case,task_binding(selected,case)


def exact_transport_template(manifest):
    """Static canonical remote paths from original recipe, never fabricate copies."""
    return {'schema':'R14ExactOriginalTransportTemplateV1','payload_access_authorized':False,
       'rows':[{'recipe_index':e['recipe_index'],'r14_partition':e['r14_partition'],
         'case_id':e['original_recipe_row']['case_id'],
         'local_source_path':e['original_recipe_row']['source_local_path'],
         'canonical_remote_source_path':e['original_recipe_row']['source_remote_path'],
         'sha256':e['original_recipe_row']['source_asset_SHA'],'bytes':None,
         'actual_transport_pin_qualified':False} for e in manifest['rows']]}
