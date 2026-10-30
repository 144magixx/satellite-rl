"""Load W32 case/layout files through the existing scientific input validator."""
from functools import lru_cache
from pathlib import Path

from satellite_rl.shared_interface.content import content_sha, read_wire
from satellite_rl.shared_interface.contracts import Authority
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
from satellite_rl.shared_interface.stages import UniformStage2Input


def authority_for(case, layout):
    registry = dict(schema='R7PublishedParentRegistryV1',
                    case_sha_by_id={case['case_id']: case['content_sha256']},
                    layout_shas=[layout['content_sha256']], allocation_shas=[])
    return Authority(registry, content_sha(registry))


@lru_cache(maxsize=700)
def _read(path):
    return read_wire(Path(path).read_bytes())


def load_input(record):
    case, layout = _read(record['case_path']), _read(record['layout_path'])
    if record['case_id'] != case['case_id'] or record['role'] != case['actual_role']:
        raise ValueError('case identity/role mismatch')
    return UniformStage2Input(case, layout, authority_for(case, layout), PHYSICS)
