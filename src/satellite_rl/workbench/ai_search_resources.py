"""Fresh lane observations and fail-closed AI17 worker admission."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import shutil

from .storage import now, read_json


def observe(archive_root):
    root = Path(archive_root).resolve()
    def query(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=12)
        if result.returncode: raise RuntimeError(result.stderr[-600:])
        return result.stdout.strip()
    foreign = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid(): continue
        try:
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
            if ('satellite_rl.workbench.runner' not in command or str(root) in command): continue
            raw = (directory / 'stat').read_text()
            foreign.append({'pid': int(directory.name), 'birth_ticks': int(raw[raw.rfind(')')+2:].split()[19]),
                            'command': command[:2500]})
        except (FileNotFoundError, PermissionError, ProcessLookupError): pass
    return {'observed_utc': now(), 'ssh_port': int(root.parent.name),
            'gpu': query(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
            'compute_apps': query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']),
            'foreign_workers': foreign, 'disk_free_bytes': shutil.disk_usage(root).free}


def admission(store, job):
    from .coverage_ai_search import GROUP, deployment, pin
    port = deployment(store)
    route = read_json(store.archive_root.parent / 'ops/resource_route.json', {})
    if route.get('group') != GROUP or str(route.get('ssh_port')) != port or route.get('ready') is not True:
        raise ValueError('AI17 lane has no current ready route')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(route['observed_utc'])).total_seconds()
    if not -5 <= age <= 120: raise ValueError('AI17 resource route is stale')
    if route.get('scientific_commit') != job.get('scientific_commit'):
        raise ValueError('resource route belongs to another scientific epoch')
    if port == '20474':
        release = read_json(pin(route.get('f20_return_receipt')), {})
        if (release.get('group') != 'S2-F20' or release.get('ssh_port') != 20474 or
                release.get('returned') is not True or
                not all(release.get('complete', {}).get(k) is True for k in ('compute', 'archives', 'report', 'resource_return'))):
            raise ValueError('74 requires F20 final compute/archive/report/resource return proof')
    observation = observe(store.archive_root)
    if observation['foreign_workers']: raise ValueError('foreign scientific workers remain active')
    gpu_uuid = observation['gpu'].split(',')[0].strip()
    if gpu_uuid != route.get('gpu_uuid'): raise ValueError('GPU identity changed after resource route observation')
    return observation
