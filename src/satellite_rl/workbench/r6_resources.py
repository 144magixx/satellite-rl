"""R6 observations retain the actual nested archive scope and explicit lane."""
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from .storage import now

BASE=PurePosixPath('/home/u2024110224/first_stage_satellite_rl/s1_ai17')

def lane_from_archive(path):
    root=PurePosixPath(path)
    if '..' in root.parts:raise ValueError('R6 archive traversal')
    try:parts=root.relative_to(BASE).parts
    except ValueError as exc:raise ValueError('outside R6 archive scope') from exc
    if len(parts) not in (3,4) or parts[0] not in ('20418','20424','20473','20474'):
        raise ValueError('R6 lane/archive scope')
    if tuple(parts[1:]) not in (('round6','archive'),('round6','learning','archive')):
        raise ValueError('R6 exact label or learning archive required')
    return int(parts[0])

def observe(archive_root):
    root=Path(archive_root).resolve();port=lane_from_archive(root.as_posix())
    def query(args):
        p=subprocess.run(args,capture_output=True,text=True,timeout=12)
        if p.returncode:raise RuntimeError(p.stderr[-600:])
        return p.stdout.strip()
    foreign=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            command=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
            if 'satellite_rl.workbench.runner' not in command or str(root) in command:continue
            raw=(directory/'stat').read_text()
            foreign.append({'pid':int(directory.name),'birth_ticks':int(raw[raw.rfind(')')+2:].split()[19]),'command':command[:2500]})
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return {'observed_utc':now(),'ssh_port':port,'archive_root':str(root),
        'gpu':query(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        'compute_apps':query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),
        'foreign_workers':foreign,'disk_free_bytes':shutil.disk_usage(root).free}
