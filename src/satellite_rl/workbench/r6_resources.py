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

def controlled_science_command(command):
 import shlex,re
 try:tokens=shlex.split(command)
 except ValueError:return False
 if not tokens or not re.fullmatch(r'python(?:[0-9.]+)?(?:\.exe)?',PurePosixPath(tokens[0]).name):return False
 modules={'satellite_rl.workbench.runner','satellite_rl.workbench.r6_model_eval_stage','satellite_rl.workbench.r6_model_d5_stage','satellite_rl.workbench.ai_search_r6_model_runner'}
 index=1;program=None;kind=None
 while index<len(tokens):
  token=tokens[index]
  if token in ('-c','-','-i','--help','--version','-V'):return False
  if token=='-m':
   if index+1>=len(tokens):return False
   kind='module';program=tokens[index+1];index+=2;break
  if token=='--':
   index+=1
   if index>=len(tokens):return False
   kind='script';program=tokens[index];index+=1;break
  if token in ('-W','-X'):
   index+=2;continue
  if token in ('-B','-u','-O','-OO','-s','-S','-E','-I','-q','-v') or token.startswith(('-W','-X')):
   index+=1;continue
  if token.startswith('-'):return False
  kind='script';program=token;index+=1;break
 recognized=program in modules if kind=='module' else bool(program and PurePosixPath(program).name=='r6_runtime_bootstrap.py')
 return recognized and any(v in tokens[index:] for v in ('--job-dir','--request','--directory'))


def belongs_to_archive(command,root):
 # Match the exact archive as an argv token or prefix ending at its separator.
 # /archive-other and /archive2 are foreign even when their prefix matches.
 import shlex
 try:tokens=shlex.split(command)
 except ValueError:return False
 value=str(root).rstrip('/')
 return any(token==value or token.startswith(value+'/') for token in tokens)
