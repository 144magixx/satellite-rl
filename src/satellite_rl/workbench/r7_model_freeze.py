"""R6 model-only complete Git source snapshots including required scripts."""
import io,subprocess,zipfile
from pathlib import Path
from .storage import inside,digest,read_json,now
from satellite_rl.training.run_monitor import atomic_json

def freeze_source(store):
 with store.lock:
  def git(*args):return subprocess.run(['git',*args],cwd=store.source_root,capture_output=True,check=True).stdout
  scopes=['src','scripts','pyproject.toml','.gitattributes']
  if git('ls-tree','HEAD','data').strip():scopes.append('data')
  if git('status','--porcelain','--',*scopes).strip():raise ValueError('complete model source must be committed clean')
  commit=git('rev-parse','HEAD').decode().strip();destination=store.root/'source_snapshots'/commit;manifest=read_json(destination/'source_manifest.json',{})
  if destination.exists():
   if manifest.get('format_version')!=2 or manifest.get('group')!='S1-AI17-R7' or not any(n.startswith('scripts/') for n in manifest.get('files',{})):raise ValueError('existing incomplete/other snapshot preserved; no retrofit')
   for name,sha in manifest['files'].items():
    if digest(inside(destination/name,destination))!=sha:raise ValueError('complete frozen source member changed')
   return destination,commit
  destination.mkdir(parents=True)
  with zipfile.ZipFile(io.BytesIO(git('archive','--format=zip',commit,*scopes))) as bundle:
   for item in bundle.infolist():inside(destination/item.filename,destination)
   bundle.extractall(destination)
  files={p.relative_to(destination).as_posix():digest(p) for p in destination.rglob('*') if p.is_file()}
  required=['src/satellite_rl/coverage/ai_search_r6/runtime_v3.py','scripts/stage1_ai_search_r7/plan_builder.py','scripts/stage1_ai_search_r5/audit.py','scripts/stage1_ai_search_r4/audit.py',
            'src/satellite_rl/shared_interface/contracts.py','src/satellite_rl/shared_interface/content.py','src/satellite_rl/shared_interface/stages.py']
  if any(n not in files for n in required):raise ValueError('required model/audit script dependency omitted')
  atomic_json(destination/'source_manifest.json',{'format_version':2,'group':'S1-AI17-R7','git_commit':commit,'source_root':str(store.source_root),'created_at':now(),'files':files,'dependency_roots':['src','.'],'no_external_cwd_fallback':True});return destination,commit
