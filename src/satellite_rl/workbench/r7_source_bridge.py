"""Require complete actual source closure and preserve original interface ancestry."""
import subprocess
from pathlib import Path
from .storage import read_json,digest
from .coverage_ai_search import pin

INTERFACE_COMMIT='6496abe61d8bdc5029388ce024508b15a7e306c7'
INTERFACE_RECEIPT_SHA='9fbf8fbd1924c977a6a6dd7c066225ce9dcfcd263c7c6bff35fa1df7c2a65e54'

def source_members(root,commit):
    root=Path(root);manifest_path=root/'source_manifest.json'
    if manifest_path.is_file():
        manifest=read_json(manifest_path,{})
        if manifest.get('format_version')!=2 or manifest.get('group')!='S1-AI17-R7' or manifest.get('git_commit')!=commit or not manifest.get('files'):raise ValueError('actual complete frozen R7 source manifest')
        names=set(manifest['files'])
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix()!='source_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        if names!=actual:raise ValueError('snapshot contains omitted or extra actual source members')
        return {name:digest(root/name) for name in sorted(names)}
    def git(*args):return subprocess.run(['git','-C',str(root),*args],capture_output=True,check=True).stdout
    if git('rev-parse','HEAD').decode().strip()!=commit:raise ValueError('actual service checkout epoch differs')
    scopes=['src','scripts','pyproject.toml','.gitattributes']
    if git('ls-tree','HEAD','data').strip():scopes.append('data')
    if git('status','--porcelain','--',*scopes).strip():raise ValueError('live source archive scope is not committed clean')
    names=[name.decode('utf8') for name in git('ls-tree','-r','--name-only','-z',commit,'--',*scopes).split(b'\0') if name]
    if not names or not any(n.startswith('scripts/') for n in names):raise ValueError('complete actual src/scripts closure missing')
    return {name:digest(root/name) for name in names}

def validate_bridge(record,root,commit,design_sha,phase):
    if record.get('schema')!='R7CompleteSourceInterfaceBridgeV1' or record.get('passed')is not True or record.get('group')!='S1-AI17-R7' or record.get('scientific_commit')!=commit or record.get('design_sha256')!=design_sha or phase not in record.get('admitted_phases',[]):raise ValueError('root exact complete-source phase bridge')
    actual=source_members(root,commit)
    if record.get('source_file_sha256')!=actual:raise ValueError('qualification must match every actual src/scripts/archive source member, not a subset')
    ancestor_ref=record['accepted_interface_ref']
    if ancestor_ref.get('sha256')!=INTERFACE_RECEIPT_SHA or record.get('accepted_interface_source_commit')!=INTERFACE_COMMIT:raise ValueError('original interface receipt/source ancestry must remain unchanged')
    ancestor=read_json(pin(ancestor_ref),{})
    if ancestor.get('passed')is not True or ancestor.get('source_commit')!=INTERFACE_COMMIT:raise ValueError('original interface acceptance differs')
    leaves=ancestor.get('actual_shared_module_SHAs',{})
    if len(leaves)!=9 or any(actual.get('src/satellite_rl/shared_interface/'+name)!=sha for name,sha in leaves.items()):raise ValueError('actual9 shared source leaves differ from original qualification')
    return actual
