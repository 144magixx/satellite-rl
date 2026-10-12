"""Reference original accepted pilot evidence without rewriting its JSON."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from . import ops
from .plan_builder import digest


def build(root_gate,source_freeze,pilot_source_manifest=None):
    root_gate=Path(root_gate).resolve();source_freeze=Path(source_freeze).resolve()
    root=json.loads(root_gate.read_text(encoding='utf-8'));freeze=json.loads(source_freeze.read_text(encoding='utf-8'))
    local={'root_gate':root_gate,'audit':Path(root['audit_ref']['path']),'gate':Path(root['gate_ref']['path']),
        'source_freeze':source_freeze,'interpretation':Path(root['G3_pre_result_interpretation_ref']['path'])}
    bundle={'aliases':{},'scientific_core_sha256':freeze['core_git_blob_sha256']};uploads={}
    from .deploy import SOURCE
    allowed={'src/satellite_rl/workbench/coverage_ai_search_r2.py','src/satellite_rl/workbench/ai_search_r2_runner.py',
        'src/satellite_rl/workbench/frozen_round_tasks.py'}
    names=subprocess.run(['git','-C',str(SOURCE),'ls-tree','-r','--name-only',freeze['scientific_commit'],'src'],capture_output=True,text=True,check=True).stdout.splitlines()
    git_blobs={relative:hashlib.sha256(subprocess.run(['git','-C',str(SOURCE),'show',freeze['scientific_commit']+':'+relative],capture_output=True,check=True).stdout).hexdigest()
        for relative in names if relative not in allowed}
    if pilot_source_manifest is None:
        catalog=json.loads((ops.LOCAL/'analysis/pilot/collection_manifest.json').read_text(encoding='utf-8'))
        original=catalog['actual_source_manifests'][0]
        pilot_source_manifest=original['local_source_manifest_ref']['path']
        bundle['original_source_manifest']=original['original_source_manifest_ref']
    signed=json.loads(Path(pilot_source_manifest).read_text(encoding='utf-8'))
    if signed['git_commit']!=freeze['scientific_commit']:raise ValueError('original executed source manifest is not f02')
    executed={k:v for k,v in signed['files'].items() if k.startswith('src/') and k not in allowed}
    if executed!=git_blobs:raise ValueError('original execution bytes differ from separately measured old Git blobs')
    bundle['scientific_dependency_sha256']=executed
    bundle['original_git_blob_sha256']=git_blobs
    for key,path in local.items():
        path=path.resolve();sha=digest(path);remote=ops.BASE+'/source/extension_admission/'+sha+'.json'
        ref={'path':remote,'sha256':sha};bundle[key]=ref;bundle['aliases'][str(path)]=ref
        uploads[remote]={'local_path':str(path),'sha256':sha}
    # Internal references use original Windows strings, not Path-normalized
    # replacements; both spellings may be needed and always share actual SHA.
    for key,original in [('audit',root['audit_ref']),('gate',root['gate_ref']),('interpretation',root['G3_pre_result_interpretation_ref'])]:
        if bundle[key]['sha256']!=original['sha256']:raise ValueError('original accepted evidence bytes changed')
        bundle['aliases'][original['path']]=bundle[key]
    return bundle,uploads


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root-gate','source-freeze','output','uploads-output'):p.add_argument('--'+name,required=True)
    p.add_argument('--pilot-source-manifest')
    args=p.parse_args();bundle,uploads=build(args.root_gate,args.source_freeze,args.pilot_source_manifest)
    atomic_json(args.output,bundle);atomic_json(args.uploads_output,uploads)
    print(json.dumps({'bundle_sha256':digest(args.output),'files':len(uploads),'science_calls':0}))
