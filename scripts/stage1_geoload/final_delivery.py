"""Explicit G16 small delivery; plan is read-only, resume never repackages.

The spec pins small files, not data volumes. This package depends on accepted
canonical archives and cannot reproduce the experiment without those volumes.
No science, source mutation, directory scans or model loading occur here.

Spec schema g16_delivery_spec_v1: group, package_id, files[{role,path,sha256,
member}], and model_receipt_fields{verified,final_sha256,checkpoint_sha256,
export_verified,export_sha256} mapping dotted receipt keys where necessary.
Roles: report11, summary, closure, snapshot, science_plan, run_plan,
publication_manifest, source_index, reproduction, task_receipts, source_receipts;
gate_passed adds model_final, model_provenance, gpu_receipt, export_receipt.
All figure artifacts in publication manifest are added from its parent directory.
source_index schema: {sources:[{role,commit,files:[{path,sha256,canonical_ref}]}]}.
Required source roles scientific, controller, postprocessing. Files refer to
precise accepted canonical members; they are not duplicate big source archives.
source_receipts is an accepted exact node index {status:'accepted',nodes:[...]}
from original source receipts (those JSON receipts should be extra pinned files).
Node shape matches archive_acceptance: id,sha256,path or archive_path/member.
report11_status='final' explicitly excludes a proposal/outline from delivery.
model_provenance pins scientific_commit,final_sha256,checkpoint_sha256,run_id,
port,canonical_final_member (an accepted closure node id owned by train_mlm).
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

SCHEMA='g16_delivery_spec_v1'
SCIENCE='495bff976c59c40ddb33de6b5b37b20a894a1347'
MAX_FILE=16 << 20
MAX_MODEL=64 << 20
MAX_TOTAL=256 << 20
SUFFIXES={'.md','.json','.csv','.png','.pdf','.py','.txt','.pt'}
REQUIRED={'report11','summary','closure','snapshot','science_plan','run_plan',
          'publication_manifest','source_index','reproduction','task_receipts','source_receipts'}


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode('utf-8')


def safe(name):
    if not isinstance(name,str) or '\\' in name or any(p in ('','.', '..') for p in name.split('/')) or PurePosixPath(name).is_absolute():
        raise ValueError('unsafe member')
    return name


def pinned(row):
    path=Path(row['path']).resolve()
    if Path(row['path']).is_symlink() or not path.is_file():raise ValueError('missing or symlink small file')
    limit=MAX_MODEL if row.get('role')=='model_final' else MAX_FILE
    if path.suffix.lower() not in SUFFIXES or path.stat().st_size>limit:raise ValueError('raw archive/oversized input refused')
    if not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')) or sha(path)!=row['sha256']:raise ValueError('small file SHA differs')
    return path


def field(value,key):
    for part in key.split('.'):
        if not isinstance(value,dict) or part not in value:raise ValueError('model receipt field absent: '+key)
        value=value[part]
    return value


def plan(spec_path):
    """Inspect explicit small pins only; failures return not_ready and write nothing."""
    try:
        spec=read(spec_path)
        if spec.get('schema')!=SCHEMA or spec.get('group')!='S1-G16':raise ValueError('wrong spec')
        ident=spec.get('package_id','')
        if not re.fullmatch('[a-zA-Z0-9_-]{1,80}',ident):raise ValueError('unsafe package_id')
        rows=spec['files'];roles={row['role']:row for row in rows}
        if len(roles)!=len(rows) or not REQUIRED.issubset(roles):raise ValueError('missing or duplicate required small role')
        paths={role:pinned(row) for role,row in roles.items()}
        summary=read(paths['summary']);closure=read(paths['closure']);snapshot=read(paths['snapshot'])
        if summary.get('group')!='S1-G16' or summary.get('formal_ready') is not True or summary.get('status')!='formal_ready' or summary.get('issues')!=[]:
            raise ValueError('strict report not formal_ready')
        if summary.get('scientific_commit')!=SCIENCE:raise ValueError('scientific source differs')
        if closure.get('status')!='accepted' or closure.get('issues')!=[] or closure.get('group')!='S1-G16':raise ValueError('canonical archive closure not accepted')
        if not snapshot.get('compute_and_archives_complete') or not snapshot.get('all_computation_completed'):raise ValueError('input snapshot incomplete')
        for role in ('science_plan','run_plan'):
            value=read(paths[role])
            if value.get('scientific_commit')!=SCIENCE or value.get('group')!='S1-G16':raise ValueError('frozen plan source differs')
        if sha(paths['run_plan'])!=summary['plan_sha256'] or sha(paths['snapshot'])!=summary['state_sha256']:raise ValueError('strict report snapshot/plan pin differs')
        if spec.get('report11_status')!='final' or paths['report11'].suffix!='.md' or len(paths['report11'].read_text(encoding='utf-8-sig').strip())<100:raise ValueError('formal report11 missing/empty/not final')
        if not paths['reproduction'].read_text(encoding='utf-8-sig').strip():raise ValueError('reproduction commands absent')
        source_receipts=read(paths['source_receipts'])
        if source_receipts.get('status')!='accepted' or not source_receipts.get('nodes'):
            raise ValueError('source receipt index not accepted/exact')
        canonical={r['id']:r for r in closure.get('nodes',[])}
        for node in source_receipts['nodes']:
            if not node.get('path') and not (node.get('archive_path') and node.get('member')):raise ValueError('source receipt lacks canonical archive/file entry point')
            if node['id'] in canonical and canonical[node['id']]['sha256']!=node['sha256']:raise ValueError('conflicting source receipt canonical identity')
            canonical[node['id']]=node
        sources=read(paths['source_index']).get('sources',[])
        byrole={r['role']:r for r in sources}
        if not {'scientific','controller','postprocessing'}.issubset(byrole):raise ValueError('source index lacks science/controller/postops')
        if byrole['scientific']['commit']!=SCIENCE or not byrole['controller']['commit'].startswith('b520'):raise ValueError('scientific/controller index source differs')
        for value in sources:
            if not re.fullmatch('[0-9a-f]{40}',value['commit']) or not value.get('files'):raise ValueError('source index not exact')
            for item in value['files']:
                if not item.get('path') or not item.get('canonical_ref') or not re.fullmatch('[0-9a-f]{64}',item.get('sha256','')):raise ValueError('source member not indexed precisely')
                node=canonical.get(item['canonical_ref'])
                if node is None or node.get('sha256')!=item['sha256']:raise ValueError('source index not backed by accepted exact node')
        receipts=read(paths['task_receipts'])
        # This is a small exact copy of snapshot archives, not a second archive audit.
        if receipts!=snapshot.get('archives') or set(receipts)!=set(snapshot.get('tasks',{})):raise ValueError('task receipt index differs from snapshot')
        pub=read(paths['publication_manifest'])
        if pub.get('synthetic') is not False or pub.get('formal_ready') is not True or pub.get('source_summary_sha256')!=sha(paths['summary']):raise ValueError('publication is not formal/from strict summary')
        artifacts=pub.get('artifacts',{})
        expected={f'{prefix}.{suffix}' for prefix in ('01_same_N','02_input_profiles','03_variable_N','04_learning','05_costs') for suffix in ('png','pdf','csv')}
        if set(artifacts)!=expected:raise ValueError('publication artifact matrix incomplete')
        figure_root=paths['publication_manifest'].parent
        for name,digest in artifacts.items():
            rows.append(dict(role='figure_'+name,path=str(figure_root/safe(name)),sha256=digest,member='figures/'+name))
        # Figure tables are exact referenced compact inputs; add them once.
        for name,digest in pub.get('sources',{}).items():
            target=paths['summary'].parent/safe(name)
            rows.append(dict(role='table_'+name,path=str(target),sha256=digest,member='strict_tables/'+name))
        if summary.get('gate_passed') is True:
            needed={'model_final','model_provenance','gpu_receipt','export_receipt'}
            if not needed.issubset(roles):raise ValueError('MLM model export/GPU evidence absent')
            gpu=read(paths['gpu_receipt']);export=read(paths['export_receipt'])
            mappings=spec.get('model_receipt_fields',{})
            keys={'verified','final_sha256','checkpoint_sha256','export_verified','export_sha256'}
            if set(mappings)!=keys:raise ValueError('explicit actual model receipt field mapping required')
            if field(gpu,mappings['verified']) is not True or field(export,mappings['export_verified']) is not True:raise ValueError('GPU/export unverified')
            if field(gpu,mappings['final_sha256'])!=roles['model_final']['sha256'] or field(export,mappings['export_sha256'])!=roles['model_final']['sha256']:raise ValueError('GPU/export final SHA differs')
            if not re.fullmatch('[0-9a-f]{64}',field(gpu,mappings['checkpoint_sha256'])):raise ValueError('checkpoint SHA absent')
            provenance=read(paths['model_provenance'])
            # Explicit small provenance adapter pins the exported endpoint back to
            # the canonical train task; actual raw receipt remains packaged too.
            if provenance.get('scientific_commit')!=SCIENCE or provenance.get('final_sha256')!=roles['model_final']['sha256'] or provenance.get('checkpoint_sha256')!=field(gpu,mappings['checkpoint_sha256']):raise ValueError('model provenance endpoint/checkpoint/source differs')
            owner=snapshot.get('tasks',{}).get('train_mlm',{})
            if provenance.get('run_id')!=owner.get('run_id') or provenance.get('port')!=owner.get('port') or not provenance.get('canonical_final_member'):raise ValueError('model provenance canonical task owner absent')
            node=canonical.get(provenance['canonical_final_member'])
            if node is None or node.get('sha256')!=roles['model_final']['sha256'] or node.get('owner')!='train_mlm':raise ValueError('small model not exact accepted canonical endpoint')
        elif summary.get('gate_passed') is not False:raise ValueError('teacher gate unknown')
        elif any(role.startswith('model_') or role in ('gpu_receipt','export_receipt') for role in roles):raise ValueError('gate-fail has unexpected MLM model')
        members={}
        total=0
        for row in rows:
            member=safe(row['member']);path=pinned(row)
            if member in members or member=='delivery_manifest.json':raise ValueError('duplicate/reserved member')
            total+=path.stat().st_size
            members[member]=dict(path=str(path),sha256=row['sha256'],bytes=path.stat().st_size,role=row['role'])
        if total>MAX_TOTAL or len(members)>1500:raise ValueError('small delivery bounded size exceeded')
        return dict(status='ready',group='S1-G16',package_id=ident,spec_path=str(Path(spec_path).resolve()),
            spec_sha256=sha(spec_path),members=members,total_bytes=total,issues=[],
            standalone_full_data=False,canonical_dependencies=list(canonical.values()))
    except Exception as error:
        return dict(status='not_ready',group='S1-G16',issues=[str(error)])


def package(spec_path,destination):
    value=plan(spec_path)
    if value['status']!='ready':raise ValueError('; '.join(value['issues']))
    destination=Path(destination)
    if destination.exists():raise FileExistsError('use resume-upload with existing receipt, never repackage')
    destination.mkdir(parents=True)
    index={name:{k:v for k,v in item.items() if k!='path'} for name,item in value['members'].items()}
    manifest=dict(schema='g16_small_delivery_v1',group='S1-G16',package_id=value['package_id'],
        spec_sha256=value['spec_sha256'],members=index,standalone_full_data=False,
        dependency_note='Canonical ZIP/tar volumes remain external; this small package is not independently complete data.',
        canonical_dependencies=value['canonical_dependencies'])
    target=destination/'delivery.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,item in value['members'].items():
            # Fail if a small file changed since read-only planning.
            if sha(item['path'])!=item['sha256']:raise ValueError('small input changed while packaging')
            archive.write(item['path'],name)
        archive.writestr('delivery_manifest.json',encoded(manifest))
    verify_zip(target,manifest)
    receipt=dict(group='S1-G16',schema='g16_delivery_receipt_v1',package_id=value['package_id'],
        zip_path=str(target.resolve()),zip_sha256=sha(target),zip_bytes=target.stat().st_size,
        manifest=manifest,status='packaged_not_uploaded',remote_verified={},
        created_utc=datetime.now(timezone.utc).isoformat())
    save_receipt(destination/'receipt.json',receipt)
    return receipt


def verify_zip(path,manifest):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!={*manifest['members'],'delivery_manifest.json'}:raise ValueError('ZIP member set differs')
        if archive.read('delivery_manifest.json')!=encoded(manifest):raise ValueError('ZIP manifest differs')
        for name,row in manifest['members'].items():
            data=archive.read(name)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('ZIP member SHA differs')


def save_receipt(path,receipt):
    path=Path(path);temporary=path.with_suffix('.json.partial')
    temporary.write_bytes(encoded(receipt));temporary.replace(path)


def resume_upload(receipt_path, *, transport=None):
    """Upload identical existing bytes, then read through both authorized endpoints."""
    if transport is None:
        from . import ops as transport
    receipt=read(receipt_path)
    if receipt.get('schema')!='g16_delivery_receipt_v1' or receipt.get('group')!='S1-G16':raise ValueError('wrong resume receipt')
    path=Path(receipt['zip_path'])
    if path.stat().st_size!=receipt['zip_bytes'] or sha(path)!=receipt['zip_sha256']:raise ValueError('resume package bytes changed')
    verify_zip(path,receipt['manifest'])
    ident=receipt['package_id']
    if not re.fullmatch('[a-zA-Z0-9_-]{1,80}',ident):raise ValueError('unsafe package identity')
    remote=transport.BASE+'/delivery/'+ident+'/delivery.zip'
    probe=f"import pathlib,hashlib,json;p=pathlib.Path({remote!r});print(json.dumps({{'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}}))"
    result=transport.remote_python(probe,20424,timeout=60)
    if result['exit_code']:raise RuntimeError('shared ZIP probe failed')
    existing=json.loads(result['stdout'].splitlines()[-1])
    if existing['exists'] and existing['sha256']!=receipt['zip_sha256']:raise ValueError('remote immutable delivery differs; refusing overwrite')
    if not existing['exists']:transport.upload(path,remote,20424)
    expected=receipt['manifest']
    code=f'''import pathlib,hashlib,json,zipfile
p=pathlib.Path({remote!r}); expected=json.loads({encoded(expected).decode()!r})
data=p.read_bytes()
if len(data)!={receipt['zip_bytes']} or hashlib.sha256(data).hexdigest()!={receipt['zip_sha256']!r}:raise ValueError('remote ZIP differs')
with zipfile.ZipFile(p) as z:
 names=z.namelist()
 if len(names)!=len(set(names)) or set(names)!=set(expected['members'])|{{'delivery_manifest.json'}}:raise ValueError('remote member set differs')
 if json.loads(z.read('delivery_manifest.json'))!=expected:raise ValueError('remote manifest differs')
 for name,row in expected['members'].items():
  b=z.read(name)
  if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('remote member SHA differs')
print(json.dumps({{'verified':True,'path':str(p),'zip_sha256':hashlib.sha256(data).hexdigest(),'members':len(expected['members'])}}))
'''
    for port in (20424,20473):
        result=transport.remote_python(code,port,timeout=60)
        if result['exit_code']:raise RuntimeError(str(port)+' delivery readback failed: '+result['stderr'][-300:])
        value=json.loads(result['stdout'].splitlines()[-1])
        if value.get('verified') is not True or value.get('zip_sha256')!=receipt['zip_sha256'] or value.get('members')!=len(expected['members']):raise ValueError('remote readback receipt differs')
        receipt['remote_verified'][str(port)]=value
        receipt['remote_path']=remote
        save_receipt(receipt_path,receipt)
    receipt['status']='dual_endpoint_verified';save_receipt(receipt_path,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','deliver','resume-upload'))
    parser.add_argument('--spec');parser.add_argument('--output');parser.add_argument('--receipt')
    args=parser.parse_args()
    if args.action=='plan':
        if not args.spec:parser.error('--spec required')
        value=plan(args.spec);print(json.dumps(value,ensure_ascii=False));return 0 if value['status']=='ready' else 2
    if args.action=='deliver':
        if not args.spec or not args.output:parser.error('--spec and --output required')
        package(args.spec,args.output);resume_upload(Path(args.output)/'receipt.json')
    else:
        if not args.receipt:parser.error('--receipt required')
        resume_upload(args.receipt)
    return 0


if __name__=='__main__':raise SystemExit(main())
