"""Read-only G16 closure audit; stream ZIP members, never extract or export.

External registry schema: {resources:[{id,path,sha256,source_paths:[],
kind:'file'|'git_bundle'|'zip',commit:...,members:{name:{sha256,bytes}},
member_source_paths:{name:[original_path]},json_members:[]}], requirements:[{source_path,sha256}],
member_aliases:[{source_path,sourceRef,sha256}], upstream_sources:[{role,
raw_archives:[{path,sha256}],parser:[...],registry:[...],inputs:[...]}]}.
Every original source path resolves to a verified canonical owner/member. Shared
files can have several source_paths; SHA equality alone never invents a path alias.
Source snapshots must be exported separately with their actual source_manifest
and all files, and registered file by file (the manifest is not the source).
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

CHUNK = 1024 * 1024
JSON_LIMIT = 16 * CHUNK


def sha_stream(stream):
    digest = hashlib.sha256(); count = 0
    for block in iter(lambda: stream.read(CHUNK), b''):
        digest.update(block); count += len(block)
    return digest.hexdigest(), count


def file_sha(path):
    with Path(path).open('rb') as stream:
        return sha_stream(stream)[0]


def safe_member(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name


def pinned_refs(value, location='$'):
    """Discover explicit file pins, including archive_path/archive_sha256 pairs."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield location, value['path'], value['sha256']
            if isinstance(value.get('source_path'), str):
                yield location+'.source_path', value['source_path'], value['sha256']
        for key, item in value.items():
            if key.endswith('_path') and isinstance(item, str):
                sha = value.get(key[:-5] + '_sha256')
                if isinstance(sha, str): yield location + '.' + key, item, sha
            elif key not in ('path','source_path') and isinstance(item,str) and isinstance(value.get(key+'_sha256'),str):
                yield location+'.'+key,item,value[key+'_sha256']
            yield from pinned_refs(item, location + '.' + key)
    elif isinstance(value, list):
        for i, item in enumerate(value): yield from pinned_refs(item, location + '[' + str(i) + ']')


def audit(plan, state, registry=None):
    registry = registry or {}; issues = []; nodes = []; edges = []; pins = []
    aliases = {}; by_sha = {}; documents = []; jobs = []
    commit = plan.get('scientific_commit')

    def issue(code, **detail): issues.append({'code': code, **detail})

    def add(node, source_paths=()):
        nodes.append(node); by_sha.setdefault(node['sha256'], []).append(node['id'])
        for source in source_paths:
            previous = aliases.get(source)
            if previous and previous['sha256'] != node['sha256']:
                issue('conflicting_source_owner', source_path=source)
            else: aliases.setdefault(source, node)

    if plan.get('group') != 'S1-G16' or state.get('group') != 'S1-G16': issue('wrong_group')
    if not state.get('all_computation_completed') or not state.get('compute_and_archives_complete'):
        issue('computation_or_export_incomplete')
    tasks = state.get('tasks', {}); archives = state.get('archives', {})
    if not tasks: issue('no_tasks')
    if set(tasks) != set(archives): issue('task_archive_set_differs', missing=sorted(set(tasks)-set(archives)))
    for key, task in tasks.items():
        receipt = archives.get(key)
        if task.get('stage') != 'completed': issue('noncompleted_task', task=key)
        if not receipt: continue
        path = Path(receipt.get('local_path', ''))
        try:
            if not path.is_file() or file_sha(path) != receipt.get('sha256'):
                raise ValueError('ZIP receipt SHA differs or file absent')
            with zipfile.ZipFile(path) as z:
                names = [i.filename for i in z.infolist() if not i.is_dir()]
                if len(names) != len(set(names)) or any(not safe_member(n) for n in names):
                    raise ValueError('duplicate or unsafe member')
                manifest = json.loads(z.read('archive_manifest.json'))
                if manifest.get('run_id') != task['run_id'] or manifest.get('scientific_commit') != commit:
                    raise ValueError('manifest task/commit differs')
                if set(names) != set(manifest['members']) | {'archive_manifest.json'}:
                    raise ValueError('member inventory differs')
                job = json.loads(z.read('job/job.json')); jobs.append((key, job))
                if job.get('experiment_id') != 'S1-G16' or job.get('run_id') != task['run_id'] or job.get('scientific_commit') != commit:
                    raise ValueError('job identity differs')
                jobroot = job['archive_root'].rstrip('/') + '/outputs/workbench/jobs/' + job['run_id']
                status = json.loads(z.read('job/status.json'))
                if status.get('stage') != 'completed': issue('archived_status_noncompleted', task=key)
                for name in names:
                    with z.open(name) as stream: sha, count = sha_stream(stream)
                    if name != 'archive_manifest.json':
                        expected = manifest['members'][name]
                        if expected['sha256'] != sha or expected['bytes'] != count:
                            raise ValueError('member SHA/bytes differ: ' + name)
                    source = (jobroot + '/' + name[4:] if name.startswith('job/') else
                              job['model_dir'].rstrip('/') + '/' + name[6:] if name.startswith('model/') else None)
                    node = {'id': key + ':' + name, 'owner': key, 'archive_path': str(path),
                            'member': name, 'sha256': sha, 'bytes': count}
                    add(node, [source] if source else [])
                    if name.endswith('.json') and count <= JSON_LIMIT and name != 'archive_manifest.json':
                        try: documents.append((node['id'], json.loads(z.read(name)), source))
                        except (ValueError, UnicodeError): issue('invalid_json', node=node['id'])
                add({'id': key + ':ZIP', 'path': str(path), 'sha256': receipt['sha256'], 'bytes':path.stat().st_size})
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            issue('archive_invalid', task=key, reason=str(error))
    bundles = []
    for item in registry.get('resources', []):
        try:
            path = Path(item['path']); sha = file_sha(path)
            if sha != item['sha256']: raise ValueError('external SHA differs')
            add({'id':item['id'], 'path':str(path), 'sha256':sha, 'bytes':path.stat().st_size}, item.get('source_paths', []))
            if item.get('kind') == 'git_bundle':
                output = subprocess.run(['git','bundle','list-heads',str(path)], check=True, capture_output=True, text=True).stdout
                if item.get('commit') != commit or commit not in [line.split()[0] for line in output.splitlines()]:
                    raise ValueError('bundle lacks scientific commit')
                bundles.append(item['id'])
            if item.get('kind') == 'zip':
                with zipfile.ZipFile(path) as z:
                    names=[i.filename for i in z.infolist() if not i.is_dir()]
                    if len(names)!=len(set(names)) or set(names)!=set(item.get('members',{})) or any(not safe_member(n) for n in names):
                        raise ValueError('external ZIP member inventory differs')
                    for name in names:
                        with z.open(name) as stream: member_sha,count=sha_stream(stream)
                        expected=item['members'][name]
                        if member_sha!=expected['sha256'] or count!=expected['bytes']:
                            raise ValueError('external member SHA/bytes differ: '+name)
                        node={'id':item['id']+':'+name,'owner':item['id'],'archive_path':str(path),
                              'member':name,'sha256':member_sha,'bytes':count}
                        sources=item.get('member_source_paths',{}).get(name,[]); add(node,sources)
                        if name in item.get('json_members',[]) and count<=JSON_LIMIT:
                            value=json.loads(z.read(name))
                            for source in sources: documents.append((node['id'],value,source))
            if item.get('parse_json', True) and path.suffix == '.json' and path.stat().st_size <= JSON_LIMIT:
                value = json.loads(path.read_text(encoding='utf-8-sig'))
                for source in item.get('source_paths', []): documents.append((item['id'], value, source))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            issue('external_invalid', resource=item.get('id'), reason=str(error))
    if not bundles: issue('scientific_git_bundle_missing')
    node_by_id={n['id']:n for n in nodes}
    for item in registry.get('member_aliases', []):
        node=node_by_id.get(item.get('sourceRef'))
        if not node or node['sha256'] != item.get('sha256'):
            issue('member_alias_invalid', source_path=item.get('source_path'))
        elif item['source_path'] in aliases and aliases[item['source_path']]['sha256'] != node['sha256']:
            issue('conflicting_source_owner', source_path=item['source_path'])
        else: aliases[item['source_path']]=node
    for owner, value, source in documents:
        for location, target, sha in pinned_refs(value):
            if not PurePosixPath(target).is_absolute() and source:
                if not safe_member(target): issue('unsafe_relative_reference', owner=owner, source_path=target); continue
                target = str(PurePosixPath(source).parent / target)
            pins.append((owner, location, target, sha))
        if isinstance(value, dict) and 'samples' in value:
            for sample in value['samples']:
                if sample.get('path') and sample.get('sha256') and source:
                    if not safe_member(sample['path']): issue('unsafe_teacher_sample', owner=owner)
                    else: pins.append((owner, '$.samples', str(PurePosixPath(source).parent / sample['path']), sample['sha256']))
        if isinstance(value, dict) and source and PurePosixPath(source).name == 'source_manifest.json':
            if value.get('git_commit') != commit: issue('source_manifest_commit_differs', owner=owner)
            if not value.get('files'): issue('source_manifest_empty', owner=owner)
            for name, sha in value.get('files', {}).items():
                if not safe_member(name): issue('unsafe_source_member', owner=owner)
                else: pins.append((owner, '$.files.'+name, str(PurePosixPath(source).parent/name), sha))
    for key, job in jobs:
        root = job.get('source_root', '').rstrip('/')
        if root + '/source_manifest.json' not in aliases: issue('source_snapshot_missing', task=key, source_path=root)
    # Mandatory upstream registry/original source requirements are supplied by the
    # explicit source registry; omitting that registry cannot produce acceptance.
    upstream = registry.get('upstream_sources', [])
    if {x.get('role') for x in upstream} != {'train','development'}:
        issue('upstream_raw_closure_not_verified')
    for item in upstream:
        for category in ('raw_archives','parser','registry','inputs'):
            records=item.get(category, [])
            if not records: issue('upstream_category_missing', role=item.get('role'), category=category)
            for pin in records:
                pins.append(('upstream:'+str(item.get('role')), category, pin['path'], pin['sha256']))
    for item in registry.get('requirements', []):
        pins.append(('registry', '$.requirements', item['source_path'], item['sha256']))
    for owner, location, target, sha in pins:
        found = aliases.get(target)
        edge = {'from':owner, 'location':location, 'source_path':target, 'expected_sha256':sha,
                'sourceRef':found['id'] if found else None}
        edges.append(edge)
        if not found: issue('dependency_missing', **edge)
        elif found['sha256'] != sha: issue('dependency_sha_differs', **edge)
    result = {'group':'S1-G16', 'schema_version':1, 'status':'not_ready' if issues else 'accepted',
              'scientific_commit':commit, 'issues':issues, 'nodes':nodes, 'edges':edges,
              'duplicates':[{'sha256':sha,'canonical_owner':ids[0],'references':ids[1:]} for sha,ids in by_sha.items() if len(ids)>1],
              'model_validation':{'status':'not_ready','reason':'archive integrity does not perform strict model load'},
              'module_sha256':file_sha(__file__)}
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',required=True); parser.add_argument('--state',required=True)
    parser.add_argument('--registry'); parser.add_argument('--output',required=True); args=parser.parse_args()
    def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    result=audit(read(args.plan),read(args.state),read(args.registry) if args.registry else None)
    output=Path(args.output)
    # Caller chooses an existing output directory; no canonical data are rewritten.
    if output.exists(): raise FileExistsError('use a fresh acceptance receipt path')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'issues':len(result['issues']),'output':str(output)}))


if __name__ == '__main__': main()
