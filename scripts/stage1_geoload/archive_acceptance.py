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
Local postprocessing dependency: requirements-postops.txt (ijson==3.3.0).
Use local DQN; installing this dependency never changes remote scientific envs.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import time
import zipfile
from urllib.parse import urlsplit

CHUNK = 1024 * 1024
JSON_LIMIT = 16 * CHUNK
IJSON_VERSION = '3.3.0'


def stream_events(stream):
    """Fixed mature parser, lazy import: numeric/layout arrays never materialize."""
    import ijson
    if ijson.__version__ != IJSON_VERSION:
        raise RuntimeError('postops requires fixed ijson==' + IJSON_VERSION)
    prefix=stream.read(3)
    class PrefixedReader:
        def __init__(self):self.prefix=b'' if prefix==b'\xef\xbb\xbf' else prefix
        def read(self,size):
            if size<0:raise ValueError('unbounded JSON read refused')
            head=self.prefix[:size];self.prefix=self.prefix[len(head):]
            return head+stream.read(size-len(head))
        def readinto(self,buffer):
            data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)
    try:yield from ijson.parse(PrefixedReader(), use_float=True, buf_size=64 * 1024)
    except ijson.JSONError as error:raise ValueError('invalid streaming JSON: '+str(error)) from error


def parser_metadata():
    import ijson
    if ijson.__version__ != IJSON_VERSION:raise RuntimeError('postops ijson version differs')
    return {'library':'ijson','version':ijson.__version__,'backend':ijson.backend,
            'buffer_bytes':64*1024,'document_retention':False,'all_json_members_scanned':True}


def stream_pinned_refs(stream, source=None, summary=None):
    """Emit pins at map end; retain only current map scalar strings and nesting.

    No case/layout/journal arrays or completed documents survive. JSON keys are
    unique in frozen worker output; ambiguous duplicates are rejected explicitly.
    The location syntax matches pinned_refs, including array indices.
    """
    stack=[];summary=summary if summary is not None else {}
    def location():
        if not stack:return '$'
        parent=stack[-1]
        if parent['kind']=='map':return parent['location']+'.'+parent['key']
        result=parent['location']+'['+str(parent['index'])+']';parent['index']+=1;return result
    for _,event,value in stream_events(stream):
        if event=='map_key':
            frame=stack[-1]
            if value in frame['seen']:raise ValueError('duplicate JSON key at '+frame['location']+'.'+value)
            frame['seen'].add(value);frame['key']=value
        elif event in ('start_map','start_array'):
            at=location()
            stack.append({'kind':'map','location':at,'key':None,'strings':{},'seen':set()} if event=='start_map' else {'kind':'array','location':at,'index':0})
        elif event=='end_map':
            frame=stack.pop();strings=frame['strings'];at=frame['location']
            for key,target,sha in pinned_refs(strings,at):yield key,target,sha
            if at=='$':summary['git_commit']=strings.get('git_commit')
            if at=='$.files' and source and PurePosixPath(source).name=='source_manifest.json':
                if frame['seen']!=set(strings):raise ValueError('nonstring source file SHA')
                summary['source_files_count']=len(strings)
                for name,sha in strings.items():
                    if not safe_member(name):raise ValueError('unsafe_source_member: '+name)
                    yield '$.files.'+name,str(PurePosixPath(source).parent/name),sha
            if at.startswith('$.samples[') and at.endswith(']') and at.count('[')==1 and source and {'path','sha256'}<=frame['seen']:
                if not strings.get('path') or not strings.get('sha256'):raise ValueError('teacher sample path/SHA type invalid')
                if not safe_member(strings['path']):raise ValueError('unsafe_teacher_sample: '+strings['path'])
                yield '$.samples',str(PurePosixPath(source).parent/strings['path']),strings['sha256']
        elif event=='end_array':stack.pop()
        elif event in ('string','number','boolean','null'):
            if stack and stack[-1]['kind']=='array':stack[-1]['index']+=1
            elif event=='string' and stack:stack[-1]['strings'][stack[-1]['key']]=value
    if stack:raise ValueError('unterminated JSON container')


def absolute_reference(target):
    return PurePosixPath(target).is_absolute() or PureWindowsPath(target).is_absolute() or urlsplit(target).scheme in ('http','https')


class HashingReader:
    """Hash the exact bytes consumed by event parsing, without a second pass."""
    def __init__(self,stream,on_bytes=None):self.stream=stream;self.hash=hashlib.sha256();self.count=0;self.on_bytes=on_bytes
    def read(self,size=-1):
        if size<0:raise ValueError('unbounded JSON read refused')
        data=self.stream.read(size);self.hash.update(data);self.count+=len(data)
        if self.on_bytes and data:self.on_bytes(len(data))
        return data
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)


def sha_stream(stream,on_bytes=None):
    digest = hashlib.sha256(); count = 0
    for block in iter(lambda: stream.read(CHUNK), b''):
        digest.update(block); count += len(block)
        if on_bytes:on_bytes(len(block))
    return digest.hexdigest(), count


def file_sha(path,on_bytes=None):
    with Path(path).open('rb') as stream:
        return sha_stream(stream,on_bytes)[0]


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


def audit(plan, state, registry=None, *, progress=None):
    registry = registry or {}; issues = []; nodes = []; edges = []; pins = []
    aliases = {}; by_sha = {}; jobs = []
    commit = plan.get('scientific_commit')
    started=time.monotonic();last_progress=started-10
    observation={'phase':'starting','current_archive':None,'current_member':None,
                 'bytes_read':0,'json_bytes_read':0,'json_documents_scanned':0,'pins_discovered':0}

    def observe(phase=None,archive=None,member=None,amount=0,json_bytes=False,force=False):
        nonlocal last_progress
        if progress is None:return
        if phase is not None:observation['phase']=phase
        if archive is not None:observation['current_archive']=str(archive)
        if member is not None:observation['current_member']=member
        observation['bytes_read']+=amount
        if json_bytes:observation['json_bytes_read']+=amount
        observation['pins_discovered']=len(pins)
        current=time.monotonic()
        if force or current-last_progress>=5:
            progress({**observation,'elapsed_seconds':current-started,
                      'byte_scope':'all volume/member reads, including complete SHA verification',
                      'nodes_verified':len(nodes),'issues':len(issues)})
            last_progress=current
    observe(force=True)

    def issue(code, **detail): issues.append({'code': code, **detail})

    def add(node, source_paths=()):
        nodes.append(node); by_sha.setdefault(node['sha256'], []).append(node['id'])
        for source in source_paths:
            previous = aliases.get(source)
            if previous and previous['sha256'] != node['sha256']:
                issue('conflicting_source_owner', source_path=source)
            else: aliases.setdefault(source, node)

    def consume_json(stream,owner,source):
        summary={}
        try:
            for location,target,sha in stream_pinned_refs(stream,source,summary):
                if not absolute_reference(target) and source:
                    if not safe_member(target):issue('unsafe_relative_reference',owner=owner,source_path=target);continue
                    target=str(PurePosixPath(source).parent/target)
                pins.append((owner,location,target,sha))
        except (ValueError,UnicodeError) as error:
            issue('invalid_json',node=owner,reason=str(error))
            # A malformed/unsafe member cannot pass, but still finishes full byte
            # hashing, including parser read-ahead already consumed by the reader.
            for _ in iter(lambda:stream.read(CHUNK),b''):pass
        if source and PurePosixPath(source).name=='source_manifest.json':
            if summary.get('git_commit')!=commit:issue('source_manifest_commit_differs',owner=owner)
            if not summary.get('source_files_count'):issue('source_manifest_empty',owner=owner)
        observation['json_documents_scanned']+=1

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
            observe('archive_volume_sha',path,'')
            if not path.is_file() or file_sha(path,lambda n:observe(amount=n)) != receipt.get('sha256'):
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
                    observe('archive_member',path,name)
                    source = (jobroot + '/' + name[4:] if name.startswith('job/') else
                              job['model_dir'].rstrip('/') + '/' + name[6:] if name.startswith('model/') else None)
                    with z.open(name) as stream:
                        if name.endswith('.json') and name!='archive_manifest.json':
                            reader=HashingReader(stream,lambda n:observe(amount=n,json_bytes=True))
                            consume_json(reader,key+':'+name,source)
                            sha,count=reader.hash.hexdigest(),reader.count
                        else:sha,count=sha_stream(stream,lambda n:observe(amount=n))
                    if name != 'archive_manifest.json':
                        expected = manifest['members'][name]
                        if expected['sha256'] != sha or expected['bytes'] != count:
                            raise ValueError('member SHA/bytes differ: ' + name)
                    node = {'id': key + ':' + name, 'owner': key, 'archive_path': str(path),
                            'member': name, 'sha256': sha, 'bytes': count}
                    add(node, [source] if source else [])
                add({'id': key + ':ZIP', 'path': str(path), 'sha256': receipt['sha256'], 'bytes':path.stat().st_size})
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            issue('archive_invalid', task=key, reason=str(error))
    bundles = []
    for item in registry.get('resources', []):
        try:
            path = Path(item['path']);observe('external_volume_sha',path,'');sha = file_sha(path,lambda n:observe(amount=n))
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
                        observe('external_member',path,name)
                        sources=item.get('member_source_paths',{}).get(name,[])
                        with z.open(name) as stream:
                            if name in item.get('json_members',[]) and sources:
                                reader=HashingReader(stream,lambda n:observe(amount=n,json_bytes=True));consume_json(reader,item['id']+':'+name,sources[0])
                                member_sha,count=reader.hash.hexdigest(),reader.count
                            else:member_sha,count=sha_stream(stream,lambda n:observe(amount=n))
                        expected=item['members'][name]
                        if member_sha!=expected['sha256'] or count!=expected['bytes']:
                            raise ValueError('external member SHA/bytes differ: '+name)
                        node={'id':item['id']+':'+name,'owner':item['id'],'archive_path':str(path),
                              'member':name,'sha256':member_sha,'bytes':count}
                        add(node,sources)
                        if name in item.get('json_members',[]):
                            for source in sources[1:]:
                                with z.open(name) as stream:consume_json(HashingReader(stream,lambda n:observe(amount=n,json_bytes=True)),node['id'],source)
            if item.get('parse_json', True) and path.suffix == '.json':
                for source in item.get('source_paths', []):
                    observe('external_json',path,'')
                    with path.open('rb') as stream:consume_json(HashingReader(stream,lambda n:observe(amount=n,json_bytes=True)),item['id'],source)
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
    observe('complete',force=True)
    result = {'group':'S1-G16', 'schema_version':1, 'status':'not_ready' if issues else 'accepted',
              'scientific_commit':commit, 'issues':issues, 'nodes':nodes, 'edges':edges,
              'duplicates':[{'sha256':sha,'canonical_owner':ids[0],'references':ids[1:]} for sha,ids in by_sha.items() if len(ids)>1],
              'model_validation':{'status':'not_ready','reason':'archive integrity does not perform strict model load'},
              'json_parser':parser_metadata(),
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
