"""Strict G16 producer-aware resolution of an already fully hashed closure cache.

Never hashes/unpacks the scientific data again. Pins the whole original cache,
all frozen inputs and every task's original manifest/receipt. All original errors
must be explained exactly; downstream JobZip must still reverify original bytes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile
from .archive_acceptance import file_sha, parser_metadata

SCIENCE='495bff976c59c40ddb33de6b5b37b20a894a1347'
CACHE_MODULE_SHA='f760dda0bab06930abcc94f102eab632cbccf715c3373700af377874d6a36b19'
PRODUCER_SHAS={
 'src/satellite_rl/workbench/geoload_runner.py':'edb1d51a9fcf7691e1f1aa71d0d6fce5fe048957226a24f997a3e9491efad909',
 'src/satellite_rl/coverage/geoload/teacher.py':'beab21f2d225adf0f8eab1860ee41531c0da5fc8359d51a2357ad6b191d7f967'}
CASE=re.compile(r'^(teacher_20424|teacher_20473):job/cases/([0-9a-f]{64})\.json$')
INDEX=re.compile(r'^\$\.samples\[(\d+)\]$')
EXPECTED_COUNTS={'teacher_relative_base':64858,'gate_original_path_alias':4}


def require_sha(path,expected):
    if file_sha(path)!=expected:raise ValueError('immutable SHA binding differs: '+str(path))


def rows(path,prefix):
    import ijson
    parser_metadata()
    with Path(path).open('rb') as stream:yield from ijson.items(stream,prefix+'.item')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def signature(edge):return tuple(edge[k] for k in ('from','location','source_path','expected_sha256'))


def node_check(nodes,node_id,sha,owner=None,member=None):
    node=nodes.get(node_id)
    if node is None or node[0]!=sha or (owner is not None and node[1]!=owner) or (member is not None and node[2]!=member):
        raise ValueError('verified canonical node binding differs: '+str(node_id))
    return node


def resolve_teacher(issue,jobs,nodes,shards,manifest_samples,case_indexes):
    hit=CASE.fullmatch(issue['from'])
    if not hit:raise ValueError('unknown teacher-case owner')
    owner,casehash=hit.groups();job=jobs[owner]
    if job.get('mode')!='teacher' or job.get('kind')!='prepare' or job.get('methods')!=['JOINT']:
        raise ValueError('job is not the frozen teacher producer')
    index=case_indexes[owner].get(casehash)
    if not index or index.get('method')!='JOINT' or index.get('outcome')!='completed' or hashlib.sha256((index['case_id']+'\0JOINT').encode()).hexdigest()!=casehash:
        raise ValueError('same-case hash/index identity differs')
    case_member='job/cases/'+casehash+'.json';node_check(nodes,issue['from'],index['sha256'],owner,case_member)
    root=job['archive_root'].rstrip('/')+'/outputs/workbench/jobs/'+job['run_id']
    shard=shards.get(issue['from']);manifest_member='job/teacher/'+casehash+'/manifest.json'
    expected_manifest=owner+':'+manifest_member
    if not shard or shard.get('sourceRef')!=expected_manifest or shard['source_path']!=root+'/teacher/'+casehash+'/manifest.json':
        raise ValueError('same-case teacher_shard absolute source binding differs')
    node_check(nodes,expected_manifest,shard['expected_sha256'],owner,manifest_member)
    bad_prefix=root+'/cases/'+casehash+'/'
    if not issue['source_path'].startswith(bad_prefix):raise ValueError('relative producer source is outside the same case')
    relative=issue['source_path'][len(bad_prefix):]
    if not re.fullmatch(r'sample_[0-9]{5}\.npz',relative):raise ValueError('teacher sample relative path is not a frozen producer file')
    correct=root+'/teacher/'+casehash+'/'+relative;member='job/teacher/'+casehash+'/'+relative;target=owner+':'+member
    node_check(nodes,target,issue['expected_sha256'],owner,member)
    hit_index=INDEX.fullmatch(issue['location'])
    if hit_index:
        anchor=manifest_samples.get((expected_manifest,issue['location']))
        reason='same manifest sample index'
    elif issue['location']=='$.samples':
        anchor=manifest_samples.get((expected_manifest,correct,issue['expected_sha256']))
        reason='exact same-manifest path and SHA duplicate pin'
    else:raise ValueError('unknown sample reference location')
    if not anchor or anchor.get('sourceRef')!=target or anchor['source_path']!=correct or anchor['expected_sha256']!=issue['expected_sha256']:
        raise ValueError('teacher manifest sample index/path/SHA does not establish the target')
    return target,{'kind':'teacher_relative_base','original_issue':issue,'resolved_source_path':correct,'sourceRef':target,
        'case_index_sourceRef':owner+':job/case_index.json','case_id':index['case_id'],
        'teacher_shard_sourceRef':expected_manifest,'teacher_shard_sha256':shard['expected_sha256'],
        'manifest_sample_location':anchor['location'],'sample_sha256':issue['expected_sha256'],'condition':reason}


def resolve_gate(issue,jobs,nodes,copied_gates):
    owner=issue['from'].split(':',1)[0]
    if issue['from']!=owner+':job/job.json' or issue['location']!='$.references.gate.source_path':raise ValueError('unknown gate reference')
    if owner not in ('teacher_20424','teacher_20473','teacher_merge','train_mlm'):raise ValueError('gate owner is outside the four frozen consumers')
    job=jobs[owner]
    gate=job.get('references',{}).get('gate',{});anchor=copied_gates.get(owner)
    root=job['archive_root'].rstrip('/')+'/outputs/workbench/jobs/'+job['run_id'];member='job/input_sources/gate/development_gate.json';target=owner+':'+member
    original=str(PurePosixPath(job['archive_root']).parent/'ops/development_gate.json')
    if gate.get('source_path')!=issue['source_path'] or gate['source_path']!=original or gate.get('sha256')!=issue['expected_sha256'] or gate.get('path')!=root+'/input_sources/gate/development_gate.json':
        raise ValueError('gate original/copy job provenance differs')
    if not anchor or anchor.get('sourceRef')!=target or anchor['source_path']!=gate['path'] or anchor['expected_sha256']!=issue['expected_sha256']:
        raise ValueError('gate copy is not already resolved by the full byte audit')
    node_check(nodes,target,issue['expected_sha256'],owner,member)
    return target,{'kind':'gate_original_path_alias','original_issue':issue,'resolved_source_path':gate['path'],
        'sourceRef':target,'job_reference_sha256':gate['sha256'],'condition':'same job explicit gate original source and retained copied pin'}


def prepare(cache,input_manifest,cache_sha,input_manifest_sha):
    require_sha(cache,cache_sha);require_sha(input_manifest,input_manifest_sha)
    manifest=read(input_manifest);inputs={}
    for key,item in manifest.items():
        path=Path(item['path']);require_sha(path,item['sha256'])
        if path.stat().st_size!=item['bytes']:raise ValueError('frozen input size differs')
        inputs[key]=read(path)
    plan,state,registry=inputs['plan'],inputs['state'],inputs['registry']
    if plan.get('group')!='S1-G16' or state.get('group')!='S1-G16' or registry.get('group')!='S1-G16' or plan.get('scientific_commit')!=SCIENCE or state.get('scientific_commit')!=SCIENCE or not state.get('compute_and_archives_complete') or not state.get('all_computation_completed'):
        raise ValueError('scientific terminal snapshot differs')
    if len(state.get('tasks',{}))!=28 or set(state['tasks'])!=set(state.get('archives',{})) or any(t['stage']!='completed' for t in state['tasks'].values()):
        raise ValueError('all original28 terminal tasks/archives required')
    if state.get('plan_sha256')!=manifest['plan']['sha256']:raise ValueError('original plan receipt differs')
    # Root scalar cache metadata only: never deserialize whole arrays.
    import ijson
    header={}
    with Path(cache).open('rb') as f:
        for prefix,event,value in ijson.parse(f,use_float=True):
            if prefix in ('group','status','scientific_commit','module_sha256') and event=='string':header[prefix]=value
    if header!={'group':'S1-G16','status':'not_ready','scientific_commit':SCIENCE,'module_sha256':CACHE_MODULE_SHA}:
        raise ValueError('cache is not the exact completed full-byte audit producer')
    nodes={};resource_ids={r['id'] for r in registry['resources']};resource_paths={}
    for node in rows(cache,'nodes'):
        identity=node['id']
        if identity in nodes:raise ValueError('duplicate cache node identity')
        nodes[identity]=(node['sha256'],sys.intern(node.get('owner','')),node.get('member',''),node['bytes'])
        if identity in resource_ids:resource_paths[identity]=node.get('path')
    for item in registry['resources']:
        node_check(nodes,item['id'],item['sha256'])
        if resource_paths.get(item['id'])!=item['path']:raise ValueError('cached external source canonical path differs')
        for name,pin in item.get('members',{}).items():
            node_check(nodes,item['id']+':'+name,pin['sha256'],item['id'],name)
    for rel,sha in PRODUCER_SHAS.items():node_check(nodes,'science_snapshot:snapshot/'+rel,sha,'science_snapshot','snapshot/'+rel)
    jobs={};case_indexes={};receipt_bindings={}
    for key,receipt in state['archives'].items():
        if receipt['run_id']!=state['tasks'][key]['run_id'] or receipt['port']!=state['tasks'][key]['port']:
            raise ValueError('receipt is not owned by the frozen task')
        node_check(nodes,key+':ZIP',receipt['sha256']);path=Path(receipt['local_path'])
        if receipt.get('verified') is not True or path.stat().st_size!=receipt['bytes'] or path.stat().st_mtime_ns>Path(cache).stat().st_mtime_ns:
            raise ValueError('archive receipt/size/immutable-cache chronology differs')
        receipt_bindings[key]={'run_id':receipt['run_id'],'sha256':receipt['sha256'],'manifest_sha256':receipt['manifest_sha256'],'bytes':receipt['bytes'],'path':str(path)}
        with zipfile.ZipFile(path) as z:
            raw=z.read('archive_manifest.json')
            if hashlib.sha256(raw).hexdigest()!=receipt['manifest_sha256']:raise ValueError('original archive manifest SHA changed')
            original=json.loads(raw)
            if original.get('run_id')!=receipt['run_id'] or original.get('scientific_commit')!=SCIENCE or original.get('group')!='S1-G16':raise ValueError('original archive/source identity differs')
            if set(z.namelist())!=set(original['members'])|{'archive_manifest.json'}:raise ValueError('original ZIP inventory differs')
            node_check(nodes,key+':archive_manifest.json',receipt['manifest_sha256'])
            for name,pin in original['members'].items():
                node=node_check(nodes,key+':'+name,pin['sha256'],key,name)
                if node[3]!=pin['bytes']:raise ValueError('cached member size differs from bound original manifest')
            if key in ('teacher_20424','teacher_20473','teacher_merge','train_mlm'):
                for name in ('job/job.json',*(['job/case_index.json'] if key.startswith('teacher_204') else [])):
                    payload=z.read(name);node_check(nodes,key+':'+name,hashlib.sha256(payload).hexdigest(),key,name)
                    value=json.loads(payload)
                    if name=='job/job.json':jobs[key]=value
                    else:case_indexes[key]=value
                if jobs[key].get('experiment_id')!='S1-G16' or jobs[key].get('scientific_commit')!=SCIENCE or jobs[key].get('run_id')!=state['tasks'][key]['run_id']:
                    raise ValueError('actual consumer job scientific identity differs')
                gate=jobs[key]['references']['gate'];planned_gate=state['gate_refs'][str(jobs[key]['ssh_port'])]
                if gate['source_path']!=planned_gate['path'] or gate['sha256']!=planned_gate['sha256']:
                    raise ValueError('actual gate original source differs from frozen controller pin')
    issues=list(rows(cache,'issues'))
    if not issues or any(i.get('code')!='dependency_missing' or i.get('sourceRef') is not None for i in issues):raise ValueError('cache contains errors outside declared missing-reference class')
    shards={};samples={};gates={};missing=Counter()
    for edge in rows(cache,'edges'):
        owner=edge['from']
        if edge.get('sourceRef') is None:missing[signature(edge)]+=1
        else:node_check(nodes,edge['sourceRef'],edge['expected_sha256'])
        if CASE.fullmatch(owner) and edge['location']=='$.teacher_shard':shards[owner]=edge
        if owner.startswith(('teacher_20424:job/teacher/','teacher_20473:job/teacher/')) and owner.endswith('/manifest.json') and INDEX.fullmatch(edge['location']):
            if edge.get('sourceRef') is None:raise ValueError('teacher manifest sample pin itself unresolved')
            node_check(nodes,edge['sourceRef'],edge['expected_sha256'])
            samples[(owner,edge['location'])]=edge;samples[(owner,edge['source_path'],edge['expected_sha256'])]=edge
        if owner in {k+':job/job.json' for k in jobs} and edge['location']=='$.references.gate':gates[owner.split(':',1)[0]]=edge
    if missing!=Counter(signature(i) for i in issues):raise ValueError('original missing-edge/error multisets differ')
    resolutions={};proofs=[];counts=Counter()
    for issue in issues:
        target,proof=resolve_teacher(issue,jobs,nodes,shards,samples,case_indexes) if CASE.fullmatch(issue['from']) else resolve_gate(issue,jobs,nodes,gates)
        sig=signature(issue)
        if sig in resolutions and resolutions[sig]!=target:raise ValueError('ambiguous resolution target')
        resolutions[sig]=target;proofs.append(proof);counts[proof['kind']]+=1
    if dict(counts)!=EXPECTED_COUNTS:raise ValueError('resolution classes are not the exact original64858 samples plus4 gates')
    return {'inputs':inputs,'input_manifest':manifest,'resolutions':resolutions,'proofs':proofs,'counts':dict(counts),
        'cache_sha256':cache_sha,'input_manifest_sha256':input_manifest_sha,'archive_bindings':receipt_bindings,
        'original_issue_count':len(issues),'original_node_count':len(nodes)}


def execute(cache,input_manifest,output,cache_sha,input_manifest_sha):
    cache=Path(cache);output=Path(output)
    if output.exists() or output.with_suffix('.partial').exists():raise FileExistsError('fresh resolved closure destination required')
    if output.resolve()==cache.resolve():raise ValueError('original closure cannot be rewritten')
    prepared=prepare(cache,input_manifest,cache_sha,input_manifest_sha)
    proof_path=output.with_suffix('.resolution_proof.jsonl');summary_path=output.with_suffix('.resolution_summary.json')
    with proof_path.open('x',encoding='utf-8') as f:
        for proof in prepared['proofs']:f.write(json.dumps(proof,ensure_ascii=False)+'\n')
    binding={'cache_path':str(cache),'cache_sha256':cache_sha,'input_manifest_path':str(input_manifest),
        'input_manifest_sha256':input_manifest_sha,'frozen_inputs':prepared['input_manifest'],
        'archive_receipts':prepared['archive_bindings'],'cache_module_sha256':CACHE_MODULE_SHA,'producer_source_shas':PRODUCER_SHAS,
        'resolution_module_sha256':file_sha(__file__),'proof_path':str(proof_path),'proof_sha256':file_sha(proof_path),
        'original_issue_count':prepared['original_issue_count'],'resolved_issue_count':len(prepared['proofs']),
        'resolution_classes':prepared['counts'],'downstream_original_JobZip_reverification_required':True,
        'scope':'cached full SHA audit plus exact producer graph resolution; scientific bytes unchanged'}
    def array(f,key,items):
        f.write(',\n'+json.dumps(key)+':[');first=True
        for item in items:
            if not first:f.write(',')
            f.write(json.dumps(item,ensure_ascii=False));first=False
        f.write(']')
    temporary=output.with_suffix('.partial');changed=0
    with temporary.open('x',encoding='utf-8') as f:
        f.write('{"group":"S1-G16","schema_version":2,"status":"accepted","scientific_commit":'+json.dumps(SCIENCE)+',"issues":[]')
        array(f,'nodes',rows(cache,'nodes'))
        def corrected():
            nonlocal changed
            for edge in rows(cache,'edges'):
                if edge.get('sourceRef') is None:
                    target=prepared['resolutions'].get(signature(edge))
                    if target is None:raise ValueError('unexplained edge remained before accepted output')
                    edge={**edge,'sourceRef':target,'producer_resolution':True};changed+=1
                yield edge
        array(f,'edges',corrected());array(f,'duplicates',rows(cache,'duplicates'))
        f.write(',"resolution_binding":'+json.dumps(binding,ensure_ascii=False))
        f.write(',"model_gpu_validation":'+json.dumps(prepared['inputs']['gpu'],ensure_ascii=False))
        f.write(',"module_sha256":'+json.dumps(file_sha(__file__))+',"json_parser":'+json.dumps(parser_metadata())+'}\n')
    if changed!=prepared['original_issue_count']:raise ValueError('resolved edges are not the complete original error set')
    require_sha(cache,cache_sha);require_sha(input_manifest,input_manifest_sha)
    temporary.replace(output);binding.update(status='accepted_cached_graph_resolution',output_path=str(output),output_sha256=file_sha(output),resolved_edges=changed)
    with summary_path.open('x',encoding='utf-8') as f:json.dump(binding,f,ensure_ascii=False,indent=2)
    return {'output':str(output),'sha256':binding['output_sha256'],'proof':str(proof_path),'summary':str(summary_path),'resolved':changed}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--input-manifest',required=True)
    parser.add_argument('--cache-sha256',required=True);parser.add_argument('--input-manifest-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(execute(args.cache,args.input_manifest,args.output,args.cache_sha256,args.input_manifest_sha256)))
