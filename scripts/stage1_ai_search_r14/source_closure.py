"""Exact conservative project source closure and snapshot; AST only."""
import ast
import hashlib
import json
import posixpath
from pathlib import Path

def relocate_refs(manifest,absolute_target_root):
    if not absolute_target_root.startswith('/'):raise ValueError('exact absolute remote source root')
    return [{'path':posixpath.join(absolute_target_root,r['relative_path']),'bytes':r['bytes'],'sha256':r['sha256']}
            for r in manifest['source_refs']]


def snapshot(repo, output):
    repo=Path(repo).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    roots=(repo/'src',repo)
    entries=['satellite_rl.coverage.ai_search_r5.core','satellite_rl.coverage.ai_search_r10.core']
    entries += ['scripts.stage1_ai_search_r14.'+p.stem for p in (repo/'scripts/stage1_ai_search_r14').glob('*.py')]
    entries += ['satellite_rl.coverage.ai_search_r14.'+p.stem for p in (repo/'src/satellite_rl/coverage/ai_search_r14').glob('*.py')]
    queue=list(entries);seen=set();files={};external=set();edges=[]
    def locate(name):
        relative=Path(*name.split('.'))
        for root in roots:
            for p in ((root/relative).with_suffix('.py'),root/relative/'__init__.py'):
                if p.is_file():return p.resolve()
    while queue:
        module=queue.pop()
        if module in seen:continue
        seen.add(module);path=locate(module)
        if path is None:
            external.add(module);continue
        if not path.is_relative_to(repo):raise ValueError('source escapes workspace')
        if '.ai_search_r11.' in module:raise ValueError('old dirtyR11 forbidden in R14 closure')
        relative=path.relative_to(repo).as_posix();files[relative]=path
        if len(files)>256:raise ValueError('finite256 source leaves')
        parts=module.split('.');package=parts if path.name=='__init__.py' else parts[:-1]
        for i in range(1,len(parts)):
            parent='.'.join(parts[:i]);p=locate(parent)
            if p is not None and p.name=='__init__.py':queue.append(parent)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []))
                if base:names.append(base)
                names += [base+'.'+a.name for a in node.names if a.name!='*' and base and locate(base+'.'+a.name)]
            for name in names:
                queue.append(name);edges.append([module,name])
    refs=[]
    for relative,path in sorted(files.items()):
        raw=path.read_bytes();target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        refs.append({'relative_path':relative,'original_path':str(path),'snapshot_path':str(target),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    result={'schema':'R14ASTTransitiveSourceSnapshotV1','entries':entries,'source_refs':refs,'edges':edges,
            'external_imports':sorted(external),'import_path_order':['<snapshot>/src','<snapshot>'],
            'dynamic_imports_must_be_reviewed':True,'actual_math_imports_or_calls':0,
            'R5_expected_sha256':'4e3f7eaa723b1f95830648658720afce506b3d2f681df64f1264b0e6b737d3e6',
            'R10_expected_sha256':'440aafce1830f4a2045f1e6ac48613cdfcf873af025f9f38b8fdb1a2bd5db1f8'}
    for prefix in ('R5','R10'):
        relative='src/satellite_rl/coverage/ai_search_'+prefix.lower()+'/core.py'
        if next(r['sha256'] for r in refs if r['relative_path']==relative)!=result[prefix+'_expected_sha256']:
            raise ValueError('original frozen mathematical bytes differ')
    (output/'source_manifest.json').write_text(json.dumps(result,ensure_ascii=True,indent=2),encoding='utf8')
    return result
