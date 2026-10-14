"""Conservative project import closure using AST only, no project execution."""
import ast
from pathlib import Path

ENTRIES=tuple('scripts.stage1_ai_search_r12.'+name for name in
    ('server','worker','engineering','collect','reader','geometry_reader','normalize','exact_order','region_v2','historical_s'))

def project_closure(repo):
    repo=Path(repo).resolve();roots=(repo/'src',repo);queue=list(ENTRIES);seen=set();files={}
    def locate(module):
        relative=Path(*module.split('.'))
        for base in roots:
            for path in ((base/relative).with_suffix('.py'),base/relative/'__init__.py'):
                if path.is_file():return path
        return None
    while queue:
        module=queue.pop()
        if module in seen:continue
        seen.add(module)
        if 'ai_search_r11' in module:raise ValueError('dirty R11 cannot enter runtime closure')
        path=locate(module)
        if path is None:continue  # external/stdlib or namespace package
        real=path.resolve()
        if not real.is_relative_to(repo):raise ValueError('project dependency outside exact frozen tree')
        files[real.relative_to(repo).as_posix()]=real
        if len(files)>256:raise ValueError('bounded256 project dependency leaves')
        parts=module.split('.')
        for i in range(1,len(parts)):
            parent='.'.join(parts[:i]);candidate=locate(parent)
            if candidate is not None and candidate.name=='__init__.py':queue.append(parent)
        package=parts if path.name=='__init__.py' else parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if isinstance(node,ast.Import):queue.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []))
                if base:queue.append(base)
                queue.extend(base+'.'+a.name for a in node.names if a.name!='*' and base)
    return dict(sorted(files.items()))
