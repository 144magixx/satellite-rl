"""Actual new postprocess epoch/source bytes, separate from original algorithm."""
import subprocess
from pathlib import Path
from .storage import read_json,digest

def execution_source():
    root=Path(__file__).resolve().parents[3];manifest=root/'source_manifest.json'
    if manifest.is_file():
        value=read_json(manifest,{})
        if value.get('format_version')!=2 or value.get('group')!='S1-AI17-R7':raise ValueError('actual new postprocess source snapshot required')
        epoch=value['git_commit']
    else:
        epoch=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        dirty=subprocess.run(['git','-C',str(root),'status','--porcelain','--','src','scripts','pyproject.toml','.gitattributes'],capture_output=True,text=True,check=True).stdout.strip()
        if dirty:raise ValueError('postprocess actual source must be committed clean before execution')
    if len(epoch)!=40:raise ValueError('actual postprocess source epoch')
    names=['scripts/stage1_ai_search_r7/analyze_node.py','scripts/stage1_ai_search_r7/analyze_labels.py','scripts/stage1_ai_search_r7/label_audit.py',
           'src/satellite_rl/workbench/r7_publish.py','src/satellite_rl/workbench/r7_postprocess_source.py','src/satellite_rl/workbench/r7_node_audit.py',
           'scripts/stage1_ai_search_r4/audit.py','scripts/stage1_ai_search_r5/audit.py']
    names+=['src/satellite_rl/shared_interface/'+p.name for p in (root/'src/satellite_rl/shared_interface').glob('*.py')]
    return dict(postprocess_source_epoch=epoch,actual_postprocess_source_file_sha256={name:digest(root/name) for name in names},
                original_algorithm_source_is_separately_retained=True)

def new_output(output):
    """One new plan-selected output root; never overwrite a prior gate/table."""
    output=Path(output).resolve();root=output.parent
    if root.exists() and (not root.is_dir() or any(root.iterdir())):raise ValueError('postprocess outputRoot already exists with content; choose a new independent plan-selected directory')
    root.mkdir(parents=True,exist_ok=True)
    return output
