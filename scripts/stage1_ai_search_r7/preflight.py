"""Read-only deployed R7 source/DQN/CUDA/resource preflight; no scientific calls."""
import argparse,json,sys,subprocess
from pathlib import Path
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.r7_resources import observe
from satellite_rl.training.run_monitor import atomic_json
from .contract import GROUP,PORTS,DESIGN_SHA

def check(spec_ref,output):
    spec=json.loads(pin(spec_ref).read_text(encoding='utf8'));port=spec['ssh_port']
    if spec.get('group')!=GROUP or port not in PORTS or spec.get('design_sha256')!=DESIGN_SHA:raise ValueError('R7 root source deployment spec')
    source=Path(spec['source_root']).resolve();lane=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17')/str(port)/'round7'
    if not source.is_relative_to(lane) or source.name!=spec['scientific_commit']:raise ValueError('isolated content-addressed scientific checkout')
    actual=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=spec['scientific_commit'] or subprocess.run(['git','-C',str(source),'status','--porcelain','--','src','scripts','pyproject.toml','.gitattributes'],capture_output=True,text=True,check=True).stdout.strip():raise ValueError('actual source commit/clean scope differs')
    members=spec['source_file_sha256']
    if not members or any(digest(source/name)!=sha for name,sha in members.items()):raise ValueError('deployed actual source members differ')
    import satellite_rl,torch,numpy
    if not Path(satellite_rl.__file__).resolve().is_relative_to(source/'src'):raise ValueError('actual imported package outside scientific checkout')
    expected='/home/u2024110224/first_stage_satellite_rl/envs/DQN'
    if Path(sys.prefix).resolve()!=Path(expected).resolve() or torch.version.cuda is None or not torch.cuda.is_available():raise ValueError('actual DQN CUDA required; no CPU/base fallback')
    archive=lane/'archive';archive.mkdir(parents=True,exist_ok=True);resource=observe(archive)
    if resource['foreign_workers'] or resource['disk_free_bytes']<8589934592:raise ValueError('actual foreign worker or8GiB reserve')
    from satellite_rl.workbench.r7_model_io import model_source_identity
    result=dict(group=GROUP,passed=True,read_only_preflight=True,ssh_port=port,scientific_commit=actual,
                python=sys.executable,environment_prefix=sys.prefix,package_file=satellite_rl.__file__,
                torch=torch.__version__,torch_cuda=torch.version.cuda,numpy=numpy.__version__,
                immutable_model_sources=model_source_identity(),resource=resource,
                root_resource_admission=False,science_POSTs=0,real_data_reads=0,prepare=0,T=0,D5=0,optimizer=0)
    atomic_json(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    check(dict(path=a.spec,sha256=a.spec_sha256),a.output)
