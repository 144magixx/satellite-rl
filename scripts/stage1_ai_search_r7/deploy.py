"""Prepare an exact bundle; explicit root-admitted staging only, no service/start."""
import argparse,json,hashlib,subprocess,os,shlex
from pathlib import Path
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import digest
from satellite_rl.training.run_monitor import atomic_json
from .contract import GROUP,PORTS,DESIGN_SHA

SCOPES=('src','scripts','pyproject.toml','.gitattributes')

def prepare_bundle(repository,commit,output):
    repo=Path(repository);output=Path(output)
    actual=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=commit or len(commit)!=40:raise ValueError('actual frozen source HEAD required')
    if subprocess.run(['git','-C',str(repo),'status','--porcelain','--',*SCOPES],capture_output=True,text=True,check=True).stdout.strip():raise ValueError('complete src/scripts source must be committed clean')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('preserve original bundle, no overwrite')
    subprocess.run(['git','-C',str(repo),'bundle','create',str(output),'HEAD'],capture_output=True,check=True)
    subprocess.run(['git','-C',str(repo),'bundle','verify',str(output)],capture_output=True,check=True)
    return dict(path=str(output),sha256=digest(output),scientific_commit=commit)

def stage(spec_ref,bundle_ref):
    """All code executes only after an explicit root source-staging receipt."""
    spec=json.loads(pin(spec_ref).read_text(encoding='utf8'));port=spec['ssh_port'];receipt=json.loads(pin(spec['root_staging_admission_ref']).read_text(encoding='utf8'))
    if port not in PORTS or spec.get('group')!=GROUP or spec.get('design_sha256')!=DESIGN_SHA or receipt.get('passed')is not True or receipt.get('scientific_commit')!=spec['scientific_commit'] or receipt.get('group')!=GROUP or receipt.get('bundle_sha256')!=bundle_ref['sha256']:raise ValueError('root exact source staging admission')
    import paramiko
    password=os.environ['BUPT_PASSWORD'];client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.160.17.102',port=port,username='u2024110224',password=password,look_for_keys=False,allow_agent=False,timeout=20);del password
    lane=f'/home/u2024110224/first_stage_satellite_rl/s1_ai17/{port}/round7';remote_bundle=lane+'/source_inputs/'+bundle_ref['sha256']+'/source.bundle';source=lane+'/source_repo/'+spec['scientific_commit']
    try:
        def command(text):
            _,stdout,stderr=client.exec_command(text,timeout=60);status=stdout.channel.recv_exit_status();out=stdout.read().decode();err=stderr.read().decode()
            if status:raise RuntimeError('source staging command failed: '+err[-800:])
            return out
        command('mkdir -p '+shlex.quote(str(Path(remote_bundle).parent).replace('\\','/'))+' '+shlex.quote(lane+'/source_repo'))
        sftp=client.open_sftp();sftp.put(str(pin(bundle_ref)),remote_bundle);sftp.close()
        actual=command('sha256sum '+shlex.quote(remote_bundle)).split()[0]
        if actual!=bundle_ref['sha256']:raise ValueError('remote bundle bytes differ')
        command('test ! -e '+shlex.quote(source))
        command('git clone --no-checkout '+shlex.quote(remote_bundle)+' '+shlex.quote(source))
        command('git -C '+shlex.quote(source)+' checkout --detach '+shlex.quote(spec['scientific_commit']))
        return dict(group=GROUP,ssh_port=port,source_root=source,bundle_ref={**bundle_ref,'path':remote_bundle},source_staged=True,service_started=False,science_POSTs=0)
    finally:client.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--commit',required=True);p.add_argument('--bundle',required=True);p.add_argument('--stage-spec');p.add_argument('--stage-spec-sha256');p.add_argument('--receipt',required=True);a=p.parse_args()
    bundle=prepare_bundle(a.repo,a.commit,a.bundle);result={'bundle_ref':bundle,'source_staged':False,'science_POSTs':0}
    if a.stage_spec:result=stage(dict(path=a.stage_spec,sha256=a.stage_spec_sha256),bundle)
    atomic_json(a.receipt,result)
