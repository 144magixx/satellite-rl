"""一次性探测账号容器：能否登录、GPU、python/torch、/home。用法：python _probe_accounts.py user port [port ...]"""
import sys
sys.path.insert(0, 'scripts/s2lab')
import paramiko
import remote

user = sys.argv[1]
for port in map(int, sys.argv[2:]):
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(remote.REMOTE_HOST, port=port, username=user, password=remote.read_password(), allow_agent=False,
                  look_for_keys=False, timeout=15, banner_timeout=20, auth_timeout=20)
        out = remote.run(c, 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; echo HOME=$HOME; ls $HOME | head -20; '
                            'ls -d $HOME/*/envs/*/bin/python 2>/dev/null | head; ls /opt/conda/envs 2>/dev/null; which python3; '
                            '/opt/conda/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1; '
                            'nproc; df -h $HOME | tail -1', check=False)
        print(f'== {user}@{port}\n{out.strip()}\n', flush=True)
        c.close()
    except Exception as e:
        print(f'== {user}@{port} FAIL {type(e).__name__}: {e}', flush=True)
