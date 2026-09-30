"""Sync a committed Git revision or securely forward the stage-one workbench.

Requires paramiko in the controller environment. BUPT_PASSWORD stays in memory;
Windows user-scope variables are supported without printing their contents.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import socketserver
import subprocess
import tempfile


def read_password():
    password = os.environ.get("BUPT_PASSWORD")
    if not password and os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            password = winreg.QueryValueEx(key, "BUPT_PASSWORD")[0]
    if not password:
        raise RuntimeError("BUPT_PASSWORD环境变量不可用")
    return password


def connect(args):
    import paramiko
    class PinnedKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if actual != args.host_key:
                raise paramiko.SSHException("远端SSH主机指纹不匹配")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(PinnedKey())
    client.connect(args.host, port=args.port, username=args.user, password=read_password(),
                   allow_agent=False, look_for_keys=False, timeout=15,
                   banner_timeout=30, auth_timeout=30)
    client.get_transport().set_keepalive(20)
    return client


def execute(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=120)
    out, err = stdout.read().decode(), stderr.read().decode()
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(err or out or f"远端命令失败: {status}")
    return out


def sync(client, args):
    root = Path(args.repository).resolve()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "data", "pyproject.toml"], cwd=root)
    if dirty.strip():
        raise ValueError("源码有未提交修改，先核验并提交后再同步")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    target = args.remote_root.rstrip("/")
    incoming = target + "/setup/source_" + revision + ".bundle"
    execute(client, "mkdir -p " + shlex.quote(target + "/setup"))
    with tempfile.TemporaryDirectory(prefix="stage1_git_") as temporary:
        bundle = Path(temporary) / "source.bundle"
        subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=root, check=True)
        with client.open_sftp() as sftp:
            sftp.put(str(bundle), incoming)
    repo = target + "/repo"
    command = ("set -eu; if [ ! -d " + shlex.quote(repo + "/.git") + " ]; then git clone " +
               shlex.quote(incoming) + " " + shlex.quote(repo) + "; fi; cd " + shlex.quote(repo) +
               "; test -z \"$(git status --porcelain -- src data pyproject.toml)\"; git fetch " +
               shlex.quote(incoming) + " HEAD; git checkout --detach " + shlex.quote(revision) + "; git rev-parse HEAD")
    lines = execute(client, command).splitlines()
    if not lines or lines[-1].strip() != revision:
        raise RuntimeError("本地与远端提交号不一致")
    print(json.dumps({"git_commit": revision, "remote_repository": repo, "verified": True}))


def tunnel(client, args):
    transport = client.get_transport()
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            channel = transport.open_channel("direct-tcpip", ("127.0.0.1", args.remote_port), self.request.getpeername())
            try:
                while True:
                    readable, _, _ = select.select([self.request, channel], [], [])
                    for stream in readable:
                        data = stream.recv(65536)
                        if not data:
                            return
                        (channel if stream is self.request else self.request).sendall(data)
            finally:
                channel.close()
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with Server(("127.0.0.1", args.local_port), Handler) as server:
        print(f"第一阶段工作台 http://127.0.0.1:{args.local_port}/#coverage", flush=True)
        server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("sync", "tunnel", "check"))
    p.add_argument("--host", default="10.160.17.102")
    p.add_argument("--port", type=int, default=20418)
    p.add_argument("--user", default="u2024110224")
    p.add_argument("--host-key", default="a6ym3EfC35pPdkqWO8lg+4As2nLDI1eKu8cur4P7WDc")
    p.add_argument("--remote-root", default="/home/u2024110224/first_stage_satellite_rl")
    p.add_argument("--repository", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--local-port", type=int, default=8876)
    p.add_argument("--remote-port", type=int, default=8876)
    args = p.parse_args()
    client = connect(args)
    try:
        if args.mode == "sync":
            sync(client, args)
        elif args.mode == "tunnel":
            tunnel(client, args)
        else:
            print(execute(client, "hostname; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
