"""Deploy one frozen scientific commit to both S2-P18 instances and start services.

Default is a read-only plan. ``--execute`` uploads a verified Git bundle, creates
the two isolated checkouts, verifies the imported package path and GPU binding,
then starts one workbench per instance. It never submits a task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402


def git(source: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def require_commit(value: str) -> str:
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("提交号必须是40位十六进制")
    return value


def plan(source: Path, commit: str) -> dict:
    source = source.resolve()
    if git(source, "rev-parse", "HEAD") != commit:
        raise RuntimeError("正式源码 HEAD 与指定提交不符")
    dirty = git(source, "status", "--porcelain")
    if dirty:
        raise RuntimeError("正式源码工作区不干净：" + dirty[:400])
    report = {"scientific_commit": commit, "source_repo": str(source), "instances": {}}
    for port in ops18.PORTS:
        info = ops18.instance(port)
        code = f'''
import json,os,pathlib,shutil,socket,subprocess
root=pathlib.Path({info["root"]!r});repo=pathlib.Path({info["repo"]!r})
def run(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return {{"exit_code":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr[-300:]}}
with socket.socket() as sock:
 sock.settimeout(2);busy=sock.connect_ex(("127.0.0.1",{info["api_port"]}))==0
existing={{"repo":repo.is_dir(),"repo_head":run(["git","-C",str(repo),"rev-parse","HEAD"]) if (repo/".git").is_dir() else None,
 "repo_dirty":run(["git","-C",str(repo),"status","--porcelain"]) if (repo/".git").is_dir() else None}}
print(json.dumps({{"gpu":run(["nvidia-smi","--query-gpu=name,uuid,memory.total,memory.used","--format=csv,noheader,nounits"]),
 "home_free_bytes":shutil.disk_usage("/home").free,"cpus":len(os.sched_getaffinity(0)),
 "api_port_busy":busy,"existing":existing}},ensure_ascii=False))
'''
        result = ops18.remote_python(code, port, timeout=120)
        if result["exit_code"]:
            raise RuntimeError(f"预检失败 {port}: {result['stderr'][-400:]}")
        detail = json.loads(result["stdout"].strip().splitlines()[-1])
        report["instances"][str(port)] = {
            "target": info["target"], "root": info["root"], "repo": info["repo"],
            "archive": info["archive"], "api_port": info["api_port"], **detail}
    return report


def deploy(source: Path, commit: str) -> dict:
    source = source.resolve()
    preflight = plan(source, commit)
    bundle = ops18.LOCAL / f"s2p18_{commit}.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.is_file():
        subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"], check=True)
    subprocess.run(["git", "-C", str(source), "bundle", "verify", str(bundle)],
                   capture_output=True, text=True, check=True)
    receipt = {"scientific_commit": commit, "bundle": str(bundle),
               "bundle_sha256": ops18.sha256_file(bundle), "instances": {}}
    for port in ops18.PORTS:
        info = ops18.instance(port)
        remote_bundle = f"{info['ops']}/s2p18_{commit}.bundle"
        upload = ops18.upload(bundle, remote_bundle, port)
        if upload["local_sha256"] != upload["remote"]:
            raise RuntimeError(f"bundle 上传校验失败 {port}")
        code = f'''
import json,pathlib,subprocess
repo=pathlib.Path({info["repo"]!r});archive=pathlib.Path({info["archive"]!r})
bundle=pathlib.Path({remote_bundle!r});commit={commit!r}
def run(args,**kw):
 p=subprocess.run(args,capture_output=True,text=True,**kw)
 return {{"exit_code":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr[-300:]}}
steps={{}}
if repo.is_dir():
 steps["clone"]={{"exit_code":0,"stdout":"already_present","stderr":""}}
else:
 steps["clone"]=run(["git","clone","--quiet",str(bundle),str(repo)])
steps["fetch"]=run(["git","-C",str(repo),"fetch","--quiet",str(bundle),"HEAD"])
steps["checkout"]=run(["git","-C",str(repo),"checkout","--quiet",commit])
steps["head"]=run(["git","-C",str(repo),"rev-parse","HEAD"])
steps["dirty"]=run(["git","-C",str(repo),"status","--porcelain"])
archive.mkdir(parents=True,exist_ok=True)
print(json.dumps({{k:v for k,v in steps.items()}},ensure_ascii=False))
'''
        result = ops18.remote_python(code, port, timeout=600)
        if result["exit_code"]:
            raise RuntimeError(f"部署失败 {port}: {result['stderr'][-500:]}")
        steps = json.loads(result["stdout"].strip().splitlines()[-1])
        if steps["head"]["stdout"] != commit or steps["dirty"]["stdout"]:
            raise RuntimeError(f"{port} 部署后 HEAD 或工作区不符")
        receipt["instances"][str(port)] = {"bundle": remote_bundle,
                                           "bundle_remote_sha256": upload["remote"], "steps": steps}
    ops18.write_json(ops18.LOCAL / "preflight" / f"deployment_{commit[:12]}.json",
                     {"status": "deployed_clean_no_workbench_or_jobs", **receipt, **preflight})
    return receipt


def serve(commit: str) -> dict:
    receipt_path = ops18.LOCAL / "preflight" / f"deployment_{commit[:12]}.json"
    if not receipt_path.is_file():
        raise RuntimeError("缺少干净部署回执，先执行 deploy")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("scientific_commit") != commit:
        raise RuntimeError("部署回执提交号不符")
    report = {}
    for port in ops18.PORTS:
        info = ops18.instance(port)
        code = f'''
import json,os,pathlib,socket,subprocess,time
repo=pathlib.Path({info["repo"]!r});archive=pathlib.Path({info["archive"]!r})
commit={commit!r};api_port={info["api_port"]!r}
head=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
dirty=subprocess.run(["git","-C",str(repo),"status","--porcelain"],capture_output=True,text=True).stdout.strip()
if head!=commit or dirty: raise RuntimeError("冻结repo与提交不符")
with socket.socket() as sock:
 sock.settimeout(2)
 if sock.connect_ex(("127.0.0.1",int(api_port)))==0:
  print(json.dumps({{"status":"port_already_busy","api_port":api_port}}));raise SystemExit
log=archive.parent/"ops"/"workbench.log"
log.parent.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
handle=open(log,"ab",buffering=0)
process=subprocess.Popen([{ops18.DQN!r},"-B","-m","satellite_rl.workbench.server",
 "--archive-root",str(archive),"--port",str(api_port),"--python",{ops18.DQN!r},
 "--max-workers","2"],stdout=handle,stderr=handle,stdin=subprocess.DEVNULL,
 cwd=str(repo/"src"),env=env,start_new_session=True)
for _ in range(60):
 time.sleep(1)
 with socket.socket() as sock:
  sock.settimeout(2)
  if sock.connect_ex(("127.0.0.1",int(api_port)))==0: break
else:
 raise RuntimeError("工作台未在60秒内监听")
print(json.dumps({{"status":"serving","pid":process.pid,"api_port":api_port,
 "log":str(log),"archive":str(archive),"commit":commit}},ensure_ascii=False))
'''
        result = ops18.remote_python(code, port, timeout=300)
        if result["exit_code"]:
            raise RuntimeError(f"起服务失败 {port}: {result['stderr'][-500:]}")
        report[str(port)] = json.loads(result["stdout"].strip().splitlines()[-1])
    ops18.write_json(ops18.LOCAL / "preflight" / f"workbench_{commit[:12]}.json",
                     {"scientific_commit": commit, "instances": report, "checked_utc": ops18.utc_now()})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--action", choices=("plan", "deploy", "serve"), required=True)
    parser.add_argument("--source", default="D:/paper")
    args = parser.parse_args()
    commit = require_commit(args.commit)
    source = Path(args.source)
    if args.action == "plan":
        output = plan(source, commit)
    elif args.action == "deploy":
        output = deploy(source, commit)
    else:
        output = serve(commit)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
