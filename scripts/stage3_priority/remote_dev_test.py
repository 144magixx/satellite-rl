"""Run short S3-P03 source checks in an isolated remote development snapshot."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute, paramiko  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
OUTPUT = Path("D:/paper/outputs/verification/stage3_si_priority_20261003/remote_dev")


def snapshot(pattern: str) -> tuple[Path, str, int]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / "dev_source.zip"
    count = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        files = sorted([p for p in (PROJECT / "src").rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"])
        files += sorted([p for p in (PROJECT / "tests").rglob(pattern) if p.is_file()])
        files += sorted([p for p in (PROJECT / "data").rglob("*") if p.is_file()])
        files += sorted([p for p in (PROJECT / "models/pretrained").rglob("*") if p.is_file()])
        files += [PROJECT / "pyproject.toml"]
        for path in files:
            rel = path.relative_to(PROJECT).as_posix()
            output.write(path, rel)
            count += 1
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as source:
        members = [{"path": name, "bytes": len(source.read(name)),
                    "sha256": hashlib.sha256(source.read(name)).hexdigest()}
                   for name in source.namelist()]
    head = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "-C", str(PROJECT), "diff", "--binary", "HEAD", "--",
                           "src", "tests", "pyproject.toml"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=True).stdout
    (OUTPUT / f"source_diff_{archive_sha[:16]}.patch").write_text(diff, encoding="utf-8")
    manifest = {"archive": str(archive), "archive_sha256": archive_sha,
                "snapshot_utc": datetime.now(timezone.utc).isoformat(),
                "base_head": head, "source_patch_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "members": members}
    (OUTPUT / f"source_manifest_{archive_sha[:16]}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return archive, archive_sha, count


def run(port: int, archive: Path, sha: str, count: int, pattern: str) -> dict:
    remote_zip = f"{BASE}/shared_data/dev_snapshots/{sha[:16]}.zip"
    transport = connect(port)
    try:
        mkdir = execute(transport, "mkdir -p " + BASE + "/shared_data/dev_snapshots", timeout=30)
        if mkdir["exit_code"]:
            raise RuntimeError(mkdir["stderr"][-500:])
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(archive), remote_zip + ".part")
        code = f'''
import datetime,hashlib,json,os,pathlib,subprocess,zipfile
base=pathlib.Path({BASE!r});part=pathlib.Path({(remote_zip+'.part')!r});z=pathlib.Path({remote_zip!r})
root=base/'shared_data/dev_snapshots'/{sha[:16]!r};dqn={DQN!r}
if hashlib.sha256(part.read_bytes()).hexdigest()!={sha!r}:raise RuntimeError('Dev snapshot transfer SHA mismatch')
part.replace(z)
root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(z) as archive:
 names=archive.namelist()
 if len(names)!={count} or any(pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts for name in names):
  raise RuntimeError('Unexpected dev snapshot members')
 archive.extractall(root)
if not (root/'.git').is_dir():
 subprocess.run(['git','init',str(root)],capture_output=True,text=True,check=True,timeout=60)
 history=base/'../s3_all_routes_20261003/20418/repo'
 history=history.resolve()
 subprocess.run(['git','-C',str(root),'fetch',str(history),'HEAD'],capture_output=True,text=True,check=True,timeout=180)
deps=base/'shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1'
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+os.pathsep+str(deps),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
probe=subprocess.run([dqn,'-B','-c','import json,sys,torch,numpy,scipy,cvxpy,clarabel,satellite_rl;print(json.dumps({{"python":sys.executable,"torch":torch.__version__,"cuda":torch.cuda.is_available(),"numpy":numpy.__version__,"scipy":scipy.__version__,"cvxpy":cvxpy.__version__,"clarabel":clarabel.__version__,"package":satellite_rl.__file__}}))'],cwd=root,env=env,capture_output=True,text=True,timeout=90)
if probe.returncode:raise RuntimeError('Dev interpreter import failed: '+probe.stderr[-1200:])
info=json.loads(probe.stdout.splitlines()[-1])
if pathlib.Path(info['package']).resolve()!=root/'src/satellite_rl/__init__.py' or not info['cuda']:
 raise RuntimeError('Dev test imported wrong package or CUDA unavailable')
test=subprocess.run([dqn,'-B','-m','unittest','discover','-s','tests','-p',{pattern!r},'-v'],
 cwd=root,env=env,capture_output=True,text=True,timeout=1800)
(root/'test_stdout.log').write_text(test.stdout)
(root/'test_stderr.log').write_text(test.stderr)
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'port':{port},
 'snapshot_sha256':{sha!r},'snapshot_root':str(root),'source_file_count':{count},'pattern':{pattern!r},'runtime':info,
 'exit_code':test.returncode,'stdout':test.stdout[-12000:],'stderr':test.stderr[-12000:],
 'stdout_sha256':hashlib.sha256(test.stdout.encode()).hexdigest(),
 'stderr_sha256':hashlib.sha256(test.stderr.encode()).hexdigest()}}))
'''
        payload = base64.b64encode(code.encode()).decode()
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=1920)
        if not result["exit_code"]:
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                suffix = f"{port}_{pattern[:-3].replace('*', 'all')}_{sha[:16]}"
                sftp.get(f"{BASE}/shared_data/dev_snapshots/{sha[:16]}/test_stdout.log", str(OUTPUT / f"stdout_{suffix}.log"))
                sftp.get(f"{BASE}/shared_data/dev_snapshots/{sha[:16]}/test_stderr.log", str(OUTPUT / f"stderr_{suffix}.log"))
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1500:])
    return json.loads(result["stdout"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=(20418, 20424), default=20418)
    parser.add_argument("--pattern", default="test_power_priority.py")
    args = parser.parse_args()
    if (args.pattern != "test_*.py" and not args.pattern.startswith("test_power_")
            and args.pattern != "test_workbench_service.py") or not args.pattern.endswith(".py") or "/" in args.pattern or "\\" in args.pattern:
        raise ValueError("Only full suite or a relevant workbench test module may be selected")
    archive, sha, count = snapshot(args.pattern)
    result = run(args.port, archive, sha, count, args.pattern)
    path = OUTPUT / f"dev_test_{args.port}_{args.pattern[:-3].replace('*', 'all')}_{sha[:16]}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "snapshot_root": result["snapshot_root"],
                      "exit_code": result["exit_code"], "stderr_tail": result["stderr"][-2500:]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
