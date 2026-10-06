"""Update the deployed repo checkout to a new commit without touching running jobs.

Already-created jobs read their own archived source snapshot, so updating the
repo files only affects jobs created afterwards. The workbench service keeps
running; no worker is signalled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

CHECK_FILES = {
    "config": "src/satellite_rl/workbench/config.py",
    "checkpoints": "src/satellite_rl/workbench/checkpoints.py",
    "algorithms": "src/satellite_rl/agents/workbench_algorithms.py",
    "networks": "src/satellite_rl/agents/workbench_networks.py",
    "environment": "src/satellite_rl/workbench/environment.py",
}


def git(source: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def _normalized_sha(data: bytes) -> str:
    """Hash file content with line endings normalised to LF.

    The Windows checkout stores CRLF while the remote checkout stores LF; the
    source content is identical, so the comparison must not depend on that.
    """
    import hashlib
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def local_hashes() -> dict:
    return {key: _normalized_sha((Path("D:/paper") / rel).read_bytes())
            for key, rel in CHECK_FILES.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="D:/paper")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    if git(source, "status", "--porcelain"):
        raise RuntimeError("本地工作区不干净，拒绝以未提交内容更新远端")
    commit = git(source, "rev-parse", "HEAD")
    bundle = ops18.LOCAL / f"s2p18_{commit}.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.is_file():
        subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"], check=True)
    subprocess.run(["git", "-C", str(source), "bundle", "verify", str(bundle)],
                   capture_output=True, text=True, check=True)
    expected = local_hashes()
    report = {"commit": commit, "bundle_sha256": ops18.sha256_file(bundle), "instances": {}}
    for port in ops18.PORTS:
        info = ops18.instance(port)
        remote_bundle = f"{info['ops']}/s2p18_{commit}.bundle"
        upload = ops18.upload(bundle, remote_bundle, port)
        if upload["local_sha256"] != upload["remote"]:
            raise RuntimeError(f"bundle 上传校验失败 {port}")
        body = (
            "import hashlib,json,pathlib,subprocess\n"
            f"repo=pathlib.Path({info['repo']!r})\n"
            f"bundle=pathlib.Path({remote_bundle!r})\n"
            f"commit={commit!r}\n"
            f"files={CHECK_FILES!r}\n"
            "def run(a):\n"
            "    p=subprocess.run(a,capture_output=True,text=True)\n"
            "    return {'code':p.returncode,'out':p.stdout.strip()[:300],'err':p.stderr.strip()[-300:]}\n"
            "steps={}\n"
            "steps['fetch']=run(['git','-C',str(repo),'fetch','--quiet',str(bundle),'HEAD'])\n"
            "steps['checkout']=run(['git','-C',str(repo),'checkout','--quiet',commit])\n"
            "steps['head']=run(['git','-C',str(repo),'rev-parse','HEAD'])\n"
            "steps['dirty']=run(['git','-C',str(repo),'status','--porcelain'])\n"
            "hashes={k:hashlib.sha256((repo/v).read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest()"
            " for k,v in files.items()}\n"
            "print(json.dumps({'steps':steps,'hashes':hashes},ensure_ascii=False))\n"
        )
        result = ops18.remote_python(body, port, timeout=600)
        if result["exit_code"]:
            raise RuntimeError(f"{port} 更新失败: {result['stderr'][-400:]}")
        detail = json.loads(result["stdout"].strip().splitlines()[-1])
        if detail["steps"]["head"]["out"] != commit or detail["steps"]["dirty"]["out"]:
            raise RuntimeError(f"{port} 更新后 HEAD 或工作区不符")
        if detail["hashes"] != expected:
            raise RuntimeError(f"{port} 关键文件SHA与本地不一致")
        report["instances"][str(port)] = detail
    ops18.write_json(ops18.LOCAL / "preflight" / f"repo_update_{commit[:12]}.json",
                     {"checked_utc": ops18.utc_now(), **report})
    print(json.dumps({"commit": commit, "bundle_sha256": report["bundle_sha256"],
                      "instances_ok": sorted(report["instances"])}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
