"""Push corrected source files to an instance by staging them as files, then commit.

Embedding base64 in a command line exceeded the shell's argument limit, so each file
is written to a small local staging file and uploaded with the existing uploader,
then moved into place inside the repo with SHA verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

FILES = {
    "src/satellite_rl/workbench/service.py": "src/satellite_rl/workbench/service.py",
    "src/satellite_rl/workbench/runner.py": "src/satellite_rl/workbench/runner.py",
}
MESSAGE = ("fix: 共同考核终点门槛与束数上限覆盖适配S2-P18\n\n"
           "终点门槛接受逐束自然布局；束数上限覆盖仅在模型配置声明不足200时注入，"
           "避免逐束自然布局被误判为来源不符。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    info = ops18.instance(args.port)
    staging = ops18.LOCAL.parent.parent / "tmp" / "sync_stage"
    staging.mkdir(parents=True, exist_ok=True)
    plan = {}
    for relative in FILES:
        local = Path(r"D:\paper") / relative
        copy = staging / Path(relative).name
        copy.write_bytes(local.read_bytes())
        plan[relative] = {"local": str(copy), "sha256": ops18.sha256_file(local)}
    if not args.execute:
        for relative, item in plan.items():
            print(f"计划 {relative} sha256={item['sha256'][:16]}")
        return
    remote_dir = f"{info['ops']}/sync"
    ops18.remote_mkdir(remote_dir, args.port) if hasattr(ops18, "remote_mkdir") else None
    ops18.remote_python(f"import pathlib;pathlib.Path({remote_dir!r}).mkdir(parents=True,exist_ok=True)",
                        args.port, timeout=120)
    for relative, item in plan.items():
        ops18.upload(Path(item["local"]), f"{remote_dir}/{Path(relative).name}", args.port)
    body_lines = [
        "import hashlib,json,pathlib,shutil,subprocess",
        f"repo=pathlib.Path({info['repo']!r})",
        f"stage=pathlib.Path({remote_dir!r})",
        "out={'files':{}}",
    ]
    for relative in FILES:
        name = Path(relative).name
        body_lines += [
            f"source=stage/{name!r}",
            f"target=repo/{relative!r}",
            "if target.is_file():",
            f"    out['files'].setdefault('before',{{}})[{relative!r}]=hashlib.sha256(target.read_bytes()).hexdigest()",
            "shutil.copy2(str(source),str(target))",
            f"out['files'].setdefault('after',{{}})[{relative!r}]=hashlib.sha256(target.read_bytes()).hexdigest()",
        ]
    body_lines += [
        "def run(*cmd):",
        "    return subprocess.run(cmd,capture_output=True,text=True,cwd=str(repo))",
        "run('git','config','user.email','s2p18@local')",
        "run('git','config','user.name','S2-P18 runner')",
        "for relative in " + repr(list(FILES)) + ":",
        "    run('git','add',relative)",
        "commit=run('git','commit','-m'," + repr(MESSAGE) + ")",
        "out['commit']=(commit.stdout or '').strip()[:200]",
        "out['head']=run('git','rev-parse','HEAD').stdout.strip()",
        "out['status']=run('git','status','--porcelain').stdout.strip()[:200]",
        "print(json.dumps(out,ensure_ascii=False))",
    ]
    result = ops18.remote_python("\n".join(body_lines), args.port, timeout=900)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(result["stderr"][-400:])
    report = json.loads(lines[-1])
    report["sha_verified"] = all(
        report["files"]["after"][relative] == plan[relative]["sha256"] for relative in FILES)
    ops18.write_json(ops18.LOCAL / "preflight" / f"sync_fix_{args.port}.json",
                     {"checked_utc": ops18.utc_now(), "port": args.port, **report})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
