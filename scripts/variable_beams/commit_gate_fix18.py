"""Configure the instance repo identity, move the backup out, and commit the gate fix."""
import json
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

PORT = 20418
MESSAGE = ("fix: 共同考核终点门槛接受S2-P18自然逐束布局\n\n"
           "仅把接受的布局扩展为可变束数上下文与自然逐束布局两种；混合束数、种子51001、"
           "四十万步预算与来源步数等约束未放宽。原文件备份移至 ops/service.py.pre_s2p18_gate。")

BODY = (
    "import json,pathlib,shutil,subprocess\n"
    f"repo=pathlib.Path({ops18.instance(PORT)['repo']!r})\n"
    f"ops=pathlib.Path({ops18.instance(PORT)['ops']!r})\n"
    "def run(*args):\n"
    "    return subprocess.run(args,capture_output=True,text=True,cwd=str(repo))\n"
    "out={}\n"
    "# 1) 仓库级身份，避免依赖全局配置\n"
    "run('git','config','user.email','s2p18@local')\n"
    "run('git','config','user.name','S2-P18 runner')\n"
    "# 2) 备份文件移出工作区\n"
    "stray=repo/'src/satellite_rl/workbench/service.py.pre_s2p18_gate'\n"
    "if stray.is_file():\n"
    "    ops.mkdir(parents=True,exist_ok=True)\n"
    "    shutil.move(str(stray), str(ops/'service.py.pre_s2p18_gate'))\n"
    "    out['backup_moved_to']=str(ops/'service.py.pre_s2p18_gate')\n"
    "out['status_before']=run('git','status','--porcelain').stdout.strip()[:300]\n"
    "run('git','add','src/satellite_rl/workbench/service.py')\n"
    "commit=run('git','commit','-m'," + repr(MESSAGE) + ")\n"
    "out['commit_stdout']=(commit.stdout or '').strip()[:200]\n"
    "out['commit_stderr']=(commit.stderr or '').strip()[:200]\n"
    "out['head']=run('git','rev-parse','HEAD').stdout.strip()\n"
    "out['status_after']=run('git','status','--porcelain').stdout.strip()[:200]\n"
    "print(json.dumps(out,ensure_ascii=False))\n"
)
result = ops18.remote_python(BODY, PORT, timeout=600)
lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
if not lines:
    raise RuntimeError(result["stderr"][-400:])
report = json.loads(lines[-1])
ops18.write_json(ops18.LOCAL / "preflight" / "gate_fix_commit_20418.json",
                 {"checked_utc": ops18.utc_now(), "port": PORT, **report})
print(json.dumps(report, ensure_ascii=False, indent=1))
