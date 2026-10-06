"""Restart the 20418 workbench with the identical command line, then verify the API."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

PORT = 20418
PID = 81357


def full_command() -> str:
    body = (
        "import json,pathlib\n"
        f"cmd=pathlib.Path('/proc/{PID}/cmdline').read_bytes().split(b'\\x00')\n"
        "print(json.dumps({'argv':[part.decode() for part in cmd if part]}))\n"
    )
    result = ops18.remote_python(body, PORT, timeout=180)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    return json.loads(lines[-1])["argv"] if lines else []


def main() -> None:
    argv = full_command()
    print("原命令行:", argv)
    if not argv:
        raise RuntimeError("未能读取原命令行，停止重启以免用错参数")
    info = ops18.instance(PORT)
    # 读取原进程的工作目录，确保新进程一致
    cwd_body = f"import os;print(os.readlink('/proc/{PID}/cwd'))"
    cwd_result = ops18.remote_python(cwd_body, PORT, timeout=120)
    cwd = cwd_result["stdout"].strip().splitlines()[-1] if cwd_result["stdout"].strip() else f"{info['repo']}/src"
    print("工作目录:", cwd)
    body = (
        "import json,os,signal,subprocess,time,socket,pathlib\n"
        f"pid={PID}\n"
        "os.kill(pid, signal.SIGTERM)\n"
        "for _ in range(30):\n"
        "    time.sleep(1)\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except OSError:\n"
        "        break\n"
        f"log=pathlib.Path({info['ops']!r})/'workbench_restart.log'\n"
        "handle=open(log,'ab',buffering=0)\n"
        f"argv={argv!r}\n"
        f"cwd={cwd!r}\n"
        "env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'\n"
        "proc=subprocess.Popen(argv,cwd=cwd,stdout=handle,stderr=handle,"
        "stdin=subprocess.DEVNULL,env=env,start_new_session=True)\n"
        "ready=False\n"
        "for _ in range(60):\n"
        "    time.sleep(1)\n"
        "    with socket.socket() as sock:\n"
        "        sock.settimeout(2)\n"
        "        if sock.connect_ex(('127.0.0.1',8898))==0:\n"
        "            ready=True\n"
        "            break\n"
        "print(json.dumps({'new_pid':proc.pid,'listening':ready}))\n"
    )
    result = ops18.remote_python(body, PORT, timeout=300)
    print("重启结果:", result["stdout"].strip()[-200:] or result["stderr"][-300:])
    for attempt in range(10):
        time.sleep(5)
        try:
            if ops18.api(PORT, "/catalog", timeout=60):
                print(f"API 正常（第 {attempt + 1} 次探测）")
                return
        except Exception as error:
            print(f"  探测 {attempt + 1}: {str(error)[:70]}")
    print("警告：API 未恢复，请检查 ops/workbench_restart.log")


if __name__ == "__main__":
    main()
