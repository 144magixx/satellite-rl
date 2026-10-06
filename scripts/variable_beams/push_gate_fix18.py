"""Push the corrected endpoint gate to the instance repo without restarting the server.

Restarting 20418's workbench while V2's SCA is running could kill that worker, so the
file is updated now and the restart is deferred until the SCA finishes.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

SOURCE = Path(r"D:\paper\src\satellite_rl\workbench\service.py")
TARGET = "src/satellite_rl/workbench/service.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, default=20418)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    info = ops18.instance(args.port)
    local = SOURCE.read_bytes()
    digest = ops18.sha256_file(SOURCE)
    remote_path = f"{info['repo']}/{TARGET}"
    if not args.execute:
        print(f"计划：上传 {SOURCE} -> {remote_path}（{len(local)} 字节，sha256={digest[:16]}）")
        return
    # 先备份远端原文件，再原子替换
    body = (
        "import hashlib,json,pathlib,shutil,base64\n"
        f"target=pathlib.Path({remote_path!r})\n"
        "backup=target.with_suffix('.py.pre_s2p18_gate')\n"
        "out={'target':str(target),'existed':target.is_file()}\n"
        "if target.is_file():\n"
        "    out['before_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()\n"
        "    if not backup.exists():\n"
        "        shutil.copy2(target,backup)\n"
        "    out['backup']=str(backup)\n"
        f"payload=base64.b64decode({__import__('base64').b64encode(local).decode()!r})\n"
        "target.write_bytes(payload)\n"
        "out['after_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()\n"
        "print(json.dumps(out))\n"
    )
    result = ops18.remote_python(body, args.port, timeout=600)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"上传失败: {result['stderr'][-400:]}")
    import json as _json
    receipt = _json.loads(lines[-1])
    ok = receipt.get("after_sha256") == digest
    ops18.write_json(ops18.LOCAL / "preflight" / f"gate_fix_{args.port}.json",
                     {"checked_utc": ops18.utc_now(), "local_sha256": digest,
                      "verified": ok, **receipt})
    print(f"已上传并校验：{ok} | 备份 {receipt.get('backup')}")


if __name__ == "__main__":
    main()
