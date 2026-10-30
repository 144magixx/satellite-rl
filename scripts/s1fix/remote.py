"""Minimal SSH/SFTP helper for the S1-FIX group on 10.160.17.102:20654."""
import os, sys, paramiko

HOST, PORT, USER = "10.160.17.102", int(os.environ.get("S1FIX_PORT", "20654")), "u2025140002"


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=os.environ["BUPT_PASSWORD"],
              allow_agent=False, look_for_keys=False, timeout=20, banner_timeout=30)
    c.get_transport().set_keepalive(20)
    return c


def run(command, timeout=600):
    c = client()
    _, out, err = c.exec_command(command, timeout=timeout)
    o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
    code = out.channel.recv_exit_status()
    c.close()
    return code, o, e


def put(local, remote):
    c = client(); s = c.open_sftp(); s.put(local, remote); s.close(); c.close()


def get(remote, local):
    c = client(); s = c.open_sftp(); s.get(remote, local); s.close(); c.close()


if __name__ == "__main__":
    if sys.argv[1] == "put":
        put(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3])
    else:
        code, o, e = run(sys.argv[1] if len(sys.argv) == 2 else " ".join(sys.argv[1:]))
        sys.stdout.write(o); sys.stderr.write(e); sys.exit(code)
