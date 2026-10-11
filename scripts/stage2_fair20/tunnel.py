"""Persistent pinned SSH forwarding for one isolated F20 loopback workbench."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import select
import socketserver
import threading
import time

from scripts.stage2_fair20 import ops


def mapping(port):
    if type(port) is not int or port not in ops.SSH_PORTS:
        raise ValueError("F20 forwarding permits only its four approved SSH lanes")
    return {"ssh_port": port, "local_address": "127.0.0.1", "local_port": ops.API_PORTS[port],
            "remote_address": "127.0.0.1", "remote_port": ops.API_PORTS[port]}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def singleton(directory):
    """An OS-held lane lock releases on death; no stale PID deletion is needed."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "tunnel.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    import paramiko
    logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-port", type=int, choices=ops.SSH_PORTS, required=True)
    parser.add_argument("--receipt-root", type=Path, default=ops.LOCAL / "preflight" / "tunnels")
    args = parser.parse_args(argv)
    pin = mapping(args.ssh_port)
    receipt = args.receipt_root.resolve() / str(args.ssh_port)
    connection = {"client": None}
    mutex = threading.Lock()

    def note(stage, **fields):
        value = {"experiment_id": "S2-F20", "pid": os.getpid(), "stage": stage,
                 "updated_utc": datetime.now(timezone.utc).isoformat(), **pin, **fields}
        atomic(receipt / "status.json", value)
        print(value["updated_utc"] + " " + stage, flush=True)

    def connected():
        with mutex:
            client = connection["client"]
            if client is not None and client.get_transport() and client.get_transport().is_active():
                return client.get_transport()
            if client is not None:
                client.close()
                connection["client"] = None
            for delay in (0, 1, 2, 4, 8, 15):
                if delay:
                    time.sleep(delay)
                try:
                    connection["client"] = ops.connect(args.ssh_port)
                    note("connected")
                    return connection["client"].get_transport()
                except (OSError, EOFError, paramiko.SSHException):
                    continue
            raise ConnectionError("pinned F20 SSH unavailable after bounded retries")

    def discard(transport):
        with mutex:
            client = connection["client"]
            if client is not None and client.get_transport() is transport:
                client.close()
                connection["client"] = None

    class Forwarder(socketserver.BaseRequestHandler):
        def handle(self):
            channel = None
            try:
                for attempt in range(2):
                    transport = connected()
                    try:
                        channel = transport.open_channel("direct-tcpip", (pin["remote_address"], pin["remote_port"]),
                                                         self.client_address, timeout=15)
                        break
                    except (OSError, EOFError, paramiko.SSHException):
                        discard(transport)
                        if attempt == 0:
                            time.sleep(1)
                if channel is None:
                    raise ConnectionError("F20 forwarding channel unavailable")
                while True:
                    ready, _, _ = select.select([self.request, channel], [], [], 30)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (channel if source is self.request else self.request).sendall(data)
            except (OSError, EOFError, paramiko.SSHException, ConnectionError) as error:
                # Report only exception type; authentication material stays in the environment.
                print(datetime.now(timezone.utc).isoformat() + " request ended: " + type(error).__name__, flush=True)
            finally:
                if channel is not None:
                    channel.close()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = False
        daemon_threads = True
        request_queue_size = 64

    with singleton(receipt):
        # Bind once, before SSH. Occupied ports fail closed instead of leaving
        # a second daemon repeatedly trying to take another owner's listener.
        with Server((pin["local_address"], pin["local_port"]), Forwarder) as server:
            note("listening")
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .5}, daemon=True)
            thread.start()
            try:
                while True:
                    try:
                        connected()
                        note("ready")
                    except (OSError, EOFError, paramiko.SSHException, ConnectionError) as error:
                        note("reconnecting", error_type=type(error).__name__)
                    time.sleep(15)
            except KeyboardInterrupt:
                return
            finally:
                server.shutdown()
                client = connection["client"]
                if client is not None:
                    client.close()


if __name__ == "__main__":
    main()
