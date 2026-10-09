"""Pinned local loopback forwarding for one S1-D12A workbench lane."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import select
import socketserver
import threading
import time

import paramiko

from scripts.stage1_drl_r04 import ops


logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-port", type=int, choices=ops.SSH_PORTS, required=True)
    args = parser.parse_args()
    api_port = ops.API_PORTS[args.ssh_port]
    lock = threading.Lock()
    connection = {"client": None}

    def note(message):
        print(datetime.now(timezone.utc).isoformat() + " " + message, flush=True)

    def connected():
        with lock:
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
                    return connection["client"].get_transport()
                except (OSError, EOFError, paramiko.SSHException):
                    continue
            raise ConnectionError("S1-D12A SSH forwarding unavailable after bounded retries")

    def discard(transport):
        with lock:
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
                        channel = transport.open_channel(
                            "direct-tcpip", ("127.0.0.1", api_port),
                            self.client_address, timeout=15)
                        break
                    except (OSError, EOFError, paramiko.SSHException):
                        discard(transport)
                        if attempt == 0:
                            time.sleep(1)
                if channel is None:
                    raise ConnectionError("S1-D12A forwarding channel unavailable")
                while True:
                    ready, _, _ = select.select([self.request, channel], [], [], 30)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (channel if source is self.request else self.request).sendall(data)
            except (OSError, EOFError, paramiko.SSHException, ConnectionError) as exc:
                note("request ended after " + type(exc).__name__)
            finally:
                if channel is not None:
                    channel.close()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
        request_queue_size = 64

    while True:
        try:
            connected()
            with Server(("127.0.0.1", api_port), Forwarder) as server:
                note(f"S1-D12A {args.ssh_port} tunnel ready on local {api_port}")
                server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            return
        except (OSError, EOFError, paramiko.SSHException, ConnectionError) as exc:
            note("listener retry after " + type(exc).__name__)
            time.sleep(15)


if __name__ == "__main__":
    main()
