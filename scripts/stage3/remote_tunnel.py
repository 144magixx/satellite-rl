"""Local authenticated tunnel for one isolated stage 3 workbench instance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import select
import socketserver
import threading
import time
import sys

sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, paramiko

logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-port", type=int, choices=(20418, 20424), required=True)
    parser.add_argument("--local-port", type=int, required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    args = parser.parse_args()
    allowed = {(20418, 8787, 8772), (20424, 8788, 8773)}
    if (args.ssh_port, args.local_port, args.remote_port) not in allowed:
        raise RuntimeError("Stage3 tunnel port mapping differs from isolated layout")
    lock = threading.Lock()
    state = {"transport": None}

    def note(message: str) -> None:
        print(datetime.now(timezone.utc).isoformat() + " " + message, flush=True)

    def verified_transport():
        with lock:
            transport = state["transport"]
            if transport is not None and transport.is_active():
                return transport
            if transport is not None:
                transport.close()
                state["transport"] = None
            for delay in (0, 1, 2, 4, 8, 15):
                if delay:
                    time.sleep(delay)
                try:
                    state["transport"] = connect(args.ssh_port)
                    return state["transport"]
                except RuntimeError:
                    raise
                except (OSError, EOFError, paramiko.SSHException):
                    continue
            raise ConnectionError("SSH temporarily unavailable after bounded retries")

    def discard(transport):
        with lock:
            if state["transport"] is transport:
                transport.close()
                state["transport"] = None

    class Forwarder(socketserver.BaseRequestHandler):
        def handle(self):
            channel = None
            try:
                for attempt in range(2):
                    transport = verified_transport()
                    try:
                        channel = transport.open_channel(
                            "direct-tcpip", ("127.0.0.1", args.remote_port), self.client_address, timeout=15)
                        break
                    except (OSError, EOFError, paramiko.SSHException):
                        discard(transport)
                        if attempt == 0:
                            time.sleep(1)
                if channel is None:
                    raise ConnectionError("SSH forwarding channel unavailable")
                while True:
                    ready, _, _ = select.select([self.request, channel], [], [], 30)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (channel if source is self.request else self.request).sendall(data)
            except (OSError, EOFError, paramiko.SSHException, ConnectionError) as error:
                note("request ended after " + type(error).__name__)
            finally:
                if channel is not None:
                    channel.close()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
        request_queue_size = 64

    while True:
        try:
            verified_transport()
            with Server(("127.0.0.1", args.local_port), Forwarder) as server:
                note(f"Stage3 tunnel ready on local port {args.local_port}")
                server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            return
        except RuntimeError:
            raise
        except (OSError, EOFError, paramiko.SSHException, ConnectionError) as error:
            note("listener retry after " + type(error).__name__)
            time.sleep(15)


if __name__ == "__main__":
    main()
