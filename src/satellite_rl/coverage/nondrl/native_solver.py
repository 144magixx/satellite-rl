"""Owned persistent CPU MILP process, with parent-enforced absolute deadlines.

The SciPy/HiGHS time_limit remains a solver option, never the hard clock. Small
socket messages point to local pickle files so a large matrix cannot block IPC.
One child per owner is reused after successful calls and killed/reaped on timeout.
Startup, payload I/O, native execution and cleanup are separately recorded.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import pickle
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
from time import perf_counter, sleep


def _message(sock, value):
    sock.sendall(json.dumps(value, allow_nan=False).encode() + b"\n")


def _receive(sock, deadline):
    data = bytearray()
    while not data.endswith(b"\n"):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("native solver IPC deadline")
        sock.settimeout(remaining)
        block = sock.recv(1)
        if not block:
            raise EOFError("native solver exited during IPC")
        data.extend(block)
        if len(data) > 65536:
            raise ValueError("native solver control packet oversized")
    return json.loads(data)


def _windows_job(process):
    """Kill-on-close job: parent death cannot leave a native CPU child behind."""
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                    ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class Io(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", Io), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    info = Extended(); info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or
            not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(process._handle))):
        if handle:
            kernel.CloseHandle(handle)
        raise OSError("cannot establish native child parent-death job guard")
    return kernel, handle


class NativeSolver:
    def __init__(self, backend="milp"):
        if backend not in ("milp", "controlled_sleep"):
            raise ValueError("unknown native backend")
        self.backend = backend
        self.process = self.channel = self.job = None
        self.root = tempfile.TemporaryDirectory(prefix="satellite_nd12_milp_")
        self.lock = threading.RLock()
        self.last_cleanup = None
        self.sequence = 0
        atexit.register(self.close)

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def start(self, deadline):
        with self.lock:
            if self.process is not None and self.process.poll() is None and self.channel is not None:
                return {"startup_seconds": 0., "worker_pid": self.pid, "reused": True}
            self._stop()
            started = perf_counter()
            token = secrets.token_hex(24)
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0)); listener.listen(1)
            port = listener.getsockname()[1]
            try:
                if deadline <= perf_counter():
                    raise TimeoutError("native child startup budget exhausted")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                    "--child", "--port", str(port), "--token", token, "--parent", str(os.getpid()), "--backend", self.backend],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    close_fds=True, start_new_session=os.name != "nt", creationflags=flags)
                if os.name == "nt":
                    self.job = _windows_job(self.process)
                listener.settimeout(max(.001, deadline - perf_counter()))
                self.channel, _ = listener.accept()
                response = _receive(self.channel, deadline)
                if response.get("token") != token or response.get("pid") != self.pid:
                    raise ValueError("native child identity handshake differs")
                if perf_counter() >= deadline:
                    raise TimeoutError("native child ready after deadline")
                return {"startup_seconds": perf_counter() - started, "worker_pid": self.pid, "reused": False}
            except BaseException:
                self._stop()
                raise
            finally:
                listener.close()

    def _stop(self):
        begin = perf_counter()
        process, self.process = self.process, None
        if self.channel is not None:
            self.channel.close(); self.channel = None
        if process is not None:
            if process.poll() is None:
                process.kill()  # SIGKILL / TerminateProcess, not cooperative native SIGINT.
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process = process
                self.last_cleanup = {"worker_pid": process.pid, "alive_after": True,
                                     "cleanup_seconds": perf_counter() - begin, "error": "hard-killed child not yet reaped"}
                # Do not lose ownership or return a normal fallback while the
                # OS has not confirmed process exit.
                raise RuntimeError("native cleanup did not confirm process exit")
            self.last_cleanup = {"worker_pid": process.pid, "exit_code": process.returncode,
                                 "alive_after": process.poll() is None, "cleanup_seconds": perf_counter() - begin}
        if self.job is not None:
            kernel, handle = self.job; kernel.CloseHandle(handle); self.job = None
        return perf_counter() - begin

    def close(self):
        with self.lock:
            self._stop()
            self.root.cleanup()

    def solve(self, payload, deadline):
        """Return a completed solution or an unknown engineering outcome, never late x."""
        with self.lock:
            begin = perf_counter()
            request = output = None
            timing = {"startup_seconds": 0., "payload_seconds": 0., "ipc_seconds": 0., "cleanup_seconds": 0.}
            try:
                timing.update(self.start(deadline))
                self.sequence += 1
                request = Path(self.root.name) / f"request_{self.sequence}.pkl"
                output = Path(self.root.name) / f"output_{self.sequence}.pkl"
                start = perf_counter()
                with request.open("wb") as stream:
                    pickle.dump(payload, stream, protocol=5)
                timing["payload_seconds"] += perf_counter() - start
                if perf_counter() >= deadline:
                    raise TimeoutError("model payload completed after hard deadline")
                start = perf_counter()
                self.channel.settimeout(max(.001, deadline - start))
                _message(self.channel, {"request": str(request), "output": str(output), "sequence": self.sequence})
                response = _receive(self.channel, deadline)
                timing["ipc_seconds"] = perf_counter() - start
                if response.get("sequence") != self.sequence:
                    raise ValueError("native solver response cursor differs")
                if perf_counter() >= deadline:
                    raise TimeoutError("native response arrived after hard deadline")
                with output.open("rb") as stream:
                    result = pickle.load(stream)
                if perf_counter() >= deadline:
                    raise TimeoutError("native solution decoded after hard deadline")
                request.unlink(); output.unlink()
                timing["wall_seconds"] = perf_counter() - begin
                if result.get("engineering_error"):
                    return {"outcome": "engineering_error", "known": False, "result": None, "timing": timing,
                            "reason": result["engineering_error"]}
                timing["native_seconds"] = result.pop("native_seconds")
                timing["ipc_overhead_seconds"] = max(0., timing["ipc_seconds"] - timing["native_seconds"])
                return {"outcome": "completed", "known": True, "result": result, "timing": timing}
            except (TimeoutError, socket.timeout):
                timing["cleanup_seconds"] = self._stop()
                timing["wall_seconds"] = perf_counter() - begin
                timing["hard_deadline_exceeded_seconds"] = max(0., perf_counter() - deadline)
                return {"outcome": "engineering_timeout", "known": False, "result": None, "timing": timing,
                        "cleanup": self.last_cleanup, "reason": "no completed response before parent hard deadline"}
            except (EOFError, ConnectionError, OSError) as error:
                timing["cleanup_seconds"] = self._stop()
                timing["wall_seconds"] = perf_counter() - begin
                return {"outcome": "engineering_error", "known": False, "result": None, "timing": timing,
                        "cleanup": self.last_cleanup, "reason": repr(error)}
            except BaseException:
                self._stop()
                raise
            finally:
                for path in (request, output):
                    if path is not None:
                        path.unlink(missing_ok=True)


_OWNED = None


def owned_solver():
    global _OWNED
    if _OWNED is None:
        _OWNED = NativeSolver()
    return _OWNED


def _child(port, token, parent, backend):
    if os.name != "nt":
        import ctypes
        import signal
        libc = ctypes.CDLL(None)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
            raise RuntimeError("native parent-death guard could not be established")
    if backend == "milp":
        from scipy.optimize import Bounds, LinearConstraint, milp
    connection = socket.create_connection(("127.0.0.1", port), timeout=10.)
    _message(connection, {"pid": os.getpid(), "token": token})
    while True:
        command = _receive(connection, perf_counter() + 24 * 3600)
        with open(command["request"], "rb") as stream:
            payload = pickle.load(stream)
        begin = perf_counter()
        try:
            if backend == "controlled_sleep":
                sleep(payload.get("delay_seconds", 0.))
                result = {"x": payload.get("x"), "status": 0, "fun": payload.get("fun", 0.),
                          "message": "controlled test backend", "mip_gap": 0., "mip_dual_bound": 0.}
            else:
                answer = milp(payload["objective"], integrality=payload["integrality"],
                    bounds=Bounds(payload["bounds_lower"], payload["bounds_upper"]),
                    constraints=LinearConstraint(payload["matrix"], payload["constraint_lower"], payload["constraint_upper"]),
                    options=payload["options"])
                result = {key: getattr(answer, key, None) for key in ("x", "status", "fun", "message", "mip_gap", "mip_dual_bound")}
            result["native_seconds"] = perf_counter() - begin
        except Exception as error:
            result = {"engineering_error": repr(error)}
        with open(command["output"], "wb") as stream:
            pickle.dump(result, stream, protocol=5)
        _message(connection, {"sequence": command["sequence"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--backend", choices=("milp", "controlled_sleep"), required=True)
    args = parser.parse_args()
    _child(args.port, args.token, args.parent, args.backend)
