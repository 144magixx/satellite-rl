"""Persist measured setup/input/audit/publish scopes even on failure; no fictitious exit time."""
import time
from contextlib import contextmanager
from .contracts import publish

@contextmanager
def measured_scope(path,name,clock=time.monotonic):
 began=clock();error=None
 try:yield
 except BaseException as exc:error=type(exc).__name__+': '+str(exc);raise
 finally:
  publish(path,dict(scope=name,observed_seconds_to_finally=clock()-began,error=error,receipt_IO_and_process_exit_excluded=True,nested_scopes_not_additive=True,complete_worker_exit_cost_seconds=None),exclusive=True)
