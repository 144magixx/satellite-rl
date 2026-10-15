"""Reuse the exact epoch marker guard inside child numerical entry boundaries."""
import time
import os
from .runtime import PauseGuard
from .bindings import read_json

def child_stop(request,started):
    epoch=read_json(request['runtime_epoch_ref'])
    own=[*epoch['own_paths'],request['own_child_stop_marker']]
    deadline=float(os.environ['R14_REQUEST_DEADLINE_MONOTONIC'])
    stop_new=float(os.environ['R14_REQUEST_STOP_NEW_MONOTONIC'])
    outer=(deadline-stop_new)/.1
    guard=PauseGuard(own,epoch['global_paths'],epoch['historical_exact_refs'],seconds=outer)
    guard.started=deadline-outer
    last=[0.,False]
    def stopped(force=False):
        now=time.monotonic()
        if now>=stop_new:return True
        if not force and now-last[0]<.02:return last[1]
        last[0]=now
        try:guard.check()
        except (RuntimeError,OSError):last[1]=True;return True
        return False
    stopped.fresh=lambda:stopped(force=True)
    return stopped
