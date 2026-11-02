"""Common CLI parsing only; worker module is fixed by admitted dispatcher."""
import argparse
import os,time
from pathlib import Path
from .bindings import read_json
from .bindings import verify_private_dependencies
from .runtime import CONTROL
from .stage_admission import verify_stage
def main(execute):
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True)
    a=p.parse_args();deadline=float(os.environ['R14_REQUEST_DEADLINE_MONOTONIC'])
    def bounded():
        if time.monotonic()>=deadline:raise TimeoutError('shared parent hard deadline')
    CONTROL.set(bounded)
    request=read_json({'path':a.request,'sha256':a.sha256,'bytes':a.bytes})
    verify_stage(request)
    from .dispatcher import worker_module
    expected=Path(request['source_root'],*worker_module(request['kind']).split('.')).with_suffix('.py').resolve()
    if Path(execute.__globals__['__file__']).resolve()!=expected:raise ValueError('actual worker differs from admitted fixed kind route')
    verify_private_dependencies(request)
    execute(request)
