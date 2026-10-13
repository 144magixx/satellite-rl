"""Linux parent-death and birth-race binding, before any algorithm import."""
import os,sys,signal,ctypes
def bind_parent(variable):
    if sys.platform!='linux':raise ValueError('formal R7 parent guard requires Linux')
    expected=int(os.environ[variable]);libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'parent-death binding failed')
    if os.getppid()!=expected:raise RuntimeError('original owning parent changed; no algorithm call')
