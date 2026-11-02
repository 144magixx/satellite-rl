"""Transport original normal/unknown kernel evidence before returning/raising."""
from pathlib import Path
from .runtime import atomic_bytes,json_bytes

def call_saved(function,args,store,out,refs):
    try:native=function(*args)
    except BaseException as error:
        evidence={'complete':False,'error':repr(error),'partial_kernel':getattr(error,'partial_kernel',None),
                  'partial_ledger':getattr(error,'partial_ledger',None)}
        try:
            ref=store.put_json(evidence);refs.append(ref)
            atomic_bytes(Path(out)/'native_objects.json',json_bytes(refs))
        except BaseException as storage_error:
            atomic_bytes(Path(out)/'unknown_kernel_small_failure.json',json_bytes({'error':repr(error),
                'storage_error':repr(storage_error),'previous_native_refs':refs,'current_kernel_unknown':True}))
        raise
    ref=store.put_json(native);refs.append(ref)
    atomic_bytes(Path(out)/'native_objects.json',json_bytes(refs));return native
