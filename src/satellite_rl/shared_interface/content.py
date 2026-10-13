"""R7 typed canonical content hashing, separate from seeded random digests."""
import hashlib
import json
import numpy as np

ARRAY_DTYPES={'lat_lon_deg':'<f8','centres_lat_lon_deg':'<f8','full_width_deg':'<f8',
    'request_bps_int64':'<i8','request_bps_by_beam_int64':'<i8','aircraft_count_by_beam':'<i8',
    'processing_indices':'<i8','actual_group':'<i8','actual_start_slot':'<i8','actual_slot_count':'<i8',
    'actual_whole_beam_power_w':'<f8'}


def declared_array(value,name):
    dtype=ARRAY_DTYPES[name];raw=np.asarray(value)
    if dtype=='<i8' and raw.dtype.kind not in 'iu':raise ValueError('integer dtype required '+name)
    if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():raise ValueError('finite numeric array '+name)
    if dtype=='<i8' and raw.dtype.kind=='u' and (raw>np.iinfo(np.int64).max).any():raise ValueError('int64 overflow')
    return np.ascontiguousarray(raw,dtype=dtype)


def typed_tree(value):
    if isinstance(value,np.ndarray):
        if value.dtype.kind not in 'if' or value.dtype.itemsize!=8 or not np.isfinite(value).all():raise ValueError('canonical arrays require finite i8/f8')
        array=np.asarray(value,dtype='<i8' if value.dtype.kind=='i' else '<f8')
        if not array.flags.c_contiguous:array=np.ascontiguousarray(array)
        return dict(dtype=array.dtype.str,shape=list(array.shape),data_hex=array.tobytes().hex())
    if isinstance(value,(float,np.floating)):
        if not np.isfinite(value):raise ValueError('nonfinite scalar')
        array=np.asarray(value,dtype='<f8')
        return dict(dtype='<f8',shape=[],data_hex=array.tobytes().hex())
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,np.integer)):return int(value)
    if isinstance(value,str):value.encode('utf8',errors='strict');return value
    if value is None:return None
    if isinstance(value,(list,tuple)):return [typed_tree(item) for item in value]
    if isinstance(value,dict):
        if any(not isinstance(key,str) for key in value):raise ValueError('canonical dictionary keys must be strings')
        return {typed_tree(key):typed_tree(item) for key,item in value.items()}
    raise ValueError('unsupported canonical type '+str(type(value)))


def canonical_bytes(value):
    return json.dumps(typed_tree(value),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')


def content_sha(record):
    return hashlib.sha256(canonical_bytes({k:v for k,v in record.items() if k!='content_sha256'})).hexdigest()


def seal(record):
    value=dict(record);value['content_sha256']=content_sha(value);return value


def verify_self(record):
    if not isinstance(record,dict) or record.get('content_sha256')!=content_sha(record):raise ValueError('content_SHA mismatch')


def _decode(value):
    if isinstance(value,dict) and set(value)=={'dtype','shape','data_hex'}:
        if value['dtype'] not in ('<i8','<f8') or any(type(n)!=int or n<0 for n in value['shape']):raise ValueError('typed array format')
        raw=bytes.fromhex(value['data_hex']);shape=tuple(value['shape']);count=int(np.prod(shape,dtype=object)) if shape else 1
        if len(raw)!=count*8:raise ValueError('typed array byte length')
        array=np.frombuffer(raw,dtype=value['dtype']).reshape(shape).copy()
        if not np.isfinite(array).all():raise ValueError('nonfinite wire array')
        return float(array) if not shape and value['dtype']=='<f8' else array
    if isinstance(value,dict):return {k:_decode(v) for k,v in value.items()}
    if isinstance(value,list):return [_decode(v) for v in value]
    return value


def wire_bytes(record):return canonical_bytes(record)


def read_wire(raw):
    value=_decode(json.loads(raw.decode('utf8') if isinstance(raw,bytes) else raw))
    verify_self(value);return value
