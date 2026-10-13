"""Lossless compressed canonical JSON assets with original byte SHA."""
import gzip,json,hashlib,os,shutil
from pathlib import Path
from .coverage_ai_search import pin
from .storage import clean_json,digest

def write_asset(path,value,max_bytes=268435456):
 path=Path(path);raw=json.dumps(clean_json(value),ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
 if len(raw)>max_bytes:raise ValueError("bounded percase publication size exceeded")
 encoded=gzip.compress(raw,compresslevel=6,mtime=0)
 if shutil.disk_usage(path.parent).free < 8589934592+len(encoded):raise OSError("8GiB archive reserve reached")
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(encoded);os.replace(tmp,path)
 return {"path":str(path),"sha256":digest(path),"canonical_json_sha256":hashlib.sha256(raw).hexdigest(),"uncompressed_bytes":len(raw),"compressed_bytes":len(encoded),"compression":"gzip"}
def read_asset(ref):
 if not 0<=ref["uncompressed_bytes"]<=268435456:raise ValueError("uncompressed asset bound")
 path=pin(ref);raw=gzip.decompress(path.read_bytes())
 if len(raw)!=ref["uncompressed_bytes"] or hashlib.sha256(raw).hexdigest()!=ref["canonical_json_sha256"]:raise ValueError("lossless asset identity differs")
 return json.loads(raw)

def pin_tree(value,seen=None):
 seen=set() if seen is None else seen
 if isinstance(value,dict):
  if "path" in value and "sha256" in value:
   key=(value["path"],value["sha256"])
   if key in seen:return
   seen.add(key);path=pin(value)
   if value.get("compression")=="gzip":pin_tree(read_asset(value),seen)
   elif path.suffix==".json":pin_tree(json.loads(path.read_text(encoding="utf8")),seen)
  for child in value.values():pin_tree(child,seen)
 elif isinstance(value,list):
  for child in value:pin_tree(child,seen)
