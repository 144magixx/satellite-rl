"""Read-only train-only input identities. No geometry/search/model execution."""
import hashlib,json
import numpy as np
from .plan_builder import selected_indices,TRAIN_SHA

def aircraft_sha(ids):return hashlib.sha256(json.dumps(list(map(str,ids)),ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def scan(path):
 from pathlib import Path
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=TRAIN_SHA:raise ValueError("exact training input bytes")
 with np.load(path,allow_pickle=False) as z:
  if len(z["case_ids"])!=2304:raise ValueError("training count")
  arrays={k:z[k] for k in ("positions","demands","aircraft_ids","offsets","case_ids","mother_ids","demand_profiles")}
  out={}
  for i in selected_indices():
   a,b=map(int,arrays["offsets"][i:i+2]);ids=tuple(map(str,arrays["aircraft_ids"][a:b]));value={"positions":np.asarray(arrays["positions"][a:b],float).tolist(),"demands":np.asarray(arrays["demands"][a:b],float).tolist(),"aircraft_ids":list(ids),"profile":"europe_v1"};input_sha=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
   out[str(i)]={"case_id":str(arrays["case_ids"][i]),"mother_id":str(arrays["mother_ids"][i]),"profile":str(arrays["demand_profiles"][i]),"role":"train","input_sha256":input_sha,"aircraft_order_sha256":aircraft_sha(ids),"n":b-a}
 return {"dataset_sha256":TRAIN_SHA,"role":"train","case_metadata":out,"cases":768,"mothers":192,"geometry_calls":0,"train_array_read_only":True,"validation_test_opened":False}
