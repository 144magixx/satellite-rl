"""Explicit root-gated train-only feature/statistics preparation, never auto-run."""
from pathlib import Path
import time,json
import numpy as np
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.r6_model_io import qualified_rows,load_target
from satellite_rl.training.run_monitor import atomic_json

def prepare(qualified_ref,gate_ref,output,feature_source_sha):
 from satellite_rl.workbench.r6_model_limits import check
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r6.features import build_features,FeatureStats,record_sha
 import inspect
 if digest(inspect.getsourcefile(build_features))!=feature_source_sha:raise ValueError("frozen feature computation source")
 directory=Path(output)
 if directory.exists():raise ValueError("preserve existing supply attempt; no implicit regenerate")
 directory.mkdir(parents=True);began=time.monotonic();atomic_json(directory/"supply_started.json",{"qualified_ref":qualified_ref,"gate_ref":gate_ref,"began_wall_ns":time.time_ns(),"charged_to_training_activity":True})
 manifest,rows=qualified_rows(qualified_ref,gate_ref);features=[];cache={};pairs=0
 for row in rows:
  check(directory)
  if time.monotonic()-began>=28800:raise RuntimeError("supply activity cap, no training")
  with np.load(pin(row["scene_ref"]),allow_pickle=False) as z:scene=Scene(z["positions"],z["demands"],tuple(map(str,z["aircraft_ids"])),str(z["profile"]))
  identity={k:row[k] for k in ("case_index","case_id","input_sha","role","label_ref_sha")};case=build_features(scene,identity=identity);features.append(case);pairs+=case.cost["NN_ray_pairs"]
  target=load_target(row,case);path=directory/(str(row["case_index"])+".npz");np.savez_compressed(path,values=case.values,neighbors=case.neighbors,aircraft_ids=case.aircraft_ids,input_sha=case.input_sha,target_owners=target.owners,target_anchor_ids=target.anchor_ids,target_slot_ids=target.slot_ids,target_label_ref_sha=row["label_ref_sha"],target_identity_sha=record_sha(row["identity"]));cache[str(row["case_index"])]={"path":str(path),"sha256":digest(path)}
 stats=FeatureStats.fit(features,role="train",qualified_table_sha=record_sha(rows));atomic_json(directory/"normalization.json",stats.to_record());atomic_json(directory/"feature_cache.json",{"role":"train","qualified_table_sha":record_sha(rows),"feature_graph_source_SHA":feature_source_sha,"cases":cache,"charged_supply_seconds":time.monotonic()-began,"NN_ray_pairs":pairs,"actual_feature_builds":len(rows),"raw_labels_generation_calls":0})
 atomic_json(directory/"supply_endpoint.json",{"charged_supply_seconds_including_publication":time.monotonic()-began,"cache_ref":{"path":str(directory/"feature_cache.json"),"sha256":digest(directory/"feature_cache.json")},"normalization_ref":{"path":str(directory/"normalization.json"),"sha256":digest(directory/"normalization.json")},"qualified_ref":qualified_ref,"science_updates":0});return directory
