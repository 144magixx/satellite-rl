"""Thin true-role wrappers around immutable b7 preparation/T/D5."""
import argparse,time,sys
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,digest
from .coverage_ai_search import pin
from .r6_io import read_asset,write_asset

def execute(req,directory):
 from satellite_rl.coverage.ai_search.core import Scene
 from satellite_rl.coverage.ai_search_r5.core import prepare_case,PreparedCase,build_teacher,build_proposal,decode_groups
 directory=Path(directory);began=time.monotonic();identity=req["identity"]
 with np.load(pin(req["scene_ref"]),allow_pickle=False) as z:scene=Scene(z["positions"],z["demands"],tuple(str(x) for x in z["aircraft_ids"]),str(z["profile"]))
 if identity["role"]!="train" or scene.input_sha256!=identity["input_sha256"]:raise ValueError("true train identity required")
 manifest=req["phase_manifest"]
 stage=req["stage"];source=req["source_sha256"];started=time.monotonic();proposal_ref=None
 if stage=="preparation":result=prepare_case(scene,identity,source,phase_manifest=manifest,time_limit=10.).to_dict()
 else:
  prepared=PreparedCase.from_dict(scene,read_asset(req["prepared_ref"])["result"])
  if prepared.identity!=identity or prepared.source_sha256!=source:raise ValueError("original prepared lineage")
  if stage=="teacher":
   refs=[]
   def accepted(snapshot):
    target=directory/("accepted_%05d.json.gz"%len(refs));refs.append(write_asset(target,{"identity":identity,"snapshot":snapshot}));atomic_json(directory/"accepted_snapshot_index.json",{"accepted_refs":refs,"identity":identity,"source_sha256":source,"partial_not_terminal":True})
   result=build_teacher(prepared,phase_manifest=manifest,time_limit=30.,on_accepted=accepted)
  elif stage=="decoder":
   teacher=read_asset(req["teacher_ref"])["result"]
   proposal=build_proposal(prepared,"EXACT",teacher=teacher,producer_sha256=req["producer_sha256"],evidence_refs=[req["teacher_ref"]])
   proposal_ref=write_asset(directory/"proposal.json.gz",proposal)
   result=decode_groups(prepared,proposal,time_limit=5.)
  else:raise ValueError("labels stage only")
 value={"stage":stage,"identity":identity,"source_sha256":source,"result":result,"proposal_ref":proposal_ref,"algorithm_seconds":time.monotonic()-started,"provider_and_algorithm_seconds":time.monotonic()-began,"python":sys.executable}
 asset=write_asset(directory/"stage_output.json.gz",value);atomic_json(directory/"stage_output.json",{"asset_ref":asset,"stage":stage,"identity":identity});return value

def main():
 p=argparse.ArgumentParser();p.add_argument("--request",required=True);p.add_argument("--directory",required=True);a=p.parse_args();execute(read_json(a.request,{}),a.directory)
if __name__=="__main__":main()
