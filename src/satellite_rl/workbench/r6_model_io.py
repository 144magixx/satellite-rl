"""Signed model-phase artifact boundaries; no implicit held-out loading."""
from pathlib import Path
import json
import numpy as np
from .coverage_ai_search import pin
from .storage import read_json,digest
from .r6_io import read_asset,pin_tree

def signed_json(ref):return read_json(pin(ref),{})
def qualified_rows(ref,gate_ref):
 m=signed_json(ref);gate=signed_json(gate_ref)
 if m.get("schema")!="R6QualifiedTrainManifestV1" or m.get("role")!="train" or not gate.get("passed") or gate.get("phase")!="labels" or m.get("labels_gate_sha256")!=gate_ref["sha256"]:raise ValueError("root qualified train gate required")
 if gate.get("qualified",0)<692 or gate.get("nontrivial",0)<154 or gate.get("capable_mothers",0)<6:raise ValueError("original label scientific gate")
 from scripts.stage1_ai_search_r6.plan_builder import selected_indices
 rows=m["qualified_cases"];indices=[r["case_index"] for r in rows]
 if not 692<=len(rows)<=768 or any(type(i)is not int or i not in set(selected_indices()) for i in indices) or indices!=sorted(set(indices)) or any(r.get("role")!="train" or r.get("label_ref_sha")!=r["original_T_terminal_ref"]["sha256"] or not r.get("input_sha") or not r.get("case_id") for r in rows):raise ValueError("immutable qualified original T table")
 if m.get("source_dataset_sha256")!="7bc2b1f8fdd583e136621e3d25bf8aed371fa72e8d4a399f21f4ed3cd27655af":raise ValueError("training dataset source SHA")
 return m,rows

def load_feature(row,cache):
 from satellite_rl.coverage.ai_search_r6.features import FeatureCase
 ref=cache["cases"][str(row["case_index"])];path=pin(ref)
 with np.load(path,allow_pickle=False) as z:
  ids=tuple(map(str,z["aircraft_ids"]));values=z["values"].copy();neighbors=z["neighbors"].copy();input_sha=str(z["input_sha"].item())
 if input_sha!=row["input_sha"]:raise ValueError("feature input differs")
 identity={k:row[k] for k in ("case_index","case_id","input_sha","role","label_ref_sha")}
 return FeatureCase(values,neighbors,ids,input_sha,{"offline_feature_cache_ref":ref,"geometry_calls_in_supply":0},identity)
def load_target(row,features):
 from satellite_rl.coverage.ai_search_r6.learning import TeacherTarget
 value=read_asset(row["original_T_terminal_ref"]);t=value["result"]
 expected=row["identity"]
 if value["identity"]!=expected or expected["role"]!="train" or expected["input_sha256"]!=row["input_sha"] or t.get("status")!="available" or any(t.get(k)!=v for k,v in expected.items()):raise ValueError("original T terminal source/role")
 layout=t["layout"];target=TeacherTarget(features.aircraft_ids,np.asarray(layout["assignment"]),tuple(layout["anchor_aircraft_ids"]),tuple(layout["beam_ids"]),identity=features.identity,label_ref_sha=row["label_ref_sha"]);target.ordered();return target

def validate_cache(cache,stats,rows,model_sources):
 from satellite_rl.coverage.ai_search_r6.features import record_sha
 if cache.get("role")!="train" or cache.get("qualified_table_sha")!=record_sha(rows) or stats.qualified_table_sha!=record_sha(rows) or cache.get("feature_graph_source_SHA")!=model_sources["feature_graph_source_SHA"] or set(cache["cases"])!=set(str(r["case_index"]) for r in rows):raise ValueError("normalization/cache exact training inputs")
 charge=cache.get("charged_supply_seconds")
 if not isinstance(charge,(int,float)) or not np.isfinite(charge) or charge<0:raise ValueError("offline supply activity missing")
 return float(charge)

def load_cached_target(row,features,cache):
 from satellite_rl.coverage.ai_search_r6.learning import TeacherTarget
 from satellite_rl.coverage.ai_search_r6.features import record_sha
 ref=cache['cases'][str(row['case_index'])]
 with np.load(pin(ref),allow_pickle=False) as z:
  if str(z['target_label_ref_sha'].item())!=row['label_ref_sha'] or str(z['target_identity_sha'].item())!=record_sha(row['identity']) or tuple(map(str,z['aircraft_ids']))!=features.aircraft_ids:raise ValueError('compact original T cache source/role/order binding')
  target=TeacherTarget(features.aircraft_ids,z['target_owners'].copy(),tuple(map(str,z['target_anchor_ids'])),tuple(map(str,z['target_slot_ids'])),identity=features.identity,label_ref_sha=row['label_ref_sha'])
 target.ordered();return target

def model_source_identity():
 root=Path(__file__).resolve().parents[1]
 names={'model_graph_source_SHA':'coverage/ai_search_r6/model.py','feature_graph_source_SHA':'coverage/ai_search_r6/features.py','learning_source_SHA':'coverage/ai_search_r6/learning.py','adapter_source_SHA':'coverage/ai_search_r6/ai_adapter.py','b7_decoder_source_SHA':'coverage/ai_search_r5/core.py','runtime_source_SHA':'coverage/ai_search_r6/runtime_v3.py'}
 return {key:digest(root/name) for key,name in names.items()}

def verify_model_source_equivalence(ref,engineering_ref,initial):
 record=signed_json(ref);actual=model_source_identity()
 if record.get('passed')is not True or record.get('accepted_engineering_admission_sha256')!=engineering_ref['sha256'] or record.get('immutable_model_source_sha256')!=actual:raise ValueError('root accepted engineering/source equivalence differs from actual model/learning/adapter/core/runtime code')
 if any(actual[k]!=initial[k] for k in ('model_graph_source_SHA','feature_graph_source_SHA')) or actual['runtime_source_SHA']!=initial['training_runtime_identity']['runtime_source_sha256']:raise ValueError('accepted engineering original graph/runtime source mismatch')
 return actual
