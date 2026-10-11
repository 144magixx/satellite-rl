"""Offline S1-ND12 evidence audit. No RF recalculation or new evaluation.

Aircraft demand, ownership, exact GEO footprints, beam load algebra and declared
TDMA arithmetic are independently checked from frozen inputs and scene shards.
The RF capacities themselves remain the frozen variable_downstream model output.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

from satellite_rl.coverage.nondrl import metrics_all
from satellite_rl.coverage.nondrl.core import layout_sha256
from satellite_rl.coverage.variable_design import Layout
from satellite_rl.coverage.variable_dataset import _loaded_arrays, load_variable_dataset
from scripts.stage1_nondrl12.execution_sources import audit_lineage, manifest_origin, verify_marker


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_close(actual, expected, name, *, atol=1e-6):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError(name + " undefined boundary differs")
    elif not np.allclose(actual, expected, rtol=1e-9, atol=atol, equal_nan=False):
        raise ValueError(name + " differs from original arrays")


class Evidence:
    """Read a job directory or canonical archive receipt, without extraction."""
    def __init__(self, location, verified_volume_cache=None, *, receipt=None):
        self.location = Path(location)
        self.members = {}
        self.prefix = ""
        if self.location.name == "receipt.json":
            receipt = receipt or json.loads(self.location.read_text(encoding="utf-8"))
            if receipt.get("experiment_id") != "S1-ND12":
                raise ValueError("wrong archive experiment")
            self.prefix = f"{receipt['port']}/{receipt['run_id']}/"
            for volume in receipt["volumes"]:
                path = Path(volume.get("canonical_path", self.location.parent / volume["file"]))
                if verified_volume_cache is not None:
                    from .archive import verify_cached_volume
                    verify_cached_volume(path,volume,verified_volume_cache)
                elif path.stat().st_size != volume["bytes"] or digest_file(path) != volume["sha256"]:
                    raise ValueError("archive volume identity differs")
                for member, item in volume["members"].items():
                    if member in self.members:
                        raise ValueError("duplicate cross-volume member")
                    self.members[member] = (path, item)
        elif not self.location.is_dir():
            raise FileNotFoundError(location)

    def names(self):
        if self.members:
            return [name[len(self.prefix):] for name in self.members if name.startswith(self.prefix)]
        return [path.relative_to(self.location).as_posix() for path in self.location.rglob("*") if path.is_file()]

    def read(self, relative):
        if relative.startswith("/") or "\\" in relative or any(p in ("", ".", "..") for p in relative.split("/")):
            raise ValueError("unsafe evidence member")
        if self.members:
            path, expected = self.members[self.prefix + relative]
            with zipfile.ZipFile(path) as archive:
                value = archive.read(self.prefix + relative)
            if len(value) != expected["bytes"] or digest_bytes(value) != expected["sha256"]:
                raise ValueError("archive member SHA differs")
            return value
        path = self.location / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.location.resolve()):
            raise ValueError("evidence path escaped job directory")
        return path.read_bytes()

    def json(self, relative):
        return json.loads(self.read(relative))


def audit_row(row, original_scene):
    """Original mother scene in, scaled demands derived from row multiplier."""
    from copy import deepcopy
    scene = deepcopy(original_scene)
    multiplier = float(row["demand_multiplier"])
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("invalid demand multiplier")
    scene.demands = np.asarray(original_scene.demands) * multiplier
    if row.get("mother_scene_id") != original_scene.scene_id:
        raise ValueError("mother scene identity differs")
    result = row["result"]
    layout = Layout.from_record(result["final_layout"])
    initial = Layout.from_record(result["initial_layout"])
    if layout_sha256(initial) != result["initial_sha256"] or layout_sha256(layout) != result["final_sha256"]:
        raise ValueError("layout SHA differs")
    if len(layout.centers) != len(initial.centers):
        raise ValueError("same-N mechanism track changed beam count")
    exact = metrics_all(scene, layout, result["config"])
    baseline = metrics_all(scene, initial, result["config"])
    metrics = result["metrics_all"]
    for name in ("load_cv_all", "load_cv_positive", "demand_band_fraction", "total_demand_bps",
                 "beam_demand_bps", "square_load_objective", "beam_count", "zero_demand_beams"):
        assert_close(metrics[name], exact[name], name)
    if not all(metrics.get(name) is True for name in ("coverage_100", "geometry_feasible", "unique_assignment", "demand_conserved")):
        raise ValueError("stored hard-constraint flags are not true")
    if exact["load_cv_all"] is not None and exact["load_cv_all"] > baseline["load_cv_all"] + 1e-9:
        raise ValueError("published exact CV worse than shared I0")
    phy = result["physical"]
    beams = phy["beams"]
    if len(beams) != len(layout.centers):
        raise ValueError("physical beam denominator excludes actual beams")
    loads = np.asarray(exact["beam_demand_bps"])
    for b, record in enumerate(beams):
        assert_close(record["beam_id"], b, "beam identity")
        if record["stable_beam_id"] != layout.beam_ids[b]:
            raise ValueError("stable beam identity differs")
        assert_close([record["lat"], record["lon"]], layout.centers[b], "beam center", atol=1e-10)
        assert_close(record["demand_bps"], loads[b], "beam demand")
        assert_close(record["width_deg"], layout.widths[b], "beam width")
        cap = float(record["capacity_bps"])
        if cap < 0 or not np.isfinite(cap):
            raise ValueError("invalid stored beam capacity")
        assert_close(record["carried_bps"], min(cap, loads[b]), "beam carried")
        assert_close(record["si"], min(cap / loads[b], 1.) if loads[b] else None, "beam SI")
    positive = [b for b in beams if b["demand_bps"] > 0]
    expected_beam_carry = sum(b["carried_bps"] for b in beams) / float(loads.sum()) if loads.sum() else None
    assert_close(phy["metrics"]["demand_weighted_carried_fraction"], expected_beam_carry, "whole-beam carried denominator")
    assert_close(phy["metrics"]["beam_mean_si"], float(np.mean([b["si"] for b in positive])) if positive else None, "mean beam SI")
    assert_close(phy["metrics"]["beam_mean_sgm"], float(np.mean([b["sgm"] for b in positive])) if positive else None, "mean stored beam SGM")
    user = phy["aircraft_service"]
    demands = np.asarray(scene.demands)
    capacities = np.asarray(user["capacity_bps"], dtype=float)
    shares = np.asarray(user["airtime_fraction"], dtype=float)
    if capacities.shape != demands.shape or shares.shape != demands.shape:
        raise ValueError("aircraft service omitted aircraft")
    if not np.isfinite(capacities).all() or np.any(capacities < 0) or not np.isfinite(shares).all() or np.any(shares < 0):
        raise ValueError("invalid aircraft TDMA arrays")
    if "full_time_rate_bps" in user:
        assert_close(capacities, np.asarray(user["full_time_rate_bps"]) * shares, "TDMA rate times airtime")
    for b, record in enumerate(beams):
        ids = layout.assignment == b
        expected_shares = demands[ids] / loads[b] if loads[b] and record["slots"] > 0 else np.zeros(ids.sum())
        assert_close(shares[ids], expected_shares, "demand-proportional beam TDMA")
    carried = np.minimum(capacities, demands)
    if "carried_bps" in user:
        assert_close(user["carried_bps"], carried, "whole-aircraft carried sum")
    total = float(demands.sum())
    carry = float(carried.sum() / total) if total else None
    assert_close(user["demand_weighted_carried_fraction"], carry, "all-demand aircraft denominator")
    if "si" in user:
        expected_si = [min(float(cap / demand), 1.) if demand else None for cap, demand in zip(capacities, demands)]
        for actual, expected in zip(user["si"], expected_si):
            assert_close(actual, expected, "aircraft SI")
    initial_carry = result["initial_physical"]["aircraft_service"]["demand_weighted_carried_fraction"]
    if carry is not None and carry < initial_carry - result["config"]["carry_tolerance"] - 1e-9:
        raise ValueError("published aircraft carry below shared I0 protection")
    power = sum(float(record["power_w"]) for record in beams)
    assert_close(phy["metrics"]["total_power_w"], power, "full-scene power sum")
    if not np.isfinite(power) or power > 3000 + 1e-7 or power < 0:
        raise ValueError("full-scene power violation")
    for name, seconds in result["timing"].items():
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid measured monotonic timing " + name)
    raw_feasible, raw_cv, raw_reason = None, None, None
    if result.get("raw_layout") is not None:
        try:
            raw_metrics = metrics_all(scene, Layout.from_record(result["raw_layout"]), result["config"])
            raw_feasible = raw_metrics["beam_count"] == len(initial.centers)
            raw_cv = raw_metrics["load_cv_all"]
        except ValueError as error:
            raw_feasible, raw_reason = False, str(error)
    return {"row_key": row["key"], "mother_scene_id": row["mother_scene_id"],
            "dataset_index": row["dataset_index"], "initial_cv": baseline["load_cv_all"],
            "initial_carry": initial_carry,
            "multiplier": multiplier, "N": len(layout.centers), "load_cv_all": exact["load_cv_all"],
            "load_cv_positive": exact["load_cv_positive"], "demand_band_fraction": exact["demand_band_fraction"],
            "aircraft_carry": carry, "beam_carry": phy["metrics"]["demand_weighted_carried_fraction"],
            "beam_mean_si": phy["metrics"]["beam_mean_si"], "beam_mean_sgm": phy["metrics"]["beam_mean_sgm"],
            "all_zero_demand": total == 0, "hard_feasible": True,
            "total_power_w": power, "initial_sha256": result["initial_sha256"],
            "raw_proposal_present": result.get("raw_layout") is not None,
            "raw_geometry_feasible": raw_feasible, "raw_load_cv_all": raw_cv,
            "raw_rejection_reason": raw_reason,
            "published_I0": result["final_sha256"] == result["initial_sha256"],
            "timing": result["timing"], "task_seconds": row.get("measured_task_seconds"),
            "shared_initial_seconds": row.get("shared_initial_seconds"),
            "total_seconds": row.get("end_to_end_segment_sum_seconds", row.get("measured_task_seconds")),
            "calls": result["calls"], "route": row.get("route"), "method": row.get("method")}


def dataset_for(evidence, config, role, path_map=None):
    source = config["datasets"][role]
    path = source["path"]
    for old, new in (path_map or {}).items():
        if path.startswith(old):
            if isinstance(new,dict):
                other=Evidence(new["receipt"])
                relative=new.get("member",new.get("member_prefix","")+path[len(old):].lstrip("/"))
                value=other.read(relative)
                if digest_bytes(value)!=source["sha256"]:
                    raise ValueError("canonical referenced dataset SHA differs")
                return _loaded_arrays(BytesIO(value))
            path = new + path[len(old):]
            break
    local = Path(path)
    if local.is_file():
        value = local.read_bytes()
        if digest_bytes(value) != source["sha256"]:
            raise ValueError("frozen input dataset SHA differs")
        return load_variable_dataset(local)
    relative = f"scene_inputs/{role}/{Path(source['path']).name}"
    if relative not in evidence.names():
        relative = f"scene_inputs/{role}/inputs.npz"
    value = evidence.read(relative)
    if digest_bytes(value) != source["sha256"]:
        raise ValueError("archived scene dataset SHA differs")
    return _loaded_arrays(BytesIO(value))


def supervision_sources(evidence, job):
    """Close wrapper, immutable supervision shards and completed problem files."""
    wrapper_bytes=evidence.read("teacher_manifest.json")
    wrapper=json.loads(wrapper_bytes)
    raw=evidence.read("teacher/manifest.json")
    if digest_bytes(raw)!=wrapper.get("sha256"):
        raise ValueError("supervision manifest SHA differs")
    manifest=json.loads(raw)
    if (manifest.get("format")!="s1_nondrl_teacher_v1" or
            manifest.get("provenance",{}).get("run_id")!=job["run_id"] or
            manifest.get("provenance",{}).get("run_id")!=job["run_id"]):
        raise ValueError("supervision source/run provenance differs")
    manifest_origin(job, manifest)
    verified={"teacher_manifest.json":digest_bytes(wrapper_bytes),"teacher/manifest.json":digest_bytes(raw)}
    for item in manifest.get("shards",[])+manifest.get("problem_receipts",[]):
        relative="teacher/"+item["path"];payload=evidence.read(relative)
        if digest_bytes(payload)!=item["sha256"]:
            raise ValueError("supervision source shard/problem SHA differs")
        if "rows" in item and len([line for line in gzip.decompress(payload).splitlines() if line.strip()])!=item["rows"]:
            raise ValueError("supervision source shard row count differs")
        verified[relative]=digest_bytes(payload)
    if sum(item["rows"] for item in manifest.get("shards",[]))!=manifest.get("rows"):
        raise ValueError("supervision source sample denominator differs")
    return verified


def audit_teacher_probe(evidence, job, config, *, path_map=None):
    """Recompute 60 real offline teacher references against frozen mother input."""
    route = config["coverage"]["nondrl"]["route"]
    if route not in ("02", "03") or job.get("preparation_mode") != "teacher_probe":
        raise ValueError("offline teacher proof is only defined for actual02/03 teacher_probe tasks")
    wrapper_bytes = evidence.read("teacher_manifest.json")
    wrapper = json.loads(wrapper_bytes)
    manifest_bytes = evidence.read("teacher/manifest.json")
    if digest_bytes(manifest_bytes) != wrapper.get("sha256"):
        raise ValueError("offline teacher wrapper/evidence manifest SHA differs")
    manifest = json.loads(manifest_bytes)
    if manifest.get("route") != route or manifest.get("status") not in ("completed", "scientific_stop", "attention_required"):
        raise ValueError("offline teacher manifest route/completion differs")
    provenance = manifest.get("provenance", {})
    manifest_origin(job, manifest)
    if manifest.get("format") != "s1_nondrl_teacher_v1":
        raise ValueError("offline teacher evidence format differs")
    for shard in manifest.get("shards", []):
        raw = evidence.read("teacher/" + shard["path"])
        if digest_bytes(raw) != shard["sha256"]:
            raise ValueError("offline teacher supervision shard SHA differs")
        if len([line for line in gzip.decompress(raw).splitlines() if line.strip()]) != shard["rows"]:
            raise ValueError("offline teacher supervision shard rows differ")
    if sum(shard["rows"] for shard in manifest.get("shards", [])) != manifest.get("rows"):
        raise ValueError("offline teacher supervision total row count differs")
    dataset = dataset_for(evidence, config, "development", path_map)
    source = config["datasets"]["development"]
    if wrapper.get("dataset_sha256") != source["sha256"]:
        raise ValueError("offline teacher development dataset differs")
    index_by_id = {scene.scene_id: index for index, scene in enumerate(dataset.scenes)}
    expected = {(scene.scene_id,float(m)) for scene in dataset.scenes for m in config["evaluation"]["demand_multipliers"]}
    receipts = manifest.get("problem_receipts", [])
    if len(receipts) != len(expected):
        raise ValueError("offline teacher full-problem denominator incomplete")
    cap = {"02": (60,38400,600.), "03": (60,7680,240.)}[route]
    if (manifest.get("problems",0) != len(expected) or type(manifest.get("trials")) is not int
            or not 0 <= manifest["trials"] <= cap[1] or len(receipts)>cap[0]
            or not np.isfinite(manifest.get("elapsed_seconds",-1)) or manifest.get("elapsed_seconds",-1)<0):
        raise ValueError("offline teacher reserved problem/trial ledger differs")
    rows, identities, candidate_calls = [], set(), 0
    for receipt in receipts:
        raw = evidence.read("teacher/" + receipt["path"])
        if digest_bytes(raw) != receipt["sha256"]:
            raise ValueError("offline teacher complete problem SHA differs")
        problem = json.loads(gzip.decompress(raw))
        if problem.get("format") != "s1_nondrl_teacher_problem_v1" or problem.get("route") != route:
            raise ValueError("offline teacher problem schema/route differs")
        if problem.get("provenance") != provenance or problem.get("target_n") != len(problem["initial_layout"]["centers"]):
            raise ValueError("offline teacher problem provenance/targetN differs")
        execution = verify_marker(job, problem.get("execution_source")) if job.get("source_repair_contract") else job["git_commit"]
        elapsed = problem.get("teacher_elapsed_seconds",-1)
        calls = problem.get("calls",{}).get("candidates",-1)
        if not np.isfinite(elapsed) or elapsed<0 or type(calls) is not int or calls<0:
            raise ValueError("offline teacher raw call/time ledger invalid")
        candidate_calls += calls
        mother, multiplier = problem["mother_id"], float(problem["demand_multiplier"])
        if receipt.get("mother_id") != mother or float(receipt.get("demand_multiplier",-1)) != multiplier:
            raise ValueError("offline teacher receipt identity differs from its raw problem")
        if (mother,multiplier) not in expected or (mother,multiplier) in identities:
            raise ValueError("offline teacher problem mother/tier duplicate or foreign")
        identities.add((mother,multiplier))
        original = dataset.scenes[index_by_id[mother]]
        assert_close(problem["scene_positions"], original.positions, "teacher positions", atol=1e-10)
        assert_close(problem["scene_demands_bps"], original.demands*multiplier, "teacher whole-aircraft demands")
        if tuple(problem["aircraft_ids"]) != tuple(original.aircraft_ids) or problem["source_hash"] != original.source_hash:
            raise ValueError("offline teacher aircraft/source identity differs")
        common = {"initial_layout": problem["initial_layout"], "initial_sha256": problem["initial_sha256"],
                  "config": problem["config"], "initial_physical": problem["initial_physical"],
                  "timing": problem["timing"], "calls": problem["calls"], "raw_layout": None}
        row = {"key": f"{index_by_id[mother]}:{multiplier:g}", "run_id": job["run_id"], "dataset_index": index_by_id[mother],
               "mother_scene_id": mother, "demand_multiplier": multiplier, "route": route, "method": "offline_teacher_probe",
               "measured_task_seconds": problem["teacher_elapsed_seconds"]}
        for label, layout_key, metrics_key, physical_key in (("initial","initial_layout","initial_metrics_all","initial_physical"),
                                                           ("final","final_layout","final_metrics_all","final_physical")):
            current = problem[layout_key]
            row["result"] = {**common, "final_layout": current, "final_sha256": layout_sha256(Layout.from_record(current)),
                             "metrics_all": problem[metrics_key], "physical": problem[physical_key]}
            audited = audit_row(row,original)
            if label == "final":
                audited.update(problem_receipt_path="teacher/"+receipt["path"], problem_receipt_sha256=receipt["sha256"],
                               teacher_elapsed_seconds=problem["teacher_elapsed_seconds"],
                               search_scope=problem.get("search_scope", {}), completion_reason=problem.get("completion_reason"),
                               execution_commit=execution)
                scope = problem.get("search_scope", {})
                audited.update(searched=scope.get("searched", calls > 0),
                               reference_only=scope.get("reference_only", False),
                               unknown_search=(scope.get("unknown", calls == 0) or scope.get("complete_search") is False))
                if audited["reference_only"] and (calls or elapsed != 0 or scope.get("trials")):
                    raise ValueError("I0-only reference improperly consumed search budget")
                rows.append(audited)
    if candidate_calls > manifest["trials"]:
        raise ValueError("offline teacher true-trial ledger smaller than raw charged candidates")
    return {"rows": rows, "wrapper_sha256": digest_bytes(wrapper_bytes), "evidence_manifest_sha256": digest_bytes(manifest_bytes),
            "dataset_sha256": source["sha256"], "true_trials": manifest["trials"],
            "complete_problems": len(receipts), "raw_candidate_calls": candidate_calls,
            "teacher_recorded_seconds": manifest["elapsed_seconds"],
            "teacher_overrun_seconds": max(0.,manifest["elapsed_seconds"]-cap[2]),
            "searched_problems": sum(bool(r["searched"]) for r in rows),
            "reference_only_problems": sum(bool(r["reference_only"]) for r in rows),
            "unknown_search_problems": sum(bool(r["unknown_search"]) for r in rows),
            "legacy_problems_without_explicit_scope":sum(not r["search_scope"] for r in rows),
            "explicit_complete_search_problems":sum(r["search_scope"].get("searched") is True and
                r["search_scope"].get("complete_search") is True for r in rows),
            "execution_segments": manifest.get("execution_segments", []),
            "scope": "actual complete offline teacher references; zero new teacher/PHY calls"}


def audit_job(location, *, path_map=None, verified_volume_cache=None):
    evidence = Evidence(location,verified_volume_cache)
    job, config, status = (evidence.json(name) for name in ("job.json", "config.json", "status.json"))
    if job.get("experiment_id") != "S1-ND12" or status.get("run_id", job["run_id"]) != job["run_id"]:
        raise ValueError("wrong task identity")
    lineage = audit_lineage(job, evidence) if job.get("source_repair_contract") else {
        "origin_commit": job["git_commit"], "execution_commit": job["git_commit"], "repaired": False}
    stage = status.get("stage")
    if job.get("preparation_mode") == "teacher_probe" and stage in ("completed", "stopped"):
        proof = audit_teacher_probe(evidence,job,config,path_map=path_map)
        return {"run_id": job["run_id"], "state": "completed" if stage=="completed" else (
                    "scientific_stop" if status.get("scientific_stop") is True else "engineering_stopped"),
                "formal_terminal": True, "kind": "prepare", "preparation_mode": "teacher_probe",
                "route": config["coverage"]["nondrl"]["route"], "dataset_role": "development",
                "git_commit": job["git_commit"], "origin_commit": lineage["origin_commit"],
                "source_lineage": lineage, "teacher_probe": proof, "rows": proof["rows"]}
    if stage == "failed":
        if "failure.json" not in evidence.names():
            raise ValueError("failed task lacks failure evidence")
        return {"run_id": job["run_id"], "state": "failure", "formal_terminal": True,
                "rows": [], "failure": evidence.json("failure.json")}
    if stage == "stopped" and status.get("scientific_stop") is True:
        gate = job.get("reachability_source")
        if job["kind"] == "train" and (gate is None or not isinstance(gate.get("sha256"), str) or len(gate["sha256"]) != 64):
            raise ValueError("scientific stop lacks pinned development gate")
        if "results.json" not in evidence.names():
            raise ValueError("scientific stop lacks actual result evidence")
        result = evidence.json("results.json")
        if not (result.get("status") == "scientific_stop" or result.get("scientific_stop") is True):
            raise ValueError("scientific stop result contradicts terminal state")
        if not result.get("stop_reason") and not status.get("stop_reason"):
            raise ValueError("scientific stop has no recorded reason")
        if gate is not None:
            name = "input_sources/reachability_source/0/" + Path(gate["path"]).name
            raw_gate = evidence.read(name)
            if digest_bytes(raw_gate) != gate["sha256"]:
                raise ValueError("scientific stop frozen gate SHA differs")
            gate_record = json.loads(raw_gate)
            if gate_record.get("experiment_id") != "S1-ND12" or gate_record.get("passed") is not True:
                raise ValueError("training scientific stop has no actual passed development source")
        return {"run_id": job["run_id"], "state": "scientific_stop", "formal_terminal": True,
                "gate_source": gate, "stop_reason": status.get("stop_reason"), "rows": []}
    partial_engineering = stage == "stopped" and lineage["repaired"] and status.get("scientific_stop") is not True
    if partial_engineering and job.get("kind") != "evaluate":
        return {"run_id":job["run_id"], "state":"engineering_stopped", "formal_terminal":True,
                "rows":[], "git_commit":job["git_commit"], "origin_commit":lineage["origin_commit"], "source_lineage":lineage}
    if stage != "completed" and not partial_engineering:
        return {"run_id": job["run_id"], "state": stage or "unknown", "formal_terminal": False, "rows": []}
    if job["kind"] in ("train", "prepare") and job.get("preparation_mode") != "initial":
        if "results.json" not in evidence.names():
            raise ValueError("completed non-evaluation task lacks results")
        result = evidence.json("results.json")
        record = {"run_id": job["run_id"], "state": "completed", "formal_terminal": True,
                  "kind": job["kind"], "results": result, "rows": [], "git_commit": job["git_commit"],
                  "origin_commit": lineage["origin_commit"], "source_lineage": lineage}
        if job.get("preparation_mode") in ("teacher","ranking_audit"):
            record.update(supervision_sources=supervision_sources(evidence,job),git_commit=job["git_commit"])
        if job["kind"] == "train":
            if result.get("updates") != config["training"]["updates"]:
                raise ValueError("completed training did not reach frozen update budget")
            losses_name = next((name for name in evidence.names() if name == "models/losses.jsonl"), None)
            if losses_name is None:
                raise ValueError("training archive lacks raw loss curve")
            losses = [json.loads(line) for line in evidence.read(losses_name).splitlines() if line.strip()]
            if [row["update"] for row in losses] != list(range(1, result["updates"] + 1)) or any(not np.isfinite(row["loss"]) for row in losses):
                raise ValueError("training curve incomplete or nonfinite")
            model_name, policy_name = "models/final/model.json", "models/final/policy.pt"
            model = evidence.json(model_name)
            policy_sha = digest_bytes(evidence.read(policy_name))
            if model.get("policy_sha256") != policy_sha or result.get("endpoint_sha256") != policy_sha:
                raise ValueError("fixed endpoint model SHA differs")
            if model.get("updates") != result["updates"] or model.get("selection") != "fixed_budget_endpoint":
                raise ValueError("model selection/update differs")
            record["curve_losses"] = losses
            validation = evidence.json("models/validation.json")
            record["curve_validation"] = validation
            for node in validation:
                prefix = f"validation/update_{node['updates']:06d}/"
                manifest = evidence.json(prefix + "manifest.json")
                dataset = dataset_for(evidence, config, "validation", path_map)
                expected_rows = len(dataset.scenes) * len(config["evaluation"]["demand_multipliers"])
                if len(manifest["rows"]) != expected_rows or node["metrics"].get("completed_scenes") != expected_rows:
                    raise ValueError("fixed validation curve node incomplete")
                measured = []
                for raw in manifest["rows"]:
                    data = evidence.read(prefix + raw["file"])
                    if digest_bytes(data) != raw["sha256"]:
                        raise ValueError("validation scene shard SHA differs")
                    row = json.loads(gzip.decompress(data))
                    measured.append(audit_row(row, dataset.scenes[row["dataset_index"]]))
                values = [m["load_cv_all"] for m in measured if m["load_cv_all"] is not None]
                assert_close(node["metrics"].get("mother_equal_cv_mean"), float(np.mean(values)) if values else None, "validation CV curve vs raw scenes")
        return record
    manifest = evidence.json("raw_scenes/manifest.json")
    if manifest.get("format") != "s1nd12_scene_shards_v1" or manifest["provenance"].get("run_id") != job["run_id"]:
        raise ValueError("scene manifest identity differs")
    manifest_origin(job, manifest)
    role = job["dataset_role"]
    dataset = dataset_for(evidence, config, role, path_map)
    indices = job.get("scene_indices", list(range(len(dataset.scenes))))
    expected = {f"{index}:{multiplier:g}" for index in indices for multiplier in config["evaluation"]["demand_multipliers"]}
    keys = [row["key"] for row in manifest["rows"]]
    if len(keys) != len(set(keys)) or (not set(keys).issubset(expected) if partial_engineering else set(keys) != expected):
        raise ValueError("completed task scene matrix missing or duplicated")
    rows = []
    for record in manifest["rows"]:
        value = evidence.read("raw_scenes/" + record["file"])
        if digest_bytes(value) != record["sha256"]:
            raise ValueError("committed scene shard SHA differs")
        row = json.loads(gzip.decompress(value))
        if row["key"] != record["key"] or row.get("run_id") != job["run_id"]:
            raise ValueError("scene shard task/key differs")
        audited = audit_row(row, dataset.scenes[row["dataset_index"]])
        execution = verify_marker(job, row.get("execution_source")) if job.get("source_repair_contract") else job["git_commit"]
        budget = config["coverage"]["nondrl"].get("limits",row["result"]["config"])["online_seconds"]
        online = row["result"].get("timing", {}).get("online_seconds")
        audited.update(execution_commit=execution, online_budget_seconds=budget,
                       online_overrun_seconds=max(row["result"].get("online_overrun_seconds", 0.),
                                                  max(0., online-budget) if online is not None else 0.))
        rows.append(audited)
    return {"run_id": job["run_id"], "state": "engineering_stopped" if partial_engineering else "completed", "formal_terminal": True,
            "expected_rows":len(expected), "actual_rows":len(rows), "complete_scene_matrix":not partial_engineering,
            "kind": job["kind"], "route": job.get("route", config["coverage"]["nondrl"]["route"]),
            "method": config["coverage"]["nondrl"]["method"], "dataset_role": role,
            "git_commit": job["git_commit"], "origin_commit": lineage["origin_commit"], "source_lineage": lineage,
            "execution_segments": manifest.get("execution_segments", []),
            "rows": rows, "raw_manifest_sha256": digest_bytes(evidence.read("raw_scenes/manifest.json")),
            "config_sha256": manifest["provenance"]["config_sha256"],
            "execution_environment": {"target": job.get("execution_target"), "device": job.get("execution_device"),
                "preflight": status.get("preflight"), "concurrency": status.get("concurrency"),
                "limitation": "missing device/thread/concurrency fields remain missing; no historical zeros filled"},
            "RF_audit_scope": "geometry and recorded TDMA algebra; RF link capacities not independently recalculated"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--path-map", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_job(args.job, path_map=json.loads(args.path_map.read_text(encoding="utf-8")) if args.path_map else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_id": result["run_id"], "state": result["state"], "rows": len(result["rows"]),
                      "formal_terminal": result["formal_terminal"]}))


if __name__ == "__main__":
    main()
