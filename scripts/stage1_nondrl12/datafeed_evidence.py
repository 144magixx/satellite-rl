"""Offline receipt checks for the one approved MLM reader extension.

The base snapshot and learning contract retain a21. Runtime components have
their own exact SHA identity, and never grant a general source exception.
"""
from __future__ import annotations
from io import BytesIO
import hashlib
import json
import math
import sqlite3
import tempfile
from pathlib import Path

RUN_ID="train_nondrl_20261012_055310_890a4cba"
EXTENSION_ID="nd12_01_compact_evidence_v1"
BASE_COMMIT="a21d50ae49f55f1ac2210ba6aaf0778e33f25f8d"
PREFIX="runtime_extensions/"+EXTENSION_ID+"/"


def validate_cache(evidence,contract,origin):
    """Check the full cache index and binary spans without running its provider."""
    import numpy as np
    cache=evidence.json(PREFIX+"cache/manifest.json")
    if (cache.get("format")!="s1_nd12_compact_supervision_v1" or cache.get("route")!="01" or
            cache.get("dataset_sha256")!=contract["teacher_manifest"]["sha256"] or
            cache.get("dataset_sha256")!=origin["contract"].get("dataset_sha") or
            cache.get("source_identity",{}).get("original_manifest_sha256")!=cache["dataset_sha256"] or
            set(cache.get("files",{}))!={"index.sqlite","data.bin"} or cache.get("filtered")!=0):
        raise ValueError("runtime cache dataset/route/raw manifest identity differs")
    expected=[]
    for index,shard in enumerate(cache["source_identity"]["shards"]):
        expected.extend((index,row) for row in range(shard["rows"]))
    if cache.get("rows")!=len(expected) or cache.get("valid_rows")!=len(expected):
        raise ValueError("runtime cache full indexed source denominator differs")
    data=evidence.read(PREFIX+"cache/data.bin")
    index=evidence.read(PREFIX+"cache/index.sqlite")
    for name,payload in (("data.bin",data),("index.sqlite",index)):
        pin=cache["files"][name]
        if len(payload)!=pin["bytes"] or hashlib.sha256(payload).hexdigest()!=pin["sha256"]:
            raise ValueError("runtime cache member byte SHA/length differs")
    with tempfile.TemporaryDirectory(prefix="nd12_cache_audit_") as directory:
        path=Path(directory)/"index.sqlite";path.write_bytes(index)
        connection=sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1",uri=True)
        try:
            records=connection.execute("SELECT shard,row,fields FROM samples ORDER BY shard,row").fetchall()
            if [(r[0],r[1]) for r in records]!=expected:
                raise ValueError("runtime cache index reordered or omitted original samples")
            for _,_,encoded in records:
                for descriptor in json.loads(encoded).values():
                    dtype=np.dtype(descriptor["dtype"]);shape=descriptor["shape"];offset=descriptor["offset"]
                    if (dtype.hasobject or dtype.kind not in ("f","i","b") or
                            any(type(v)is not int or v<0 for v in shape) or type(offset)is not int or offset<0 or
                            offset+math.prod(shape)*dtype.itemsize>len(data)):
                        raise ValueError("runtime cache tensor descriptor escaped its pinned binary")
        finally:connection.close()
    return {"rows":len(expected),"valid_rows":cache["valid_rows"],"filtered":cache["filtered"],
        "files":cache["files"],"canonical_owner":RUN_ID,"source_identity":cache["source_identity"]}


def torch_record(payload):
    import torch
    return torch.load(BytesIO(payload),map_location="cpu",weights_only=False)


def validate_training_artifacts(evidence,job,marker,config,*,completed):
    """Independently compare the actual old and resumed checkpoint contracts."""
    if (job.get("run_id")!=RUN_ID or job.get("git_commit")!=BASE_COMMIT or job.get("kind")!="train" or
            config["coverage"]["nondrl"]["route"]!="01" or config["training"].get("seed")!=51001 or
            config["training"].get("updates")!=10000 or config["training"].get("batch_size")!=32 or
            config["training"].get("microbatch_size")!=4):
        raise ValueError("runtime reader extension is outside the exact MLM budget/config/source")
    contract=evidence.json(PREFIX+"contract.json")
    if (hashlib.sha256(evidence.read("config.json")).hexdigest()!=contract["config"]["sha256"] or
            hashlib.sha256(evidence.read("source/source_manifest.json")).hexdigest()!=contract["base_snapshot"]["sha256"]):
        raise ValueError("extension original configuration/base snapshot SHA differs")
    origin=torch_record(evidence.read(PREFIX+"origin_checkpoint.pt"))
    checkpoint=torch_record(evidence.read("models/checkpoint.pt"))
    start=contract["start_update"]
    if (origin.get("format")!="s1_nondrl_checkpoint_v1" or origin.get("updates")!=start or
            origin.get("total_updates")!=10000 or origin.get("cursor")!=start*32 or
            origin.get("runtime_extensions") or origin.get("contract",{}).get("route")!="01"):
        raise ValueError("original full checkpoint differs from the authorized activation boundary")
    if (origin.get("contract",{}).get("source_sha")!=BASE_COMMIT or
            origin.get("contract",{}).get("source_manifest_sha256")!=contract["base_snapshot"]["sha256"]):
        raise ValueError("original checkpoint base learning source differs")
    for state in (origin,checkpoint):
        if (not state.get("model") or not state.get("optimizer",{}).get("state") or
                set(state.get("rng",{}))!={"python","numpy","torch","cuda","sampler"} or
                not state["rng"]["cuda"] or not isinstance(state.get("validation"),list)):
            raise ValueError("extension checkpoint is missing model/Adam/RNG/fixed-validation state")
    if checkpoint.get("contract")!=origin["contract"] or checkpoint.get("network")!=origin.get("network"):
        raise ValueError("extension changed the frozen learning contract or network")
    if checkpoint["optimizer"].get("param_groups")!=origin["optimizer"].get("param_groups"):
        raise ValueError("extension changed Adam configuration or parameter ordering")
    if checkpoint.get("runtime_extensions")!=[marker] and not (not completed and
            hashlib.sha256(evidence.read("models/checkpoint.pt")).hexdigest()==contract["checkpoint"]["sha256"]):
        raise ValueError("current checkpoint runtime extension marker differs")
    prefix=evidence.read(PREFIX+"origin_losses_prefix.bin")
    loss_pin=contract["loss_prefix"]
    if (len(prefix)!=loss_pin["bytes"] or hashlib.sha256(prefix).hexdigest()!=loss_pin["sha256"] or
            origin.get("loss_bytes")!=len(prefix) or origin.get("loss_sha256")!=loss_pin["sha256"]):
        raise ValueError("original checkpoint/loss prefix differs")
    losses=evidence.read("models/losses.jsonl")
    if not losses.startswith(prefix):raise ValueError("reader activation rewrote the original loss prefix")
    current_prefix=losses[:checkpoint["loss_bytes"]]
    if hashlib.sha256(current_prefix).hexdigest()!=checkpoint["loss_sha256"] or len(current_prefix)!=checkpoint["loss_bytes"]:
        raise ValueError("current full checkpoint loss bytes/hash differ")
    original_rows=[json.loads(line) for line in prefix.splitlines() if line.strip()]
    if ([r["update"] for r in original_rows]!=list(range(1,start+1)) or
            loss_pin["rows"]!=start or any(r.get("cursor")!=r["update"]*32 for r in original_rows)):
        raise ValueError("original full loss prefix update/sample cursor differs")
    expected_nodes=list(range(1000,start+1,1000))
    if [v["updates"] for v in origin["validation"]]!=expected_nodes:
        raise ValueError("reader activation lacks every preceding fixed validation node")
    if [v["updates"] for v in contract["validation_manifests"]]!=expected_nodes:
        raise ValueError("runtime contract preceding validation denominator differs")
    for pin in contract["validation_manifests"]:
        payload=evidence.read(f"validation/update_{pin['updates']:06d}/manifest.json")
        saved=json.loads(payload)
        if (hashlib.sha256(payload).hexdigest()!=pin["sha256"] or pin.get("rows")!=60 or
                len(saved.get("rows",[]))!=60 or {r["key"] for r in saved["rows"]}!=
                {f"{i}:{m:g}" for i in range(20) for m in (.75,1.,1.25)}):
            raise ValueError("preceding fixed validation original manifest SHA/denominator differs")
        for row in saved["rows"]:
            payload=evidence.read(f"validation/update_{pin['updates']:06d}/"+row["file"])
            if hashlib.sha256(payload).hexdigest()!=row["sha256"]:
                raise ValueError("preceding fixed validation original scene shard SHA differs")
    audit=evidence.json(PREFIX+"checkpoint_audit.json")
    status=evidence.json(PREFIX+"origin_status.json")
    if (audit.get("format")!="s1_nd12_datafeed_safe_checkpoint_v1" or audit.get("passed") is not True or
            audit.get("run_id")!=RUN_ID or audit.get("updates")!=start or audit.get("complete_optimizer_rng_cursor") is not True or
            audit.get("complete_validation_nodes") is not True or audit.get("base_learning_contract")!=origin["contract"] or
            audit.get("checkpoint_sha256")!=contract["checkpoint"]["sha256"] or
            audit.get("config_sha256")!=contract["config"]["sha256"] or audit.get("loss_sha256")!=loss_pin["sha256"] or
            audit.get("loss_bytes")!=len(prefix) or audit.get("validation_manifests")!=contract["validation_manifests"] or
            status.get("stage")!="stopped" or status.get("scientific_stop") is True or
            status.get("pid") not in (None,audit.get("original_pid")) or not audit.get("original_pid") or
            not isinstance(audit.get("original_process_start_time"),(int,float))):
        raise ValueError("safe activation full-state/status/fixed-validation audit differs")
    history=job.get("runtime_history",[])
    if (len(history)!=1 or history[0].get("marker")!=marker or history[0].get("safe_start_update")!=start or
            history[0].get("origin_checkpoint_sha256")!=contract["checkpoint"]["sha256"] or
            history[0].get("original_pid")!=audit["original_pid"] or
            history[0].get("original_process_start_time")!=audit["original_process_start_time"]):
        raise ValueError("runtime activation history differs from the original stopped birth/checkpoint")
    from satellite_rl.workbench import nondrl_datafeed_bootstrap as frozen_bridge
    current_rows=[json.loads(line) for line in current_prefix.splitlines() if line.strip()]
    frozen_bridge.validate_loss_rows(current_rows,checkpoint)
    for state in (origin,checkpoint):
        if any(float(slot["step"])!=state["updates"] for slot in state["optimizer"]["state"].values()):
            raise ValueError("complete checkpoint Adam step count differs from actual updates")
    if {p["sha256"] for p in contract["equivalence_proofs"]}!=frozen_bridge.PROOF_SHAS:
        raise ValueError("runtime numerical proofs differ from the reviewed frozen pair")
    formats=set();proofs=[]
    for index,pin in enumerate(contract["equivalence_proofs"]):
        proof=evidence.json(PREFIX+f"proofs/{index:03d}.json")
        if proof.get("passed") is not True:raise ValueError("runtime reader numerical proof did not pass")
        formats.add(proof.get("format"));proofs.append({"format":proof.get("format"),"sha256":pin["sha256"]})
        if proof.get("format")=="s1_nd12_evidence_cache_engineering_v1":
            if any(proof.get(key)is not True for key in
                    ("bitwise_tensors","bitwise_parameters_optimizer","same_dataset_sha","same_sampler_and_torch_rng")):
                raise ValueError("runtime reader lacks actual bitwise tensor/Adam/RNG equivalence")
            draws=proof.get("draw_benchmarks",[]);updates=proof.get("cuda_update_proofs",[])
            if (len(draws)!=2 or any(v.get("samples")!=1024 for v in draws) or
                    any(draws[0].get(k)!=draws[1].get(k) for k in ("positions_sha256","tensors_sha256")) or
                    len(updates)!=2 or any(v.get("updates")!=2 or v.get("samples")!=64 for v in updates) or
                    any(updates[0].get(k)!=updates[1].get(k) for k in ("parameters_sha256","optimizer_sha256",
                        "torch_rng_sha256","cuda_rng_sha256","positions_sha256","sampler_rng_sha256"))):
                raise ValueError("runtime reader actual1024 samples/twoCUDA updates full-state proof differs")
        elif proof.get("format")=="s1_nd12_evidence_cache_compatibility_v1":
            if (proof.get("normalized_full_module_AST_equal") is not True or
                    proof.get("final_candidate_sha256")!=marker["provider_sha256"] or
                    proof.get("verified_existing_cache_manifest_sha256")!=marker["cache_manifest_sha256"] or
                    proof.get("dataset_sha256")!=origin["contract"]["dataset_sha"] or
                    proof.get("cuda_updates_added")!=0 or proof.get("draw_batches_added")!=0 or
                    proof.get("source_original_files_changed") is not False):
                raise ValueError("final provider compatibility differs from the actually proved candidate")
    if not {"s1_nd12_evidence_cache_engineering_v1","s1_nd12_evidence_cache_compatibility_v1"}.issubset(formats):
        raise ValueError("runtime reader equivalence/compatibility proof closure is missing")
    cache=validate_cache(evidence,contract,origin)
    if completed:
        launch=evidence.json("launch.json");command=launch.get("command",[])
        expected_bootstrap=contract["config"]["path"].replace("\\","/").rsplit("/",1)[0]+"/"+PREFIX+"bootstrap.py"
        if (len(command)<3 or command[2].replace("\\","/")!=expected_bootstrap or "--resume" not in command or
                "--contract-sha256" not in command or
                command.index("--contract-sha256")+1>=len(command) or
                command[command.index("--contract-sha256")+1]!=marker["contract_sha256"]):
            raise ValueError("actual completed worker command did not use the frozen external bootstrap")
        final=torch_record(evidence.read("models/final/policy.pt"))
        model=evidence.json("models/final/model.json")
        status=evidence.json("models/status.json")
        for state in (final,model,status):
            if state.get("runtime_extensions")!=[marker]:
                raise ValueError("fixed endpoint/model/status runtime extension marker differs")
            if state.get("contract")!=origin["contract"]:
                raise ValueError("fixed endpoint/model/status base learning contract differs")
        if final.get("format")!="s1_nondrl_supervised_v1" or model.get("format")!="s1_nondrl_supervised_v1":
            raise ValueError("fixed policy/model schema differs from the original supervised endpoint")
        if (checkpoint.get("updates")!=10000 or checkpoint.get("cursor")!=320000 or
                final.get("updates")!=10000 or final.get("selection")!="fixed_budget_endpoint" or
                model.get("updates")!=10000 or model.get("selection")!="fixed_budget_endpoint"):
            raise ValueError("reader extension did not end at the original10000-update endpoint")
        import torch
        if set(checkpoint["model"])!=set(final.get("model",{})) or any(
                not torch.equal(value,final["model"][key]) for key,value in checkpoint["model"].items()):
            raise ValueError("final policy differs from the complete endpoint checkpoint")
    return {"run_id":RUN_ID,"marker":marker,"activation_update":start,
        "preserved_loss_rows":len(original_rows),"original_validation_nodes":expected_nodes,
        "base_contract_preserved":True,"original_active_seconds":origin.get("elapsed_seconds"),
        "complete_active_seconds":checkpoint.get("elapsed_seconds"),
        "cache_canonical_owner":RUN_ID,"cache_manifest_sha256":marker["cache_manifest_sha256"],
        "equivalence_proofs":proofs,"runtime_history":history,"cache":cache}


def audit_runtime(job,evidence,config,*,completed=False):
    extensions=job.get("runtime_extensions",[])
    inherited=job.get("model_runtime_extensions",[])
    if extensions:
        from satellite_rl.workbench.nondrl_datafeed import validate_job_extension
        marker=validate_job_extension(job,evidence.location,read_artifact=evidence.read)
        return validate_training_artifacts(evidence,job,marker,config,completed=completed)
    if inherited:
        if (job.get("experiment_id")!="S1-ND12" or job.get("kind")!="evaluate" or
                job.get("git_commit")!=BASE_COMMIT or job.get("model_source_run_id")!=RUN_ID or
                config["coverage"]["nondrl"]["route"]!="01" or
                config["coverage"]["nondrl"].get("method") not in ("main","M","MO") or len(inherited)!=1 or
                inherited[0].get("extension_id")!=EXTENSION_ID or
                inherited[0].get("base_execution_commit")!=BASE_COMMIT or
                inherited[0].get("scientific_contract_preserved") is not True):
            raise ValueError("model consumer runtime extension is outside the one approved MLM lineage")
        return {"model_source_run_id":RUN_ID,"marker":inherited[0],"execution_role":"model_only_consumer",
            "provider_execution_authorized":False,"cache_canonical_owner":RUN_ID,
            "model_files":job.get("model_files",[])}
    return None


def training_segments(losses,extension):
    """Separate actual active times; resumed supply is not one speed condition."""
    start=extension["activation_update"]
    result=[]
    for first,last,identity in ((1,start,"base_gzip_json_reader"),(start+1,10000,"equivalent_compact_reader")):
        rows=[r for r in losses if first<=r["update"]<=last]
        result.append({"first_update":first,"last_update":last,"recorded_updates":len(rows),"datafeed":identity,
            "loss_elapsed_first":rows[0].get("elapsed_seconds") if rows else None,
            "loss_elapsed_last":rows[-1].get("elapsed_seconds") if rows else None})
    result[0]["checkpoint_active_seconds"]=extension.get("original_active_seconds")
    before,after=extension.get("original_active_seconds"),extension.get("complete_active_seconds")
    result[1]["checkpoint_active_seconds"]=after-before if before is not None and after is not None else None
    return result
