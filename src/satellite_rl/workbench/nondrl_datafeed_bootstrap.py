"""Frozen external bootstrap for one explicitly approved equivalent data reader.

Invoke this file directly; the base scientific snapshot remains first on
PYTHONPATH. Added runtime metadata does not change the base learning contract.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import random
import sys

FORMAT="s1_nd12_datafeed_extension_v1"
EXTENSION_ID="nd12_01_compact_evidence_v1"
RUN_ID="train_nondrl_20261012_055310_890a4cba"
BASE_COMMIT="a21d50ae49f55f1ac2210ba6aaf0778e33f25f8d"
PROVIDER_SHA="e891e952461edd0b3b3250c0c23609807916e8e4bd49d200e9a289c9a866fc22"
CACHE_SHA="475bfd03c5e8983a6c2874cb1b56480180d3b451b64ff1c54293fa68491cb529"
TEACHER_SHA="ba9e6bd2e2a93a96dc52db6805d248f252197386255dab936420b1a3f3f3c35e"
PROOF_SHAS={"8996d22bcc5abaa722c34feb0cf579e976375d7cd8cdfd9b4ec61026a47e9499",
            "d410c122c9685508af11a817ea83647f4fc7a702f7f5e3795acadbe5d4eda59b"}


def sha(path,byte_count=None):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        remaining=byte_count
        while remaining is None or remaining>0:
            block=source.read(1048576 if remaining is None else min(1048576,remaining))
            if not block:
                if remaining not in (None,0):raise ValueError("loss prefix is shorter than frozen boundary")
                break
            h.update(block)
            if remaining is not None:remaining-=len(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pin(item):
    path=Path(item["path"])
    if path.is_symlink() or not path.is_file() or sha(path)!=item["sha256"]:
        raise ValueError("frozen runtime extension artifact SHA differs: "+str(path))
    return path


def marker(contract,contract_sha):
    return {"extension_id":EXTENSION_ID,"contract_sha256":contract_sha,
            "base_execution_commit":BASE_COMMIT,"base_manifest_sha256":contract["base_snapshot"]["sha256"],
            "provider_sha256":contract["provider"]["sha256"],"bridge_sha256":contract["bridge"]["sha256"],
            "cache_manifest_sha256":contract["cache_manifest"]["sha256"],
            "scientific_contract_preserved":True,"runtime_identity":"base_snapshot_plus_frozen_equivalent_reader"}


def validate(contract_path,contract_sha,job_dir,*,initial=False):
    contract=read(pin({"path":str(contract_path),"sha256":contract_sha}))
    directory=Path(job_dir).resolve();job=read(directory/"job.json")
    if (contract.get("format")!=FORMAT or contract.get("extension_id")!=EXTENSION_ID or
            contract.get("run_id")!=RUN_ID or job.get("run_id")!=RUN_ID or
            job.get("experiment_id")!="S1-ND12" or job.get("kind")!="train" or job.get("variant")!="01" or
            contract.get("base_execution_commit")!=BASE_COMMIT or job.get("git_commit")!=BASE_COMMIT or
            contract["provider"]["sha256"]!=PROVIDER_SHA or contract["cache_manifest"]["sha256"]!=CACHE_SHA or
            contract["teacher_manifest"]["sha256"]!=TEACHER_SHA):
        raise ValueError("datafeed extension is outside its single-task/source/provider allowlist")
    base=Path(job["archive_root"]).resolve().parent.parent
    for key in ("provider","bridge","cache_manifest","checkpoint_audit","teacher_manifest"):
        path=pin(contract[key])
        if not path.resolve().is_relative_to(base):raise ValueError("extension source/cache escaped the ND12 group")
    cache=read(contract["cache_manifest"]["path"])
    if set(cache.get("files",{}))!={"data.bin","index.sqlite"}:
        raise ValueError("cache artifact inventory differs")
    for name,item in cache["files"].items():
        pin({"path":str(Path(contract["cache_manifest"]["path"]).parent/name),"sha256":item["sha256"]})
    if Path(contract["config"]["path"]).resolve()!=directory/"config.json":
        raise ValueError("extension config path differs from this task")
    config=read(pin(contract["config"]))
    if (config["training"].get("updates")!=10000 or config["training"].get("seed")!=51001 or
            config["training"].get("batch_size")!=32 or config["training"].get("microbatch_size")!=4):
        raise ValueError("extension cannot alter the existing budget/batch/microbatch/seed")
    snapshot_path=Path(job["source_root"])/"source_manifest.json"
    if Path(contract["base_snapshot"]["path"]).resolve()!=snapshot_path.resolve():
        raise ValueError("base snapshot path differs")
    manifest=read(pin(contract["base_snapshot"]))
    if manifest.get("git_commit")!=BASE_COMMIT:raise ValueError("base scientific source differs")
    for relative,expected in manifest["files"].items():
        path=(snapshot_path.parent/relative).resolve()
        if not path.is_relative_to(snapshot_path.parent.resolve()):raise ValueError("snapshot member escaped")
        pin({"path":str(path),"sha256":expected})
    loss=contract["loss_prefix"]
    if (Path(loss["path"]).resolve()!=Path(job["model_dir"]).resolve()/"losses.jsonl" or
            sha(loss["path"],loss["bytes"])!=loss["sha256"]):
        raise ValueError("original loss curve prefix differs")
    for item in contract["validation_manifests"]:pin(item)
    proof=read(contract["checkpoint_audit"]["path"])
    if (proof.get("format")!="s1_nd12_datafeed_safe_checkpoint_v1" or proof.get("passed") is not True or
            proof.get("run_id")!=RUN_ID or proof.get("updates")!=contract["start_update"] or
            proof.get("checkpoint_sha256")!=contract["checkpoint"]["sha256"] or
            proof.get("config_sha256")!=contract["config"]["sha256"] or
            proof.get("loss_sha256")!=loss["sha256"] or proof.get("loss_bytes")!=loss["bytes"] or
            proof.get("validation_manifests")!=contract["validation_manifests"] or
            proof.get("origin_records")!=contract.get("origin_records") or
            proof.get("complete_optimizer_rng_cursor") is not True or proof.get("complete_validation_nodes") is not True):
        raise ValueError("safe checkpoint/curve/validation proof is incomplete")
    update=contract["start_update"]
    if type(update)is not int or not 1000<update<10000 or not 25<=update%1000<=900:
        raise ValueError("extension activation must follow a complete validation in a safe update window")
    proofs=contract["equivalence_proofs"]
    if {item["sha256"] for item in proofs}!=PROOF_SHAS:
        raise ValueError("extension must use the actual frozen equivalence receipts")
    required={"s1_nd12_evidence_cache_engineering_v1","s1_nd12_evidence_cache_compatibility_v1"}
    observed=set()
    for item in proofs:
        proof=read(pin(item));observed.add(proof.get("format"))
        if proof.get("passed") is not True:raise ValueError("equivalence proof failed")
        if proof.get("format")=="s1_nd12_evidence_cache_engineering_v1" and any(proof.get(key)is not True for key in
                ("bitwise_tensors","bitwise_parameters_optimizer","same_dataset_sha","same_sampler_and_torch_rng")):
            raise ValueError("equivalence proof lacks bitwise parameter/optimizer/RNG/data checks")
    if not required.issubset(observed):raise ValueError("provider equivalence/compatibility proofs are absent")
    if initial:
        pin(contract["checkpoint"])
        if set(contract.get("origin_records",{}))!={"job","launch","status"}:
            raise ValueError("extension requires all original job/launch/status record pins")
        for name,item in contract["origin_records"].items():
            if Path(item["path"]).resolve()!=directory/(name+".json"):
                raise ValueError("original record path differs from the task")
            pin(item)
    return contract,job,marker(contract,contract_sha)


def current_checkpoint(job,contract,runtime):
    import torch
    path=Path(job["model_dir"])/"checkpoint.pt"
    before=sha(path)
    state=torch.load(path,map_location="cpu",weights_only=False)
    original=read(contract["checkpoint_audit"]["path"])
    validate_rng(state.get("rng"))
    if (state.get("format")!="s1_nondrl_checkpoint_v1" or state.get("total_updates")!=10000 or
            not contract["start_update"]<=state.get("updates",-1)<=10000 or
            state.get("cursor",-1)!=state["updates"]*32 or
            not state.get("model") or not state.get("optimizer",{}).get("state") or
            set(state.get("rng",{}))!={"python","numpy","torch","cuda","sampler"} or
            not state["rng"]["cuda"] or state["contract"].get("source_sha")!=BASE_COMMIT or
            state["contract"]!=original.get("base_learning_contract") or
            state["contract"].get("dataset_sha")!=TEACHER_SHA or
            state["contract"].get("source_manifest_sha256")!=contract["base_snapshot"]["sha256"]):
        raise ValueError("complete checkpoint/base source/optimizer/RNG/cursor differs")
    if sha(path)!=contract["checkpoint"]["sha256"] and state.get("runtime_extensions")!=[runtime]:
        raise ValueError("advanced checkpoint has no exact composite runtime marker")
    if any(float(entry.get("step",-1))!=state["updates"] for entry in state["optimizer"]["state"].values()):
        raise ValueError("Adam update counter differs from the learning cursor")
    loss=Path(job["model_dir"])/"losses.jsonl"
    if sha(loss,state["loss_bytes"])!=state["loss_sha256"]:
        raise ValueError("current checkpoint loss prefix differs")
    with loss.open("rb") as source:rows=[json.loads(line) for line in source.read(state["loss_bytes"]).splitlines() if line.strip()]
    validate_loss_rows(rows,state)
    if sha(path)!=before:raise ValueError("checkpoint changed while validating its full state")
    return state


def validate_rng(value):
    """Check private generator state formats without advancing any global RNG."""
    import numpy as np
    import torch
    try:
        if not isinstance(value,dict) or set(value)!={"python","numpy","torch","cuda","sampler"}:
            raise ValueError("RNG keys differ")
        private_python=random.Random(0);private_python.setstate(value["python"])
        private_numpy=np.random.RandomState(0);private_numpy.set_state(value["numpy"])
        private_sampler=np.random.PCG64(0);private_sampler.state=value["sampler"]
        cpu=value["torch"]
        if not isinstance(cpu,torch.Tensor) or cpu.dtype!=torch.uint8 or cpu.ndim!=1:
            raise ValueError("CPU RNG byte tensor format differs")
        torch.Generator(device="cpu").set_state(cpu.detach().cpu())
        cuda=value["cuda"]
        if not isinstance(cuda,list) or not cuda:raise ValueError("CUDA RNG topology is absent")
        if torch.cuda.is_available() and len(cuda)!=torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs from this execution environment")
        for index,state in enumerate(cuda):
            # PyTorch2.3 stores Philox seed/offset as two uint64 values. CUDA's
            # offset advances in groups of four. Offline audits can check the
            # same byte format even without a CUDA runtime.
            if (not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.ndim!=1 or state.numel()!=16 or
                    int.from_bytes(bytes(state.detach().cpu().tolist()[8:]),"little")%4):
                raise ValueError("CUDA2.3 Philox RNG byte format differs")
            if torch.cuda.is_available():
                torch.Generator(device=f"cuda:{index}").set_state(state.detach().cpu())
    except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as error:
        raise ValueError("incomplete or invalid private RNG state: "+str(error)) from error


def validate_loss_rows(rows,state):
    if (len(rows)!=state["updates"] or any(row.get("update")!=i or row.get("cursor")!=i*32 or
            len(row.get("sample_refs",[]))!=32 for i,row in enumerate(rows,1)) or
            rows[-1]["sample_refs"]!=state.get("sampled_last")):
        raise ValueError("loss curve updates/cursors/sample references do not close")


def install(contract, runtime, job):
    """Install only the equivalent reader and bounded top-level runtime metadata."""
    import satellite_rl.coverage.nondrl.learning as learning
    extension=Path(contract["config"]["path"]).parent/"runtime_extensions"/EXTENSION_ID
    provider_path=extension/"provider.py" if (extension/"provider.py").is_file() else Path(contract["provider"]["path"])
    pin({"path":str(provider_path),"sha256":contract["provider"]["sha256"]})
    cache_path=extension/"cache/manifest.json" if (extension/"cache/manifest.json").is_file() else Path(contract["cache_manifest"]["path"])
    spec=importlib.util.spec_from_file_location("satellite_rl.coverage.nondrl.evidence_cache_extension",str(provider_path))
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    original_dataset=learning.EvidenceDataset
    opened=[]
    def dataset(path,route):
        source=Path(path).resolve()
        if source.is_dir():source/= "manifest.json"
        if route!="01" or source!=Path(contract["teacher_manifest"]["path"]).resolve() or sha(source)!=contract["teacher_manifest"]["sha256"]:
            raise ValueError("equivalent reader cannot consume a different route or teacher")
        reader=provider.CompactEvidenceDataset(cache_path,expected_source=source,
            expected_sha256=contract["cache_manifest"]["sha256"],verify_original=True)
        opened.append(reader)
        return reader
    learning.EvidenceDataset=dataset
    original_save,original_json=learning.save_torch,learning.atomic_json
    model_root=Path(job["model_dir"]).resolve()
    def annotate(path,value):
        path=Path(path).resolve()
        if isinstance(value,dict) and path.is_relative_to(model_root):
            # Preserve contract, tensors, optimizer, RNG, validation and cursor.
            value={**value,"runtime_extensions":[runtime]}
        return value
    def save_torch(path,value):
        selected=(Path(path).resolve().is_relative_to(model_root) and isinstance(value,dict) and
                  value.get("format") in (learning.MODEL_FORMAT,learning.CHECKPOINT_FORMAT))
        original_save(path,annotate(path,value) if selected else value)
    learning.save_torch=save_torch
    learning.atomic_json=lambda path,value:original_json(path,annotate(path,value))
    def restore():
        try:
            for reader in opened:reader.close()
        finally:
            opened.clear()
            learning.EvidenceDataset=original_dataset;learning.save_torch=original_save;learning.atomic_json=original_json
    return restore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir",required=True,type=Path)
    parser.add_argument("--contract",required=True,type=Path)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--resume",action="store_true",required=True)
    args=parser.parse_args()
    contract,job,runtime=validate(args.contract,args.contract_sha256,args.job_dir)
    current_checkpoint(job,contract,runtime)
    if sha(__file__)!=contract["bridge"]["sha256"]:raise ValueError("actual frozen bootstrap bytes differ")
    if Path(job["source_root"]).resolve()/"src"!=Path(sys.path[0]).resolve():
        # Direct-script sys.path[0] is its external folder. Force only the
        # pinned base src ahead of it; do not import source from the ops repo.
        sys.path.insert(0,str(Path(job["source_root"]).resolve()/"src"))
    import satellite_rl.coverage.nondrl.learning as learning
    if not Path(learning.__file__).resolve().is_relative_to(Path(job["source_root"]).resolve()):
        raise ValueError("bootstrap imported learning from outside the frozen base")
    restore=install(contract,runtime,job)
    try:
        sys.argv=["satellite_rl.workbench.runner","--job-dir",str(args.job_dir),"--resume"]
        runpy.run_module("satellite_rl.workbench.runner",run_name="__main__")
    finally:
        restore()


if __name__=="__main__":main()
