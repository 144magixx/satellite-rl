"""Strict, archived recovery of stopped jobs after an equivalent atomic-I/O fix."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, inside, new_id, now, read_json


REPAIR_ID = "atomic_io_retry_v1"
WORKER_FILES = {"src/satellite_rl/workbench/runner.py", "src/satellite_rl/workbench/checkpoints.py"}
CONTROL_FILES = {"src/satellite_rl/workbench/runtime.py", "src/satellite_rl/workbench/service.py",
                 "src/satellite_rl/workbench/recovery.py"}
RETRY_MODULE = "satellite_rl.training.run_monitor"

S2P17_BOUNDARY_REPAIR_ID = "sca_basic_threshold_exact_v1"
S2P17_ORIGIN_COMMIT = "0a4742a9a15af93478501497fac5cd36579fa075"
S2P17_FAILED_RUN = "refine_sac_20261004_215930_e814197c"
S2P17_FAILED_REQUEST = "s2p17-v0-jan29-sca"
S2P17_FAILED_CASE = "scene_00033_r0_sca_si_lp"
S2P17_SOURCE_ALLOWLIST = {
    "src/satellite_rl/power_refinement/priority.py",
    "src/satellite_rl/power_refinement/runner.py",
    "src/satellite_rl/workbench/recovery.py",
    "src/satellite_rl/workbench/runtime.py",
    "src/satellite_rl/workbench/service.py",
    "src/satellite_rl/workbench/power_refinement.py",
}

I0_BUDGET_REPAIR_ID = "s2p19_i0_validation_budget_v1"
I0_ORIGIN_COMMIT = "13ea014f1c1709c4a82059fc89ba3092e75010e1"
I0_RUN_ID = "train_sac_20261007_174121_7d0b687c"
I0_REQUEST_ID = "s2p19-i0-train-51001"
I0_CONFIG_SHA256 = "fdd0b03f742324c166c98bc0a70bc833820cbd516d84a15e1446b98e55a383fe"
I0_EQUIVALENCE_120K_SHA256 = "7ff1020f179df269184e60df1551cb1d8018462fef65a055ead77a5cb30ef91d"
I0_SAVED_VALIDATION_STEPS = (0, 50000, 97112, 100000)
I0_BUDGET_SOURCE_FILES = {
    "src/satellite_rl/sac_improvement/runner.py",
    "src/satellite_rl/workbench/recovery.py",
    "src/satellite_rl/workbench/runtime.py",
}


def verify_i0_validation_budget_chain(old_root, new_root, new_commit, validation):
    """Bind a scheduling-only source change to the original 13ea I0 run."""
    old = verified_manifest(old_root, I0_ORIGIN_COMMIT)
    new = verified_manifest(new_root, new_commit)
    names = set(old["files"]) | set(new["files"])
    changed = {name for name in names if old["files"].get(name) != new["files"].get(name)}
    if changed != I0_BUDGET_SOURCE_FILES:
        raise ValueError("I0验证预算修复源码超出runner/恢复入口三文件")
    criteria = validation.get("criteria") or {}
    required = {"saved_nodes_sha_unchanged", "same_checkpoint_learning_state",
                "one_gpu_update_equal", "training_rng_equal", "validation_rng_restored",
                "future_direct_nodes_at_most_six", "skip_only_future_300k_validation"}
    if (validation.get("format") != "s2p19_i0_validation_budget_repair_v1" or
            validation.get("repair_id") != I0_BUDGET_REPAIR_ID or
            validation.get("origin_commit") != I0_ORIGIN_COMMIT or
            validation.get("repair_commit") != new_commit or
            validation.get("run_id") != I0_RUN_ID or
            validation.get("tests_passed") is not True or
            set(criteria) != required or any(criteria[key] is not True for key in required)):
        raise ValueError("I0验证预算修复缺少同断点/更新/RNG等价证据")
    return {"repair_id": I0_BUDGET_REPAIR_ID,
            "origin_commit": I0_ORIGIN_COMMIT, "repair_commit": new_commit,
            "old_source_root": str(Path(old_root).resolve()),
            "new_source_root": str(Path(new_root).resolve()),
            "old_manifest_sha256": digest(Path(old_root) / "source_manifest.json"),
            "new_manifest_sha256": digest(Path(new_root) / "source_manifest.json"),
            "changed_files": sorted(changed),
            "source_file_sha256_before": {name: old["files"][name] for name in sorted(changed)},
            "source_file_sha256_after": {name: new["files"][name] for name in sorted(changed)},
            "scope": "I0 future validation cadence only; frozen train inputs and update semantics unchanged"}


def _canonical_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_s2p17_boundary_chain(old_root, new_root, new_commit, validation):
    """File-allowlisted behavior repair; never describe it as AST equivalence."""
    old = verified_manifest(old_root, S2P17_ORIGIN_COMMIT)
    new = verified_manifest(new_root, new_commit)
    names = set(old["files"]) | set(new["files"])
    changed = sorted(name for name in names if old["files"].get(name) != new["files"].get(name))
    if (not {"src/satellite_rl/power_refinement/priority.py",
             "src/satellite_rl/power_refinement/runner.py"}.issubset(changed)
            or any(name not in S2P17_SOURCE_ALLOWLIST for name in changed)):
        raise ValueError("S2-P17修复源码超出精确文件允许名单")
    if (validation.get("format") != "s2p17_boundary_validation_v1"
            or validation.get("repair_id") != S2P17_BOUNDARY_REPAIR_ID
            or validation.get("origin_science_commit") != S2P17_ORIGIN_COMMIT
            or validation.get("repair_commit") != new_commit
            or validation.get("original_299_cases_verified") is not True
            or validation.get("failed_B_fallback_to_A_verified") is not True
            or validation.get("strict_final_guard_unchanged") is not True
            or validation.get("candidate_boundary_consistent") is not True
            or validation.get("tests_passed") is not True):
        raise ValueError("S2-P17数值边界修复缺少完整工程验证回执")
    return {"repair_id": S2P17_BOUNDARY_REPAIR_ID,
            "origin_science_commit": S2P17_ORIGIN_COMMIT,
            "repair_commit": new_commit,
            "old_source_root": str(Path(old_root).resolve()),
            "new_source_root": str(Path(new_root).resolve()),
            "old_manifest_sha256": digest(Path(old_root) / "source_manifest.json"),
            "new_manifest_sha256": digest(Path(new_root) / "source_manifest.json"),
            "changed_files": changed,
            "source_file_sha256_before": {name: old["files"].get(name) for name in changed},
            "source_file_sha256_after": {name: new["files"].get(name) for name in changed},
            "behavior_note": "Candidate basic-service boundary acceptance changed; final strict guard unchanged",
            "ast_equivalent": False}


def verified_manifest(root, commit):
    root = Path(root).resolve()
    manifest = read_json(root / "source_manifest.json")
    if not manifest or manifest.get("git_commit") != commit or not manifest.get("files"):
        raise ValueError("恢复源码的冻结清单或提交号不一致")
    for name, expected in manifest["files"].items():
        path = inside(root / name, root)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("恢复源码校验失败：" + name)
    return manifest


class _NormalizeRetry(ast.NodeTransformer):
    """Only two-argument calls to the already audited helper may replace os.replace."""

    def visit_ImportFrom(self, node):
        if node.module == RETRY_MODULE:
            remaining = [item for item in node.names
                         if not (item.name == "replace_with_retry" and item.asname is None)]
            if not remaining:
                return None
            node.names = remaining
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "replace_with_retry":
            if len(node.args) != 2 or node.keywords:
                raise ValueError("I/O恢复只允许无额外参数的原子替换重试")
            node.func = ast.Attribute(value=ast.Name(id="os", ctx=ast.Load()),
                                      attr="replace", ctx=ast.Load())
        return node


def _worker_ast(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    return ast.dump(_NormalizeRetry().visit(tree), include_attributes=False)


def _control_ast(path):
    """The scheduler may only change resume, and service only forwards its option."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    if Path(path).name == "runtime.py":
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
                node.body = [item for item in node.body
                             if not (isinstance(item, ast.FunctionDef) and item.name == "resume")]
    else:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "resume" and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "scheduler"):
                for keyword in node.keywords:
                    if keyword.arg == "source_repair" and ast.dump(keyword.value) != ast.dump(
                            ast.parse('body.get("source_repair")', mode="eval").body):
                        raise ValueError("恢复接口仅允许转发指定的修复类型")
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != "source_repair"]
    return ast.dump(tree, include_attributes=False)


def verify_io_retry_repair(old_root, old_commit, new_root, new_commit):
    """Reject any learning/data/config change; check worker code beyond I/O by AST."""
    old_root, new_root = Path(old_root), Path(new_root)
    old = verified_manifest(old_root, old_commit)
    new = verified_manifest(new_root, new_commit)
    names = set(old["files"]) | set(new["files"])
    changed = sorted(name for name in names if old["files"].get(name) != new["files"].get(name))
    if not WORKER_FILES.issubset(changed) or any(name not in WORKER_FILES | CONTROL_FILES for name in changed):
        raise ValueError("恢复版本含未许可的学习/环境/数据改动，或缺少必要I/O修复")
    for name in CONTROL_FILES - {"src/satellite_rl/workbench/recovery.py"}:
        if name in changed and (name not in old["files"] or name not in new["files"]
                                or _control_ast(old_root / name) != _control_ast(new_root / name)):
            raise ValueError("恢复版本改变了恢复接口以外的任务启动/服务逻辑：" + name)
    for name in WORKER_FILES:
        if name not in old["files"] or name not in new["files"]:
            raise ValueError("恢复版本缺少原训练入口或断点模块")
        if _worker_ast(old_root / name) != _worker_ast(new_root / name):
            raise ValueError("恢复版本改变了原子替换以外的训练/断点逻辑：" + name)
        tree = ast.parse((new_root / name).read_text(encoding="utf-8-sig"))
        helpers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "replace_with_retry"]
        direct = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == "os" and n.func.attr == "replace"]
        helper_import = any(isinstance(n, ast.ImportFrom) and n.module == RETRY_MODULE
                            and any(a.name == "replace_with_retry" and a.asname is None for a in n.names)
                            for n in tree.body)
        if not helpers or direct or not helper_import:
            raise ValueError("恢复版本没有完整使用既有原子替换重试")
    return {"repair_id": REPAIR_ID, "old_commit": old_commit, "new_commit": new_commit,
            "old_source_root": str(old_root), "new_source_root": str(new_root),
            "changed_files": changed, "worker_ast_equivalent_except_atomic_retry": True,
            "unchanged_file_count": len(names) - len(changed),
            "old_manifest_sha256": digest(old_root / "source_manifest.json"),
            "new_manifest_sha256": digest(new_root / "source_manifest.json")}


def prepare_io_retry_recovery(store, job):
    """Prepare a new immutable binding, with checkpoint and failure evidence copied first.

    Called with Scheduler.lock held after checking that no worker is alive. Does
    not queue or mutate job.json itself, so failed validation cannot resume a job.
    """
    directory = store.job_dir(job["run_id"])
    old_root = inside(job["source_root"], store.root / "source_snapshots")
    previous = job.get("io_repair", {})
    if previous.get("repair_id") == REPAIR_ID and previous.get("new_commit") == job["git_commit"]:
        # A prior API attempt may have published the new job but failed to queue.
        verified_manifest(old_root, job["git_commit"])
        if digest(directory / "checkpoint.pt") != previous.get("checkpoint_sha256"):
            raise ValueError("此任务已应用I/O修复并产生新断点；请检查新故障并使用普通断点恢复")
        return job
    failure = read_json(directory / "failure.json", {})
    if not re.search(r"\[WinError (5|32)\]", failure.get("error", "")) or "replace" not in failure.get("traceback", ""):
        raise ValueError("此修复恢复仅用于已记录的Windows原子替换占用错误")
    checkpoint = directory / "checkpoint.pt"
    checkpoint_sha = digest(checkpoint)
    reference = failure.get("last_normal_checkpoint", {})
    if reference.get("sha256") != checkpoint_sha:
        raise ValueError("当前断点校验值与失败时最后正常断点不一致")
    import torch
    from .checkpoints import config_hash

    # This is a trusted local job-owned full checkpoint, never an arbitrary import.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = read_json(directory / "config.json")
    progress = payload.get("progress", {})
    step = progress.get("env_step")
    if (payload.get("format") != "satellite_rl_workbench_checkpoint_v1"
            or payload.get("config_hash") != config_hash(config)
            or step != reference.get("env_step") or not isinstance(step, int)
            or not 0 <= step < config["training"]["total_steps"]
            or not payload.get("rng") or not payload.get("agent", {}).get("rng")):
        raise ValueError("恢复断点的配置、进度或随机状态不完整")
    update_step = int(payload["agent"]["update_step"])
    del payload
    new_root, new_commit = store.freeze_source()
    proof = verify_io_retry_repair(old_root, job["git_commit"], new_root, new_commit)
    archive = inside(directory / "recoveries" / new_id(REPAIR_ID), directory)
    archive.mkdir(parents=True)
    backup_files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            copied = archive / path.name
            shutil.copy2(path, copied)
            checksum = digest(path)
            if checksum != digest(copied):
                raise RuntimeError("恢复前备份校验失败：" + path.name)
            backup_files[path.name] = checksum
    proof.update(at=now(), archive_path=str(archive), checkpoint_sha256=checkpoint_sha,
                 checkpoint_step=step, checkpoint_update_step=update_step,
                 backup_files_sha256=backup_files, reason="Windows transient atomic replacement failures")
    atomic_json(archive / "recovery_manifest.json", proof)
    updated = deepcopy(job)
    updated.setdefault("original_git_commit", job["git_commit"])
    updated.setdefault("source_history", []).append(proof)
    updated.update(source_root=str(new_root), git_commit=new_commit, io_repair=proof)
    return updated


def prepare_i0_validation_budget_recovery(store, job, validation_path,
                                          validation_sha256, expected_repair_commit):
    """Preserve the approved I0 run and rebind only its future validation cadence."""
    import torch

    if (job.get("run_id") != I0_RUN_ID or job.get("request_id") != I0_REQUEST_ID or
            job.get("kind") != "train" or job.get("experiment_id") != "S2-P19" or
            job.get("research_stage") != "sac_improvement" or job.get("mode") != "train" or
            job.get("seed") != 51001 or job.get("total_steps") != 400000 or
            not isinstance(expected_repair_commit, str) or
            not re.fullmatch(r"[0-9a-f]{40}", expected_repair_commit)):
        raise ValueError("I0验证预算修复只允许原单种子40万步训练任务")
    directory = store.job_dir(I0_RUN_ID)
    previous = job.get("i0_validation_budget_repair") or {}
    if previous:
        proof_path = Path(previous.get("archive_path", "")) / "repair_proof.json"
        if (previous.get("repair_id") != I0_BUDGET_REPAIR_ID or
                previous.get("repair_commit") != expected_repair_commit or
                previous.get("validation_receipt_sha256") != validation_sha256 or
                job.get("git_commit") != expected_repair_commit or
                job.get("source_root") != previous.get("new_source_root") or
                read_json(proof_path) != previous):
            raise ValueError("I0先前预算修复来源/证明不一致，禁止再次替换")
        verified_manifest(job["source_root"], expected_repair_commit)
        return job
    if job.get("git_commit") != I0_ORIGIN_COMMIT:
        raise ValueError("I0训练原冻结源码不是13ea")
    status = read_json(directory / "status.json", {})
    config_path, checkpoint = directory / "config.json", directory / "checkpoint.pt"
    config = read_json(config_path, {})
    controls = config.get("i0") or {}
    if (digest(config_path) != I0_CONFIG_SHA256 or
            job.get("i0_config") != controls or
            controls.get("seed") != 51001 or controls.get("total_steps") != 400000 or
            controls.get("pilot_stop_steps") != 100000 or
            controls.get("pilot_max_episodes") != 3125 or
            controls.get("max_episodes") != 12500 or controls.get("device") != "cuda" or
            (config.get("training") or {}).get("seed") != 51001 or
            (config.get("training") or {}).get("total_steps") != 400000):
        raise ValueError("I0冻结配置/种子/训练预算不一致")
    checkpoint_sha = digest(checkpoint)
    step, episodes, updates = (status.get(key) for key in ("env_step", "episodes", "updates"))
    if (status.get("stage") != "stopped" or status.get("stop_reason") != "user_stop" or
            status.get("pilot_gate_passed") is not True or
            type(step) is not int or not 100000 < step < 300000 or
            type(episodes) is not int or not 3125 < episodes < 12500 or
            type(updates) is not int or updates <= 0 or
            status.get("checkpoint_step") != step or
            status.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("I0没有300k之前、pilot已通过的正常完整停止断点")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = payload.get("agent") or {}
    rng = payload.get("rng") or {}
    if (payload.get("version") != 1 or payload.get("run_id") != I0_RUN_ID or
            payload.get("frozen_config_sha256") != I0_CONFIG_SHA256 or
            payload.get("i0_config") != controls or
            (payload.get("env_step"), payload.get("episodes")) != (step, episodes) or
            payload.get("pilot_gate_passed") is not True or
            agent.get("updates") != updates or
            not {"actor", "critic1", "critic2", "target1", "target2",
                 "actor_opt", "critic1_opt", "critic2_opt", "replay"}.issubset(agent) or
            set(rng) != {"python", "numpy", "torch", "cuda"} or
            any(rng[key] is None for key in rng) or
            not (isinstance(payload.get("environment"), dict) or
                 (payload.get("environment") is None and payload.get("scene_index") is None))):
        raise ValueError("I0断点缺完整策略/优化器/回放/RNG/环境状态")
    del payload
    gate = read_json(directory / "pilot_gate.json", {})
    pilot = read_json(directory / "pilot_artifacts/manifest.json", {})
    audit_path = Path(gate.get("audit_path", ""))
    audit = read_json(audit_path, {})
    pilot_sha = digest(directory / "pilot_artifacts/checkpoint.pt")
    if (gate.get("pass") is not True or gate.get("run_id") != I0_RUN_ID or
            gate.get("checkpoint_step") != 97112 or
            gate.get("checkpoint_sha256") != pilot_sha or
            pilot.get("run_id") != I0_RUN_ID or pilot.get("env_step") != 97112 or
            pilot.get("checkpoint_sha256") != pilot_sha or
            pilot.get("config_sha256") != I0_CONFIG_SHA256 or
            pilot.get("git_commit") != I0_ORIGIN_COMMIT or
            pilot.get("audit_path") != gate.get("audit_path") or
            pilot.get("audit_sha256") != gate.get("audit_sha256") or
            not audit_path.is_file() or not audit_path.resolve().is_relative_to(store.archive_root.resolve()) or
            digest(audit_path) != gate.get("audit_sha256") or
            audit.get("run_id") != I0_RUN_ID or audit.get("checkpoint_sha256") != pilot_sha or
            audit.get("checkpoint_step") != 97112 or audit.get("passed") is not True or
            not audit.get("criteria") or not audit.get("source_files") or
            any(value is not True for value in audit["criteria"].values())):
        raise ValueError("I0已批准97112试点/原始审计证据不一致")
    for reference in audit.get("source_files") or []:
        if digest(reference["path"]) != reference["sha256"]:
            raise ValueError("I0试点审计原始来源SHA变化")
    validation_dir = directory / "validation"
    found = sorted(int(path.parent.name.removeprefix("step_"))
                   for path in validation_dir.glob("step_*/summary.json"))
    mandatory = set(I0_SAVED_VALIDATION_STEPS)
    if (not mandatory.issubset(found) or
            set(found) not in (mandatory, mandatory | {200000}) or
            (validation_dir / "step_000300000").exists() or
            (directory / "validation_skips/step_000300000.json").exists()):
        raise ValueError("I0已存验证节点不符合四个原节点与最多200k的预算")
    node_refs = []
    archive_names = ["checkpoint.pt", "config.json", "job.json", "status.json",
                     "pilot_gate.json", "pilot_artifacts/checkpoint.pt",
                     "pilot_artifacts/manifest.json"]
    for node in found:
        relative = f"validation/step_{node:09d}"
        summary_path = directory / relative / "summary.json"
        summary = read_json(summary_path, {})
        cases = summary.get("cases") or []
        if (summary.get("env_step") != node or summary.get("count") != 72 or
                len(cases) != 72 or
                len(list((directory / relative).glob("scene_*.json"))) != 72 or
                not re.fullmatch(r"[0-9a-f]{64}", str(summary.get("checkpoint_sha256", "")))):
            raise ValueError("I0已存验证摘要/72案例不完整")
        case_hashes = []
        for index, reference in enumerate(cases):
            case_path = inside(reference["path"], directory / relative)
            if (case_path.name != f"scene_{index:05d}.json" or
                    digest(case_path) != reference["sha256"]):
                raise ValueError("I0已存验证案例路径或SHA变化")
            case = read_json(case_path, {})
            if (case.get("env_step") != node or
                    case.get("checkpoint_sha256") != summary["checkpoint_sha256"]):
                raise ValueError("I0已存验证案例与原节点断点不一致")
            case_hashes.append(reference["sha256"])
            archive_names.append(f"{relative}/{case_path.name}")
        archive_names.append(f"{relative}/summary.json")
        node_refs.append({"step": node, "summary_sha256": digest(summary_path),
                          "case_hashes_sha256": _canonical_sha(case_hashes),
                          "checkpoint_sha256": summary["checkpoint_sha256"]})
    validation_path = inside(validation_path, directory / "recoveries")
    if digest(validation_path) != validation_sha256:
        raise ValueError("I0验证预算修复回执SHA不符")
    validation = read_json(validation_path, {})
    actual = validation.get("actual_recovery_checkpoint") or {}
    equivalent = validation.get("equivalence_checkpoint") or {}
    equivalent_path = Path(equivalent.get("proof_path", ""))
    equivalent_proof = read_json(equivalent_path, {})
    if (actual != {"sha256": checkpoint_sha, "env_step": step,
                   "episodes": episodes, "updates": updates} or
            equivalent.get("sha256") != I0_EQUIVALENCE_120K_SHA256 or
            equivalent.get("env_step") != 120000 or
            equivalent.get("source_commit") != I0_ORIGIN_COMMIT or
            not equivalent_path.is_file() or
            not equivalent_path.resolve().is_relative_to(store.archive_root.resolve()) or
            digest(equivalent_path) != equivalent.get("proof_sha256") or
            equivalent_proof.get("strict_equivalence_passed") is not True or
            equivalent_proof.get("copy_sha256_before") != I0_EQUIVALENCE_120K_SHA256 or
            equivalent_proof.get("copy_sha256_after") != I0_EQUIVALENCE_120K_SHA256 or
            equivalent_proof.get("copied_env_step") != 120000 or
            equivalent_proof.get("old_source_commit") != I0_ORIGIN_COMMIT or
            equivalent_proof.get("patch_commit") != "708c5264f9ebcd2a0b423da946c7042337211d74" or
            equivalent_proof.get("old_new_config_equal") is not True or
            equivalent_proof.get("old_new_checkpoint_loader_source_equal") is not True or
            equivalent_proof.get("gpu_replay_sample_equal") is not True or
            equivalent_proof.get("partial_env_state_equal") is not True or
            equivalent_proof.get("simulated_skip", {}).get("rng_unchanged") is not True or
            equivalent_proof.get("simulated_skip", {}).get("no_validation_case_or_summary_created") is not True or
            equivalent_proof.get("simulated_skip", {}).get("300k_checkpoint_due") is not True or
            equivalent_proof.get("diagnostic_max_abs") != 0 or
            equivalent_proof.get("post_network_max_abs") != 0 or
            equivalent_proof.get("post_optimizer_max_abs") != 0 or
            set(equivalent_proof.get("actor_double_q_parameter_changes") or {}) !=
            {"actor", "critic1", "critic2"} or
            any(value <= 0 for value in
                (equivalent_proof.get("actor_double_q_parameter_changes") or {}).values()) or
            validation.get("config_sha256") != I0_CONFIG_SHA256 or
            validation.get("pilot_checkpoint_sha256") != pilot_sha or
            validation.get("pilot_audit_sha256") != gate["audit_sha256"] or
            validation.get("saved_nodes") != node_refs):
        raise ValueError("I0修复回执把代表性120k等价证据与实际停点混淆，或旧节点SHA不符")
    evidence = validation.get("evidence_files") or []
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("I0修复回执缺少可核原始等价证据")
    for item in evidence:
        path = inside(item["path"], store.archive_root)
        if digest(path) != item["sha256"]:
            raise ValueError("I0等价证据文件SHA变化")
    old_root = inside(job["source_root"], store.root / "source_snapshots")
    new_root, new_commit = store.freeze_source()
    if new_commit != expected_repair_commit or new_commit == I0_ORIGIN_COMMIT:
        raise ValueError("I0修复服务源码不是指定干净提交")
    source_proof = verify_i0_validation_budget_chain(old_root, new_root, new_commit, validation)
    archive = inside(directory / "recoveries" /
                     (I0_BUDGET_REPAIR_ID + "_" + new_commit[:12]), directory)
    original = archive / "original"
    if archive.exists():
        proof = read_json(archive / "repair_proof.json", {})
        backup = proof.get("backup_files_sha256") or {}
        if (any(proof.get(key) != value for key, value in source_proof.items()) or
                proof.get("run_id") != I0_RUN_ID or
                proof.get("checkpoint_sha256") != checkpoint_sha or
                proof.get("validation_receipt_sha256") != validation_sha256 or
                proof.get("saved_nodes") != node_refs or
                set(backup) != set(archive_names) or
                any(digest(original / name) != backup[name] or
                    digest(directory / name) != backup[name] for name in archive_names)):
            raise ValueError("I0既有修复归档不完整或来源不同，禁止覆盖")
    else:
        original.mkdir(parents=True)
        backup = {}
        for relative in archive_names:
            source = inside(directory / relative, directory)
            destination = original / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(source) != digest(destination):
                raise RuntimeError("I0旧断点/验证案例归档SHA失败：" + relative)
            backup[relative] = digest(source)
        shutil.copy2(validation_path, archive / "validation_receipt.json")
        if digest(archive / "validation_receipt.json") != validation_sha256:
            raise RuntimeError("I0验证回执归档SHA失败")
        proof = {**source_proof, "run_id": I0_RUN_ID, "at": now(),
                 "archive_path": str(archive), "checkpoint_sha256": checkpoint_sha,
                 "checkpoint_step": step, "checkpoint_episodes": episodes,
                 "checkpoint_updates": updates, "config_sha256": I0_CONFIG_SHA256,
                 "pilot_checkpoint_sha256": pilot_sha,
                 "pilot_gate_sha256": digest(directory / "pilot_gate.json"),
                 "pilot_audit_sha256": gate["audit_sha256"],
                 "equivalence_checkpoint_sha256": I0_EQUIVALENCE_120K_SHA256,
                 "equivalence_proof_sha256": equivalent["proof_sha256"],
                 "saved_nodes": node_refs, "backup_files_sha256": backup,
                 "validation_receipt_path": str(validation_path),
                 "validation_receipt_sha256": validation_sha256,
                 "reason": "reserve one direct validation for actual terminal within six nodes"}
        atomic_json(archive / "repair_proof.json", proof)
    updated = deepcopy(job)
    updated.setdefault("original_git_commit", I0_ORIGIN_COMMIT)
    updated.setdefault("source_history", []).append(proof)
    updated.update(source_root=str(new_root), git_commit=new_commit,
                   scientific_origin_commit=I0_ORIGIN_COMMIT,
                   i0_validation_budget_repair=proof)
    return updated


def prepare_s2p17_boundary_recovery(store, job, validation_path, validation_sha256,
                                     expected_repair_commit):
    """Prepare only the failed V0 SCA task; queueing remains Scheduler's job.

    This is a behavior-changing boundary fix, not an I/O-equivalent source
    replacement. Every previously committed case stays byte-for-byte intact.
    """
    import math
    import json

    directory = store.job_dir(job["run_id"])
    previous = job.get("boundary_repair") or {}
    if previous.get("repair_id") == S2P17_BOUNDARY_REPAIR_ID:
        proof_path = Path(previous["archive_path"]) / "repair_proof.json"
        fallback_path = directory / "failed_attempt_fallback.json"
        if (job.get("run_id") != S2P17_FAILED_RUN or
                job.get("scientific_origin_commit") != S2P17_ORIGIN_COMMIT or
                job.get("git_commit") != expected_repair_commit or
                previous.get("repair_commit") != expected_repair_commit or
                job.get("source_root") != previous.get("new_source_root") or
                read_json(proof_path) != previous or
                digest(fallback_path) != previous.get("fallback_contract_sha256") or
                digest(validation_path) != previous.get("validation_receipt_sha256")):
            raise ValueError("先前S2-P17修复准备状态不一致，禁止覆盖")
        verified_manifest(job["source_root"], expected_repair_commit)
        return job
    if (job.get("run_id") != S2P17_FAILED_RUN or
            job.get("request_id") != S2P17_FAILED_REQUEST or
            job.get("kind") != "refine" or job.get("experiment_id") != "S2-P17" or
            job.get("git_commit") != S2P17_ORIGIN_COMMIT or
            (job.get("initial_source") or {}).get("group") != "model"):
        raise ValueError("此数值边界修复只允许原S2-P17 V0细调失败任务")
    status = read_json(directory / "status.json", {})
    failure = read_json(directory / "failure.json", {})
    if (status.get("stage") != "failed" or status.get("env_step") != 299 or
            status.get("budget_env_steps") != 3960 or
            "SI优先结果新增严格基本服务跨下波束" not in failure.get("error", "")):
        raise ValueError("S2-P17失败状态、299/3960预算或边界故障签名不符")
    old_root = inside(job["source_root"], store.root / "source_snapshots")
    verified_manifest(old_root, S2P17_ORIGIN_COMMIT)
    validation_path = inside(validation_path, directory / "recoveries")
    if digest(validation_path) != validation_sha256:
        raise ValueError("S2-P17修复验证回执SHA不符")
    validation = read_json(validation_path, {})
    new_root, new_commit = store.freeze_source()
    if new_commit != expected_repair_commit or new_commit == S2P17_ORIGIN_COMMIT:
        raise ValueError("运行中的修复源码不是指定干净提交")
    source_proof = verify_s2p17_boundary_chain(old_root, new_root, new_commit, validation)
    case_dir = directory / "power_cases"
    methods = ("none", "sca", "sca_si_lp")
    expected_names = [f"scene_{index:05d}_r{repeat}_{method}.json"
                      for index in range(440) for repeat in range(3) for method in methods][:299]
    found_names = {path.name for path in case_dir.glob("*.json")}
    if found_names != set(expected_names) or len(expected_names) != 299 or \
            (case_dir / (S2P17_FAILED_CASE + ".json")).exists():
        raise ValueError("已提交SCA案例不是连续299件，或失败B已被写入")
    inventory = [{"name": name, "sha256": digest(case_dir / name)} for name in expected_names]
    inventory_sha = _canonical_sha(inventory)
    prior_manifest = read_json(directory / "refinement_manifest.json", {})
    checkpoint = read_json(directory / "refinement_checkpoint.json", {})
    a_path = case_dir / "scene_00033_r0_sca.json"
    a_sha = digest(a_path)
    a_case = read_json(a_path, {})
    if (prior_manifest.get("budget_cases") != 3960 or
            checkpoint.get("identity") != prior_manifest.get("identity") or
            checkpoint.get("completed_cases") != 299 or
            checkpoint.get("last_case_sha256") != a_sha or
            Path(checkpoint.get("last_case", "")).resolve() != a_path.resolve() or
            a_case.get("method") != "sca" or a_case.get("repeat") != 0):
        raise ValueError("S2-P17细调manifest/299断点/A参照不一致")
    iterations_source = directory / "power_iterations.jsonl"
    callback_rows = []
    with iterations_source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("method") == "sca_si_lp" and row.get("repeat") == 0 and
                    row.get("scene_id") == a_case.get("scene_id")):
                callback_rows.append(row)
    last = callback_rows[-1] if callback_rows else {}
    if (len(callback_rows) != 49 or last.get("evaluation") != 684 or
            not math.isclose(float(last.get("elapsed_seconds", -1)),
                             1.0373822273686528, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("失败B原接受回调下界与冻结日志不一致")
    archive = inside(directory / "recoveries" /
                     (S2P17_BOUNDARY_REPAIR_ID + "_" + new_commit[:12]), directory)
    if archive.exists():
        prior_proof = read_json(archive / "repair_proof.json", {})
        prior_fallback = directory / "failed_attempt_fallback.json"
        if (prior_proof.get("repair_id") != S2P17_BOUNDARY_REPAIR_ID or
                prior_proof.get("repair_commit") != new_commit or
                prior_proof.get("old_manifest_sha256") != source_proof["old_manifest_sha256"] or
                prior_proof.get("new_manifest_sha256") != source_proof["new_manifest_sha256"] or
                prior_proof.get("original_299_inventory_sha256") != inventory_sha or
                prior_proof.get("original_A_case_sha256") != a_sha or
                prior_proof.get("validation_receipt_sha256") != validation_sha256 or
                not prior_fallback.is_file() or
                digest(prior_fallback) != prior_proof.get("fallback_contract_sha256") or
                digest(prior_proof.get("validation_receipt_path", "")) != validation_sha256 or
                any(digest(archive / "original/power_cases" / item["name"]) != item["sha256"]
                    for item in inventory)):
            raise ValueError("修复证据目录为未完成或不同的API准备，保留现场并拒绝覆盖")
        updated = deepcopy(job)
        updated.setdefault("original_git_commit", S2P17_ORIGIN_COMMIT)
        updated.setdefault("source_history", []).append(prior_proof)
        updated.update(source_root=str(new_root), git_commit=new_commit,
                       boundary_repair=prior_proof,
                       failed_attempt_fallback_path=str(prior_fallback),
                       scientific_origin_commit=S2P17_ORIGIN_COMMIT)
        return updated
    archive.mkdir(parents=True)
    original = archive / "original"
    copied = {}
    for name in ("failure.json", "status.json", "job.json", "config.json",
                 "worker.log", "power_iterations.jsonl", "events.jsonl",
                 "refinement_manifest.json", "refinement_checkpoint.json"):
        source = directory / name
        if source.is_file():
            target = original / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum = digest(source)
            if digest(target) != checksum:
                raise RuntimeError("S2-P17恢复前证据副本SHA不符：" + name)
            copied[name] = {"path": str(target), "sha256": checksum}
    for name in expected_names:
        source = case_dir / name
        target = original / "power_cases" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != digest(source):
            raise RuntimeError("S2-P17原299件案例备份SHA不符")
    for source in sorted((directory / "initial_allocations").glob("*.json")):
        target = original / "initial_allocations" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != digest(source):
            raise RuntimeError("S2-P17初解副本SHA不符")
    validation_copy = original / "validation_receipt.json"
    shutil.copy2(validation_path, validation_copy)
    if digest(validation_copy) != validation_sha256:
        raise RuntimeError("S2-P17验证回执证据副本SHA不符")
    if not all(name in copied for name in ("failure.json", "status.json", "job.json",
                                           "power_iterations.jsonl", "refinement_manifest.json",
                                           "refinement_checkpoint.json")):
        raise ValueError("S2-P17失败现场缺少必要原文件")
    if digest(Path(copied["power_iterations.jsonl"]["path"])) != digest(iterations_source):
        raise RuntimeError("冻结失败尝试日志在证据复制时改变")
    contract = {
        "repair_type": S2P17_BOUNDARY_REPAIR_ID,
        "origin_science_commit": S2P17_ORIGIN_COMMIT,
        "original_run_id": S2P17_FAILED_RUN,
        "request_id": S2P17_FAILED_REQUEST,
        "case_key": S2P17_FAILED_CASE,
        "scene_sha": a_case["source_scene_sha256"],
        "initial_sha": a_case["initial_allocation_sha256"],
        "A_case_path": str(a_path), "A_case_sha256": a_sha,
        "A_case_backup_path": str(original / "power_cases" / a_path.name),
        "failure_path": copied["failure.json"]["path"],
        "failure_sha256": copied["failure.json"]["sha256"],
        "iterations_path": copied["power_iterations.jsonl"]["path"],
        "iterations_sha256": copied["power_iterations.jsonl"]["sha256"],
        "callback_lower_bounds": {"capacity_calls": 684,
                                  "elapsed_seconds": 1.0373822273686528,
                                  "accepted_callbacks": 49},
        "unknown_exact_fields": ["original_attempt_capacity_calls",
                                 "original_attempt_optimizer_wall_seconds"],
        "missing_reason": "solver result not persisted after final guard failure",
        "completed_prefix_case_count": 299,
        "completed_prefix_inventory_sha256": inventory_sha,
        "fallback": "failed_attempt_fallback_to_A",
    }
    fallback_path = directory / "failed_attempt_fallback.json"
    if fallback_path.exists():
        raise ValueError("失败回退合同已存在，禁止覆盖或再次准备")
    atomic_json(fallback_path, contract)
    fallback_sha = digest(fallback_path)
    source_proof.update(at=now(), original_run_id=S2P17_FAILED_RUN,
                        original_request_id=S2P17_FAILED_REQUEST,
                        original_status_sha256=copied["status.json"]["sha256"],
                        original_failure_sha256=copied["failure.json"]["sha256"],
                        original_iterations_sha256=copied["power_iterations.jsonl"]["sha256"],
                        original_299_inventory_sha256=inventory_sha,
                        original_299_inventory=inventory,
                        original_A_case_sha256=a_sha,
                        archive_path=str(archive), backup_files=copied,
                        fallback_contract_path=str(fallback_path),
                        fallback_contract_sha256=fallback_sha,
                        validation_receipt_path=str(validation_copy),
                        validation_receipt_sha256=validation_sha256)
    atomic_json(archive / "repair_proof.json", source_proof)
    updated = deepcopy(job)
    updated.setdefault("original_git_commit", S2P17_ORIGIN_COMMIT)
    updated.setdefault("source_history", []).append(source_proof)
    updated.update(source_root=str(new_root), git_commit=new_commit,
                   boundary_repair=source_proof,
                   failed_attempt_fallback_path=str(fallback_path),
                   scientific_origin_commit=S2P17_ORIGIN_COMMIT)
    return updated
