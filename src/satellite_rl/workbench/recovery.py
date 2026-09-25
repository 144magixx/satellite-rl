"""Strict, archived recovery of stopped jobs after an equivalent atomic-I/O fix."""

from __future__ import annotations

import ast
from copy import deepcopy
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
