"""Independent S1-03-R1 controller: reuse audited starts, train once from zero.

This is an orchestrator, not a scientific worker. It submits jobs only through
the new s1_03_r1 workbench API and never resumes or changes original S1-03 jobs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time

try:
    from . import controller as old
    from . import remote
except ImportError:
    import controller as old
    import remote


GROUP = "s1_03_r1"
OLD_GROUP = "s1_03"
OLD_COMMIT = "e1745eccc1f47aa0b2f4ede678ce7789e030c1ea"
OLD_BASE = "/home/u2024110224/first_stage_satellite_rl/s1_03"
NEW_BASE = "/home/u2024110224/first_stage_satellite_rl/s1_03_r1"
ROLE_COUNTS = {"train": 576, "validation": 20, "test": 100}
SOURCE_TARGET = "u2024110224@10.160.17.102:20418"
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_r1_plan(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw.decode("utf-8-sig"))
    if (plan.get("experiment_id") != "S1-03-R1" or plan.get("hostroot") != NEW_BASE or
            plan.get("ports") != [20418, 20424] or
            not COMMIT.fullmatch(plan.get("science_commit", "")) or
            plan["science_commit"] == OLD_COMMIT or
            plan.get("gates") != old.GATES):
        raise ValueError("R1 plan source/host/approved budget gate differs")
    template = plan.get("config_template", {})
    training = template.get("training", {})
    if (template.get("schema_version") != 4 or template.get("research_stage") != "coverage" or
            training.get("seed") != 51001 or training.get("total_steps") != 262144 or
            training.get("mechanism_gate_step") != 65536):
        raise ValueError("R1 train must be a new full-budget S1-03 policy")
    if not all(mode in plan.get("audit_commands", {}) for mode in
               ("development", "mechanism", "formal")) or not plan.get("s2_release", {}).get(
                   "controller_manifest") or plan["s2_release"].get("allow_file") is not None:
        raise ValueError("R1 requires frozen audits and live S2 release, without manual bypass")
    for mode in ("development", "mechanism", "formal"):
        command = plan["audit_commands"][mode]
        port = 20424 if mode == "formal" else 20418
        expected_script = (f"{NEW_BASE}/{port}/repo/scripts/"
                           "stage1_reconstruction/audit.py")
        if (not isinstance(command, dict) or not isinstance(command.get("argv"), list) or
                len(command["argv"]) < 2 or
                not all(isinstance(arg, str) for arg in command["argv"]) or
                command["argv"][0] != remote.DQN or
                command["argv"][1] != expected_script or
                not SHA.fullmatch(command.get("script_sha256", ""))):
            raise ValueError(f"R1 {mode} audit must use DQN and this group's frozen repo script")
    reuse = plan.get("source_reuse", {})
    if (reuse.get("source_group") != OLD_GROUP or reuse.get("source_target") != "20418" or
            reuse.get("old_science_commit") != OLD_COMMIT or
            not SHA.fullmatch(reuse.get("old_plan_sha256", "")) or
            not SHA.fullmatch(reuse.get("old_source_manifest_sha256", "")) or
            not SHA.fullmatch(reuse.get("new_source_manifest_sha256", ""))):
        raise ValueError("R1 reuse source is not frozen old S1-03/20418")
    roles = reuse.get("roles", {})
    if set(roles) != set(ROLE_COUNTS):
        raise ValueError("R1 must bind all three old data/start partitions")
    for role, record in roles.items():
        if (record.get("scene_count") != ROLE_COUNTS[role] or
                not isinstance(record.get("source_dataset_id"), str) or
                not isinstance(record.get("source_run_id"), str) or
                not SHA.fullmatch(record.get("dataset_sha256", "")) or
                not SHA.fullmatch(record.get("starts_sha256", "")) or
                type(record.get("qualified_count")) is not int or
                not 0 <= record["qualified_count"] <= ROLE_COUNTS[role]):
            raise ValueError(f"R1 {role} source data/start/qualification contract differs")
    if roles["train"]["qualified_count"] != 572:
        raise ValueError("R1 training qualification must remain 572/576")
    audit = reuse.get("audit", {})
    audit_path = PurePosixPath(audit.get("path", ""))
    if (not audit_path.is_absolute() or
            not str(audit_path).startswith(NEW_BASE + "/20418/ops/") or
            not SHA.fullmatch(audit.get("sha256", ""))):
        raise ValueError("R1 audit must be a SHA-frozen receipt in new 20418 ops")
    return plan, hashlib.sha256(raw).hexdigest()


class RerunController(old.Controller):
    def __init__(self, plan, plan_sha256, state_path):
        super().__init__(plan, plan_sha256, state_path)
        self.state.setdefault("source_reuse", None)

    def api(self, port, endpoint, body=None, *, timeout=120, retry=None):
        if retry is None:
            retry = 3 if body is None or endpoint == "coverage/runs" else 1
        return self._remote(port, lambda client: remote.api(
            client, port, endpoint, body, timeout=timeout, group=GROUP), retry=retry)

    def _verify_reuse(self):
        reuse = self.plan["source_reuse"]
        proof = reuse["audit"]
        observation = self._read_remote_json(20418, proof["path"])
        if not observation.get("exists"):
            self.state["phase"] = "awaiting_reuse_audit"
            return False
        receipt = observation.get("value") or {}
        if (observation.get("sha256") != proof["sha256"] or
                receipt.get("passed") is not True or receipt.get("issues") or
                receipt.get("source_group") != OLD_GROUP or
                receipt.get("source_target") != "20418" or
                receipt.get("old_science_commit") != OLD_COMMIT or
                receipt.get("new_science_commit") != self.plan["science_commit"] or
                receipt.get("old_plan_sha256") != reuse["old_plan_sha256"] or
                receipt.get("old_source_manifest_sha256") != reuse["old_source_manifest_sha256"] or
                receipt.get("new_source_manifest_sha256") != reuse["new_source_manifest_sha256"] or
                receipt.get("total_scenes") != 696 or
                receipt.get("training_qualified") != 572):
            raise old.Attention("R1 696-scene old/new source reuse audit failed or changed")
        audited_roles = receipt.get("roles", {})
        if set(audited_roles) != set(ROLE_COUNTS):
            raise old.Attention("R1 reuse audit lacks a scene partition")
        for role, expected in reuse["roles"].items():
            row = audited_roles[role]
            dataset = receipt.get("datasets", {}).get(role, {})
            start = receipt.get("starts", {}).get(role, {})
            if (dataset.get("source_dataset_id") != expected["source_dataset_id"] or
                    dataset.get("sha256") != expected["dataset_sha256"] or
                    dataset.get("scene_count") != expected["scene_count"] or
                    start.get("source_run_id") != expected["source_run_id"] or
                    start.get("sha256") != expected["starts_sha256"] or
                    start.get("dataset_sha256") != expected["dataset_sha256"] or
                    row.get("source_dataset_id") != expected["source_dataset_id"] or
                    row.get("source_run_id") != expected["source_run_id"] or
                    row.get("issues")):
                raise old.Attention(f"R1 {role} reuse data/start provenance differs")
            for plan_name, audit_name in (("source_run_id", "prepare_run_id"),
                                          ("dataset_sha256", "dataset_sha256"),
                                          ("starts_sha256", "starts_sha256"),
                                          ("scene_count", "scene_count"),
                                          ("qualified_count", "qualified_count")):
                if row.get(audit_name) != expected[plan_name]:
                    raise old.Attention(f"R1 {role} audited source/start identity differs")
            if (row.get("failed_count") != expected["scene_count"] -
                    expected["qualified_count"] or
                    not isinstance(row.get("scenes"), list) or
                    len(row["scenes"]) != expected["scene_count"]):
                raise old.Attention(f"R1 {role} physical qualification denominator differs")
        code = f'''
import hashlib,json
from pathlib import Path
paths={{'old':{OLD_BASE!r}+'/20418/outputs/workbench/source_snapshots/'+{OLD_COMMIT!r}+'/source_manifest.json',
       'new':{NEW_BASE!r}+'/20418/outputs/workbench/source_snapshots/'+{self.plan['science_commit']!r}+'/source_manifest.json'}}
out={{}}
for name,path in paths.items():
 p=Path(path)
 if not p.is_file():raise FileNotFoundError(path)
 out[name]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out))
'''
        manifests = self.code(20418, code)
        if (manifests.get("old") != reuse["old_source_manifest_sha256"] or
                manifests.get("new") != reuse["new_source_manifest_sha256"]):
            raise old.Attention("R1 live old/new frozen source manifests differ from reuse audit")
        recorded = {"observed_utc": old.utcnow(), "audit_path": proof["path"],
                    "audit_sha256": proof["sha256"], "old_group": OLD_GROUP,
                    "old_science_commit": OLD_COMMIT,
                    "new_science_commit": self.plan["science_commit"],
                    "total_scenes": 696, "training_qualified": 572}
        self.state["source_reuse"] = recorded
        return True

    def _import_role(self, role, port):
        if self.state.get("source_reuse") is None:
            raise old.Attention("R1 cannot import before source reuse audit")
        source = self.plan["source_reuse"]["roles"][role]
        dataset_key = f"{port}:{role}"
        datasets = self.api(port, "coverage/datasets")["datasets"]
        matches = [row for row in datasets
                   if row.get("source_group") == OLD_GROUP and
                   row.get("source_target") == SOURCE_TARGET and
                   row.get("source_dataset_id") == source["source_dataset_id"] and
                   row.get("partition_role") == role]
        if len(matches) > 1:
            raise old.Attention(f"R1 {dataset_key} has duplicate imports")
        dataset = matches[0] if matches else self.api(port,
            "coverage/reconstruction/import-dataset", {
                "source_group": OLD_GROUP, "source_target": "20418",
                "source_dataset_id": source["source_dataset_id"],
                "partition_role": role}, retry=1)
        if (dataset.get("source_group") != OLD_GROUP or
                dataset.get("source_target") != SOURCE_TARGET or
                dataset.get("source_dataset_id") != source["source_dataset_id"] or
                dataset.get("sha256") != source["dataset_sha256"] or
                dataset.get("count") != ROLE_COUNTS[role]):
            raise old.Attention(f"R1 {dataset_key} imported data SHA/source differs")
        self.state["datasets"][dataset_key] = dataset["id"]
        starts = self.api(port, "coverage/reconstruction/imports")["starts"]
        candidates = [row for row in starts
                      if row.get("source_group") == OLD_GROUP and
                      row.get("source_target") == SOURCE_TARGET and
                      row.get("source_run_id") == source["source_run_id"] and
                      row.get("partition_role") == role]
        if len(candidates) > 1:
            raise old.Attention(f"R1 {dataset_key} has duplicate start imports")
        start = candidates[0] if candidates else self.api(port,
            "coverage/reconstruction/import-start", {
                "source_group": OLD_GROUP, "source_target": "20418",
                "source_run_id": source["source_run_id"],
                "partition_role": role}, retry=1)
        if (start.get("source_group") != OLD_GROUP or
                start.get("source_target") != SOURCE_TARGET or
                start.get("source_run_id") != source["source_run_id"] or
                start.get("sha256") != source["starts_sha256"] or
                start.get("dataset_sha256") != source["dataset_sha256"]):
            raise old.Attention(f"R1 {dataset_key} imported start SHA/source differs")
        self.state["imports"][f"{dataset_key}:start"] = start["id"]
        self.state["observed"][f"reuse_{dataset_key}"] = {
            "observed_utc": old.utcnow(), "dataset_id": dataset["id"],
            "dataset_sha256": dataset["sha256"], "start_id": start["id"],
            "starts_sha256": start["sha256"],
            "source_dataset_id": source["source_dataset_id"],
            "source_run_id": source["source_run_id"]}
        return True

    def _start_reference(self, role, port):
        return self.state["imports"][f"{port}:{role}:start"]

    def _import_model_20424(self):
        key = "P1_model_20424"
        source_run_id = self.state["jobs"]["train_P1"]
        models = self.api(20424, "coverage/models")["models"]
        matches = [row for row in models if row.get("source_group") == GROUP and
                   row.get("source_run_id") == source_run_id and
                   row.get("source_target") == SOURCE_TARGET]
        if len(matches) > 1:
            raise old.Attention("R1 P1 model imported more than once")
        model = matches[0] if matches else self.api(20424,
            "coverage/reconstruction/import-model", {
                "source_group": GROUP, "source_target": "20418",
                "source_run_id": source_run_id}, retry=1)
        if (model.get("source_group") != GROUP or
                model.get("source_target") != SOURCE_TARGET or
                model.get("source_run_id") != source_run_id or
                not SHA.fullmatch(model.get("sha256", ""))):
            raise old.Attention("R1 P1 model origin/SHA differs")
        self.state["imports"][key] = model["id"]
        return True

    def _lane_20418(self):
        if not self._verify_reuse():
            return
        for role in ("train", "validation"):
            self._import_role(role, 20418)
        self.state["phase"] = "development"
        complete = []
        for route in ("R0", "R1", "R2", "R3"):
            complete.append(self._job(f"dev_{route}", 20418, route, "evaluate", "validation"))
        if not all(complete):
            return
        if "development" not in self.state["audits"]:
            self.state["phase"] = "development_audit"
            if self._run_audit("development", 20418) is None:
                return
            self.save()
        self.state["phase"] = "training"
        if not self._job("train_P1", 20418, "P1", "train"):
            return
        self.state["phase"] = "training_completed"

    def _lane_20424(self):
        if not self._s2_released():
            self.state["observed"]["s2_waiting_utc"] = old.utcnow()
            return
        if self.state.get("audits", {}).get("development", {}).get("passed") is not True:
            self.state["phase"] = "awaiting_development_audit_for_independent_test"
            return
        if not self._verify_reuse():
            return
        self._import_role("test", 20424)
        self.state["phase"] = "independent_baselines"
        complete = []
        for route in ("R0", "R1", "R2", "R3"):
            complete.append(self._job(f"formal_{route}", 20424, route,
                                      "evaluate", "test"))
        if not all(complete):
            return
        if self.state["observed"].get("train_P1", {}).get("stage") != "completed":
            return
        self._import_model_20424()
        if not self._job("formal_P1", 20424, "P1", "evaluate", "test"):
            return
        if "formal" not in self.state["audits"]:
            self.state["phase"] = "formal_audit"
            if self._run_audit("formal", 20424) is None:
                return
        self.state["phase"] = "awaiting_final_report"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--daemon", action="store_true")
    args = parser.parse_args(argv)
    plan, digest = load_r1_plan(args.plan)
    with old.single_controller_lock(args.state.with_suffix(args.state.suffix + ".lock")):
        controller = RerunController(plan, digest, args.state)
        while True:
            controller.tick()
            print(json.dumps({"observed_utc": old.utcnow(), "phase": controller.state["phase"],
                              "jobs": controller.state["jobs"],
                              "attention": controller.state["attention"]}, ensure_ascii=False), flush=True)
            if args.once or controller.state["phase"] == "awaiting_final_report":
                return 0
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
