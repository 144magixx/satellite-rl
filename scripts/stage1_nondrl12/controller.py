"""Persistent ND12 workbench coordinator. No scientific worker bypasses the API.

Run plan and dataset registry are immutable inputs. Missing mechanism evidence
requires engineering attention; measured failed gates are scientific stops.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from satellite_rl.coverage.nondrl.runner import worker_lock
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_nondrl import default_nondrl_config, METHODS, MODEL_METHODS
from satellite_rl.workbench.runtime import process_start_time
from satellite_rl.workbench.storage import digest, now, read_json
from scripts.stage1_nondrl12 import ops

PORTS = (20418, 20424, 20473, 20474)
TERMINAL = {"completed", "stopped", "failed", "interrupted", "cancelled"}
PHASES = ("initial", "development", "teacher", "train", "test", "report_pending")


def stable_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_plan(plan, registry):
    if (plan.get("experiment_id", plan.get("group")) != "S1-ND12" or
            registry.get("experiment_id") != "S1-ND12" or
            not isinstance(plan.get("scientific_commit"), str) or len(plan["scientific_commit"]) != 40 or
            registry.get("scientific_commit") != plan["scientific_commit"]):
        raise ValueError("plan/registry group or scientific source differs")
    if set(registry.get("lanes", {})) != {str(p) for p in PORTS}:
        raise ValueError("registry must cover four authorized lanes")
    for role in ("train", "development", "validation", "test"):
        records = [registry["lanes"][str(p)].get(role, {}) for p in PORTS]
        if any(not r.get("id") or not r.get("path") or len(r.get("sha256", "")) != 64 or
               type(r.get("count")) is not int or r["count"] <= 0 for r in records):
            raise ValueError(f"registry {role} partition not ready")
        if len({(r["sha256"], r["count"]) for r in records}) != 1:
            raise ValueError(f"registry {role} differs across lanes")
    if len(registry.get("partition_manifest_sha256", "")) != 64:
        raise ValueError("registry lacks frozen partition manifest SHA")
    if set(plan.get("methods", {})) != set(METHODS) or set(plan.get("gate_thresholds", {})) != set(METHODS):
        raise ValueError("plan must freeze methods and gates for every route")
    for route, methods in plan["methods"].items():
        if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or any(m not in METHODS[route] + ("I0", "BH") for m in methods):
            raise ValueError(f"invalid frozen method matrix for {route}")
        gate = plan["gate_thresholds"][route]
        if (not isinstance(gate.get("development_methods"), list) or not gate["development_methods"] or
                any(m not in METHODS[route] + ("I0", "BH") or m in MODEL_METHODS.get(route, ()) for m in gate["development_methods"]) or
                gate.get("primary_method") not in gate["development_methods"] or
                type(gate.get("minimum_improved_mothers")) is not int or
                not 0 <= gate["minimum_improved_mothers"] <= 20 or
                not isinstance(gate.get("relative_cv_improvement"), (int, float))):
            raise ValueError(f"invalid or learned-before-training development gate {route}")
    return plan


class Controller:
    def __init__(self, directory, plan, registry, transport=ops):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan, self.registry, self.transport = validate_plan(plan, registry), registry, transport
        self.plan_sha, self.registry_sha = stable_sha(plan), stable_sha(registry)
        self.path = self.root / "state.json"
        self.state = read_json(self.path) or {"experiment_id": "S1-ND12", "phase": "initial", "jobs": {},
                                           "routes": {}, "plan_sha256": self.plan_sha, "registry_sha256": self.registry_sha,
                                           "created_at": now()}
        if self.state["plan_sha256"] != self.plan_sha or self.state["registry_sha256"] != self.registry_sha:
            raise ValueError("existing controller belongs to another frozen plan/registry")
        self.state.update(pid=os.getpid(), process_start_time=process_start_time(os.getpid()))
        self.save()

    def save(self):
        self.state["heartbeat_at"] = now()
        atomic_json(self.path, self.state)

    def event(self, event_name, **fields):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "event": event_name, **fields}, ensure_ascii=False, allow_nan=False) + "\n")

    def attention(self, reason, **fields):
        self.state.update(phase="attention_required", attention={"reason": reason, "at": now(), **fields})
        self.event("attention_required", reason=reason, **fields)
        self.save()

    def record(self, port, role):
        return self.registry["lanes"][str(port)][role]

    def initial_refs(self, role):
        return [{"port": str(j["port"]), "run_id": j["run_id"], "route": "06"}
                for key, j in sorted(self.state["jobs"].items()) if key.startswith(f"initial/{role}/")]

    def body(self, route, method, kind, role, port, **extra):
        config = default_nondrl_config(route)
        config["coverage"]["nondrl"]["method"] = method
        config["name"] = f"S1-ND12 {route}/{method} {kind} {role or 'supervised'}"
        result = {"config": config, "kind": kind, "experiment_role": "verification" if kind == "evaluate" and role == "development" else "formal"}
        if kind == "train":
            teacher_role = "development" if route == "05" else "train"
            result["dataset_ids"] = {r: self.record(port, r)["id"] for r in (teacher_role, "validation")}
        else:
            result["dataset_ids"] = {role: self.record(port, role)["id"]}
        if kind == "evaluate" or extra.get("preparation_mode") == "initial":
            result["dataset_role"] = role
        if kind == "evaluate":
            result["initial_source"] = self.initial_refs(role)
        result.update(extra)
        return result

    def ensure(self, key, port, body):
        expected = stable_sha(body)
        existing = self.state["jobs"].get(key)
        if existing:
            if existing["body_sha256"] != expected or existing["port"] != port:
                raise ValueError("planned task body changed after publication")
            return existing
        body = {**body, "request_id": "nd12_" + stable_sha({"plan": self.plan_sha, "key": key})[:40]}
        job = self.transport.api(port, "/api/coverage/nondrl/runs", body)
        if job.get("git_commit") != self.plan["scientific_commit"]:
            self.attention("new job scientific source differs; no further publication", key=key, run_id=job.get("run_id"))
            return None
        record = {"port": port, "run_id": job["run_id"], "body_sha256": expected, "kind": body["kind"],
                  "role": body.get("dataset_role"), "route": body["config"]["coverage"]["nondrl"]["route"],
                  "method": body["config"]["coverage"]["nondrl"]["method"], "stage": "queued", "retries": 0}
        self.state["jobs"][key] = record
        self.event("published", key=key, **record)
        self.save()
        return record

    def refresh(self):
        snapshots = {}
        for port in PORTS:
            snapshots[port] = {r["run_id"]: r for r in self.transport.api(port, "/api/coverage/nondrl/runs")["runs"]}
        for key, record in self.state["jobs"].items():
            row = snapshots[record["port"]].get(record["run_id"])
            if not row:
                self.attention("published workbench job missing", key=key)
                return False
            record.update(stage=row.get("stage"), scientific_stop=row.get("scientific_stop", False),
                          stop_reason=row.get("stop_reason"), last_heartbeat=row.get("heartbeat_at"),
                          pid=row.get("pid"), completed_scenes=row.get("completed_scenes"), updates=row.get("updates"))
            if row.get("stage") in ("failed", "interrupted"):
                if record["retries"] >= 2:
                    self.attention("same-ID recovery bound reached", key=key, status=row)
                    return False
                try:
                    self.transport.api(record["port"], f"/api/runs/{record['run_id']}/resume", {})
                except Exception as error:
                    self.attention("job requires diagnosis before safe same-ID recovery", key=key, error=str(error))
                    return False
                record.update(retries=record["retries"] + 1, stage="queued")
                self.event("same_id_resume", key=key, retry=record["retries"])
            if row.get("stage") in ("stopped", "cancelled") and row.get("scientific_stop") is not True:
                self.attention("workbench task explicitly stopped; continuation needs diagnosis", key=key, status=row)
                return False
        self.save()
        return True

    def finished(self, prefix):
        rows = [j for k, j in self.state["jobs"].items() if k.startswith(prefix)]
        return bool(rows) and all(j["stage"] == "completed" for j in rows)

    def initial(self):
        for role in ("development", "validation", "test", "train"):
            count = self.record(PORTS[0], role)["count"]
            for lane, port in enumerate(PORTS):
                indices = list(range(lane, count, len(PORTS)))
                if indices:
                    self.ensure(f"initial/{role}/{port}", port, self.body("06", "I0", "prepare", role, port,
                        preparation_mode="initial", scene_indices=indices))
                if self.state["phase"] == "attention_required":
                    return
        if self.finished("initial/development/") and self.finished("initial/validation/"):
            self.state["phase"] = "development"

    def development(self):
        count = self.record(PORTS[0], "development")["count"]
        for route in METHODS:
            for method in self.plan["gate_thresholds"][route]["development_methods"]:
                for lane, port in enumerate(PORTS):
                    self.ensure(f"development/{route}/{method}/{port}", port,
                        self.body(route, method, "evaluate", "development", port,
                                  scene_indices=list(range(lane, count, len(PORTS)))))
                    if self.state["phase"] == "attention_required":
                        return
        if not self.finished("development/"):
            return
        for route in METHODS:
            if route in self.state["routes"]:
                continue
            gate = self.make_gate(route)
            if gate is None:
                return
            self.state["routes"][route] = gate
            self.save()
        self.state["phase"] = "teacher"

    def summaries(self, jobs):
        """Read only compact recomputation from the published workbench raw shards."""
        output = []
        for record in jobs:
            run_id = record["run_id"]
            code = f'''import json,gzip,hashlib
from pathlib import Path
d=Path({(ops.BASE+'/'+str(record['port'])+'/archive/outputs/workbench/jobs/'+run_id)!r})
m=json.loads((d/'raw_scenes/manifest.json').read_text())
rows=[]
for x in m['rows']:
 p=d/'raw_scenes'/x['file']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']: raise ValueError('raw scene SHA differs')
 r=json.loads(gzip.decompress(p.read_bytes()))
 z=r['result']; a=z['metrics_all']; b=z['initial_metrics_all']
 rows.append({{'index':r['dataset_index'],'multiplier':r['demand_multiplier'],'mother_id':r['mother_scene_id'],'cv':a['load_cv_all'],'initial_cv':b['load_cv_all'],'carry':z['physical']['aircraft_service']['demand_weighted_carried_fraction'],'initial_carry':z['initial_physical']['aircraft_service']['demand_weighted_carried_fraction'],'feasible':all(a.get(k) is True for k in ['geometry_feasible','unique_assignment','demand_conserved']),'initial_sha256':z['initial_sha256'],'final_sha256':z['final_sha256']}})
print(json.dumps({{'run_id':{run_id!r},'port':{record['port']},'manifest_path':str(d/'raw_scenes/manifest.json'),'manifest_sha256':hashlib.sha256((d/'raw_scenes/manifest.json').read_bytes()).hexdigest(),'source_commit':m['provenance']['git_commit'],'config_sha256':m['provenance']['config_sha256'],'rows':rows}}))
'''
            receipt = self.transport.remote_python(code, record["port"], timeout=120)
            if receipt["exit_code"]:
                raise ValueError("raw gate recomputation failed: " + receipt["stderr"][-500:])
            output.append(json.loads(receipt["stdout"].splitlines()[-1]))
        return output

    def make_gate(self, route):
        threshold = self.plan["gate_thresholds"][route]
        keys = [j for k, j in self.state["jobs"].items() if k.startswith(f"development/{route}/{threshold['primary_method']}/")]
        sources = self.summaries(keys)
        rows = [r for s in sources for r in s["rows"]]
        expected_count = self.record(PORTS[0], "development")["count"]
        expected = {(i, m) for i in range(expected_count) for m in (.75, 1., 1.25)}
        if ({(r["index"], r["multiplier"]) for r in rows} != expected or len(rows) != len(expected) or
                any(s["source_commit"] != self.plan["scientific_commit"] for s in sources)):
            self.attention("development evidence incomplete or source differs", route=route)
            return None
        mothers = []
        for index in range(expected_count):
            group = [r for r in rows if r["index"] == index]
            cvs = [r["cv"] for r in group if r["cv"] is not None]
            base = [r["initial_cv"] for r in group if r["initial_cv"] is not None]
            carry = [r["carry"] - r["initial_carry"] for r in group if r["carry"] is not None and r["initial_carry"] is not None]
            baseline, final = sum(base) / len(base) if base else None, sum(cvs) / len(cvs) if cvs else None
            improvement = (baseline - final) / baseline if baseline is not None and baseline > 1e-12 else None
            mothers.append({"index": index, "mother_id": group[0]["mother_id"], "initial_cv": baseline,
                            "cv": final, "relative_improvement": improvement,
                            "carry_delta": sum(carry) / len(carry) if carry else None,
                            "feasible": all(r["feasible"] for r in group)})
        improved = [m["index"] for m in mothers if m["relative_improvement"] is not None and
                    m["relative_improvement"] > threshold["relative_cv_improvement"] and
                    m["feasible"] and (m["carry_delta"] is None or m["carry_delta"] >= -.005)]
        geometry_pass = all(m["feasible"] for m in mothers)
        passed = geometry_pass and len(improved) >= threshold["minimum_improved_mothers"]
        # Oracle-region and fixed ranking labels require their own actual audit.
        # A missing audit is an engineering gap, never a measured scientific fail.
        audit = threshold.get("mechanism_audit")
        if passed and (threshold.get("require_mechanism_audit") or threshold.get("required_mechanism")):
            if not isinstance(audit, dict) or not audit.get("path") or not audit.get("sha256"):
                self.attention("missing route mechanism audit; repair audit interface", route=route)
                return None
            receipt = self.transport.remote_python(f"import json,hashlib; from pathlib import Path; p=Path({audit['path']!r}); assert hashlib.sha256(p.read_bytes()).hexdigest()=={audit['sha256']!r}; print(p.read_text())", PORTS[0])
            actual = json.loads(receipt["stdout"])
            if actual.get("route") != route or actual.get("experiment_id") != "S1-ND12" or "passed" not in actual:
                self.attention("invalid mechanism audit receipt", route=route)
                return None
            passed = passed and actual["passed"] is True
        result = {"experiment_id": "S1-ND12", "route": route, "passed": passed,
                  "status": "passed" if passed else "science_stopped", "source_commit": self.plan["scientific_commit"],
                  "partition_manifest_sha256": self.registry["partition_manifest_sha256"],
                  "dataset_sha256": self.record(PORTS[0], "development")["sha256"], "mother_count": len(mothers),
                  "expected_row_count": len(expected), "actual_row_count": len(rows), "improved_mothers": improved,
                  "geometry_pass": geometry_pass, "mother_metrics": mothers, "sources": sources,
                  "threshold": threshold, "created_at": now()}
        path = self.root / "gates" / f"route_{route}.json"
        atomic_json(path, result)
        remote = ops.BASE + "/source/gates/" + path.name
        self.transport.upload(path, remote, PORTS[0])
        result["gate_source"] = {"path": remote, "sha256": digest(path)}
        self.event("development_gate", route=route, passed=passed, improved_mothers=improved)
        return result

    def teacher(self):
        for index, route in enumerate(("01", "02", "03", "04", "05")):
            if not self.state["routes"][route]["passed"]:
                continue
            port = PORTS[index % len(PORTS)]
            role = "development" if route == "05" else "train"
            self.ensure(f"teacher/{route}", port, self.body(route, "main", "prepare", role, port, preparation_mode="teacher"))
            if self.state["phase"] == "attention_required":
                return
        rows = [j for k, j in self.state["jobs"].items() if k.startswith("teacher/")]
        if all(j["stage"] in TERMINAL for j in rows):
            for j in rows:
                if j.get("scientific_stop"):
                    self.state["routes"][j["route"]].update(status="science_stopped", passed=False, stop_reason=j.get("stop_reason"))
            self.state["phase"] = "train"

    def training(self):
        if any(self.state["routes"][r]["passed"] for r in ("03", "04")) and not self.plan.get("pilot_audit_implementation_receipt"):
            self.attention("03/04 pilot telemetry implementation receipt missing; complete engineering before publication")
            return
        for route in ("01", "02", "03", "04", "05"):
            if not self.state["routes"][route]["passed"]:
                continue
            teacher = self.state["jobs"][f"teacher/{route}"]
            port = teacher["port"]
            self.ensure(f"train/{route}", port, self.body(route, "main", "train", None, port,
                teacher_source={"port": str(port), "run_id": teacher["run_id"]},
                gate_source=self.state["routes"][route]["gate_source"],
                validation_initial_source=self.initial_refs("validation")))
            if self.state["phase"] == "attention_required":
                return
        rows = [j for k, j in self.state["jobs"].items() if k.startswith("train/")]
        if all(j["stage"] in TERMINAL for j in rows):
            for j in rows:
                if j.get("scientific_stop"):
                    self.state["routes"][j["route"]].update(status="science_stopped", passed=False, stop_reason=j.get("stop_reason"))
            self.state["phase"] = "test"

    def testing(self):
        count = self.record(PORTS[0], "test")["count"]
        # One test task per lane at a time fixes concurrency for paired timings.
        # Training and preparation previously use all scheduler concurrency.
        for lane, port in enumerate(PORTS):
            if any(j["port"] == port and k.startswith("test/") and j["stage"] not in TERMINAL for k, j in self.state["jobs"].items()):
                continue
            for route, methods in self.plan["methods"].items():
                if not self.state["routes"][route]["passed"]:
                    continue
                for method in methods:
                    key = f"test/{route}/{method}/{port}"
                    if key in self.state["jobs"]:
                        continue
                    extra = {"scene_indices": list(range(lane, count, len(PORTS)))}
                    if method in MODEL_METHODS.get(route, ()):
                        model = self.state["jobs"][f"train/{route}"]
                        extra["model_source"] = {"port": str(model["port"]), "run_id": model["run_id"]}
                    self.ensure(key, port, self.body(route, method, "evaluate", "test", port, **extra))
                    break
                else:
                    continue
                break
        expected = sum(len(methods) * len(PORTS) for r, methods in self.plan["methods"].items() if self.state["routes"][r]["passed"])
        rows = [j for k, j in self.state["jobs"].items() if k.startswith("test/")]
        if len(rows) == expected and all(j["stage"] == "completed" for j in rows):
            self.state.update(phase="report_pending", completed_at=now(), report_scope="actual passed/failed routes; no fabricated endpoints")
            self.event("report_pending", test_tasks=len(rows), routes=self.state["routes"])

    def tick(self):
        if self.state["phase"] in ("attention_required", "report_pending", "stopped"):
            return self.state
        if not self.refresh():
            return self.state
        phase = self.state["phase"]
        {"initial": self.initial, "development": self.development, "teacher": self.teacher,
         "train": self.training, "test": self.testing}[phase]()
        self.save()
        return self.state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    args.directory.mkdir(parents=True, exist_ok=True)
    with worker_lock(args.directory):
        controller = Controller(args.directory, read_json(args.plan), read_json(args.registry))
        while True:
            try:
                state = controller.tick()
            except Exception as error:
                controller.attention("controller exception; retain current tasks", error=repr(error))
                state = controller.state
            print(json.dumps({"at": now(), "phase": state["phase"], "jobs": len(state["jobs"])}, ensure_ascii=False), flush=True)
            if args.once or state["phase"] in ("report_pending", "attention_required", "stopped"):
                return
            if (args.directory / "stop.request").exists():
                controller.state.update(phase="stopped", stop_reason="controller_user_stop")
                controller.save()
                return
            time.sleep(20)


if __name__ == "__main__":
    main()
