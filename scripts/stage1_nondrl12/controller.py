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
from scripts.stage1_nondrl12.execution_sources import contract_bytes, role_allowed

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
        if route in ("02", "03") and gate.get("teacher_probe_required") is not True:
            raise ValueError("02/03 must require actual strong development teacher proof")
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
        self.source_policy = read_json(self.root / "source_policy.json")
        self.source_contract = None
        if self.source_policy:
            policy = self.source_policy
            payload = Path(policy["local_contract_path"]).read_bytes()
            self.source_contract = contract_bytes(payload, policy["contract_sha256"])
            if self.source_contract["origin_commit"] != plan["scientific_commit"]:
                raise ValueError("source policy belongs to another frozen plan")
            previous = self.state.get("source_policy_sha256")
            if previous and previous != policy["contract_sha256"]:
                raise ValueError("controller source policy cannot change after binding")
            self.state["source_policy_sha256"] = policy["contract_sha256"]
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
        if kind == "evaluate" or extra.get("preparation_mode") in ("initial", "ranking_audit", "teacher_probe"):
            result["dataset_role"] = role
        if kind == "evaluate":
            result["initial_source"] = self.initial_refs(role)
        if kind == "prepare" and extra.get("preparation_mode") in ("teacher", "ranking_audit", "teacher_probe"):
            result["initial_source"] = self.initial_refs(role)
        result.update(extra)
        return result

    def ensure(self, key, port, body):
        expected = stable_sha(body)
        existing = self.state["jobs"].get(key)
        if existing:
            if existing["body_sha256"] != expected or existing["port"] != port:
                raise ValueError("planned task body changed after publication")
            if (self.source_contract and existing["run_id"] in self.source_contract["allowed_run_ids"]
                    and existing.get("source_repair_contract_sha256") != self.source_policy["contract_sha256"]):
                self.verify_execution(existing)
            return existing
        body = {**body, "request_id": "nd12_" + stable_sha({"plan": self.plan_sha, "key": key})[:40]}
        if self.source_contract and role_allowed(self.source_contract, body):
            body["execution_source"] = {k: self.source_policy[k] for k in ("contract_path", "contract_sha256")}
        job = self.transport.api(port, "/api/coverage/nondrl/runs", body)
        expected_commit = self.source_contract["execution_commit"] if body.get("execution_source") else self.plan["scientific_commit"]
        if job.get("git_commit") != expected_commit:
            self.attention("new job scientific source differs; no further publication", key=key, run_id=job.get("run_id"))
            return None
        record = {"port": port, "run_id": job["run_id"], "body_sha256": expected, "kind": body["kind"],
                  "role": body.get("dataset_role"), "route": body["config"]["coverage"]["nondrl"]["route"],
                  "method": body["config"]["coverage"]["nondrl"]["method"], "stage": "queued", "retries": 0}
        record.update(origin_commit=self.plan["scientific_commit"], source_commit=job["git_commit"],
                      publication_body_sha256=stable_sha(body))
        if body.get("execution_source"):
            self.verify_execution(record)
        self.state["jobs"][key] = record
        self.event("published", key=key, **record)
        self.save()
        return record

    def verify_execution(self, record):
        """Actual API detail and on-host immutable bytes, never a state-only waiver."""
        detail = self.transport.api(record["port"], f"/api/runs/{record['run_id']}")
        job = detail["job"]
        if job.get("git_commit") == self.plan["scientific_commit"]:
            if self.source_contract and record["run_id"] in self.source_contract["allowed_run_ids"]:
                raise ValueError("authorized existing repair has not been activated through the API")
            return None
        item = job.get("source_repair_contract", {})
        if (not self.source_contract or job["git_commit"] != self.source_contract["execution_commit"] or
                item.get("contract_sha256") != self.source_policy["contract_sha256"] or
                job.get("origin_commit") != self.plan["scientific_commit"]):
            raise ValueError("actual workbench execution source is outside the pinned policy")
        code = f'''import json,sys
from pathlib import Path
d=Path({(ops.BASE+'/'+str(record['port'])+'/archive/outputs/workbench/jobs/'+record['run_id'])!r})
j=json.loads((d/'job.json').read_text());sys.path.insert(0,str(Path(j['source_root'])/'src'))
from satellite_rl.workbench.nondrl_source_repair import validate_job_lineage
validate_job_lineage(j,d)
assert j['git_commit']=={job['git_commit']!r}
assert j['source_repair_contract']['contract_sha256']=={self.source_policy['contract_sha256']!r}
print(json.dumps({{'run_id':j['run_id'],'execution_commit':j['git_commit'],'contract_sha256':j['source_repair_contract']['contract_sha256'],'passed':True}}))
'''
        result = self.transport.remote_python(code, record["port"], timeout=120)
        if result["exit_code"]:
            raise ValueError("actual repaired job lineage audit failed: " + result["stderr"][-500:])
        proof = json.loads(result["stdout"].splitlines()[-1])
        if (proof.get("passed") is not True or proof.get("run_id") != record["run_id"] or
                proof.get("contract_sha256") != self.source_policy["contract_sha256"]):
            raise ValueError("actual repaired source audit receipt differs")
        record.update(source_commit=job["git_commit"], origin_commit=job["origin_commit"],
                      source_repair_contract_sha256=item["contract_sha256"], source_audit={**proof, "at": now()})
        return detail

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
            code = f'''import json,gzip,hashlib,sys
from pathlib import Path
d=Path({(ops.BASE+'/'+str(record['port'])+'/archive/outputs/workbench/jobs/'+run_id)!r})
job=json.loads((d/'job.json').read_text())
assert job.get('origin_commit',job['git_commit'])=={self.plan['scientific_commit']!r}
if job['git_commit']!={self.plan['scientific_commit']!r}:
 sys.path.insert(0,str(Path(job['source_root'])/'src'))
 from satellite_rl.workbench.nondrl_source_repair import validate_job_lineage
 validate_job_lineage(job,d)
m=json.loads((d/'raw_scenes/manifest.json').read_text())
assert m['provenance']['git_commit'] in (job['git_commit'],job.get('origin_commit',job['git_commit']))
rows=[]
for x in m['rows']:
 p=d/'raw_scenes'/x['file']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']: raise ValueError('raw scene SHA differs')
 r=json.loads(gzip.decompress(p.read_bytes()))
 z=r['result']; a=z['metrics_all']; b=z['initial_metrics_all']
 rows.append({{'index':r['dataset_index'],'multiplier':r['demand_multiplier'],'mother_id':r['mother_scene_id'],'cv':a['load_cv_all'],'initial_cv':b['load_cv_all'],'carry':z['physical']['aircraft_service']['demand_weighted_carried_fraction'],'initial_carry':z['initial_physical']['aircraft_service']['demand_weighted_carried_fraction'],'feasible':all(a.get(k) is True for k in ['geometry_feasible','unique_assignment','demand_conserved']),'initial_sha256':z['initial_sha256'],'final_sha256':z['final_sha256']}})
print(json.dumps({{'run_id':{run_id!r},'port':{record['port']},'manifest_path':str(d/'raw_scenes/manifest.json'),'manifest_sha256':hashlib.sha256((d/'raw_scenes/manifest.json').read_bytes()).hexdigest(),'source_commit':job['git_commit'],'origin_commit':job.get('origin_commit',job['git_commit']),'contract_sha256':job.get('source_repair_contract',{{}}).get('contract_sha256'),'config_sha256':m['provenance']['config_sha256'],'rows':rows}}))
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
                any(s.get("origin_commit",s["source_commit"]) != self.plan["scientific_commit"] for s in sources)):
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
        if route in ("02", "03"):
            measured = self.teacher_reachability(route)
            if measured is None:
                return None
            if measured.get("status") == "attention_required":
                self.attention("development strong teacher evidence requires engineering repair", route=route, receipt=measured)
                return None
            # Rule/online-search results remain diagnostics, rather than
            # standing in for the separately budgeted strong offline teacher.
            passed = measured["passed"] is True
        else:
            measured = None
        # Oracle-region and fixed ranking labels require their own actual audit.
        # A missing audit is an engineering gap, never a measured scientific fail.
        if passed and (threshold.get("require_mechanism_audit") or threshold.get("required_mechanism")):
            actual = self.mechanism(route, keys)
            if actual is None:
                return None
            if actual.get("status") == "attention_required":
                self.attention("route mechanism evidence requires engineering repair", route=route, receipt=actual)
                return None
            passed = passed and actual["passed"] is True
        else:
            actual = None
        result = {"experiment_id": "S1-ND12", "route": route, "passed": passed,
                  "status": "passed" if passed else "science_stopped", "source_commit": self.plan["scientific_commit"],
                  "partition_manifest_sha256": self.registry["partition_manifest_sha256"],
                  "dataset_sha256": self.record(PORTS[0], "development")["sha256"], "mother_count": len(mothers),
                  "expected_row_count": len(expected), "actual_row_count": len(rows), "improved_mothers": improved,
                  "geometry_pass": geometry_pass, "mother_metrics": mothers, "sources": sources,
                  "threshold": threshold, "mechanism": actual, "teacher_reachability": measured, "created_at": now()}
        if measured is not None:
            result["authority"] = "actual_strong_teacher_probe"
            result["proxy_diagnostics"] = {"method": threshold["primary_method"], "improved_mothers": improved,
                                            "mother_metrics": mothers, "sources": sources}
            result["improved_mothers"] = measured["statistics"]["improved_mothers"]
            result["mother_metrics"] = measured["statistics"].get("mother_metrics", [])
            result["sources"] = measured.get("sources", [])
        path = self.root / "gates" / f"route_{route}.json"
        atomic_json(path, result)
        remote = ops.BASE + "/source/gates/" + path.name
        self.transport.upload(path, remote, PORTS[0])
        result["gate_source"] = {"path": remote, "sha256": digest(path)}
        self.event("development_gate", route=route, passed=passed, improved_mothers=improved)
        return result

    def mechanism(self, route, primary_jobs):
        """Generate a receipt from actual worker evidence after it exists.

        Its eventual hash lives in controller state and the training gate; the
        immutable plan freezes the rule, rather than guessing a result hash.
        """
        if route == "05":
            job = self.ensure("teacher/05", PORTS[0],
                              self.body("05", "main", "prepare", "development", PORTS[0], preparation_mode="teacher"))
            if job is None or job["stage"] not in TERMINAL:
                return None
            if job["stage"] != "completed":
                return {"status": "attention_required", "passed": False,
                        "reason": "region teacher did not produce a complete oracle evidence source", "run_id": job["run_id"]}
            selected = [job]
        else:
            selected = list(primary_jobs)
        control_jobs = {method: [j for k, j in self.state["jobs"].items()
                                if k.startswith(f"development/{route}/{method}/")]
                        for method in ("voronoi", "integer")} if route == "08" else {}
        tasks = [{"port": j["port"], "run_id": j["run_id"]} for j in selected]
        controls = {method: [{"port": j["port"], "run_id": j["run_id"]} for j in jobs]
                    for method, jobs in control_jobs.items()}
        code = f'''import json,gzip,hashlib,sys
from pathlib import Path
base=Path({ops.BASE!r}); tasks={tasks!r}; controls={controls!r}
expected_commit={self.plan['scientific_commit']!r}
def folder(ref): return base/str(ref['port'])/'archive/outputs/workbench/jobs'/ref['run_id']
first=folder(tasks[0]); job=json.loads((first/'job.json').read_text())
assert job['git_commit']==expected_commit
sys.path.insert(0,str(Path(job['source_root'])/'src'))
from satellite_rl.coverage.nondrl.mechanism_audit import audit_scene_traces,audit_region_oracle
def read_rows(refs):
 rows=[]; sources=[]
 for ref in refs:
  d=folder(ref); j=json.loads((d/'job.json').read_text()); assert j['git_commit']==expected_commit
  p=d/'raw_scenes/manifest.json'; m=json.loads(p.read_text())
  assert m['provenance']['git_commit']==expected_commit
  for item in m['rows']:
   f=(d/'raw_scenes'/item['file']).resolve(); assert f.is_relative_to((d/'raw_scenes').resolve())
   assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
   rows.append(json.loads(gzip.decompress(f.read_bytes())))
  sources.append({{'run_id':ref['run_id'],'port':ref['port'],'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'git_commit':expected_commit,'config_sha256':m['provenance']['config_sha256']}})
 return rows,sources
if {route!r}=='05':
 wrapper=first/'teacher_manifest.json'; config=json.loads((first/'config.json').read_text())
 record=config['datasets']['development']; p=Path(record['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
 from satellite_rl.coverage.variable_dataset import load_variable_dataset
 mother_indices={{s.scene_id:i for i,s in enumerate(load_variable_dataset(p).scenes)}}
 result=audit_region_oracle(wrapper,hashlib.sha256(wrapper.read_bytes()).hexdigest(),mother_indices=mother_indices,sources=[{{'run_id':tasks[0]['run_id'],'port':tasks[0]['port'],'dataset_sha256':record['sha256'],'source_commit':expected_commit,'path':str(wrapper),'sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest()}}])
else:
 rows,sources=read_rows(tasks)
 other={{k:read_rows(v)[0] for k,v in controls.items()}}
 result=audit_scene_traces(rows,{route!r},sources,controls=other)
print(json.dumps(result,allow_nan=False))
'''
        response = self.transport.remote_python(code, PORTS[0], timeout=120)
        if response["exit_code"]:
            return {"status": "attention_required", "passed": False,
                    "reason": "mechanism helper/source interface failed", "error": response["stderr"][-1000:]}
        actual = json.loads(response["stdout"].splitlines()[-1])
        if "passed" not in actual or actual.get("status") not in ("passed", "science_stopped", "attention_required"):
            return {"status": "attention_required", "passed": False, "reason": "invalid mechanism receipt schema"}
        path = self.root / "mechanisms" / f"route_{route}.json"
        atomic_json(path, actual)
        actual["receipt_source"] = {"path": str(path), "sha256": digest(path)}
        return actual

    def teacher_reachability(self, route):
        port = PORTS[1] if route == "02" else PORTS[2]
        task = self.ensure(f"teacher_probe/{route}", port,
                           self.body(route, "main", "prepare", "development", port, preparation_mode="teacher_probe"))
        if task is None or task["stage"] not in TERMINAL:
            return None
        if task["stage"] != "completed" and task.get("scientific_stop") is not True:
            return {"status": "attention_required", "passed": False, "reason": "strong teacher probe did not retain terminal evidence"}
        directory = ops.BASE + f"/{port}/archive/outputs/workbench/jobs/{task['run_id']}"
        code = f'''import json,hashlib,sys
from pathlib import Path
d=Path({directory!r}); job=json.loads((d/'job.json').read_text()); assert job.get('origin_commit',job['git_commit'])=={self.plan['scientific_commit']!r}
sys.path.insert(0,str(Path(job['source_root'])/'src'))
if job['git_commit']!={self.plan['scientific_commit']!r}:
 from satellite_rl.workbench.nondrl_source_repair import validate_job_lineage
 validate_job_lineage(job,d)
from satellite_rl.coverage.nondrl.mechanism_audit import audit_teacher_reachability
from satellite_rl.coverage.variable_dataset import load_variable_dataset
config=json.loads((d/'config.json').read_text()); data=config['datasets']['development']; p=Path(data['path'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==data['sha256']
indices={{s.scene_id:i for i,s in enumerate(load_variable_dataset(p).scenes)}}
wrapper=d/'teacher_manifest.json'
result=audit_teacher_reachability(wrapper,hashlib.sha256(wrapper.read_bytes()).hexdigest(),{route!r},mother_indices=indices)
result['task_source']={{'run_id':job['run_id'],'port':{port},'wrapper_path':str(wrapper),'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),'dataset_sha256':data['sha256'],'scientific_commit':job.get('origin_commit',job['git_commit']),'execution_commit':job['git_commit'],'contract_sha256':job.get('source_repair_contract',{{}}).get('contract_sha256')}}
print(json.dumps(result,allow_nan=False))
'''
        response = self.transport.remote_python(code, port, timeout=120)
        if response["exit_code"]:
            return {"status": "attention_required", "passed": False,
                    "reason": "strong teacher reachability audit interface failed", "error": response["stderr"][-1000:]}
        actual = json.loads(response["stdout"].splitlines()[-1])
        if "passed" not in actual or actual.get("status") not in ("passed", "science_stopped", "attention_required"):
            return {"status": "attention_required", "passed": False, "reason": "invalid strong teacher receipt"}
        path = self.root / "teacher_reachability" / f"route_{route}.json"
        atomic_json(path, actual)
        actual["receipt_source"] = {"path": str(path), "sha256": digest(path)}
        return actual

    def teacher(self):
        for index, route in enumerate(("01", "02", "03", "04", "05")):
            if not self.state["routes"][route]["passed"]:
                continue
            port = PORTS[index % len(PORTS)]
            role = "development" if route == "05" else "train"
            if not self.finished(f"initial/{role}/"):
                continue
            self.ensure(f"teacher/{route}", port, self.body(route, "main", "prepare", role, port, preparation_mode="teacher"))
            if self.state["phase"] == "attention_required":
                return
        rows = [j for k, j in self.state["jobs"].items() if k.startswith("teacher/")]
        for j in rows:
            if j.get("scientific_stop"):
                self.state["routes"][j["route"]].update(status="science_stopped", passed=False, stop_reason=j.get("stop_reason"))
        expected = sum(self.state["routes"][r]["passed"] for r in ("01", "02", "03", "04", "05"))
        available = [j for j in rows if self.state["routes"][j["route"]]["passed"]]
        if len(available) == expected and all(j["stage"] == "completed" for j in available):
            self.state["phase"] = "train"
        self.training()

    def training(self):
        audit_ready = False
        if self.state["routes"]["04"]["passed"]:
            teacher = self.state["jobs"].get("teacher/04")
            if teacher and teacher["stage"] == "completed":
                port = teacher["port"]
                audit = self.ensure("ranking_audit/04", port,
                                    self.body("04", "main", "prepare", "validation", port, preparation_mode="ranking_audit"))
                audit_ready = audit is not None and audit["stage"] == "completed"
                if audit is not None and audit["stage"] in TERMINAL and not audit_ready:
                    self.attention("fixed route04 validation labels require engineering review", run_id=audit["run_id"])
                    return
        for route in ("01", "02", "03", "04", "05"):
            if not self.state["routes"][route]["passed"]:
                continue
            teacher = self.state["jobs"].get(f"teacher/{route}")
            if not teacher or teacher["stage"] != "completed" or route == "04" and not audit_ready:
                continue
            port = teacher["port"]
            extra = {"teacher_source": {"port": str(port), "run_id": teacher["run_id"]},
                     "gate_source": self.state["routes"][route]["gate_source"],
                     "validation_initial_source": self.initial_refs("validation")}
            if route == "04":
                audit = self.state["jobs"]["ranking_audit/04"]
                extra["ranking_audit_source"] = {"port": str(audit["port"]), "run_id": audit["run_id"]}
            self.ensure(f"train/{route}", port, self.body(route, "main", "train", None, port, **extra))
            if self.state["phase"] == "attention_required":
                return
        rows = [j for k, j in self.state["jobs"].items() if k.startswith("train/")]
        expected = sum(self.state["routes"][r]["passed"] for r in ("01", "02", "03", "04", "05"))
        if len(rows) == expected and all(j["stage"] in TERMINAL for j in rows):
            for j in rows:
                if j.get("scientific_stop"):
                    self.state["routes"][j["route"]].update(status="science_stopped", passed=False, stop_reason=j.get("stop_reason"))
            self.state["phase"] = "test"

    def testing(self):
        if not self.finished("initial/test/"):
            return
        count = self.record(PORTS[0], "test")["count"]
        # One test task per lane at a time fixes concurrency for paired timings.
        # Training and preparation previously use all scheduler concurrency.
        for lane, port in enumerate(PORTS):
            if any(j["port"] == port and k.startswith("test/") and j["stage"] not in TERMINAL for k, j in self.state["jobs"].items()):
                continue
            for route, methods in self.plan["methods"].items():
                for method in methods:
                    if method != "BH" and not self.state["routes"][route]["passed"]:
                        continue
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
        expected = sum(sum(m == "BH" or self.state["routes"][r]["passed"] for m in methods) * len(PORTS)
                       for r, methods in self.plan["methods"].items())
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
    if (args.directory / "execution_placement.json").exists():
        raise RuntimeError("executor placement is active; use launcher_pipeline or PipelineController instead of the legacy CLI")
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
