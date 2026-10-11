"""Local operations DAG: each ND12 route advances on its own closed evidence.

The original controller's task bodies, keys, request IDs and scientific gates
are reused verbatim. This module does not replace worker code or the run plan.
Final test publication keeps a group-wide terminal/quiet barrier.
"""
from __future__ import annotations

import json

from scripts.stage1_nondrl12.controller import Controller, PORTS, TERMINAL
from satellite_rl.workbench.coverage_nondrl import METHODS
from satellite_rl.workbench.storage import now

LEARNED = ("01", "02", "03", "04", "05")


def route_of(key):
    parts = key.split("/")
    return parts[1] if len(parts) > 1 and parts[0] != "initial" and parts[1] in METHODS else None


class PipelineController(Controller):
    def route_attention(self, route, reason, **fields):
        record = {"reason": reason, "at": now(), **fields}
        old = self.state.setdefault("route_attention", {}).get(route)
        if old is None or old.get("reason") != reason:
            self.event("route_attention_required", route=route, **record)
        self.state["route_attention"][route] = record
        self.save()

    def attention(self, reason, **fields):
        # Evidence gaps for one route remain engineering attention. Other
        # ready branches may advance; global identity/API faults still stop.
        route = fields.get("route")
        if route in METHODS:
            self.route_attention(route, reason, **{k: v for k, v in fields.items() if k != "route"})
        else:
            super().attention(reason, **fields)

    def refresh(self):
        snapshots = {p: {r["run_id"]: r for r in self.transport.api(p, "/api/coverage/nondrl/runs")["runs"]}
                     for p in PORTS}
        for key, record in self.state["jobs"].items():
            row = snapshots[record["port"]].get(record["run_id"])
            if row is None:
                super().attention("published workbench job missing", key=key)
                return False
            record.update(stage=row.get("stage"), scientific_stop=row.get("scientific_stop", False),
                          stop_reason=row.get("stop_reason"), last_heartbeat=row.get("heartbeat_at"),
                          pid=row.get("pid"), completed_scenes=row.get("completed_scenes"), updates=row.get("updates"))
            route = route_of(key)
            blocked = route in self.state.get("route_attention", {})
            if row.get("stage") in ("failed", "interrupted") and not blocked:
                reason = None
                if record.get("retries", 0) >= 2:
                    reason = "same-ID recovery bound reached"
                else:
                    try:
                        self.transport.api(record["port"], f"/api/runs/{record['run_id']}/resume", {})
                    except Exception as error:
                        reason = "job requires diagnosis before safe same-ID recovery: " + str(error)
                    else:
                        record.update(retries=record.get("retries", 0) + 1, stage="queued")
                        self.event("same_id_resume", key=key, retry=record["retries"])
                if reason:
                    if route:
                        self.route_attention(route, reason, key=key, status=row)
                    else:
                        super().attention(reason, key=key, status=row)
                        return False
            if row.get("stage") in ("stopped", "cancelled") and row.get("scientific_stop") is not True and not blocked:
                reason = "workbench task explicitly stopped; continuation needs diagnosis"
                if route:
                    self.route_attention(route, reason, key=key, status=row)
                else:
                    super().attention(reason, key=key, status=row)
                    return False
        self.save()
        return True

    def development_ready(self, route):
        keys = [f"development/{route}/{method}/{port}"
                for method in self.plan["gate_thresholds"][route]["development_methods"] for port in PORTS]
        return all(key in self.state["jobs"] and self.state["jobs"][key]["stage"] == "completed" for key in keys)

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
            if (route in self.state["routes"] or route in self.state.get("route_attention", {}) or
                    not self.development_ready(route)):
                continue
            gate = self.make_gate(route)
            if gate is not None:
                self.state["routes"][route] = gate
                self.event("route_gate_closed", route=route, passed=gate["passed"])
                self.save()
            if self.state["phase"] == "attention_required":
                return
        self.advance_learning()

    def advance_learning(self):
        for index, route in enumerate(LEARNED):
            gate = self.state["routes"].get(route)
            if not gate or not gate["passed"] or route in self.state.get("route_attention", {}):
                continue
            teacher_key, train_key = f"teacher/{route}", f"train/{route}"
            for key in (teacher_key, train_key):
                record = self.state["jobs"].get(key)
                if record and record.get("scientific_stop") is True:
                    gate.update(development_gate_passed=True, passed=False, status="science_stopped",
                                stopped_phase=key.split("/")[0], stop_reason=record.get("stop_reason"))
                    self.event("route_scientific_stop", route=route, key=key, stop_reason=record.get("stop_reason"))
                    self.save()
                    break
            if not gate["passed"]:
                continue
            port = PORTS[index % len(PORTS)]
            role = "development" if route == "05" else "train"
            if not self.finished(f"initial/{role}/") or not self.finished("initial/validation/"):
                continue
            teacher = self.ensure(teacher_key, port, self.body(route, "main", "prepare", role, port, preparation_mode="teacher"))
            if teacher is None or teacher["stage"] != "completed":
                continue
            extra = {"teacher_source": {"port": str(port), "run_id": teacher["run_id"]},
                     "gate_source": gate["gate_source"], "validation_initial_source": self.initial_refs("validation")}
            if route == "04":
                audit = self.ensure("ranking_audit/04", port,
                                    self.body("04", "main", "prepare", "validation", port, preparation_mode="ranking_audit"))
                if audit is None or audit["stage"] not in TERMINAL:
                    continue
                if audit["stage"] != "completed":
                    self.route_attention(route, "fixed route04 validation labels require engineering review", run_id=audit["run_id"])
                    continue
                extra["ranking_audit_source"] = {"port": str(port), "run_id": audit["run_id"]}
            self.ensure(train_key, port, self.body(route, "main", "train", None, port, **extra))
            if self.state["phase"] == "attention_required":
                return

    def learning_terminal(self):
        for route in LEARNED:
            gate = self.state["routes"].get(route)
            if gate is None:
                return False
            if gate["passed"]:
                model = self.state["jobs"].get(f"train/{route}")
                if model is None or model["stage"] != "completed":
                    return False
        return True

    def ready_for_test(self):
        return (set(self.state["routes"]) == set(METHODS) and not self.state.get("route_attention") and
                all(type(g.get("passed")) is bool and g.get("status") in ("passed", "science_stopped")
                    for g in self.state["routes"].values()) and
                self.learning_terminal() and self.finished("initial/test/") and
                all(j["stage"] in TERMINAL for key, j in self.state["jobs"].items() if not key.startswith("test/")))

    def workers_quiet(self):
        """Check old process births; a residual/reused PID is not evidence of work."""
        for port in PORTS:
            keys = [j["run_id"] for key, j in self.state["jobs"].items()
                    if j["port"] == port and not key.startswith("test/")]
            code = f'''import json,os,pathlib
base=pathlib.Path({('/home/u2024110224/first_stage_satellite_rl/s1_nd12/'+str(port)+'/archive/outputs/workbench/jobs')!r})
boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime '));hz=os.sysconf('SC_CLK_TCK')
live=[];unknown=[]
for key in {keys!r}:
 d=base/key;status=json.loads((d/'status.json').read_text());p=d/'launch.json'
 launch=json.loads(p.read_text()) if p.exists() else {{}};pid=status.get('pid') or launch.get('pid')
 if not pid:continue
 p=pathlib.Path('/proc')/str(pid)/'stat'
 try:f=p.read_text().rsplit(')',1)[1].split()
 except FileNotFoundError:continue
 if f[0]=='Z':continue
 actual=boot+int(f[19])/hz;expected=launch.get('process_start_time')
 if expected is None:unknown.append(key)
 elif abs(actual-expected)<.01:live.append(key)
print(json.dumps({{'quiet':not live and not unknown,'live':live,'unknown_birth':unknown}}))
'''
            response = self.transport.remote_python(code, port, timeout=60)
            if response["exit_code"]:
                super().attention("final timing isolation process audit failed", port=port, error=response["stderr"][-800:])
                return False
            receipt = json.loads(response["stdout"].splitlines()[-1])
            if receipt.get("unknown_birth"):
                super().attention("final timing isolation lacks worker birth evidence", port=port, receipt=receipt)
                return False
            if not receipt["quiet"]:
                self.state["test_barrier"] = {"reason": "waiting_for_workers_to_exit", "port": port, **receipt}
                return False
        return True

    def testing(self):
        if not self.ready_for_test():
            self.state["test_barrier"] = {"reason": "waiting_for_all_route_gates_and_learning_terminal"}
            return
        if not self.workers_quiet():
            return
        self.state["phase"] = "test"
        super().testing()

    def update_route_view(self):
        view = {}
        for route in METHODS:
            gate = self.state["routes"].get(route)
            if route in self.state.get("route_attention", {}):
                phase = "engineering_attention"
            elif gate is None:
                phase = "gate_pending" if self.development_ready(route) else "development_pending"
            elif not gate["passed"]:
                phase = "science_stopped"
            elif route not in LEARNED:
                phase = "ready_for_final_test"
            else:
                teacher, train = self.state["jobs"].get(f"teacher/{route}"), self.state["jobs"].get(f"train/{route}")
                phase = ("ready_for_final_test" if train and train["stage"] == "completed" else
                         "training" if train else "teacher" if teacher else "teacher_pending")
            view[route] = {"phase": phase, "development_complete": self.development_ready(route)}
        self.state.update(pipeline_version="route_readiness_v1", pipeline_routes=view)

    def tick(self):
        if self.state["phase"] in ("attention_required", "report_pending", "stopped"):
            return self.state
        if not self.refresh():
            return self.state
        if self.state["phase"] == "initial":
            self.initial()
        if self.state["phase"] != "initial":
            self.development()
            if self.state["phase"] != "attention_required":
                self.testing()
        self.update_route_view()
        self.save()
        return self.state
