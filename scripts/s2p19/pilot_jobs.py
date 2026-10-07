"""Submit the already budgeted validation baseline and fixed-node B jobs.

One bounded pass uses only workbench APIs. It never resumes a training job or
selects a model. Request IDs name the scientific node, so retries cannot create
additional evaluations. Immutable saved validation cases are the input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from . import ops


RUNS = {"C0": (20418, "train_sac_20261007_152717_479365fb"),
        "C1": (20424, "train_sac_20261007_153047_1faf6d10")}
VALIDATION_SCENE = "scenes_20261007_152633_e7a0290f"
OUTPUT = ops.LOCAL / "controller" / "validation_jobs.json"


def _write(value):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)


def _match(rows, request_id):
    matches = [row for row in rows if row.get("request_id") == request_id]
    if len(matches) > 1:
        raise ValueError("duplicate scientific request ID: " + request_id)
    return matches[0] if matches else None


def controls(group):
    saved = json.loads(Path("D:/paper/outputs/verification/s2p19_research_20261007/p17_sca_configs.json").read_text(encoding="utf-8"))
    result = deepcopy(saved["B1" if group == "greedy" else "V1"]["config_power_refinement"])
    result["repeats"] = 1
    if result["methods"] != ["none", "sca", "sca_si_lp"] or result["action_seed"] != 53001:
        raise ValueError("frozen complete B route differs")
    return result


def pass_once(dispatch=False):
    with ThreadPoolExecutor(max_workers=2) as pool:
        catalogs = dict(zip(ops.SSH_PORTS, pool.map(lambda port: ops.api(port, "/api/s2p19/runs")["runs"], ops.SSH_PORTS)))
    result = {"experiment_id": "S2-P19", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "dispatch": dispatch, "jobs": {}, "new_submissions": [],
              "complete_B_budget": {"greedy_validation": 72, "model_pilot_and_endpoint_max": 432},
              "timing_scope": "validation alongside training; not final isolated timing comparison"}

    def ensure(key, port, request_id, endpoint, body, ready):
        existing = _match(catalogs[port], request_id)
        if existing:
            detail = ops.api(port, "/api/runs/" + existing["run_id"])
            result["jobs"][key] = {"port": port, "request_id": request_id,
                                   "run_id": existing["run_id"], "detail": detail}
            return existing
        result["jobs"][key] = {"port": port, "request_id": request_id,
                               "state": "ready" if ready else "waiting_source"}
        if ready and dispatch:
            # Bound this pass to one submission. Re-read before creating to
            # protect against an asynchronous progress run using the same ID.
            if result["new_submissions"]:
                return None
            again = _match(ops.api(port, "/api/s2p19/runs")["runs"], request_id)
            if again:
                return again
            created = ops.api(port, endpoint, {**body, "request_id": request_id}, "POST", timeout=180)
            result["new_submissions"].append({"key": key, "port": port, "request_id": request_id, "response": created})
            result["jobs"][key]["submission"] = created
        return None

    baseline = ensure("greedy_direct", 20418, "s2p19-validation-greedy-direct-51001",
                      "/api/s2p19/improvement",
                      {"mode": "baselines", "partition": "validation", "scene_id": VALIDATION_SCENE,
                       "methods": ["greedy"], "repeats": 1, "action_seed": 53001}, True)
    if baseline:
        ensure("greedy_B", 20418, "s2p19-validation-greedy-B", "/api/power-refinement/runs",
               {"source_run_id": baseline["run_id"], "source_group": "greedy",
                "partition": "validation", "config": controls("greedy")}, baseline.get("stage") == "completed")

    i0 = _match(catalogs[20418], "s2p19-i0-train-51001")
    runs = dict(RUNS)
    if i0:
        runs["I0"] = (20418, i0["run_id"])
    for arm, (port, run_id) in runs.items():
        current = next((row for row in catalogs[port] if row.get("run_id") == run_id), None)
        if current is None:
            raise RuntimeError("authorized training task disappeared: " + run_id)
        stage, step = current.get("stage"), current.get("env_step")
        node = ("pilot" if stage == "stopped" and current.get("stop_reason") == "s2p19_pilot_gate_pending"
                else "endpoint" if stage == "completed" else None)
        if node is None:
            result["jobs"][arm + "_B"] = {"state": "waiting_fixed_validation", "run_id": run_id,
                                               "stage": stage, "env_step": step}
            continue
        if type(step) is not int or step <= 0 or (arm != "I0" and step not in (100000, 400000)):
            raise ValueError("fixed validation step does not meet the frozen route: " + arm)
        ensure(arm + "_" + node + "_B", port, f"s2p19-{arm.lower()}-{node}-B",
               "/api/power-refinement/runs",
               {"source_run_id": run_id, "source_group": "model", "partition": "validation",
                "validation_step": step, "config": controls("model")}, True)
    _write(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(pass_once(args.dispatch), ensure_ascii=False, indent=2))
