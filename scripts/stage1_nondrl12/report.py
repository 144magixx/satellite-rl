"""Evidence-driven ND12 report builder; never fabricates stopped endpoints."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.stage1_nondrl12.strict_audit import Evidence, audit_job, assert_close


def stable_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def statistics(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite summary input")
    return {"count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def aggregate(rows):
    """Three related demand rows -> one mother; mothers then equally weighted."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["mother_scene_id"]].append(row)
    keys = ("load_cv_all", "load_cv_positive", "demand_band_fraction", "aircraft_carry", "beam_carry",
            "beam_mean_si", "beam_mean_sgm", "N", "total_power_w", "total_seconds", "task_seconds", "shared_initial_seconds")
    mothers = []
    for mother, values in sorted(groups.items()):
        multipliers = [r["multiplier"] for r in values]
        if len(values) != 3 or sorted(multipliers) != [.75, 1., 1.25]:
            raise ValueError("mother lacks exactly three frozen demand levels")
        result = {"mother_scene_id": mother, "all_zero_demand": all(r["all_zero_demand"] for r in values),
                  "hard_feasible": all(r["hard_feasible"] for r in values)}
        for key in keys:
            observed = [r.get(key) for r in values]
            if any(v is None for v in observed) and not all(v is None for v in observed):
                raise ValueError("partially missing mother metric " + key)
            result[key] = float(np.mean(observed)) if observed[0] is not None else None
        timing_keys = set.intersection(*(set(r["timing"]) for r in values))
        result["timing"] = {key: float(np.mean([r["timing"][key] for r in values])) for key in timing_keys}
        mothers.append(result)
    summary = {key: statistics([m[key] for m in mothers]) for key in keys}
    summary.update(mother_count=len(mothers), row_count=len(rows),
                   zero_demand_mothers=sum(m["all_zero_demand"] for m in mothers),
                   published_I0_rows=sum(r["published_I0"] for r in rows),
                   raw_proposal_rows=sum(r["raw_proposal_present"] for r in rows),
                   raw_geometry_feasible_rows=sum(r.get("raw_geometry_feasible") is True for r in rows),
                   hard_feasible=all(m["hard_feasible"] for m in mothers),
                   timing={key: statistics([m["timing"].get(key) for m in mothers])
                           for key in sorted({k for m in mothers for k in m["timing"]})})
    return summary, mothers


def paired_comparison(method, baseline):
    a = {m["mother_scene_id"]: m for m in method["mothers"]}
    b = {m["mother_scene_id"]: m for m in baseline["mothers"]}
    if set(a) != set(b):
        raise ValueError("paired comparison mother identities differ")
    pairs = [(a[key], b[key]) for key in sorted(a)]
    cv_a = [x["load_cv_all"] for x,y in pairs if x["load_cv_all"] is not None and y["load_cv_all"] is not None]
    cv_b = [y["load_cv_all"] for x,y in pairs if x["load_cv_all"] is not None and y["load_cv_all"] is not None]
    base_mean = float(np.mean(cv_b)) if cv_b else None
    relative = float(np.mean(cv_a)) / base_mean - 1 if base_mean is not None and base_mean > 1e-12 else None
    carried = [x["aircraft_carry"]-y["aircraft_carry"] for x,y in pairs if x["aircraft_carry"] is not None and y["aircraft_carry"] is not None]
    carry_delta = float(np.mean(carried)) if carried else None
    times_a = statistics([x["total_seconds"] for x,y in pairs])
    times_b = statistics([y["total_seconds"] for x,y in pairs])
    protected = bool(carried) and carry_delta >= -.005 and all(x["hard_feasible"] for x,y in pairs)
    quality = protected and relative is not None and relative <= -.05
    comparable_cv = relative <= .02 if relative is not None else bool(cv_a) and max(abs(x-y) for x,y in zip(cv_a,cv_b)) <= 1e-10
    efficiency = (protected and comparable_cv and times_a["median"] is not None and times_b["median"] is not None
                  and times_a["median"] <= .5*times_b["median"] and times_a["p95"] <= times_b["p95"])
    return {"paired_mothers": len(pairs), "relative_CV_change_ratio_of_means": relative,
            "absolute_CV_change": float(np.mean(np.asarray(cv_a)-cv_b)) if cv_a else None,
            "aircraft_carry_delta": carry_delta, "quality_branch_met": quality,
            "efficiency_branch_met": bool(efficiency), "zero_baseline_relative_undefined": base_mean is not None and base_mean <= 1e-12,
            "worse_CV_mothers": sum(x["load_cv_all"] > y["load_cv_all"]+1e-10 for x,y in pairs if x["load_cv_all"] is not None),
            "worse_carry_mothers": sum(x["aircraft_carry"] < y["aircraft_carry"]-1e-10 for x,y in pairs if x["aircraft_carry"] is not None)}


def verify_gate(gate, threshold, audits_by_run, path_map=None):
    """Recompute the scientific gate from audited exact scene evidence."""
    if gate.get("experiment_id") != "S1-ND12" or gate.get("threshold") != threshold:
        raise ValueError("development gate group/threshold differs")
    if gate.get("authority") == "actual_strong_teacher_probe":
        return verify_teacher_gate(gate, audits_by_run)
    rows = []
    for source in gate.get("sources", []):
        audit = audits_by_run.get(source["run_id"])
        if not audit or audit["state"] != "completed" or audit.get("raw_manifest_sha256") != source["manifest_sha256"]:
            raise ValueError("development gate source does not close to audited task")
        actual = {(r["mother_scene_id"], r["multiplier"]): r for r in audit["rows"]}
        for row in source["rows"]:
            match = actual.get((row["mother_id"], row["multiplier"]))
            if match is None or match["dataset_index"] != row["index"]:
                raise ValueError("gate row CV differs from audited shard")
            for source_key, actual_key in (("cv", "load_cv_all"), ("initial_cv", "initial_cv"),
                                           ("carry", "aircraft_carry"), ("initial_carry", "initial_carry")):
                assert_close(row[source_key], match[actual_key], "gate row " + source_key)
            if row["feasible"] is not match["hard_feasible"] or row["initial_sha256"] != match["initial_sha256"]:
                raise ValueError("gate geometry/initial SHA differs")
        rows.extend(source["rows"])
    if len(rows) != gate.get("actual_row_count") or len(rows) != gate.get("expected_row_count"):
        raise ValueError("gate row accounting incomplete")
    groups = defaultdict(list)
    for row in rows:
        groups[row["index"]].append(row)
    if len(groups) != gate.get("mother_count") or any(sorted(r["multiplier"] for r in group) != [.75, 1., 1.25] for group in groups.values()):
        raise ValueError("gate mother/three-level accounting differs")
    improved = []
    geometry = True
    for index, group in groups.items():
        geometry &= all(r["feasible"] for r in group)
        base = [r["initial_cv"] for r in group if r["initial_cv"] is not None]
        final = [r["cv"] for r in group if r["cv"] is not None]
        carried = [r["carry"] - r["initial_carry"] for r in group if r["carry"] is not None]
        base_mean = float(np.mean(base)) if base else None
        gain = (base_mean - float(np.mean(final))) / base_mean if base_mean is not None and base_mean > 1e-12 else None
        if (gain is not None and gain > threshold["relative_cv_improvement"] and all(r["feasible"] for r in group)
                and (not carried or float(np.mean(carried)) >= -.005)):
            improved.append(index)
    passed = geometry and len(improved) >= threshold["minimum_improved_mothers"]
    if sorted(improved) != sorted(gate["improved_mothers"]) or gate["geometry_pass"] != geometry:
        raise ValueError("gate improvements/feasibility differ")
    # A separate required mechanism audit can veto a quality-passing gate.
    required=threshold.get("require_mechanism_audit") or threshold.get("required_mechanism")
    if not passed and gate["passed"] is True:
        raise ValueError("quality-failed gate cannot be marked passed by a mechanism")
    if not required and gate["passed"] != passed:
        raise ValueError("gate outcome differs from measured threshold")
    if threshold.get("required_mechanism") and passed:
        mechanism_record=verify_dynamic_mechanism(gate,audits_by_run,path_map)
        if gate["passed"]!=(passed and mechanism_record["passed"] is True):
            raise ValueError("gate dynamic mechanism outcome differs")
    elif threshold.get("require_mechanism_audit"):
        mechanism = threshold.get("mechanism_audit")
        if not isinstance(mechanism, dict) or not mechanism.get("sha256") or not mechanism.get("path"):
            raise ValueError("required mechanism audit missing; engineering gap, not science failure")
        location = mechanism["path"]
        for old,new in (path_map or {}).items():
            if location.startswith(old):
                location = new + location[len(old):]
                break
        path = Path(location)
        if not path.is_file() or digest_bytes(path.read_bytes()) != mechanism["sha256"]:
            raise ValueError("required mechanism audit source does not close")
        mechanism_record = json.loads(path.read_text(encoding="utf-8"))
        if mechanism_record.get("experiment_id") != "S1-ND12" or mechanism_record.get("route") != gate["route"]:
            raise ValueError("required mechanism audit identity differs")
        if gate["passed"] != (passed and mechanism_record.get("passed") is True):
            raise ValueError("gate mechanism outcome differs")
    return {"passed_quality": passed, "actual_passed": gate["passed"], "improved_mothers": len(improved)}


def verify_dynamic_mechanism(gate,audits,path_map=None):
    receipt=gate.get("mechanism",{})
    source=receipt.get("receipt_source",{})
    location=source.get("path","")
    for old,new in (path_map or {}).items():
        if location.startswith(old):location=new+location[len(old):];break
    path=Path(location)
    if not path.is_file() or digest_bytes(path.read_bytes())!=source.get("sha256"):
        raise ValueError("required dynamic mechanism receipt/source missing; engineering gap")
    saved=json.loads(path.read_text(encoding="utf-8"))
    if saved!={k:v for k,v in receipt.items() if k!="receipt_source"}:
        raise ValueError("dynamic mechanism receipt fields differ from pinned content")
    if (saved.get("experiment_id")!="S1-ND12" or saved.get("route")!=gate["route"]
            or saved.get("status") not in ("passed","science_stopped")):
        raise ValueError("dynamic mechanism is incomplete; engineering attention")
    if not saved.get("sources"):
        raise ValueError("dynamic mechanism has no actual task source")
    for item in saved["sources"]:
        run_id=item.get("run_id")
        if run_id is None:
            run_id=next((key for key in audits if f"/jobs/{key}/" in item.get("path","")),None)
        audit=audits.get(run_id)
        if not audit or audit["state"]!="completed":
            raise ValueError("dynamic mechanism source task is not fully audited")
        if audit.get("raw_manifest_sha256")==item.get("sha256"):
            continue
        marker=f"/jobs/{run_id}/"
        relative=item.get("path","").split(marker,1)[-1]
        if audit.get("supervision_sources",{}).get(relative)!=item.get("sha256"):
            raise ValueError("dynamic mechanism source SHA does not close to immutable task evidence")
    return saved


def verify_teacher_gate(gate, audits_by_run):
    """02/03 gate authority is real offline references, never online T/HR."""
    measured = gate.get("teacher_reachability", {})
    source = measured.get("task_source", {})
    audit = audits_by_run.get(source.get("run_id"))
    if (gate.get("route") not in ("02", "03") or measured.get("status") not in ("passed", "science_stopped")
            or not audit or audit.get("preparation_mode") != "teacher_probe" or not audit.get("teacher_probe")):
        raise ValueError("strong teacher complete proof missing; engineering attention")
    proof = audit["teacher_probe"]
    if (source.get("wrapper_sha256") != proof["wrapper_sha256"] or source.get("dataset_sha256") != proof["dataset_sha256"]
            or source.get("scientific_commit") != audit.get("git_commit") or audit.get("git_commit") != gate["source_commit"]):
        raise ValueError("strong teacher task/source SHA closure differs")
    rows = proof["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset_index"]].append(row)
    if (proof["complete_problems"] != 60 or set(groups) != set(range(20)) or
            any(sorted(r["multiplier"] for r in values) != [.75,1.,1.25] for values in groups.values())):
        raise ValueError("strong teacher60 problems/20 mothers incomplete")
    improved = []
    for index, values in groups.items():
        before = [r["initial_cv"] for r in values if r["initial_cv"] is not None]
        after = [r["load_cv_all"] for r in values if r["load_cv_all"] is not None]
        initial = float(np.mean(before)) if before else None
        gain = (initial-float(np.mean(after)))/initial if initial is not None and initial>1e-12 else None
        if gain is not None and gain>.01 and all(r["hard_feasible"] for r in values):
            improved.append(index)
    passed = len(improved)>=6 and all(r["hard_feasible"] for r in rows)
    statistics = measured.get("statistics", {})
    if (statistics.get("actual_complete_problems") != 60 or statistics.get("legal_reference_mothers") != list(range(20))
            or sorted(statistics.get("improved_mothers", [])) != sorted(improved)
            or sorted(gate.get("improved_mothers", [])) != sorted(improved) or measured.get("passed") is not passed):
        raise ValueError("strong teacher measured gate statistics differ from full raw references")
    if gate.get("passed") != passed:
        mechanism = gate.get("mechanism")
        if not mechanism or mechanism.get("status") != "science_stopped" or mechanism.get("passed") is not False:
            raise ValueError("strong teacher gate outcome differs")
    return {"passed_quality": passed, "actual_passed": gate["passed"], "improved_mothers": len(improved),
            "authority": "actual_strong_teacher_probe", "complete_problems": 60,
            "proxy_diagnostics_retained": bool(gate.get("proxy_diagnostics"))}


def task_location(root, record):
    if record.get("local_job_dir"):
        return Path(record["local_job_dir"])
    receipt = root / "job_archives" / str(record["port"]) / record["run_id"] / "receipt.json"
    if receipt.is_file():
        return receipt
    for path in (root / str(record["port"]) / "archive/outputs/workbench/jobs" / record["run_id"],
                 root / "jobs" / record["run_id"]):
        if path.is_dir():
            return path
    raise FileNotFoundError("task evidence missing: " + record["run_id"])


def shared_baseline(audits):
    """BH is a common baseline task family, independent of route06 passing."""
    candidates = [audit for audit in audits.values() if audit.get("dataset_role") == "test"
                  and audit.get("method") == "BH" and
                  (audit.get("key", "").startswith("test/06/BH/") or
                   audit.get("key", "").startswith("shared_baseline/"))]
    if len(candidates) != 4 or {audit.get("port") for audit in candidates} != {20418,20424,20473,20474}:
        return None
    if any(audit["state"] != "completed" for audit in candidates):
        return None
    rows = [row for audit in candidates for row in audit["rows"]]
    summary, mothers = aggregate(rows)
    return {"summary": summary, "mothers": mothers, "run_ids": [audit["run_id"] for audit in candidates],
            "scope": "single shared BH family; does not count as route06 algorithm passing",
            "evidence_keys": [audit["key"] for audit in candidates]}


def build(plan, state, root, *, path_map=None, gate_directory=None, verified_volume_cache=None):
    root = Path(root)
    if plan.get("experiment_id") != "S1-ND12" or state.get("experiment_id") != "S1-ND12":
        raise ValueError("report inputs must belong to S1-ND12")
    if state.get("plan_sha256") != stable_sha(plan):
        raise ValueError("controller frozen plan SHA differs")
    issues, audits, locations = [], {}, {}
    for key, record in state.get("jobs", {}).items():
        try:
            location = task_location(root, record)
            audit = audit_job(location, path_map=path_map,verified_volume_cache=verified_volume_cache)
            audit.update(key=key, port=record["port"])
            if audit["run_id"] != record["run_id"] or (audit.get("git_commit") and audit["git_commit"] != plan["scientific_commit"]):
                raise ValueError("controller task/source differs")
            audits[record["run_id"]] = audit
            locations[record["run_id"]] = str(location)
            if not audit["formal_terminal"] or audit["state"] == "failure":
                issues.append(key + ": " + audit["state"])
        except (ValueError, KeyError, FileNotFoundError) as error:
            issues.append(key + ": " + str(error))
    routes = {}
    common_initials = {}
    for audit in audits.values():
        if audit.get("key", "").startswith("initial/"):
            role = audit.get("dataset_role")
            for row in audit["rows"]:
                key = (role, row["mother_scene_id"], row["multiplier"])
                if key in common_initials:
                    issues.append("duplicate common initial mother/level " + str(key))
                common_initials[key] = row["initial_sha256"]
    for audit in audits.values():
        if audit.get("key", "").startswith(("test/", "development/", "shared_baseline/", "teacher_probe/")):
            role = audit.get("dataset_role")
            for row in audit["rows"]:
                key = (role, row["mother_scene_id"], row["multiplier"])
                if common_initials.get(key) != row["initial_sha256"]:
                    issues.append(audit["key"] + ": shared I0 source does not close " + str(key))
    common_BH = shared_baseline(audits)
    if common_BH is None:
        issues.append("shared BH baseline lacks four completed, independent lane shards")
    for route in (f"{i:02d}" for i in range(1, 13)):
        status = state.get("routes", {}).get(route, {})
        result = {"status": status.get("status", "pending"), "methods": {}, "gate": None}
        if not status:
            issues.append(route + ": missing development route state")
        else:
            try:
                gate_path = Path(gate_directory or root / "controller/gates") / f"route_{route}.json"
                raw = gate_path.read_bytes()
                if digest_bytes(raw) != status["gate_source"]["sha256"]:
                    raise ValueError("gate SHA differs")
                result["gate"] = verify_gate(json.loads(raw), plan["gate_thresholds"][route], audits, path_map)
            except (KeyError, ValueError, FileNotFoundError) as error:
                issues.append(route + ": " + str(error))
        if status.get("passed"):
            for method in plan["methods"][route]:
                if method == "BH" and common_BH is not None:
                    if route == "06":
                        result["methods"]["BH"] = common_BH
                    continue
                subset = [a for a in audits.values() if a.get("key", "").startswith(f"test/{route}/{method}/")]
                if len(subset) != 4 or any(a["state"] != "completed" for a in subset):
                    issues.append(f"{route}/{method}: incomplete four-lane test evidence")
                    continue
                rows = [r for a in subset for r in a["rows"]]
                summary, mothers = aggregate(rows)
                result["methods"][method] = {"summary": summary, "mothers": mothers,
                                             "run_ids": [a["run_id"] for a in subset]}
        elif status.get("status") == "science_stopped":
            if result["gate"] and result["gate"]["actual_passed"]:
                stops = [a for a in audits.values() if a.get("key") in (f"teacher/{route}", f"train/{route}")
                         and a["state"] == "scientific_stop"]
                if not stops:
                    issues.append(route + ": route stopped without measured failure or task scientific stop")
        else:
            issues.append(route + ": pending or engineering attention, no scientific failure inferred")
        if route == "06" and common_BH is not None:
            result["methods"]["BH"] = common_BH
        routes[route] = result
        result["comparisons"] = {}
        required = {"01": (("M","H"),("MO","HO"),("MO","O")),
                    "02": (("BC","H"),("BC","T")), "03": (("DR","HR"),),
                    "04": (("L","H"),), "05": (("L","rule"),),
                    "08": (("main","voronoi"),("main","integer")),
                    "12": (("main","BH"),("main","single_objective"))}.get(route, (("main","BH"),))
        for primary, baseline in required:
            baseline_data = common_BH if baseline == "BH" else result["methods"].get(baseline)
            if primary in result["methods"] and baseline_data is not None:
                result["comparisons"][primary+"/"+baseline] = paired_comparison(result["methods"][primary], baseline_data)
            elif status.get("passed"):
                issues.append(route+": required attribution comparison missing "+primary+"/"+baseline)
    if state.get("phase") != "report_pending":
        issues.append("controller has not reached report_pending")
    if not state.get("jobs"):
        issues.append("no workbench task evidence")
    return {"experiment_id": "S1-ND12", "scientific_commit": plan["scientific_commit"],
            "formal_ready": not issues, "issues": issues, "routes": routes,
            "tasks": {run_id: {k: v for k, v in a.items() if k != "rows"} for run_id, a in audits.items()},
            "row_audits": {run_id: a["rows"] for run_id, a in audits.items()}, "locations": locations,
            "shared_baseline": common_BH,
            "statistics_unit": "mean of three levels per mother, then equal mothers",
            "RF_scope": "frozen model estimate; recorded TDMA algebra verified; no independent RF recomputation"}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def markdown(report):
    lines = ["# 第一阶段非DRL十二路线实验报告", "", "状态：" + ("证据已闭合" if report["formal_ready"] else "草稿，证据尚未闭合"), "",
             "本轮为固定0.5°、每景共同N的机制验证。全部飞机必须覆盖且唯一归属，需求不可拆。",
             "逐机承载为冻结下游模型估计，审计核对GEO几何、归属、负载和TDMA代数，没有独立重算RF。", "",
             "三档需求先在母场景内求均值，再对母场景等权汇总；全零需求CV和承载比例不补零。", "",
             "| 路线 | 实际状态/方法 | 全体束CV | 正需求束CV | 逐机承载 | N | 总时均值/中位/P95(s) |", "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "缺失/未定义" if value is None else f"{value:.4f}"
    for route, record in report["routes"].items():
        if not record["methods"]:
            lines.append(f"| {route} | {record['status']} | — | — | — | — | — |")
        for method, data in record["methods"].items():
            s = data["summary"]
            t = s["total_seconds"]
            lines.append(f"| {route} | {method} | {fmt(s['load_cv_all']['mean'])} | {fmt(s['load_cv_positive']['mean'])} | {fmt(s['aircraft_carry']['mean'])} | {fmt(s['N']['mean'])} | {fmt(t['mean'])}/{fmt(t['median'])}/{fmt(t['p95'])} |")
    lines.extend(["", "所有失败、科学停止与未完成任务均保留。主法和消融分别列示；只有通过冻结开发门槛的路线才进入后续训练与新考核。单种子51001不能支持多种子稳定性结论。", ""])
    lines += ["| 路线/方法 | 需求带内率 | 波束中心模型承载 | 波束SI | 波束SGM | 全景功率(W) | 原提议几何可行/有提议行 | 发布I0行 | 全零需求母 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for route,record in report["routes"].items():
        for method,data in record["methods"].items():
            s=data["summary"]
            lines.append(f"| {route}/{method} | {fmt(s['demand_band_fraction']['mean'])} | {fmt(s['beam_carry']['mean'])} | {fmt(s['beam_mean_si']['mean'])} | {fmt(s['beam_mean_sgm']['mean'])} | {fmt(s['total_power_w']['mean'])} | {s['raw_geometry_feasible_rows']}/{s['raw_proposal_rows']} | {s['published_I0_rows']} | {s['zero_demand_mothers']} |")
    lines += ["", "分段时间均为实测：共同I0成本由来源任务加入；算法在线时间包含内部候选与物理核验，求解/推理是其中子段，不能再次叠加。总时是共同初解与当前完整任务实测分段合计，不是“不调整”函数的近零耗时。", "",
              "| 路线/方法 | 共同I0均值(s) | 在线算法均值(s) | 其中求解均值(s) | 终局核验均值(s) | 总时中位/P95(s) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for route,record in report["routes"].items():
        for method,data in record["methods"].items():
            s=data["summary"];t=s["timing"]
            mean=lambda key:t.get(key,{}).get("mean")
            lines.append(f"| {route}/{method} | {fmt(s['shared_initial_seconds']['mean'])} | {fmt(mean('online_seconds'))} | {fmt(mean('solver_seconds'))} | {fmt(mean('final_verification_seconds'))} | {fmt(s['total_seconds']['median'])}/{fmt(s['total_seconds']['p95'])} |")
    lines += ["", "设备、软件和并发条件保留于各任务的 execution_environment 与原始 job/status/preflight；缺项如实标缺失，异设备或不同争用条件的耗时不能当作严格同条件比较。", ""]
    lines += ["配对归因检查：", ""]
    for route,record in report["routes"].items():
        for label,comparison in record.get("comparisons",{}).items():
            lines.append(f"- {route} {label}：CV均值相对变化 {fmt(comparison['relative_CV_change_ratio_of_means'])}；逐机承载差 {fmt(comparison['aircraft_carry_delta'])}；质量门槛{'达标' if comparison['quality_branch_met'] else '未达标'}，效率门槛{'达标' if comparison['efficiency_branch_met'] else '未达标'}。")
    if report["issues"]:
        lines += ["证据缺口：", ""] + ["- " + issue for issue in report["issues"]]
    return "\n".join(lines) + "\n"


def figures(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    groups = [(f"{r}/{m}", d["mothers"]) for r, x in report["routes"].items() for m, d in x["methods"].items()]
    outputs = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        for metric, label in (("load_cv_all", "All-beam load CV"), ("aircraft_carry", "Aircraft TDMA carried fraction"),
                              ("total_seconds", "End-to-end segment time (s)")):
            if not groups:
                continue
            fig, ax = plt.subplots(figsize=(max(8, len(groups)*.28), 4.5), layout="constrained")
            for index, (name, mothers) in enumerate(groups):
                values = [m[metric] for m in mothers if m[metric] is not None]
                ax.scatter(np.full(len(values), index), values, s=12, marker=("o", "s", "^", "D")[index%4],
                           alpha=.45, color=("#0072B2", "#D55E00", "#009E73", "#000000")[index%4])
                if values:
                    ax.plot([index-.22,index+.22], [np.mean(values)]*2, color="black", linewidth=2)
            ax.set_xticks(range(len(groups)), [name for name, _ in groups], rotation=90)
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.set_title("Each point: mother mean of 3 demand levels; black line: equal-mother mean")
            path = directory / f"{metric}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            outputs.append({"file": str(path), "sha256": digest_bytes(path.read_bytes()), "statistical_unit": "mother"})
        for run_id, task in report["tasks"].items():
            losses = task.get("curve_losses", [])
            validation = task.get("curve_validation", [])
            if not losses:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
            axes[0].plot([v["update"] for v in losses], [v["loss"] for v in losses], color="#0072B2", linewidth=.7)
            axes[0].set(xlabel="Supervised update", ylabel="Raw supervised loss", title="Every recorded update; no smoothing")
            xs = [v["updates"] for v in validation]
            ys = [v["metrics"].get("mother_equal_cv_mean", np.nan) for v in validation]
            axes[1].plot(xs, [np.nan if y is None else y for y in ys], color="#D55E00", marker="o")
            axes[1].set(xlabel="Supervised update", ylabel="Fixed validation all-beam CV", title="Raw fixed nodes; missing values shown as gaps")
            path = directory / ("training_" + run_id + ".png")
            fig.savefig(path, dpi=180)
            plt.close(fig)
            outputs.append({"file": str(path), "sha256": digest_bytes(path.read_bytes()), "training_run": run_id,
                            "loss_updates": len(losses), "validation_nodes": len(validation)})
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--path-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(json.loads(args.plan.read_text(encoding="utf-8")), json.loads(args.state.read_text(encoding="utf-8")), args.root,
                   path_map=json.loads(args.path_map.read_text(encoding="utf-8")) if args.path_map else None,
                   gate_directory=args.state.parent / "gates")
    args.output.mkdir(parents=True, exist_ok=True)
    report["figures"] = figures(report, args.output / "figures")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"formal_ready": report["formal_ready"], "issues": len(report["issues"]),
                      "tasks": len(report["tasks"]), "figures": len(report["figures"])}))


if __name__ == "__main__":
    main()
