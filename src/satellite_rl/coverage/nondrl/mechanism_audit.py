"""Read-only, source-closed mechanism gates for S1-ND12.

This module never generates teacher labels or launches jobs. It recomputes
saved supervision/scene traces; missing or unknown evidence is engineering
attention, whereas a measured threshold failure is a scientific stop.
"""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np

from .learning import audit_ranker
from .teachers import EVIDENCE_FORMAT, digest

MULTIPLIERS = (.75, 1., 1.25)
FORMAT = "s1_nondrl_mechanism_audit_v1"


class EvidenceGap(ValueError):
    pass


def _receipt(route, sources, statistics, thresholds, passed=None, reason=None):
    return {"format": FORMAT, "experiment_id": "S1-ND12", "route": route,
            "status": "attention_required" if passed is None else "passed" if passed else "science_stopped",
            "passed": passed is True, "attention_required": passed is None,
            "scientific_stop": passed is False, "statistics": statistics,
            "thresholds": thresholds, "sources": sources or [], "reason": reason}


def _sources(sources):
    if not sources:
        raise EvidenceGap("mechanism source manifest/SHA is missing")
    normalized = []
    for source in sources:
        path = source.get("manifest_path", source.get("path"))
        expected = source.get("manifest_sha256", source.get("sha256"))
        if not path or not expected or not Path(path).is_file() or digest(path) != expected:
            raise EvidenceGap("mechanism source manifest/SHA differs or is unavailable")
        normalized.append({**source, "verified_sha256": digest(path)})
    return normalized


def _scene_rows(rows, count, multipliers):
    lookup, names = {}, {}
    for row in rows:
        index, multiplier = row.get("dataset_index", row.get("mother_index")), row.get("demand_multiplier")
        if type(index) is not int or multiplier is None:
            raise EvidenceGap("scene lacks actual mother index/demand multiplier")
        key = index, float(multiplier)
        if key in lookup:
            raise EvidenceGap("duplicate mother/multiplier scene evidence")
        name = row.get("mother_scene_id", row.get("mother_id"))
        if not name or names.setdefault(index, name) != name:
            raise EvidenceGap("mother identities differ across demand tiers")
        lookup[key] = row
    expected = {(i, float(m)) for i in range(count) for m in multipliers}
    if set(lookup) != expected or len(rows) != len(expected):
        raise EvidenceGap(f"scene denominator differs: expected {len(expected)}, got {len(rows)}")
    return lookup


def _cv(result):
    return result["metrics_all"]["load_cv_all"]


def _feasible(result):
    metrics = result.get("metrics_all", {})
    return all(metrics.get(k) is True for k in ("geometry_feasible", "unique_assignment", "demand_conserved"))


def _qualified(result):
    if not _feasible(result):
        return False
    current = result.get("physical", {}).get("aircraft_service", {}).get("demand_weighted_carried_fraction")
    initial = result.get("initial_physical", {}).get("aircraft_service", {}).get("demand_weighted_carried_fraction")
    if initial is not None and (current is None or current < initial - .005 - 1e-12):
        return False
    return True


def _improvement(before, after):
    if before is None or after is None:
        return None
    if not np.isfinite(before) or not np.isfinite(after):
        raise EvidenceGap("nonfinite CV in source evidence")
    return float((before - after) / before) if before > 1e-12 else 0.


def _mother_final(lookup, count, multipliers):
    output = []
    for index in range(count):
        group = [lookup[index, float(m)] for m in multipliers]
        before = [row["result"]["initial_metrics_all"]["load_cv_all"] for row in group]
        after = [_cv(row["result"]) for row in group]
        score = _improvement(float(np.mean(before)), float(np.mean(after))) if None not in before + after else None
        output.append({"dataset_index": index, "mother_id": group[0].get("mother_scene_id", group[0].get("mother_id")),
                       "relative_cv_improvement": score,
                       "unscored_cv": score is None,
                       "published_qualified": all(_qualified(row["result"]) for row in group)})
    return output


def audit_direct_construction(rows, scenes, config=None, sources=None, *, expected_mother_count=20,
                              multipliers=MULTIPLIERS, at_update=2000):
    """Recheck actual D raw layouts; compare published DR with the same-I0 HR."""
    thresholds = {"update": 2000, "raw_feasible_fraction_minimum": .8,
                  "minimum_improved_mothers": 6, "relative_cv_improvement_strictly_greater": .01}
    statistics = {"expected_mothers": expected_mother_count, "expected_raw_rows": expected_mother_count * len(multipliers),
                  "actual_raw_rows": len(rows), "model_update": at_update}
    try:
        verified = _sources(sources)
        lookup = _scene_rows(rows, expected_mother_count, multipliers)
        if len(scenes) != expected_mother_count or at_update != 2000:
            raise EvidenceGap("03 pilot scene count/update differs")
        from .core import metrics_all
        from ..variable_design import Layout
        raw_rows, paired = [], []
        for index in range(expected_mother_count):
            model_cvs, rule_cvs, qualified = [], [], True
            for multiplier in multipliers:
                row = lookup[index, float(multiplier)]
                result, control = row["result"], row.get("HR_result")
                if control is None:
                    raise EvidenceGap("03 same-I0 HR result is missing")
                if not result.get("initial_sha256") or result.get("initial_sha256") != control.get("initial_sha256"):
                    raise EvidenceGap("03 model and HR do not share I0")
                scene = replace(scenes[index], demands=np.asarray(scenes[index].demands) * multiplier)
                if row.get("mother_scene_id", row.get("mother_id")) != scenes[index].scene_id:
                    raise EvidenceGap("03 mother identity differs from frozen input scene")
                raw = result.get("raw_layout")
                raw = Layout.from_record(raw) if isinstance(raw, dict) else raw
                feasible, reason = False, "no_raw_proposal"
                if raw is not None:
                    try:
                        metrics = metrics_all(scene, raw, config)
                        feasible = (metrics["beam_count"] == result["initial_metrics_all"]["beam_count"] and
                                    all(metrics.get(k) is True for k in ("geometry_feasible", "unique_assignment", "demand_conserved")))
                        reason = "raw_verified" if feasible else "raw_fixed_N_violation"
                    except (ValueError, TypeError, AttributeError) as error:
                        reason = str(error)
                raw_rows.append({"dataset_index": index, "demand_multiplier": multiplier,
                                 "raw_feasible": feasible, "reason": reason})
                model_cvs.append(_cv(result))
                rule_cvs.append(_cv(control))
                qualified &= _qualified(result) and _qualified(control)
            gain = _improvement(float(np.mean(rule_cvs)), float(np.mean(model_cvs))) if None not in model_cvs + rule_cvs else None
            paired.append({"dataset_index": index, "relative_cv_improvement_over_HR": gain,
                           "qualified": qualified, "demand_tiers": len(multipliers)})
        rate = sum(row["raw_feasible"] for row in raw_rows) / len(raw_rows)
        improved = [row["dataset_index"] for row in paired if row["qualified"] and
                    row["relative_cv_improvement_over_HR"] is not None and row["relative_cv_improvement_over_HR"] > .01]
        statistics.update(raw_feasible_fraction=rate, raw_rows=raw_rows, mother_metrics=paired,
                          improved_mothers=improved, denominator_policy="all_raw_proposals_including_invalid_and_missing")
        passed = rate >= .8 and len(improved) >= 6
        return _receipt("03", verified, statistics, thresholds, passed,
                        None if passed else "measured_03_raw_feasibility_or_HR_gain_gate_failed")
    except (EvidenceGap, KeyError) as error:
        return _receipt("03", sources, statistics, thresholds, None, str(error))


def _teacher_samples(path, expected_sha=None, *, allow_reference_gap=False):
    path = Path(path)
    if not path.is_file() or expected_sha is not None and digest(path) != expected_sha:
        raise EvidenceGap("teacher wrapper/evidence manifest SHA differs")
    record = json.loads(path.read_text(encoding="utf-8"))
    sources = [{"path": str(path), "sha256": digest(path)}]
    if "evidence_path" in record:
        target = Path(record["evidence_path"])
        expected = record.get("sha256", record.get("evidence_sha256"))
        if not expected or not target.is_file() or digest(target) != expected:
            raise EvidenceGap("wrapped teacher evidence SHA differs")
        path, record = target, json.loads(target.read_text(encoding="utf-8"))
        sources.append({"path": str(target), "sha256": digest(target)})
    statuses = ("completed", "attention_required") if allow_reference_gap else ("completed",)
    if record.get("format") != EVIDENCE_FORMAT or record.get("status") not in statuses:
        raise EvidenceGap("teacher evidence is not a complete audited manifest")
    samples = []
    for shard in record["shards"]:
        target = (path.parent / shard["path"]).resolve()
        if not target.is_relative_to(path.parent.resolve()) or digest(target) != shard["sha256"]:
            raise EvidenceGap("teacher shard path/SHA differs")
        with gzip.open(target, "rt", encoding="utf-8") as stream:
            values = [json.loads(line) for line in stream]
        if len(values) != shard["rows"]:
            raise EvidenceGap("teacher shard row count differs")
        samples.extend(values)
    if len(samples) != record["rows"]:
        raise EvidenceGap("teacher total row count differs")
    return record, samples, sources


def audit_fixed_ranker(policy, manifest_path, expected_sha=None, *, expected_mother_count=20, at_update=2000):
    """Frozen one-state-per-new-validation-mother pools; no new true trials."""
    thresholds = {"update": 2000, "topk": 8, "recall_advantage_minimum": .05,
                  "completed_best_improvement_not_worse": True}
    statistics = {"expected_mothers": expected_mother_count, "expected_label_states": expected_mother_count,
                  "model_update": at_update}
    sources = []
    try:
        manifest, samples, sources = _teacher_samples(manifest_path, expected_sha)
        if manifest.get("route") != "04" or len(samples) != expected_mother_count:
            raise EvidenceGap("04 fixed label states/route differs")
        if len({sample["mother_id"] for sample in samples}) != expected_mother_count:
            raise EvidenceGap("04 pools are not exactly one per independent mother")
        if any(float(sample.get("demand_multiplier", 1.)) != 1. or len(sample["actions"]) > 64 for sample in samples):
            raise EvidenceGap("04 fixed labels must be one 1.0 tier and at most 64 candidates")
        if manifest["trials"] > 1280:
            raise EvidenceGap("04 fixed labels exceeded reserved 1280 true trials")
        audits = [audit_ranker(policy, sample, 8) | {"mother_id": sample["mother_id"]} for sample in samples]
        if not all(row["complete"] for row in audits):
            raise EvidenceGap("04 fixed pools contain unknown/unfinished trial labels")
        scored = [row for row in audits if not row["no_improvement_state"]]
        advantage = float(np.mean([row["model_topk_recall"] - row["rule_topk_recall"] for row in scored])) if scored else None
        model_gain = float(np.mean([row["model_completed_best_improvement"] for row in audits]))
        rule_gain = float(np.mean([row["rule_completed_best_improvement"] for row in audits]))
        statistics.update(actual_label_states=len(samples), scored_recall_mothers=len(scored),
                          no_improvement_mothers=[row["mother_id"] for row in audits if row["no_improvement_state"]],
                          recall_advantage=advantage, model_completed_best_improvement_mean=model_gain,
                          rule_completed_best_improvement_mean=rule_gain, audits=audits,
                          true_trials=manifest["trials"], true_trial_budget=1280,
                          denominator_policy="mother_equal_mean; recall excludes zero-positive pools explicitly")
        passed = advantage is not None and advantage >= .05 and model_gain >= rule_gain - 1e-12
        return _receipt("04", sources, statistics, thresholds, passed,
                        None if passed else "measured_04_top8_recall_or_true_gain_gate_failed")
    except (EvidenceGap, KeyError, ValueError) as error:
        return _receipt("04", sources, statistics, thresholds, None, str(error))


def _normalized_layout_cv(layout, demands, n):
    labels = np.asarray(layout["assignment"] if isinstance(layout, dict) else layout.assignment)
    if labels.shape != demands.shape or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= n):
        raise EvidenceGap("saved layout aircraft ownership is invalid")
    if len(np.unique(labels)) != n:
        raise EvidenceGap("saved accepted layout has empty beams")
    loads = np.bincount(labels, weights=demands, minlength=n)
    return float(loads.std() / loads.mean()) if loads.mean() else None


def audit_region_oracle(teacher_manifest_path, expected_sha=None, *, expected_mother_count=20,
                        multipliers=MULTIPLIERS, sources=None, mother_indices=None):
    """Recompute one saved 05 label collection, stratified state→tier→mother."""
    thresholds = {"minimum_improved_mothers": 8, "oracle_uniform_advantage_minimum": .01,
                  "max_regions_per_state": 12, "carry_drop_maximum": .005}
    statistics = {"expected_mothers": expected_mother_count, "expected_mother_tiers": expected_mother_count * len(multipliers)}
    verified = []
    try:
        manifest, samples, verified = _teacher_samples(teacher_manifest_path, expected_sha)
        if manifest.get("route") != "05":
            raise EvidenceGap("region oracle requires the existing route05 development teacher")
        if sources:
            verified.extend(_sources(sources))
        if mother_indices is None:
            origin = Path(teacher_manifest_path)
            for config_path in (origin.parent / "config.json", origin.parent.parent / "config.json"):
                if config_path.is_file():
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    data = config.get("datasets", {}).get("development")
                    if data and digest(data["path"]) == data["sha256"]:
                        from ..variable_dataset import load_variable_dataset
                        dataset = load_variable_dataset(data["path"])
                        mother_indices = {scene.scene_id: i for i, scene in enumerate(dataset.scenes)}
                        verified.append({"path": data["path"], "sha256": data["sha256"], "role": "development"})
                        break
        if mother_indices is None or set(mother_indices.values()) != set(range(expected_mother_count)):
            raise EvidenceGap("05 authoritative development mother indices are missing")
        states, grouped = [], {}
        for sample in samples:
            name, multiplier = sample["mother_id"], float(sample["demand_multiplier"])
            if name not in mother_indices or multiplier not in multipliers:
                raise EvidenceGap("05 labels come from an unexpected mother/tier")
            objects, trials = sample.get("objects", []), sample.get("trials", [])
            if (not 1 <= len(objects) <= 12 or len(objects) != len(trials) or
                    len(sample["known"]) != len(objects) or len(sample["utility"]) != len(objects) or not all(sample["known"])):
                raise EvidenceGap("05 region pool is incomplete or contains unknown trials")
            if any(key not in sample for key in ("scene_demands_bps", "scene_positions", "initial_carry", "initial_metrics", "initial_layout_sha256")):
                raise EvidenceGap("05 raw demand/position/I0 audit payload is incomplete")
            demands = np.asarray(sample["scene_demands_bps"], dtype=float)
            if demands.ndim != 1 or np.any(demands < 0) or not np.isfinite(demands).all():
                raise EvidenceGap("05 raw scene demands are invalid")
            normalized = np.asarray(sample["aircraft"], dtype=float)[:, 5]
            if not np.allclose(normalized, demands / max(float(demands.sum()), 1.), rtol=1e-8, atol=1e-10):
                raise EvidenceGap("05 raw demands disagree with immutable network demand features")
            before_n = len(sample["state"]["centers"])
            before = _normalized_layout_cv(sample["state"], demands, before_n)
            gains = []
            for index, (obj, trial) in enumerate(zip(objects, trials)):
                if trial.get("object") != obj or not trial.get("label", {}).get("known"):
                    raise EvidenceGap("05 trial does not correspond to the saved region menu")
                result = trial.get("result")
                if not result:
                    raise EvidenceGap("05 known region lacks an actual trial result")
                gain = 0.
                if result.get("accepted"):
                    layout = result.get("layout")
                    if layout is None or len(layout["centers"]) != before_n:
                        raise EvidenceGap("05 accepted region does not preserve N")
                    after = _normalized_layout_cv(layout, demands, before_n)
                    outside_beams = np.flatnonzero(~np.isin(np.arange(before_n), obj["beams"]))
                    outside_aircraft = ~np.isin(np.asarray(sample["state"]["assignment"]), obj["beams"])
                    if (not np.array_equal(np.asarray(layout["assignment"])[outside_aircraft],
                                           np.asarray(sample["state"]["assignment"])[outside_aircraft]) or
                            not np.array_equal(np.asarray(layout["centers"])[outside_beams],
                                               np.asarray(sample["state"]["centers"])[outside_beams])):
                        raise EvidenceGap("05 accepted regional trial changed outside assignments/centers")
                    recorded = result.get("metrics_all", {}).get("load_cv_all")
                    if after is None or recorded is None or not np.isclose(after, recorded, rtol=1e-8, atol=1e-10):
                        raise EvidenceGap("05 saved true CV differs from demand/assignment recomputation")
                    old_carry = sample["initial_carry"]
                    new_carry = result.get("physical", {}).get("aircraft_service", {}).get("demand_weighted_carried_fraction")
                    if new_carry is None:
                        raise EvidenceGap("05 accepted region lacks full-scene physical carry")
                    if old_carry is not None and new_carry < old_carry - .005 - 1e-12:
                        raise EvidenceGap("05 accepted trial violates the initial carry safeguard")
                    gain = max(0., _improvement(before, after) or 0.)
                    if not np.isclose(sample["utility"][index], gain, rtol=1e-8, atol=1e-10):
                        raise EvidenceGap("05 utility label differs from actual full-scene gain")
                elif "unknown" in str(result.get("reason", "")):
                    raise EvidenceGap("05 unknown region was labelled as known")
                gains.append(gain)
            oracle, uniform = max(gains), float(np.mean(gains))
            row = {"dataset_index": mother_indices[name], "mother_id": name, "demand_multiplier": multiplier,
                   "state_index": sample.get("trajectory_step", 0), "regions": len(gains),
                   "cv_before": before, "all_zero_demand": before is None,
                   "oracle_gain": oracle if before is not None else None,
                   "uniform_expected_gain": uniform if before is not None else None,
                   "oracle_uniform_advantage": oracle - uniform if before is not None else None,
                   "region_gains": gains}
            states.append(row)
            grouped.setdefault((mother_indices[name], multiplier), []).append(row)
        expected = {(i, float(m)) for i in range(expected_mother_count) for m in multipliers}
        if set(grouped) != expected:
            raise EvidenceGap("05 complete three-tier evidence is missing for some mothers")
        mother_rows = []
        def mean_defined(values):
            defined = [value for value in values if value is not None]
            return float(np.mean(defined)) if defined else None
        for index in range(expected_mother_count):
            tier = [{"multiplier": m, "states": len(grouped[index, float(m)]),
                     "oracle_gain": mean_defined([r["oracle_gain"] for r in grouped[index, float(m)]]),
                     "advantage": mean_defined([r["oracle_uniform_advantage"] for r in grouped[index, float(m)]])}
                    for m in multipliers]
            mother_rows.append({"dataset_index": index, "oracle_gain": mean_defined([r["oracle_gain"] for r in tier]),
                                "oracle_uniform_advantage": mean_defined([r["advantage"] for r in tier]), "tiers": tier})
        advantage = mean_defined([row["oracle_uniform_advantage"] for row in mother_rows])
        improved = [row["dataset_index"] for row in mother_rows if row["oracle_gain"] is not None and row["oracle_gain"] > 1e-12]
        statistics.update(actual_states=len(states), actual_mother_tiers=len(grouped),
                          mother_count=len(mother_rows), improved_mothers=improved,
                          unscored_cv_mothers=[row["dataset_index"] for row in mother_rows if row["oracle_gain"] is None],
                          oracle_uniform_advantage=advantage, mother_metrics=mother_rows,
                          state_metrics=states, true_trials=manifest["trials"],
                          denominator_policy="average states within tier, tiers within mother, then equal mothers")
        passed = len(improved) >= 8 and advantage is not None and advantage >= .01
        return _receipt("05", verified, statistics, thresholds, passed,
                        None if passed else "measured_05_oracle_uniform_selection_gate_failed")
    except (EvidenceGap, KeyError, ValueError) as error:
        return _receipt("05", verified or sources, statistics, thresholds, None, str(error))


def audit_teacher_reachability(manifest_path, expected_sha=None, route=None, *, mother_indices=None,
                               expected_mother_count=20, multipliers=MULTIPLIERS, sources=None):
    """Actual offline teacher probes for02/03, never online T/HR substitutes.

    Recompute complete layouts and demand CV from each problem receipt. Reuse
    the recorded full-scene physical evaluation, preserving the I0 carry guard;
    no new teacher search or physical call is performed.
    """
    thresholds = {"minimum_improved_mothers": 6, "relative_cv_improvement_strictly_greater": .01,
                  "every_mother_has_legal_reference": True, "carry_drop_maximum": .005,
                  "probe_budget": {"02": {"problems": 60, "trials": 38400, "seconds": 600.},
                                   "03": {"problems": 60, "trials": 7680, "seconds": 240.}}}
    statistics = {"expected_mothers": expected_mother_count,
                  "expected_complete_problems": expected_mother_count * len(multipliers)}
    verified = []
    route = str(route).zfill(2) if route is not None else None
    try:
        manifest, _, verified = _teacher_samples(manifest_path, expected_sha, allow_reference_gap=True)
        route = route or manifest.get("route")
        if route not in ("02", "03") or manifest.get("route") != route:
            raise EvidenceGap("reachability audit needs an actual route02/03 teacher probe")
        if sources:
            verified.extend(_sources(sources))
        if mother_indices is None or set(mother_indices.values()) != set(range(expected_mother_count)):
            raise EvidenceGap("teacher probe authoritative development mother indices are missing")
        source_path = Path(manifest_path)
        wrapper = json.loads(source_path.read_text(encoding="utf-8"))
        evidence_path = Path(wrapper["evidence_path"]) if "evidence_path" in wrapper else source_path
        receipts = manifest.get("problem_receipts", [])
        statistics["actual_complete_problems"] = len(receipts)
        if len(receipts) != statistics["expected_complete_problems"]:
            raise EvidenceGap("teacher probe complete problem receipt denominator differs")
        cap = thresholds["probe_budget"][route]
        if manifest.get("trials", 0) > cap["trials"] or manifest.get("problems", 0) > cap["problems"]:
            raise EvidenceGap("teacher probe exceeds its reserved problem/true-trial subledger")
        from .core import layout_sha256, metrics_all
        from ..scenes import CoverageScene
        from ..variable_design import Layout
        rows, identities = [], set()
        searched, fallback, unknown, complete_searches = 0, 0, 0, 0
        for receipt in receipts:
            path = (evidence_path.parent / receipt["path"]).resolve()
            if not path.is_relative_to(evidence_path.parent.resolve()) or digest(path) != receipt["sha256"]:
                raise EvidenceGap("teacher problem receipt path/SHA differs")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                problem = json.load(stream)
            if problem.get("format") != "s1_nondrl_teacher_problem_v1" or problem.get("route") != route:
                raise EvidenceGap("teacher problem receipt schema/route differs")
            origin = manifest["provenance"]
            actual = problem.get("provenance", {})
            if actual != origin:
                marker = problem.get("execution_source", {})
                segments = [s.get("execution_source") for s in manifest.get("execution_segments", [])]
                if (marker not in segments or marker.get("git_commit") != actual.get("git_commit") or
                        marker.get("source_sha256") != actual.get("source_sha256") or
                        any(actual.get(k) != origin.get(k) for k in set(actual) | set(origin)
                            if k not in ("git_commit", "source_sha256"))):
                    raise EvidenceGap("teacher problem source/config/dataset provenance differs")
            scope = problem.get("search_scope", {})
            calls = problem.get("calls", {}).get("candidates", 0)
            is_fallback = scope.get("reference_only") is True
            is_searched = scope.get("searched", calls > 0) is True
            is_unknown = scope.get("unknown") is True or is_fallback or (not scope and calls == 0)
            is_complete = scope.get("complete_search", not is_unknown and "unknown" not in problem.get("completion_reason", "")) is True
            if is_fallback and (calls or scope.get("trials") or problem.get("teacher_elapsed_seconds") != 0):
                raise EvidenceGap("I0 reference-only receipt consumed a search budget")
            searched += is_searched
            fallback += is_fallback
            unknown += is_unknown or not is_complete
            complete_searches += is_searched and is_complete
            name, multiplier = problem["mother_id"], float(problem["demand_multiplier"])
            if name not in mother_indices or multiplier not in multipliers:
                raise EvidenceGap("teacher probe belongs to an unexpected mother/tier")
            key = mother_indices[name], multiplier
            if key in identities:
                raise EvidenceGap("teacher probe duplicates a mother/tier problem")
            identities.add(key)
            positions, demands = np.asarray(problem["scene_positions"]), np.asarray(problem["scene_demands_bps"])
            scene = CoverageScene(name, positions, demands, np.empty((0, 2)), problem["aircraft_ids"], {}, problem["source_hash"])
            initial, final = Layout.from_record(problem["initial_layout"]), Layout.from_record(problem["final_layout"])
            if layout_sha256(initial) != problem["initial_sha256"]:
                raise EvidenceGap("teacher probe actual I0 layout SHA differs")
            first, last = metrics_all(scene, initial, problem["config"]), metrics_all(scene, final, problem["config"])
            if first["beam_count"] != last["beam_count"] or first["beam_count"] != problem["target_n"]:
                raise EvidenceGap("teacher probe changed the fixed-N mechanism")
            for computed, saved in ((first, problem["initial_metrics_all"]), (last, problem["final_metrics_all"])):
                if computed["load_cv_all"] is None or saved["load_cv_all"] is None:
                    if computed["load_cv_all"] != saved["load_cv_all"]:
                        raise EvidenceGap("teacher probe zero-demand CV differs")
                elif not np.isclose(computed["load_cv_all"], saved["load_cv_all"], rtol=1e-9, atol=1e-11):
                    raise EvidenceGap("teacher probe CV differs from full raw-layout recomputation")
            before_carry = problem["initial_physical"]["aircraft_service"]["demand_weighted_carried_fraction"]
            after_carry = problem["final_physical"]["aircraft_service"]["demand_weighted_carried_fraction"]
            if before_carry is not None and (after_carry is None or after_carry < before_carry - .005 - 1e-12):
                raise EvidenceGap("teacher probe final reference violates actual I0 carry guard")
            rows.append({"dataset_index": key[0], "mother_scene_id": name, "demand_multiplier": multiplier,
                         "result": {"metrics_all": last, "initial_metrics_all": first,
                                    "physical": problem["final_physical"], "initial_physical": problem["initial_physical"]},
                         "initial_sha256": problem["initial_sha256"], "receipt_path": str(path),
                         "receipt_sha256": receipt["sha256"], "teacher_elapsed_seconds": problem["teacher_elapsed_seconds"],
                         "search_scope": scope, "searched": is_searched, "reference_only": is_fallback,
                         "unknown_search": is_unknown or not is_complete,
                         "execution_source": problem.get("execution_source")})
        lookup = _scene_rows(rows, expected_mother_count, multipliers)
        mother_rows = _mother_final(lookup, expected_mother_count, multipliers)
        improved = [row["dataset_index"] for row in mother_rows if row["published_qualified"] and
                    row["relative_cv_improvement"] is not None and row["relative_cv_improvement"] > .01]
        statistics.update(improved_mothers=improved, mother_metrics=mother_rows,
                          legal_reference_mothers=sorted(mother_indices.values()), problem_metrics=rows,
                          true_trials=manifest["trials"], recorded_teacher_seconds=manifest["elapsed_seconds"],
                          budget_overrun_seconds=max(0., manifest["elapsed_seconds"] - cap["seconds"]),
                          searched_problems=searched, reference_only_problems=fallback,
                          unknown_search_problems=unknown, complete_search_problems=complete_searches,
                          denominator_policy="mean three demand tiers within mother; 20 independent mothers",
                          no_new_physical_or_teacher_calls=True)
        passed = len(improved) >= 6 and all(row["published_qualified"] for row in mother_rows)
        if not passed and unknown:
            return _receipt(route, verified, statistics, thresholds, None,
                            "offline_teacher_search_evidence_gap: bounded unknown/reference-only problems remain")
        return _receipt(route, verified, statistics, thresholds, passed,
                        None if passed else "measured_offline_teacher_reachability_gate_failed")
    except (EvidenceGap, KeyError, TypeError, ValueError) as error:
        return _receipt(route or "teacher", verified or sources, statistics, thresholds, None, str(error))


def audit_scene_traces(rows, route, sources, *, controls=None, expected_mother_count=20, multipliers=MULTIPLIERS):
    """06–08 finite online mechanism counts, not a new optimization/evaluation."""
    route = str(route).zfill(2)
    thresholds = {"relative_cv_improvement_strictly_greater": .01, "minimum_improved_mothers": 8,
                  "06_minimum_opportunity_mothers": 12, "07_minimum_nontrivial_proposal_mothers": 15,
                  "08_minimum_completed_pipeline_mothers": 12, "08_net_improved_mother_advantage": 3}
    statistics = {"expected_mothers": expected_mother_count, "expected_rows": expected_mother_count * len(multipliers),
                  "actual_rows": len(rows)}
    try:
        if route not in ("06", "07", "08"):
            raise EvidenceGap("trace mechanism audit is only defined for routes06–08")
        verified = _sources(sources)
        lookup = _scene_rows(rows, expected_mother_count, multipliers)
        mothers = _mother_final(lookup, expected_mother_count, multipliers)
        improved = [row["dataset_index"] for row in mothers if row["published_qualified"] and
                    row["relative_cv_improvement"] is not None and row["relative_cv_improvement"] > .01]
        mechanism_mothers, diagnostics = [], []
        for index in range(expected_mother_count):
            successes = []
            for multiplier in multipliers:
                result = lookup[index, float(multiplier)]["result"]
                trace = result.get("trace")
                if not isinstance(trace, list):
                    raise EvidenceGap("raw scene trace is missing")
                base = result["initial_metrics_all"]["load_cv_all"]
                if route == "06":
                    opportunities = [r for r in trace if r.get("accepted") and r.get("source", "").startswith("06_") and
                                     _improvement(base, r.get("metrics_all", {}).get("load_cv_all")) is not None and
                                     _improvement(base, r.get("metrics_all", {}).get("load_cv_all")) > 0]
                    success = bool(opportunities)
                elif route == "07":
                    initial = result.get("initial_layout")
                    initial = initial.to_record() if hasattr(initial, "to_record") else initial
                    if initial is None:
                        raise EvidenceGap("07 initial raw layout is missing")
                    proposals = [r for r in trace if r.get("source", "").startswith("07_")]
                    success = any(r.get("layout") is not None and r.get("metrics_all") and
                                  r["metrics_all"].get("beam_count") == result["initial_metrics_all"]["beam_count"] and
                                  all(r["metrics_all"].get(k) is True for k in ("geometry_feasible", "unique_assignment", "demand_conserved")) and
                                  (r["layout"]["assignment"] != initial["assignment"] or
                                   r["layout"]["centers"] != initial["centers"]) for r in proposals)
                else:
                    transport = [r for r in trace if r.get("source") == "08_transport" and
                                 r.get("model") == "row_conserving_entropy_soft_quota_v1"]
                    for item in transport:
                        if item.get("row_residual") is None or not np.isfinite(item["row_residual"]):
                            raise EvidenceGap("08 soft transport residual is missing/nonfinite")
                    hard = [r for r in trace if r.get("source", "").startswith("08_OT_") and
                            r.get("metrics_all") and not r.get("finished_after_deadline") and
                            all(r["metrics_all"].get(k) is True for k in ("geometry_feasible", "unique_assignment", "demand_conserved"))]
                    success = bool(transport) and all(r["row_residual"] <= 1e-8 for r in transport) and bool(hard)
                successes.append(success)
                diagnostics.append({"dataset_index": index, "demand_multiplier": multiplier,
                                    "mechanism_completed": success, "trace_rows": len(trace)})
            if (all(successes) if route == "08" else any(successes)):
                mechanism_mothers.append(index)
        minimum = {"06": 12, "07": 15, "08": 12}[route]
        passed = len(improved) >= 8 and len(mechanism_mothers) >= minimum
        control_statistics = {}
        if route == "08":
            if not controls or not all(method in controls for method in ("voronoi", "integer")):
                raise EvidenceGap("08 matching noOT Voronoi and integer controls are missing")
            for method in ("voronoi", "integer"):
                comparator = _scene_rows(controls[method], expected_mother_count, multipliers)
                for key, row in comparator.items():
                    if not row["result"].get("initial_sha256") or row["result"].get("initial_sha256") != lookup[key]["result"].get("initial_sha256"):
                        raise EvidenceGap("08 noOT controls do not share the same I0")
                control_mothers = _mother_final(comparator, expected_mother_count, multipliers)
                counts = [r["dataset_index"] for r in control_mothers if r["published_qualified"] and
                          r["relative_cv_improvement"] is not None and r["relative_cv_improvement"] > .01]
                advantage = len(improved) - len(counts)
                control_statistics[method] = {"improved_mothers": counts, "net_improved_mother_advantage": advantage,
                                              "mother_metrics": control_mothers}
                passed &= advantage >= 3
        statistics.update(mother_metrics=mothers, improved_mothers=improved,
                          mechanism_mothers=mechanism_mothers, mechanism_count=len(mechanism_mothers),
                          scene_diagnostics=diagnostics, controls=control_statistics,
                          denominator_policy="three tiers summarized per actual mother; never count tiers as mothers")
        return _receipt(route, verified, statistics, thresholds, passed,
                        None if passed else f"measured_{route}_mechanism_gate_failed")
    except (EvidenceGap, KeyError, TypeError, ValueError) as error:
        return _receipt(route, sources, statistics, thresholds, None, str(error))
