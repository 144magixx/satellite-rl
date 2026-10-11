"""Supervised non-DRL training, strict recovery, and common online executor."""
from __future__ import annotations

from collections import OrderedDict
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .models import (MODEL_FORMAT, canonical_route, make_model, supervised_loss)
from .teachers import (EVIDENCE_FORMAT, atomic_json, beam_features, digest, feature16,
                       mlm_sample, scene_features, state_sample)

UPDATE_CAPS = {"01": 10000, "02": 10000, "03": 12000, "04": 8000, "05": 4000}
CHECKPOINT_FORMAT = "s1_nondrl_checkpoint_v1"
LEARNING_METHODS = {"01": ("M", "MO", "H", "O", "HO"), "02": ("BC", "H", "T", "R"),
                    "03": ("D", "DR", "H", "HR"), "04": ("L", "H", "R", "F"),
                    "05": ("L", "rule", "worst", "random")}


class ScientificStop(RuntimeError):
    pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_sample(sample, route):
    route = canonical_route(route)
    if sample.get("route", route) != route:
        raise ValueError("supervision route differs")
    aircraft = np.asarray(sample["aircraft"], dtype=float)
    if aircraft.ndim != 2 or aircraft.shape[1] != 8 or not len(aircraft) or not np.isfinite(aircraft).all():
        raise ValueError("invalid aircraft features")
    if not sample.get("mother_id"):
        raise ValueError("sample lacks mother identity")
    if route == "01":
        candidates = np.asarray(sample["candidates"], dtype=float)
        labels, visible = np.asarray(sample["labels"]), np.asarray(sample["visible"])
        hidden, legal = np.asarray(sample["hidden"], dtype=bool), np.asarray(sample["legal"], dtype=bool)
        if candidates.ndim != 2 or candidates.shape[1] != 8 or not 0 < len(candidates) <= 1000:
            raise ValueError("invalid MLM candidate set")
        if legal.shape != (len(aircraft), len(candidates)) or labels.shape != (len(aircraft),):
            raise ValueError("invalid MLM edge/label shape")
        if visible.shape != labels.shape or hidden.shape != labels.shape or not hidden.any():
            raise ValueError("invalid MLM masking")
        if ((labels < 0) | (labels >= len(candidates))).any() or not legal[np.arange(len(labels)), labels].all():
            raise ValueError("MLM labels outside legal candidate domain")
        if not np.array_equal(visible < 0, hidden) or not np.array_equal(visible[~hidden], labels[~hidden]):
            raise ValueError("MLM visible constraints conflict with labels")
        # Verify visible-load fields algebraically, so hidden labels cannot leak.
        known = visible >= 0
        demand = aircraft[:, 5]
        loads = np.bincount(visible[known], weights=demand[known], minlength=len(candidates))
        counts = np.bincount(visible[known], minlength=len(candidates)) / len(visible)
        if not np.allclose(candidates[:, 5], loads, atol=1e-7) or not np.allclose(candidates[:, 6], counts, atol=1e-7):
            raise ValueError("MLM hidden full loads/counts leaked into input")
        original = np.asarray(sample["original_assignment"])
        if original.shape != labels.shape or not legal[np.arange(len(labels)), original].all():
            raise ValueError("MLM original assignment is not retained")
        return True
    if route == "03":
        if any(key in sample for key in ("beams", "candidates", "visible", "state")):
            raise ValueError("direct constructor input contains layout context")
        n = sample["target_n"]
        if type(n) is not int or not 1 <= n <= 200:
            raise ValueError("constructor N outside 1..200")
        if np.asarray(sample["positions"]).shape != (len(aircraft), 2):
            raise ValueError("constructor positions do not align")
        refs = sample["references"]
        if not 1 <= len(refs) <= 4:
            raise ValueError("constructor needs one to four whole references")
        for ref in refs:
            labels = np.asarray(ref["assignment"])
            if np.asarray(ref["centers"]).shape != (n, 2) or labels.shape != (len(aircraft),):
                raise ValueError("constructor reference shape differs")
            if set(labels.tolist()) != set(range(n)):
                raise ValueError("constructor reference has empty/nonexistent beam")
        return True
    actions, beams = np.asarray(sample["actions"]), np.asarray(sample["beams"])
    if (actions.ndim != 2 or actions.shape[1] != 16 or not len(actions) or
            beams.ndim != 2 or beams.shape[1] != 8 or not len(beams)):
        raise ValueError("invalid candidate/state shape")
    if not np.isfinite(actions).all() or not np.isfinite(beams).all():
        raise ValueError("non-finite state features")
    if route == "02":
        positives = sample["positive_actions"]
        if not positives and not sample.get("stop_positive", False):
            raise ValueError("truncated BC trajectory cannot supply STOP label")
        if any(type(i) is not int or not 0 <= i < len(actions) for i in positives):
            raise ValueError("BC demonstration not in execution menu")
        trials = sample.get("trials", [])
        if positives and (len(trials) != len(actions) or
                          any(not trials[i].get("result", {}).get("accepted") for i in positives)):
            raise ValueError("BC positive action lacks a completed executable trial")
        if sample.get("stop_positive") and sample.get("stop_reason") not in (
                "verified_no_improvement", "verified_empty_menu"):
            raise ValueError("BC STOP is not exhaustive verification")
        return True
    known, utility = np.asarray(sample["known"], bool), np.asarray(sample["utility"], float)
    if known.shape != (len(actions),) or utility.shape != known.shape or not np.isfinite(utility).all():
        raise ValueError("invalid true trial labels")
    if np.any(utility[~known] != 0):
        raise ValueError("unknown candidate cannot have positive/negative utility")
    if route == "04":
        values = utility[known]
        return len(values) > 1 and float(np.ptp(values)) > 1e-8
    return bool(known.any())


class EvidenceDataset:
    """Verified compressed shards, bounded resident cache, mother-balanced draws."""
    def __init__(self, path, route):
        self.path, self.route = Path(path), canonical_route(route)
        if self.path.is_dir():
            self.path /= "manifest.json"
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != EVIDENCE_FORMAT or self.manifest.get("route") != self.route:
            raise ValueError("teacher manifest schema/route differs")
        if self.manifest.get("status") not in ("completed", "stopped"):
            raise ScientificStop("teacher has not produced accepted complete evidence")
        self.sha256, self.cache = digest(self.path), OrderedDict()
        self.mothers, self.filtered = {}, 0
        for index, record in enumerate(self.manifest["shards"]):
            target = (self.path.parent / record["path"]).resolve()
            if not target.is_relative_to(self.path.parent.resolve()) or digest(target) != record["sha256"]:
                raise ValueError("teacher shard path/hash differs")
            rows = self.shard(index)
            if len(rows) != record["rows"]:
                raise ValueError("teacher shard row count differs")
            for row, sample in enumerate(rows):
                if validate_sample(sample, self.route):
                    self.mothers.setdefault(sample["mother_id"], []).append((index, row))
                else:
                    self.filtered += 1
        self.keys = sorted(self.mothers)
        if not self.keys:
            raise ScientificStop("no trainable supervised states; unknown/tied records retained in evidence")

    def shard(self, index):
        if index not in self.cache:
            path = self.path.parent / self.manifest["shards"][index]["path"]
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                self.cache[index] = [json.loads(line) for line in stream]
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def draw(self, rng, size):
        samples, positions = [], []
        for _ in range(size):
            mother = self.keys[int(rng.integers(len(self.keys)))]
            entries = self.mothers[mother]
            shard, row = entries[int(rng.integers(len(entries)))]
            samples.append(self.shard(shard)[row])
            positions.append({"mother_id": mother, "shard": shard, "row": row})
        return samples, positions


def collate(samples, route, device="cpu"):
    route = canonical_route(route)
    batch = {}

    def pad(name, feature=8, mask_name=None):
        lengths = [len(sample[name]) for sample in samples]
        maximum = max(lengths)
        values = np.zeros((len(samples), maximum, feature), dtype=np.float32)
        mask = np.zeros((len(samples), maximum), dtype=bool)
        for i, sample in enumerate(samples):
            values[i, :lengths[i]], mask[i, :lengths[i]] = sample[name], True
        batch[name] = torch.as_tensor(values, device=device)
        batch[mask_name or name.rstrip("s") + "_mask"] = torch.as_tensor(mask, device=device)

    pad("aircraft", mask_name="aircraft_mask")
    if route == "01":
        pad("candidates", mask_name="candidate_mask")
        a, c = batch["aircraft"].shape[1], batch["candidates"].shape[1]
        legal = np.zeros((len(samples), a, c), bool)
        visible, labels = np.full((len(samples), a), -1, dtype=np.int64), np.zeros((len(samples), a), dtype=np.int64)
        hidden = np.zeros((len(samples), a), bool)
        for i, sample in enumerate(samples):
            length, count = len(sample["aircraft"]), len(sample["candidates"])
            legal[i, :length, :count] = sample["legal"]
            visible[i, :length], labels[i, :length], hidden[i, :length] = sample["visible"], sample["labels"], sample["hidden"]
        batch.update(legal=torch.as_tensor(legal, device=device), visible=torch.as_tensor(visible, device=device),
                     labels=torch.as_tensor(labels, device=device), hidden=torch.as_tensor(hidden, device=device))
    elif route == "03":
        n = max(sample["target_n"] for sample in samples)
        slots = np.zeros((len(samples), n), bool)
        for i, sample in enumerate(samples):
            slots[i, :sample["target_n"]] = True
        pad("positions", feature=2, mask_name="positions_mask")
        batch["slot_mask"] = torch.as_tensor(slots, device=device)
    else:
        pad("beams", mask_name="beam_mask")
        pad("actions", feature=16, mask_name="action_mask")
        size = batch["actions"].shape[1]
        if route == "02":
            positive = np.zeros((len(samples), size + 1), bool)
            for i, sample in enumerate(samples):
                positive[i, sample.get("positive_actions", [])] = True
                positive[i, -1] = sample.get("stop_positive", False)
            batch["positives"] = torch.as_tensor(positive, device=device)
        else:
            known, utility = np.zeros((len(samples), size), bool), np.zeros((len(samples), size), np.float32)
            for i, sample in enumerate(samples):
                known[i, :len(sample["actions"])], utility[i, :len(sample["actions"])] = (
                    sample.get("known", [False] * len(sample["actions"])),
                    sample.get("utility", [0.] * len(sample["actions"])))
            batch.update(known=torch.as_tensor(known, device=device), utility=torch.as_tensor(utility, device=device))
    return batch


def rng_state(sampler):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": sampler.bit_generator.state}


def restore_rng(state, sampler):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG topology differs")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    sampler.bit_generator.state = state["sampler"]


def isolated_callback(callback, payload, sampler):
    if callback is None:
        return None
    state = rng_state(sampler)
    try:
        return callback(payload)
    finally:
        restore_rng(state, sampler)


def save_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def torch_load(path, device="cpu"):
    # Explicitly trusted task-owned checkpoints, never arbitrary downloaded PT.
    return torch.load(path, map_location=device, weights_only=False)


def train(evidence_path, route, config, output, resume=False, stop_callback=None,
          validation_callback=None, progress_callback=None, provenance=None):
    """One fixed endpoint, with all state needed for byte-consistent recovery.

    The runner owns formal workbench authorization and reachability gate. A
    caller must set engineering=True for a deliberately tiny CPU check; formal
    training requires a real CUDA device and seed 51001.
    """
    route = canonical_route(route)
    config, provenance = dict(config), dict(provenance or {})
    seed, total = int(config.get("seed", 51001)), int(config.get("updates", UPDATE_CAPS[route]))
    device = str(config.get("device", "cuda"))
    engineering = bool(config.get("engineering", False))
    if seed != 51001 or not 0 < total <= UPDATE_CAPS[route]:
        raise ValueError("learning seed/update budget differs from approved route")
    if not engineering and (not device.startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("formal supervised training requires verified DQN CUDA")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; refusing silent CPU fallback")
    batch_size = int(config.get("batch_size", 16 if route in ("03", "04") else 32))
    interval = int(config.get("checkpoint_interval", 1000))
    microbatch_size = min(batch_size, int(config.get("microbatch_size", 4)))
    if batch_size < 1 or interval < 1 or microbatch_size < 1:
        raise ValueError("invalid batch/checkpoint interval")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path, loss_path = folder / "checkpoint.pt", folder / "losses.jsonl"
    dataset = EvidenceDataset(evidence_path, route)
    if config.get("dataset_sha") not in (None, dataset.sha256):
        raise ValueError("frozen dataset hash differs")
    contract = {"format": MODEL_FORMAT, "route": route, "config_sha": canonical_hash(config),
                "source_sha": provenance.get("source_sha", provenance.get("git_commit", config.get("source_sha"))),
                "source_manifest_sha256": provenance.get("source_sha256"),
                "schema_version": provenance.get("schema_version", 7),
                "dataset_sha": dataset.sha256, "provenance": provenance}
    if not engineering and (not contract["source_sha"] or not provenance.get("reachability_sha")):
        raise ValueError("formal learning needs frozen source and reachability provenance")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sampler = np.random.default_rng(seed)
    model = make_model(route, config.get("network")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", .0003)))
    updates, cursor, previous_elapsed = 0, 0, 0.
    if resume:
        state = torch_load(checkpoint_path, device)
        if state.get("format") != CHECKPOINT_FORMAT or state["contract"] != contract:
            raise ValueError("checkpoint schema/source/dataset/config/provenance differs")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], sampler)
        updates, cursor, previous_elapsed = state["updates"], state["cursor"], state["elapsed_seconds"]
        # Complete log records beyond a checkpoint are retained as recovery evidence.
        if loss_path.exists():
            raw = loss_path.read_bytes()
            offset = state["loss_bytes"]
            if len(raw) < offset or hashlib.sha256(raw[:offset]).hexdigest() != state["loss_sha256"]:
                raise ValueError("checkpoint loss prefix differs")
            if len(raw) > offset:
                orphan = folder / f"losses_after_checkpoint_{updates}_{digest(loss_path)[:12]}.jsonl"
                if not orphan.exists():
                    orphan.write_bytes(raw[offset:])
                loss_path.write_bytes(raw[:offset])
        elif state["loss_bytes"]:
            raise ValueError("checkpoint loss log is missing")
    elif checkpoint_path.exists() or loss_path.exists():
        raise FileExistsError("training output exists; use strict same-contract resume")
    if not 0 <= updates <= total:
        raise ValueError("checkpoint update counter exceeds fixed budget")
    started, last_loss, status = time.perf_counter(), None, "training"
    validation_rows = list(state.get("validation", [])) if resume else []
    sampled_last = []

    def checkpoint():
        elapsed = previous_elapsed + time.perf_counter() - started
        raw = loss_path.read_bytes() if loss_path.exists() else b""
        state = {"format": CHECKPOINT_FORMAT, "contract": contract, "network": config.get("network", {}),
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(sampler),
                 "updates": updates, "total_updates": total, "cursor": cursor,
                 "sampled_last": sampled_last, "elapsed_seconds": elapsed,
                 "validation": validation_rows, "loss_bytes": len(raw),
                 "loss_sha256": hashlib.sha256(raw).hexdigest()}
        save_torch(checkpoint_path, state)
        record = {"status": status, "route": route, "updates": updates, "total_updates": total,
                  "loss": last_loss, "cursor": cursor, "elapsed_seconds": elapsed,
                  "device": device, "engineering": engineering, "contract": contract,
                  "batch_size": batch_size, "microbatch_size": microbatch_size,
                  "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
                  "filtered_unordered_states": dataset.filtered,
                  "parameters": sum(p.numel() for p in model.parameters())}
        atomic_json(folder / "status.json", record)
        isolated_callback(progress_callback, {**record, "stage": "training"}, sampler)
        return record

    try:
        while updates < total:
            if stop_callback and isolated_callback(lambda _: stop_callback(), {}, sampler):
                status = "stopped"
                break
            samples, sampled_last = dataset.draw(sampler, batch_size)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            measured_loss = 0.
            # One optimizer update for the frozen full batch. Exact per-state
            # loss weighting permits bounded GPU activation memory.
            for offset in range(0, len(samples), microbatch_size):
                subset = samples[offset:offset + microbatch_size]
                batch = collate(subset, route, device)
                loss = supervised_loss(route, model(batch), batch, subset) * len(subset) / len(samples)
                if not torch.isfinite(loss):
                    raise ScientificStop("nonfinite supervised objective")
                loss.backward()
                measured_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.)))
            if not torch.isfinite(norm):
                raise ScientificStop("nonfinite supervised gradient")
            optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            updates, cursor, last_loss = updates + 1, cursor + len(samples), measured_loss
            with loss_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"update": updates, "loss": last_loss,
                                         "gradient_norm": float(norm), "cursor": cursor,
                                         "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                                         "sample_refs": sampled_last}, allow_nan=False) + "\n")
            if updates % 25 == 0:
                isolated_callback(progress_callback, {"stage": "training", "updates": updates,
                                  "total_updates": total, "loss": last_loss,
                                  "checkpoint_path": str(checkpoint_path), "cursor": cursor}, sampler)
            if updates % interval == 0 or updates == total:
                if validation_callback:
                    model.eval()
                    result = isolated_callback(validation_callback,
                                               {"model": model, "route": route, "updates": updates,
                                                "device": device, "contract": contract}, sampler)
                    validation_rows.append({"updates": updates, "metrics": result})
                    atomic_json(folder / "validation.json", validation_rows)
                    if isinstance(result, dict) and result.get("attention_required"):
                        status = "stopped"
                        record = checkpoint()
                        record.update(attention_required=True,
                                      stop_reason="engineering_attention: " + str(result.get("reason", "incomplete_validation_evidence")))
                        atomic_json(folder / "status.json", record)
                        return record
                    if isinstance(result, dict) and result.get("scientific_stop"):
                        status = "scientific_stop"
                        record = checkpoint()
                        return {**record, "stop_reason": result.get("stop_reason", "fixed_validation_gate_failed")}
                checkpoint()
        if status == "training":
            status = "completed"
        record = checkpoint()
        if status == "completed":
            endpoint = folder / "final" / "policy.pt"
            endpoint.parent.mkdir(parents=True, exist_ok=True)
            save_torch(endpoint, {"format": MODEL_FORMAT, "contract": contract,
                                  "network": config.get("network", {}), "model": model.state_dict(),
                                  "updates": updates, "selection": "fixed_budget_endpoint"})
            record.update(endpoint_path=str(endpoint), endpoint_sha256=digest(endpoint))
            atomic_json(endpoint.parent / "model.json", {"format": MODEL_FORMAT, "route": route,
                        "schema_version": contract["schema_version"], "contract": contract,
                        "provenance": provenance, "updates": updates, "selection": "fixed_budget_endpoint",
                        "policy_sha256": record["endpoint_sha256"]})
            atomic_json(folder / "status.json", record)
        return record
    except Exception as error:
        status = "scientific_stop" if isinstance(error, ScientificStop) else "failed"
        record = checkpoint()
        atomic_json(folder / "failure.json", {"type": type(error).__name__, "reason": str(error),
                                               "updates": updates, "checkpoint_sha256": record["checkpoint_sha256"]})
        if isinstance(error, ScientificStop):
            record["stop_reason"] = str(error)
            atomic_json(folder / "status.json", record)
            return record
        raise


class LearningPolicy:
    def __init__(self, route, model, contract=None, device="cpu", mode=None):
        self.route, self.model, self.contract = canonical_route(route), model, contract or {}
        self.device, self.mode = device, mode
        self.model.to(device).eval()

    def predict(self, sample):
        batch = collate([sample], self.route, self.device)
        with torch.no_grad():
            output = self.model(batch)
        return {key: value[0].cpu().numpy() for key, value in output.items()}


def load_policy(path, expected_provenance=None, device="cpu", mode=None):
    state = torch_load(path, device)
    if state.get("format") != MODEL_FORMAT or state.get("selection") != "fixed_budget_endpoint":
        raise ValueError("online policy must be a supervised fixed endpoint")
    contract = state["contract"]
    if expected_provenance:
        for key, expected in expected_provenance.items():
            if contract.get(key, contract.get("provenance", {}).get(key)) != expected:
                raise ValueError(f"policy frozen provenance differs: {key}")
    route = canonical_route(contract["route"])
    if not 0 < state["updates"] <= UPDATE_CAPS[route]:
        raise ValueError("policy update provenance differs")
    model = make_model(route, state["network"])
    model.load_state_dict(state["model"], strict=True)
    return LearningPolicy(route, model, contract, device, mode)


def retained_candidates(logits, legal, original, maximum=9):
    """Equal cardinality per aircraft, stable geometry tie order supplied by pool."""
    logits, legal, original = np.asarray(logits), np.asarray(legal, bool), np.asarray(original, int)
    retained = np.zeros_like(legal)
    for i in range(len(original)):
        if not legal[i, original[i]]:
            raise ValueError("original assignment absent from candidate pool")
        ids = np.flatnonzero(legal[i])
        order = ids[np.lexsort((ids, -logits[i, ids]))]
        selected = [int(original[i])] + [int(j) for j in order if j != original[i]][:maximum - 1]
        retained[i, selected] = True
    return retained


def audit_ranker(policy, frozen_complete_labels, topk=8):
    """Reuse one frozen labelled pool; performs NO new trial/physical calls.

    Unknown pools are reported as incomplete engineering evidence. A state
    without qualified improvement has undefined recall and is listed separately.
    The rule comparator is the same pretrial squared-load proxy as online H.
    """
    if policy.route != "04" or not 1 <= topk <= 64:
        raise ValueError("rank audit requires route 04 and bounded K")
    sample = frozen_complete_labels
    validate_sample(sample, "04")
    known = np.asarray(sample["known"], dtype=bool)
    utility = np.asarray(sample["utility"], dtype=float)
    if not known.all():
        return {"complete": False, "reason": "frozen_candidate_pool_contains_unknown_trials",
                "known_candidates": int(known.sum()), "candidates": len(known), "recall": None}
    scores = policy.predict(sample)["scores"][:len(known)]
    objects = sample.get("objects")
    if not objects or len(objects) != len(known):
        raise ValueError("rank audit lacks original candidate identities")
    proxy = -np.asarray([obj["features"][4] for obj in objects])
    order = np.argsort(-scores, kind="stable")[:topk]
    rule_order = np.argsort(-proxy, kind="stable")[:topk]
    positive = utility > 0
    denominator = int(positive.sum())
    return {"complete": True, "candidates": len(known), "qualified_improvements": denominator,
            "no_improvement_state": denominator == 0,
            "model_topk_recall": float(positive[order].sum() / denominator) if denominator else None,
            "rule_topk_recall": float(positive[rule_order].sum() / denominator) if denominator else None,
            "model_completed_best_improvement": float(max(0., utility[order].max(initial=0.))),
            "rule_completed_best_improvement": float(max(0., utility[rule_order].max(initial=0.))),
            "pool_completed_best_improvement": float(max(0., utility.max(initial=0.))),
            "model_selected": order.tolist(), "rule_selected": rule_order.tolist(),
            "label_pool_sha256": canonical_hash(sample), "topk": topk}


def run_learning(problem, route, policy, method="main"):
    """Execute model proposals solely through common charged validation hooks."""
    route = canonical_route(str(route).split("_")[0])
    options = LEARNING_METHODS
    main = {"01": "MO", "02": "BC", "03": "DR", "04": "L", "05": "L"}
    method = main[route] if method == "main" else method
    if method not in options[route]:
        raise ValueError(f"unsupported route {route} method {method}")
    learned = method in {"M", "MO", "BC", "D", "DR", "L"}
    if learned and (policy is None or policy.route != route):
        raise ValueError("learning route requires its own strict trained endpoint")
    from .core import BudgetStop, clone
    from .classical import direct_rule_layout, region_pool, repair_layout, run_lns
    random_order = np.random.default_rng(problem.config.seed)
    try:
        if route == "03":
            sample = {"aircraft": scene_features(problem.scene).tolist(),
                      "positions": problem.scene.positions.tolist(), "target_n": problem.n}
            start = time.perf_counter()
            if learned:
                output = policy.predict(sample)
                problem.times["inference_seconds"] = time.perf_counter() - start
                raw = clone(problem.initial)
                raw.centers = output["centers"][:problem.n].astype(np.float64)
                raw.assignment = output["logits"][:len(problem.demands), :problem.n].argmax(-1)
            else:
                raw = direct_rule_layout(problem)
            trial = problem.consider(raw, "03_" + method + "_raw")
            if trial["accepted"]:
                problem.commit(trial)
            elif raw is not None and problem.available() and method in ("DR", "HR"):
                repaired = repair_layout(problem, raw)
                raw_saved = clone(raw)
                problem.commit(problem.consider(repaired, "03_" + method + "_repaired"))
                problem.last_raw = raw_saved
            return
        if route == "01":
            # Shared 2..6-beam region generator; region-external labels remain locked.
            for region in problem.regions()[:problem.config.max_rounds]:
                if not problem.available():
                    break
                before = clone(problem.current)
                local = np.isin(before.assignment, region["beams"])
                pool = region_pool(problem, region)
                sample = mlm_sample(problem, before, np.random.default_rng(51001), 1., before, region, pool)
                ids = np.asarray(sample["members"], dtype=np.int64)
                visible = np.where(local, -1, before.assignment)
                start = time.perf_counter()
                if learned:
                    output = policy.predict(sample)
                else:
                    rays = np.asarray(sample["candidates"])[:, 2:5]
                    aircraft_rays = np.asarray(sample["aircraft"])[:, 2:5]
                    output = {"logits": -np.sum((aircraft_rays[:, None] - rays[None]) ** 2, -1)}
                problem.times["inference_seconds"] = problem.times.get("inference_seconds", 0.) + time.perf_counter() - start
                if method in ("MO", "O", "HO"):
                    keep = (np.asarray(sample["legal"], dtype=bool) if method == "O" else
                            retained_candidates(output["logits"], sample["legal"], sample["original_assignment"]))
                    proposal = problem.rebuild(region, candidate_pool=pool, visible=visible, edge_mask=keep)
                else:
                    proposal = clone(before)
                    logits = output["logits"].copy()
                    logits[~np.asarray(sample["legal"], dtype=bool)] = -1e9
                    chosen = np.full(len(ids), -1, dtype=np.int64)
                    decode_history = []
                    # Freeze confidence order within a fill batch, then expose
                    # only the filled assignments and recompute visible loads.
                    for decode_round in range(8):
                        pending = np.flatnonzero(chosen < 0)
                        if not len(pending) or not problem.available():
                            break
                        if decode_round:
                            sample["visible"] = chosen.tolist()
                            sample["hidden"] = (chosen < 0).tolist()
                            features = np.asarray(sample["candidates"], dtype=float)
                            filled = chosen >= 0
                            demand = np.asarray(sample["aircraft"])[:, 5]
                            features[:, 5] = np.bincount(chosen[filled], weights=demand[filled], minlength=len(pool))
                            features[:, 6] = np.bincount(chosen[filled], minlength=len(pool)) / len(ids)
                            sample["candidates"] = features.tolist()
                            if learned:
                                start = time.perf_counter()
                                logits = policy.predict(sample)["logits"]
                                problem.times["inference_seconds"] += time.perf_counter() - start
                                logits[~np.asarray(sample["legal"], dtype=bool)] = -1e9
                        shifted = logits[pending] - logits[pending].max(-1, keepdims=True)
                        probabilities = np.exp(shifted)
                        probabilities /= probabilities.sum(-1, keepdims=True)
                        confidence = probabilities.max(-1)
                        positions = problem.positions[ids[pending]]
                        order = np.lexsort((positions[:, 1], positions[:, 0], -confidence))
                        fill = pending[order[:max(1, int(np.ceil(len(pending) / (8 - decode_round))))]]
                        chosen[fill] = logits[fill].argmax(-1)
                        decode_history.append({"round": decode_round, "filled_local_aircraft": fill.tolist(),
                                               "candidate_assignment": chosen[fill].tolist()})
                    selected = sorted(set(chosen.tolist()))
                    fixed_n = -1 not in selected and len(selected) == len(region["beams"])
                    if fixed_n:
                        proposal.centers[list(region["beams"])] = pool[selected]
                        proposal.assignment[ids] = np.asarray(region["beams"])[np.searchsorted(selected, chosen)]
                    else:
                        proposal = None
                    problem.trace.append({"source": "01_prediction", "method": method,
                                          "pool_hash": sample["pool_hash"], "raw_candidate_assignment": chosen.tolist(),
                                          "raw_scores": logits.tolist(), "raw_fixed_N": fixed_n,
                                          "decode_history": decode_history})
                if proposal is not None:
                    problem.commit(problem.consider(proposal, "01_" + method))
            return
        if route == "05" and method in ("rule", "random"):
            run_lns(problem, selector=method)
            return
        failures, tried_regions = 0, set()
        for _ in range(problem.config.max_rebuilds if route == "05" else problem.config.max_rounds):
            if not problem.available():
                break
            objects = problem.regions() if route == "05" else problem.menu()
            if route == "05":
                objects = [obj for obj in objects if (tuple(obj["beams"]), tuple(problem.current.assignment)) not in tried_regions]
            if not objects:
                break
            sample = state_sample(problem, objects)
            start = time.perf_counter()
            if learned:
                prediction = policy.predict(sample)
            else:
                if route == "05" and method == "worst":
                    scores = np.asarray([obj["features"][4] - obj["features"][5] for obj in objects])
                else:
                    scores = (-np.asarray([obj["features"][4] for obj in objects]) if method != "R"
                              else random_order.random(len(objects)))
                prediction = {"scores": scores, "logits": np.r_[scores, -np.inf]}
            problem.times["inference_seconds"] = problem.times.get("inference_seconds", 0.) + time.perf_counter() - start
            scores = prediction["logits"] if route == "02" else prediction["scores"]
            order = np.argsort(-scores[:len(objects)], kind="stable")
            if route == "02" and learned and scores[-1] >= scores[order[0]]:
                problem.trace.append({"source": "02_stop", "reason": "model_stop", "scores": scores.tolist()})
                break
            count = (len(objects) if method in ("T", "F") else
                     1 if route == "05" else 4 if route == "02" else 8)
            completed = []
            for index in order[:count]:
                if not problem.available():
                    break
                obj = objects[int(index)]
                if route == "05":
                    tried_regions.add((tuple(obj["beams"]), tuple(problem.current.assignment)))
                    proposal = problem.rebuild(obj)
                    trial = problem.consider(proposal, "05_region")
                else:
                    trial = problem.preview(obj)
                problem.trace.append({"source": "model_score", "route": route,
                                      "object": obj, "score": float(scores[index])})
                if trial.get("accepted"):
                    completed.append(trial)
                    if route == "02":
                        break
            if completed:
                best = min(completed, key=lambda trial: trial["metrics_all"]["load_cv_all"])
                problem.commit(best)
                failures = 0
            else:
                failures += 1
                if route != "05" or failures >= 2:
                    break
    except BudgetStop:
        # Common executor has recorded the exact exhausted budget.
        return
