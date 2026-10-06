"""Durable, immutable per-allocation evidence for the approved variable-beam assessment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from satellite_rl.training.run_monitor import replace_with_retry

from .experiments import is_variable_beam_experiment


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha256(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class VariableEvaluationCases:
    """A case is committed by one atomic file containing its own content hash.

    Complete files are immutable. A missing case may be run; a corrupt or
    contract-mismatched case blocks resume rather than replacing evidence.
    """

    def __init__(self, directory, contract):
        experiment_id = contract.get("experiment_id")
        if not is_variable_beam_experiment(experiment_id):
            raise ValueError(f"Variable case store requires a variable-beam experiment, got {experiment_id!r}")
        self.experiment_id = experiment_id
        self.directory = Path(directory) / "cases"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.contract_sha256 = _sha256(contract)
        path = self.directory / "contract.json"
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"{experiment_id} case contract is corrupt") from error
            if saved != {"contract": contract, "sha256": self.contract_sha256}:
                raise ValueError(f"{experiment_id} evaluation inputs differ from frozen case contract")
        else:
            if any(self.directory.glob("*.json")):
                raise ValueError(f"{experiment_id} case files exist without a frozen contract")
            _atomic_json(path, {"contract": contract, "sha256": self.contract_sha256})

    def path(self, case_id):
        if not re.fullmatch(r"(?:model|fixed|greedy)_\d{5}_\d{2}", case_id):
            raise ValueError(f"Invalid {self.experiment_id} case identity")
        return self.directory / f"{case_id}.json"

    def load(self, case_id, expected):
        path = self.path(case_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{self.experiment_id} case is corrupt: {case_id}") from error
        body = envelope.get("body") if isinstance(envelope, dict) else None
        if (not isinstance(body, dict) or envelope.get("sha256") != _sha256(body) or
                body.get("case_id") != case_id or body.get("contract_sha256") != self.contract_sha256 or
                body.get("expected") != expected or not isinstance(body.get("result"), dict)):
            raise ValueError(f"{self.experiment_id} case checksum or frozen input mismatch: {case_id}")
        result = body["result"]
        scene = result.get("scene")
        if (result.get("case_id", case_id) != case_id or
                result.get("action_seed") != expected.get("action_seed") or
                result.get("repeat_index") != expected.get("repeat_index") or
                not isinstance(scene, dict) or
                scene.get("source_sha256") != expected.get("scene_sha256") or
                result.get("beam_count") != expected.get("beam_count") or
                ("scene_metadata" in expected and
                 result.get("scene_metadata") != expected["scene_metadata"])):
            raise ValueError(f"{self.experiment_id} case result identity mismatch: {case_id}")
        return result

    def save(self, case_id, expected, result):
        if self.load(case_id, expected) is not None:
            raise ValueError(f"{self.experiment_id} completed case cannot be overwritten: {case_id}")
        body = {"case_id": case_id, "contract_sha256": self.contract_sha256,
                "expected": expected, "result": result}
        _atomic_json(self.path(case_id), {"body": body, "sha256": _sha256(body)})

    def require_all(self, cases):
        """Reject unknown files and yield results in the frozen planned order."""
        planned = {case_id for case_id, _ in cases}
        found = {path.stem for path in self.directory.glob("*.json") if path.name != "contract.json"}
        if found != planned:
            raise ValueError(f"{self.experiment_id} case inventory differs from approved evaluation plan")
        for case_id, expected in cases:
            result = self.load(case_id, expected)
            if result is None:
                raise ValueError(f"{self.experiment_id} planned case is missing: {case_id}")
            yield case_id, result
