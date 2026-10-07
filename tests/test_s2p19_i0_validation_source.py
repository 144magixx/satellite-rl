"""I0 pilot saved-source contract; fixture values are not model performance results."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from satellite_rl.envs.regions import profile_metadata
from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.power_refinement.config import (
    VARIABLE_BEAM_METHODS, default_config as refinement_defaults)
from satellite_rl.sac_improvement.config import default_config as i0_defaults
from satellite_rl.workbench.power_refinement import PowerRefinementApplication
from satellite_rl.workbench.refinement_sources import prepare_source, source_rows
from satellite_rl.workbench.storage import digest


COUNTS = (50, 80, 100, 120, 160, 200)
LOADS = (0.75, 1.25)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _fixture(root, step=98765):
    """Mirror runner's JSON layout with zero-service structural fixtures."""
    jobdir = root / "outputs" / "workbench" / "jobs" / "i0_train"
    scene_dir = root / "scenes"
    scene_dir.mkdir()
    checkpoint = jobdir / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"bounded-checkpoint-fixture")
    checkpoint_sha = digest(checkpoint)
    profile = profile_metadata("europe_v1")
    scenes, summary_cases = [], []
    for mother in range(6):
        for count in COUNTS:
            np.random.seed(51001 + mother * 1000 + count)
            beam = BeamInfo(profile="europe_v1", count=count)
            arrays = {key: np.asarray(value).copy() for key, value in beam.beam_info.items()}
            arrays["original_indices"] = np.asarray(beam.shuffle_indices).copy()
            for load in LOADS:
                index = len(scenes)
                path = scene_dir / f"scene_{index:05d}.npz"
                np.savez_compressed(path, **arrays, fixture_index=np.asarray([index]))
                scene_sha = digest(path)
                metadata = {"path": str(path), "sha256": scene_sha,
                            "beam_count": count, "mother_scene_id": f"mother-{mother}",
                            "source_sha256": f"{mother + 1:064x}",
                            "selected_time_utc": f"2024-01-12T{mother:02d}:00:00+00:00",
                            "load_multiplier": load, "partition_role": "validation"}
                scenes.append(metadata)
                # The source contract requires complete beam rows; these zero-service
                # fixtures exercise provenance and API limits without claiming an I0 result.
                beams = [{"beam_id": beam_id, "group": beam_id % 8,
                          "freq": 0, "slots": 0, "power_w": 0.0,
                          "required_bps": 1e6, "rate_bps": 0.0}
                         for beam_id in range(count)]
                case = {"beams": beams,
                        "metrics": {"si": 0.0, "sgm": 0.0, "initial_si": 0.0,
                                    "initial_sgm": 0.0, "power_w": 0.0,
                                    "total_seconds": 0.01,
                                    "initialization_seconds": 0.005,
                                    "candidate_generation_seconds": 0.0,
                                    "probe_seconds": 0.0,
                                    "decisions": 0, "probes": 0, "invalid": 0},
                        "trace": [], "scene_sha256": scene_sha,
                        "scene_metadata": metadata, "seed": 53001,
                        "action_seed": 53001, "method": "I0_validation",
                        "repeat_index": 0, "env_step": step,
                        "checkpoint_sha256": checkpoint_sha}
                case_path = jobdir / "validation" / f"step_{step:09d}" / f"scene_{index:05d}.json"
                _write(case_path, case)
                summary_cases.append({"path": str(case_path), "sha256": digest(case_path)})
    record = {"id": "validation_72", "name": "validation_72",
              "path": str(scene_dir), "count": 72, "files": scenes,
              "profile": "europe_v1", "environment_profile": profile,
              "environment_version": profile["environment_version"],
              "source": {"partition_role": "validation"}}
    controls = i0_defaults()
    _write(jobdir / "job.json", {"run_id": "i0_train", "name": "I0 bounded pilot",
                                  "kind": "train", "algorithm": "sac", "mode": "train",
                                  "research_stage": "sac_improvement",
                                  "experiment_id": "S2-P19", "i0_config": controls,
                                  "action_seed": 53001, "repeats": 1,
                                  "scene_sets": {"validation": record}})
    _write(jobdir / "config.json", {"research_stage": "sac_improvement", "i0": controls})
    _write(jobdir / "status.json", {"stage": "stopped", "env_step": step,
                                     "checkpoint_step": step, "episodes": 3125,
                                     "checkpoint_sha256": checkpoint_sha,
                                     "stop_reason": "s2p19_pilot_gate_pending"})
    summary_path = jobdir / "validation" / f"step_{step:09d}" / "summary.json"
    _write(summary_path, {"env_step": step, "checkpoint_sha256": checkpoint_sha,
                          "source_set_id": record["id"], "count": 72,
                          "mean_si": 0.0, "mean_sgm": 0.0,
                          "mean_si_delta_vs_b1": 0.0, "cases": summary_cases})
    return jobdir, summary_path, record


def _complete_fixture(jobdir, *, due="episode_cap", episodes=12500):
    """Mirror observed I0 terminal status: no checkpoint_step, fixed final pair."""
    job_path, status_path = jobdir / "job.json", jobdir / "status.json"
    job, status = (json.loads(path.read_text(encoding="utf-8"))
                   for path in (job_path, status_path))
    job.update(seed=51001, total_steps=400000, model_dir=str(jobdir / "models"))
    _write(job_path, job)
    final = Path(job["model_dir"]) / "final"
    final.mkdir(parents=True)
    model_path = final / "policy.pt"
    model_path.write_bytes(b"fixed-terminal-policy-fixture")
    policy_sha = digest(model_path)
    status.update(run_id=job["run_id"], stage="completed", episodes=episodes,
                  updates=395113, completed_due_to=due, model_path=str(model_path),
                  model_sha256=policy_sha)
    status.pop("checkpoint_step", None)
    status.pop("stop_reason", None)
    _write(status_path, status)
    _write(final / "model.json", {"run_id": job["run_id"],
        "env_step": status["env_step"], "episodes": episodes,
        "updates": status["updates"], "config_sha256": digest(jobdir / "config.json"),
        "policy_sha256": policy_sha,
        "provenance": {"run_id": job["run_id"], "env_step": status["env_step"]}})
    return status, final / "model.json"


class I0PilotSavedSourceTests(unittest.TestCase):
    def test_completed_12500_episode_terminal_uses_saved_72_cases_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobdir, _, record = _fixture(Path(tmp), step=397112)
            status, model_meta = _complete_fixture(jobdir)

            class App:
                store = SimpleNamespace(create_job=lambda config, **kwargs:
                                        {"config": config, **kwargs})

                def resolve_run(self, run_id):
                    self_run = "i0_train"
                    if run_id != self_run:
                        raise FileNotFoundError(run_id)
                    return jobdir, False

                def scene_dates(self, _record):
                    return {"2024-01-12"}

            body = {"source_run_id": "i0_train", "source_group": "model",
                    "partition": "validation", "validation_step": 397112}
            _, _, _, reference = prepare_source(App(), body)
            self.assertEqual(reference["source_stage"], "final")
            self.assertEqual(reference["source_checkpoint_sha256"],
                             status["checkpoint_sha256"])
            self.assertEqual(reference["source_model_sha256"], status["model_sha256"])
            self.assertEqual(len(source_rows(reference)), 72)
            controls = refinement_defaults()
            controls.update(methods=list(VARIABLE_BEAM_METHODS), repeats=1,
                            action_mode="stochastic", action_seed=53001)
            submitted = PowerRefinementApplication(App()).submit_saved({
                **body, "request_id": "i0-endpoint-refine-fixture", "config": controls})
            self.assertEqual(submitted["extra"]["scene_sets"]["validation"]["id"], record["id"])
            self.assertEqual(submitted["config"]["training"]["total_steps"], 216)
            with self.assertRaises(ValueError):
                prepare_source(App(), {**body, "validation_step": 200000})
            saved = json.loads(model_meta.read_text(encoding="utf-8"))
            saved["env_step"] = 200000
            _write(model_meta, saved)
            with self.assertRaises(ValueError):
                prepare_source(App(), body)

    def test_completed_exact_400k_and_false_terminal_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobdir, _, _ = _fixture(Path(tmp), step=400000)
            _complete_fixture(jobdir, due="env_steps", episodes=12000)

            class App:
                def resolve_run(self, _run_id):
                    return jobdir, False

                def scene_dates(self, _record):
                    return {"2024-01-12"}

            body = {"source_run_id": "i0_train", "source_group": "model",
                    "partition": "validation", "validation_step": 400000}
            self.assertEqual(prepare_source(App(), body)[3]["source_stage"], "final")
            status_path = jobdir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            for key, value in (("completed_due_to", "episode_cap"),
                               ("episodes", 12500),
                               ("checkpoint_sha256", "0" * 64)):
                altered = dict(status)
                altered[key] = value
                _write(status_path, altered)
                with self.assertRaises(ValueError, msg=key):
                    prepare_source(App(), body)
            _write(status_path, status)

    def test_bounded_pilot_72_cases_reach_saved_refinement_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobdir, summary_path, record = _fixture(Path(tmp))

            class App:
                store = SimpleNamespace(create_job=lambda config, **kwargs:
                                        {"config": config, **kwargs})

                def resolve_run(self, run_id):
                    if run_id != "i0_train":
                        raise FileNotFoundError(run_id)
                    return jobdir, False

                def scene_dates(self, _record):
                    return {"2024-01-12"}

            app = App()
            body = {"source_run_id": "i0_train", "source_group": "model",
                    "partition": "validation", "validation_step": 98765}
            _, _, _, reference = prepare_source(app, body)
            self.assertEqual(reference["source_kind"], "i0_train_validation")
            self.assertEqual(reference["source_checkpoint_sha256"],
                             digest(jobdir / "checkpoint.pt"))
            self.assertEqual(len(source_rows(reference)), 72)
            controls = refinement_defaults()
            controls.update(methods=list(VARIABLE_BEAM_METHODS), repeats=1,
                            action_mode="stochastic", action_seed=53001)
            submitted = PowerRefinementApplication(app).submit_saved(
                {**body, "request_id": "i0-pilot-refine-fixture", "config": controls})
            self.assertEqual(submitted["extra"]["scene_sets"]["validation"]["id"], record["id"])
            self.assertEqual(submitted["extra"]["model_files"], [])
            self.assertEqual(submitted["config"]["training"]["total_steps"], 216)
            with self.assertRaises(ValueError):
                prepare_source(app, {**body, "validation_step": 100000})
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            original_summary = json.loads(json.dumps(summary))
            summary["cases"][0]["sha256"] = "0" * 64
            _write(summary_path, summary)
            with self.assertRaises(ValueError):
                source_rows(reference)
            case_path = Path(original_summary["cases"][0]["path"])
            case = json.loads(case_path.read_text(encoding="utf-8"))
            case["action_seed"] = 53002
            _write(case_path, case)
            original_summary["cases"][0]["sha256"] = digest(case_path)
            _write(summary_path, original_summary)
            with self.assertRaises(ValueError):
                prepare_source(app, body)

    def test_incomplete_grid_and_wrong_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobdir, _, record = _fixture(Path(tmp))

            class App:
                def resolve_run(self, _run_id):
                    return jobdir, False

                def scene_dates(self, _record):
                    return {"2024-01-12"}

            app = App()
            body = {"source_run_id": "i0_train", "source_group": "model",
                    "partition": "validation", "validation_step": 98765}
            status_path = jobdir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["checkpoint_sha256"] = "0" * 64
            _write(status_path, status)
            with self.assertRaises(ValueError):
                prepare_source(app, body)
            status["checkpoint_sha256"] = digest(jobdir / "checkpoint.pt")
            _write(status_path, status)
            job_path = jobdir / "job.json"
            job = json.loads(job_path.read_text(encoding="utf-8"))
            job["scene_sets"]["validation"]["files"][1]["load_multiplier"] = 0.75
            _write(job_path, job)
            with self.assertRaises(ValueError):
                prepare_source(app, body)


if __name__ == "__main__":
    unittest.main()
