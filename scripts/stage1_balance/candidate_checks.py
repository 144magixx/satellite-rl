"""Bounded remote DQN checks before any S1-02 formal workbench submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen


DQN = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python")
STAGED_VALIDATION = Path("/home/u2024110224/first_stage_satellite_rl/s1_02/20418/data/source_validation")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def api(port: int, endpoint: str, body=None):
    url = f"http://127.0.0.1:{port}/api/" + endpoint.lstrip("/")
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data,
                      headers={"Content-Type": "application/json"} if data is not None else {})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def run_logged(args: list[str], cwd: Path, env: dict, log: Path, timeout: int) -> None:
    with log.open("wb") as stream:
        result = subprocess.run(args, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Short verification failed ({result.returncode}): {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a2-start", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8893)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "status.json"
    status = {"status": "running", "started_utc": now(),
              "source_commit": args.commit, "repo": str(repo),
              "verification_root": str(output)}
    save(status_file, status)
    server = None
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain",
                                         "--", "src", "data", "pyproject.toml"], text=True).strip()
        if commit != args.commit or dirty:
            raise RuntimeError("Verification source differs from frozen science commit")
        env = dict(os.environ, PYTHONPATH=str(repo / "src"),
                   SATELLITE_RL_OUTPUT_ROOT=str(output / "isolated_outputs"),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        runtime = subprocess.check_output([str(DQN), "-B", "-c",
            "import json,sys,torch,satellite_rl;print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda_build':torch.version.cuda,'available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'package':satellite_rl.__file__}))"],
            cwd=repo, env=env, text=True, timeout=45)
        runtime = json.loads(runtime)
        if (runtime["cuda_build"] != "12.1" or not runtime["available"] or
                "4090 D" not in runtime["gpu"] or
                Path(runtime["package"]).resolve() != repo / "src/satellite_rl/__init__.py"):
            raise RuntimeError("Remote DQN/CUDA or actual package path differs")
        status.update(status="full_unittest", runtime=runtime)
        save(status_file, status)
        run_logged([str(DQN), "-B", "-m", "unittest", "discover", "-s", "tests",
                    "-v"], repo, env,
                   output / "full_unittest.log", 1800)
        test_text = (output / "full_unittest.log").read_text(encoding="utf-8", errors="replace")
        count = re.search(r"Ran (\d+) tests? in", test_text)
        if count is None or int(count.group(1)) < 450:
            raise RuntimeError("Full remote unittest run did not include the expected project suite")
        status["full_unittest"] = {"status": "passed", "tests_run": int(count.group(1)),
                                   "log": str(output / "full_unittest.log")}
        sys.path.insert(0, str(repo / "scripts"))
        from report_stage1_balance import _matplotlib
        from matplotlib.ft2font import FT2Font

        font = Path("/home/u2024110224/second_stage_satellite_rl/fonts/NotoSansSC-VF.ttf")
        if not font.is_file():
            raise RuntimeError("Existing shared Noto CJK font is missing")
        font_sha = hashlib.sha256(font.read_bytes()).hexdigest()
        if font_sha != "763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a":
            raise RuntimeError("Existing CJK font SHA differs from preflight")
        cmap = FT2Font(str(font)).get_charmap()
        if not all(ord(character) in cmap for character in "波束覆盖需求"):
            raise RuntimeError("CJK font lacks required plot glyphs")
        plt, font_metadata = _matplotlib()
        figure, axis = plt.subplots(figsize=(4, 2))
        axis.set_title("波束覆盖与业务需求")
        axis.plot([0, 1], [0, 1])
        figure.savefig(output / "cjk_font_sample.png", dpi=120)
        plt.close(figure)
        status["cjk_font"] = {"path": str(font), "sha256": font_sha,
                              "sample": str(output / "cjk_font_sample.png")}
        status["status"] = "real_jan12_x4"
        save(status_file, status)
        run_logged([str(DQN), "-B", str(Path(__file__).with_name("verify_real_scene.py")),
                    "--repo", str(repo), "--source",
                    str(STAGED_VALIDATION / "source_observations_00000.jsonl.gz"),
                    "--start", str(args.a2_start),
                    "--output", str(output / "real_scene_x4.json")],
                   repo, env, output / "real_scene_x4.log", 600)
        status["real_scene_x4"] = json.loads((output / "real_scene_x4.json").read_text())
        status["status"] = "isolated_workbench_ui_and_api"
        save(status_file, status)
        workbench_archive = output / "workbench"
        with (output / "workbench.log").open("ab") as log:
            server = subprocess.Popen([str(DQN), "-B", "-m", "satellite_rl.workbench.server",
                                       "--archive-root", str(workbench_archive),
                                       "--port", str(args.port), "--python", str(DQN),
                                       "--no-scheduler"], cwd=repo, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        system = None
        for _ in range(30):
            if server.poll() is not None:
                raise RuntimeError("Isolated verification workbench exited")
            try:
                system = api(args.port, "system")
                break
            except Exception:
                time.sleep(1)
        if system is None or Path(system["source_root"]).resolve() != repo:
            raise RuntimeError("Isolated verification workbench did not serve frozen source")
        with urlopen(f"http://127.0.0.1:{args.port}/", timeout=10) as response:
            page = response.read().decode("utf-8")
        with urlopen(f"http://127.0.0.1:{args.port}/coverage.js", timeout=10) as response:
            page_script = response.read().decode("utf-8")
        if "coverage" not in page or "X4" not in page_script or "balance" not in page_script:
            raise RuntimeError("Served first-stage page does not expose S1-02 controls")
        catalog = api(args.port, "coverage/catalog")
        config = catalog["balance_defaults"]
        config["coverage"]["balance"].update(route="X4", action_menu="X4")
        config["coverage"]["method"] = "local_search"
        validated = api(args.port, "coverage/validate", config)
        if (validated["config"]["coverage"]["balance"]["action_menu"] != "X4" or
                validated["summary"]["action_dim"] != 37):
            raise RuntimeError("X4 page/API configuration was not preserved")
        prepare_config = catalog["balance_defaults"]
        prepare_config["coverage"]["balance"]["route"] = "prepare"
        prepare_config["coverage"]["method"] = "none"
        imported = api(args.port, "coverage/datasets/import", {
            "path": str(STAGED_VALIDATION / "source_observations_00000.jsonl.gz"),
            "partition_role": "validation", "coverage": prepare_config["coverage"],
            "name": "S1-02 isolated one-scene verification"})
        if imported.get("count") != 1 or imported.get("dates") != ["2024-01-12"]:
            raise RuntimeError("Verification-only scene import differs")
        job = api(args.port, "coverage/runs", {
            "kind": "evaluate", "config": prepare_config,
            "dataset_ids": {"validation": imported["id"]},
            "dataset_role": "validation", "experiment_role": "verification",
            "request_id": "S1_02_candidate_prepare_verification"})
        detail = api(args.port, "runs/" + job["run_id"])
        if (detail["status"].get("stage") != "queued" or
                detail["job"].get("git_commit") != args.commit or
                detail["config"]["coverage"]["balance"]["route"] != "prepare"):
            raise RuntimeError("No-scheduler verification task record/source differs")
        status.update(status="completed", completed_utc=now(),
                      workbench={"system": system, "page_controls": True,
                                 "x4_action_dim": 37, "verification_dataset_id": imported["id"],
                                 "verification_run_id": job["run_id"],
                                 "queued_without_scheduler": True,
                                 "archive_root": str(workbench_archive)})
    except BaseException as error:
        status.update(status="failed", failed_utc=now(),
                      error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        save(status_file, status)
    print(json.dumps({"status": status["status"],
                      "verification_run_id": status["workbench"]["verification_run_id"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
