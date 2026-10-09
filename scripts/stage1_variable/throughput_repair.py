"""Reviewable S1-V10 same-ID throughput migration; no action runs implicitly.

The CLI is deliberately split into bounded, receipt-writing steps. ``plan`` is
read-only. Stop/deploy/resume actions require the reviewed repair commit and
the two exact formal PPO IDs; they never create another scientific run.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.stage1_variable import ops
from scripts.stage1_variable.data_controller import stage_throughput_canonical_contract
from scripts.stage1_variable.deploy import serve_lane


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
from satellite_rl.workbench.variable_throughput_recovery import (
    CRITERIA, DATA_CONTRACT_SHA256, ORIGIN_COMMIT, REPAIR_ID,
    SOURCE_ALLOWLIST, TARGETS, TRAIN_SHA256, VALIDATION_SHA256)


OUT = ops.LOCAL / "ops" / "throughput_repair"
CONTROLLER_LAUNCH = ops.LOCAL / "preflight" / "controller_launch_b311.json"
CONTROLLER_STATE = ops.LOCAL / "controller" / "state.json"
ACTIVE = {"starting", "initializing", "training", "evaluating", "preparing"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)
    return path


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def api(port, endpoint, body=None):
    response = ops.api(port, endpoint, body)
    if response["status"] != 200:
        raise RuntimeError(f"{port} {endpoint} HTTP {response['status']}: {response.get('error')}")
    return response["body"]


def remote(port, code, timeout=120):
    response = ops.remote_python(code, port, timeout=timeout)
    if response["exit_code"]:
        raise RuntimeError(f"{port} remote: {response['stderr'][-900:]}")
    return json.loads(response["stdout"].splitlines()[-1])


def commit_diff(new_commit):
    if not isinstance(new_commit, str) or len(new_commit) != 40 or any(
            digit not in "0123456789abcdef" for digit in new_commit):
        raise ValueError("exact 40-character repair commit required")
    def git(*args):
        return subprocess.run(["git", "-C", str(REPOSITORY), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    if git("rev-parse", "HEAD") != new_commit or git("status", "--porcelain"):
        raise RuntimeError("reviewed repair must be a clean current checkout")
    if git("merge-base", ORIGIN_COMMIT, new_commit) != ORIGIN_COMMIT:
        raise RuntimeError("repair commit is not descended from original science")
    changed = set(git("diff", "--name-only", ORIGIN_COMMIT, new_commit).splitlines())
    control = {
        "scripts/stage1_variable/throughput_repair.py",
        "scripts/stage1_variable/controller.py",
        "scripts/stage1_variable/data_controller.py",
        "scripts/stage1_variable/report.py",
        "scripts/stage1_variable/archive_inventory.py",
        "tests/test_s1v10_throughput_recovery.py",
        "tests/test_stage1_variable_controller.py",
        "tests/test_stage1_variable_report.py",
        "tests/test_stage1_variable_archive.py",
        "tests/test_coverage_variable_learning.py",
        "tests/test_coverage_variable_design.py",
        "tests/test_coverage_variable_workbench.py",
    }
    if changed - (SOURCE_ALLOWLIST | control):
        raise RuntimeError(f"repair changed out-of-scope files: {sorted(changed - (SOURCE_ALLOWLIST | control))}")
    if not {"src/satellite_rl/coverage/variable_design.py",
            "src/satellite_rl/coverage/variable_runner.py",
            "src/satellite_rl/workbench/variable_throughput_recovery.py",
            "src/satellite_rl/workbench/runtime.py",
            "src/satellite_rl/workbench/coverage_variable.py"} <= changed:
        raise RuntimeError("repair lacks one of its required reviewed source files")
    return {"origin_commit": ORIGIN_COMMIT, "repair_commit": new_commit,
            "changed_files": sorted(changed)}


def job_detail(run_id):
    expected = TARGETS[run_id]
    port = int(expected["port"])
    detail = api(port, "/api/runs/" + run_id)
    job, config, status = (detail[key] for key in ("job", "config", "status"))
    if (job.get("run_id") != run_id or job.get("request_id") != expected["request_id"] or
            job.get("experiment_id") != "S1-V10" or job.get("kind") != "train" or
            job.get("variable_route") != expected["route"] or
            config.get("schema_version") != 5 or
            config.get("training", {}).get("seed") != 51001 or
            config.get("training", {}).get("total_steps") != 262144 or
            {role: row["sha256"] for role, row in config.get("datasets", {}).items()} !=
            {"train": TRAIN_SHA256, "validation": VALIDATION_SHA256}):
        raise RuntimeError("target PPO job/request/data/seed/budget identity differs")
    return {"port": port, "run_id": run_id, "job": job, "config": config,
            "status": status, "directory": detail["path"]}


def controller_process():
    import psutil
    record = json.loads(CONTROLLER_LAUNCH.read_text(encoding="utf-8-sig"))
    pid = int(record["pid"])
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None
    command = " ".join(process.cmdline())
    if (abs(process.create_time() - datetime.fromisoformat(
            record["creation_date"]).timestamp()) > 1 or
            "scripts.stage1_variable.controller" not in command or
            ORIGIN_COMMIT not in command):
        raise RuntimeError("local controller PID/birth/command differs")
    return process


def repaired_controller_process():
    import psutil
    path = OUT / "controller_after_repair_launch.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    try:
        process = psutil.Process(int(record["pid"]))
    except psutil.NoSuchProcess:
        return None
    if (abs(process.create_time() - record["birth"]) > .1 or
            "scripts.stage1_variable.controller" not in " ".join(process.cmdline()) or
            ORIGIN_COMMIT not in " ".join(process.cmdline())):
        raise RuntimeError("repaired local controller PID/birth/command differs")
    return process


def plan(new_commit=None):
    rows = {}
    for run_id in TARGETS:
        detail = job_detail(run_id)
        job, status = detail["job"], detail["status"]
        if job.get("git_commit") != ORIGIN_COMMIT:
            raise RuntimeError("one target job source has already changed")
        rows[run_id] = {"port": detail["port"], "request_id": job["request_id"],
                        "route": job["variable_route"], "stage": status.get("stage"),
                        "env_step": status.get("env_step"),
                        "checkpoint_sha256": status.get("checkpoint_sha256"),
                        "pid": status.get("pid"), "source_root": job["source_root"]}
    process = controller_process()
    result = {"observed_utc": now(), "repair_id": REPAIR_ID,
              "origin_commit": ORIGIN_COMMIT, "target_jobs": rows,
              "controller_pid": process.pid if process else None,
              "source_diff": commit_diff(new_commit) if new_commit else None,
              "action": "read_only_plan_no_worker_stop"}
    save("plan.json", result)
    return result


def pause_controller():
    process = controller_process()
    if process is None:
        raise RuntimeError("original local controller is absent; inspect ownership")
    state = json.loads(CONTROLLER_STATE.read_text(encoding="utf-8"))
    if (state.get("scientific_commit") != ORIGIN_COMMIT or
            any(state.get("throughput_source_overrides", {}))):
        raise RuntimeError("controller state has a different science source")
    before = save("controller_before_repair.json", state)
    process.terminate()
    process.wait(timeout=15)
    result = {"observed_utc": now(), "pid": process.pid,
              "controller_state_sha256": sha(before),
              "controller_state_path": str(before),
              "old_controller_stopped": True}
    save("controller_pause.json", result)
    return result


def hold(port):
    result = api(port, "/api/dispatch/hold", {"ttl_seconds": 600,
        "reason": "S1-V10 exact same-ID throughput checkpoint/source migration"})
    if not result.get("active") or not api(port, "/api/dispatch/hold").get("active"):
        raise RuntimeError("dispatch hold was not persisted")
    return result


def stop_training():
    if controller_process() is not None or not (OUT / "controller_pause.json").is_file():
        raise RuntimeError("pause the exact original local controller before stopping science workers")
    result = {"requested_utc": now(), "runs": {}}
    for run_id in TARGETS:
        detail = job_detail(run_id)
        port = detail["port"]
        if detail["job"].get("git_commit") != ORIGIN_COMMIT:
            raise RuntimeError("target source changed before stop")
        if detail["status"].get("stage") not in ("training", "evaluating"):
            raise RuntimeError(f"{run_id} not in formal training or its validation stage")
        held = hold(port)
        response = api(port, f"/api/runs/{run_id}/stop", {})
        result["runs"][run_id] = {"port": port, "request_id": detail["job"]["request_id"],
                                  "status_before": detail["status"], "hold": held,
                                  "stop_api": response}
        save("stop_requests.json", result)
    return result


def stopped_checkpoint(run_id):
    detail = job_detail(run_id)
    port = detail["port"]
    job, status = detail["job"], detail["status"]
    if (job.get("git_commit") != ORIGIN_COMMIT or
            status.get("stage") != "stopped" or
            status.get("stop_reason") not in ("requested", "validation_interrupted") or
            type(status.get("env_step")) is not int or
            status["env_step"] % 2048 or not 0 < status["env_step"] < 262144 or
            not api(port, "/api/dispatch/hold").get("active")):
        raise RuntimeError("target is not cleanly stopped at an original-source 2048 checkpoint")
    directory = detail["directory"]
    code = f'''
import hashlib,json,os,pathlib,subprocess
root=pathlib.Path({directory!r});status=json.loads((root/'status.json').read_text());launch=json.loads((root/'launch.json').read_text())
path=root/'checkpoint.pt';h=hashlib.sha256()
with path.open('rb') as stream:
 for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
pid=launch['pid'];p=pathlib.Path('/proc')/str(pid);alive=False
if p.exists():
 raw=(p/'stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19]);boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime '));start=boot+ticks/os.sysconf('SC_CLK_TCK')
 alive=abs(start-launch['process_start_time'])<.02
sessions=subprocess.run(['ps','-eo','sess,pid,args'],capture_output=True,text=True,timeout=10,check=True).stdout.splitlines()[1:]
orphan=[line for line in sessions if line.split(None,2) and line.split(None,2)[0].isdigit()
 and int(line.split(None,2)[0])==pid and int(line.split(None,2)[1])!=pid]
print(json.dumps({{'sha256':h.hexdigest(),'bytes':path.stat().st_size,
 'worker_alive_same_birth':alive,'launch_pid':pid,'launch_start':launch['process_start_time'],
 'step':status['env_step'],'updates':status['update_step'],'episode':status['episode'],
 'stop_reason':status.get('stop_reason'),
 'orphan_session_processes':orphan}}))
'''
    remote_state = remote(port, code)
    if (remote_state["worker_alive_same_birth"] or remote_state["orphan_session_processes"] or
            remote_state["sha256"] != status.get("checkpoint_sha256") or
            remote_state["step"] != status["env_step"]):
        raise RuntimeError("stopped worker PID/birth/checkpoint SHA differs")
    return {"port": port, "run_id": run_id, "request_id": job["request_id"],
            "source_commit": ORIGIN_COMMIT, "directory": directory,
            "observed_utc": now(), "checkpoint": remote_state}


def wait_stopped():
    if not (OUT / "stop_requests.json").is_file():
        raise RuntimeError("no exact stop request receipt")
    for port in (20418, 20424):
        hold(port)
    rows = {run_id: stopped_checkpoint(run_id) for run_id in TARGETS}
    save("stopped_checkpoints.json", rows)
    return rows


def _check_stopped_and_held():
    saved = load("stopped_checkpoints.json")
    if set(saved) != set(TARGETS):
        raise RuntimeError("both exact checkpoint stop receipts are required")
    for run_id in TARGETS:
        current = stopped_checkpoint(run_id)
        if (current["checkpoint"]["sha256"] != saved[run_id]["checkpoint"]["sha256"] or
                current["checkpoint"]["step"] != saved[run_id]["checkpoint"]["step"]):
            raise RuntimeError("stopped checkpoint changed while awaiting repair")
    return saved


def _service_stop_and_checkout(port, new_commit, remote_bundle, bundle_sha):
    info = ops.instance(port)
    target_id = next(run_id for run_id, row in TARGETS.items() if row["port"] == str(port))
    code = f'''
import datetime,json,os,pathlib,signal,socket,subprocess,time
root=pathlib.Path({info['root']!r});repo=root/'repo';record=json.loads((root/'ops/service.json').read_text())
if (record['source_commit']!={ORIGIN_COMMIT!r} or record['archive_root']!=str(root) or
    record['api_port']!={info['api_port']} or record['ssh_port']!={port} or
    record['max_workers']!=2):raise RuntimeError('old service ownership differs')
jobs=[]
for file in (root/'outputs/workbench/jobs').glob('*/job.json'):
 job=json.loads(file.read_text());status=json.loads((file.parent/'status.json').read_text())
 if job.get('experiment_id')=='S1-V10':jobs.append((job['run_id'],status['stage']))
if ({target_id!r},'stopped') not in jobs or any(stage in ('queued','starting','initializing','training','evaluating','preparing') for _,stage in jobs):
 raise RuntimeError('another S1-V10 task is active/queued or target not stopped: '+repr(jobs))
def git(*args):
 p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=True)
 return p.stdout.strip()
if git('rev-parse','HEAD')!={ORIGIN_COMMIT!r} or git('status','--porcelain'):
 raise RuntimeError('old workbench checkout differs or dirty')
p=pathlib.Path('/proc')/str(record['pid']);raw=(p/'stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
cmd=(p/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace')
if (ticks!=record['start_ticks'] or str(root) not in cmd or
    'satellite_rl.workbench.server' not in cmd or '--port {info['api_port']}' not in cmd):
 raise RuntimeError('old service PID/start/command differs')
os.kill(record['pid'],signal.SIGTERM)
for _ in range(100):
 time.sleep(.1)
 if not p.exists():break
 stat=(p/'stat').read_text();new_ticks=int(stat[stat.rfind(')')+2:].split()[19])
 if new_ticks!=ticks:break
else:raise RuntimeError('old owned service did not exit')
with socket.socket() as sock:
 sock.settimeout(2)
 if sock.connect_ex(('127.0.0.1',{info['api_port']}))==0:raise RuntimeError('reserved port still occupied')
bundle=pathlib.Path({remote_bundle!r})
h=__import__('hashlib').sha256(bundle.read_bytes()).hexdigest()
if h!={bundle_sha!r}:raise RuntimeError('reviewed source bundle SHA differs')
subprocess.run(['git','-C',str(repo),'fetch','--quiet',str(bundle),'HEAD'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--quiet','--detach',{new_commit!r}],check=True)
if git('rev-parse','HEAD')!={new_commit!r} or git('status','--porcelain'):
 raise RuntimeError('reviewed new workbench checkout differs')
value={{'port':{port},'target_run_id':{target_id!r},'old_commit':{ORIGIN_COMMIT!r},
 'new_commit':{new_commit!r},'old_service_pid':record['pid'],
 'old_service_start_ticks':ticks,'jobs_at_stop':jobs,'bundle_sha256':h,
 'stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
(root/'ops/throughput_service_migration.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''
    return remote(port, code, timeout=180)


def deploy_repair(new_commit):
    if controller_process() is not None or not (OUT / "controller_pause.json").is_file():
        raise RuntimeError("pause the original controller before changing workbench sources")
    for port in (20418, 20424):
        hold(port)
    stopped = _check_stopped_and_held()
    diff = commit_diff(new_commit)
    bundle = OUT / f"s1v10_throughput_{new_commit}.bundle"
    if not bundle.exists():
        subprocess.run(["git", "-C", str(REPOSITORY), "bundle", "create",
                        str(bundle), "HEAD"], check=True)
    subprocess.run(["git", "-C", str(REPOSITORY), "bundle", "verify", str(bundle)],
                   check=True, capture_output=True)
    bundle_sha = sha(bundle)
    result = {"repair_commit": new_commit, "origin_commit": ORIGIN_COMMIT,
              "source_diff": diff, "bundle_sha256": bundle_sha,
              "stopped_checkpoints": stopped, "lanes": {}}
    for port in (20418, 20424):
        hold(port)  # Renew a bounded hold after review of the stopped state.
        info = ops.instance(port)
        target = f"{info['ops']}/{bundle.name}"
        sent = ops.upload(bundle, target, port)
        if sent["remote"] != bundle_sha or sent["local_sha256"] != bundle_sha:
            raise RuntimeError(f"{port}: reviewed repair bundle SHA differs")
        migrated = _service_stop_and_checkout(port, new_commit, target, bundle_sha)
        started = serve_lane(port, new_commit, 2)
        if (started["source_commit"] != new_commit or started["max_workers"] != 2 or
                not api(port, "/api/dispatch/hold").get("active")):
            raise RuntimeError(f"{port}: repaired service source/hold differs")
        code = f'''
import json,os,pathlib,sys
repo=pathlib.Path({info['repo']!r});sys.path.insert(0,str(repo/'src'))
from satellite_rl.workbench.storage import Store,digest
store=Store({info['root']!r},source_root=str(repo))
snapshot,commit=store.freeze_source()
print(json.dumps({{'source_root':str(snapshot),'commit':commit,
                  'source_manifest_sha256':digest(snapshot/'source_manifest.json')}}))
'''
        frozen = remote(port, code, timeout=180)
        if frozen["commit"] != new_commit:
            raise RuntimeError(f"{port}: repaired frozen source commit differs")
        result["lanes"][str(port)] = {"migration": migrated,
                                     "new_service": started,
                                     "new_frozen_source": frozen}
        save("deployment.json", result)
    result["throughput_dataset_contract"] = stage_throughput_canonical_contract(new_commit)
    save("deployment.json", result)
    return result


EVIDENCE = {
    "constructor_1728.json": "preflight/construct_full/full_equivalence_summary.json",
    "pool_packet_state_rng.json": "preflight/throughput_pool_summary.json",
    "trajectory_128.json": "preflight/runner_equivalence_128/summary.json",
    "spawn_short_runner.json": "preflight/processpool_short_runner/verification.json",
    "same_gpu_update.json": "preflight/gpu_one_rollout_audit_final.json",
    "frozen_checkpoint_loader.json": "preflight/frozen_checkpoint_dev_loader_compatibility.json",
    "throughput_diagnosis.json": "preflight/throughput_diagnosis.json",
}


def _static_equivalence(route):
    paths = {name: ops.LOCAL / relative for name, relative in EVIDENCE.items()}
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("a required raw equivalence evidence file is missing")
    rows = {name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()}
    constructor = rows["constructor_1728.json"]
    pool = rows["pool_packet_state_rng.json"]["routes"][route]["comparisons"]
    trajectory = rows["trajectory_128.json"]
    gpu = rows["same_gpu_update.json"]["routes"][route]
    runner_blob = subprocess.check_output(["git", "-C", str(REPOSITORY), "show",
        "HEAD:src/satellite_rl/coverage/variable_runner.py"])
    expected_runner_sha = hashlib.sha256(runner_blob).hexdigest()
    working_runner = REPOSITORY / "src/satellite_rl/coverage/variable_runner.py"
    dirty_runner = subprocess.check_output(["git", "-C", str(REPOSITORY),
        "status", "--porcelain", "--", "src/satellite_rl/coverage/variable_runner.py"],
        text=True).strip()
    normalized_working = working_runner.read_bytes().replace(b"\r\n", b"\n")
    if dirty_runner or hashlib.sha256(normalized_working).hexdigest() != expected_runner_sha:
        raise RuntimeError("GPU equivalence candidate runner is not the reviewed Git content")
    candidate_runner_sha = hashlib.sha256(working_runner.read_bytes()).hexdigest()
    loader = next((item for item in rows["frozen_checkpoint_loader.json"]
                   if item.get("route") == route), None)
    checks = {
        "constructor_1728_exact": (constructor.get("pairs") == 1728 and
                                    constructor.get("all_layout_info_exact") is True and
                                    constructor.get("all_rng_unchanged") is True),
        "pool_packet_state_rng_equal": bool(pool) and all(value is True for value in pool.values()),
        "trajectory_128_weight_optimizer_rng_equal": (
            trajectory.get("decisions") == 128 and trajectory.get("all_equal") is True and
            all(value is True for value in trajectory.get("checks", {}).values())),
        "same_input_action_gpu_update_equal": (
            gpu.get("same_4090d") is True and gpu.get("requests_exact") is True and
            gpu.get("candidate_runner_sha256") in (expected_runner_sha, candidate_runner_sha) and
            gpu.get("candidate_identity_exact") is True and
            gpu.get("actions_excluding_wall_time_exact") is True and
            gpu.get("episodes_excluding_wall_time_exact") is True and
            gpu.get("updates_excluding_wall_time_exact") is True and
            gpu.get("stream_state_exact") is True and
            gpu.get("policy", {}).get("bitwise_equal") is True and
            gpu.get("optimizer", {}).get("bitwise_equal") is True and
            all(gpu.get("rng", {}).get(key) is True for key in ("python", "numpy", "torch", "cuda"))),
        "prior_frozen_checkpoint_loader_complete": (
            loader is not None and loader.get("loaded") is True and
            loader.get("format_version") == 2 and loader.get("streams") == 32 and
            loader.get("source_commit") == ORIGIN_COMMIT),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{route}: static CUDA/constructor/checkpoint equivalence evidence differs: {checks}")
    return paths, checks


def _upload_immutable(port, source, destination):
    expected = sha(source)
    code = f'''
import hashlib,json,pathlib
path=pathlib.Path({destination!r});path.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'exists':path.exists(),
 'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}}))
'''
    current = remote(port, code)
    if current["exists"] and current["sha256"] != expected:
        raise RuntimeError("immutable recovery evidence already exists with another SHA")
    if not current["exists"]:
        uploaded = ops.upload(source, destination, port)
        if uploaded["remote"] != expected or uploaded["local_sha256"] != expected:
            raise RuntimeError("recovery evidence remote SHA differs")
    return {"path": destination, "sha256": expected}


def _actual_state(port, directory):
    info = ops.instance(port)
    code = f'''
import json,pathlib,sys
repo=pathlib.Path({info['repo']!r});sys.path.insert(0,str(repo/'src'))
from satellite_rl.workbench.storage import Store,digest,read_json
from satellite_rl.workbench.variable_throughput_recovery import _checkpoint_state
root=pathlib.Path({directory!r});job=read_json(root/'job.json');config=read_json(root/'config.json');status=read_json(root/'status.json')
store=Store({info['root']!r},source_root=str(repo))
state=_checkpoint_state(root,job,config,status)
print(json.dumps({{'state':state,'config_sha256':digest(root/'config.json'),
 'old_source_root':job['source_root'],'job_git_commit':job['git_commit'],
 'request_id':job['request_id'],'dataset_sha256':job['variable_dataset_sha256']}}))
'''
    return remote(port, code, timeout=180)


def stage_validation_receipt(run_id, new_commit, approval_commit):
    if run_id not in TARGETS or approval_commit != new_commit:
        raise ValueError("exact target run and explicit reviewed commit approval required")
    deployment = load("deployment.json")
    if deployment.get("repair_commit") != new_commit or set(deployment.get("lanes", {})) != {"20418", "20424"}:
        raise RuntimeError("both reviewed throughput workbench sources must be deployed")
    for hold_port in (20418, 20424):
        hold(hold_port)
    stopped = _check_stopped_and_held()[run_id]
    detail = job_detail(run_id)
    port, route = detail["port"], TARGETS[run_id]["route"]
    if detail["job"].get("git_commit") != ORIGIN_COMMIT or detail["status"].get("stage") != "stopped":
        raise RuntimeError("original same-ID job changed before validation receipt")
    paths, static = _static_equivalence(route)
    evidence = []
    for label, path in paths.items():
        remote_path = f"{detail['directory']}/recoveries/evidence/{label}"
        evidence.append(_upload_immutable(port, path, remote_path))
    actual = _actual_state(port, detail["directory"])
    state = actual["state"]
    if (actual["job_git_commit"] != ORIGIN_COMMIT or
            actual["request_id"] != TARGETS[run_id]["request_id"] or
            actual["dataset_sha256"] != {"train": TRAIN_SHA256,
                                         "validation": VALIDATION_SHA256} or
            state["checkpoint_sha256"] != stopped["checkpoint"]["sha256"] or
            state["checkpoint_step"] != stopped["checkpoint"]["step"]):
        raise RuntimeError("actual stopped checkpoint differs from prior identity receipt")
    new_root = deployment["lanes"][str(port)]["new_frozen_source"]["source_root"]
    if new_root != f"{ops.instance(port)['root']}/outputs/workbench/source_snapshots/{new_commit}":
        raise RuntimeError("new frozen source path differs from exact lane")
    criteria = {key: True for key in CRITERIA}
    if (not all(static.values()) or state["stream_count"] != 32 or
            state["checkpoint_step"] % 2048 or
            state["ledger_counts"]["actions.jsonl"] != state["checkpoint_step"]):
        raise RuntimeError("actual state/ledger proof does not support all seven criteria")
    receipt = {"format": "s1v10_throughput_repair_validation_v1",
               "repair_id": REPAIR_ID, "origin_commit": ORIGIN_COMMIT,
               "repair_commit": new_commit,
               "old_source_root": actual["old_source_root"],
               "new_source_root": new_root,
               "run_id": run_id, "request_id": TARGETS[run_id]["request_id"],
               "route": route, "config_sha256": actual["config_sha256"],
               "config_hash": state["config_hash"],
               "dataset_sha256": {"train": TRAIN_SHA256,
                                  "validation": VALIDATION_SHA256},
               "stream_count": 32,
               "stop_reason": state["stop_reason"],
               "saved_validation_nodes": state["saved_validation_nodes"],
               "actual_checkpoint": {"sha256": state["checkpoint_sha256"],
                                     "step": state["checkpoint_step"],
                                     "updates": state["checkpoint_updates"],
                                     "episode": state["checkpoint_episode"]},
               "tests_passed": True, "approved": True,
               "criteria": criteria, "evidence_files": evidence,
               "static_checks": static,
               "created_utc": now(),
               "scope": "same-ID frozen-config CPU supply only; 64-step GPU equivalence is not formal throughput"}
    local = save(f"validation_{run_id}.json", receipt)
    remote_path = f"{detail['directory']}/recoveries/validation_receipt.json"
    reference = _upload_immutable(port, local, remote_path)
    code = f'''
import json,pathlib,sys
repo=pathlib.Path({ops.instance(port)['repo']!r});sys.path.insert(0,str(repo/'src'))
from satellite_rl.workbench.storage import Store,read_json
from satellite_rl.workbench.variable_throughput_recovery import _checkpoint_state,_validation_receipt
root=pathlib.Path({detail['directory']!r});store=Store({ops.instance(port)['root']!r},source_root=str(repo))
job=read_json(root/'job.json');config=read_json(root/'config.json');status=read_json(root/'status.json')
state=_checkpoint_state(root,job,config,status)
verified=_validation_receipt(store,root,{remote_path!r},{reference['sha256']!r},
 job=job,config=config,status=status,checkpoint_state=state,new_commit={new_commit!r})
print(json.dumps({{'passed':True,'run_id':verified['run_id'],
 'checkpoint_sha256':state['checkpoint_sha256'],'receipt_sha256':{reference['sha256']!r}}}))
'''
    verified = remote(port, code, timeout=180)
    if not verified["passed"]:
        raise RuntimeError("remote source-repair receipt verification failed")
    result = {"run_id": run_id, "port": port, "route": route,
              "validation_receipt": reference,
              "actual_checkpoint": receipt["actual_checkpoint"],
              "remote_validation": verified,
              "evidence_files": evidence,
              "observed_utc": now()}
    save(f"receipt_staged_{run_id}.json", result)
    return result


def resume_training(run_id, new_commit):
    if run_id not in TARGETS:
        raise ValueError("only two exact formal S1-V10 training IDs may resume")
    receipt = load(f"receipt_staged_{run_id}.json")
    deployment = load("deployment.json")
    port = int(TARGETS[run_id]["port"])
    for hold_port in (20418, 20424):
        hold(hold_port)
    if (deployment.get("repair_commit") != new_commit or
            receipt.get("run_id") != run_id or receipt.get("port") != port or
            not api(port, "/api/dispatch/hold").get("active")):
        raise RuntimeError("same-ID repair deployment/receipt/dispatch hold differs")
    stopped = stopped_checkpoint(run_id)
    if stopped["checkpoint"]["sha256"] != receipt["actual_checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint changed since exact validation receipt")
    path = receipt["validation_receipt"]["path"]
    proof_sha = receipt["validation_receipt"]["sha256"]
    payload = {"source_repair": REPAIR_ID,
               "repair_validation_path": path,
               "repair_validation_sha256": proof_sha,
               "expected_origin_commit": ORIGIN_COMMIT,
               "expected_repair_commit": new_commit}
    response = api(port, f"/api/runs/{run_id}/resume", payload)
    detail = job_detail(run_id)
    job = detail["job"]
    proof = job.get("variable_throughput_repair") or {}
    if (response.get("stage") != "queued" or
            detail["status"]["stage"] != "queued" or
            job.get("git_commit") != new_commit or
            job.get("source_root") != deployment["lanes"][str(port)]["new_frozen_source"]["source_root"] or
            job.get("run_id") != run_id or job.get("request_id") != TARGETS[run_id]["request_id"] or
            proof.get("repair_id") != REPAIR_ID or
            proof.get("checkpoint_sha256") != stopped["checkpoint"]["sha256"] or
            proof.get("validation_receipt_sha256") != proof_sha or
            proof.get("origin_commit") != ORIGIN_COMMIT or
            proof.get("repair_commit") != new_commit):
        raise RuntimeError("workbench same-ID resume did not bind exact original state and new source")
    result = {"run_id": run_id, "port": port, "queue_status": response,
              "old_source_commit": ORIGIN_COMMIT, "new_source_commit": new_commit,
              "checkpoint_sha256": proof["checkpoint_sha256"],
              "checkpoint_step": proof["checkpoint_step"],
              "validation_receipt_sha256": proof_sha,
              "repair_archive_path": proof["archive_path"],
              "observed_utc": now()}
    # The archive lives on the remote host. Read its SHA through pinned SSH.
    code = f'''
import hashlib,json,pathlib
p=pathlib.Path({proof['archive_path']!r})/'repair_proof.json'
print(json.dumps({{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                  'path':str(p)}}))
'''
    result["remote_repair_proof"] = remote(port, code)
    save(f"resume_{run_id}.json", result)
    return result


def rebind_controller(new_commit):
    if controller_process() is not None or not (OUT / "controller_pause.json").is_file():
        raise RuntimeError("original controller still runs or has no verified pause receipt")
    for port in (20418, 20424):
        hold(port)
    deployment = load("deployment.json")
    contract = load("deployment.json").get("throughput_dataset_contract") or {}
    if deployment.get("repair_commit") != new_commit or len(contract.get("sha256", "")) != 64:
        raise RuntimeError("reviewed new source/learned evaluation scene sidecar missing")
    state = json.loads(CONTROLLER_STATE.read_text(encoding="utf-8"))
    if (state.get("scientific_commit") != ORIGIN_COMMIT or
            state.get("throughput_source_overrides")):
        raise RuntimeError("controller science source or prior repair mapping differs")
    overrides = {}
    for run_id, target in TARGETS.items():
        resume = load(f"resume_{run_id}.json")
        route = target["route"]
        train_key, test_key = f"train_{route}", f"test_{route}"
        if (resume.get("run_id") != run_id or
                state["tasks"][train_key]["run_id"] != run_id or
                resume.get("new_source_commit") != new_commit):
            raise RuntimeError("controller training task ID differs from repaired worker")
        overrides[train_key] = {
            "origin_commit": ORIGIN_COMMIT, "repair_commit": new_commit,
            "validation_receipt_sha256": resume["validation_receipt_sha256"],
            "origin_checkpoint_sha256": resume["checkpoint_sha256"],
            "repair_proof_sha256": resume["remote_repair_proof"]["sha256"]}
        overrides[test_key] = {
            "origin_commit": ORIGIN_COMMIT, "repair_commit": new_commit,
            "dataset_contract_sha256": contract["sha256"],
            "source_training_run_id": run_id}
    before = save("controller_state_before_source_override.json", state)
    state["throughput_source_overrides"] = overrides
    state["throughput_repair_controller_receipt"] = str(before)
    for key in ("train_R08", "train_R09"):
        state.get("need_attention", {}).pop(key, None)
    from scripts.stage1_variable import controller
    controller.validate_throughput_overrides(state)
    controller.save(CONTROLLER_STATE, state)
    controller.write_run_manifest(state)
    receipt = {"at": now(), "source_commit": ORIGIN_COMMIT,
               "repair_commit": new_commit,
               "controller_state_before_sha256": sha(before),
               "controller_state_after_sha256": sha(CONTROLLER_STATE),
               "overrides": overrides}
    save("controller_rebind.json", receipt)
    return receipt


def start_controller():
    import psutil
    if controller_process() is not None or repaired_controller_process() is not None:
        raise RuntimeError("another S1-V10 controller still runs")
    state = json.loads(CONTROLLER_STATE.read_text(encoding="utf-8"))
    if set(state.get("throughput_source_overrides", {})) != {
            "train_R08", "test_R08", "train_R09", "test_R09"}:
        raise RuntimeError("controller has not adopted both exact throughput repair routes")
    from scripts.stage1_variable import controller
    controller.validate_throughput_overrides(state)
    command = [sys.executable, "-B", "-m", "scripts.stage1_variable.controller",
               "run", "--commit", ORIGIN_COMMIT, "--interval", "60"]
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    stdout = (OUT / "controller_after_repair.stdout.log").open("ab", buffering=0)
    stderr = (OUT / "controller_after_repair.stderr.log").open("ab", buffering=0)
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
             if os.name == "nt" else 0)
    process = subprocess.Popen(command, cwd=REPOSITORY, env=env,
                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                               creationflags=flags, start_new_session=os.name != "nt",
                               close_fds=True)
    time.sleep(1)
    actual = psutil.Process(process.pid)
    if not actual.is_running() or "scripts.stage1_variable.controller" not in " ".join(actual.cmdline()):
        raise RuntimeError("new local controller did not start with expected module")
    receipt = {"at": now(), "pid": process.pid, "birth": actual.create_time(),
               "command": command, "controller_state_sha256": sha(CONTROLLER_STATE),
               "repair_commit": state["throughput_source_overrides"]["train_R08"]["repair_commit"]}
    save("controller_after_repair_launch.json", receipt)
    return receipt


def release_holds():
    if repaired_controller_process() is None:
        raise RuntimeError("new source-aware controller must be running before dispatch release")
    result = {}
    for port in (20418, 20424):
        row = api(port, "/api/dispatch/hold/release", {})
        if row.get("active") or api(port, "/api/dispatch/hold").get("active"):
            raise RuntimeError(f"{port}: dispatch hold remains active")
        result[str(port)] = {"active": False, "released_utc": now()}
    save("dispatch_release.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "plan", "pause-controller", "stop-training", "wait-stopped",
        "deploy", "stage-receipt", "resume", "rebind-controller",
        "start-controller", "release-holds"))
    parser.add_argument("--repair-commit")
    parser.add_argument("--approval-commit")
    parser.add_argument("--run-id", choices=tuple(TARGETS))
    args = parser.parse_args()
    if args.action != "plan":
        if (not args.repair_commit or args.approval_commit != args.repair_commit):
            parser.error("all mutation/recovery steps require exact --repair-commit and matching --approval-commit")
        commit_diff(args.repair_commit)
    if args.action in ("stage-receipt", "resume") and not args.run_id:
        parser.error("--run-id required")
    result = (
        plan(args.repair_commit) if args.action == "plan" else
        pause_controller() if args.action == "pause-controller" else
        stop_training() if args.action == "stop-training" else
        wait_stopped() if args.action == "wait-stopped" else
        deploy_repair(args.repair_commit) if args.action == "deploy" else
        stage_validation_receipt(args.run_id, args.repair_commit, args.approval_commit)
            if args.action == "stage-receipt" else
        resume_training(args.run_id, args.repair_commit) if args.action == "resume" else
        rebind_controller(args.repair_commit) if args.action == "rebind-controller" else
        start_controller() if args.action == "start-controller" else
        release_holds())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
