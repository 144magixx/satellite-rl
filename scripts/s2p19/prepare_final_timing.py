"""Audited S2-P19 final-timing lane handoff; default is strictly read-only.

Only after three original training runs and three unique endpoint B jobs have
completed may ``--execute`` stop this group's local controller and replace the
20418 workbench server with the same 13ea source at one worker/four 1-thread
settings. No training, evaluation, or other-instance job is submitted here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from . import controller, ops, plan_builder


EXPERIMENT = "S2-P19"
SERVICE_COMMIT = "13ea014f1c1709c4a82059fc89ba3092e75010e1"
CONTROLLER_COMMIT = "ee0a60c7636212671964c5003f322016b41a880e"
CONTROLLER_DIR = Path("C:/Users/Admin/.codex/worktrees/s2p19-ops-timing/paper")
ARMS = {
    "C0": (20418, "train_sac_20261007_152717_479365fb", "s2p19-c0-train-51001",
           "48c7c4feac71a6da9090d2274cbce73c9cb78e2f"),
    "C1": (20424, "train_sac_20261007_153047_1faf6d10", "s2p19-c1-train-51001",
           "48c7c4feac71a6da9090d2274cbce73c9cb78e2f"),
    "I0": (20418, "train_sac_20261007_174121_7d0b687c", "s2p19-i0-train-51001",
           "62e33c1baa066c2a577a351cd18ecb487f4e11e8"),
}
SWITCH_DIR = ops.LOCAL / "preflight/final_timing_switch"
SERVICE_RECEIPT = ops.LOCAL / "preflight/service_13ea/deployment_receipt.json"
SWITCH_RECEIPT = SWITCH_DIR / "switch_receipt.json"
ORIGINAL_RECEIPT = SWITCH_DIR / "original_service_13ea_deployment_receipt.json"
INTENT = SWITCH_DIR / "intent.json"
THREAD_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


class TimingGateError(RuntimeError):
    """A source or identity drift that must not become a wait/retry."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique(rows: list[dict], request_id: str) -> dict | None:
    selected = [row for row in rows if row.get("request_id") == request_id]
    if len(selected) > 1:
        raise TimingGateError("duplicate workbench request ID: " + request_id)
    return selected[0] if selected else None


def _details(port: int, run_id: str) -> tuple[dict, dict]:
    detail = ops.api(port, "/api/runs/" + run_id)
    return detail.get("job") or {}, detail.get("status") or {}


def _status_summary(status: dict) -> dict:
    return {key: status.get(key) for key in
            ("stage", "env_step", "episodes", "updates", "update_step",
             "checkpoint_sha256", "completed_due_to", "heartbeat_at")}


def _endpoint_checkpoint(arm: str, job: dict, status: dict) -> dict:
    """Read the actual job-owned final checkpoint, even when status omits SHA."""
    port, run_id, _, commit = ARMS[arm]
    archive = ops.instance(port)["archive"]
    source = f"{archive}/outputs/workbench/source_snapshots/{commit}"
    if job.get("source_root") != source:
        raise TimingGateError(arm + " final checkpoint source root escaped its frozen commit")
    code = f'''
import hashlib,json,pathlib,sys,torch
archive=pathlib.Path({archive!r});run={run_id!r};source=pathlib.Path({source!r})
directory=archive/'outputs/workbench/jobs'/run
checkpoint=directory/'checkpoint.pt';config_path=directory/'config.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
config=json.loads(config_path.read_text(encoding='utf-8'))
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
if {arm!r}=='I0':
 result={{'checkpoint_sha256':sha(checkpoint),'format':'s2p19_i0_checkpoint_v1',
  'step':payload.get('env_step'),'episodes':payload.get('episodes'),
  'run_id':payload.get('run_id'),'frozen_config_sha256':payload.get('frozen_config_sha256'),
  'config_sha256':sha(config_path),'version':payload.get('version'),
  'pilot_gate_passed':payload.get('pilot_gate_passed')}}
else:
 sys.path.insert(0,str(source/'src'))
 from satellite_rl.workbench.checkpoints import config_hash
 progress=payload.get('progress') or {{}}
 provenance=payload.get('environment_state') or {{}}
 result={{'checkpoint_sha256':sha(checkpoint),'format':payload.get('format'),
  'step':progress.get('env_step'),'last_checkpoint_step':progress.get('last_checkpoint_step'),
  'run_id':provenance.get('run_id'),'provenance_commit':provenance.get('git_commit'),
  'provenance_profile':provenance.get('profile'),
  'config_hash':payload.get('config_hash'),'computed_config_hash':config_hash(config),
  'config_sha256':sha(config_path)}}
print(json.dumps(result))
'''
    response = ops.remote_python(code, port, timeout=180)
    if response["exit_code"]:
        raise TimingGateError(arm + " final checkpoint unavailable: " + response["stderr"][-400:])
    saved = json.loads(response["stdout"].splitlines()[-1])
    digest = saved.get("checkpoint_sha256")
    step = status.get("env_step")
    if (not isinstance(digest, str) or len(digest) != 64 or
            saved.get("run_id") != run_id or saved.get("step") != step or
            (arm != "I0" and status.get("checkpoint_step") != step) or
            (arm == "I0" and status.get("checkpoint_step") not in (None, step)) or
            (status.get("checkpoint_sha256") is not None and
             status["checkpoint_sha256"] != digest)):
        raise TimingGateError(arm + " final status/actual checkpoint identity differs")
    if arm == "I0":
        if (saved.get("version") != 1 or saved.get("format") != "s2p19_i0_checkpoint_v1" or
                saved.get("episodes") != status.get("episodes") or
                saved.get("frozen_config_sha256") != saved.get("config_sha256") or
                saved.get("pilot_gate_passed") is not True):
            raise TimingGateError("I0 final complete checkpoint/config/episode differs")
    elif (saved.get("format") != "satellite_rl_workbench_checkpoint_v1" or
          saved.get("last_checkpoint_step") != 400000 or
          saved.get("provenance_commit") != commit or
          saved.get("provenance_profile") != "europe_v1" or
          saved.get("config_hash") != saved.get("computed_config_hash")):
        raise TimingGateError(arm + " final 400000 checkpoint provenance/config differs")
    return saved


def _training_gate(arm: str, catalog: list[dict]) -> tuple[dict, str | None]:
    port, run_id, request_id, expected_commit = ARMS[arm]
    row = _unique(catalog, request_id)
    if row is None or row.get("run_id") != run_id:
        raise TimingGateError(arm + " original training request/run ID changed")
    job, status = _details(port, run_id)
    if (job.get("run_id") != run_id or job.get("request_id") != request_id or
            job.get("git_commit") != expected_commit or job.get("kind") != "train" or
            job.get("experiment_id") != EXPERIMENT or status.get("run_id") != run_id):
        raise TimingGateError(arm + " original workbench training identity/source differs")
    stage, step = status.get("stage"), status.get("env_step")
    if stage in ("failed", "cancelled", "interrupted"):
        raise TimingGateError(arm + " training ended abnormally")
    reason = None
    if stage != "completed":
        reason = arm + " training has not completed"
    elif arm in ("C0", "C1") and step != 400000:
        raise TimingGateError(arm + " completed away from fixed 400000 endpoint")
    elif arm == "I0" and not (
            (step == 400000 and status.get("completed_due_to") == "env_steps") or
            (status.get("episodes") == 12500 and
             status.get("completed_due_to") == "episode_cap" and
             type(step) is int and 0 < step < 400000)):
        raise TimingGateError("I0 completed away from its fixed env-step/episode endpoint")
    summary = _status_summary(status)
    checkpoint = None
    if stage == "completed":
        checkpoint = _endpoint_checkpoint(arm, job, status)
        summary["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
    return {"port": port, "run_id": run_id, "request_id": request_id,
            "git_commit": expected_commit, "status": summary,
            "checkpoint": checkpoint}, reason


def _endpoint_B_gate(arm: str, catalog: list[dict], train: dict) -> tuple[dict, str | None]:
    port, run_id, _, _ = ARMS[arm]
    request_id = f"s2p19-{arm.lower()}-endpoint-B"
    row = _unique(catalog, request_id)
    if row is None:
        return {"request_id": request_id, "state": "absent"}, arm + " endpoint B absent"
    b_id = row.get("run_id")
    if not isinstance(b_id, str) or not b_id.startswith("refine_"):
        raise TimingGateError(arm + " endpoint B has invalid workbench run ID")
    job, status = _details(port, b_id)
    if (job.get("run_id") != b_id or job.get("request_id") != request_id or
            job.get("kind") != "refine" or job.get("experiment_id") != EXPERIMENT or
            status.get("run_id") != b_id):
        raise TimingGateError(arm + " endpoint B workbench identity differs")
    source = job.get("initial_source") or {}
    train_status = train["status"]
    if (source.get("run_id") != run_id or source.get("group") != "model" or
            source.get("partition") != "validation" or
            source.get("validation_step") != train_status["env_step"] or
            source.get("source_checkpoint_sha256") != train_status["checkpoint_sha256"] or
            source.get("scene_count") != 72 or
            len(source.get("case_files") or []) != 72):
        raise TimingGateError(arm + " endpoint B is not bound to this exact final model/step")
    if status.get("stage") in ("failed", "stopped", "cancelled", "interrupted"):
        raise TimingGateError(arm + " endpoint B failed or stopped")
    result = {"run_id": b_id, "request_id": request_id,
              "source_run_id": source["run_id"], "validation_step": source["validation_step"],
              "source_checkpoint_sha256": source["source_checkpoint_sha256"],
              "status": _status_summary(status)}
    if status.get("stage") != "completed":
        return result, arm + " endpoint B has not completed"
    if status.get("env_step") != 216 or status.get("budget_env_steps") != 216:
        raise TimingGateError(arm + " endpoint B is not the complete 72×3 method budget")
    return result, None


def _controller_state_ready() -> str | None:
    plan, plan_sha = controller.load_plan()
    saved = json.loads(controller.STATE.read_text(encoding="utf-8"))
    if (saved.get("plan_sha256") != plan_sha or
            saved.get("plan_revision") != plan.get("revision") or
            plan.get("service_commit") != SERVICE_COMMIT):
        raise TimingGateError("controller plan/state/service identity differs")
    if plan.get("timing_isolation_receipt"):
        raise TimingGateError("final timing is already pinned by another appended plan")
    for arm, (port, run_id, _, _) in ARMS.items():
        receipt = saved.get("tasks", {}).get(f"{arm}_pilot_{port}") or {}
        if receipt.get("run_id") != run_id:
            raise TimingGateError(arm + " controller task changed original run identity")
        if receipt.get("state") != "completed":
            return arm + " controller has not observed completed training"
    return None


def inspect() -> dict:
    """Actual workbench API and GPU observations; no local or remote writes."""
    catalogs = {port: ops.api(port, "/api/s2p19/runs")["runs"] for port in ops.SSH_PORTS}
    trains, refinements, reasons = {}, {}, []
    for arm in ARMS:
        trains[arm], reason = _training_gate(arm, catalogs[ARMS[arm][0]])
        if reason:
            reasons.append(reason)
        if reason is None:
            refinements[arm], reason = _endpoint_B_gate(
                arm, catalogs[ARMS[arm][0]], trains[arm])
            if reason:
                reasons.append(reason)
        else:
            refinements[arm] = {"state": "waiting_training"}
    lane = {}
    if not reasons:
        controller_reason = _controller_state_ready()
        if controller_reason:
            reasons.append(controller_reason)
        for port in ops.SSH_PORTS:
            snapshot = plan_builder._timing_lane_observation(port)
            lane[str(port)] = snapshot
            if (snapshot.get("active_jobs") or snapshot.get("external_gpu_pids") or
                    len(snapshot.get("server") or []) != 1 or
                    len(snapshot.get("gpu_uuids") or []) != 1):
                reasons.append(f"{port} lane still has active jobs, external GPU use, or ambiguous server")
    return {"experiment_id": EXPERIMENT, "observed_utc": now(),
            "status": "ready" if not reasons else "waiting", "reasons": reasons,
            "training": trains, "endpoint_B": refinements, "lanes": lane,
            "mutation_count": 0}


def _process_identity() -> dict:
    import psutil
    matches = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "create_time"]):
        try:
            args = process.info["cmdline"] or []
            cwd = process.info["cwd"]
            if (len(args) >= 5 and args[1:5] ==
                    ["-u", "-m", "scripts.s2p19.controller", "--run"] and
                    Path(cwd or "").resolve() == CONTROLLER_DIR.resolve()):
                matches.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    if len(matches) != 1:
        raise TimingGateError("need exactly one live ee controller process")
    process = matches[0]
    if (process.children(recursive=True) or not process.is_running() or
            os.path.abspath(process.exe()).lower() !=
            os.path.abspath("C:/Users/Admin/anaconda3/python.exe").lower()):
        raise TimingGateError("ee controller process ownership/children differ")
    import subprocess
    head = subprocess.run(["git", "-C", str(CONTROLLER_DIR), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != CONTROLLER_COMMIT:
        raise TimingGateError("ee controller checkout is not its frozen commit")
    return {"pid": process.pid, "start_unix": process.create_time(),
            "cmdline": process.cmdline(), "cwd": process.cwd(), "commit": head}


def _service_process() -> dict:
    """Identify the single 8900 owner by PID birth, source, args and thread env."""
    info = ops.instance(20418)
    saved = json.loads(SERVICE_RECEIPT.read_text(encoding="utf-8"))
    pinned = saved["instances"]["20418"]
    if saved.get("service_commit") != SERVICE_COMMIT:
        raise TimingGateError("13ea service deployment receipt source differs")
    code = f'''
import json,os,pathlib,socket,subprocess
repo=pathlib.Path({pinned['service_repo']!r})
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
rows=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  parts=[item.decode('utf-8','replace') for item in (proc/'cmdline').read_bytes().split(b'\\x00') if item]
  if ('satellite_rl.workbench.server' not in parts or '--archive-root' not in parts or
      {info['archive']!r} not in parts or '--port' not in parts or '8900' not in parts):continue
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  environment=(proc/'environ').read_bytes().split(b'\\x00')
  values={{name:next((item.decode('utf-8','replace').split('=',1)[1] for item in environment
                       if item.startswith((name+'=').encode())),None)
          for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS','PYTHONPATH','SATELLITE_RL_PRIORITY_DEPS')}}
  rows.append({{'pid':int(proc.name),'start_ticks':ticks,'cmdline':parts,
               'cwd':os.readlink(proc/'cwd'),'environment':values}})
 except (OSError,ValueError,IndexError):continue
with socket.socket() as sock:
 sock.settimeout(2);busy=sock.connect_ex(('127.0.0.1',8900))==0
print(json.dumps({{'head':head,'dirty':dirty,'rows':rows,'port_busy':busy}}))
'''
    result = ops.remote_python(code, 20418, timeout=45)
    if result["exit_code"]:
        raise TimingGateError("cannot inspect exact 8900 service: " + result["stderr"][-400:])
    observed = json.loads(result["stdout"].splitlines()[-1])
    rows = observed["rows"]
    command = rows[0]["cmdline"] if len(rows) == 1 else []
    workers = (command[command.index("--max-workers") + 1]
               if "--max-workers" in command and
               command.index("--max-workers") + 1 < len(command) else None)
    worker_python = (command[command.index("--python") + 1]
                     if "--python" in command and
                     command.index("--python") + 1 < len(command) else None)
    environment = rows[0]["environment"] if len(rows) == 1 else {}
    pythonpath = (environment.get("PYTHONPATH") or "").split(":")
    solver_env = environment.get("SATELLITE_RL_PRIORITY_DEPS")
    if (observed["head"] != SERVICE_COMMIT or observed["dirty"] or
            len(rows) != 1 or not observed["port_busy"] or
            rows[0]["cwd"] != pinned["service_repo"] or
            pinned["service_repo"] + "/src" not in pythonpath or
            ops.PRIORITY_DEPS not in pythonpath or
            workers not in ("1", "2") or
            (workers == "1" and solver_env != ops.PRIORITY_DEPS) or
            (workers == "2" and solver_env not in (None, ops.PRIORITY_DEPS)) or
            worker_python != ops.DQN or
            "--no-scheduler" in command or not command or command[0] != ops.DQN):
        raise TimingGateError("8900 owner/source/package path is ambiguous")
    row = rows[0]
    row["max_workers"] = workers
    row["source_commit"] = observed["head"]
    row["original_receipt_pid"] = pinned["new_server_pid"]
    row["original_receipt_ticks"] = pinned["new_server_start_ticks"]
    return row


def _desired_service(row: dict) -> bool:
    return (row.get("max_workers") == "1" and
            all(row.get("environment", {}).get(key) == "1" for key in THREAD_NAMES))


def _write(path: Path, value: dict) -> None:
    controller.write_json(path, value)


def _stop_controller(identity: dict) -> None:
    import psutil
    process = psutil.Process(identity["pid"])
    if (process.create_time() != identity["start_unix"] or
            process.cmdline() != identity["cmdline"] or
            process.cwd() != identity["cwd"] or process.children(recursive=True)):
        raise TimingGateError("ee controller changed before exact stop")
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired as error:
        raise TimingGateError("exact ee controller did not exit") from error


def _stop_old_service(row: dict) -> dict:
    if (row["pid"] != row["original_receipt_pid"] or
            row["start_ticks"] != row["original_receipt_ticks"] or
            row["max_workers"] != "2"):
        raise TimingGateError("old 8900 service is not the pinned two-worker process")
    code = f'''
import json,os,pathlib,signal,socket,time
pid={row['pid']};ticks={row['start_ticks']};cwd={row['cwd']!r};p=pathlib.Path('/proc')/str(pid)
if not p.is_dir():raise RuntimeError('pinned old server disappeared')
stat=(p/'stat').read_text();actual=int(stat[stat.rfind(')')+2:].split()[19])
args=(p/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace')
if actual!=ticks or os.readlink(p/'cwd')!=cwd or '--port 8900' not in args or '--max-workers 2' not in args:
 raise RuntimeError('old server process changed before signal')
os.kill(pid,signal.SIGTERM)
for _ in range(60):
 time.sleep(.1)
 if not p.is_dir():break
with socket.socket() as sock:
 sock.settimeout(2);busy=sock.connect_ex(('127.0.0.1',8900))==0
print(json.dumps({{'old_pid':pid,'old_ticks':ticks,'closed':not p.is_dir() and not busy}}))
'''
    response = ops.remote_python(code, 20418, timeout=30)
    if response["exit_code"]:
        raise TimingGateError("old 8900 service stop uncertain: " + response["stderr"][-500:])
    observed = json.loads(response["stdout"].splitlines()[-1])
    if not observed["closed"]:
        raise TimingGateError("old 8900 service still owns its port")
    return observed


def _launch_one_worker() -> dict:
    info = ops.instance(20418)
    saved = json.loads(SERVICE_RECEIPT.read_text(encoding="utf-8"))
    repo = saved["instances"]["20418"]["service_repo"]
    log = info["ops"] + "/service_13ea/final_timing_8900.log"
    code = f'''
import json,os,pathlib,socket,subprocess,time,urllib.request
repo=pathlib.Path({repo!r});archive=pathlib.Path({info['archive']!r});log=pathlib.Path({log!r})
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!={SERVICE_COMMIT!r} or dirty:raise RuntimeError('same-source 13ea checkout changed')
with socket.socket() as sock:
 sock.settimeout(2)
 if sock.connect_ex(('127.0.0.1',8900))==0:raise RuntimeError('8900 already has an owner')
deps=pathlib.Path({ops.PRIORITY_DEPS!r})
if not deps.is_dir():raise RuntimeError('pinned solver dependencies missing')
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src')+os.pathsep+str(deps),
 SATELLITE_RL_PRIORITY_DEPS=str(deps),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
identity=subprocess.run([{ops.DQN!r},'-c',
 'import json,pathlib,torch,cvxpy,clarabel,satellite_rl;print(json.dumps({{"package":str(pathlib.Path(satellite_rl.__file__).resolve()),"cuda":torch.cuda.is_available(),"cuda_build":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"cvxpy":cvxpy.__version__,"clarabel":clarabel.__version__}}))'],
 cwd=str(repo),env=env,capture_output=True,text=True,timeout=30,check=True)
packages=json.loads(identity.stdout.splitlines()[-1])
if (packages['package']!=str(repo/'src/satellite_rl/__init__.py') or not packages['cuda'] or
    packages['cvxpy']!='1.6.5' or packages['clarabel']!='0.11.1'):
 raise RuntimeError('DQN CUDA/package/pinned solver identity differs')
with log.open('ab',buffering=0) as output:
 proc=subprocess.Popen([{ops.DQN!r},'-B','-m','satellite_rl.workbench.server',
  '--archive-root',str(archive),'--port','8900','--python',{ops.DQN!r},
  '--max-workers','1'],cwd=str(repo),env=env,stdin=subprocess.DEVNULL,
  stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
for _ in range(80):
 time.sleep(.25)
 try:
  with urllib.request.urlopen('http://127.0.0.1:8900/api/catalog',timeout=2) as response:
   json.loads(response.read().decode('utf-8'));break
 except Exception:
  if proc.poll() is not None:raise RuntimeError('one-worker server exited: '+log.read_text(errors='replace')[-500:])
else:raise RuntimeError('one-worker server did not answer')
p=pathlib.Path('/proc')/str(proc.pid)
stat=(p/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({{'pid':proc.pid,'start_ticks':ticks,'repo':str(repo),'archive':str(archive),
 'log':str(log),'service_commit':head,'max_workers':1,'packages':packages}}))
'''
    response = ops.remote_python(code, 20418, timeout=90)
    if response["exit_code"]:
        raise TimingGateError("new 8900 launch outcome unknown; inspect port owner before any retry: " +
                              response["stderr"][-500:])
    return json.loads(response["stdout"].splitlines()[-1])


def _update_deployment_receipt(row: dict, *, original: dict) -> dict:
    updated = json.loads(json.dumps(original))
    instance = updated["instances"]["20418"]
    instance.update(new_server_pid=row["pid"], new_server_start_ticks=row["start_ticks"],
                    service_repo=row["cwd"],
                    service_package=row["cwd"] + "/src/satellite_rl/__init__.py",
                    final_timing_max_workers=1,
                    final_timing_thread_environment={key: "1" for key in THREAD_NAMES})
    updated["final_timing_switch"] = {"at": now(), "target_port": 20418,
                                      "same_service_commit": SERVICE_COMMIT,
                                      "original_receipt": str(ORIGINAL_RECEIPT),
                                      "new_server_pid": row["pid"],
                                      "new_server_start_ticks": row["start_ticks"]}
    _write(SERVICE_RECEIPT, updated)
    return updated


def _adopt_existing_switch() -> dict:
    """Recover an unknown response by identity, never by starting another server."""
    if not INTENT.is_file() or not ORIGINAL_RECEIPT.is_file():
        raise TimingGateError("same-target adoption requires the original intent and receipt")
    intent = json.loads(INTENT.read_text(encoding="utf-8"))
    if controller.digest(ORIGINAL_RECEIPT) != intent.get("original_receipt_sha256"):
        raise TimingGateError("original 13ea deployment receipt changed during recovery")
    import psutil
    for process in psutil.process_iter(["cmdline", "cwd"]):
        try:
            if ("scripts.s2p19.controller" in (process.info["cmdline"] or []) and
                    Path(process.info["cwd"] or "").resolve() == CONTROLLER_DIR.resolve()):
                raise TimingGateError("ee controller is active during same-target adoption")
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    service = _service_process()
    if not _desired_service(service):
        raise TimingGateError("prior switch has no exact one-worker target; do not relaunch blindly")
    original = json.loads(ORIGINAL_RECEIPT.read_text(encoding="utf-8"))
    if (service["cwd"] != original["instances"]["20418"]["service_repo"] or
            service["source_commit"] != SERVICE_COMMIT):
        raise TimingGateError("existing one-worker service is not the approved same-source target")
    current = json.loads(SERVICE_RECEIPT.read_text(encoding="utf-8"))
    pinned = current["instances"]["20418"]
    if (pinned["new_server_pid"], pinned["new_server_start_ticks"]) != (
            service["pid"], service["start_ticks"]):
        if (pinned["new_server_pid"], pinned["new_server_start_ticks"]) != (
                original["instances"]["20418"]["new_server_pid"],
                original["instances"]["20418"]["new_server_start_ticks"]):
            raise TimingGateError("deployment receipt names a third 8900 process")
        _update_deployment_receipt(service, original=original)
    for port in ops.SSH_PORTS:
        controller._service_identity(port, SERVICE_COMMIT)
    timing = plan_builder.capture_final_timing_isolation(["C0", "C1", "I0"])
    receipt = {"format": "s2p19_final_timing_switch_v1", "at": now(),
               "recovered_unknown_outcome": True, "old_service": intent["old_service"],
               "new_server_pid": service["pid"],
               "new_server_start_ticks": service["start_ticks"],
               "new_service": service, "same_service_commit": SERVICE_COMMIT,
               "controller_stopped": intent["controller"],
               "original_receipt_sha256": controller.digest(ORIGINAL_RECEIPT),
               "updated_receipt_sha256": controller.digest(SERVICE_RECEIPT),
               "timing_isolation": timing, "no_training_or_evaluation_API_POST": True}
    _write(SWITCH_RECEIPT, receipt)
    return {"status": "adopted_same_target", "receipt": str(SWITCH_RECEIPT),
            "timing_isolation": timing, "new_server_pid": service["pid"]}


def execute() -> dict:
    if SWITCH_RECEIPT.exists():
        previous = json.loads(SWITCH_RECEIPT.read_text(encoding="utf-8"))
        service = _service_process()
        if (previous.get("new_server_pid") != service["pid"] or
                previous.get("new_server_start_ticks") != service["start_ticks"] or
                not _desired_service(service)):
            raise TimingGateError("existing final timing switch receipt differs; no duplicate launch")
        return {"status": "already_switched", "receipt": str(SWITCH_RECEIPT),
                "service": service, "mutation_count": 0}
    checked = inspect()
    if checked["status"] != "ready":
        return checked  # No local or remote mutation when any gate is waiting.
    if INTENT.exists():
        return _adopt_existing_switch()
    before = _service_process()
    if _desired_service(before):
        raise TimingGateError("one-worker 8900 exists without a saved switch receipt; inspect owner")
    control = _process_identity()
    if (before["pid"] != before["original_receipt_pid"] or
            before["start_ticks"] != before["original_receipt_ticks"]):
        raise TimingGateError("8900 current service differs from original deployment receipt")
    SWITCH_DIR.mkdir(parents=True, exist_ok=True)
    if ORIGINAL_RECEIPT.exists():
        raise TimingGateError("original service receipt backup already exists without switch receipt")
    shutil.copy2(SERVICE_RECEIPT, ORIGINAL_RECEIPT)
    if controller.digest(SERVICE_RECEIPT) != controller.digest(ORIGINAL_RECEIPT):
        raise TimingGateError("13ea service deployment backup SHA differs")
    _write(INTENT, {"format": "s2p19_final_timing_switch_intent_v1", "at": now(),
                    "training_and_B_gate": checked, "controller": control,
                    "old_service": before,
                    "original_receipt_sha256": controller.digest(ORIGINAL_RECEIPT)})
    _stop_controller(control)
    # One final read prevents a concurrent workbench job from starting between
    # the API gate and stopping the old scheduler.
    final = inspect()
    if final["status"] != "ready":
        raise TimingGateError("final timing gate changed after controller stop; leave old service running")
    stopped = _stop_old_service(before)
    launched = _launch_one_worker()
    service = _service_process()
    if (service["pid"] != launched["pid"] or
            service["start_ticks"] != launched["start_ticks"] or
            not _desired_service(service)):
        raise TimingGateError("new one-worker 8900 did not pass exact PID/threads/source verification")
    original = json.loads(ORIGINAL_RECEIPT.read_text(encoding="utf-8"))
    _update_deployment_receipt(service, original=original)
    for port in ops.SSH_PORTS:
        controller._service_identity(port, SERVICE_COMMIT)
    timing = plan_builder.capture_final_timing_isolation(["C0", "C1", "I0"])
    receipt = {"format": "s2p19_final_timing_switch_v1", "at": now(),
               "old_service": before, "stopped_service": stopped,
               "new_server_pid": service["pid"],
               "new_server_start_ticks": service["start_ticks"],
               "new_service": service, "same_service_commit": SERVICE_COMMIT,
               "controller_stopped": control, "original_receipt_sha256":
               controller.digest(ORIGINAL_RECEIPT),
               "updated_receipt_sha256": controller.digest(SERVICE_RECEIPT),
               "timing_isolation": timing,
               "no_training_or_evaluation_API_POST": True}
    _write(SWITCH_RECEIPT, receipt)
    return {"status": "switched", "receipt": str(SWITCH_RECEIPT),
            "timing_isolation": timing, "new_server_pid": service["pid"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Read actual API and lane status only")
    action.add_argument("--execute", action="store_true", help="Execute the exact approved same-source switch")
    args = parser.parse_args(argv)
    result = execute() if args.execute else inspect()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
