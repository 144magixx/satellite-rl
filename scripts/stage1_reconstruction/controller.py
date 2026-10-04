"""Persistent local S1-03 workbench controller.

The controller performs orchestration only. All scenario preparation, training,
evaluation and audits run under the frozen remote DQN/workbench environment.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import traceback

try:
    from . import remote
except ImportError:  # Direct execution: python scripts/stage1_reconstruction/controller.py
    import remote


ACTIVE = {"queued", "starting", "initializing", "preparing", "training", "running",
          "validating", "evaluating", "refining", "warmup"}
S2_SCA_JOBS = ("V0_sca", "V1_sca", "H0_sca", "fixed_sca", "greedy_sca")
ROUTE_METHOD = {"prepare": "none", "R0": "none", "R1": "local_search",
                "R2": "local_search", "R3": "local_search", "P1": "ppo"}
GATES = {"development_min_valid_scenes": 16, "development_min_improved_scenes": 4,
         "min_quality_gain": 1e-6, "mechanism_step": 65536,
         "mechanism_min_improved_scenes": 4,
         "minimum_recent_nonstop_acceptance_rate": 0.5,
         "maximum_immediate_stop_fraction": 0.9, "formal_scenes": 100}


class Attention(RuntimeError):
    """A durable stop requiring investigation; never infer a retry is safe."""


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_json(value):
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


@contextmanager
def single_controller_lock(path):
    """Keep the lock file; the OS releases its byte lock on process death."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise Attention("另一个S1-03控制器已持有本地锁") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise Attention("另一个S1-03控制器已持有本地锁") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_plan(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw.decode("utf-8-sig"))
    required = {"science_commit", "config_template", "ports", "hostroot",
                "local_output_root", "raw_inputs", "gates", "audit_commands", "s2_release"}
    if not isinstance(plan, dict) or not required <= set(plan):
        raise ValueError("S1-03 plan缺少冻结源码、配置、数据、闸门或S2放行信息")
    if not re.fullmatch(r"[0-9a-f]{40}", plan["science_commit"]):
        raise ValueError("science_commit须为完整40位Git提交")
    if plan["ports"] != [20418, 20424] or plan["hostroot"] != remote.REMOTE_BASE:
        raise ValueError("S1-03 plan执行端必须为批准的双实例归档")
    template = plan["config_template"]
    if (template.get("schema_version") != 4 or template.get("research_stage") != "coverage" or
            template.get("algorithm") != "ppo" or
            template.get("training", {}).get("total_steps") != 262144 or
            template.get("training", {}).get("seed") != 51001 or
            template.get("training", {}).get("mechanism_gate_step") != 65536):
        raise ValueError("S1-03 plan配置或单种子预算不符合授权")
    if set(plan["raw_inputs"]) != {"train", "validation", "test"}:
        raise ValueError("必须冻结train/validation/test三份原始场景")
    if plan["gates"] != GATES:
        raise ValueError("S1-03开发、机制及100景正式审计门槛须与方案冻结值一致")
    for role, record in plan["raw_inputs"].items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError(f"{role}原始输入路径缺失")
    test = plan["raw_inputs"]["test"]
    if not all(test.get(key) for key in ("record_path", "manifest_path", "selection_sha256")):
        raise ValueError("新独立考核须冻结record/manifest/selection SHA")
    if not all(key in plan["audit_commands"] for key in ("development", "mechanism", "formal")):
        raise ValueError("三项独立审计命令未配置")
    for mode, command in plan["audit_commands"].items():
        if (not isinstance(command, dict) or not isinstance(command.get("argv"), list) or
                not command["argv"] or not all(isinstance(arg, str) for arg in command["argv"]) or
                len(command["argv"]) < 2 or
                command["argv"][0] != remote.DQN or
                not PurePosixPath(command["argv"][1]).is_absolute() or
                not re.fullmatch(r"[0-9a-f]{64}", command.get("script_sha256", ""))):
            raise ValueError(f"{mode}审计命令须为固定脚本SHA和完整argv")
    if not plan["s2_release"].get("controller_manifest"):
        raise ValueError("S2放行须绑定原控制器实时manifest")
    if plan["s2_release"].get("allow_file") is not None:
        raise ValueError("本轮S2放行不得使用手工文件旁路，须实时核五SCA与8781")
    return plan, sha_bytes(raw)


class Controller:
    def __init__(self, plan, plan_sha256, state_path):
        self.plan = plan
        self.plan_sha256 = plan_sha256
        self.state_path = Path(state_path).resolve()
        self.archive = {port: f"{plan['hostroot']}/{port}" for port in (20418, 20424)}
        existing = read_json(self.state_path)
        if existing is not None and existing.get("plan_sha256") != plan_sha256:
            raise Attention("已保存控制器状态绑定另一份计划；禁止静默改变科学范围")
        self.state = existing or {"schema_version": 1, "plan_sha256": plan_sha256,
                                  "science_commit": plan["science_commit"],
                                  "created_utc": utcnow(), "phase": "preparing",
                                  "datasets": {}, "jobs": {}, "audits": {}, "imports": {},
                                  "observed": {}, "concurrency": {},
                                  "s2_release": None, "test_release": None,
                                  "attention": None}
        self.state.setdefault("concurrency", {})

    def save(self):
        self.state["updated_utc"] = utcnow()
        remote.save(self.state_path, self.state)

    def _remote(self, port, operation, *, retry=3):
        last = None
        for attempt in range(retry):
            client = None
            try:
                client = remote.connect(port)
                return operation(client)
            except Exception as error:
                last = error
                if attempt + 1 < retry:
                    time.sleep(min(3, attempt + 1))
            finally:
                if client is not None:
                    client.close()
        raise Attention(f"{port}远端API/SSH连续{retry}次失败：{last}") from last

    def api(self, port, endpoint, body=None, *, timeout=120, retry=None):
        if retry is None:
            retry = 3 if body is None or endpoint == "coverage/runs" else 1
        return self._remote(port, lambda client: remote.api(client, port, endpoint,
                                                             body, timeout=timeout), retry=retry)

    def code(self, port, code, *, timeout=60, retry=3):
        return self._remote(port, lambda client: remote.run_code(client, code, timeout=timeout), retry=retry)

    def observe(self, label, port, detail):
        self.state["observed"][label] = {"observed_utc": utcnow(), "port": port,
                                          "run_id": detail.get("job", {}).get("run_id"),
                                          "stage": detail.get("status", {}).get("stage"),
                                          "heartbeat_at": detail.get("status", {}).get("heartbeat_at"),
                                          "env_step": detail.get("status", {}).get("env_step"),
                                          "evaluated_scenes": detail.get("status", {}).get("evaluated_scenes"),
                                          "prepared_scenes": detail.get("status", {}).get("prepared_scenes")}

    def _dataset(self, role, port):
        key = f"{port}:{role}"
        source = self.plan["raw_inputs"][role]
        name = f"S1-03-{self.plan_sha256[:12]}-{role}"
        catalog = self.api(port, "coverage/datasets")["datasets"]
        record = next((row for row in catalog if row.get("id") == self.state["datasets"].get(key)), None)
        if record is None:
            matches = [row for row in catalog if row.get("name") == name and
                       row.get("partition_role") == role and row.get("source_path") == source["path"]]
            if len(matches) > 1:
                raise Attention(f"{key}存在多个同计划数据记录，须人工核对")
            record = matches[0] if matches else None
        if record is None:
            record = self.api(port, "coverage/datasets/import", {
                "path": source["path"], "partition_role": role,
                "coverage": self.plan["config_template"]["coverage"], "name": name}, timeout=1800, retry=1)
        if (record.get("research_stage") != "coverage" or record.get("partition_role") != role or
                record.get("source_path") != source["path"] or record.get("initialization") != "cluster" or
                record.get("count") != {"train": 576, "validation": 20, "test": 100}[role] or
                not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise Attention(f"{key}工作台数据来源/用途/SHA不一致")
        self.state["datasets"][key] = record["id"]
        self.state["observed"]["dataset_" + key] = {"observed_utc": utcnow(),
                                                       "sha256": record["sha256"],
                                                       "source_path": record["source_path"],
                                                       "count": record.get("count")}
        return True

    def _job_config(self, label, route):
        config = deepcopy(self.plan["config_template"])
        config["name"] = f"S1-03 {label}"
        config["coverage"]["reconstruction"]["route"] = route
        config["coverage"]["method"] = ROUTE_METHOD[route]
        config["datasets"] = {}
        config["reconstruction_starts"] = {}
        return config

    def _record_concurrency(self, label, port):
        if label in self.state["concurrency"]:
            return
        other = 20424 if port == 20418 else 20418
        system = self.api(port, "system")
        sibling = self.api(other, "coverage/runs")["runs"]
        self.state["concurrency"][label] = {
            "observed_utc": utcnow(), "port": port,
            "device": system.get("device"), "gpu_snapshot": system.get("gpu"),
            "python": system.get("python"), "torch": system.get("torch"),
            "cuda_build": system.get("cuda_build"),
            "server_time": system.get("server_time"),
            "other_instance_active": [{"run_id": row.get("run_id"), "stage": row.get("stage")}
                                      for row in sibling if row.get("stage") in ACTIVE],
        }

    def _job(self, label, port, route, kind, role=None):
        request_id = f"s103_{self.plan_sha256[:16]}_{label}"
        run_id = self.state["jobs"].get(label)
        if not run_id:
            matches = [row for row in self.api(port, "coverage/runs")["runs"]
                       if row.get("request_id") == request_id]
            if len(matches) > 1:
                raise Attention(f"{label}请求编号出现多个任务")
            run_id = matches[0]["run_id"] if matches else None
        if not run_id:
            roles = ("train", "validation") if kind == "train" else (role,)
            if label == "train_P1" or label.startswith("formal_"):
                self._record_concurrency(label, port)
            body = {"kind": kind, "config": self._job_config(label, route),
                    "dataset_ids": {r: self.state["datasets"][f"{port}:{r}"] for r in roles},
                    "dataset_role": role or "test", "experiment_role":
                    "verification" if role == "validation" else "formal",
                    "request_id": request_id}
            if route != "prepare":
                body["start_run_ids"] = {r: self.state["jobs"][f"prepare_{r}"] for r in roles}
            if kind == "evaluate" and route == "P1":
                body["model_id"] = self.state["imports"]["P1_model_20424"]
            job = self.api(port, "coverage/runs", body, timeout=180, retry=3)
            run_id = job["run_id"]
            self.state["jobs"][label] = run_id
            self.save()
        detail = self.api(port, "runs/" + run_id)
        self.observe(label, port, detail)
        job, status = detail["job"], detail["status"]
        if (job.get("request_id") != request_id or job.get("git_commit") != self.plan["science_commit"] or
                job.get("execution_target") != f"u2024110224@10.160.17.102:{port}" or
                job.get("reconstruction_route") != route or job.get("kind") != kind):
            raise Attention(f"{label}任务身份、冻结源码、路线或归属实例不一致")
        self.state["jobs"][label] = run_id
        stage = status.get("stage")
        if stage == "completed":
            return True
        if stage in ACTIVE or stage == "preparing":
            return False
        if stage == "stopped" and label == "train_P1" and status.get("stop_reason") == "mechanism_gate_pending":
            return self._mechanism_gate(run_id, detail)
        raise Attention(f"{label}任务状态{stage}需要保留现场核查，不自动重跑")

    def _read_remote_json(self, port, path):
        code = f"""
import hashlib,json
from pathlib import Path
p=Path({path!r})
if not p.is_file():
 print(json.dumps({{'exists':False}}))
else:
 b=p.read_bytes()
 print(json.dumps({{'exists':True,'sha256':hashlib.sha256(b).hexdigest(),
                    'value':json.loads(b.decode('utf-8-sig'))}},ensure_ascii=False))
"""
        return self.code(port, code)

    def _test_source_ready(self):
        if self.state.get("test_release"):
            return True
        source = self.plan["raw_inputs"]["test"]
        code = f"""
import hashlib,json
from pathlib import Path
record_path=Path({source['record_path']!r})
manifest_path=Path({source['manifest_path']!r})
if not record_path.is_file() or not manifest_path.is_file():
 print(json.dumps({{'ready':False,'reason':'test record or completion manifest absent'}}))
else:
 rb=record_path.read_bytes(); mb=manifest_path.read_bytes()
 record=json.loads(rb.decode('utf-8-sig')); manifest=json.loads(mb.decode('utf-8-sig'))
 selected=record.get('source',{{}}).get('selected_times',[])
 selection_sha=hashlib.sha256(json.dumps(selected,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 files=record.get('files',[])
 files_ok=len(files)==100
 if files_ok:
  for item in files:
   path=Path(item.get('path',''))
   if not path.resolve().is_relative_to(Path({source['path']!r}).resolve()) or not path.is_file(): files_ok=False;break
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
   if h.hexdigest()!=item.get('sha256'):files_ok=False;break
 print(json.dumps({{'ready':True,'record_sha256':hashlib.sha256(rb).hexdigest(),
   'manifest_sha256':hashlib.sha256(mb).hexdigest(),'record':record,
   'manifest':manifest,'selection_sha256':selection_sha,'scene_files_verified':files_ok}},ensure_ascii=False))
"""
        observed = self.code(20418, code, timeout=60)
        if not observed.get("ready"):
            self.state["observed"]["test_source"] = {"observed_utc": utcnow(), **observed}
            return False
        record, manifest = observed["record"], observed["manifest"]
        good = (record.get("path") == source["path"] and record.get("count") == 100 and
                manifest.get("status") == "prepared_not_imported_or_evaluated" and
                manifest.get("count") == 100 and
                manifest.get("staged_record_sha256") == observed["record_sha256"] and
                observed["selection_sha256"] == source["selection_sha256"] and
                observed["scene_files_verified"] is True)
        if source.get("record_sha256") and observed["record_sha256"] != source["record_sha256"]:
            good = False
        if source.get("manifest_sha256") and observed["manifest_sha256"] != source["manifest_sha256"]:
            good = False
        if source.get("date_utc") and manifest.get("date_utc") != source["date_utc"]:
            good = False
        if not good:
            raise Attention("新Jan独立考核完成记录、100场景、选择SHA或逐文件SHA未通过")
        self.state["test_release"] = {"observed_utc": utcnow(),
                                       "record_sha256": observed["record_sha256"],
                                       "manifest_sha256": observed["manifest_sha256"],
                                       "selection_sha256": observed["selection_sha256"],
                                       "scene_count": 100}
        return True

    def _s2_released(self):
        release = self.plan["s2_release"]
        if release.get("allow_file") is not None:
            raise Attention("S2手工放行旁路未获本轮计划支持")
        manifest_path = release["controller_manifest"]
        code = f"""
import json,urllib.request
from pathlib import Path
p=Path({manifest_path!r})
if not p.is_file():
 print(json.dumps({{'ready':False,'reason':'S2 controller manifest absent'}}))
else:
 manifest=json.loads(p.read_text(encoding='utf-8-sig'))
 jobs={{}}
 for key in {list(S2_SCA_JOBS)!r}:
  entry=manifest.get('jobs',{{}}).get(key,{{}})
  status_path=Path(entry.get('status_path') or '')
  status=json.loads(status_path.read_text(encoding='utf-8-sig')) if status_path.is_file() else {{}}
  jobs[key]={{'run_id':entry.get('run_id'),'manifest_stage':entry.get('stage'),
             'status_stage':status.get('stage'),'status_path':str(status_path)}}
 try:
  with urllib.request.urlopen('http://127.0.0.1:8781/api/runs',timeout=10) as response:
   runs=json.load(response).get('runs',[])
  live=[{{'run_id':r.get('run_id'),'stage':r.get('stage')}} for r in runs
        if r.get('stage') in {sorted(ACTIVE)!r}]
  error=None
 except Exception as exc:
  live=[];error=str(exc)
 print(json.dumps({{'ready':True,'phase':manifest.get('phase'),'jobs':jobs,
                    'live':live,'api_error':error}},ensure_ascii=False))
"""
        observed = self.code(20424, code)
        self.state["observed"]["s2_release"] = {"observed_utc": utcnow(), **observed}
        if not observed.get("ready"):
            return False
        if observed.get("api_error"):
            raise Attention("S2-P17工作台8781不可用，不能确认无活动任务：" + observed["api_error"])
        if observed.get("phase") != "awaiting_report_and_archive" or observed["live"]:
            return False
        jobs = observed["jobs"]
        if any(jobs[key]["manifest_stage"] != "completed" or
               jobs[key]["status_stage"] != "completed" or
               not jobs[key]["run_id"] for key in S2_SCA_JOBS):
            return False
        receipt = {"observed_utc": utcnow(), "mode": "S2_controller_terminal_and_live_API",
                   "phase": observed["phase"], "jobs": jobs, "active_runs": []}
        self.state["s2_release"] = receipt
        proof = Path(self.plan["local_output_root"]) / "ops" / "s2_20424_release.json"
        remote.save(proof, receipt)
        return True

    def _audit_context(self, mode, port, checkpoint_sha=None):
        methods = {}
        if mode == "development":
            labels = [f"dev_{route}" for route in ("R0", "R1", "R2", "R3")]
        elif mode == "formal":
            labels = [f"formal_{route}" for route in ("R0", "R1", "R2", "R3", "P1")]
        else:
            labels = [f"dev_{route}" for route in ("R0", "R1", "R2", "R3")]
        for label in labels:
            route = label.rsplit("_", 1)[1]
            methods[route] = f"{self.archive[20418 if mode != 'formal' else 20424]}/outputs/workbench/jobs/{self.state['jobs'][label]}"
        if mode == "mechanism":
            methods["P1"] = f"{self.archive[20418]}/outputs/workbench/jobs/{self.state['jobs']['train_P1']}"
        base = f"{self.archive[port]}/ops/audits"
        context = {"mode": mode, "plan_sha256": self.plan_sha256,
                   "science_commit": self.plan["science_commit"],
                   "expected_scenes": 100 if mode == "formal" else 20,
                   "methods": methods, "criteria": self.plan["gates"],
                   "training_job_dir": methods["P1"] if mode == "mechanism" else None,
                   "output_path": f"{base}/{mode}_audit.json"}
        if mode == "mechanism":
            context.update(run_id=self.state["jobs"]["train_P1"], env_step=65536,
                           checkpoint_sha256=checkpoint_sha)
        return context, f"{base}/{mode}_context.json"

    def _write_audit_context(self, port, context, path):
        payload = base64.b64encode(json.dumps(context, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":")).encode()).decode()
        code = f"""
import base64,json,os
from pathlib import Path
p=Path({path!r});p.parent.mkdir(parents=True,exist_ok=True)
b=base64.b64decode({payload!r})
if p.is_file() and p.read_bytes()!=b:raise RuntimeError('Frozen audit context differs')
if not p.is_file():
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(b);os.replace(tmp,p)
print(json.dumps({{'path':str(p),'bytes':len(b)}}))
"""
        self.code(port, code, retry=1)

    def _audit_process(self, mode, port, argv, context_path, context_sha, script_sha):
        base = f"{self.archive[port]}/ops/audits/{mode}"
        parameters = {"argv": argv, "argv_sha256": sha_json(argv),
                      "context_path": context_path, "context_sha256": context_sha,
                      "script_sha256": script_sha, "script_path": argv[1],
                      "source_root": (None if mode == "mechanism" else
                                      f"{self.archive[port]}/outputs/workbench/source_snapshots/{self.plan['science_commit']}"),
                      "science_commit": self.plan["science_commit"], "port": port,
                      "archive_root": self.archive[port],
                      "python": remote.DQN, "launch": base + "_launch.json",
                      "exit": base + "_exit.json", "child": base + "_child.json",
                      "preflight": base + "_preflight.json",
                      "stdout": base + "_stdout.log", "stderr": base + "_stderr.log",
                      "wrapper_stdout": base + "_wrapper_stdout.log",
                      "wrapper_stderr": base + "_wrapper_stderr.log"}
        payload = base64.b64encode(json.dumps(parameters, ensure_ascii=False,
                                              sort_keys=True).encode()).decode()
        wrapper = "import base64,hashlib,json,os,subprocess,sys,traceback\n" \
                  "from datetime import datetime,timezone\nfrom pathlib import Path\n" \
                  f"P=json.loads(base64.b64decode({payload!r}))\n" \
                  "def save(path,value):\n p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');" \
                  "tmp.write_text(json.dumps(value,sort_keys=True),encoding='utf-8');os.replace(tmp,p)\n" \
                  "def birth(pid):\n" \
                  " try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]\n" \
                  " except Exception:return None\n" \
                  "def sha(path):\n" \
                  " h=hashlib.sha256()\n" \
                  " with Path(path).open('rb') as stream:\n" \
                  "  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)\n" \
                  " return h.hexdigest()\n" \
                  "def verify():\n" \
                  " context=json.loads(Path(P['context_path']).read_text(encoding='utf-8'))\n" \
                  " if context['mode']=='mechanism':\n" \
                  "  job=json.loads((Path(context['training_job_dir'])/'job.json').read_text(encoding='utf-8'))\n" \
                  "  root=Path(job['source_root']).resolve()\n" \
                  "  if job['git_commit']!=P['science_commit']:raise RuntimeError('mechanism job commit mismatch')\n" \
                  " else:root=Path(P['source_root']).resolve()\n" \
                  " manifest_path=root/'source_manifest.json'\n" \
                  " manifest=json.loads(manifest_path.read_text(encoding='utf-8'))\n" \
                  " if manifest.get('git_commit')!=P['science_commit']:raise RuntimeError('frozen commit mismatch')\n" \
                  " files=manifest.get('files')\n" \
                  " if not isinstance(files,dict) or not files:raise RuntimeError('empty frozen source manifest')\n" \
                  " for name,expected in files.items():\n" \
                  "  path=(root/name).resolve()\n" \
                  "  if not path.is_relative_to(root) or not path.is_file() or sha(path)!=expected:\n" \
                  "   raise RuntimeError('frozen source checksum mismatch: '+name)\n" \
                  " expected_prefix=Path(P['python']).parent.parent.resolve()\n" \
                  " if Path(sys.prefix).resolve()!=expected_prefix:raise RuntimeError('audit not under DQN prefix')\n" \
                  " env=dict(os.environ);env['PYTHONPATH']=str(root/'src')\n" \
                  " env['SATELLITE_RL_SSH_PORT']=str(P['port'])\n" \
                  " env['SATELLITE_RL_OUTPUT_ROOT']=P['archive_root']\n" \
                  " for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1';os.environ[key]='1'\n" \
                  " os.environ['PYTHONPATH']=env['PYTHONPATH']\n" \
                  " sys.path.insert(0,env['PYTHONPATH'])\n" \
                  " import satellite_rl,torch\n" \
                  " package=Path(satellite_rl.__file__).resolve()\n" \
                  " if not package.is_relative_to((root/'src').resolve()):raise RuntimeError('audit imports nonfrozen satellite_rl')\n" \
                  " if torch.version.cuda is None or not torch.cuda.is_available() or '4090 D' not in torch.cuda.get_device_name(0):\n" \
                  "  raise RuntimeError('audit DQN/CUDA/4090D unavailable')\n" \
                  " return root,env,{'source_root':str(root),'git_commit':P['science_commit']," \
                  "'source_manifest_sha256':sha(manifest_path),'source_file_count':len(files)," \
                  "'python_prefix':sys.prefix,'package_file':str(package),'torch':torch.__version__," \
                  "'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)," \
                  "'ssh_port':P['port'],'threads':1,'checked_utc':datetime.now(timezone.utc).isoformat()}\n" \
                  "try:\n" \
                  " root,env,preflight=verify();save(P['preflight'],preflight);preflight_sha=sha(P['preflight'])\n" \
                  " with open(P['stdout'],'wb') as out,open(P['stderr'],'wb') as err:\n" \
                  "  child=subprocess.Popen(P['argv'],stdout=out,stderr=err,env=env,cwd=root,start_new_session=False)\n" \
                  "  child_birth=birth(child.pid)\n" \
                  "  save(P['child'],{'pid':child.pid,'birth_ticks':child_birth," \
                  "'context_sha256':P['context_sha256'],'script_sha256':P['script_sha256']," \
                  "'preflight_sha256':preflight_sha})\n" \
                  "  code=child.wait()\n" \
                  " save(P['exit'],{'returncode':code,'pid':child.pid,'birth_ticks':child_birth," \
                  "'context_sha256':P['context_sha256'],'script_sha256':P['script_sha256']," \
                  "'preflight_path':P['preflight'],'preflight_sha256':preflight_sha," \
                  "'finished_utc':datetime.now(timezone.utc).isoformat()})\n" \
                  "except BaseException as exc:\n" \
                  " save(P['exit'],{'returncode':-1,'error':str(exc),'traceback':traceback.format_exc()[-3000:]," \
                  "'context_sha256':P['context_sha256'],'script_sha256':P['script_sha256']})\n"
        encoded_wrapper = base64.b64encode(wrapper.encode()).decode()
        code = "import base64,hashlib,json,os,subprocess,time\nfrom pathlib import Path\n" + \
               f"P=json.loads(base64.b64decode({payload!r}))\n" + \
               f"WRAPPER={encoded_wrapper!r}\n" + \
               "def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n" + \
               "def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))\n" + \
               "def birth(pid):\n" + \
               " try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]\n" + \
               " except Exception:return None\n" + \
               "launch=Path(P['launch']);launch.parent.mkdir(parents=True,exist_ok=True)\n" + \
               "if sha(P['context_path'])!=P['context_sha256'] or sha(P['script_path'])!=P['script_sha256']:" + \
               "raise RuntimeError('frozen audit context or script SHA changed')\n" + \
               "if not launch.exists():\n" + \
               " try:fd=os.open(str(launch),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\n" + \
               " except FileExistsError:fd=None\n" + \
               " if fd is not None:\n" + \
               "  os.write(fd,json.dumps({'status':'starting','context_sha256':P['context_sha256']," + \
               "'script_sha256':P['script_sha256'],'argv_sha256':P['argv_sha256']}).encode());os.close(fd)\n" + \
               "  with open(P['wrapper_stdout'],'wb') as out,open(P['wrapper_stderr'],'wb') as err:\n" + \
               "   process=subprocess.Popen([P['python'],'-B','-c'," + \
               "'exec(__import__(\\\"base64\\\").b64decode(\\\"'+WRAPPER+'\\\"))']," + \
               "stdout=out,stderr=err,start_new_session=True,close_fds=True)\n" + \
               "  record={'status':'running','pid':process.pid,'birth_ticks':birth(process.pid)," + \
               "'context_sha256':P['context_sha256'],'script_sha256':P['script_sha256']," + \
               "'argv_sha256':P['argv_sha256']}\n" + \
               "  tmp=launch.with_suffix('.json.tmp');tmp.write_text(json.dumps(record),encoding='utf-8');os.replace(tmp,launch)\n" + \
               "record=read(launch)\n" + \
               "if any(record.get(k)!=P[k] for k in ('context_sha256','script_sha256','argv_sha256')):" + \
               "raise RuntimeError('audit launch belongs to a different frozen context')\n" + \
               "exit_path=Path(P['exit'])\n" + \
               "if exit_path.is_file():\n" + \
               " done=read(exit_path)\n" + \
               " print(json.dumps({'state':'completed','launch':record,'exit':done,'stdout':P['stdout'],'stderr':P['stderr']}))\n" + \
               "else:\n" + \
               " pid=record.get('pid');live=bool(pid and record.get('birth_ticks') and birth(pid)==record.get('birth_ticks'))\n" + \
               " if pid and not live:raise RuntimeError('audit wrapper exited without exit receipt')\n" + \
               " if not pid and time.time()-launch.stat().st_mtime>120:raise RuntimeError('audit launch stuck before PID receipt')\n" + \
               " print(json.dumps({'state':'running','launch':record,'live':live}))\n"
        return self.code(port, code, timeout=30, retry=1)

    def _run_audit(self, mode, port, checkpoint_sha=None):
        context, context_path = self._audit_context(mode, port, checkpoint_sha)
        self._write_audit_context(port, context, context_path)
        output = context["output_path"]
        command = self.plan["audit_commands"][mode]
        replacements = {"port": str(port), "archive_root": self.archive[port],
                        "job_dir": context.get("training_job_dir") or "",
                        "run_id": context.get("run_id", ""),
                        "checkpoint_sha256": checkpoint_sha or "",
                        "plan_sha256": self.plan_sha256,
                        "audit_context_path": context_path}
        argv = [arg.format(**replacements) for arg in command["argv"]]
        progress = self._audit_process(mode, port, argv, context_path, sha_json(context),
                                       command["script_sha256"])
        self.state["observed"]["audit_" + mode] = {"observed_utc": utcnow(), **progress}
        if progress["state"] == "running":
            return None
        finished = progress["exit"]
        if (finished.get("returncode") != 0 or
                finished.get("context_sha256") != sha_json(context) or
                finished.get("script_sha256") != command["script_sha256"]):
            raise Attention(f"{mode}独立审计后台异常退出；请核stdout/stderr及exit回执")
        preflight = self._read_remote_json(port, finished.get("preflight_path", ""))
        expected_root = f"{self.archive[port]}/outputs/workbench/source_snapshots/{self.plan['science_commit']}"
        info = preflight.get("value") or {}
        if (not preflight.get("exists") or preflight.get("sha256") != finished.get("preflight_sha256") or
                info.get("source_root") != expected_root or
                info.get("git_commit") != self.plan["science_commit"] or
                info.get("ssh_port") != port or
                "4090 D" not in str(info.get("gpu")) or
                not str(info.get("package_file", "")).startswith(expected_root + "/src/")):
            raise Attention(f"{mode}审计DQN/CUDA/冻结源码preflight回执无效")
        observed = self._read_remote_json(port, output)
        audit = observed.get("value") or {}
        if (not observed.get("exists") or audit.get("passed") is not True or
                audit.get("mode") != mode or audit.get("plan_sha256") != self.plan_sha256):
            raise Attention(f"{mode}独立审计文件未通过或计划身份不一致")
        if mode == "mechanism" and (audit.get("run_id") != self.state["jobs"]["train_P1"] or
                                    audit.get("env_step") != 65536 or
                                    audit.get("checkpoint_sha256") != checkpoint_sha):
            raise Attention("65k机制审计未绑定当前任务/交互步/断点SHA")
        receipt = {"mode": mode, "port": port, "passed": True, "audit_path": output,
                   "audit_sha256": observed["sha256"], "context_path": context_path,
                   "preflight_path": finished["preflight_path"],
                   "preflight_sha256": finished["preflight_sha256"],
                   "source_root": info["source_root"], "gpu": info["gpu"],
                   "observed_utc": utcnow()}
        self.state["audits"][mode] = receipt
        return receipt

    def _mechanism_gate(self, run_id, detail):
        status = detail["status"]
        if status.get("env_step") != 65536 or not re.fullmatch(r"[0-9a-f]{64}", status.get("checkpoint_sha256", "")):
            raise Attention("P1机制闸门缺少65536完整步数或断点SHA")
        sha = status["checkpoint_sha256"]
        code = f"""
import hashlib,json
from pathlib import Path
p=Path({self.archive[20418]!r})/'outputs/workbench/jobs'/{run_id!r}/'checkpoint.pt'
h=hashlib.sha256()
with p.open('rb') as stream:
 for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
print(json.dumps({{'sha256':h.hexdigest()}}))
"""
        if self.code(20418, code)["sha256"] != sha:
            raise Attention("65k机制闸门断点文件SHA与任务状态不一致")
        audit = self._run_audit("mechanism", 20418, sha)
        if audit is None:
            return False
        process = self._worker_process(20418, run_id)
        self.state["observed"]["mechanism_worker_exit"] = {"observed_utc": utcnow(), **process}
        if process["state"] == "same_process_alive":
            return False
        if process["state"] != "exited":
            raise Attention("65k旧worker PID被复用或出生信息不可信；不终止进程、不提交恢复")
        self.api(20418, f"coverage/runs/{run_id}/mechanism-gate", {
            "audit_path": audit["audit_path"], "audit_sha256": audit["audit_sha256"]}, retry=1)
        self.api(20418, f"runs/{run_id}/resume", {}, retry=1)
        self.state["observed"]["mechanism_resume"] = {"observed_utc": utcnow(),
                                                          "run_id": run_id, "checkpoint_sha256": sha,
                                                          "audit_sha256": audit["audit_sha256"]}
        return False

    def _worker_process(self, port, run_id):
        directory = f"{self.archive[port]}/outputs/workbench/jobs/{run_id}"
        code = f"""
import json,os
from pathlib import Path
root=Path({directory!r})
launch=json.loads((root/'launch.json').read_text(encoding='utf-8-sig')) if (root/'launch.json').is_file() else {{}}
status=json.loads((root/'status.json').read_text(encoding='utf-8-sig')) if (root/'status.json').is_file() else {{}}
pid=launch.get('pid'); expected=launch.get('process_start_time')
def actual(pid):
 try:
  stat=Path('/proc/%d/stat'%int(pid)).read_text()
  ticks=int(stat.rsplit(')',1)[1].split()[19])
  boot=next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
  return boot+ticks/os.sysconf('SC_CLK_TCK')
 except Exception:return None
observed=actual(pid) if pid else None
if not pid or expected is None or status.get('pid')!=pid or status.get('stage')!='stopped':
 state='identity_unverified'
elif observed is None:
 state='exited'
elif abs(float(observed)-float(expected))<0.01:
 state='same_process_alive'
else:
 state='pid_reused'
print(json.dumps({{'state':state,'pid':pid,'expected_start_time':expected,
                  'actual_start_time':observed,'status_stage':status.get('stage'),
                  'status_pid':status.get('pid')}}))
"""
        return self.code(port, code)

    def _import_model_20424(self):
        key = "P1_model_20424"
        source_run_id = self.state["jobs"]["train_P1"]
        models = self.api(20424, "coverage/models")["models"]
        model = next((row for row in models if row.get("id") == self.state["imports"].get(key)), None)
        if model is None:
            matches = [row for row in models if row.get("source_run_id") == source_run_id and
                       row.get("source_target") == "u2024110224@10.160.17.102:20418"]
            if len(matches) > 1:
                raise Attention("跨实例P1模型导入记录不唯一")
            model = matches[0] if matches else None
        if model is None:
            model = self.api(20424, "coverage/reconstruction/import-model", {
                "source_target": "20418", "source_run_id": source_run_id}, retry=1)
        if (model.get("source_run_id") != source_run_id or
                model.get("source_target") != "u2024110224@10.160.17.102:20418" or
                not re.fullmatch(r"[0-9a-f]{64}", model.get("sha256", ""))):
            raise Attention("P1跨实例模型原任务、原实例或权重SHA不一致")
        self.state["imports"][key] = model["id"]
        self.state["observed"][key] = {"observed_utc": utcnow(), "sha256": model["sha256"],
                                       "source_run_id": source_run_id}
        return True

    def _lane_20418(self):
        port = 20418
        self.state["phase"] = "preparing_development"
        for role in ("train", "validation"):
            if not self._dataset(role, port):
                return
        if not self._job("prepare_validation", port, "prepare", "evaluate", "validation"):
            return
        for route in ("R0", "R1", "R2", "R3"):
            if not self._job(f"dev_{route}", port, route, "evaluate", "validation"):
                return
        if "development" not in self.state["audits"]:
            self.state["phase"] = "development_audit"
            if self._run_audit("development", port) is None:
                return
            self.save()
        if not self._job("prepare_train", port, "prepare", "evaluate", "train"):
            self.state["phase"] = "preparing_train"
            return
        if not (self.state.get("test_release") and self.state["datasets"].get("20424:test") and
                self.state["jobs"].get("prepare_test") and
                self.state["observed"].get("prepare_test", {}).get("stage") == "completed"):
            self.state["phase"] = "awaiting_frozen_independent_test"
            return
        self.state["phase"] = "training"
        if not self._job("train_P1", port, "P1", "train"):
            return
        self.state["phase"] = "training_completed"

    def _lane_20424(self):
        port = 20424
        if not self._s2_released():
            self.state["observed"]["s2_waiting_utc"] = utcnow()
            return
        if not self._test_source_ready():
            return
        self._dataset("test", port)
        if not self._job("prepare_test", port, "prepare", "evaluate", "test"):
            return
        if ("development" not in self.state["audits"] or
                self.state["observed"].get("prepare_train", {}).get("stage") != "completed"):
            return
        for route in ("R0", "R1", "R2", "R3"):
            if not self._job(f"formal_{route}", port, route, "evaluate", "test"):
                return
        if self.state["observed"].get("train_P1", {}).get("stage") != "completed":
            return
        self._import_model_20424()
        if not self._job("formal_P1", port, "P1", "evaluate", "test"):
            return
        if "formal" not in self.state["audits"]:
            self.state["phase"] = "formal_audit"
            if self._run_audit("formal", port) is None:
                return
        self.state["phase"] = "awaiting_final_report"

    def tick(self):
        if self.state.get("attention"):
            raise Attention("控制器有未处理异常；先核现场再明确清除attention")
        if self.state.get("phase") == "awaiting_final_report":
            return self.state
        try:
            self._lane_20418()
            self.save()
            self._lane_20424()
            self.save()
        except Exception as error:
            self.state["phase"] = "attention_required"
            self.state["attention"] = {"observed_utc": utcnow(), "error": str(error),
                                       "traceback": traceback.format_exc()[-12000:]}
            self.save()
            raise
        return self.state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--daemon", action="store_true")
    args = parser.parse_args(argv)
    plan, plan_sha = load_plan(args.plan)
    with single_controller_lock(args.state.with_suffix(args.state.suffix + ".lock")):
        controller = Controller(plan, plan_sha, args.state)
        while True:
            controller.tick()
            print(json.dumps({"observed_utc": utcnow(), "phase": controller.state["phase"],
                              "jobs": controller.state["jobs"],
                              "attention": controller.state["attention"]}, ensure_ascii=False), flush=True)
            if args.once or controller.state["phase"] == "awaiting_final_report":
                return 0
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
