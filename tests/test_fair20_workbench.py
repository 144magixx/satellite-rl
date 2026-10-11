"""Typed API, durable queue, immutable inputs and finite F20 bookkeeping."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.fair20_service import (
    Fair20Application, default_config, validate_config, select_candidate,
    freeze_fair20_sources,
)
from satellite_rl.workbench.storage import Store, digest, read_json
from satellite_rl.workbench.fair20_runner import validate_datasets, save_torch, _train, _matrix, full_b_case, Fair20RecoveryAttention, FrozenActionPolicy


class DummyScheduler:
    def __init__(self):
        self.lock = threading.RLock()


class DummyApplication:
    def __init__(self, store):
        self.store, self.scheduler = store, DummyScheduler()

    @staticmethod
    def observed_status(status):
        return dict(status)

    def run_detail(self, identifier):
        directory = self.store.job_dir(identifier)
        return {"job": read_json(directory / "job.json"), "status": read_json(directory / "status.json")}


class Fair20WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "archive", source_root=self.root)
        snapshot = self.root / "frozen"
        snapshot.mkdir()
        self.store.freeze_source = lambda: (snapshot, "a"*40)
        self.app = Fair20Application(DummyApplication(self.store))

    def tearDown(self):
        self.temp.cleanup()

    def task(self, identifier, mode, candidate="A", completed=True, algorithm="sac"):
        directory = self.store.job_dir(identifier)
        directory.mkdir()
        job = {"run_id": identifier, "experiment_id": "S2-F20", "mode": mode, "algorithm": algorithm,
               "prepare_run_id": "prepared", "kind": "train", "model_dir": str(self.root / identifier)}
        config = default_config(algorithm, candidate, mode)
        atomic_json(directory / "job.json", job)
        atomic_json(directory / "config.json", config)
        atomic_json(directory / "status.json", {"stage": "completed" if completed else "failed", "env_step": config["training"]["total_steps"]})
        return directory

    def test_finite_common_config_and_hyperparameters(self):
        for algorithm in ("sac", "ppo", "dqn"):
            cfg = default_config(algorithm)
            self.assertEqual(validate_config(cfg), cfg)
            self.assertEqual(cfg["training"]["total_steps"], 400000)
            self.assertEqual(cfg["training"]["checkpoint_nodes"], [100000,200000,300000,400000])
            self.assertEqual(default_config(algorithm, mode="shorttune")["training"]["total_steps"], 32768)
        self.assertEqual(default_config("sac","B")["learning"]["alpha"], .05)
        self.assertEqual(default_config("ppo","B")["learning"]["lr"], 3e-4)

    def test_reject_contract_drift(self):
        for section, key, value in (("training", "total_steps", 400001), ("network", "heads", 8),
                                    ("environment", "power_rule", "learned"), ("training", "seed", 51002)):
            cfg = default_config(); cfg[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        with self.assertRaises(ValueError):
            default_config("random", mode="train")

    def test_candidate_selection_predefined_ties(self):
        a = {"candidate":"A", "SI":.5, "SGM":.4}
        b = {"candidate":"B", "SI":.502, "SGM":.1}
        self.assertEqual(select_candidate([a,b]), "B")
        b.update(SI=.5005, SGM=.5)
        self.assertEqual(select_candidate([a,b]), "B")
        b.update(SGM=.4)
        self.assertEqual(select_candidate([b,a]), "A")

    def test_tune_selection_requires_real_completed_budget(self):
        prepared = self.task("prepared", "prepare", algorithm="prepare")
        atomic_json(prepared / "results.json", {"datasets": {}})
        with self.assertRaises(ValueError):
            self.app.submit({"algorithm":"sac", "prepare_run_id":"prepared", "selection_run_ids":[]}, "train")

    def test_freeze_checks_sha_and_preserves_source(self):
        source = self.root / "scene.npz"; source.write_bytes(b"actual-test-source")
        detail = self.root / "detail.jsonl.gz"; detail.write_bytes(b"real-detail-reference")
        ref = {"path":str(source), "sha256":digest(source), "mother_scene_id":"m1",
               "observation_detail":{"path":str(detail),"sha256":digest(detail)}}
        extra = {"datasets":{"train":{"files":[ref,deepcopy(ref)],"source":{"partition_role":"train"}}}}
        frozen = freeze_fair20_sources(extra, self.root / "input")
        self.assertNotEqual(frozen["datasets"]["train"]["files"][0]["path"], str(source))
        self.assertEqual(extra["datasets"]["train"]["files"][0]["path"], str(source))
        refs = frozen["datasets"]["train"]["files"]
        self.assertEqual(refs[0]["observation_detail"]["path"],refs[1]["observation_detail"]["path"])
        source.write_bytes(b"changed")
        self.assertEqual(Path(refs[0]["path"]).read_bytes(),b"actual-test-source")
        with self.assertRaises(ValueError):
            freeze_fair20_sources(extra, self.root / "other")

    def test_sourceprep_freezes_record_paths(self):
        scene = self.root / "scene.npz"; scene.write_bytes(b"s")
        record_path = self.root / "record.json"
        record = {"files":[{"path":str(scene),"sha256":digest(scene)}], "source":{"demand_model":{"x":1}}}
        atomic_json(record_path, record)
        archive = self.root / "day.tar"; archive.write_bytes(b"a")
        extra = {"datasets":{"train":record}, "source_files":[{"path":str(record_path),"sha256":digest(record_path)},
                  {"path":str(archive),"sha256":digest(archive)}],
                 "sourceprep":{"train_record_path":str(record_path),"train_record_sha256":digest(record_path),
                               "source_assets":[{"path":str(archive)}]}}
        frozen = freeze_fair20_sources(extra,self.root/"frozen_inputs")
        copied = read_json(frozen["sourceprep"]["train_record_path"])
        self.assertTrue(Path(copied["files"][0]["path"]).is_relative_to(self.root/"frozen_inputs"))
        self.assertEqual(digest(frozen["sourceprep"]["train_record_path"]),frozen["sourceprep"]["train_record_sha256"])

    def test_queued_resource_job_is_published_with_semantic_kind(self):
        with patch.dict("os.environ", {"SATELLITE_RL_F20_REQUIRE_ADMISSION":"1"}):
            job = self.app.submit({"request_id":"unit-prepared","datasets":{}},"prepare")
        self.assertEqual(job["kind"],"prepare")
        self.assertEqual(job["research_stage"],"fair20")
        self.assertEqual(read_json(self.store.job_dir(job["run_id"])/"status.json")["stage"],"queued_resources")
        with patch.dict("os.environ", {"SATELLITE_RL_F20_REQUIRE_ADMISSION":"1"}):
            repeat = self.app.submit({"request_id":"unit-prepared","datasets":{}},"prepare")
        self.assertEqual(repeat["run_id"],job["run_id"])
        with self.assertRaises(ValueError):
            self.app.submit({"request_id":"unit-prepared","datasets":{"new":{}}},"prepare")

    def test_admission_is_sha_and_run_bound(self):
        with patch.dict("os.environ", {"SATELLITE_RL_F20_REQUIRE_ADMISSION":"1"}):
            job = self.app.submit({"request_id":"unit-admission","datasets":{}},"prepare")
        proof = self.root/"admission.json"
        receipt=self.root/"release.json";atomic_json(receipt,{"released":True,"released_ssh_port":20474,
            "old_coordinator_excluded_ports":[20474],"persistent_coordinator_exclusion_verified":True,
            "coordinator_evidence":{"persistent":"real-test-evidence"},"active_jobs":[],"pending_jobs":[],"future_lane_reserved":False})
        atomic_json(proof,{"admitted":True,"run_id":job["run_id"], "release_receipt_path":str(receipt),
                          "release_receipt_sha256":digest(receipt), "release":{"port":20474,"scientific_commit":job["git_commit"],
                             "gpu_uuid":"GPU-test", "old_lane_all_work_terminal":True,"old_future_training_reserved":False},
                          "resource_observation":{
                              "checked_at_utc":datetime.now(timezone.utc).isoformat(),"port":20474,
                              "scientific_commit":job["git_commit"],"worker_free":True,"gpu_free":True,
                              "gpu":"GPU-test,4090D,0,24000,0,40", "gpu_query_exit":0,"compute_query_exit":0,
                              "compute_apps":"", "processes":[],"old_tasks":{"runs":[]},"cpu_affinity":[0],
                              "cgroup":{"cpu.max":"100000 100000"},"disk_free":20*1024**3,"memory":{"MemAvailable":10*1024**3}}})
        status = self.app.api("POST",["admit"],{"run_id":job["run_id"],"proof_path":str(proof),"proof_sha256":digest(proof)})
        self.assertEqual(status["stage"],"queued")
        valid=read_json(proof)
        contradicted=deepcopy(valid);contradicted["resource_observation"]["compute_apps"]="999,GPU-test,100"
        atomic_json(proof,contradicted)
        with self.assertRaises(ValueError):
            self.app.api("POST",["admit"],{"run_id":job["run_id"],"proof_path":str(proof),"proof_sha256":digest(proof)})
        atomic_json(proof,valid)
        false_release=read_json(receipt);false_release["released"]=False;atomic_json(receipt,false_release)
        valid["release_receipt_sha256"]=digest(receipt);atomic_json(proof,valid)
        with self.assertRaises(ValueError):
            self.app.api("POST",["admit"],{"run_id":job["run_id"],"proof_path":str(proof),"proof_sha256":digest(proof)})
        atomic_json(proof,{"admitted":True,"run_id":"wrong"})
        with self.assertRaises(ValueError):
            self.app.api("POST",["admit"],{"run_id":job["run_id"],"proof_path":str(proof),"proof_sha256":digest(proof)})

    def test_scheduler_never_creates_worker_without_admission(self):
        from satellite_rl.workbench.runtime import Scheduler
        with patch.dict("os.environ", {"SATELLITE_RL_F20_REQUIRE_ADMISSION":"1"}):
            job = self.app.submit({"request_id":"unit-worker-guard","datasets":{}},"prepare")
        directory=self.store.job_dir(job["run_id"])
        status=read_json(directory/"status.json")
        scheduler=Scheduler(self.store)
        with patch("satellite_rl.workbench.runtime.subprocess.Popen") as spawn:
            scheduler._launch(job,status)
            spawn.assert_not_called()
        self.assertEqual(read_json(directory/"status.json")["stage"],"queued_resources")

    def test_detail_reads_actual_results(self):
        directory = self.task("short", "shorttune")
        result = {"validation_summary":{"SI":.3,"SGM":.2}}
        atomic_json(directory/"results.json",result)
        self.assertEqual(self.app.api("GET",["tasks","short"],{})["result"],result)

    def test_resume_fails_without_full_checkpoint_and_keeps_failure(self):
        directory = self.task("broken","train",completed=False)
        with self.assertRaises(ValueError):
            self.app.resume("broken")
        (directory/"checkpoint.pt").write_bytes(b"opaque-complete-state-test")
        atomic_json(directory/"failure.json",{"original":"failure"})
        with patch("satellite_rl.workbench.runtime.pid_alive",return_value=False):
            status = self.app.resume("broken")
        self.assertEqual(status["stage"],"queued")
        self.assertEqual(read_json(directory/"failure.json"),{"original":"failure"})
        self.assertTrue(list((directory/"recovery_history").glob("*/failure.json")))

    def test_engineering_dataset_leak_rejected(self):
        records = {role:{"files":[{"path":"test", "sha256":role,"mother_scene_id":role}],
                         "source":{"partition_role":role}} for role in ("train","validation","test")}
        validate_datasets(records,engineering=True)
        records["test"]["files"][0]["mother_scene_id"] = "train"
        with self.assertRaises(ValueError):
            validate_datasets(records,engineering=True)

    def test_atomic_torch_checkpoint_round_trip(self):
        import torch
        path=self.root/"checkpoint.pt"
        save_torch(path,{"step":123,"tensor":torch.tensor([1,2,3])})
        value=torch.load(path,weights_only=False)
        self.assertEqual(value["step"],123)
        self.assertFalse(path.with_name("checkpoint.pt.tmp").exists())

    def test_B_recovery_reuses_solvers_and_accumulates_prior_time(self):
        import numpy as np
        from types import SimpleNamespace
        from satellite_rl.workbench.fair20_service import fingerprint
        initial={"scene":{}, "beams":[{"power_w":1500.,"rate_bps":50.,"required_bps":100.},
                                       {"power_w":1500.,"rate_bps":50.,"required_bps":100.}],
                 "metrics":{}, "timing":{"direct_total_seconds":3.}}
        solver={"power_w":[1500.,1500.],"evaluations":3,"elapsed_seconds":5.,"termination_reason":"completed"}
        journal=self.root/"optimizer.json"
        atomic_json(journal,{"phase":"B_complete","initial_sha256":fingerprint(initial),"A":solver,"B":solver,
                            "A_seconds":7.,"B_seconds":9.,"overhead_seconds":2.})
        ev=SimpleNamespace(capacity_bps=np.array([50.,50.]),beam_si=np.array([.5,.5]),beam_sgm=np.array([.6,.6]),si=.5,sgm=.6)
        problem=SimpleNamespace(demand_bps=np.array([100.,100.]),evaluate=lambda power:ev)
        def verify(snap):snap["metrics"].update(SI=.5,SGM=.6);return problem
        with patch("satellite_rl.workbench.fair20_runner.verify_snapshot",side_effect=verify),patch("satellite_rl.power_refinement.optimizer.optimize_power") as A,patch("satellite_rl.power_refinement.priority.optimize_priority") as B:
            result=full_b_case(initial,lambda:False,journal=journal)
        A.assert_not_called();B.assert_not_called()
        self.assertGreaterEqual(result["timing"]["refinement_total_seconds"],18.)
        self.assertGreaterEqual(result["timing"]["direct_plus_refine_total_seconds"],21.)
        self.assertTrue(result["timing"]["recovered_A"])
        self.assertEqual(read_json(journal)["phase"],"complete")

    def test_model_from_different_preparation_rejected(self):
        prepared=self.task("prepared","prepare",algorithm="prepare")
        datasets={role:{"files":[{"sha256":role,"mother_scene_id":role}]} for role in ("train","validation","test")}
        atomic_json(prepared/"results.json",{"datasets":datasets})
        model=self.task("model","train")
        job=read_json(model/"job.json");job["prepare_run_id"]="different";atomic_json(model/"job.json",job)
        with self.assertRaisesRegex(ValueError,"准备任务来源"):
            self.app.submit({"request_id":"cross-prep-rejected","algorithm":"sac","prepare_run_id":"prepared","model_run_id":"model"},"evaluate")

    def test_schema_contains_matching_arm_mode_candidates(self):
        schema=self.app.api("GET",["schema"],{})
        self.assertEqual(schema["configurations"]["shorttune/ppo/B"]["learning"]["lr"],3e-4)
        self.assertEqual(schema["configurations"]["evaluate/random/A"]["training"]["total_steps"],0)
        with self.assertRaises(ValueError):self.app.submit({"datasets":{}},"prepare")

    def test_matrix_interleaves_five_methods_once_and_resumes_without_new_cases(self):
        import numpy as np
        from types import SimpleNamespace
        class StubEnv:
            def __init__(self,scene):pass
            def reset(self):return {}
            def action_mask(self):return np.ones(1000,bool)
        agent=SimpleNamespace(act=lambda *args,**kwargs:(0,{}))
        refs=[{"path":"opaque", "sha256":str(i),"mother_scene_id":"m"+str(i),"beam_count":50,"load_multiplier":1.}
              for i in range(2)]
        job={"datasets":{"test":{"files":refs}},"models_by_algorithm":{arm:{"candidate":"A"} for arm in ("sac","ppo","dqn")},
             "method_order":{"methods":["sac","ppo","dqn","random","greedy"],
                "rule":"scene_index modulo5 rotates fixed method sequence","warmup_network_calls_per_model":4,"all_cases_once":True}}
        directory=self.root/"matrix";directory.mkdir()
        calls=[]
        def direct(ref,method,model,index,stop):
            calls.append((index,method))
            return {"scene_reference":ref,"method":method,"metrics":{"SI":.5,"SGM":.6},"timing":{"direct_total_seconds":3.}}
        def B(initial,*args,**kwargs):
            result=deepcopy(initial);result["timing"]["direct_plus_refine_total_seconds"]=5.;return result
        with patch("satellite_rl.fair_comparison.Fair20Env",StubEnv),patch("satellite_rl.workbench.fair20_runner.load_model",return_value=agent),patch("satellite_rl.workbench.fair20_runner.read_scene",return_value={}),patch("satellite_rl.workbench.fair20_runner.direct_case",side_effect=direct) as direct_call,patch("satellite_rl.workbench.fair20_runner.full_b_case",side_effect=B) as B_call,patch("satellite_rl.workbench.runtime.gpu_snapshot",return_value={"compute_pids":[]}),patch("satellite_rl.workbench.fair20_runner.torch.cuda.synchronize"):
            result=_matrix(directory,job,default_config("prepare",mode="matrix"),lambda **kw:None,lambda:False)
            continued=_matrix(directory,job,default_config("prepare",mode="matrix"),lambda **kw:None,lambda:False)
        self.assertEqual(direct_call.call_count,10);self.assertEqual(B_call.call_count,10)
        self.assertEqual(calls[:5],[(0,m) for m in ("sac","ppo","dqn","random","greedy")])
        self.assertEqual(calls[5:],[(1,m) for m in ("ppo","dqn","random","greedy","sac")])
        self.assertEqual(result["direct_count"],10);self.assertEqual(continued["full_B_count"],10)
        self.assertEqual({m:r["count"] for m,r in result["summary_by_method"].items()},dict.fromkeys(("sac","ppo","dqn","random","greedy"),2))

    def test_cgroup_v1_or_unexposed_limits_are_truthfully_recorded(self):
        from satellite_rl.workbench.fair20_guard import cgroup_inventory
        root=self.root/"cgroup";root.mkdir();proc=self.root/"proc_cgroup"
        proc.write_text("7:cpu,cpuacct:/docker/actual\n",encoding="utf-8")
        first=cgroup_inventory(root,proc)
        self.assertEqual(first["version"],"unexposed")
        self.assertFalse(first["limits_exposed"])
        self.assertTrue(first["attempted"])
        self.assertIn("not asserted",first["limitation"])
        (root/"cpu").mkdir();(root/"cpu"/"cpu.cfs_quota_us").write_text("100000")
        (root/"cpu"/"cpu.cfs_period_us").write_text("100000")
        second=cgroup_inventory(root,proc)
        self.assertEqual(second["version"],1)
        self.assertEqual(second["limits"]["cpu.cfs_quota_us"],"100000")
        self.assertFalse(second["timing_cpu_quota_control_verified"])

    def test_prepare_uses_server_side_sourceprep_id_and_compact_task_inventory(self):
        source=self.task("sourceprep","sourceprep",algorithm="prepare")
        scene=self.root/"actual-input.npz";scene.write_bytes(b"isolated typed-input fixture")
        datasets={"train":{"files":[{"path":str(scene),"sha256":digest(scene),"mother_scene_id":"m1"}],
                           "source":{"partition_role":"train"}}}
        atomic_json(source/"results.json",{"datasets":datasets})
        body={"request_id":"unit-prepare-source-id","sourceprep_run_id":"sourceprep"}
        job=self.app.submit(body,"prepare")
        self.assertEqual(job["sourceprep_run_id"],"sourceprep")
        self.assertNotIn("datasets",body)
        frozen=job["datasets"]["train"]["files"][0]
        self.assertEqual(Path(frozen["path"]).read_bytes(),scene.read_bytes())
        inventory=self.app.api("GET",["tasks"],{})
        self.assertTrue(all("datasets" not in row["job"] for row in inventory["tasks"]))

    def test_budget_tail_truncates_without_reset_or_extra_interactions(self):
        """A three-step orchestration stub, not a model training measurement."""
        import numpy as np
        from types import SimpleNamespace
        class StubEnv:
            def __init__(self, scene):self.index=0
            def reset(self):self.index=0;return {"x":0}
            def observation(self):return {"x":self.index}
            def action_mask(self):return np.ones(1000,bool)
            def step(self,action):
                self.index+=1
                return self.observation(),.1,False,{"actual_action":action,"requested_action":action}
            def state_dict(self):return {"index":self.index}
            def load_state_dict(self,value):self.index=value["index"]
        class StubAgent:
            def __init__(self):
                self.interactions=self.update_count=0;self.last_update={};self.models={};self.initialization_metadata={};self.truncations=[]
                self.policy=SimpleNamespace(state_dict=lambda:{},parameters=lambda:[])
            def act(self,obs,mask):return 0,{}
            def observe(self,*args,**kwargs):
                self.interactions+=1;self.truncations.append(kwargs["truncated"]);return {}
            def flush(self):return {}
            def state_dict(self):return {"interactions":self.interactions,"truncations":self.truncations}
            def load_state_dict(self,value):self.interactions=value["interactions"];self.truncations=value["truncations"]
        config=default_config("dqn",mode="engineering");config["training"]["total_steps"]=3
        directory=self.root/"stub_job";directory.mkdir();atomic_json(directory/"config.json",config)
        snapshot=self.root/"snapshot";snapshot.mkdir();atomic_json(snapshot/"source_manifest.json",{"files":{}})
        job={"run_id":"stub","source_root":str(snapshot),"git_commit":"0"*40,"model_dir":str(self.root/"stub_model"),
             "datasets":{"train":{"files":[{"path":"opaque","sha256":"input"}]}}}
        first=StubAgent()
        captured=[]
        ticks=iter(range(0,1000,6))
        with patch("satellite_rl.fair_comparison.Fair20Env",StubEnv),patch("satellite_rl.fair_comparison.make_agent",return_value=first),patch("satellite_rl.workbench.fair20_runner.read_scene",return_value={}),patch("satellite_rl.workbench.fair20_runner.time.monotonic",side_effect=lambda:next(ticks)):
            result=_train(directory,job,config,False,lambda **kw:captured.append(kw),lambda:False)
        self.assertEqual(result["env_step"],3)
        self.assertEqual(first.truncations,[False,False,True])
        self.assertEqual(result["episodes"],0)
        self.assertTrue(any(row.get("stage")=="training" and row.get("env_step")==1 and row.get("updates")==0 for row in captured))
        self.assertTrue(any(row.get("stage")=="training" and row.get("env_step")==2 and row.get("updates")==0 for row in captured))
        restored=StubAgent()
        with patch("satellite_rl.fair_comparison.Fair20Env",StubEnv),patch("satellite_rl.fair_comparison.make_agent",return_value=restored),patch("satellite_rl.workbench.fair20_runner.read_scene",return_value={}):
            continued=_train(directory,job,config,True,lambda **kw:None,lambda:False)
        self.assertEqual(continued["env_step"],3)
        self.assertEqual(restored.interactions,3)
        self.assertEqual(restored.truncations,[False,False,True])
        trace=directory/"traces.jsonl"
        with trace.open("a",encoding="utf-8") as stream:stream.write('{"env_step":4,"actual_action":0}\n')
        before=trace.read_bytes();blocked=StubAgent()
        with patch("satellite_rl.fair_comparison.Fair20Env",StubEnv),patch("satellite_rl.fair_comparison.make_agent",return_value=blocked),patch("satellite_rl.workbench.fair20_runner.read_scene",return_value={}):
            with self.assertRaises(Fair20RecoveryAttention):
                _train(directory,job,config,True,lambda **kw:None,lambda:False)
        self.assertEqual(blocked.interactions,0)
        self.assertEqual(trace.read_bytes(),before)
        proof=read_json(directory/"recovery_history"/"strict_budget_attempt_0000"/"budget_attention.json")
        self.assertEqual(proof["known_uncommitted_interactions"],1)
        self.assertTrue(proof["automatic_reexecution_forbidden"])

    def test_frozen_deployment_uses_only_one_legal_argmax_forward(self):
        import numpy as np
        import torch
        class StubNetwork(torch.nn.Module):
            def __init__(self,output):super().__init__();self.calls=0
            def forward(self,obs):
                self.calls+=1
                value=torch.zeros((1,1000));value[0,7]=2;value[0,9]=2;value[0,900]=99
                return value
        mask=np.zeros(1000,bool);mask[[7,9]]=True
        with patch("satellite_rl.fair_comparison.networks.TransformerNetwork",StubNetwork),patch("satellite_rl.fair_comparison.networks.tensor_observations",return_value={}):
            deployed=FrozenActionPolicy("ppo","B",{},device="cpu")
            action,_=deployed.act({},mask,deterministic=True)
        self.assertEqual(action,7)
        self.assertEqual(deployed.policy.calls,1)
        self.assertFalse(hasattr(deployed,"value"))
        self.assertFalse(hasattr(deployed,"q1"))


if __name__ == "__main__":
    unittest.main()
