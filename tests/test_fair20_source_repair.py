"""Local synthetic source-extension transactions; no training or solvers."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import Store,digest,read_json
from satellite_rl.workbench.fair20_service import default_config
from satellite_rl.workbench import fair20_source_repair as repair


class SourceRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.science=self.root/"science";self.science.mkdir()
        def file(name,text):
            path=self.science/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
        file("src/satellite_rl/workbench/fair20_runner.py","original validation predicate\n")
        file("src/satellite_rl/fair_comparison/algorithms.py","immutable learning\n")
        file("data/example.txt","immutable physical source\n");file("pyproject.toml","[project]\nname='synthetic'\n")
        self.git(self.science,"init");self.git(self.science,"add",".");self.git(self.science,"commit","-m","synthetic original")
        self.origin=self.git(self.science,"rev-parse","HEAD").decode().strip()
        self.execution=self.root/"execution";shutil.copytree(self.science,self.execution)
        path=self.execution/"src/satellite_rl/workbench/fair20_runner.py";path.write_text("strict verified direct fallback\n",encoding="utf-8")
        self.git(self.execution,"add",".");self.git(self.execution,"commit","-m","synthetic validation repair")
        self.execution_commit=self.git(self.execution,"rev-parse","HEAD").decode().strip()
        self.patch=patch.object(repair,"ORIGIN_COMMIT",self.origin);self.patch.start()
        self.store=Store(self.root/"archive",source_root=self.science)
        self.scheduler=type("Scheduler",(),{"store":self.store,"lock":threading.RLock()})()
        self.snapshot,_=self.store.freeze_source()
        proof=self.root/"validation.json";atomic_json(proof,{"passed":True,"zero_learning_interactions":True})
        changed="src/satellite_rl/workbench/fair20_runner.py"
        self.contract={"format":repair.FORMAT,"repair_id":repair.REPAIR_ID,"scientific_commit":self.origin,
            "execution_commit":self.execution_commit,"execution_source_repo":str(self.execution),"scientific_source_repo":str(self.science),
            "allowed_future_modes":["train","matrix","evaluate","refine"],"runs":{},
            "diff_files":[{"path":changed,"origin_sha256":hashlib.sha256(self.git(self.execution,"show",self.origin+":"+changed)).hexdigest(),
                           "execution_sha256":hashlib.sha256(self.git(self.execution,"show",self.execution_commit+":"+changed)).hexdigest()}],
            "validation_receipts":[{"path":str(proof),"sha256":digest(proof)}],
            "learning_and_environment_unchanged":True,"optimizer_and_budgets_unchanged":True,
            "acceptance_policy":"strict_SI_SGM_service_resource_else_verified_direct_no_extra_solve"}
        self.contract_path=self.root/"contract.json"

    def tearDown(self):self.patch.stop();self.temp.cleanup()

    @staticmethod
    def git(repo,*args):
        return subprocess.run(["git","-c","user.name=F20 fixture","-c","user.email=fixture@example.invalid",*args],
                              cwd=repo,capture_output=True,check=True).stdout

    def body(self):
        atomic_json(self.contract_path,self.contract)
        return {"contract_path":str(self.contract_path),"contract_sha256":digest(self.contract_path)}

    def job(self,identifier="formal"):
        directory=self.store.job_dir(identifier);directory.mkdir()
        config=default_config("sac")
        job={"run_id":identifier,"kind":"train","mode":"train","experiment_id":"S2-F20","git_commit":self.origin,
             "source_root":str(self.snapshot),"archive_root":str(self.store.archive_root),"resource_hold":True}
        atomic_json(directory/"job.json",job);atomic_json(directory/"config.json",config)
        atomic_json(directory/"status.json",{"stage":"queued_resources","env_step":0})
        return directory,job

    def test_future_binding_keeps_scientific_source_and_configuration(self):
        reference=repair.register_group(self.scheduler,self.body())
        directory,job=self.job();old_config=(directory/"config.json").read_bytes()
        bound=repair.bind_future_job(self.store,job)
        self.assertEqual(bound["git_commit"],self.origin);self.assertEqual(bound["source_root"],str(self.snapshot))
        self.assertEqual((directory/"config.json").read_bytes(),old_config)
        self.assertEqual(repair.execution_view(bound)["git_commit"],self.execution_commit)
        self.assertEqual(repair.execution_view(bound)["source_root"],reference["snapshot_path"])
        self.assertEqual(self.store.source_root,self.science.resolve())
        repair.validate_execution_job(bound,directory)
        bound["git_commit"]=self.execution_commit
        with self.assertRaises(ValueError):repair.validate_execution_job(bound,directory)

    def test_contract_rejects_learning_changes_even_when_declared(self):
        path=self.execution/"src/satellite_rl/fair_comparison/algorithms.py";path.write_text("changed learning\n")
        self.git(self.execution,"add",".");self.git(self.execution,"commit","-m","forbidden fixture change")
        self.contract["execution_commit"]=self.git(self.execution,"rev-parse","HEAD").decode().strip()
        body=self.body()
        with self.assertRaisesRegex(ValueError,"unapproved"):
            repair.validate_contract(self.store,body["contract_path"],body["contract_sha256"])

    def test_queued_existing_task_same_id_repair_and_idempotent_retry(self):
        directory,job=self.job("queued")
        self.contract["runs"]["queued"]={name+"_sha256":digest(directory/(name+".json")) for name in ("job","status","config")}
        self.contract["runs"]["queued"].update(origin_manifest_sha256=digest(self.snapshot/"source_manifest.json"),artifacts=[])
        body=self.body();config_sha=digest(directory/"config.json")
        status=repair.repair_job(self.scheduler,"queued",body)
        self.assertEqual(status["stage"],"queued_resources")
        actual=read_json(directory/"job.json")
        self.assertEqual(actual["run_id"],"queued");self.assertEqual(actual["git_commit"],self.origin)
        self.assertNotIn("resume",actual);self.assertEqual(digest(directory/"config.json"),config_sha)
        self.assertEqual(repair.repair_job(self.scheduler,"queued",body)["stage"],"queued_resources")
        self.assertEqual(read_json(directory/"source_lineage"/"origin"/"job.json"),job)

    def test_failed_training_without_validation_only_proof_is_rejected(self):
        directory,job=self.job("failed")
        atomic_json(directory/"status.json",{"stage":"failed","env_step":100000,"phase":"validation_100000"})
        atomic_json(directory/"launch.json",{"pid":12345,"process_start_time":1.})
        (directory/"checkpoint.pt").write_bytes(b"pinned complete checkpoint fixture")
        atomic_json(directory/"failure.json",{"run_id":"failed","error":"validation"})
        artifacts=[{"relative_path":name,"sha256":digest(directory/name)} for name in ("checkpoint.pt","failure.json")]
        self.contract["runs"]["failed"]={name+"_sha256":digest(directory/(name+".json")) for name in ("job","status","config","launch")}
        self.contract["runs"]["failed"].update(original_pid=12345,original_process_start_time=1.,
            origin_manifest_sha256=digest(self.snapshot/"source_manifest.json"),artifacts=artifacts)
        with patch("satellite_rl.workbench.runtime.pid_alive",return_value=False):
            with self.assertRaisesRegex(ValueError,"no-extra-interaction proof"):
                repair.repair_job(self.scheduler,"failed",self.body())
        self.assertEqual(read_json(directory/"job.json"),job)

    def test_failed_validation_binds_immutable_proof_without_changing_checkpoint(self):
        directory,job=self.job("validation")
        atomic_json(directory/"status.json",{"stage":"failed","env_step":100000,"phase":"validation_100000"})
        atomic_json(directory/"launch.json",{"pid":12345,"process_start_time":1.})
        (directory/"checkpoint.pt").write_bytes(b"synthetic frozen complete learning state")
        atomic_json(directory/"failure.json",{"run_id":"validation","error":"validation only"})
        for name in ("traces.jsonl","episodes.jsonl","updates.jsonl"):(directory/name).write_bytes(b"")
        names=("checkpoint.pt","failure.json","traces.jsonl","episodes.jsonl","updates.jsonl")
        item={name+"_sha256":digest(directory/(name+".json")) for name in ("job","status","config","launch")}
        item.update(original_pid=12345,original_process_start_time=1.,origin_manifest_sha256=digest(self.snapshot/"source_manifest.json"),
                    artifacts=[{"relative_path":name,"sha256":digest(directory/name)} for name in names])
        checkpoint_sha=digest(directory/"checkpoint.pt")
        proof=self.root/"validation-only.json"
        atomic_json(proof,{"version":repair.PROOF_FORMAT,"run_id":"validation","scientific_commit":self.origin,
            "execution_commit":self.execution_commit,"checkpoint_step":100000,"agent_interactions":100000,
            "failed_phase":"validation_100000","failed_status_sha256":item["status_sha256"],
            "failure_sha256":digest(directory/"failure.json"),"checkpoint_sha256":checkpoint_sha,
            "origin_records":{**{name:{"path":str(directory/"source_lineage"/"origin"/(name+".json")),"sha256":item[name+"_sha256"]}
                                    for name in ("job","status","config","launch")},
                **{name:{"path":str(directory/"source_lineage"/"origin"/"artifacts"/(name+extension)),"sha256":digest(directory/(name+extension))}
                   for name,extension in (("checkpoint",".pt"),("failure",".json"))}},
            "training_logs":{name:{"path":str(directory/name),"sha256":digest(directory/name),"saved_size":0,"size":0}
                             for name in ("traces.jsonl","episodes.jsonl","updates.jsonl")},
            "training_log_sizes":dict.fromkeys(("traces.jsonl","episodes.jsonl","updates.jsonl"),0)})
        item["validation_only_resume_proof"]={"path":str(proof),"sha256":digest(proof)}
        self.contract["runs"]["validation"]=item
        with patch("satellite_rl.workbench.runtime.pid_alive",return_value=False):
            status=repair.repair_job(self.scheduler,"validation",self.body())
        changed=read_json(directory/"job.json")
        self.assertEqual(status["stage"],"queued_resources")
        self.assertEqual(changed["git_commit"],self.origin);self.assertEqual(changed["source_root"],job["source_root"])
        self.assertTrue(changed["resume"]);self.assertEqual(changed["resume_origin_stage"],"failed")
        self.assertEqual(digest(directory/"checkpoint.pt"),checkpoint_sha)
        self.assertEqual(digest(changed["validation_only_resume_proof"]["proof_path"]),digest(proof))
        repair.validate_execution_job(changed,directory)
        preserved=directory/"source_lineage"/"origin"/"artifacts"/"failure.json"
        preserved.write_text("tampered origin")
        with self.assertRaises(ValueError):repair.validate_execution_job(changed,directory)

    def test_generic_resume_endpoint_cannot_bypass_f20_validation_failure_protection(self):
        from satellite_rl.workbench.service import Application
        directory,job=self.job("generic-blocked")
        atomic_json(directory/"status.json",{"stage":"failed","env_step":100000,"phase":"validation_100000"})
        atomic_json(directory/"failure.json",{"traceback":"in validate_node\nin _evaluate\nin full_b_case"})
        (directory/"checkpoint.pt").write_bytes(b"complete frozen checkpoint fixture")
        before={name:(directory/name).read_bytes() for name in ("job.json","status.json","config.json","checkpoint.pt")}
        application=Application(self.store.archive_root,source_root=self.science)
        with patch("satellite_rl.workbench.runtime.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError,"通用恢复入口"):
                application.api("POST",["runs","generic-blocked","resume"],{}, {})
            with self.assertRaisesRegex(ValueError,"通用恢复入口"):
                application.scheduler.resume("generic-blocked",source_repair="atomic_io_retry_v1")
            spawn.assert_not_called()
        self.assertEqual(read_json(directory/"status.json")["stage"],"failed")
        for name,payload in before.items():self.assertEqual((directory/name).read_bytes(),payload)


if __name__=="__main__":unittest.main()
