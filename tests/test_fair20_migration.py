"""Synthetic control-plane migration/ownership/peer checks, zero learning."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import Store,digest,read_json
from satellite_rl.workbench.fair20_service import default_config,Fair20Application
from satellite_rl.workbench import fair20_migration as migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=Store(self.root/"20474"/"archive");self.target=Store(self.root/"20418"/"archive")
        self.schedulers={port:type("Scheduler",(),{"store":store,"lock":threading.RLock()})() for port,store in ((74,self.source),(18,self.target))}
        self.identifier="queued_dqn_A";self.directory=self.source.job_dir(self.identifier);self.directory.mkdir()
        model_dir=self.source.models_root/self.identifier;model_dir.mkdir()
        config=default_config("dqn","A","shorttune")
        self.job={"run_id":self.identifier,"request_id":"original-dqn-short-A","experiment_id":"S2-F20","kind":"train",
            "algorithm":"dqn","candidate":"A","mode":"shorttune","git_commit":migration.SCIENCE,
            "source_root":str(self.root/"20474"/"readonly-science"),"model_dir":str(model_dir),"archive_root":str(self.source.archive_root),
            "frozen_input_sha256":"fixed-original-input-sha","execution_source":{"execution_commit":migration.EXECUTION},"resource_hold":True}
        atomic_json(self.directory/"job.json",self.job);atomic_json(self.directory/"config.json",config)
        atomic_json(self.directory/"status.json",{"stage":"queued_resources","env_step":0,"update_step":0})
        self.contract={"format":migration.FORMAT,"run_id":self.identifier,"request_id":self.job["request_id"],"source_port":20474,"target_port":20418,
            "source_archive_root":str(self.source.archive_root),"target_archive_root":str(self.target.archive_root),
            "source_job_sha256":digest(self.directory/"job.json"),"source_status_sha256":digest(self.directory/"status.json"),
            "config_sha256":digest(self.directory/"config.json"),"frozen_input_sha256":self.job["frozen_input_sha256"],
            "scientific_commit":migration.SCIENCE,"execution_commit":migration.EXECUTION}
        self.path=self.root/"migration-contract.json";atomic_json(self.path,self.contract)
        self.body={"contract_path":str(self.path),"contract_sha256":digest(self.path)}

    def tearDown(self):self.temp.cleanup()

    def export(self):return migration.export_job(self.schedulers[74],self.body)

    def adopt(self,receipt):
        return migration.adopt_job(self.schedulers[18],{**self.body,"export_receipt_path":receipt["export_receipt_path"],"export_receipt_sha256":receipt["export_receipt_sha256"]})

    def test_export_adopt_keeps_id_config_inputs_and_single_owner(self):
        before=(self.directory/"config.json").read_bytes()
        receipt=self.export();self.assertEqual(read_json(self.directory/"status.json")["stage"],"migrated")
        with self.assertRaises(ValueError):
            with migration.launch_ownership(self.source,self.job):self.fail("source cannot launch after export")
        pending=self.target.job_dir(self.identifier)
        with self.assertRaises(ValueError):
            with migration.launch_ownership(self.target,self.job):self.fail("pending target cannot launch")
        adopted=self.adopt(receipt)
        self.assertEqual(adopted["job"]["run_id"],self.identifier)
        self.assertEqual(adopted["job"]["request_id"],self.job["request_id"])
        self.assertEqual(adopted["job"]["frozen_input_sha256"],self.job["frozen_input_sha256"])
        self.assertEqual(adopted["job"]["source_root"],self.job["source_root"])
        self.assertEqual((pending/"config.json").read_bytes(),before)
        self.assertEqual(adopted["job"]["executor_port"],20418)
        self.assertEqual(adopted["job"]["archive_root"],str(self.target.archive_root))
        with migration.launch_ownership(self.target,adopted["job"]) as owner:self.assertEqual(owner["owner_port"],20418)
        self.assertEqual(self.adopt(receipt)["job"],adopted["job"])
        self.assertEqual(self.export()["export_receipt_sha256"],receipt["export_receipt_sha256"])

    def test_started_or_nonzero_task_cannot_export(self):
        status=read_json(self.directory/"status.json");status.update(stage="training",env_step=1)
        atomic_json(self.directory/"status.json",status)
        self.contract["source_status_sha256"]=digest(self.directory/"status.json");atomic_json(self.path,self.contract)
        self.body["contract_sha256"]=digest(self.path)
        with self.assertRaisesRegex(ValueError,"queued0"):self.export()
        self.assertEqual(read_json(self.directory/"status.json")["env_step"],1)

    def test_target_without_source_revocation_is_rejected(self):
        receipt=self.root/"fake-export.json";atomic_json(receipt,{"source_dispatch_revoked":False})
        with self.assertRaises(ValueError):
            migration.adopt_job(self.schedulers[18],{**self.body,"export_receipt_path":str(receipt),"export_receipt_sha256":digest(receipt)})

    def test_peer_dependency_is_sha_bound_completed_and_read_only(self):
        directory=self.source.job_dir("dqn_B");directory.mkdir()
        job={"run_id":"dqn_B","git_commit":migration.SCIENCE,"experiment_id":"S2-F20","algorithm":"dqn","candidate":"B","mode":"shorttune"}
        atomic_json(directory/"job.json",job);atomic_json(directory/"status.json",{"run_id":"dqn_B","stage":"completed","env_step":32768})
        atomic_json(directory/"config.json",default_config("dqn","B","shorttune"));atomic_json(directory/"results.json",{"validation_summary":{"SI":.5,"SGM":.6}})
        ref={"archive_root":str(self.source.archive_root),"run_id":"dqn_B","role":"shorttune",**{name+"_sha256":digest(directory/(name+".json")) for name in ("job","status","config","results")}}
        first=migration.import_peer(self.target,{"source_ref":ref})
        self.assertTrue(first["read_only"]);self.assertEqual(migration.import_peer(self.target,{"source_ref":ref}),first)
        app=type("Application",(),{"store":self.target,"scheduler":self.schedulers[18]})()
        api=Fair20Application(app);resolved,source_job,status=api.task("dqn_B",True)
        self.assertEqual(resolved,directory)
        with self.assertRaisesRegex(ValueError,"read-only"):api.resume("dqn_B")
        atomic_json(directory/"results.json",{"tampered":True})
        with self.assertRaises(ValueError):migration.resolve_peer(self.target,"dqn_B")

    def test_old_source_scheduler_cannot_spawn_after_migration(self):
        from satellite_rl.workbench.runtime import Scheduler
        self.export();scheduler=Scheduler(self.source)
        with patch("satellite_rl.workbench.runtime.subprocess.Popen") as spawn:
            scheduler._launch(self.job,read_json(self.directory/"status.json"));spawn.assert_not_called()
        self.assertEqual(read_json(self.directory/"status.json")["stage"],"migrated")

    def test_original_shorttune_without_execution_extension_remains_original_science(self):
        job=read_json(self.directory/"job.json");job.pop("execution_source");atomic_json(self.directory/"job.json",job)
        self.contract["execution_commit"]=migration.SCIENCE;self.contract["source_job_sha256"]=digest(self.directory/"job.json")
        atomic_json(self.path,self.contract);self.body["contract_sha256"]=digest(self.path)
        receipt=self.export();adopted=self.adopt(receipt)
        self.assertEqual(receipt["execution_commit"],migration.SCIENCE)
        self.assertNotIn("execution_source",adopted["job"])
        self.assertEqual(adopted["job"]["source_root"],job["source_root"])

    def test_export_journal_closes_crash_window_before_owner_publication(self):
        receipt=self.export()
        owner_path=migration.ownership_root(self.source)/(self.identifier+".json");owner_path.unlink()
        original=read_json(self.directory/"migration"/"origin"/"status.json");atomic_json(self.directory/"status.json",original)
        with self.assertRaisesRegex(ValueError,"export journal"):
            with migration.launch_ownership(self.source,self.job):self.fail("exported source must never relaunch")
        retry=self.export();self.assertEqual(retry["export_receipt_sha256"],receipt["export_receipt_sha256"])
        self.assertEqual(self.adopt(retry)["job"]["run_id"],self.identifier)

    def test_control_service_takeover_recognizes_alive_PPO_and_does_not_relaunch(self):
        from satellite_rl.workbench.runtime import Scheduler
        directory=self.source.job_dir("existing_ppo");directory.mkdir()
        atomic_json(directory/"job.json",{"run_id":"existing_ppo","experiment_id":"S2-F20","kind":"train","algorithm":"ppo"})
        status={"run_id":"existing_ppo","stage":"training","env_step":139583,"pid":6789}
        atomic_json(directory/"status.json",status);atomic_json(directory/"launch.json",{"pid":6789,"process_start_time":42.})
        scheduler=Scheduler(self.source,max_workers=1)
        with patch("satellite_rl.workbench.runtime.pid_alive",side_effect=lambda pid,birth=None:pid==6789 and birth==42.),patch.object(scheduler,"_launch") as spawn,patch("satellite_rl.workbench.runtime.gpu_snapshot") as gpu:
            scheduler.tick();spawn.assert_not_called();gpu.assert_not_called()
        self.assertEqual(read_json(directory/"status.json"),status)
        self.assertFalse((directory/"stop.request").exists())

    def test_peer_identity_and_actual_execution_are_required_even_with_valid_shas(self):
        directory=self.source.job_dir("model_peer");directory.mkdir()
        job={"run_id":"model_peer","git_commit":migration.SCIENCE,"experiment_id":"S2-F20","algorithm":"dqn",
             "candidate":"A","mode":"train","execution_source":{"execution_commit":"b474"}}
        atomic_json(directory/"job.json",job);atomic_json(directory/"status.json",{"run_id":"model_peer","stage":"completed","env_step":400000})
        atomic_json(directory/"config.json",default_config("dqn","A","train"));atomic_json(directory/"results.json",{})
        def reference(role="model"):
            return {"archive_root":str(self.source.archive_root),"run_id":"model_peer","role":role,
                    **{name+"_sha256":digest(directory/(name+".json")) for name in ("job","status","config","results")}}
        with self.assertRaisesRegex(ValueError,"fbf"):migration.peer_directory(self.target,reference())
        job["execution_source"]["execution_commit"]=migration.EXECUTION;atomic_json(directory/"job.json",job)
        self.assertEqual(migration.peer_directory(self.target,reference()),directory)
        status=read_json(directory/"status.json");status["run_id"]="wrong";atomic_json(directory/"status.json",status)
        with self.assertRaises(ValueError):migration.peer_directory(self.target,reference())
        status.update(run_id="model_peer",env_step=32768);atomic_json(directory/"status.json",status)
        job.update(mode="shorttune");atomic_json(directory/"job.json",job)
        with self.assertRaisesRegex(ValueError,"2fc"):migration.peer_directory(self.target,reference("shorttune"))


if __name__=="__main__":unittest.main()
