import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import random
from unittest.mock import patch

import numpy as np
import torch
from satellite_rl.workbench import nondrl_datafeed as feed, nondrl_datafeed_bootstrap as bridge
from satellite_rl.workbench.storage import Store, digest, read_json
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.coverage.nondrl import evidence_cache, learning
from test_nondrl_evidence_cache import evidence, nested_equal


class DatafeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.store=Store(self.base/"20424/archive",self.base/"repo")
        self.directory=self.store.job_dir(bridge.RUN_ID);self.directory.mkdir()
        self.model=self.store.models_root/bridge.RUN_ID;self.model.mkdir()
        self.source=evidence(self.base/"20418/source","01")
        self.cache=evidence_cache.build_cache(self.source,self.base/"20424/cache","01")
        snapshot=self.base/"20424/snapshot";snapshot.mkdir()
        (snapshot/"original.py").write_text("unchanged=True")
        atomic_json(snapshot/"source_manifest.json",{"git_commit":bridge.BASE_COMMIT,
             "files":{"original.py":digest(snapshot/"original.py")}})
        self.config={"training":{"updates":10000,"seed":51001,"batch_size":32,"microbatch_size":4}}
        self.job={"run_id":bridge.RUN_ID,"experiment_id":"S1-ND12","kind":"train","variant":"01",
             "git_commit":bridge.BASE_COMMIT,"source_root":str(snapshot),"archive_root":str(self.store.archive_root),"model_dir":str(self.model)}
        atomic_json(self.directory/"job.json",self.job);atomic_json(self.directory/"config.json",self.config)
        atomic_json(self.directory/"status.json",{"stage":"stopped","pid":991222})
        atomic_json(self.directory/"launch.json",{"pid":991222,"process_start_time":100.})
        validation=self.directory/"validation/update_001000";validation.mkdir(parents=True)
        rows=[]
        for index in range(20):
            for scale in (.75,1.,1.25):
                key=f"{index}:{scale:g}";name=f"row_{len(rows):03d}.json.gz"
                with gzip.open(validation/name,"wb") as stream:stream.write(json.dumps({"key":key}).encode())
                rows.append({"key":key,"file":name,"sha256":digest(validation/name)})
        atomic_json(validation/"manifest.json",{"rows":rows})
        nodes=[{"updates":1000,"metrics":{"raw_manifest_sha256":digest(validation/"manifest.json")}}]
        self.refs=[{"mother_id":"a","shard":0,"row":0} for _ in range(32)]
        losses=b"".join((json.dumps({"update":i,"cursor":i*32,"sample_refs":self.refs})+"\n").encode() for i in range(1,1026))
        (self.model/"losses.jsonl").write_bytes(losses);atomic_json(self.model/"validation.json",nodes)
        self.state={"format":"s1_nondrl_checkpoint_v1","updates":1025,"total_updates":10000,"cursor":32800,
             "model":{"w":torch.ones(2)},"optimizer":{"state":{0:{"step":torch.tensor(1025.),"exp_avg":torch.zeros(2),"exp_avg_sq":torch.zeros(2)}},"param_groups":[{"params":[0]}]},
             "rng":learning.rng_state(np.random.default_rng(51001)),
             "contract":{"source_sha":bridge.BASE_COMMIT,"dataset_sha":digest(self.source),"source_manifest_sha256":digest(snapshot/"source_manifest.json")},
             "validation":nodes,"sampled_last":self.refs,"loss_bytes":len(losses),"loss_sha256":hashlib.sha256(losses).hexdigest()}
        torch.save(self.state,self.model/"checkpoint.pt")
        self.provider=self.base/"20424/provider.py";self.provider.write_bytes(Path(evidence_cache.__file__).read_bytes())
        self.bootstrap=self.base/"20424/bootstrap.py";self.bootstrap.write_bytes(Path(bridge.__file__).read_bytes())
        self.proofs=[]
        for i,value in enumerate(({"format":"s1_nd12_evidence_cache_engineering_v1","passed":True,
               "bitwise_tensors":True,"bitwise_parameters_optimizer":True,"same_dataset_sha":True,"same_sampler_and_torch_rng":True},
               {"format":"s1_nd12_evidence_cache_compatibility_v1","passed":True})):
            path=self.base/"20424"/f"proof_{i}.json";atomic_json(path,value);self.proofs.append({"path":str(path),"sha256":digest(path)})
        self.constants=patch.multiple(bridge,PROVIDER_SHA=digest(self.provider),CACHE_SHA=digest(self.cache),TEACHER_SHA=digest(self.source),
               PROOF_SHAS={p["sha256"] for p in self.proofs})
        self.constants.start();self.alive=patch("satellite_rl.workbench.runtime.pid_alive",return_value=False);self.alive.start()
        self.audit_path=self.base/"20424/audit.json";atomic_json(self.audit_path,feed.audit_checkpoint(self.directory))
        def item(path):return {"path":str(path),"sha256":digest(path)}
        audit=read_json(self.audit_path)
        self.contract={"format":bridge.FORMAT,"extension_id":bridge.EXTENSION_ID,"run_id":bridge.RUN_ID,
              "base_execution_commit":bridge.BASE_COMMIT,"base_snapshot":item(snapshot/"source_manifest.json"),
              "config":item(self.directory/"config.json"),"checkpoint":item(self.model/"checkpoint.pt"),
              "teacher_manifest":item(self.source),"cache_manifest":item(self.cache),"provider":item(self.provider),
              "bridge":item(self.bootstrap),"checkpoint_audit":item(self.audit_path),"start_update":1025,
              "loss_prefix":{**item(self.model/"losses.jsonl"),"bytes":len(losses),"rows":1025},
              "validation_manifests":audit["validation_manifests"],"equivalence_proofs":self.proofs}
        self.contract["origin_records"]=audit["origin_records"]
        self.path=self.base/"20424/contract.json";self.publish();self.scheduler=Scheduler(self.store)

    def publish(self):
        atomic_json(self.path,self.contract);self.body={"contract_path":str(self.path),"contract_sha256":digest(self.path)}

    def tearDown(self):
        self.alive.stop();self.constants.stop();self.tmp.cleanup()

    def test_exact_single_id_install_and_composite_command_preserve_base_bytes(self):
        old=(self.directory/"config.json").read_bytes();checkpoint=(self.model/"checkpoint.pt").read_bytes()
        result=feed.extend_job(self.scheduler,bridge.RUN_ID,self.body)
        self.assertEqual(result["stage"],"queued")
        current=read_json(self.directory/"job.json")
        runtime=feed.validate_job_extension(current,self.directory)
        self.assertEqual(current["git_commit"],bridge.BASE_COMMIT)
        self.assertEqual((self.directory/"config.json").read_bytes(),old)
        self.assertEqual((self.model/"checkpoint.pt").read_bytes(),checkpoint)
        command=feed.worker_command(current,self.directory,"python")
        self.assertIn("bootstrap.py",command[2]);self.assertIn("--resume",command)
        self.assertEqual(runtime,feed.validate_job_extension(current,Path("/archive_only"),read_artifact=lambda r:(self.directory/r).read_bytes()))
        origin=self.directory/"runtime_extensions"/bridge.EXTENSION_ID
        for name,item in self.contract["origin_records"].items():
            self.assertEqual(digest(origin/("origin_"+name+".json")),item["sha256"])

    def test_live_worker_unknown_run_and_changed_provider_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"one approved"):
            feed.extend_job(self.scheduler,"another",self.body)
        self.alive.stop()
        with patch("satellite_rl.workbench.runtime.pid_alive",return_value=True),self.assertRaisesRegex(ValueError,"birth exited"):
            feed.extend_job(self.scheduler,bridge.RUN_ID,self.body)
        self.alive.start()
        self.provider.write_text("tampered=True")
        with self.assertRaisesRegex(ValueError,"SHA"):
            feed.extend_job(self.scheduler,bridge.RUN_ID,self.body)

    def test_partial_validation_or_missing_curve_cannot_be_skipped(self):
        (self.directory/"validation/update_002000").mkdir()
        with self.assertRaisesRegex(ValueError,"unfinished"):
            feed.audit_checkpoint(self.directory)
        (self.directory/"validation/update_002000").rmdir()
        self.state["validation"]=[];torch.save(self.state,self.model/"checkpoint.pt")
        with self.assertRaisesRegex(ValueError,"missing/unfinished"):
            feed.audit_checkpoint(self.directory)

    def test_initial_bridge_must_be_reviewed_but_existing_resume_keeps_frozen_copy(self):
        original=self.bootstrap.read_bytes()
        self.bootstrap.write_bytes(original+b"\n# arbitrary extra script\n")
        self.contract["bridge"]["sha256"]=digest(self.bootstrap);self.publish()
        with self.assertRaisesRegex(ValueError,"currently reviewed"):
            feed.extend_job(self.scheduler,bridge.RUN_ID,self.body)
        self.assertNotIn("runtime_extensions",read_json(self.directory/"job.json"))
        self.bootstrap.write_bytes(original);self.contract["bridge"]["sha256"]=digest(self.bootstrap);self.publish()
        feed.extend_job(self.scheduler,bridge.RUN_ID,self.body);job=read_json(self.directory/"job.json")
        with patch.object(bridge,"__file__",str(self.base/"unrelated_new_service.py")):
            self.assertIsNotNone(feed.worker_command(job,self.directory,"python"))

    def test_rng_formats_are_real_and_private_checks_preserve_global_states(self):
        sampler=np.random.default_rng(19)
        before=learning.rng_state(sampler)
        bridge.validate_rng(copy.deepcopy(self.state["rng"]))
        after=learning.rng_state(sampler)
        self.assertEqual(before["python"],after["python"])
        self.assertEqual(before["numpy"][0],after["numpy"][0]);np.testing.assert_array_equal(before["numpy"][1],after["numpy"][1])
        self.assertEqual(before["numpy"][2:],after["numpy"][2:]);self.assertTrue(torch.equal(before["torch"],after["torch"]))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before["cuda"],after["cuda"])));self.assertEqual(before["sampler"],after["sampler"])
        for key,bad in (("python",None),("numpy",None),("sampler",{}),("torch",torch.zeros(8,dtype=torch.uint8)),("cuda",[torch.zeros(8,dtype=torch.uint8)])):
            value=copy.deepcopy(self.state["rng"]);value[key]=bad
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"invalid private RNG"):
                bridge.validate_rng(value)

    def test_linux_zombie_requires_exact_birth_no_fds_and_released_lock(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        proc=self.base/"mock_proc";process=proc/"991222";process.mkdir(parents=True);(process/"fd").mkdir()
        (process/"stat").write_text("991222 (worker) Z "+"0 "*30)
        (self.directory/"worker.lock").write_bytes(b"0")
        flock=MagicMock();fcntl=SimpleNamespace(LOCK_EX=2,LOCK_NB=4,LOCK_UN=8,flock=flock)
        with patch("satellite_rl.workbench.runtime.process_start_time",return_value=100.),patch.dict("sys.modules",{"fcntl":fcntl}):
            self.assertTrue(feed.linux_worker_exited(self.directory,991222,100.,proc))
            self.assertEqual(flock.call_count,2)
            (process/"fd/3").write_bytes(b"open")
            self.assertFalse(feed.linux_worker_exited(self.directory,991222,100.,proc));(process/"fd/3").unlink()
            flock.side_effect=BlockingIOError("worker lock still held")
            self.assertFalse(feed.linux_worker_exited(self.directory,991222,100.,proc))
        with patch("satellite_rl.workbench.runtime.process_start_time",return_value=None):
            self.assertFalse(feed.linux_worker_exited(self.directory,991222,100.,proc))

    def test_crash_tail_is_saved_and_bad_cursor_or_adam_count_cannot_pass(self):
        feed.extend_job(self.scheduler,bridge.RUN_ID,self.body);job=read_json(self.directory/"job.json")
        tail=b'{"uncommitted_after_checkpoint":true}\n'
        with (self.model/"losses.jsonl").open("ab") as source:source.write(tail)
        feed.preserve_checkpoint_tail(job,self.directory)
        paths=list((self.directory/"runtime_extensions"/bridge.EXTENSION_ID/"recovery_tails").glob("*.jsonl"))
        self.assertEqual(len(paths),1);self.assertEqual(paths[0].read_bytes(),tail)
        self.assertTrue((self.model/"losses.jsonl").read_bytes().endswith(tail))
        self.state["cursor"]-=1;torch.save(self.state,self.model/"checkpoint.pt")
        with self.assertRaisesRegex(ValueError,"cursor"):
            feed.validate_job_extension(job,self.directory)
        self.state["cursor"]+=1;self.state["runtime_extensions"]=[job["runtime_extensions"][0]["marker"]]
        self.state["optimizer"]["state"][0]["step"]=torch.tensor(1024.)
        torch.save(self.state,self.model/"checkpoint.pt")
        with self.assertRaisesRegex(ValueError,"Adam"):
            feed.validate_job_extension(job,self.directory)

    def test_api_and_partial_activation_recovery_only_queue_one_existing_id(self):
        from satellite_rl.workbench.coverage_nondrl import api
        from types import SimpleNamespace
        application=SimpleNamespace(store=self.store,app=SimpleNamespace(scheduler=self.scheduler))
        with patch("satellite_rl.workbench.coverage_nondrl.deployment"):
            status=api(application,"POST",["runs",bridge.RUN_ID,"datafeed-extension"],self.body)
        self.assertEqual(status["stage"],"queued")
        job=read_json(self.directory/"job.json");job["runtime_extension_pending"]=True
        atomic_json(self.directory/"job.json",job)
        with self.assertRaisesRegex(ValueError,"not committed"):
            feed.worker_command(job,self.directory,"python")
        self.assertEqual(feed.extend_job(self.scheduler,bridge.RUN_ID,self.body)["stage"],"queued")
        self.assertFalse(read_json(self.directory/"job.json")["runtime_extension_pending"])

    def test_consumer_requires_the_real_model_marker_and_never_gets_bootstrap(self):
        feed.extend_job(self.scheduler,bridge.RUN_ID,self.body);job=read_json(self.directory/"job.json")
        marker=job["runtime_extensions"][0]["marker"]
        policy=self.model/"consumer_policy.pt";torch.save({"runtime_extensions":[marker]},policy)
        self.assertEqual(feed.validate_model_consumer(job,self.directory,{"runtime_extensions":[marker]},policy),[marker])
        with self.assertRaisesRegex(ValueError,"markers differ"):
            feed.validate_model_consumer(job,self.directory,{"runtime_extensions":[]},policy)
        with self.assertRaisesRegex(ValueError,"authorized producer"):
            feed.validate_model_consumer({"run_id":"different"},self.directory,{"runtime_extensions":[marker]},policy)

    def test_advanced_checkpoint_marker_and_original_curve_prefix_are_required(self):
        feed.extend_job(self.scheduler,bridge.RUN_ID,self.body);job=read_json(self.directory/"job.json")
        self.state["extra_uncommitted_marker"]=True;torch.save(self.state,self.model/"checkpoint.pt")
        with self.assertRaisesRegex(ValueError,"composite runtime marker"):
            feed.validate_job_extension(job,self.directory)
        self.state["runtime_extensions"]=[job["runtime_extensions"][0]["marker"]]
        self.state["updates"]=1026
        torch.save(self.state,self.model/"checkpoint.pt")
        with self.assertRaisesRegex(ValueError,"cursor"):
            feed.validate_job_extension(job,self.directory)
        with (self.model/"losses.jsonl").open("ab") as source:
            source.write((json.dumps({"update":1026,"cursor":1026*32,"sample_refs":self.refs})+"\n").encode())
        self.state.update(cursor=1026*32,loss_bytes=(self.model/"losses.jsonl").stat().st_size,loss_sha256=digest(self.model/"losses.jsonl"))
        self.state["optimizer"]["state"][0]["step"]=torch.tensor(1026.)
        torch.save(self.state,self.model/"checkpoint.pt")
        feed.validate_job_extension(job,self.directory)
        with (self.model/"losses.jsonl").open("r+b") as source:source.write(b"bad")
        with self.assertRaisesRegex(ValueError,"prefix"):
            feed.validate_job_extension(job,self.directory)

    def test_bridge_reader_rng_tensor_equivalence_and_metadata_preserve_real_updates(self):
        config={"engineering":True,"updates":2,"seed":51001,"device":"cpu","batch_size":32,
                "microbatch_size":4,"checkpoint_interval":1}
        learning.train(self.source,"01",config,self.model/"tiny_original")
        baseline=torch.load(self.model/"tiny_original/checkpoint.pt",weights_only=False)
        runtime=bridge.marker(self.contract,self.body["contract_sha256"])
        original=learning.EvidenceDataset(self.source,"01")
        restore=bridge.install(self.contract,runtime,self.job)
        try:
            equivalent=learning.EvidenceDataset(self.source,"01")
            a,b=np.random.default_rng(51001),np.random.default_rng(51001)
            x,px=original.draw(a,32);y,py=equivalent.draw(b,32)
            self.assertEqual(px,py);self.assertEqual(a.bit_generator.state,b.bit_generator.state)
            self.assertTrue(nested_equal(learning.collate(x,"01","cpu"),learning.collate(y,"01","cpu")))
            target=self.model/"metadata.pt";value={"format":learning.CHECKPOINT_FORMAT,"model":{"w":torch.ones(2)},"optimizer":self.state["optimizer"],"rng":self.state["rng"]}
            learning.save_torch(target,value);saved=torch.load(target,weights_only=False)
            self.assertTrue(nested_equal(saved["model"],value["model"]));self.assertTrue(nested_equal(saved["optimizer"],value["optimizer"]))
            self.assertEqual(saved["runtime_extensions"],[runtime]);self.assertNotIn("runtime_extensions",value)
            unrelated=self.base/"unrelated.pt"
            learning.save_torch(unrelated,{"purpose":"unrelated storage"})
            self.assertEqual(torch.load(unrelated,weights_only=False),{"purpose":"unrelated storage"})
            equivalent.close()
            learning.train(self.source,"01",config,self.model/"tiny_composite")
            updated=torch.load(self.model/"tiny_composite/checkpoint.pt",weights_only=False)
            self.assertTrue(nested_equal(baseline["model"],updated["model"]))
            self.assertTrue(nested_equal(baseline["optimizer"],updated["optimizer"]))
            self.assertEqual(baseline["rng"]["sampler"],updated["rng"]["sampler"])
            self.assertTrue(torch.equal(baseline["rng"]["torch"],updated["rng"]["torch"]))
            self.assertEqual(baseline["contract"],updated["contract"])
            self.assertEqual(updated["runtime_extensions"],[runtime])
            final=torch.load(self.model/"tiny_composite/final/policy.pt",weights_only=False)
            metadata=read_json(self.model/"tiny_composite/final/model.json")
            self.assertEqual(final["runtime_extensions"],[runtime]);self.assertEqual(metadata["runtime_extensions"],[runtime])
        finally:restore()


if __name__=="__main__":unittest.main()
