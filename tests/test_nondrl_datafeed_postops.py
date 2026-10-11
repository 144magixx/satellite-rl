"""Synthetic full artifact receipts, not additional formal training updates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from satellite_rl.workbench import nondrl_datafeed as feed, nondrl_datafeed_bootstrap as bridge
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest,read_json
from scripts.stage1_nondrl12 import archive
from scripts.stage1_nondrl12.datafeed_evidence import audit_runtime,training_segments,PREFIX
from scripts.stage1_nondrl12.strict_audit import Evidence
import test_nondrl_datafeed as feed_fixture
from test_nondrl_postprocessing import make_volume


class DatafeedPostopsTests(unittest.TestCase):
    def setUp(self):
        self.case=feed_fixture.DatafeedTests("test_exact_single_id_install_and_composite_command_preserve_base_bytes")
        self.case.setUp();case=self.case
        self.addCleanup(case.tearDown)
        case.config["coverage"]={"nondrl":{"route":"01","method":"main"}}
        atomic_json(case.directory/"config.json",case.config)
        case.state["contract"]["route"]="01";case.state["network"]={}
        losses=b"".join((json.dumps({"update":i,"cursor":i*32,"elapsed_seconds":i/2,"sample_refs":case.refs})+"\n").encode() for i in range(1,1026))
        (case.model/"losses.jsonl").write_bytes(losses)
        case.state.update(loss_bytes=len(losses),loss_sha256=hashlib.sha256(losses).hexdigest(),elapsed_seconds=512.5)
        torch.save(case.state,case.model/"checkpoint.pt")
        atomic_json(case.audit_path,feed.audit_checkpoint(case.directory))
        case.contract["config"]["sha256"]=digest(case.directory/"config.json")
        case.contract["checkpoint"]["sha256"]=digest(case.model/"checkpoint.pt")
        case.contract["checkpoint_audit"]["sha256"]=digest(case.audit_path)
        case.contract["loss_prefix"].update(sha256=case.state["loss_sha256"],bytes=len(losses))
        proof=read_json(case.proofs[0]["path"])
        proof["draw_benchmarks"]=[{"samples":1024,"positions_sha256":"a"*64,"tensors_sha256":"b"*64} for _ in range(2)]
        proof["cuda_update_proofs"]=[{"updates":2,"samples":64,**{k:"c"*64 for k in
            ("parameters_sha256","optimizer_sha256","torch_rng_sha256","cuda_rng_sha256","positions_sha256","sampler_rng_sha256")}} for _ in range(2)]
        atomic_json(case.proofs[0]["path"],proof);case.proofs[0]["sha256"]=digest(case.proofs[0]["path"])
        compat=read_json(case.proofs[1]["path"])
        compat.update(normalized_full_module_AST_equal=True,final_candidate_sha256=digest(case.provider),
            verified_existing_cache_manifest_sha256=digest(case.cache),dataset_sha256=digest(case.source),
            cuda_updates_added=0,draw_batches_added=0,source_original_files_changed=False)
        atomic_json(case.proofs[1]["path"],compat);case.proofs[1]["sha256"]=digest(case.proofs[1]["path"])
        self.proof_constants=patch.object(bridge,"PROOF_SHAS",{v["sha256"] for v in case.proofs});self.proof_constants.start()
        self.addCleanup(self.proof_constants.stop)
        case.publish();feed.extend_job(case.scheduler,bridge.RUN_ID,case.body)
        self.job=read_json(case.directory/"job.json");self.marker=self.job["runtime_extensions"][0]["marker"]
        atomic_json(case.directory/"launch.json",{"command":["python","-u",str(case.directory/PREFIX/"bootstrap.py"),
            "--resume","--contract-sha256",self.marker["contract_sha256"]]})
        shutil.copytree(case.job["source_root"],case.directory/"source")
        # Read artifacts through the canonical archive layout, while the base
        # fixture's models directory remains the producer location.
        self.completed=deepcopy(case.state);self.completed.update(updates=10000,cursor=320000,runtime_extensions=[self.marker],elapsed_seconds=1000.)
        self.completed["optimizer"]["state"][0]["step"]=torch.tensor(10000.)
        complete_losses=losses+b"".join((json.dumps({"update":i,"cursor":i*32,"elapsed_seconds":512.5+(i-1025)*.04,"sample_refs":case.refs})+"\n").encode() for i in range(1026,10001))
        self.completed.update(loss_bytes=len(complete_losses),loss_sha256=hashlib.sha256(complete_losses).hexdigest())
        final={"format":"s1_nondrl_supervised_v1","contract":case.state["contract"],"network":{},"model":case.state["model"],"runtime_extensions":[self.marker],
            "updates":10000,"selection":"fixed_budget_endpoint"}
        destination=case.directory/"models";destination.mkdir()
        torch.save(self.completed,destination/"checkpoint.pt");(destination/"losses.jsonl").write_bytes(complete_losses)
        (destination/"final").mkdir();torch.save(final,destination/"final/policy.pt")
        atomic_json(destination/"final/model.json",{**final,"model":None})
        atomic_json(destination/"status.json",{"contract":case.state["contract"],"runtime_extensions":[self.marker]})

    def test_full_original_checkpoint_prefix_cache_and_endpoint_close_in_zip(self):
        case=self.case;proof=audit_runtime(self.job,Evidence(case.directory),case.config,completed=True)
        self.assertEqual(proof["activation_update"],1025)
        self.assertEqual(proof["preserved_loss_rows"],1025)
        self.assertTrue(proof["base_contract_preserved"])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prefix="20418/"+bridge.RUN_ID+"/"
            values={prefix+p.relative_to(case.directory).as_posix():p.read_bytes() for p in case.directory.rglob("*") if p.is_file()}
            volume=make_volume(root,"only_cache_once.zip",values)
            receipt={"experiment_id":"S1-ND12","port":20418,"run_id":bridge.RUN_ID,"volumes":[volume]}
            archived=audit_runtime(self.job,Evidence(root/"receipt.json",receipt=receipt),case.config,completed=True)
            self.assertEqual(archived,proof)
            self.assertEqual(len([n for n in volume["members"] if n.endswith("/cache/data.bin")]),1)

    def test_changed_optimizer_base_contract_marker_prefix_or_cached_bytes_rejected(self):
        path=self.case.directory/"models/checkpoint.pt";original=path.read_bytes()
        for field in ("contract","runtime_extensions","optimizer"):
            with self.subTest(field=field):
                state=deepcopy(self.completed)
                if field=="contract":state[field]["route"]="02"
                elif field=="runtime_extensions":state[field]=[]
                else:state[field]["param_groups"]=[{"params":[99]}]
                torch.save(state,path)
                with self.assertRaises(ValueError):audit_runtime(self.job,Evidence(self.case.directory),self.case.config,completed=True)
                path.write_bytes(original)
        loss=self.case.directory/"models/losses.jsonl";loss.write_bytes(b"changed original prefix")
        with self.assertRaisesRegex(ValueError,"prefix"):
            audit_runtime(self.job,Evidence(self.case.directory),self.case.config,completed=True)
        cache=self.case.directory/PREFIX/"cache/data.bin";cache.write_bytes(b"changed cache")
        with self.assertRaisesRegex(ValueError,"artifact"):
            audit_runtime(self.job,Evidence(self.case.directory),self.case.config,completed=True)

    def test_consumers_only_reference_exact_producer_without_reader_execution(self):
        job={"experiment_id":"S1-ND12","kind":"evaluate","git_commit":bridge.BASE_COMMIT,"model_source_run_id":bridge.RUN_ID,
             "model_runtime_extensions":[self.marker],"model_files":[{"sha256":"a"*64}]}
        config={"coverage":{"nondrl":{"route":"01","method":"MO"}}}
        proof=audit_runtime(job,None,config)
        self.assertFalse(proof["provider_execution_authorized"])
        for key,value in (("model_source_run_id","different"),("git_commit","f"*40),("kind","train")):
            invalid={**job,key:value}
            with self.assertRaises(ValueError):audit_runtime(invalid,None,config)

    def test_source_boundary_segments_are_separate_and_remote_archive_compiles(self):
        rows=[{"update":u,"elapsed_seconds":u/2} for u in range(1,10001)]
        parts=training_segments(rows,{"activation_update":1025,"original_active_seconds":512.5,"complete_active_seconds":1000.})
        self.assertEqual([s["recorded_updates"] for s in parts],[1025,8975])
        self.assertEqual(parts[1]["checkpoint_active_seconds"],487.5)
        compile("REQUEST_JSON='{}'\n"+archive.REMOTE_BUILD,"frozen composite archive","exec")


if __name__=="__main__":unittest.main()
