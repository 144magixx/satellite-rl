import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.stage1_nondrl12 import archive, final_delivery as delivery, placement_transport
from test_nondrl_postprocessing import make_volume


def raw(value):
    return json.dumps(value).encode()


class FakeRemote:
    def __init__(self, payload=b"", fail_after=None):
        self.payload=payload;self.fail_after=fail_after
    def stat(self,path):
        return SimpleNamespace(st_size=len(self.payload))
    def open(self,path,mode):
        remote=self
        class Stream(io.BytesIO):
            def write(self,block):
                if remote.fail_after is not None:
                    allowed=max(0,remote.fail_after-self.tell())
                    super().write(block[:allowed]);remote.payload=self.getvalue()
                    raise OSError("simulated disconnected upload")
                return super().write(block)
            def close(self):
                if not self.closed and mode!="rb":remote.payload=self.getvalue()
                super().close()
        return Stream(self.payload if mode!="wb" else b"")


class FinalDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.commit="a"*40
        self.report_path=self.root/"10_非DRL正式报告.md"
        self.report_path.write_text("# 非DRL十二路线正式实验报告\n\n"+"本研究按冻结科学范围记录完成、停止和预算不足的真实证据。"*20,encoding="utf-8")
        delivery.save(self.root/"controller/execution_placement.json",placement_transport.CURRENT)
        self.source_dir=self.root/"job_archives/20418/shared_sources";self.source_dir.mkdir(parents=True)
        files={};refs=[]
        for role,day in (("validation","2024-02-10"),("test","2024-02-11"),("test","2024-02-12")):
            prefix=f"shared_sources/{role}/{day}";payload=day.encode();run="prep_"+day
            files[prefix+"/raw.tar"]=payload
            files[prefix+"/source_preflight.json"]=b"pin"
            files[prefix+"/selection_contract.json"]=b"selection"
            files[prefix+"/source_job_identity.json"]=raw({"experiment_id":"S1-ND12","git_commit":self.commit,"run_id":run})
            files[prefix+"/source_job_status.json"]=raw({"stage":"completed"})
            files[prefix+"/source_job_artifacts.json"]=raw({"raw_archive_sha256":hashlib.sha256(payload).hexdigest()})
            refs.append({"role":role,"date_utc":day,"source_bundle_member":prefix+"/raw.tar",
                "source_run_id":run,"source_port":20418,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),
                "source_preflight_sha256":hashlib.sha256(b"pin").hexdigest(),
                "selection_contract_sha256":hashlib.sha256(b"selection").hexdigest()})
        files["shared_sources/partition_manifest.json"]=b"partition"
        source={"experiment_id":"S1-ND12","run_id":"shared_sources","source_commit":self.commit,
            "source_refs":refs,"partition_manifest_sha256":hashlib.sha256(b"partition").hexdigest(),
            "volumes":[make_volume(self.source_dir,"source.zip",files)]}
        delivery.save(self.source_dir/"receipt.json",source)
        self.job_dir=self.root/"job_archives/20473/eval_06";self.job_dir.mkdir(parents=True)
        self.job={"experiment_id":"S1-ND12","run_id":"eval_06","git_commit":self.commit,"kind":"evaluate"}
        prefix="20473/eval_06/"
        self.task_receipt={"experiment_id":"S1-ND12","port":20473,"run_id":"eval_06","source_commit":self.commit,
            "status":"completed","volumes":[make_volume(self.job_dir,"task.zip",{
                prefix+"job.json":raw(self.job),prefix+"status.json":raw({"stage":"completed"}),prefix+"config.json":b"{}"})]}
        delivery.save(self.job_dir/"receipt.json",self.task_receipt)
        state={"experiment_id":"S1-ND12","scientific_commit":self.commit,"phase":"report_pending",
               "jobs":{"test/06/BH/20473":{"port":20473,"run_id":"eval_06","kind":"evaluate"}}}
        prepared={"experiment_id":"S1-ND12","scientific_commit":self.commit,"tasks":{}}
        snapshot_files={}
        for name,item in (("state",state),("preparations",prepared),("plan",{"experiment_id":"S1-ND12"})):
            path=self.root/"analysis/final_input"/(name+".json");delivery.save(path,item)
            snapshot_files[name]={"path":str(path),"sha256":delivery.sha(path),"bytes":path.stat().st_size}
        delivery.save(self.root/"analysis/final_input/snapshot.json",{"scientific_commit":self.commit,"files":snapshot_files})
        delivery.save(self.root/"analysis/final_input/archive_plan.json",archive.plan(state,prepared))
        delivery.save(self.root/"analysis/final_input/archive_closure.json",{"complete":True,"tasks":[{"run_id":"eval_06","closed":True}]})
        self.report={"experiment_id":"S1-ND12","scientific_commit":self.commit,"formal_ready":True,"issues":[],
            "routes":{f"{i:02d}":{"status":"budget_limited_inconclusive" if i in (2,3) else "science_stopped","methods":{}} for i in range(1,13)},
            "figures":[],"endpoints":[],"row_audits":{"deliberately_not_embedded":[1,2,3]}}
        self.write_report()
        self.dest=self.root/"canonical_delivery"

    def write_report(self):
        result=self.root/"analysis/final_report/report.json";delivery.save(result,self.report)
        md=self.root/"analysis/final_report/report.md";md.write_text("audit markdown")
        delivery.save(self.root/"analysis/final_report/completion.json",{"formal_ready":True,"scientific_commit":self.commit,
            "tasks":1,"endpoint_count":0,"figure_count":0,"report_sha256":delivery.sha(result),"markdown_sha256":delivery.sha(md)})
        delivery.save(self.root/"controller/finalizer/status.json",{"phase":"complete","report_sha256":delivery.sha(result)})

    def tearDown(self):
        self.temp.cleanup()

    def gate(self):
        return delivery.gate_state(self.root,self.report_path)

    def package(self):
        with patch.object(delivery,"free_bytes",return_value=delivery.RESERVE+(1<<30)):
            return delivery.package(self.root,self.report_path,roots=[self.dest])

    def test_inconclusive_routes_are_ready_and_raw_data_are_only_indexed(self):
        self.assertTrue(self.gate()["ready"],self.gate()["errors"])
        package=self.package();manifest=delivery.read(package["input_manifest_path"])
        names={r["member"] for r in manifest["members"]}
        self.assertIn("report/report10.md",names)
        self.assertIn("indexes/source_refs.json",names)
        self.assertFalse(any(n.endswith((".zip",".tar",".npz",".jsonl")) for n in names))
        self.assertNotIn("receipts/strict_report.json",names)
        self.assertEqual(delivery.load_package(package["local_receipt_path"])[0]["zip_sha256"],package["zip_sha256"])
        self.assertEqual(self.package()["zip_sha256"],package["zip_sha256"])

    def test_missing_report_or_incomplete_finalizer_never_packages(self):
        self.report_path.unlink()
        self.assertFalse(self.gate()["ready"])
        self.assertFalse(self.dest.exists())
        with self.assertRaisesRegex(RuntimeError,"not ready"):
            self.package()
        delivery.save(self.root/"controller/finalizer/status.json",{"phase":"waiting"})
        self.assertIn("finalizer",self.gate()["errors"][0])

    def test_formal_issues_or_tampered_frozen_inputs_fail_closed(self):
        self.report["issues"]=["missing test"];self.write_report()
        self.assertFalse(self.gate()["ready"])
        self.report["issues"]=[];self.write_report()
        (self.root/"analysis/final_input/state.json").write_text("{}")
        self.assertIn("SHA",self.gate()["errors"][0])

    def test_canonical_zip_tampering_and_missing_export_receipt_fail_closed(self):
        path=self.job_dir/"task.zip"
        path.write_bytes(path.read_bytes()+b"tampering")
        self.assertFalse(self.gate()["ready"])

    def test_completed_training_cannot_lack_small_endpoint_exports(self):
        self.job["kind"]="train"
        prefix="20473/eval_06/"
        self.task_receipt["volumes"]=[make_volume(self.job_dir,"task.zip",{
            prefix+"job.json":raw(self.job),prefix+"status.json":raw({"stage":"completed"}),prefix+"config.json":b"{}"})]
        delivery.save(self.job_dir/"receipt.json",self.task_receipt)
        state_path=self.root/"analysis/final_input/state.json"
        state=delivery.read(state_path);state["jobs"]["test/06/BH/20473"]["kind"]="train"
        delivery.save(state_path,state)
        snapshot_path=self.root/"analysis/final_input/snapshot.json"
        snapshot=delivery.read(snapshot_path);snapshot["files"]["state"]["sha256"]=delivery.sha(state_path)
        delivery.save(snapshot_path,snapshot)
        prepared=delivery.read(self.root/"analysis/final_input/preparations.json")
        delivery.save(self.root/"analysis/final_input/archive_plan.json",archive.plan(state,prepared))
        gate=self.gate()
        self.assertFalse(gate["ready"])
        self.assertIn("endpoint export",gate["errors"][0])

    def test_space_and_large_raw_assets_are_rejected(self):
        with patch.object(delivery,"free_bytes",return_value=delivery.RESERVE-1):
            with self.assertRaisesRegex(RuntimeError,"8GiB"):
                delivery.package(self.root,self.report_path,roots=[self.dest])
        data=self.root/"teacher_reach100MB.json";data.write_bytes(b"x")
        assets=self.root/"assets.json"
        delivery.save(assets,{"experiment_id":"S1-ND12","files":[{"member":"teacher_reach.json","path":str(data),"sha256":delivery.sha(data)}]})
        with self.assertRaisesRegex(ValueError,"source_ref"):
            delivery._inputs(self.root,self.gate(),assets)

    def test_resume_stream_reuses_exact_prefix_after_disconnect(self):
        data=self.root/"package";data.write_bytes(b"abcdefghij")
        remote=FakeRemote(b"abc",fail_after=6)
        with self.assertRaisesRegex(OSError,"disconnected"):
            delivery.resume_stream(data,remote,"partial",10)
        self.assertEqual(remote.payload,b"abcdef")
        remote.fail_after=None
        result=delivery.resume_stream(data,remote,"partial",10)
        self.assertEqual(result["resumed_bytes"],6)
        self.assertEqual(remote.payload,b"abcdefghij")
        with self.assertRaisesRegex(ValueError,"prefix"):
            delivery.resume_stream(data,FakeRemote(b"bad"),"partial",10)

    def test_resume_upload_does_not_repackage_and_rejects_changed_input(self):
        package=self.package()
        self.report_path.write_text(self.report_path.read_text(encoding="utf-8")+"改变正式正文",encoding="utf-8")
        with patch.object(delivery,"package") as repackage:
            with self.assertRaisesRegex(RuntimeError,"no repackage"):
                delivery.resume_upload(package["local_receipt_path"],self.root,self.report_path)
            repackage.assert_not_called()

    def test_placement_never_accepts_physical74(self):
        value=json.loads(json.dumps(placement_transport.CURRENT));value["mapping"]["20474"]=20474;value["released_physical_ports"]=[]
        delivery.save(self.root/"controller/execution_placement.json",value)
        self.assertFalse(self.gate()["ready"])
        self.assertIn("only physical",self.gate()["errors"][0])

    def test_upload_uses_one_shared_package_and_reads_only_three_physical_hosts(self):
        package=self.package();data=Path(package["zip_path"]).read_bytes()
        class Remote(FakeRemote):
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def stat(self,path):
                if path.endswith(".zip"):raise FileNotFoundError(path)
                return super().stat(path)
            def posix_rename(self,old,new):
                self.renamed=(old,new)
        remote=Remote(data[:17])
        client=SimpleNamespace(open_sftp=lambda:remote,close=lambda:None)
        real=delivery.placement(self.root)
        observed=[]
        def verify(transport,port,path,receipt):
            observed.append(port)
            self.assertEqual(remote.payload,data)
            return {"physical_port":port,"device":1,"inode":2,"zip_sha256":package["zip_sha256"]}
        with patch.object(delivery,"placement",return_value=real), patch.object(real,"connect",return_value=client) as connect,patch.object(real,"remote_python",return_value={"exit_code":0}),patch.object(delivery,"_remote_verify",side_effect=verify),patch.object(delivery,"package") as repackage:
            result=delivery.resume_upload(package["local_receipt_path"],self.root,self.report_path)
            repackage.assert_not_called();connect.assert_called_once_with(20418)
        self.assertEqual(observed,[20418,20418,20424,20473])
        self.assertEqual(result["transfer"]["resumed_bytes"],17)
        self.assertTrue(result["all_three_physical_readback_verified"])
        self.assertEqual(set(result["physical_access"]),{"20418","20424","20473"})


if __name__=="__main__":
    unittest.main()
