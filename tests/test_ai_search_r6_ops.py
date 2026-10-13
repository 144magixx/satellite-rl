import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r6.plan_builder import selected_indices,audit,DESIGN_SHA
from scripts.stage1_ai_search_r6.label_audit import gate
from satellite_rl.workbench.r6_io import write_asset,read_asset
from satellite_rl.workbench.ai_search_r6_runner import guarded
class R6OpsTests(unittest.TestCase):
 def test_exact_selection(self):
  x=selected_indices();self.assertEqual(len(x),768);self.assertEqual(len(set(i//4 for i in x)),192);self.assertEqual(x[:4],[0,1,2,3]);self.assertEqual(x[-4:],[2300,2301,2302,2303])
 def test_lossless_asset(self):
  with tempfile.TemporaryDirectory() as d:
   value={"scores":[[.123,0.,-2]],"role":"train"};ref=write_asset(Path(d)/"x.gz",value);self.assertEqual(read_asset(ref),value);self.assertEqual(len(ref["canonical_json_sha256"]),64)
 def test_unknown_wal_no_replay(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,"started.json").write_text("{}")
   with patch("satellite_rl.workbench.ai_search_r6_runner.run_stage") as stage:
    with self.assertRaisesRegex(ValueError,"unknown WAL"):guarded(d,{},15,lambda:False)
    stage.assert_not_called()
 def test_gate_fixed_denominator(self):
  with self.assertRaises(ValueError):gate([])
  rows=[{"qualified":i<692,"nontrivial":i<154,"capability":i<6,"mother_id":str(i)} for i in range(768)];self.assertTrue(gate(rows)["passed"]);rows[0]["qualified"]=False;self.assertFalse(gate(rows)["passed"])
class StagePipelineMocks(unittest.TestCase):
 def test_available_stage_guard_asset_chain(self):
  import numpy as np
  from types import SimpleNamespace
  from satellite_rl.workbench.ai_search_r6_stage import execute
  from satellite_rl.workbench.storage import digest
  import satellite_rl.coverage.ai_search_r5.core as core
  ident={"case_id":"train:m:base","mother_id":"m","profile":"base","role":"train","identity_sha256":"a"*64,"input_sha256":"b"*64}
  manifest={"stage":"labels","role":"train","admitted":True,"identity_sha256":"a"*64,"dataset_sha256":"c"*64,"cases":{ident["case_id"]:ident}}
  prep=SimpleNamespace(identity=ident,source_sha256="d"*64,to_dict=lambda:{"schema":"PreparedCaseV1","identity":ident,"status":"available"})
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);raw=root/"scene.npz";np.savez(raw,positions=np.zeros((1,2)),demands=[1.],aircraft_ids=["a"],profile="europe_v1")
   req={"identity":ident,"scene_ref":{"path":str(raw),"sha256":digest(raw)},"phase_manifest":manifest,"source_sha256":"d"*64,"producer_sha256":"e"*64}
   def fake_guard(command,directory,*args):
    execute(json.loads(Path(directory,"request.json").read_text()),directory)
    return {"status":"completed","output_ref":{"path":str(Path(directory,"stage_output.json")),"sha256":digest(Path(directory,"stage_output.json"))},"unknown_algorithm_spend":False}
   with patch("satellite_rl.coverage.ai_search.core.Scene",return_value=SimpleNamespace(input_sha256="b"*64)),patch.object(core,"prepare_case",return_value=prep) as preparation,patch.object(core.PreparedCase,"from_dict",return_value=prep),patch.object(core,"build_teacher",return_value={"status":"available","layout":{"widths_deg":[.4]}}) as teacher,patch.object(core,"build_proposal",return_value={"status":"ready","schema":"ProposalV2"}) as proposal,patch.object(core,"decode_groups",return_value={"status":"success"}) as decoder,patch("satellite_rl.workbench.ai_search_r6_runner.run_stage",side_effect=fake_guard):
    _,_,pref=guarded(root/"p",{**req,"stage":"preparation"},20,lambda:False)
    _,_,tref=guarded(root/"t",{**req,"stage":"teacher","prepared_ref":pref},45,lambda:False)
    _,value,_=guarded(root/"d",{**req,"stage":"decoder","prepared_ref":pref,"teacher_ref":tref},15,lambda:False)
    self.assertEqual(value["identity"],ident);self.assertEqual(value["result"]["status"],"success");self.assertEqual(preparation.call_args.kwargs["phase_manifest"],manifest);self.assertEqual(teacher.call_args.kwargs["phase_manifest"],manifest);self.assertEqual(proposal.call_args.args[1],"EXACT");decoder.assert_called_once()
    guarded(root/"p",{**req,"stage":"preparation"},20,lambda:False);preparation.assert_called_once()
    with self.assertRaisesRegex(ValueError,"original request changed"):guarded(root/"p",{**req,"stage":"preparation","source_sha256":"x"*64},20,lambda:False)

if __name__=="__main__":unittest.main()
