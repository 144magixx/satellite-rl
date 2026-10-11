"""Bounded inconclusive reporting and small immutable helper snapshots."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_nondrl12 import finalizer, report
from scripts.stage1_nondrl12.execution_sources import ORIGIN_COMMIT
from test_nondrl_execution_sources import raw,sha
from test_nondrl_finalizer import fixture


def evidence(root):
    run="prepare_nondrl_20261012_045228_1d87b8e0"
    receipt={"status":"attention_required","passed":False,"scientific_stop":False,
        "reason":"offline_teacher_search_evidence_gap: bounded unknown/reference-only problems remain",
        "statistics":{"actual_complete_problems":60,"legal_reference_mothers":list(range(20)),"improved_mothers":[0,1],
            "unknown_search_problems":7,"recorded_teacher_seconds":614.,"true_trials":617},
        "task_source":{"run_id":run,"contract_sha256":"c"*64,"wrapper_sha256":"4"*64,"evidence_manifest_sha256":"5"*64}}
    helper=root/"helper.json";helper.write_bytes(raw(receipt))
    item={"run_id":run,"status":"budget_limited_inconclusive","reason":"bounded evidence insufficiency",
        "teacher_wrapper_sha256":"4"*64,"teacher_evidence_manifest_sha256":"5"*64,
        "helper_receipt_path":str(helper),"helper_receipt_sha256":sha(helper.read_bytes())}
    scope={"format":"s1_nd12_bounded_dispositions_v1","origin_commit":ORIGIN_COMMIT,"contract_sha256":"c"*64,
        "scientific_stop":False,"entries":{"02":item}}
    path=root/"disposition.json";path.write_bytes(raw(scope))
    rows=[{"dataset_index":index,"multiplier":level,"initial_cv":1.,"load_cv_all":.9 if index<2 else 1.,"hard_feasible":True}
        for index in range(20) for level in (.75,1.,1.25)]
    audit={"key":"teacher_probe/02","state":"completed","source_lineage":{"contract_sha256":"c"*64},
        "teacher_probe":{"wrapper_sha256":"4"*64,"evidence_manifest_sha256":"5"*64,"complete_problems":60,
            "teacher_recorded_seconds":614.,"rows":rows,"unknown_search_problems":7,"searched_problems":53,
            "reference_only_problems":7,"legacy_problems_without_explicit_scope":53,"explicit_complete_search_problems":0}}
    gate={"route":"02","status":"budget_limited_inconclusive","passed":False,"scientific_stop":False,
        "disposition_source":{"path":str(path),"sha256":sha(path.read_bytes())},"teacher_reachability":receipt,
        "original_helper_receipt_source":{"path":str(helper),"sha256":sha(helper.read_bytes())}}
    return gate,{run:audit}


class BoundedClosureTests(unittest.TestCase):
    def test_exhausted_valid_references_stay_inconclusive_and_cannot_get_training(self):
        with tempfile.TemporaryDirectory() as folder:
            gate,audits=evidence(Path(folder))
            result=report.verify_bounded_teacher_disposition(gate,audits)
            self.assertEqual(result["status"],"budget_limited_inconclusive")
            self.assertFalse(result["scientific_stop"])
            self.assertEqual(result["legacy_problems_without_explicit_scope"],53)
            self.assertEqual(result["explicit_complete_search_problems"],0)
            self.assertFalse(result["training_performed"])
            bad=deepcopy(audits);bad["unallowed_train"]={"key":"train/02"}
            with self.assertRaisesRegex(ValueError,"training"):
                report.verify_bounded_teacher_disposition(gate,bad)

    def test_changed_pin_threshold_denominator_or_budget_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            gate,audits=evidence(Path(folder));run=next(iter(audits))
            for key,value in (("evidence_manifest_sha256","0"*64),("complete_problems",59),("teacher_recorded_seconds",599.),("unknown_search_problems",0)):
                with self.subTest(key=key):
                    bad=deepcopy(audits);bad[run]["teacher_probe"][key]=value
                    with self.assertRaises(ValueError):report.verify_bounded_teacher_disposition(gate,bad)
            bad=deepcopy(audits)
            for row in bad[run]["teacher_probe"]["rows"]:
                if row["dataset_index"]<6:row["load_cv_all"]=.9
            with self.assertRaisesRegex(ValueError,"quality"):
                report.verify_bounded_teacher_disposition(gate,bad)

    def test_large_original_helper_is_not_copied_into_final_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=fixture(root)
            helper=root/"controller/teacher_reachability/route_02.json";helper.parent.mkdir(parents=True)
            value={"status":"attention_required","statistics":{"problem_metrics":[{"dataset_index":0,
                "result":{"physical":{"per_aircraft":[1]*600000}}}]},"sources":[{"path":"teacher canonical","sha256":"a"*64}]}
            original=raw(value);helper.write_bytes(original)
            with patch.object(finalizer,"reserve"):
                snapshot=finalizer.freeze_inputs(root,paths)
            item=snapshot["files"]["teacher_reachability/route_02.json"]
            self.assertLess(item["bytes"],4096)
            saved=json.loads(Path(item["path"]).read_bytes())
            self.assertEqual(saved["original_receipt_source"]["sha256"],sha(original))
            self.assertEqual(helper.read_bytes(),original)
            self.assertNotIn("per_aircraft",json.dumps(saved))


if __name__=="__main__":unittest.main()
