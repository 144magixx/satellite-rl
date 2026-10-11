"""Executor routing preserves logical API/archive identity and released resources."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_nondrl12 import placement_transport as placement


class PlacementTests(unittest.TestCase):
    def test_api74_uses_physical73_and_api8943_with_identical_body(self):
        transport = placement.Transport()
        body = {"kind": "evaluate", "request_id": "existing_fixed_id", "dataset_ids": {"test": "unchanged"}}
        with patch.object(placement.ops, "remote_python", return_value={"exit_code": 0, "stdout": json.dumps({"status": 200, "body": {"run_id": "same"}}), "stderr": ""}) as call:
            self.assertEqual(transport.api(20474, "/api/coverage/nondrl/runs", body), {"run_id": "same"})
            args = call.call_args.args
            self.assertEqual(args[1], 20473)
            self.assertIn("127.0.0.1:8943/api/coverage/nondrl/runs", args[0])
            self.assertIn("existing_fixed_id", args[0])
            self.assertIn("unchanged", args[0])
        self.assertIn("/20474/", transport.instance(20474)["archive"])

    def test_connect_upload_and_compute_never_open_released74(self):
        transport = placement.Transport()
        with patch.object(placement.ops, "connect") as connect, patch.object(placement.ops, "upload") as upload, patch.object(placement.ops, "remote_python") as compute:
            transport.connect(20474)
            transport.upload("local", placement.ops.BASE + "/20474/archive/proof.json", 20474)
            transport.remote_python("read-only or archive code", 20474)
            connect.assert_called_once_with(20473)
            self.assertEqual(upload.call_args.args[2], 20473)
            self.assertEqual(compute.call_args.args[1], 20473)
            self.assertIn("/20474/", upload.call_args.args[1])

    def test_released_mapping_and_mutated_bytes_fail_closed(self):
        invalid = deepcopy(placement.CURRENT); invalid["mapping"]["20474"] = 20474
        with self.assertRaises(ValueError):
            placement.Transport(invalid)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "placement.json"
            path.write_text(json.dumps(placement.CURRENT))
            transport = placement.Transport.from_file(path)
            self.assertEqual(transport.physical_port(20474), 20473)
            path.write_text(json.dumps(placement.CURRENT, indent=2))
            with self.assertRaises(RuntimeError):
                transport.physical_port(20474)


if __name__ == "__main__":
    unittest.main()
