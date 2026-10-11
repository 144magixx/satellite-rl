"""Placement closure and routed tunnel checks without starting helper processes."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_nondrl12 import archive, finalizer, tunnel, tunnel_placement
from scripts.stage1_nondrl12.placement_transport import CURRENT, Transport


class FinalizerPlacementTests(unittest.TestCase):
    def test_default_existing_placement_routes_archives_and_preserves_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / "controller/finalizer"
            directory.mkdir(parents=True)
            mapping = root / "controller/execution_placement.json"
            mapping.write_text(json.dumps(CURRENT))
            deadline = 1791921433.3436897
            (directory / "status.json").write_text(json.dumps({"phase": "waiting", "deadline_unix": deadline,
                "started_at": "original_start", "module_sha256": {"original": "sha"}}))
            original = archive.ops
            seen = []
            def finalize_mock(root, paths, guard, progress):
                guard()
                seen.append(archive.ops)
                self.assertIsInstance(archive.ops, Transport)
                self.assertEqual(archive.ops.physical_port(20474), 20473)
                return {"closed": True}
            with patch.object(finalizer, "reserve"), patch.object(finalizer, "readiness", return_value=True), patch.object(finalizer, "finalize", side_effect=finalize_mock), redirect_stdout(StringIO()):
                result = finalizer.run(root, resume_after_attention=True)
            self.assertEqual(result["deadline_unix"], deadline)
            self.assertEqual(result["started_at"], "original_start")
            self.assertEqual(result["physical_ports"], [20418, 20424, 20473])
            self.assertEqual(result["logical_to_physical"][20474], 20473)
            self.assertIn(str(mapping.resolve()), result["module_sha256"])
            self.assertTrue(any(path.endswith("placement_transport.py") for path in result["module_sha256"]))
            self.assertIs(archive.ops, original)
            self.assertEqual(len(seen), 1)

    def test_placement_mutation_stops_before_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "controller").mkdir()
            mapping = root / "controller/execution_placement.json"
            mapping.write_text(json.dumps(CURRENT))
            def readiness(root, paths):
                mapping.write_text(json.dumps(CURRENT, indent=2))
                return True
            def finalize_mock(root, paths, guard, progress):
                guard()
                self.fail("mutated placement reached archive")
            with patch.object(finalizer, "reserve"), patch.object(finalizer, "readiness", side_effect=readiness), patch.object(finalizer, "finalize", side_effect=finalize_mock), patch.object(archive, "one_task") as task, patch.object(archive, "one_source") as source, redirect_stdout(StringIO()):
                result = finalizer.run(root)
            self.assertEqual(result["phase"], "attention_required")
            task.assert_not_called(); source.assert_not_called()

    def test_routed_tunnel_keeps_logical74_api8943(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "placement.json"
            path.write_text(json.dumps(CURRENT))
            original = tunnel.ops
            def stub(argv):
                self.assertEqual(argv, ["--ssh-port", "20474"])
                self.assertIsInstance(tunnel.ops, Transport)
                self.assertEqual(tunnel.ops.physical_port(20474), 20473)
                self.assertEqual(tunnel.mapping(20474)["local_port"], 8943)
                return "mock_no_listener"
            with patch.object(tunnel, "main", side_effect=stub):
                result = tunnel_placement.main(["--placement-file", str(path), "--ssh-port", "20474"])
            self.assertEqual(result, "mock_no_listener")
            self.assertIs(tunnel.ops, original)


if __name__ == "__main__":
    unittest.main()
