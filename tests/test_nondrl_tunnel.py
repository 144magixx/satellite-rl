"""Narrow infrastructure checks; no remote API mutation or scientific work."""
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_nondrl12.tunnel import mapping, singleton, atomic


class NondrlTunnelTests(unittest.TestCase):
    def test_only_fixed_loopback_mapping(self):
        for ssh, api in zip((20418, 20424, 20473, 20474), (8940, 8941, 8942, 8943)):
            actual = mapping(ssh)
            self.assertEqual((actual["local_address"], actual["remote_address"]), ("127.0.0.1", "127.0.0.1"))
            self.assertEqual((actual["local_port"], actual["remote_port"]), (api, api))
        for invalid in (20417, 8920, "20418", True):
            with self.assertRaises(ValueError):
                mapping(invalid)

    def test_singleton_refuses_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with singleton(tmp):
                with self.assertRaises(OSError):
                    with singleton(tmp):
                        self.fail("duplicate lane lock acquired")
            with singleton(tmp):
                atomic(Path(tmp) / "status.json", {"stage": "unit_fixture"})
            self.assertTrue((Path(tmp) / "status.json").is_file())


if __name__ == "__main__":
    unittest.main()
