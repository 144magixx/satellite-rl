"""The S2-P19 page is reachable without changing the legacy app router."""

from html.parser import HTMLParser
import json
import tempfile
import threading
import unittest
from urllib.request import urlopen

from satellite_rl.workbench.server import WorkbenchServer
from satellite_rl.workbench.service import Application


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href"))
        if tag == "script":
            self.scripts.append(attrs.get("src"))


class S2P19FrontendTests(unittest.TestCase):
    def test_page_is_served_and_exposes_required_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            application = Application(directory, source_root=directory)
            server = WorkbenchServer(("127.0.0.1", 0), application)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/", timeout=5) as response:
                    index = Elements()
                    index.feed(response.read().decode("utf-8"))
                self.assertIn("/s2p19.html", index.links)
                with urlopen(base + "/s2p19.html", timeout=5) as response:
                    self.assertIn("text/html", response.headers["Content-Type"])
                    page = Elements()
                    page.feed(response.read().decode("utf-8"))
                self.assertEqual(len(page.ids), len(set(page.ids)))
                self.assertIn("/s2p19.js", page.scripts)
                for control in ("s2-preparation-type", "s2-source-sha", "s2-arm",
                                "s2-profile", "s2-teacher-path", "s2-teacher-sha",
                                "s2-train-scene", "s2-validation-scene", "s2-pilot",
                                "s2-total", "s2-i0-mode", "s2-gate-run",
                                "s2-gate-audit-sha", "s2-sca-source", "s2-sca-partition",
                                "s2-sca-step", "s2-run-logs", "s2-run-results"):
                    self.assertIn(control, page.ids)
                with urlopen(base + "/s2p19.js", timeout=5) as response:
                    self.assertIn("javascript", response.headers["Content-Type"])
                    script = response.read().decode("utf-8")
                for route in ("/api/s2p19/catalog", "/api/s2p19/runs",
                              "/api/s2p19/prepare", "/api/s2p19/train",
                              "/api/s2p19/improvement", "/approve-pilot",
                              "/api/power-refinement/runs", "/api/validate"):
                    self.assertIn(route, script)
                with urlopen(base + "/api/s2p19/catalog", timeout=5) as response:
                    catalog = json.load(response)
                self.assertEqual(catalog["experiment_id"], "S2-P19")
                self.assertEqual(set(catalog["defaults"]), {"C0", "C1", "I0"})
                with urlopen(base + "/api/s2p19/runs", timeout=5) as response:
                    self.assertIn("runs", json.load(response))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
