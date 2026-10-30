import gzip
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np
from satellite_rl.coverage.s1fix.data import extract_day, load_day


class DataTests(unittest.TestCase):
    def test_latest_invalid_blocks_older_position_and_grids(self):
        date = "2024-01-08"
        midnight = datetime(2024, 1, 8, tzinfo=timezone.utc).timestamp()
        traces = {
            "000001": [[60, 50, 10, 30000, 400, 90, 0], [110, 50, 10, "ground", 0, 90, 0]],
            "000002": [[65, 50, 10, 30000, None, None, 0]],
            "000003": [[65, 50, 10, 30000, 400, 90, 0], [115, 50, 10, 30000, 400, 90, 1]],
            "000004": [[65, 50, 10, 30000, 400, 90, 0], [115, None, 10, 30000, 400, 90, 0]],
            "000005": [[59, 50, 10, 30000, 400, 90, 0]],
            "000006": [[21600, 50, 10, 30000, 400, 90, 0]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "day.tar"
            with tarfile.open(path, "w") as archive:
                for icao, trace in traces.items():
                    content = gzip.compress(json.dumps(dict(icao=icao, timestamp=midnight, trace=trace)).encode())
                    entry = tarfile.TarInfo(f"trace_full_{icao}.json")
                    entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
            extract_day(date, path, tmp)
            static, dynamic = load_day(tmp, date), load_day(tmp, date, "dynamic")
            self.assertEqual((len(static), len(dynamic)), (288, 961))
            np.testing.assert_array_equal(static[0].icao, [2])
            self.assertTrue(np.isnan(static[0].gs_kt[0]))
            np.testing.assert_array_equal(dynamic[0].icao, [6])
            np.testing.assert_array_equal(dynamic[1].icao, [6])
            self.assertEqual(len(dynamic[2].icao), 0)


if __name__ == "__main__":
    unittest.main()
