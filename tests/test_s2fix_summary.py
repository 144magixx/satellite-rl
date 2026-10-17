"""Synthetic seed weighting and date/scene paired bootstrap regression."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd


class SummaryTests(unittest.TestCase):
    def test_seed_means_precede_scene_pairing_and_date_bootstrap(self):
        script = Path(__file__).resolve().parents[1] / "scripts/s2fix/summarize.py"
        spec = importlib.util.spec_from_file_location("s2fix_summary", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = []
        for scene in range(10):
            offset = 0 if scene < 9 else 30
            common = dict(set="synthetic", scene_id=str(scene),
                          date="2024-02-23" if scene < 9 else "2024-02-24")
            rows.append(dict(common, method="G1", seed=0, repeat=0, after_W=0.))
            # Unequal repetition counts must not give seed 1 extra weight.
            for seed, repeat, value in ((1,0,0), (1,1,20), (2,0,50)):
                rows.append(dict(common,method="L1",seed=seed,repeat=repeat,after_W=value+offset))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.csv"
            target = Path(directory) / "summary"
            pd.DataFrame(rows).to_csv(source,index=False)
            with patch("sys.argv", [str(script), "--input", str(source), "--output", str(target)]):
                module.main()
            scenes = pd.read_csv(target / "scene_means.csv")
            learned = scenes[scenes.method == "L1"]
            self.assertAlmostEqual(learned.after_W.mean(), 33.)
            paired = pd.read_csv(target / "paired_differences.csv").iloc[0]
            self.assertEqual(paired.scenes, 10)
            self.assertAlmostEqual(paired.difference, 33.)
            # Resampling dates can select the small second date twice; a flat
            # bootstrap over ten scenes would have a much narrower upper tail.
            self.assertAlmostEqual(paired.ci95_low, 30.)
            self.assertAlmostEqual(paired.ci95_high, 60.)


if __name__ == "__main__":
    unittest.main()
