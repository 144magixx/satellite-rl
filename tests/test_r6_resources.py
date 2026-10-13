import unittest
from satellite_rl.workbench.r6_resources import lane_from_archive

class ResourceScopeTests(unittest.TestCase):
    def test_actual_nested_archives(self):
        for port in (20418,20424,20473,20474):
            for suffix in ('round6/archive','round6/learning/archive'):
                self.assertEqual(lane_from_archive(f'/home/u2024110224/first_stage_satellite_rl/s1_ai17/{port}/{suffix}'),port)
    def test_other_scope_and_traversal_rejected(self):
        for path in ('/tmp/20418/round6/archive','/home/u2024110224/first_stage_satellite_rl/s1_ai17/20418/archive',
                     '/home/u2024110224/first_stage_satellite_rl/s1_ai17/20418/round6/archive/../archive',
                     '/home/u2024110224/first_stage_satellite_rl/s1_ai17/20419/round6/archive'):
            with self.subTest(path=path),self.assertRaises(ValueError):lane_from_archive(path)

if __name__=='__main__':unittest.main()
