import hashlib
import unittest
from scripts.stage1_drl_r04.archive import remote_blocks


class VectorReadTests(unittest.TestCase):
    def test_bounded_windows_keep_byte_identity_and_reject_short_read(self):
        data = bytes(range(256)) * 23000
        calls = []
        class Remote:
            def readv(self, chunks):
                calls.extend(chunks)
                return [data[offset:offset + size] for offset, size in chunks]
        blocks = list(remote_blocks(Remote(), len(data)))
        self.assertEqual(hashlib.sha256(b"".join(blocks)).digest(), hashlib.sha256(data).digest())
        self.assertTrue(all(size <= 2 << 20 for _, size in calls))
        self.assertEqual([offset for offset, _ in calls], [0, 2 << 20, 4 << 20])
        class ShortRemote:
            def readv(self, chunks):
                return [b"short"]
        with self.assertRaisesRegex(ValueError, "short"):
            list(remote_blocks(ShortRemote(), len(data)))

if __name__ == "__main__":
    unittest.main()
