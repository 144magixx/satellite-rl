import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from satellite_rl.workbench.fair20_download import acquire_asset


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Length': str(len(data))}
        if content_range:
            self.headers['Content-Range'] = content_range


class Fair20DownloadTests(unittest.TestCase):
    def asset(self):
        tag = 'v2024.02.13-planes-readsb-staging-0'
        name = tag + '.tar'
        url = 'https://github.com/adsblol/globe_history_2024/releases/download/' + tag + '/' + name
        entry = {'id': 1, 'name': name, 'size': 6, 'browser_download_url': url, 'digest': 'sha256:' + hashlib.sha256(b'abcdef').hexdigest()}
        metadata = {'tag_name': tag, 'assets': [entry]}
        canonical = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        return {'date': '2024-02-13', 'release_tag': tag, 'asset_name': name, 'asset_id': 1, 'url': url, 'expected_bytes': 6, 'official_metadata_sha256': 'a' * 64, 'official_metadata_canonical_sha256': canonical, 'official_metadata': metadata}

    def test_download_then_immutable_resume_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('urllib.request.urlopen', return_value=Response(b'abcdef')) as call:
                first = acquire_asset(directory, self.asset())
                self.assertEqual(call.call_count, 1)
            with patch('urllib.request.urlopen', side_effect=AssertionError('must reuse verified source')):
                second = acquire_asset(directory, self.asset())
            self.assertEqual(first['sha256'], second['sha256'])

    def test_range_resume_preserves_exact_bytes(self):
        asset = self.asset()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'mother_sources/raw'
            raw.mkdir(parents=True)
            (raw / (asset['asset_name'] + '.part')).write_bytes(b'abc')
            with patch('urllib.request.urlopen', return_value=Response(b'def', 206, 'bytes 3-5/6')) as call:
                result = acquire_asset(directory, asset)
                self.assertEqual(call.call_args.args[0].headers['Range'], 'bytes=3-')
            self.assertEqual(Path(result['path']).read_bytes(), b'abcdef')

    def test_changed_official_metadata_rejected_before_network(self):
        asset = self.asset()
        asset['expected_bytes'] = 7
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen') as call:
            with self.assertRaises(ValueError):
                acquire_asset(directory, asset)
            call.assert_not_called()

    def test_wrong_range_never_appended(self):
        asset = self.asset()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'mother_sources/raw'
            raw.mkdir(parents=True)
            partial = raw / (asset['asset_name'] + '.part')
            partial.write_bytes(b'abc')
            with patch('urllib.request.urlopen', return_value=Response(b'def', 206, 'bytes 2-4/6')):
                with self.assertRaises(ValueError):
                    acquire_asset(directory, asset)
            self.assertEqual(partial.read_bytes(), b'abc')


if __name__ == '__main__':
    unittest.main()
