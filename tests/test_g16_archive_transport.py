"""Synthetic bounded transfer tests; no SSH, production archive or computation."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile
from scripts.stage1_geoload import archive_transport as transfer


class Remote(io.BytesIO):
    def __init__(self,data,calls):super().__init__(data);self.calls=calls
    def settimeout(self,value):pass
    def readv(self,requests):
        self.calls.extend(requests)
        for offset,size in requests:self.seek(offset);yield self.read(size)


class SFTP:
    def __init__(self,transport):self.transport=transport
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def stat(self,path):return SimpleNamespace(st_size=len(self.transport.data))
    def open(self,path,mode):return Remote(self.transport.data,self.transport.calls)


class Client:
    def __init__(self,transport):self.transport=transport
    def open_sftp(self):return SFTP(self.transport)
    def close(self):pass


class Fake:
    def __init__(self,data):self.data=data;self.calls=[];self.ports=[]
    def connect(self,port):self.ports.append(port);return Client(self)


def fixture(size=1024):
    stream=io.BytesIO();content=b'a'*size
    manifest=dict(group='S1-G16',run_id='fixture',scientific_commit=transfer.SCIENCE,
        members={'job/test.bin':dict(sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))})
    raw=json.dumps(manifest,sort_keys=True).encode()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('job/test.bin',content);z.writestr('archive_manifest.json',raw)
    data=stream.getvalue()
    receipt=dict(path='/isolated/fixture.zip',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),run_id='fixture',port=20424,scientific_commit=transfer.SCIENCE)
    return data,receipt


class ArchiveTransportTests(unittest.TestCase):
    def test_2_and_4_mib_bounded_exact_windows(self):
        data,receipt=fixture(9 << 20)
        for window,expected in [(2 << 20,5),(4 << 20,3)]:
            with tempfile.TemporaryDirectory() as temp:
                fake=Fake(data);path=Path(temp)/'archive.zip'
                result=transfer.download(receipt,path,window=window,transport=fake,free_bytes=100 << 30)
                self.assertEqual(len(fake.calls),expected);self.assertTrue(all(size<=window for _,size in fake.calls))
                self.assertEqual(path.read_bytes(),data);self.assertTrue(result['verified']);self.assertEqual(fake.ports,[20424])

    def test_resume_bound_prefix_and_final_full_sha(self):
        data,receipt=fixture(3 << 20)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'archive.zip';partial=path.with_suffix('.zip.part');partial.write_bytes(data[:12345])
            path.with_suffix('.zip.part.binding.json').write_bytes(transfer.encoded(transfer.binding(receipt)))
            fake=Fake(data);result=transfer.download(receipt,path,transport=fake,free_bytes=100 << 30)
            self.assertEqual(fake.calls[0][0],12345);self.assertEqual(result['download_resumed_from_bytes'],12345)
            self.assertEqual(path.read_bytes(),data)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'archive.zip';path.with_suffix('.zip.part').write_bytes(b'badprefix')
            path.with_suffix('.zip.part.binding.json').write_bytes(transfer.encoded(transfer.binding(receipt)))
            with self.assertRaisesRegex(ValueError,'fullSHA'):transfer.download(receipt,path,transport=Fake(data),free_bytes=100 << 30)
            self.assertFalse(path.exists());self.assertTrue(path.with_suffix('.zip.part').exists())

    def test_unbound_partial_and_reserve_and_oversize_refused(self):
        data,receipt=fixture()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'archive.zip';path.with_suffix('.zip.part').write_bytes(data[:10])
            with self.assertRaisesRegex(ValueError,'bound'):transfer.download(receipt,path,transport=Fake(data),free_bytes=100 << 30)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'8GiB'):transfer.download(receipt,Path(temp)/'x.zip',transport=Fake(data),free_bytes=transfer.RESERVE)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'archive.zip';path.with_suffix('.zip.part').write_bytes(data+b'overflow')
            with self.assertRaisesRegex(ValueError,'exceeds'):transfer.download(receipt,path,transport=Fake(data),free_bytes=100 << 30)

    def test_manifest_identity_reuse_and_short_read(self):
        data,receipt=fixture()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'archive.zip';path.write_bytes(data)
            fake=Fake(data);value=transfer.download(receipt,path,transport=fake)
            self.assertTrue(value['download_reused']);self.assertEqual(fake.calls,[])
            bad={**receipt,'manifest_sha256':'0'*64}
            with self.assertRaisesRegex(ValueError,'manifest'):transfer.download(bad,path,transport=fake)
        class Short:
            def readv(self,requests):return [b'wrong']
        with self.assertRaisesRegex(ValueError,'short'):list(transfer.blocks(Short(),10))

    def test_source_ports_and_packer_worker_guards(self):
        _,receipt=fixture()
        for port in (20418,20474):
            with self.assertRaises(ValueError):transfer.binding({**receipt,'port':port})
        code=transfer.export_code(20424,'fixture',{'archive':'/isolated/archive','ops':'/isolated/ops'})
        compile(code,'remote_export','exec')
        compile(transfer.remote_verify_code(receipt),'remote_verify','exec')
        self.assertIn("pathlib.Path('/proc/'+str(pid)).exists()",code)
        self.assertIn('fcntl.LOCK_EX|fcntl.LOCK_NB',code)
        self.assertIn('unknown archive requires pinned receipt',code)
        self.assertIn('unknown packer/source reader active',code)
        self.assertNotIn('unlink(',code)


if __name__=='__main__':unittest.main()
