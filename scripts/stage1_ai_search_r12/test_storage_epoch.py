"""Pure metadata race mocks; no scientific payload or process execution."""
import stat
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from .runtime_epoch import BASE,ROOT
from .worker import _storage

class StorageTests(unittest.TestCase):
    def observe(self,name='node_abc.pending',directory=None,races=1,mode=stat.S_IFREG):
        directory=directory or ROOT+'/science/mother_0'
        calls=dict(walk=0,missing=0)
        def walk(*args,**kwargs):
            calls['walk']+=1
            names=['stable.json',name]
            if calls['walk']==2 and races==1:
                names=['stable.json'] if name=='owner.lock' else ['stable.json','published.json']
            return iter([(directory,[],names)])
        def lstat(path):
            if path==Path(BASE):return types.SimpleNamespace(st_mode=stat.S_IFDIR,st_size=0)
            if path.name=='published.json':return types.SimpleNamespace(st_mode=stat.S_IFREG,st_size=7)
            if path.name==name:
                if calls['missing']<races:
                    calls['missing']+=1;raise FileNotFoundError(str(path))
                return types.SimpleNamespace(st_mode=mode,st_size=7)
            return types.SimpleNamespace(st_mode=stat.S_IFREG,st_size=11)
        with patch('scripts.stage1_ai_search_r12.worker.os.walk',side_effect=walk),patch.object(Path,'lstat',autospec=True,side_effect=lstat),patch('scripts.stage1_ai_search_r12.worker.shutil.disk_usage',return_value=types.SimpleNamespace(free=10**10)),patch('builtins.print'):
            result=_storage(BASE)
        return result,calls
    def test_pending_publish_rescan_from_zero(self):
        result,calls=self.observe()
        self.assertEqual(calls['walk'],2);self.assertEqual(result['inventory_passes'],2)
        self.assertEqual(result['persistent_bytes'],18);self.assertEqual(result['temporary_bytes'],0)
        self.assertFalse(result['scientific_payload_audited']);self.assertIsNotNone(result['rescan'])
    def test_owner_release_rescan_allowed(self):
        result,_=self.observe('owner.lock',ROOT+'/service')
        self.assertEqual(result['persistent_bytes'],11)
        self.assertEqual(result['inventory_passes'],2)
    def test_second_collision_fails(self):
        with self.assertRaisesRegex(RuntimeError,'repeated'):self.observe(races=2)
    def test_ordinary_missing_fails(self):
        with self.assertRaisesRegex(RuntimeError,'missing'):self.observe('node_abc.json')
    def test_outside_exact_journal_or_illegal_key_fails(self):
        for directory,name in ((ROOT+'/science/mother_1','node_abc.pending'),(BASE+'/engineering','node_abc.pending'),(ROOT+'/science/mother_0','bad.key.pending')):
            with self.assertRaisesRegex(RuntimeError,'missing'):self.observe(name,directory)
    def test_links_nonregular_never_rescan_or_count(self):
        for mode in (stat.S_IFLNK,stat.S_IFIFO,stat.S_IFSOCK):
            with self.assertRaisesRegex(RuntimeError,'nonregular'):self.observe(races=0,mode=mode)
    def test_stable_first_pass_has_no_rescan(self):
        result,calls=self.observe(races=0)
        self.assertEqual(calls['walk'],1);self.assertIsNone(result['rescan'])

if __name__=='__main__':unittest.main()

