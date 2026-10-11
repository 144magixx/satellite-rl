"""Safety/idempotence simulation only: no SSH, signal or live migration."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch
from contextlib import redirect_stdout
from io import StringIO
from scripts.stage1_nondrl12 import relocation as move


class RelocationTests(unittest.TestCase):
    def client(self,root):
        path=root/'client.json'
        move.save(path,{'experiment_id':'S1-ND12','pid':38528,'process_start_time':100.,'exited':True,
                        'command':['python','-m','scripts.stage1_nondrl12.launcher_pipeline']})
        return path,move.digest(path)

    def test_client_exact_birth_and_release_hard_gate(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);path,sha=self.client(root)
            with patch.object(move,'pid_alive',return_value=True):
                with self.assertRaisesRegex(ValueError,'alive'):move.verify_client_release(path,sha)
            with patch.object(move,'pid_alive',return_value=False):move.verify_client_release(path,sha)
            record={'experiment_id':'S1-ND12','migration_id':move.MIGRATION_ID,'source_commit':move.SOURCE,
                'run_id':move.RUN_ID,'logical_ssh_port':20474,'released_physical_port':20474,'new_physical_ssh_port':20473,
                'queued_count':0,'released':True,'old_owner_absent':True,'old_service_closed':True,'old_worker_exited':True}
            move.validate_release(record)
            for key in ('released','old_owner_absent','old_service_closed','old_worker_exited'):
                bad=dict(record);bad[key]=False
                with self.assertRaises(ValueError):move.validate_release(bad)

    def test_remote_programs_compile_and_force_is_opt_in(self):
        for code in (move.INSPECT,move.RELEASE,move.ACTIVATE):compile(code,'migration_remote','exec')
        self.assertLess(move.RELEASE.index("api('/api/runs/'"),move.RELEASE.index('signal.SIGTERM'))
        self.assertNotIn('killpg',move.RELEASE)
        self.assertIn("if not r['allow_force']",move.RELEASE)
        self.assertIn("SATELLITE_RL_SSH_PORT='20474'",move.ACTIVATE)
        self.assertIn("SATELLITE_RL_PHYSICAL_SSH_PORT='20473'",move.ACTIVATE)
        self.assertLess(move.ACTIVATE.index("api('/api/dispatch/hold/release'"),move.ACTIVATE.index("resume=api('/api/runs/'"))

    def test_native_identity_null_arguments_and_pid_birth_guard(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);fake_proc=root/'fakeproc';process=fake_proc/'77';process.mkdir(parents=True)
            values=['0']*22;values[0]='R';values[11]='130';values[12]='20';values[19]='1000';values[21]='50'
            (process/'stat').write_text('77 (python) '+' '.join(values),encoding='utf-8')
            (fake_proc/'stat').write_text('btime 100\n',encoding='utf-8')
            command=['python','-u','-m','satellite_rl.workbench.runner','--job-dir','/exact/job']
            (process/'cmdline').write_bytes(b'\0'.join(v.encode() for v in command)+b'\0')
            ns={'REQUEST_JSON':json.dumps(move.request(root=str(root)))}
            with patch.dict(sys.modules,{'fcntl':SimpleNamespace()}):exec(move.COMMON,ns)
            original_path=Path
            ns['pathlib']=SimpleNamespace(Path=lambda value:original_path(str(value).replace('/proc',str(fake_proc),1)))
            kill=Mock();ns['os']=SimpleNamespace(sysconf=lambda key:4096 if key=='SC_PAGE_SIZE' else 100,kill=kill)
            actual=ns['proc'](77)
            self.assertEqual(actual['command'],command);self.assertEqual(actual['process_start_time'],110.)
            with self.assertRaisesRegex(RuntimeError,'birth'):
                ns['send_exact'](77,111.,command,15)
            kill.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'command'):
                ns['send_exact'](77,110.,['another_job'],15)
            kill.assert_not_called()
            ns['send_exact'](77,110.,command,15);kill.assert_called_once_with(77,15)

    def test_activation_reuses_guarded_service_and_no_repeat_resume(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);archive=root/'archive';jobdir=archive/'outputs/workbench/jobs'/move.RUN_ID
            jobdir.mkdir(parents=True);receipt_dir=root/'ops/relocation'/move.MIGRATION_ID;receipt_dir.mkdir(parents=True)
            rows=[{'key':'0:1','file':'scene_000000.json.gz','sha256':'b'*64}]
            retained={'provenance':{'source':'same'},'rows':rows,'committed_count':1,'manifest_sha256':'a'*64}
            (jobdir/'config.json').write_text('{}',encoding='utf-8')
            release={'released':True,'old_owner_absent':True,'old_service_closed':True,'old_worker_exited':True,
                'client_release_sha256':'c'*64,'source_commit':move.SOURCE,'config_sha256':move.digest(jobdir/'config.json'),
                'retained_ledger':retained,'before':{'launch':{'process_start_time':100.}}}
            move.save(receipt_dir/'release_receipt.json',release)
            sha=move.digest(receipt_dir/'release_receipt.json')
            command=[move.ops.DQN,'-B','-m','satellite_rl.workbench.server','--archive-root',str(archive),
                     '--port','8943','--python',move.ops.DQN,'--max-workers','2']
            service={'pid':44,'start_ticks':200,'process_start_time':200.,'source_commit':move.SOURCE,
                'logical_ssh_port':20474,'physical_ssh_port':20473,'relocation_receipt_sha256':sha,
                'archive_root':str(archive),'actual_execution_target':'u2024110224@10.160.17.102:20473'}
            move.save(root/'ops/service.json',service)
            request=move.request(root=str(root),release_sha256=sha,client_release_sha256='c'*64,workers=2)
            fake_fcntl=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=Mock())
            for stage,expected_resume in (('interrupted',1),('queued',0),('evaluating',0),('completed',0)):
                move.save(jobdir/'status.json',{'stage':stage})
                move.save(jobdir/'launch.json',{'pid':55,'process_start_time':300.,'command':['worker']})
                ns={'REQUEST_JSON':json.dumps(request)}
                with patch.dict(sys.modules,{'fcntl':fake_fcntl}):exec(move.COMMON,ns)
                ns['listening']=lambda:True
                ns['ledger']=lambda:{**retained,'rows':rows+[{'key':'1:1','file':'new','sha256':'d'*64}],'committed_count':2,'manifest_sha256':'different_after_resume'}
                ns['identity']=lambda pid,birth=None,cmd=None,ticks=None:{'pid':pid,'alive':True,'command':command if pid==44 else ['worker'],'process_start_time':birth}
                calls=[]
                def api(path,body=None):
                    calls.append(path)
                    return {'active':False} if '/hold/release' in path else {'stage':'queued'}
                ns['api']=api
                def run(argv,**kwargs):
                    if argv[0]=='git':return SimpleNamespace(stdout=move.SOURCE+'\n' if 'rev-parse' in argv else '')
                    return SimpleNamespace(stdout=json.dumps({'cuda':True,'package':str(root/'repo/src/satellite_rl/__init__.py')}))
                fake_subprocess=SimpleNamespace(run=run,Popen=Mock(side_effect=AssertionError('must not double spawn')))
                ns['subprocess']=fake_subprocess
                with redirect_stdout(StringIO()):exec(move.ACTIVATE[len(move.COMMON):],ns)
                self.assertEqual(sum(path.endswith('/resume') for path in calls),expected_resume)
                self.assertLess(calls.index('/api/dispatch/hold/release'),len(calls)-expected_resume+1)
                value=move.read(receipt_dir/'activation_receipt.json')
                self.assertTrue(value['reused_service']);self.assertEqual(value['committed_rows_retained'],1)
                self.assertEqual(value['committed_rows_after'],2)
                fake_subprocess.Popen.assert_not_called()

    def test_activate_blocks_reappeared_old_owner_before_new_remote(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);client,csha=self.client(root);release=root/'release.json'
            value={'experiment_id':'S1-ND12','migration_id':move.MIGRATION_ID,'source_commit':move.SOURCE,
                'run_id':move.RUN_ID,'logical_ssh_port':20474,'released_physical_port':20474,'new_physical_ssh_port':20473,
                'queued_count':0,'released':True,'old_owner_absent':True,'old_service_closed':True,'old_worker_exited':True}
            move.save(release,value)
            with patch.object(move,'pid_alive',return_value=False),patch.object(move,'remote',return_value={'listening':True,'processes':[],'queued':[]}) as remote:
                with self.assertRaisesRegex(ValueError,'reappeared'):move.activate(client,csha,release,move.digest(release))
                self.assertEqual(remote.call_count,1)


if __name__=='__main__':unittest.main()
