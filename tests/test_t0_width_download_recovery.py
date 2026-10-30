import json
from pathlib import Path
from unittest.mock import patch
from test_t0_width_workbench import WorkbenchTests
from satellite_rl.t0_width.workbench import sha,read,boot_id,birth
from satellite_rl.t0_width.logging import atomic_json,content_sha
import os

class DownloadRecoveryTests(WorkbenchTests):
    def recovery_fixture(self):
        canonical=self.root/'official';canonical.mkdir()
        assets=[];records=[];provenance=[]
        admission=self.root/'download_source.json'
        for i in range(2):
            day='2024-01-'+str(i+10);path=canonical/(day+'.tar');path.write_bytes(('official'+str(i)).encode())
            asset=dict(date=day,source_url='https://api.github.com/repos/adsblol/globe_history_2024/releases/assets/'+str(i+1),release_asset_id=i+1,release_asset_bytes=path.stat().st_size)
            assets.append(asset);record=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path));records.append(record)
            provenance.append(dict(**record,date=day,source_url=asset['source_url'],release_asset_id=i+1))
        atomic_json(admission,dict(schema='W26OfficialDownloadAdmissionV1',root_approved=True,remote_port=20418,canonical_directory=str(canonical),assets=assets))
        config=dict(request_id='original-download',download_admission_ref=dict(path=str(admission),sha256=sha(admission)))
        oldplan=read(self.plan);oldplan['tasks']={'D':dict(executor_port=20418,mode='download',config=config)}
        oldpath=self.root/'oldplan.json';atomic_json(oldpath,oldplan)
        self.workbench.plan['tasks']['D']=oldplan['tasks']['D']
        with patch.object(self.workbench,'launch',side_effect=self.fake_launch):status=self.workbench.submit(dict(request_id='original-download',task_key='D',plan_sha256=self.workbench.plan_sha))
        directory=Path(status['run_dir']);job=read(directory/'job.json')
        status.update(status='failed',plan_sha256=sha(oldpath));atomic_json(directory/'status.json',status)
        job['status']['plan_sha256']=sha(oldpath);atomic_json(directory/'job.json',job)
        for asset in assets:
            atomic_json(canonical/(asset['date']+'.owner.json'),dict(request_id=config['request_id'],source_admission_SHA=sha(admission),source_url=asset['source_url'],release_asset_id=asset['release_asset_id'],release_asset_bytes=asset['release_asset_bytes']))
        transport=self.root/'transport.json';atomic_json(transport,dict(schema='W26OfficialTransportReceiptV1',assets=provenance))
        receipt=self.root/'recovery.json';atomic_json(receipt,dict(schema='W26DownloadTransportRecoveryV1',root_approved=True,passed=True,experiment_id='S2-W26',executor_port=20418,run_id=status['run_id'],original_job_SHA=sha(directory/'job.json'),original_config_SHA=content_sha(read(directory/'config.json')),original_plan_ref=dict(path=str(oldpath),sha256=sha(oldpath)),canonical_assets=records,transport_receipt_ref=dict(path=str(transport),sha256=sha(transport))))
        return status,receipt,dict(plan_sha256=self.workbench.plan_sha,recovery_admission_ref=dict(path=str(receipt),sha256=sha(receipt)))

    def test_controlled_download_same_id_and_original_identity(self):
        status,receipt,body=self.recovery_fixture();directory=Path(status['run_dir'])
        before=[sha(directory/name) for name in ('job.json','config.json')]
        with patch.object(self.workbench,'launch',return_value=status) as launch:
            result=self.workbench.recover_download(status['run_id'],body)
        self.assertEqual(result['run_id'],status['run_id'])
        self.assertEqual(before,[sha(directory/name) for name in ('job.json','config.json')])
        self.assertEqual(launch.call_args.kwargs['plan_path'],read(receipt)['original_plan_ref']['path'])
        self.assertTrue((directory/'recovery_history'/sha(receipt)/'original_status.json').exists())

    def test_wrong_scope_hash_and_alive_rejected(self):
        status,receipt,body=self.recovery_fixture();record=read(receipt);record['executor_port']=20424;atomic_json(receipt,record);body['recovery_admission_ref']['sha256']=sha(receipt)
        with self.assertRaisesRegex(ValueError,'scope'):self.workbench.recover_download(status['run_id'],body)
        record['executor_port']=20418;record['original_job_SHA']='0'*64;atomic_json(receipt,record);body['recovery_admission_ref']['sha256']=sha(receipt)
        with self.assertRaisesRegex(ValueError,'job SHA'):self.workbench.recover_download(status['run_id'],body)
        status.update(pid=os.getpid(),process_birth=birth(os.getpid()),boot_id=boot_id());atomic_json(self.workbench.status_path(status['run_id']),status)
        with self.assertRaisesRegex(ValueError,'remains alive'):self.workbench.recover_download(status['run_id'],body)
