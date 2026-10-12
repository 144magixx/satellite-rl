"""Small synthetic package and fake shared transport only; never contacts hosts."""
import json
import base64
import contextlib
import hashlib
import io
import re
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.stage1_geoload import final_delivery as delivery


def fixture(root,gate=False):
    root.mkdir()
    rows=[]
    def add(role,value,suffix='.json'):
        path=root/(role+suffix)
        path.write_bytes(delivery.encoded(value) if suffix=='.json' else value)
        row=dict(role=role,path=str(path),sha256=delivery.sha(path),member='evidence/'+path.name)
        rows.append(row);return row
    science=add('science_plan',dict(group='S1-G16',scientific_commit=delivery.SCIENCE))
    run=add('run_plan',dict(group='S1-G16',scientific_commit=delivery.SCIENCE))
    receipts={'common_test_20424':dict(run_id='fake',port=20424,verified=True)}
    tasks={'common_test_20424':dict(run_id='fake',port=20424)}
    if gate:
        receipts['train_mlm']=dict(run_id='fake_model',port=20473,verified=True)
        tasks['train_mlm']=dict(run_id='fake_model',port=20473)
    snapshot=add('snapshot',dict(all_computation_completed=True,compute_and_archives_complete=True,archives=receipts,tasks=tasks))
    summary=add('summary',dict(group='S1-G16',formal_ready=True,status='formal_ready',issues=[],scientific_commit=delivery.SCIENCE,
        gate_passed=gate,plan_sha256=run['sha256'],state_sha256=snapshot['sha256']))
    add('closure',dict(group='S1-G16',status='accepted',issues=[],nodes=[dict(id='canonical_fake',path='outside/raw.zip',sha256='a'*64)]))
    add('report11',('SYNTHETIC FORMAL REPORT QA ONLY\n'+'test evidence '*20).encode(),'.md')
    add('reproduction',b'python -m scripts.stage1_geoload.report --help\n','.txt')
    add('task_receipts',receipts)
    nodes=[dict(id='canonical_fake:member.py',sha256='b'*64,archive_path='outside/source.zip',member='member.py')]
    add('source_index',dict(sources=[dict(role=role,commit=commit,files=[dict(path='fake.py',sha256='b'*64,canonical_ref='canonical_fake:member.py')]) for role,commit in [('scientific',delivery.SCIENCE),('controller','b520'+'1'*36),('postprocessing','c'*40)]]))
    artifacts={}
    for prefix in ('01_same_N','02_input_profiles','03_variable_N','04_learning','05_costs'):
        for suffix in ('png','pdf','csv'):
            name=prefix+'.'+suffix;(root/name).write_bytes(b'SYNTHETIC QA '+name.encode());artifacts[name]=delivery.sha(root/name)
    add('publication_manifest',dict(synthetic=False,formal_ready=True,source_summary_sha256=summary['sha256'],artifacts=artifacts,sources={}))
    mappings={}
    if gate:
        model=add('model_final',b'FAKE SYNTHETIC MODEL NEVER LOAD','.pt')
        checkpoint='d'*64
        add('gpu_receipt',dict(verified=True,final=model['sha256'],checkpoint=checkpoint))
        add('export_receipt',dict(verified=True,sha=model['sha256']))
        add('model_provenance',dict(scientific_commit=delivery.SCIENCE,final_sha256=model['sha256'],checkpoint_sha256=checkpoint,run_id='fake_model',port=20473,canonical_final_member='train_mlm:model/final.pt'))
        nodes.append(dict(id='train_mlm:model/final.pt',sha256=model['sha256'],owner='train_mlm',archive_path='outside/train.zip',member='model/final.pt'))
        mappings=dict(verified='verified',final_sha256='final',checkpoint_sha256='checkpoint',export_verified='verified',export_sha256='sha')
    add('source_receipts',dict(status='accepted',nodes=nodes))
    spec=root/'spec.json'
    spec.write_bytes(delivery.encoded(dict(schema=delivery.SCHEMA,group='S1-G16',package_id='synthetic_fixture',report11_status='final',files=rows,model_receipt_fields=mappings)))
    return spec


class FakeTransport:
    BASE='/home/u2024110224/first_stage_satellite_rl/s1_g16'
    def __init__(self):self.uploads=0;self.data=None;self.ports=[]
    def upload(self,path,remote,port):self.uploads+=1;self.data=Path(path).read_bytes()
    def remote_python(self,code,port,timeout=60):
        self.ports.append(port)
        if 'expected=json.loads' not in code:
            value=dict(exists=self.data is not None,sha256=__import__('hashlib').sha256(self.data).hexdigest() if self.data else None)
        else:
            pin=re.search(r"'manifest_sha256':\s*'([0-9a-f]{64})'",code).group(1)
            value=dict(verified=True,zip_sha256=__import__('hashlib').sha256(self.data).hexdigest(),manifest_sha256=pin,members=26)
        return dict(exit_code=0,stdout=json.dumps(value),stderr='')


class DeliveryTests(unittest.TestCase):
    def test_readonly_plan_and_complete_small_package(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'input';spec=fixture(source)
            before=set(source.iterdir());value=delivery.plan(spec)
            self.assertEqual(value['status'],'ready');self.assertEqual(set(source.iterdir()),before)
            out=Path(temp)/'out';receipt=delivery.package(spec,out)
            self.assertFalse(receipt['manifest']['standalone_full_data'])
            with zipfile.ZipFile(out/'delivery.zip') as z:self.assertFalse(any(n.endswith('.tar') or n.endswith('raw.zip') for n in z.namelist()))
            with self.assertRaises(FileExistsError):delivery.package(spec,out)

    def test_failure_is_readonly_and_incomplete_gate_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp)/'input');data=delivery.read(spec)
            summary=next(r for r in data['files'] if r['role']=='summary')
            content=delivery.read(summary['path']);content['formal_ready']=False
            Path(summary['path']).write_bytes(delivery.encoded(content));summary['sha256']=delivery.sha(summary['path']);spec.write_bytes(delivery.encoded(data))
            self.assertEqual(delivery.plan(spec)['status'],'not_ready')
            with self.assertRaises(ValueError):delivery.package(spec,Path(temp)/'out')
            self.assertFalse((Path(temp)/'out').exists())

    def test_actual_model_mapping_and_provenance_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp)/'input',gate=True)
            self.assertEqual(delivery.plan(spec)['status'],'ready')
            data=delivery.read(spec);data['model_receipt_fields']['checkpoint_sha256']='missing'
            spec.write_bytes(delivery.encoded(data));self.assertEqual(delivery.plan(spec)['status'],'not_ready')

    def test_exact_source_node_and_figure_sha_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp)/'input');data=delivery.read(spec)
            index=next(r for r in data['files'] if r['role']=='source_index')
            value=delivery.read(index['path']);value['sources'][0]['files'][0]['canonical_ref']='not_real'
            Path(index['path']).write_bytes(delivery.encoded(value));index['sha256']=delivery.sha(index['path'])
            spec.write_bytes(delivery.encoded(data));self.assertEqual(delivery.plan(spec)['status'],'not_ready')
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp)/'input');(spec.parent/'01_same_N.png').write_bytes(b'tampered')
            self.assertEqual(delivery.plan(spec)['status'],'not_ready')

    def test_resume_identical_package_one_upload_dual_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp)/'input');out=Path(temp)/'out';delivery.package(spec,out)
            transport=FakeTransport();receipt=delivery.resume_upload(out/'receipt.json',transport=transport)
            self.assertEqual(receipt['status'],'dual_endpoint_verified');self.assertEqual(set(receipt['remote_verified']),{'20424','20473'})
            self.assertEqual(transport.uploads,1)
            original=delivery.sha(out/'delivery.zip');delivery.resume_upload(out/'receipt.json',transport=transport)
            self.assertEqual(transport.uploads,1);self.assertEqual(delivery.sha(out/'delivery.zip'),original)
            (out/'delivery.zip').write_bytes(b'changed')
            with self.assertRaises(ValueError):delivery.resume_upload(out/'receipt.json',transport=transport)

    def test_path_and_asset_boundaries(self):
        with self.assertRaises(ValueError):delivery.safe('../raw.zip')
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp)/'input');data=delivery.read(spec)
            row=data['files'][0];path=Path(temp)/'input'/'fake.tar';path.write_bytes(b'raw')
            row['path']=str(path);row['sha256']=delivery.sha(path);spec.write_bytes(delivery.encoded(data))
            self.assertEqual(delivery.plan(spec)['status'],'not_ready')

    def test_remote_command_is_bounded_independent_of_large_dependency_index(self):
        small={'members':{},'canonical_dependencies':[]}
        large={'members':{},'canonical_dependencies':[{'id':'source_'+str(i),'path':'/canonical/'+'x'*250,'sha256':'a'*64} for i in range(10000)]}
        def code(value):return delivery.remote_verify_code('/canonical/delivery.zip','b'*64,123,hashlib.sha256(delivery.encoded(value)).hexdigest(),0)
        a,b=code(small),code(large)
        self.assertEqual(len(a),len(b));self.assertLess(len(base64.b64encode(b.encode())),16*1024)
        self.assertNotIn('canonical_dependencies',b);compile(b,'remote_verify','exec')

    def test_real_remote_code_rejects_volume_manifest_member_and_inventory_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'delivery.zip';payload=b'correct member'
            expected={'schema':'g16_small_delivery_v1','members':{'safe.txt':{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}},'canonical_dependencies':[{'id':'external'}]}
            manifest=delivery.encoded(expected);pin=hashlib.sha256(manifest).hexdigest()
            def write(member=payload,manifest_bytes=manifest,extra=False):
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('safe.txt',member);z.writestr('delivery_manifest.json',manifest_bytes)
                    if extra:z.writestr('unlisted.txt',b'unknown')
            def run(zip_sha=None,manifest_sha=pin):
                code=delivery.remote_verify_code(str(path),zip_sha or delivery.sha(path),path.stat().st_size,manifest_sha,1)
                out=io.StringIO()
                with contextlib.redirect_stdout(out):exec(compile(code,'remote_verify','exec'),{})
                return json.loads(out.getvalue())
            write();self.assertTrue(run()['verified'])
            with self.assertRaisesRegex(ValueError,'ZIP differs'):run(zip_sha='0'*64)
            write(manifest_bytes=manifest+b' ')
            with self.assertRaisesRegex(ValueError,'manifest differs'):run()
            write(member=b'tampered bytes')
            with self.assertRaisesRegex(ValueError,'member SHA'):run()
            write(extra=True)
            with self.assertRaisesRegex(ValueError,'member set'):run()


if __name__=='__main__':unittest.main()
