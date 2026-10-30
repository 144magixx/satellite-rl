import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.t0_width.data import read_ref, validate_manifest, load_case, load_dataset, download_official_sources, prepare_binding, prepare_candidate


class WidthDataTests(unittest.TestCase):
    def test_pinned_archive_member_without_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'inputs.tar';payload=b'{"case":1}'
            with tarfile.open(path,'w') as archive:
                member=tarfile.TarInfo('canonical/case.json');member.size=len(payload)
                archive.addfile(member,io.BytesIO(payload))
            ref={'archive_path':str(path),'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'member':'canonical/case.json','sha256':hashlib.sha256(payload).hexdigest()}
            self.assertEqual(read_ref(ref),payload)
            self.assertEqual([p.name for p in Path(folder).iterdir()],['inputs.tar'])
            with self.assertRaises(ValueError):
                read_ref({**ref,'sha256':'0'*64})
            with self.assertRaises(ValueError):
                read_ref({**ref,'member':'../case.json'})

    def test_teacher_subset_is_not_a_dataset(self):
        rows=[{'role':'train','case_id':str(i),'UTC':str(i)} for i in range(519)]
        with self.assertRaises(ValueError):
            validate_manifest({'schema':'W26WenvPreparationManifestV1','method':'Wenv','rows':rows})

    def test_confirmation_is_rejected_before_any_read(self):
        with self.assertRaisesRegex(ValueError,'sealed'):
            load_case({'role':'test','case_ref':{'path':'NONEXISTENT'}})

    def test_self_hashed_manifest_does_not_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json'
            raw=json.dumps({'schema':'W26PublishedDatasetV1','root_approved':False}).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'root dataset'):
                load_dataset({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()},'train')

    def test_archive_duplicate_names_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'inputs.tar';payload=b'x'
            with tarfile.open(path,'w') as archive:
                for _ in range(2):
                    member=tarfile.TarInfo('case');member.size=1
                    archive.addfile(member,io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError,'duplicate'):
                read_ref({'archive_path':str(path),'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'member':'case','sha256':hashlib.sha256(payload).hexdigest()})

    def test_http200_is_never_appended_to_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);payload=b'complete';url='https://github.com/adsblol/globe_history_2024/releases/download/tag/file.tar'
            assets=[{'date':day,'source_url':url,'release_asset_id':i+1,'release_asset_bytes':len(payload)}
                    for i,day in enumerate(('2024-02-23','2024-02-24'))]
            admission={'schema':'W26OfficialDownloadAdmissionV1','root_approved':True,'remote_port':20418,
                       'canonical_directory':str(root),'download_wall_seconds':10,'assets':assets}
            raw=json.dumps(admission).encode();path=root/'admission.json';path.write_bytes(raw)
            ref={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            asset=assets[0]
            owner={'request_id':'owned','source_admission_SHA':ref['sha256'],'source_url':url,
                   'release_asset_id':asset['release_asset_id'],'release_asset_bytes':len(payload)}
            (root/'2024-02-23.owner.json').write_text(json.dumps(owner))
            partial=root/'2024-02-23.tar.part';partial.write_bytes(b'co')
            class Response:
                status=200
                headers={}
                def __enter__(self):return self
                def __exit__(self,*args):pass
            with patch('urllib.request.urlopen',return_value=Response()),patch('shutil.disk_usage') as free:
                free.return_value.free=10*1024**3
                with self.assertRaisesRegex(ValueError,'HTTP206'):
                    download_official_sources({'download_admission_ref':ref,'request_id':'owned'},root)
            self.assertEqual(partial.read_bytes(),b'co')

    def test_synthetic_wenv_is_candidate_not_published(self):
        from satellite_rl.shared_interface.stages import make_case
        from satellite_rl.shared_interface.content import wire_bytes, content_sha, read_wire
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            case=make_case({'case_id':'synthetic_train','actual_role':'train','source_role':'train',
                'original_mother_id':'synthetic','UTC':'2024-01-08T00:02:00Z','source_asset_SHA':'1'*64,'source':{}},
                ['a','b'],[[45.,10.],[45.01,10.01]])
            registry={'schema':'R7PublishedParentRegistryV1','case_sha_by_id':{'synthetic_train':case['content_sha256']},
                      'layout_shas':[],'allocation_shas':[]}
            cpath=root/'case.json';craw=wire_bytes(case);cpath.write_bytes(craw)
            apath=root/'authority.json';araw=json.dumps(registry).encode();apath.write_bytes(araw)
            row={'role':'train','case_id':'synthetic_train','case_content_SHA':case['content_sha256'],
                'case_ref':{'path':str(cpath),'sha256':hashlib.sha256(craw).hexdigest()},
                'case_authority_ref':{'path':str(apath),'sha256':hashlib.sha256(araw).hexdigest()},
                'case_authority_content_SHA':content_sha(registry)}
            binding=prepare_binding(row,'2'*64,'3'*64)
            request={'row':row,'source_SHA':'4'*64,'identity':binding['prepare_identity'],
                     'phase_manifest':binding['prepare_phase_manifest']}
            prepare_candidate(request,root/'output')
            layout=read_wire((root/'output'/'candidate.json').read_bytes())
            self.assertEqual(sum(layout['aircraft_count_by_beam']),2)
            self.assertEqual(sum(layout['request_bps_by_beam_int64']),43200000)
            self.assertEqual(layout['full_width_deg'].tolist(),[.3])
            self.assertEqual(json.loads((root/'output'/'stage_output.json').read_bytes())['status'],'candidate_only')
            published={**row,'layout_ref':{'path':str(root/'output'/'candidate.json'),
                'sha256':hashlib.sha256((root/'output'/'candidate.json').read_bytes()).hexdigest()},
                'layout_content_SHA':layout['content_sha256'],'publication_authority_ref':row['case_authority_ref'],
                'publication_authority_content_SHA':row['case_authority_content_SHA']}
            from satellite_rl.t0_width.data import load_published_layout
            with self.assertRaisesRegex(ValueError,'published parent'):
                load_published_layout(published,role='train')
            # Original prepare import must not call the prepare primitive again.
            original=json.loads((root/'output'/'prepared.json').read_bytes())
            outer={'stage':'preparation','source_sha256':'4'*64,'result':original}
            native_bytes=json.dumps(outer).encode();native_path=root/'native.json';native_path.write_bytes(native_bytes)
            guard_bytes=json.dumps({'status':'completed','exit_code':0}).encode();guard_path=root/'guard.json';guard_path.write_bytes(guard_bytes)
            native_ref={'path':str(native_path),'sha256':hashlib.sha256(native_bytes).hexdigest()}
            guard_ref={'path':str(guard_path),'sha256':hashlib.sha256(guard_bytes).hexdigest()}
            admission={'schema':'W26NativeWenvAdmissionV1','root_approved':True,'source_SHA':'4'*64,
                       'case_content_SHA':case['content_sha256'],'prepared_file_SHA':native_ref['sha256'],
                       'original_guard_file_SHA':guard_ref['sha256']}
            admission_bytes=json.dumps(admission).encode();admission_path=root/'native_admission.json';admission_path.write_bytes(admission_bytes)
            native_row={**row,'native_prepared_ref':native_ref,'native_guard_ref':guard_ref,
                        'native_wenv_admission_ref':{'path':str(admission_path),'sha256':hashlib.sha256(admission_bytes).hexdigest()}}
            from satellite_rl.t0_width.data import prepare_native_candidate
            with patch('satellite_rl.coverage.ai_search_r5.core.prepare_case',side_effect=AssertionError('new W forbidden')):
                prepare_native_candidate({**request,'row':native_row},root/'native_output')
            native_layout=read_wire((root/'native_output'/'candidate.json').read_bytes())
            self.assertEqual(native_layout['full_width_deg'].tolist(),layout['full_width_deg'].tolist())
            self.assertEqual(json.loads((root/'native_output'/'stage_output.json').read_bytes())['new_W_calls'],0)


if __name__=='__main__':
    unittest.main()
