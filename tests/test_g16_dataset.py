import ast
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
from satellite_rl.coverage.scenes import CoverageScene, CoverageDataset
from satellite_rl.coverage.variable_dataset import save_variable_dataset
from satellite_rl.coverage.geoload.dataset import prepare_dataset, demand_profiles, digest, _requests

class G16DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def source(self,name,stamp):
        path=self.root/('source_'+name)/'inputs.npz'
        scene=CoverageScene(name,np.array([[45.,5.],[45.,5.1],[45.,30.]]),np.array([10.,0.,5.]),
            np.array([[0.,0.]]),['aa','bb','cc'],{'selected_time_utc':stamp,'capacity_600Mbps':True})
        save_variable_dataset(CoverageDataset([scene],{}),path)
        return {'path':str(path),'sha256':digest(path)}
    def test_import_no_mutation_zero_preservation_profiles(self):
        source=self.source('one','2024-02-14T00:02:00Z');before=Path(source['path']).read_bytes()
        record=prepare_dataset({'kind':'import_existing','role':'test','count':1,'sources':[source]},self.root/'out',self.root)
        self.assertEqual(record['count'],4);self.assertEqual(Path(source['path']).read_bytes(),before)
        with np.load(record['path'],allow_pickle=False) as values:
            self.assertEqual(values['offsets'].tolist(),[0,3,6,9,12])
            self.assertEqual(values['aircraft_ids'].tolist(),['aa','bb','cc']*4)
            self.assertNotIn('centers',values.files)
            for i in range(4):self.assertAlmostEqual(values['demands'][i*3:(i+1)*3].sum(),15.)
            self.assertTrue((values['demands'][[1,4,7,10]]==0).all())
            self.assertNotIn('capacity_600Mbps',str(values['metadata_json']))
    def test_sha_rejection(self):
        source=self.source('one','2024-02-14T00:02:00Z');source['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA'):
            prepare_dataset({'kind':'import_existing','role':'test','count':1,'sources':[source]},self.root/'out',self.root)
    def test_import_stable_id_sort_records_original_permutation(self):
        path=self.root/'unsorted'/'inputs.npz'
        positions=np.array([[45.,30.],[45.,5.],[45.,5.1]])
        demands=np.array([5.,10.,0.]);ids=['cc','aa','bb']
        save_variable_dataset(CoverageDataset([CoverageScene('raw',positions,demands,np.empty((0,2)),ids,
            {'selected_time_utc':'2024-02-14T00:02:00Z'})],{}),path)
        before=path.read_bytes()
        record=prepare_dataset({'kind':'import_existing','role':'test','count':1,
            'sources':[{'path':str(path),'sha256':digest(path)}]},self.root/'sorted',self.root)
        with np.load(record['path'],allow_pickle=False) as values:
            np.testing.assert_array_equal(values['positions'][:3],positions[[1,2,0]])
            np.testing.assert_array_equal(values['demands'][:3],demands[[1,2,0]])
            self.assertEqual(values['aircraft_ids'][:3].tolist(),['aa','bb','cc'])
            meta=json.loads(str(values['metadata_json']))['mothers'][0]
            self.assertEqual(meta['canonical_to_source_index'],[1,2,0])
            self.assertEqual(meta['source_aircraft_ids'],ids)
        self.assertEqual(path.read_bytes(),before)
    def test_merge_keeps_values_and_source(self):
        refs=[]
        for i in range(2):
            source=self.source(str(i),f'2024-02-{14+i}T00:02:00Z')
            refs.append(prepare_dataset({'kind':'import_existing','role':'test','count':1,'sources':[source]},self.root/str(i),self.root))
        merged=prepare_dataset({'kind':'merge_existing','role':'test','count':2,'sources':refs},self.root/'merged',self.root)
        with np.load(merged['path'],allow_pickle=False) as values:
            self.assertEqual(len(set(values['mother_ids'])),2)
            expected=[]
            for ref in refs:
                with np.load(ref['path'],allow_pickle=False) as original:expected.append(original['demands'])
            np.testing.assert_array_equal(values['demands'],np.concatenate(expected))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            prepare_dataset({'kind':'merge_existing','role':'test','count':2,'sources':[refs[0],refs[0]]},self.root/'dup',self.root)
    def test_requests_timestamp_stability(self):
        rows=[('bb','2024-02-14T00:02:00Z',45.,5.1,10000.),('aa','2024-02-14T00:02:00Z',45.,5.,10000.)]
        a=_requests(rows,'2024-02-14T00:02:00.000Z');b=_requests(rows[::-1],'2024-02-14T00:02:00Z')
        np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[2],['aa','bb'])
    def test_requests_match_existing_real_adapter_across_timestamp_formats_and_roles(self):
        # This invokes the existing adapter, rather than reproducing its RNG
        # formula in the assertion. Its geometric result is discarded.
        from satellite_rl.workbench.variable_preparation import _scene_from_rows
        stamps=('2024-01-12T12:02:00Z','2024-02-13T00:02:00.000Z','2024-02-13T00:07:05.250Z')
        for index,stamp in enumerate(stamps):
            rows=[(identifier,stamp,45.,5.+i*.01,10000.)
                  for i,identifier in enumerate(('ee','aa','dd','bb','cc'))]
            expected=_requests(rows,stamp)
            for role in ('train','development','test'):
                work=self.root/f'paired_{index}_{role}';work.mkdir()
                actual=_scene_from_rows(work,0,stamp,rows,role,'0'*64)
                self.assertEqual(actual.aircraft_ids,expected[2])
                np.testing.assert_array_equal(actual.positions,expected[0])
                np.testing.assert_array_equal(actual.demands,expected[1])
    def test_demand_seed_override_is_rejected(self):
        source=self.source('seed','2024-02-14T00:02:00Z')
        with self.assertRaisesRegex(ValueError,'51001'):
            prepare_dataset({'kind':'import_existing','role':'test','count':1,'seed':51002,
                             'sources':[source]},self.root/'seed_out',self.root)
    def test_empty_and_allzero_profiles(self):
        for values in demand_profiles([[45,5],[45,30]],[0,0],['a','b']).values():
            np.testing.assert_array_equal(values,[0,0])
        for values in demand_profiles(np.empty((0,2)),[],[]).values():self.assertEqual(len(values),0)
    def test_pinned_official_parser_tar_timestamp_expiry_input_selection(self):
        parser=Path(os.environ.get('G16_TEST_PARSER','D:/paper/outputs/experiments/first_stage_nondrl12_20261012/source/preparation_reference.py'))
        if not parser.is_file():self.skipTest('set G16_TEST_PARSER to the SHA-pinned official reference parser')
        raw=self.root/'raw';raw.mkdir();day='2024-02-13';name='v2024.02.13-planes-readsb-prod-0.tar';archive=raw/name
        base=datetime(2024,2,13,0,2,tzinfo=timezone.utc).timestamp()
        trace={'icao':'abc123','timestamp':base,'trace':[[0,45,5,30000,0,0,0],[300,45,5.1,30000,0,0,0],[600,45,5.2,'ground',0,0,0]]}
        body=json.dumps(trace).encode()
        with tarfile.open(archive,'w') as stream:
            member=tarfile.TarInfo('./traces/ab/trace_full_abc123.json');member.size=len(body);stream.addfile(member,io.BytesIO(body))
        metadata=self.root/'release_api.json'
        asset={'id':99,'name':name,'size':archive.stat().st_size,'browser_download_url':'https://example.invalid/official.tar'}
        metadata.write_text(json.dumps({'assets':[asset]}))
        spec={'kind':'official_day','role':'validation','date':day,'count':2,'source_instance':'prod-0',
            'raw_shared_root':str(raw),'parser':{'path':str(parser),'sha256':digest(parser)},
            'metadata_pins':{'release_api':{'path':str(metadata),'sha256':digest(metadata)}},
            'archive':{'name':name,'url':asset['browser_download_url'],'asset_id':99,
                       'expected_bytes':asset['size'],'sha256':digest(archive)}}
        result=prepare_dataset(spec,self.root/'official',Path(__file__).resolve().parents[1])
        self.assertEqual(result['mother_count'],2)
        audit=json.loads((self.root/'official/all_288_candidate_audit.json').read_text())
        self.assertEqual(len(audit),288);self.assertTrue(audit[0]['eligible']);self.assertTrue(audit[1]['eligible'])
        self.assertEqual(audit[2]['counts']['ground'],1);self.assertEqual(audit[3]['counts']['expired'],1)
        self.assertEqual(audit[2]['rejection_reason'],'empty_after_trace_filters')
        self.assertEqual(len(list(raw.glob('*.tar'))),1)
        spec['count']=3
        unavailable=prepare_dataset(spec,self.root/'insufficient',Path(__file__).resolve().parents[1])
        self.assertEqual(unavailable['status'],'data_unavailable');self.assertEqual(unavailable['available_mothers'],2)
    def test_no_downstream_imports(self):
        from satellite_rl.coverage.geoload import dataset
        for node in ast.walk(ast.parse(Path(dataset.__file__).read_text())):
            if isinstance(node,ast.ImportFrom):
                self.assertFalse(any(key in (node.module or '') for key in ('physical','nondrl','aviation','variable_preparation','tdma','allocation')))

if __name__=='__main__':unittest.main()
