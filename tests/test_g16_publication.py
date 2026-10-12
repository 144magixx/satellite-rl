"""Bounded synthetic QA, no production inputs or scientific computation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage1_geoload.publication import TABLES, build, load, number, paired, variable_requested_units


def fixture(root):
    root.mkdir()
    (root/'summary.json').write_text(json.dumps(dict(group='S1-G16', status='data_unknown',
        formal_ready=False, synthetic_fixture=True, issues=['fixture'], gate_passed=False)),encoding='utf-8')
    cases=[]
    for p in ('base','one_hotspot','two_hotspots','edge_heavy'):
        for method, value in [('JOINT',2),('rule',2),('09-LEX',3)]:
            cases.append(dict(task='common_test_20424',role='test',case_id='fixture_'+p,
                mother_id='fixture_2024-02-14',date='2024-02-14',demand_profile=p,
                method=method,outcome='completed',N=2,P=value,T=value,CV=.5))
        cases.append(dict(task='variable_20424',role='test',case_id='fixture_'+p,
            mother_id='fixture_2024-02-14',date='2024-02-14',demand_profile=p,
            method='VARIABLE',stage='constructor',target_N=1,outcome='not_found'))
        cases.append(dict(task='variable_20424',role='test',case_id='fixture_'+p,
            mother_id='fixture_2024-02-14',date='2024-02-14',demand_profile=p,
            method='VARIABLE',stage='quota',target_N=3,outcome='completed',N=3,total_demand=180,P=1.5,T=1.5,CV=.6))
    rows={'per_case':cases,'per_mother':[], 'timing':[], 'candidate':[], 'learningcurve':[]}
    for name in TABLES:
        values=rows[name];keys=list(dict.fromkeys(k for r in values for k in r))
        with (root/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)


class PublicationTests(unittest.TestCase):
    def test_refuse_incomplete_production_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'report'; fixture(source); out=Path(temp)/'figures'
            with self.assertRaisesRegex(ValueError,'not formal_ready'):build(source,out)
            self.assertFalse(out.exists())

    def test_qa_requires_explicit_marker_and_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'report'; fixture(source)
            with self.assertRaisesRegex(ValueError,'synthetic fixture'):load(source,Path(temp)/'figures',True)
            data=json.loads((source/'summary.json').read_text());data.pop('synthetic_fixture')
            (source/'summary.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'synthetic fixture'):load(source,Path(temp)/'synthetic_qa',True)

    def test_missing_is_not_zero_and_nonfinite_rejected(self):
        self.assertIsNone(number({'P':''},'P'))
        with self.assertRaises(ValueError):number({'P':'nan'},'P')

    def test_variable_original_unit_algebra_and_failures(self):
        row=dict(outcome='completed',N='4',total_demand='800',P='3',T='2')
        result=variable_requested_units(row)
        self.assertEqual(result['maxload'],600);self.assertEqual(result['top10_mean'],400)
        self.assertNotIn('maxload',row)
        for failure in ('not_found','not_run_constructor_not_found'):
            self.assertEqual(variable_requested_units({**row,'outcome':failure})['maxload'],'')
        for n in ('0','-1','2.5',''):
            with self.assertRaises(ValueError):variable_requested_units({**row,'N':n})
        self.assertEqual(variable_requested_units({**row,'total_demand':'0','P':'','T':''})['maxload'],'')

    def test_optional_real_node_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'report';fixture(source)
            with (source/'validation_cases.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['node','mother_id','demand_profile','P','T','CV'])
                writer.writeheader()
                for node in (2500,5000,7500,10000):
                    writer.writerow(dict(node=node,mother_id='fixture',demand_profile='base',P=2,T=2,CV=.5))
            _,tables=load(source,Path(temp)/'synthetic_qa',True)
            self.assertEqual([number(r,'node') for r in tables['validation_cases']],[2500,5000,7500,10000])

    def test_pairing_keeps_same_N_and_duplicate_rejected(self):
        rows=[dict(case_id='c',mother_id='m',demand_profile='base',method='JOINT',N='2',P='3'),
              dict(case_id='c',mother_id='m',demand_profile='base',method='MLM',N='2',P='2')]
        self.assertEqual(paired(rows,'P')[0]['difference'],-1)
        rows[1]['N']='3'; self.assertEqual(paired(rows,'P'),[])
        rows[0]['N']='';rows[1]['N']='';self.assertEqual(paired(rows,'P'),[])
        rows[1]['mother_id']='other'
        with self.assertRaisesRegex(ValueError,'identity'):paired(rows,'P')
        rows[1]['mother_id']='m'
        with self.assertRaises(ValueError):paired(rows+[rows[0]],'P')

    def test_synthetic_exports_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'report';fixture(source)
            out=Path(temp)/'synthetic_qa'; manifest=build(source,out,synthetic=True)
            self.assertTrue(manifest['synthetic']);self.assertEqual(len(manifest['artifacts']),15)
            with (out/'03_variable_N.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            failed=[r for r in rows if r['outcome']=='not_found']
            self.assertEqual(len(failed),4);self.assertTrue(all(r['maxload']=='' for r in failed))
            self.assertTrue(all(len(value)==64 for value in manifest['artifacts'].values()))
            self.assertTrue(manifest['unavailable'])


if __name__=='__main__':unittest.main()
