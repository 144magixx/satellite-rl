"""Pure stored-array/mock-angle checks. No real projection/owner GEO is called."""
import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
from . import audit as a

def case():
    return dict(identity=dict(case_id='pure_case',source_role='preserved_original'),source_ref=dict(sha256='1'*64,bytes=9),aircraft_ids=['a','b'],positions=[[45.,13.],[45.,13.]],demands=[a.REQUEST,a.REQUEST],physical_parameters=dict(version='pure_given_constants',earth_radius_km=6371,orbit_radius_km=42164,satellite_longitude_deg=13,min_elevation_deg=10,latitude_bounds_deg=[40,60],longitude_bounds_deg=[0,30]))
def layout(width=.3):return dict(beam_ids=['beam'],anchor_aircraft_ids=['a'],centers=[[45.,13.]],widths_deg=[width],assignment=[0,0])
def stored_cache(c):
    # Synthetic predefined unit vectors; the projection primitive is never called.
    l=a.AuditLedger()
    with patch.object(a,'_project_rays_visibility',return_value=(np.array([[1.,0.,0.],[1.,0.,0.]],dtype='<f8'),np.array([True,True]))):record=a.build_ray_cache(c,l)
    l.complete()
    return a.bind_cache(c,record,expected_cache_sha=record['cache_SHA']),record
def checked(c,L,cache,ledger,**kw):
    with patch.object(a,'_owner_chord',return_value=np.zeros(2)),patch.object(a,'_minimal_widths',return_value=np.array([.3])):
        return a.audit_layout(c,L,cache,ledger,scope=kw.pop('scope','offline'),envelope=kw.pop('envelope','minimal'),Ncap=2,Omega_cap=.1,**kw)

class PureAuditTests(unittest.TestCase):
    def test_mock_projection_once_predebit_and_durable_completion(self):
        c=case();events=[];l=a.AuditLedger(persist=lambda s:events.append(s))
        def supplied(positions,physics):
            self.assertEqual(l.state['rays'],2);self.assertIsNotNone(l.state['inflight']);return np.array([[1.,0.,0.],[1.,0.,0.]],dtype='<f8'),np.array([True,True])
        with patch.object(a,'_project_rays_visibility',side_effect=supplied) as primitive:
            record=a.build_ray_cache(c,l);self.assertEqual(primitive.call_count,1)
            self.assertIsNotNone(l.state['inflight']);l.complete()
            with self.assertRaises(a.AuditError):a.build_ray_cache(c,l)
            self.assertEqual(primitive.call_count,1)
        self.assertEqual(events[0]['rays'],2);self.assertIsNone(events[-1]['inflight']);self.assertEqual(record['dtype'],'<f8')
    def test_cache_full_binding_endian_sha_and_no_teacher(self):
        c=case();bound,record=stored_cache(c);self.assertFalse(bound.rays.flags.writeable)
        for field,value in (('aircraft_ids',['b','a']),('positions',[[45.,13.],[45.1,13.]]),('demands',[a.REQUEST,1])):
            bad=copy.deepcopy(c);bad[field]=value
            with self.assertRaises(a.AuditError):a.bind_cache(bad,record,expected_cache_sha=record['cache_SHA'])
        bad=copy.deepcopy(record);bad['dtype']='>f8';bad['cache_SHA']=a.sha({k:v for k,v in bad.items() if k!='cache_SHA'})
        with self.assertRaises(a.AuditError):a.bind_cache(c,bad,expected_cache_sha=bad['cache_SHA'])
        with self.assertRaises(a.AuditError):a.bind_cache(c,record,expected_cache_sha='0'*64)
        self.assertNotIn('teacher',record)
    def test_separate_online_offline_geometry_debits_and_metrics(self):
        c=case();cache,_=stored_cache(c);l=a.AuditLedger();r=checked(c,layout(),cache,l)
        self.assertEqual(l.state['offline_layouts'],1);self.assertEqual(l.state['offline_pairs'],2);self.assertEqual(l.state['online_pairs'],0)
        self.assertIsNotNone(l.state['inflight']);self.assertEqual(r['metrics']['maxload'],2*a.REQUEST);self.assertEqual(r['metrics']['total_request'],2*a.REQUEST);self.assertEqual(r['metrics']['CV'],0)
        l.complete();checked(c,layout(),cache,l,scope='online');self.assertEqual(l.state['online_layouts'],1);self.assertEqual(l.state['online_pairs'],2);self.assertEqual(l.state['rays'],0)
    def test_raw_W_coverage_witness_does_not_require_minimum_envelope(self):
        c=case();cache,_=stored_cache(c);l=a.AuditLedger();r=checked(c,layout(.6),cache,l,envelope='raw_W');self.assertEqual(r['envelope'],'raw_W');l.complete()
        with self.assertRaises(a.AuditError):checked(c,layout(.6),cache,l,envelope='minimal')
        self.assertIsNotNone(l.state['inflight'])
    def test_non_anchor_or_empty_or_incomplete_owner_fails_before_geometry(self):
        c=case();cache,_=stored_cache(c)
        for mutate in (lambda L:L.update(anchor_aircraft_ids=['not_original']),lambda L:L.update(assignment=[0]),lambda L:L.update(assignment=[0.,0.]),lambda L:L.update(centers=[[44.,13.]]),lambda L:L.update(widths_deg=[1.01])):
            L=layout();mutate(L);l=a.AuditLedger()
            with patch.object(a,'_owner_chord') as primitive:
                with self.assertRaises(a.AuditError):a.audit_layout(c,L,cache,l,scope='offline',envelope='minimal',Ncap=2,Omega_cap=.1)
                primitive.assert_not_called();self.assertEqual(l.state['offline_pairs'],0)
    def test_given_invalid_angle_is_charged_then_unresolved_no_retry(self):
        c=case();cache,_=stored_cache(c);l=a.AuditLedger()
        with patch.object(a,'_owner_chord',return_value=np.array([0.,1.])):
            with self.assertRaises(a.AuditError):a.audit_layout(c,layout(),cache,l,scope='offline',envelope='minimal',Ncap=2,Omega_cap=.1)
        self.assertEqual(l.state['offline_pairs'],2)
        with self.assertRaises(a.AuditError):a.AuditLedger(l.state)
    def test_caps_reserve_before_mock_and_stop_has_zero_entry(self):
        c=case();cache,_=stored_cache(c);l=a.AuditLedger();l.state['offline_layouts']=a.OFFLINE_LAYOUTS
        with patch.object(a,'_owner_chord') as primitive:
            with self.assertRaises(a.AuditError):a.audit_layout(c,layout(),cache,l,scope='offline',envelope='minimal',Ncap=2,Omega_cap=.1)
            primitive.assert_not_called()
        l=a.AuditLedger()
        with self.assertRaises(a.AuditError):checked(c,layout(),cache,l,stop=lambda:True)
        self.assertEqual(l.state['offline_pairs'],0)
    def test_supplied_math_Omega_cap_and_saved_metric_failure(self):
        c=case();cache,_=stored_cache(c);l=a.AuditLedger();r=checked(c,layout(),cache,l);l.complete()
        expected=2*math.pi*(1-math.cos(math.radians(.3)/2));self.assertEqual(r['metrics']['actual_Omega'],expected)
        wrong=dict(r['metrics'],total_request=0)
        with self.assertRaises(a.AuditError):checked(c,layout(),cache,l,saved_metrics=wrong)
    def test_static_eight_fixture_definition_only(self):
        with patch.object(a,'_project_rays_visibility') as project,patch.object(a,'_owner_chord') as geometry:
            fs=a.engineering_fixture_definitions();project.assert_not_called();geometry.assert_not_called()
        self.assertEqual([len(f['aircraft_ids']) for f in fs],[8,8,16,16,32,32,64,64]);self.assertEqual(sum(len(f['aircraft_ids']) for f in fs),240)
        self.assertEqual(fs[0]['positions'][0],[45.,13.]);self.assertEqual(fs[7]['positions'][8],[45.36,13.]);self.assertTrue(all(f['Ncap']==len(f['layout']['beam_ids'])+2 for f in fs))

if __name__=='__main__':unittest.main()
