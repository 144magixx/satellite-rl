"""Original independent ray predicates; no new geometry or decoder algorithm."""
import numpy as np
from ..ai_search.core import Layout
from ..ai_search_r2.validation import validate_owned

def omega(widths):return float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(np.asarray(widths,dtype=np.float64)/2)))))
class FrozenGeometry:
    def __init__(self,scene):self.scene=scene
    def audit(self,centers,widths,owners,beam_ids,ledger):
        return validate_owned(self.scene,Layout(centers,widths,owners,tuple(beam_ids)),budget=ledger)
    def distances(self,centers,ledger):
        from ..ai_search_r5.core import _angles_and_chord
        ledger.charge('distance_table_elements',len(self.scene.demands)*len(centers))
        ledger.geometry(len(self.scene.demands)*len(centers),'initial_fixed_anchor_distances')
        rays=ledger.center_rays(centers,self.scene.profile)
        # Same chord->angle arithmetic as the unchanged envelope helper.
        angles,_=_angles_and_chord(self.scene.rays[:,None,:],rays[None,:,:])
        return np.asarray(angles,dtype=np.float64)
    def widths_for_move(self,owners,source,destination,widths,distances,ledger):
        result=np.array(widths,dtype=np.float64,copy=True)
        for beam in (source,destination):
            ledger.charge('envelope_owner_scan_items',len(owners))
            members=np.flatnonzero(owners==beam);ledger.charge('cached_member_distance_scan',len(members))
            if not len(members):return None
            required=2*float(distances[members,beam].max())
            if required>1.:return None
            result[beam]=max(.3,min(1.,float(np.nextafter(required,np.inf))))
        return result
