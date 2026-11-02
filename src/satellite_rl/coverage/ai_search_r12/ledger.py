"""Owned R12 atomic ledger, reusing only R10 generic counter mechanics."""
from ..ai_search_r10.flow import Ledger as BaseLedger, ResourceUnknown
class Ledger(BaseLedger):
    CAPS = dict(raypairs=None, flow_queries=8, residual_edge_inspections=8388608,
        anchor_selection_passes=34, graph_arc_initialization_writes=65536,
        augmentation_capacity_writes=524288, full_vector_node_visits=1048576,
        sort_key_comparisons=2000000, scan_input_count=16777216)
    def __init__(self,n,stop=None):
        super().__init__(n,stop=stop)
        self.caps['raypairs']=16384+4*n
        self.outer=self
    def sort(self,label,count):
        # Conservative comparisons for the frozen audit's NumPy internal sort.
        # All other ordered() paths meter actual key comparisons directly.
        if label=='audit_top10_load_order':self.tick('sort_key_comparisons',count*count)
        super().sort(label,count)
    def record(self):
        return dict(super().record(),scope='one R12 local kernel; no prior budget transfer')
class MotherLedger(BaseLedger):
    CAPS = dict(raypairs=None, flow_queries=48, kernels=6,
        residual_edge_inspections=6*8388608, anchor_selection_passes=6*34,
        graph_arc_initialization_writes=6*65536, augmentation_capacity_writes=6*524288,
        full_vector_node_visits=6*1048576, sort_key_comparisons=6*2000000,
        scan_input_count=6*16777216)
    def __init__(self,n,stop=None):
        super().__init__(n,stop=stop)
        self.caps['raypairs']=98504+28*n
    def sort(self,label,count):
        if label=='audit_top10_load_order':self.tick('sort_key_comparisons',count*count)
        super().sort(label,count)
    def absorb(self,record):
        for key in self.caps:
            self.tick(key,record.get(key,0))
