"""Hand supplied abstract graphs only: no scientific scene or geometry call."""
import itertools
import unittest
from satellite_rl.coverage.ai_search_r10.flow import Ledger, ResourceUnknown, solve


class ExactFlowTests(unittest.TestCase):
    def check_graph(self, adjacency, costs, q, aircraft=None, beams=None):
        n, b = len(adjacency), len(adjacency[0])
        aircraft = aircraft or ['a'+str(i) for i in range(n)]
        beams = beams or ['b'+str(j) for j in range(b)]
        legal = []
        for assignment in itertools.product(range(b), repeat=n):
            counts = [assignment.count(j) for j in range(b)]
            if all(1 <= c <= q for c in counts) and all(adjacency[i][j] for i,j in enumerate(assignment)):
                legal.append((sum(costs[i][j] for i,j in enumerate(assignment)), assignment))
        ledger = Ledger(n)
        result = solve(aircraft, beams, adjacency, costs, q, ledger)
        self.assertEqual(result['feasible'], bool(legal))
        if legal:
            self.assertEqual(int(result['total_cost']), min(x[0] for x in legal))
            self.assertIn(tuple(result['assignment']), [x[1] for x in legal])
            self.assertEqual(ledger.data['augmentations'], n)
        self.assertEqual(ledger.data['heap_pushes_plus_pops'], ledger.data['heap_pushes']+ledger.data['heap_pops'])
        self.assertEqual(ledger.data['heap_pushes'], ledger.data['heap_pops'])
        self.assertEqual(ledger.data['flow_queries'], 1)
        return result, ledger

    def test_lower_one_prevents_empty_cheaper_beam(self):
        result, _ = self.check_graph([[True,True]]*3, [[0,9]]*3, 3)
        self.assertEqual(sorted(result['counts']), [1,2])

    def test_rerouting_reverse_arcs(self):
        self.check_graph([[True,True],[True,False]], [[0,2],[1,0]], 1)

    def test_no_pinned_anchor_membership(self):
        result, _ = self.check_graph([[False,True],[True,False]], [[0,0],[0,0]], 1)
        self.assertEqual(result['assignment'], [1,0])

    def test_complete_ties_deterministic_unsorted_IDs(self):
        args = ([[True,True]]*4, [[0,0]]*4, 2)
        a, la = self.check_graph(*args, aircraft=['z','a','m','b'], beams=['z','a'])
        b, lb = self.check_graph(*args, aircraft=['z','a','m','b'], beams=['z','a'])
        self.assertEqual(a,b)
        self.assertEqual(la.record(),lb.record())

    def test_infeasible_finite_graph(self):
        result, _ = self.check_graph([[True,False]]*3, [[0,0]]*3, 2)
        self.assertEqual(result['status'],'graph_infeasible')

    def test_integer_beyond_float_and_int64(self):
        huge = 2**80
        self.check_graph([[True,True]]*3, [[huge+1,huge],[huge,huge+3],[huge+2,huge]], 2)

    def test_full_reachable_scan_and_stale_entries(self):
        _, ledger = self.check_graph([[True,True,True]]*3, [[0,9,8],[1,0,2],[8,1,0]], 1)
        self.assertGreater(ledger.data['heap_key_comparisons'], 0)
        self.assertGreater(ledger.data['residual_edge_inspections'], 0)

    def test_unknown_is_not_infeasible(self):
        ledger = Ledger(2, caps={'residual_edge_inspections':0})
        with self.assertRaises(ResourceUnknown) as caught:
            solve(['a','b'],['x','y'],[[True]*2]*2,[[0]*2]*2,1,ledger)
        self.assertEqual(caught.exception.partial_ledger['flow_results'], [])

    def test_invalid_inputs_do_not_enter_solver(self):
        ledger=Ledger(3)
        with self.assertRaises(ValueError):
            solve(['a','b','c'],['x'],[[True]]*3,[[0]]*3,2,ledger)
        self.assertEqual(ledger.data['flow_queries'],0)


if __name__ == '__main__':
    unittest.main()
