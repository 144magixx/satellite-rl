"""Pure R12 v2 exact occupancy / original-float Omega ordering."""
from fractions import Fraction
import math
import numpy as np


def exact_evidence(counts, widths, ledger=None):
    counts=list(counts);widths=list(widths)
    if not counts or len(counts)!=len(widths) or any(type(c) is not int or c<=0 for c in counts):
        raise ValueError('complete positive integer counts and widths')
    if any(not math.isfinite(float(w)) or not .3<=w<=1. for w in widths):
        raise ValueError('finite legal full widths')
    if ledger:
        ledger.scan('exact_counts_widths',len(counts)*4)
        ordered=ledger.ordered(counts,lambda c:-c,'exact_tail_count_order')
        widths=ledger.ordered(widths,lambda w:(float(w),float(w).hex()),'exact_canonical_width_order')
    else:
        ordered=sorted(counts,reverse=True)
        widths=sorted(widths,key=lambda w:(float(w),float(w).hex()))
    N=len(counts);t=(N+9)//10;U=sum(c*c for c in counts)
    total=Fraction(0)
    for w in widths:
        # Same NumPy float64 per-beam expression as the frozen omega().
        term=float(2*np.pi*(1-np.cos(np.deg2rad(np.float64(w)/2))))
        if ledger:ledger.scan('exact_Omega_ratio_arithmetic',1)
        total+=Fraction(*term.as_integer_ratio())
    return dict(counts=list(reversed(ordered)),n=sum(counts),N=N,T=sum(ordered[:t]),t=t,U=U,NU=N*U,
                Omega_key=dict(numerator=total.numerator,denominator=total.denominator))


def exact_key(e,ledger=None):
    c=e['counts'];N=e['N'];t=(N+9)//10
    if ledger:
        ledger.scan('exact_key_validation_arithmetic',8*N+12)
        ledger.tick('sort_key_comparisons',N*N+5)
    if not c or any(type(x) is not int or x<=0 for x in c) or len(c)!=N:
        raise ValueError('exact count evidence')
    if (e['n'],e['T'],e['t'],e['U'],e['NU'])!=(sum(c),sum(sorted(c,reverse=True)[:t]),t,sum(x*x for x in c),N*sum(x*x for x in c)):
        raise ValueError('exact sufficient statistics')
    o=e['Omega_key']
    return max(c),Fraction(e['T'],t),e['NU'],N,Fraction(o['numerator'],o['denominator'])


def layout_evidence(layout,ledger=None):
    counts=[0]*len(layout['beam_ids'])
    for owner in layout['assignment']:
        if type(owner) is not int or not 0<=owner<len(counts):raise ValueError('complete owner domain')
        counts[owner]+=1
    if ledger:ledger.scan('exact_owner_counts',len(layout['assignment'])+len(counts))
    return exact_evidence(counts,layout['widths_deg'],ledger)
