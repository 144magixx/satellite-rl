"""Independent V2 exact ordering. Pure component path performs no geometry."""
from fractions import Fraction

def components(counts,omega_components):
    if not counts or any(type(c) is not int or c<=0 for c in counts):raise ValueError('complete positive integer counts')
    if len(omega_components)!=len(counts):raise ValueError('one original float Omega component per beam')
    N=len(counts);t=(N+9)//10;T=sum(sorted(counts,reverse=True)[:t]);U=sum(c*c for c in counts)
    omega=sum((Fraction(*float(v).as_integer_ratio()) for v in omega_components),Fraction(0))
    evidence=dict(counts=sorted(counts),n=sum(counts),N=N,T=T,t=t,U=U,NU=N*U,
                  Omega_key=dict(numerator=omega.numerator,denominator=omega.denominator))
    return evidence

def key(evidence):
    required=('counts','n','N','T','t','U','NU','Omega_key')
    if any(k not in evidence for k in required):raise ValueError('missing complete exact evidence')
    counts=evidence['counts'];N=len(counts);t=(N+9)//10
    if not counts or any(type(c) is not int or c<=0 for c in counts):raise ValueError('nonempty integer exact counts')
    if any(evidence[k]!=v for k,v in dict(n=sum(counts),N=N,T=sum(sorted(counts,reverse=True)[:t]),t=t,U=sum(c*c for c in counts),NU=N*sum(c*c for c in counts)).items()):raise ValueError('exact evidence arithmetic mismatch')
    value=evidence['Omega_key']
    if type(value.get('numerator')) is not int or type(value.get('denominator')) is not int or value['denominator']<=0:raise ValueError('integer ratio exact Omega')
    return (max(counts),Fraction(evidence['T'],t),evidence['NU'],N,Fraction(value['numerator'],value['denominator']))

def layout_evidence(layout):
    # Admitted numerical reader only. The pure tests call components() with
    # hand-given Omega values, never this cosine/width path.
    import numpy as np
    N=len(layout['beam_ids']);counts=[0]*N
    for owner in layout['assignment']:
        if type(owner) is not int or not 0<=owner<N:raise ValueError('full exact ownership')
        counts[owner]+=1
    return counts_widths_evidence(counts,layout['widths_deg'])

def counts_widths_evidence(counts,widths):
    import numpy as np
    N=len(counts)
    widths=sorted((float(w) for w in widths),key=lambda w:(w,w.hex()))
    if len(widths)!=N or any(not np.isfinite(w) or not .3<=w<=1 for w in widths):raise ValueError('actual exact-order widths')
    values=[float(2*np.pi*(1-np.cos(np.deg2rad(np.float64(w)/2)))) for w in widths]
    return components(counts,values)

def better(a,b):
    if a['n']!=b['n']:raise ValueError('exact ordering requires fixed full n')
    return key(a)<key(b)
