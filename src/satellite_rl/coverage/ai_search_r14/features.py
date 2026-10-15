"""Twelve initial-B scalar features; no geometry or future-trajectory access."""
import math

FEATURE_NAMES=('aircraft_count','N','N_over_Ncap','Omega_over_cap','max_beam_count',
 'top10_mean_count','CV','peak_multiplicity_over_N','median_beam_count','mean_width','p90_width','max_width')

def nearest_rank(values,q):
    if not values or not 0<q<=1:raise ValueError('nonempty nearest-rank domain')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('finite feature values')
    return sorted(values)[math.ceil(q*len(values))-1]

def extract_features(aircraft_ids,layout,*,Ncap,Omega_cap):
    """Input B must be independently audited first; scalar checks do not prove GEO."""
    n=len(aircraft_ids);ids=layout['beam_ids'];b=len(ids)
    if not n or len(set(aircraft_ids))!=n or not all(type(x) is str and x for x in aircraft_ids):raise ValueError('full original aircraft identities')
    if type(Ncap) is not int or not 1<=b<=Ncap<=min(200,n) or len(set(ids))!=b:raise ValueError('actual nonempty beam/cap domain')
    if type(Omega_cap) not in (int,float) or not math.isfinite(Omega_cap) or Omega_cap<=0:raise ValueError('positive Omega denominator')
    owners=layout['assignment'];widths=layout['widths_deg']
    if len(owners)!=n or len(widths)!=b or any(type(j) is not int or not 0<=j<b for j in owners):raise ValueError('full original ownership')
    if any(type(w) not in (int,float) or not math.isfinite(w) or not .3<=w<=1. for w in widths):raise ValueError('finite widths in degrees')
    counts=[0]*b
    for j in owners:counts[j]+=1
    if any(c==0 for c in counts):raise ValueError('empty beam forbidden')
    mean=n/b;peak=max(counts);top=max(1,math.ceil(.1*b))
    # Only scalar width-to-solid-angle arithmetic; no projection or ray pair.
    omega=math.fsum(2*math.pi*(1-math.cos(math.radians(w/2))) for w in widths)
    if omega>Omega_cap+1e-12:raise ValueError('audited B scalar Omega exceeds cap')
    values=[n,b,b/Ncap,omega/Omega_cap,peak,math.fsum(sorted(counts)[-top:])/top,
        math.sqrt(math.fsum((c-mean)**2 for c in counts)/b)/mean,
        counts.count(peak)/b,nearest_rank(counts,.5),math.fsum(widths)/b,nearest_rank(widths,.9),max(widths)]
    if any(not math.isfinite(v) for v in values):raise ValueError('no feature imputation')
    return values

def validate_vector(values):
    if len(values)!=12 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('exact12 finite causal features')
    return list(values)
