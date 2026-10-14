"""Independent pure V2 support selector fed pre-given chord/angle rows."""

def domain(counts):
    if not counts or any(type(c) is not int or c<=0 for c in counts):raise ValueError('full positive current counts')
    p=max(counts);peaks=[i for i,c in enumerate(counts) if c==p];s=len(peaks)
    candidates=[j for j,c in enumerate(counts) if c<p and s<=30 and s*p+c<=128]
    return peaks,candidates

def select(counts,beam_ids,contact_rows,limit):
    if len(counts)!=len(beam_ids) or len(set(beam_ids))!=len(beam_ids):raise ValueError('stable full beam IDs')
    peaks,candidates=domain(counts);s=len(peaks);p=max(counts)
    if not candidates:return dict(peaks=peaks,support=None,region=[],reason='unsupported_no_slack_support')
    contacting=[]
    for j in candidates:
        rows=contact_rows.get(j)
        if rows is None or len(rows)!=s:raise ValueError('complete independently cached support-to-peak rows')
        if any(chord<=limit for chord,angle in rows):
            contacting.append(j)
    if not contacting:return dict(peaks=peaks,support=None,region=[],reason='unsupported_no_contacting_support')
    support=min(contacting,key=lambda j:(((s*p+counts[j])+s)//(s+1),counts[j],min(angle for chord,angle in contact_rows[j]),beam_ids[j]))
    return dict(peaks=peaks,support=support,region=sorted(peaks+[support],key=lambda j:beam_ids[j]),reason=None)

def reader_pair_reservation(n,m,*,new_contact,new_region,eligible):
    if type(n) is not int or n<1 or type(m) is not int or m<1:raise ValueError('complete positive dimensions')
    if eligible and m>128:raise ValueError('no over128 active member matrix')
    return 2*n+(5100 if new_contact else 0)+(m*m if eligible and new_region else 0)+(m if eligible else 0)
