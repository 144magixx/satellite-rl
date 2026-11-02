"""V2 fixed half-up integer assignment costs; physical predicates stay float64."""
import math

FAMILIES = ('Gd', 'Gr', 'Gw')


def edge_cost(distance_deg, reference_fullwidth, changed, n, family):
    if (type(n) is not int or n < 1 or family not in FAMILIES or
        type(changed) is not bool or not math.isfinite(distance_deg) or
        not math.isfinite(reference_fullwidth) or distance_deg < 0 or
        not .3 <= reference_fullwidth <= 1.):
        raise ValueError('qualified cost inputs')
    d, ref = float(distance_deg), float(reference_fullwidth)
    distance = math.floor(1000000*(d*d)+.5)
    w = 2*d
    expansion = math.floor(1000*max(0., (w*w)-(ref*ref))+.5)
    if distance > 250001 or expansion > 1001:
        raise ValueError('eligible edge exceeds registered integer bounds')
    weight = n*250001+1
    return (distance if family == 'Gd' else
            int(changed)*weight+distance if family == 'Gr' else
            expansion*weight+distance)


def matrix(distances, eligible, beam_ids, original_owner_ids, reference_widths,
           family, ledger):
    n, b = len(original_owner_ids), len(beam_ids)
    if len(distances) != n or len(eligible) != n or len(reference_widths) != b:
        raise ValueError('full cost table shapes')
    result = []
    ledger.scan('cost_full_table_entries', n*b)
    for i in range(n):
        if len(distances[i]) != b or len(eligible[i]) != b:
            raise ValueError('complete cost table rows')
        row=[]
        for j in range(b):
            row.append(edge_cost(distances[i][j], reference_widths[j],
                       beam_ids[j] != original_owner_ids[i], n, family)
                       if eligible[i][j] else 0)
        result.append(row)
    return result
