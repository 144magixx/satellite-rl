"""Pure native ownership adjudication; no spawning or signalling from preparation."""
KEYS=('pid','birth_ticks','boot_id','uid')
def same_process(left,right):return bool(left and right and all(left.get(k)==right.get(k) for k in KEYS))

def adjudicate(registered,observed,expected_boot):
 """A numeric process group, argv or later start time never registers a child."""
 if any(v.get('boot_id')!=expected_boot for v in registered+observed):raise ValueError('boot differs, ownership unknown')
 if len({v['pid'] for v in registered})!=len(registered):raise ValueError('ambiguous registered PID')
 known={v['pid']:v for v in registered};live=[];unknown=[];zombies=[]
 for value in observed:
  match=same_process(known.get(value['pid']),value)
  if value['state']=='Z':zombies.append(dict(identity=value,registered_exact=match));continue
  if not match:unknown.append(value)
  else:live.append(value)
 return dict(execution_ended=not live and not unknown,registered_live=live,unknown_live=unknown,zombies=zombies,signal_eligible=[] if unknown else live,no_group_signal=True)
