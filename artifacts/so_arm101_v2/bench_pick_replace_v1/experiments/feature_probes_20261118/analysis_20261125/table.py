import json,re,os,sys,importlib.util
import numpy as np
spec=importlib.util.spec_from_file_location('fp','tools/feature_probes.py'); fp=importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
m=json.load(open(R+'/matrix.json')); rec=json.load(open(os.path.dirname(__file__)+'/recount.json'))
def pooled(cell,probe):
    mem=[c for c in rec if re.sub(r"_s\d+(?=_exec\d+$|__)","",c)==cell and c!=cell]
    have=[rec[c][probe] for c in mem]
    agg={k:sum(h[k] for h in have) for k in ("fired","control_pickups","probe_pickups","probe_successes","control_successes")}
    g=[h['follow_gain_mean'] for h in have if h.get('follow_gain_mean') is not None]
    if g: agg['follow_gain_mean']=float(np.mean(g)); agg['follow_gain_ci95']=None; agg['seeds']=g
    if 'regrasps' in have[0]: agg['regrasps']=sum(h['regrasps'] for h in have); agg['regrasp_rate']=agg['regrasps']/agg['fired'] if agg['fired'] else None
    if probe=='move_square': agg['compared']=sum(h['compared'] for h in have); agg['on_moved_square']=sum(h['on_moved_square'] for h in have)
    agg['verdict']=fp.verdict(probe,agg); return agg
names={'base':'baseline','pert':'perturbed','physv2':'physics v2','lift':'high-lift','fair':'fair'}
scn={'so101':'nominal','so101_servo':'servo','so101_servo_lift100':'high-lift'}
def label(cell):
    pol,sc=cell.split('__'); parts=pol.split('_'); ex='30' if 'exec30' in pol else '90'
    return f"{names[parts[0]]} {parts[1].upper()}, execute {ex}", scn[sc]
tot={'held':0,'failed':0}
rows=[]
for p in m['predictions']:
    cell,probe=p['cell'],p['probe']
    e=(m['pooled'].get(cell) or m['cells'].get(cell) or {}).get(probe); src='matrix.json'
    if e is None:
        e=rec[cell][probe] if cell in rec else pooled(cell,probe); src='recomputed'
    v=e['verdict']; held=v in p['predicted']; tot['held' if held else 'failed']+=1
    if probe in('shift_placement',): num=f"gain {e['follow_gain_mean']:+.2f}, pickups {e['probe_pickups']} of {e['control_pickups']}"
    elif probe=='swap_elbow': num=f"pickups {e['probe_pickups']} of {e['control_pickups']}"
    elif probe=='drop_cube': num=f"{e['regrasps']} of {e['fired']} drops"
    else:
        num='nothing to compare (0 pickups unprobed)' if e.get('follow_gain_mean') is None else f"gain {e['follow_gain_mean']:+.2f}, on moved square {e.get('on_moved_square','?')} of {e.get('compared','?')}"
    pol,sc=label(cell)
    rows.append((probe,sc,pol,' or '.join(p['predicted']),v,num,'held' if held else 'FAILED',src))
order={'shift_placement':0,'swap_elbow':1,'drop_cube':2,'move_square':3}; so={'nominal':0,'servo':1,'high-lift':2}
rows.sort(key=lambda r:(order[r[0]],so[r[1]]))
print('| Probe | Scene | Policy | Predicted | Observed | Numbers | Result | Source |\n|---|---|---|---|---|---|---|---|')
for r in rows: print('| `'+r[0]+'` | '+' | '.join(r[1:])+' |')
print(tot)
from collections import Counter
print(Counter((r[0],r[6],r[7]) for r in rows))
