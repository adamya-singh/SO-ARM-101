import json,glob,os,sys,re,importlib.util
import numpy as np
sys.path.insert(0,'src')
from so_arm101_v2.simulation import probes as P
spec=importlib.util.spec_from_file_location('fp','tools/feature_probes.py'); fp=importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
m=json.load(open(R+'/matrix.json'))
out={}
for c in sorted(glob.glob(R+'/cells/*')):
    n=os.path.basename(c)
    idn=json.load(open(c+'/identity.json'))
    rows={};tr={}
    for p in ('control','shift_cube','drop_cube','move_square','shift_placement'):
        rows[p]=json.load(open(f'{c}/rows_{p}.json')); tr[p]=dict(np.load(f'{c}/trace_{p}.npz'))
    out[n]={}
    for p in ('shift_cube','drop_cube','move_square','shift_placement'):
        s=P.ProbeSpec(p,at_action=idn['at_action'],magnitude_m=idn['magnitude_m'],cube_magnitude_m=idn['cube_magnitude_m'],seed=idn['probe_seed'])
        have={r['scenario_id'] for r in rows['control']}
        keep=[i for i,r in enumerate(rows[p]) if r['scenario_id'] in have]
        rr=[rows[p][i] for i in keep]; tt={k:v[keep] for k,v in tr[p].items()}
        q=P.paired_metrics(s,rows['control'],tr['control'],rr,tt); q['verdict']=fp.verdict(p,q)
        out[n][p]=q
        # parity with matrix where matrix has it
        mm=m['cells'].get(n,{}).get(p)
        if mm is not None:
            for k in ('follow_gain_mean','regrasp_rate','probe_pickups','compared','fired'):
                if k in mm and mm[k]!=q.get(k): print('MISMATCH',n,p,k,mm[k],q.get(k))
json.dump(out,open(os.path.dirname(__file__)+'/recount.json','w'),indent=1,default=float)
def f(x): return 'None' if x is None else f'{x:+.3f}'
for n,e in out.items():
    if n.endswith('__so101'):
        q=e['move_square']; d=e['drop_cube']; sc=e['shift_cube']
        print(n,'MS cmp',q['compared'],'fired',q['fired'],'gain',f(q['follow_gain_mean']),'med',f(q['follow_gain_median']),'ci',q['follow_gain_ci95'] and [round(a,2) for a in q['follow_gain_ci95']],'onsq',q['on_moved_square'],'succ',q['probe_successes'],q['control_successes'],q['verdict'],'| DROP',d['regrasps'],d['fired'],d['verdict'],'| SC',f(sc['follow_gain_mean']),sc['probe_pickups'],sc['control_pickups'],sc['verdict'])
# pooled as cmd_report
fam={}
for n in out:
    pol,scn=n.split('__',1); fa=re.sub(r"_s\d+(?=_exec\d+$|$)","",pol)
    if fa!=pol: fam.setdefault(f'{fa}__{scn}',[]).append(n)
for fa,mem in fam.items():
    for p in ('move_square','drop_cube','shift_cube'):
        have=[out[c][p] for c in mem]
        agg={k:sum(h[k] for h in have) for k in ("fired","control_pickups","probe_pickups","probe_successes","control_successes")}
        g=[h['follow_gain_mean'] for h in have if h.get('follow_gain_mean') is not None]
        if g: agg['follow_gain_mean']=float(np.mean(g)); agg['follow_gain_ci95']=None
        if 'regrasps' in have[0]:
            agg['regrasps']=sum(h['regrasps'] for h in have); agg['regrasp_rate']=agg['regrasps']/agg['fired'] if agg['fired'] else None
        if p=='move_square': agg['compared']=sum(h['compared'] for h in have); agg['on']=sum(h['on_moved_square'] for h in have)
        print('POOL',fa,p,{k:(round(v,3) if isinstance(v,float) else v) for k,v in agg.items()},fp.verdict(p,agg))
