import json,glob,os
import numpy as np
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
m=json.load(open(R+'/matrix.json'))
pts=m['link3']['points']
hi=[p for p in pts if p['follow_gain']>=0.5]; lo=[p for p in pts if p['follow_gain']<=0.2]; mid=[p for p in pts if 0.2<p['follow_gain']<0.5]
print('hi',len(hi),'meet',sum(p['pickup_retention']>=0.7 for p in hi)); print([ (p['cell'],round(p['follow_gain'],2),round(p['pickup_retention'],3)) for p in hi if p['pickup_retention']<0.7])
print('lo',len(lo),'meet',sum(p['pickup_retention']<=0.3 for p in lo)); print([ (p['cell'],round(p['follow_gain'],2),round(p['pickup_retention'],3)) for p in lo if p['pickup_retention']>0.3])
print('mid',[(p['cell'],round(p['follow_gain'],2),round(p['pickup_retention'],3)) for p in mid])
rank=lambda v: np.argsort(np.argsort(v)).astype(float)
nom=[p for p in pts if p['cell'].endswith('__so101')]
g=np.array([p['follow_gain'] for p in nom]);k=np.array([p['pickup_retention'] for p in nom]); print('nominal',len(nom),np.corrcoef(rank(g),rank(k))[0,1])
# per-episode within cell
print('per-episode: among control-picked, probe pickup share by episode gain>=0.5 / <=0.2')
for c in sorted(glob.glob(R+'/cells/*')):
    n=os.path.basename(c)
    if n.startswith('teacher'): pass
    rc=json.load(open(c+'/rows_control.json')); rp=json.load(open(c+'/rows_shift_placement.json'))
    tc=np.load(c+'/trace_control.npz')['jaw']; tp=np.load(c+'/trace_shift_placement.npz')['jaw']
    idx={r['scenario_id']:i for i,r in enumerate(rc)}
    keep=[i for i,r in enumerate(rp) if r['scenario_id'] in idx]
    ic=np.array([idx[rp[i]['scenario_id']] for i in keep]); ip=np.array(keep)
    sh=np.array([rp[i]['shift_xy_m'] for i in ip]); mv=(tp[ip,250,:2]-tc[ic,250,:2]).astype(float)
    g=(mv*sh).sum(1)/(sh*sh).sum(1)
    pc=np.array([rc[i]['pickup_completed'] for i in ic]); pp=np.array([rp[i]['pickup_completed'] for i in ip])
    a=pc&(g>=0.5); b=pc&(g<=0.2); mid=pc&(g>0.2)&(g<0.5)
    res=np.linalg.norm(mv-sh,axis=1)*1000
    print(f"{n}: ctl picked {int(pc.sum())}; both {int((pc&pp).sum())}; probe-only {int((~pc&pp).sum())}; gain>=.5: {int(pp[a].sum())}/{int(a.sum())}; .2-.5: {int(pp[mid].sum())}/{int(mid.sum())}; <=.2: {int(pp[b].sum())}/{int(b.sum())}; residual mm med where ctl picked&probe picked {np.median(res[pc&pp]) if (pc&pp).any() else float('nan'):.1f} / ctl picked&probe missed {np.median(res[pc&~pp]) if (pc&~pp).any() else float('nan'):.1f}")
