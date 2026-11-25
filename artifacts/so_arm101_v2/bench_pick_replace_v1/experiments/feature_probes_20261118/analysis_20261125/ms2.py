import json,glob,os,sys
import numpy as np
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
for c in sorted(glob.glob(R+'/cells/*')):
    n=os.path.basename(c)
    rc=json.load(open(c+'/rows_control.json')); rp=json.load(open(c+'/rows_move_square.json'))
    tc=dict(np.load(c+'/trace_control.npz')); tp=dict(np.load(c+'/trace_move_square.npz'))
    idx={r['scenario_id']:i for i,r in enumerate(rc)}
    sel=[(i,idx[r['scenario_id']]) for i,r in enumerate(rp) if r['scenario_id'] in idx and r['event_action'] is not None and rc[idx[r['scenario_id']]]['pickup_completed']]
    if not sel: print(n,'no pairs'); continue
    ip=np.array([a for a,b in sel]); ic=np.array([b for a,b in sel])
    ev=np.array([rp[i]['event_action'] for i in ip]); sh=np.array([rp[i]['shift_xy_m'] for i in ip]); L=np.linalg.norm(sh,axis=1); u=sh/L[:,None]
    dev=np.abs(tp['requested'][ip]-tc['requested'][ic]).max(2)
    T=np.arange(dev.shape[1])[None,:]
    pre=np.where(T<ev[:,None],dev,0).max(1)
    ex=30 if 'exec30' in n else 90
    nxt=(np.ceil(ev/ex)*ex).astype(int)
    mid=np.array([dev[k,ev[k]:nxt[k]].max() if nxt[k]>ev[k] else 0 for k in range(len(ev))])
    post=np.where(T>=nxt[:,None],dev,0).max(1)
    first=np.array([(np.nonzero(d>0.005)[0][0] if (d>0.005).any() else -1) for d in dev])
    ok=first>=0
    jawpre=np.array([np.abs(tp['jaw'][ip[k],:nxt[k],:2]-tc['jaw'][ic[k],:nxt[k],:2]).max()*1000 for k in range(len(ev))])
    cubepre=np.array([np.abs(tp['cube'][ip[k],:nxt[k]]-tc['cube'][ic[k],:nxt[k]]).max()*1000 for k in range(len(ev))])
    print(f"{n} | n {len(ev)} | event med {int(np.median(ev))} | cmd dev: before event max {pre.max():.4f}, event..next read max {mid.max():.4f} med {np.median(mid):.5f}, after next read med {np.median(post):.3f} | first dev>0.005: none {int((~ok).sum())}, at next read {int((first[ok]==nxt[ok]).sum())}/{int(ok.sum())}, before next read {int((first[ok]<nxt[ok]).sum())} | jaw xy diff before next read max {jawpre.max():.3f} mm, cube diff {cubepre.max():.3f} mm")
