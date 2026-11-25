import json,glob,os,sys
import numpy as np
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
names=sys.argv[1:] or [os.path.basename(c) for c in sorted(glob.glob(R+'/cells/*'))]
print('cell | n | event act med[min,max] | first cmd deviation act: med, share at multiple of 30/90, share before next mult of exec | gain mean | perp/|shift| mean, abs-mean | share gain>=.5, <=.2, <0 | dz mm med, max|dz| | jaw gain at event+? | inside moved sq: gain>=.5, gain<=.2 | control cube final travel from start mm med')
for n in names:
    c=f'{R}/cells/{n}'
    rc=json.load(open(c+'/rows_control.json')); rp=json.load(open(c+'/rows_move_square.json'))
    tc=dict(np.load(c+'/trace_control.npz')); tp=dict(np.load(c+'/trace_move_square.npz'))
    idx={r['scenario_id']:i for i,r in enumerate(rc)}
    sel=[(i,idx[r['scenario_id']]) for i,r in enumerate(rp) if r['scenario_id'] in idx and r['event_action'] is not None and rc[idx[r['scenario_id']]]['pickup_completed']]
    if not sel: print(n,'no pairs'); continue
    ip=np.array([a for a,b in sel]); ic=np.array([b for a,b in sel])
    ev=np.array([rp[i]['event_action'] for i in ip]); sh=np.array([rp[i]['shift_xy_m'] for i in ip])
    L=np.linalg.norm(sh,axis=1); u=sh/L[:,None]; perp=np.stack([-u[:,1],u[:,0]],1)
    mv=(tp['cube'][ip,-1,:2]-tc['cube'][ic,-1,:2]).astype(float)
    g=(mv*u).sum(1)/L; pp=(mv*perp).sum(1)/L
    dz=(tp['cube'][ip,-1,2]-tc['cube'][ic,-1,2])*1000
    dev=np.abs(tp['requested'][ip]-tc['requested'][ic]).max(2)
    first=np.array([ (np.nonzero(d>1e-6)[0][0] if (d>1e-6).any() else -1) for d in dev])
    ok=first>=0
    ex=30 if 'exec30' in n else 90
    before=np.mean(first[ok]<ev[ok]); m30=np.mean(first[ok]%30==0); m90=np.mean(first[ok]%90==0)
    nxt=np.ceil(ev/ex)*ex
    atnext=np.mean(first[ok]==nxt[ok])
    # jaw xy displacement projected, at the action where control cube height last >5mm (approx release): use time of control max jaw separation? use final-60
    jg=[]
    for k in (300,330,360,390,420,450,479):
        jm=(tp['jaw'][ip,k,:2]-tc['jaw'][ic,k,:2]).astype(float); jg.append(round(float(((jm*u).sum(1)/L).mean()),2))
    ins=np.array([bool(rp[i]['final_footprint_inside']) for i in ip])
    hi=g>=0.5; lo=g<=0.2
    trav=np.linalg.norm(tc['cube'][ic,-1,:2]-tc['cube'][ic,0,:2],axis=1)*1000
    print(n,'|',len(ip),'|',int(np.median(ev)),[int(ev.min()),int(ev.max())],'| first',int(np.median(first[ok])),'undeviated',int((~ok).sum()),'before_event',round(float(before),2),'m30',round(float(m30),2),'m90',round(float(m90),2),'at_next_read',round(float(atnext),2),'| gain',round(float(g.mean()),3),'| perp',round(float(pp.mean()),3),round(float(np.abs(pp).mean()),3),'| hi',round(float(hi.mean()),2),'lo',round(float(lo.mean()),2),'neg',round(float((g<0).mean()),2),'| dz',round(float(np.median(dz)),2),round(float(np.abs(dz).max()),1),'| jawgain@300..479',jg,'| inside hi',int(ins[hi].sum()),int(hi.sum()),'lo',int(ins[lo].sum()),int(lo.sum()),'| ctl travel',round(float(np.median(trav)),1), 'ctl final z-start z mm med',round(float(np.median((tc['cube'][ic,-1,2]-tc['cube'][ic,0,2])*1000)),2))
