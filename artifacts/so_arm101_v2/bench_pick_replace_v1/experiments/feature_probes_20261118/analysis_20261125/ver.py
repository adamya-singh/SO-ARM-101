import json,glob,os,numpy as np
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
st=[]
for c in sorted(glob.glob(R+'/cells/*')):
    n=os.path.basename(c)
    rc=json.load(open(c+'/rows_control.json')); rp=json.load(open(c+'/rows_move_square.json'))
    tc=dict(np.load(c+'/trace_control.npz')); tp=dict(np.load(c+'/trace_move_square.npz'))
    idx={r['scenario_id']:i for i,r in enumerate(rc)}
    sel=[(i,idx[r['scenario_id']]) for i,r in enumerate(rp) if r['scenario_id'] in idx and r['event_action'] is not None and rc[idx[r['scenario_id']]]['pickup_completed']]
    if not sel: continue
    ip=np.array([a for a,b in sel]); ic=np.array([b for a,b in sel])
    sh=np.array([rp[i]['shift_xy_m'] for i in ip]); L=np.linalg.norm(sh,axis=1); u=sh/L[:,None]; perp=np.stack([-u[:,1],u[:,0]],1)
    mv=(tp['cube'][ip,-1,:2]-tc['cube'][ic,-1,:2]).astype(float); g=(mv*u).sum(1)/L; pp=(mv*perp).sum(1)/L
    dz=np.abs(tp['cube'][ip,-1,2]-tc['cube'][ic,-1,2])*1000
    if n.startswith('teacher'):
        jd=np.abs(tp['jaw'][ip]-tc['jaw'][ic]).max()*1000
        print(n,'whole-episode max jaw diff mm',round(float(jd),4),'gain',round(float(g.mean()),4)); continue
    st.append((n,len(ip),g.mean(),pp.mean(),(g<0).mean(),np.median(dz),dz.max()))
for s in st: print(s[0],s[1],*[round(float(x),3) for x in s[2:]])
a=np.array([s[2:] for s in st],float); print('gain',a[:,0].min(),a[:,0].max(),'perp',a[:,1].min(),a[:,1].max(),'neg max',a[:,2].max(),'neg max n>=45',max(s[4] for s in st if s[1]>=45),'dz med max',a[:,3].max(),'dz max',a[:,4].max())
for grp in (['pert_cnn_s101__so101','pert_cnn_s202__so101','pert_cnn_s303__so101'],['pert_act__so101'],['base_act_exec30__so101']):
    tot=0;up=0;reo=0
    for n in grp:
        c=f'{R}/cells/{n}'; rp=json.load(open(c+'/rows_drop_cube.json')); t=dict(np.load(c+'/trace_drop_cube.npz'))
        for i,r in enumerate(rp):
            e=r['event_action']
            if e is None or r['max_height_gain_after_event_m']<0.010: continue
            tot+=1; h=t['height'][i]; g=t['gripper'][i]
            up+= h[min(e+10,479)]>=0.010
            a=e+10+int(np.argmax(h[e+10:]>=0.010)); reo+= g[e:a+1].max()>g[e]+0.05
    print(grp[0],tot,'up at first counted',int(up),'reopened',int(reo))
