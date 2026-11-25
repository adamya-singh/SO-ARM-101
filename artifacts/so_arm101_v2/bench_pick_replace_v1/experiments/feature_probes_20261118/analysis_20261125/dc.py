import json,os,numpy as np
R='artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/feature_probes_20261118'
for n in ['pert_cnn_s101__so101','pert_cnn_s202__so101','pert_cnn_s303__so101','pert_act__so101','base_act_exec30__so101','base_act__so101','base_cnn_s101__so101']:
    c=f'{R}/cells/{n}'; rp=json.load(open(c+'/rows_drop_cube.json')); t=dict(np.load(c+'/trace_drop_cube.npz'))
    out=[]
    for i,r in enumerate(rp):
        e=r['event_action']
        if e is None or r['max_height_gain_after_event_m'] is None or r['max_height_gain_after_event_m']<0.010: continue
        h=t['height'][i]; g=t['gripper'][i]
        a=e+10+int(np.argmax(h[e+10:]>=0.010))
        peak=float(h[e+10:].max())*1000
        dur=int((h[e+10:]>=0.010).sum())
        strict=int(((t['flags'][i,e+10:]&1)>0).sum())
        out.append((e,a,round(peak,1),dur,strict,round(float(g[e]),2),round(float(g[e:a+1].max()),2),round(float(h[-1])*1000,1)))
    print(n,len(out),'(event, first action >=10mm, peak mm, actions above 10mm, strict-grasp frames after, gripper at event, max gripper event..relift, final height mm)')
    for o in out[:8]: print('   ',o)
    if out:
        o=np.array(out); print('    median first-relift delay',np.median(o[:,1]-o[:,0]),'median peak',np.median(o[:,2]),'median dur',np.median(o[:,3]),'with strict frames',int((o[:,4]>0).sum()),'gripper reopened (max>event+0.1)',int((o[:,6]>o[:,5]+0.1).sum()))
