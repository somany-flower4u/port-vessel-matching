import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
import sys, zlib, numpy as np, pandas as pd
SP=os.environ.get("SEOKCHEON_WS", ".") + ''
sys.path.insert(0, SP+'/work/v27a/model/ko')
from vm_ko.dualenc import DualEncoder
from pathlib import Path
W=SP+'/work/'
cat=pd.read_csv(W+'vessel_catalog.csv',encoding='utf-8-sig'); ids=cat.vessel_id.to_numpy(); names=cat.vessel_name.to_numpy(); id2=dict(zip(ids,range(len(ids))))
gt=cat.gross_tonnage.to_numpy(float)
tr=pd.read_csv(W+'train.csv',encoding='utf-8-sig'); ko=tr[tr.vessel_name.str.contains('[가-힣]')]
u=ko.drop_duplicates(['vessel_name','target_vessel_id']).copy()
u['fold']=[zlib.crc32(str(v).encode())%5 for v in u.target_vessel_id]
u['t']=u.target_vessel_id.map(id2)
dirs=dict(a.split('=',1) for a in sys.argv[1:])
res={}
for lab,d in dirs.items():
    ranks=np.zeros(len(u),int); wr=np.zeros(len(u),int)
    for k in range(5):
        m=DualEncoder(Path(d+f'/de_f{k}_weights.npz'),Path(d+f'/de_f{k}_cat.npy'),ids,names)
        sel=np.flatnonzero(u.fold.values==k)
        for j in sel:
            r=u.iloc[j]; s=m.scores(r.vessel_name); t=r.t
            ranks[j]=int((s>s[t]).sum())
            g=r.gross_tonnage; win=np.isfinite(gt)&(gt>=g*0.94)&(gt<=g*1.065)
            wr[j]=int((s[win]>s[t]).sum()) if win[t] else -1
        del m
    res[lab]=ranks
    kr=u.target_vessel_id.map(dict(zip(cat.vessel_id,cat.flag_name))).values=='South Korea'
    print(f'{lab:18s} OOF real train Hangul names n={len(u)}: R@1 {np.mean(ranks==0):.3f} R@5 {np.mean(ranks<5):.3f} R@60 {np.mean(ranks<60):.3f} | in-GT-window R@1 {np.mean(wr==0):.3f} | KR-flag R@1 {np.mean(ranks[kr]==0):.3f} foreign R@1 {np.mean(ranks[~kr]==0):.3f}', flush=True)
print('GATE', {k:round(float(np.mean(v==0)),4) for k,v in res.items()})
