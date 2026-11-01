"""Set E: real train Hangul rows, out-of-fold DE (v28i fold models, vessel held out) hardest negatives inside the tonnage gate."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import sys, zlib, numpy as np, pandas as pd
from pathlib import Path
SP=os.environ.get("SEOKCHEON_WS", ".") + ''
sys.path.insert(0, SP+'/work/v27a/model/ko')
from vm_ko.dualenc import DualEncoder
cat=pd.read_csv(SP+'/work/vessel_catalog.csv',encoding='utf-8-sig'); ids=cat.vessel_id.to_numpy(); names=cat.vessel_name.to_numpy(); id2=dict(zip(ids,range(len(ids))))
gt=cat.gross_tonnage.to_numpy(float)
tr=pd.read_csv(SP+'/work/train.csv',encoding='utf-8-sig'); ko=tr[tr.vessel_name.str.contains('[가-힣]',na=False)].copy()
ko['fold']=[zlib.crc32(str(v).encode())%5 for v in ko.target_vessel_id]
d=SP+'/agents/round4/v28i/folds'; out=[]
for k in range(5):
    m=DualEncoder(Path(d+f'/de_f{k}_weights.npz'),Path(d+f'/de_f{k}_cat.npy'),ids,names)
    for _,r in ko[ko.fold==k].iterrows():
        s=m.scores(r.vessel_name).astype(float); t=id2[r.target_vessel_id]
        g=r.gross_tonnage
        if np.isfinite(g) and g>0:
            with np.errstate(divide='ignore',invalid='ignore'): ok=np.abs(np.log(g/gt))<=0.2
            s=np.where(ok,s,-np.inf); s[t]=m.scores(r.vessel_name)[t]
        s2=s.copy(); s2[t]=-np.inf; o=np.argsort(-s2)[:2]
        out.append(dict(sample_id=r.sample_id,ko=r.vessel_name,true_id=r.target_vessel_id,true_en=names[t],s_true=s[t],
                        other_id=ids[o[0]],other_en=names[o[0]],s_other=s2[o[0]],other2_id=ids[o[1]],other2_en=names[o[1]],s_other2=s2[o[1]],de_wrong=bool(s2[o[0]]>s[t])))
    del m
E=pd.DataFrame(out); E.to_csv(SP+'/agents/llmjudge/setE.csv',index=False)
print(len(E),'rows; OOF DE wrong (gated)',int(E.de_wrong.sum()), 'vessels', E[E.de_wrong].true_id.nunique())
print(E[E.de_wrong].drop_duplicates(['ko','other_en']).head(30)[['ko','true_en','other_en']].to_string())
