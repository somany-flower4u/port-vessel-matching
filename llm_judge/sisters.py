"""Real train Hangul rows whose truth vessel has a same-GT, same-flag, same-length catalog sister differing by exactly one token."""
import os
import numpy as np, pandas as pd, sys
SP=os.environ.get("SEOKCHEON_WS", ".") + ''
cat=pd.read_csv(SP+'/work/vessel_catalog.csv',encoding='utf-8-sig'); ids=cat.vessel_id.to_numpy(); names=cat.vessel_name.to_numpy(); id2=dict(zip(ids,range(len(ids))))
gt=cat.gross_tonnage.to_numpy(float); fl=cat.flag_name.to_numpy()
tr=pd.read_csv(SP+'/work/train.csv',encoding='utf-8-sig'); ko=tr[tr.vessel_name.str.contains('[가-힣]',na=False)].copy()
toks=[tuple(str(n).upper().replace('.',' ').replace('-',' ').split()) for n in names]
def sisters(t):
    T=toks[t]; c=np.flatnonzero((gt==gt[t])&(fl==fl[t]))
    return [j for j in c if j!=t and len(toks[j])==len(T) and sum(a!=b for a,b in zip(toks[j],T))==1 and names[j].upper()!=names[t].upper()]
S={v:sisters(id2[v]) for v in ko.target_vessel_id.unique()}
S={v:s for v,s in S.items() if s}
print('train Hangul vessels with a one-word sister:',len(S),' rows:',int(ko.target_vessel_id.isin(S).sum()))
out=[]
for _,r in ko[ko.target_vessel_id.isin(S)].iterrows():
    for j in S[r.target_vessel_id]:
        out.append(dict(sample_id=r.sample_id,ko=r.vessel_name,true_id=r.target_vessel_id,true_en=names[id2[r.target_vessel_id]],other_id=ids[j],other_en=names[j]))
pd.DataFrame(out).to_csv(sys.argv[1],index=False)
print(len(out),'pairs')
