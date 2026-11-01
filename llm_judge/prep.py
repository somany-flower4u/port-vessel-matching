"""Build the evaluation sets and the unique (en, ko) pair list to score.
A real sister pairs, B real truth vs ranker rank-2/3 (v29v, train rows), C domtest (synthetic Hangul, v29v ranker top-3, m<MMAX),
D round4 sister bench (synthetic) v28i wrong rows: truth vs wrong prediction."""
import os
import pandas as pd, numpy as np, sys
SP=os.environ.get("SEOKCHEON_WS", ".") + ''
L=SP+'/agents/llmjudge'; sys.path.insert(0,L)
from judge import SHOT_NAMES
cat=pd.read_csv(SP+'/work/vessel_catalog.csv',encoding='utf-8-sig'); nm=dict(zip(cat.vessel_id,cat.vessel_name))
shot_ids={v for v,n in nm.items() if n in SHOT_NAMES}
MMAX=6.0
A=pd.read_csv(L+'/sister_pairs.csv'); A['set']='A'
B=pd.read_pickle(L+'/v29v_train.pkl'); B=B[~B.true_id.isin(shot_ids)]
rows=[]
for _,r in B.iterrows():
    for k in (2,3):
        if isinstance(r.get(f'id{k}'),str): rows.append(dict(set='B',sample_id=r.sample_id,ko=r.vessel_name,true_id=r.true_id,true_en=nm[r.true_id],other_id=r[f'id{k}'],other_en=nm[r[f'id{k}']],k=k,m=r.s1-r[f's{k}']))
B=pd.DataFrame(rows)
C=pd.read_pickle(L+'/v29v_domtest.pkl'); C['m']=C.s1-C.s2; C=C[C.m<MMAX]
rows=[]
for _,r in C.iterrows():
    for k in (1,2,3):
        if isinstance(r.get(f'id{k}'),str): rows.append(dict(set='C',sample_id=r.sample_id,ko=r.vessel_name,true_id=r.true_id,cand_id=r[f'id{k}'],cand_en=nm[r[f'id{k}']],k=k,s=r[f's{k}']))
C=pd.DataFrame(rows)
D=pd.concat([pd.read_csv(SP+f'/agents/round4/sister/res_{v}.csv').assign(pkg=v) for v in ('v27a','v28i')]); D=D[~D.ok].drop_duplicates(['sample_id','pred'])
D=pd.DataFrame(dict(set='D',sample_id=D.sample_id,ko=D.vessel_name,true_id=D.target_vessel_id,true_en=D.cat_name,other_id=D.pred,other_en=D.pred.map(nm),type=D.type))
for n,x in (('A',A),('B',B),('C',C),('D',D)): x.to_csv(f'{L}/set{n}.csv',index=False); print(n,len(x))
U=pd.concat([A[['true_en','ko']].set_axis(['en','ko'],axis=1),A[['other_en','ko']].set_axis(['en','ko'],axis=1),
             B[['true_en','ko']].set_axis(['en','ko'],axis=1),B[['other_en','ko']].set_axis(['en','ko'],axis=1),
             C[['cand_en','ko']].set_axis(['en','ko'],axis=1),
             D[['true_en','ko']].set_axis(['en','ko'],axis=1),D[['other_en','ko']].set_axis(['en','ko'],axis=1)]).drop_duplicates().reset_index(drop=True)
U.to_csv(L+'/pairs.csv',index=False); print('unique pairs',len(U))
