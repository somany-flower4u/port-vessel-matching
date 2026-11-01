"""project.py MODEL_TAG FORM : projected fixed/broken on the hidden test (1,064 Hangul rows, 18 errors, ~11 with truth at rank 2)
under 'swap to rank-2 if ranker margin < tau and l2 - l1 > kappa'.
Touched-row counts per tau: v29v ranker margins on domtest (correct rows / rank2-true errors), scaled to the test.
Judge rates from REAL Hangul pairs: hard set H = real sister pairs of type roman/digit/close (A) + real OOF-DE error pairs (E);
fix(kappa)=P(d>kappa) on H, flip(kappa)=P(d<-kappa) on H (conservative: near-tie correct rows are look-alike pairs);
flipB(kappa)= break rate vs actual ranker rank-2 on all real train rows (B, unconditional)."""
import os
import sys, numpy as np, pandas as pd
sys.argv=[sys.argv[0]]+sys.argv[1:]
tag,form=sys.argv[1],sys.argv[2]
L=os.environ.get("SEOKCHEON_WS", ".") + '/agents/llmjudge'
import importlib.util
sys.argv=['analyze.py',f'{L}/s{tag}_{form}.csv']
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(open(L+'/analyze.py').read())
H1=Au[Au.typ!='far'][['ko','true_en','other_en','d']]
Ew=E[E.de_wrong].drop_duplicates(['ko','other_en'])[['ko','true_en','other_en']].assign(d=E[E.de_wrong].drop_duplicates(['ko','other_en']).d1)
H=pd.concat([H1,Ew]).drop_duplicates(['ko','true_en','other_en'])
b2=B[B.k==2]
dm=pd.read_pickle(L+'/v29v_domtest.pkl'); dm['m']=dm.s1-dm.s2; c=dm[dm.ok1]; f=dm[~dm.ok1&dm.ok2]
NC,NF=1064-18,11
print(f'== {tag} {form}: hard real pairs n={len(H)}; Dsynthetic n={len(D)}')
print('kappa  fix(H)  flip(H)  flipB  fixD')
for k in KS: print(f'{k:5}  {(H.d>k).mean():.3f}  {(H.d<-k).mean():.3f}  {(b2.d<-k).mean():.4f}  {(D.d>k).mean():.2f}')
print('projected on hidden test: cells fixed / broken(H-rate) / broken(B-rate)')
print('tau  touchedC touchedF | '+' '.join(f'{"k="+str(k):>17}' for k in KS))
for t in (1,2,3,5,8):
    tc=(c.m<t).mean()*NC; tf=(f.m<t).mean()*NF
    print(f'{t:<4} {tc:7.1f} {tf:7.1f}  | '+' '.join(f'{tf*(H.d>k).mean():5.1f}/{tc*(H.d<-k).mean():4.1f}/{tc*(b2.d<-k).mean():4.2f}' for k in KS))
