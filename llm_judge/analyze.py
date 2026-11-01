"""analyze.py SCORES_CSV [SCORES_CSV2 ...]  (scores from judge.py over pairs.csv; several files = summed lp ensemble)"""
import sys, re, numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
L=os.environ.get("SEOKCHEON_WS", ".") + '/agents/llmjudge'
S=None
for f in sys.argv[1:]:
    x=pd.read_csv(f)[['en','ko','lp']]
    S=x if S is None else S.merge(x,on=['en','ko'],suffixes=('','_b')).assign(lp=lambda d:d.lp+d.lp_b).drop(columns='lp_b')
LP={(e,k):v for e,k,v in zip(S.en,S.ko,S.lp)}
lp=lambda e,k: LP.get((e,k),np.nan)
KS=[0,0.5,1,2,3,5,8]
def q(v): return ' '.join(f'{p}%:{np.percentile(v,p):+.2f}' for p in (5,10,25,50,75,90))
def toks(n): return str(n).upper().replace('.',' ').replace('-',' ').split()
ROM=set('I II III IV V VI VII VIII IX X XI XII'.split())
def dtype(a,b):
    ta,tb=toks(a),toks(b)
    if len(ta)!=len(tb): return 'other'
    d=[(x,y) for x,y in zip(ta,tb) if x!=y]
    if not d: return 'other'
    x,y=d[0]
    if x in ROM or y in ROM: return 'roman'
    if re.search(r'\d',x+y): return 'digit'
    return 'close' if Levenshtein.normalized_similarity(x,y)>=0.5 else 'far'
print('='*20, ' + '.join(sys.argv[1:]))
# ---- A
A=pd.read_csv(L+'/setA.csv'); A['d']=[lp(t,k)-lp(o,k) for t,k,o in zip(A.true_en,A.ko,A.other_en)]
A['typ']=[dtype(t,o) for t,o in zip(A.true_en,A.other_en)]
Au=A.drop_duplicates(['ko','true_en','other_en'])
print(f"A real sister pairs: rows-pairs {len(A)} acc {(A.d>0).mean():.3f} | unique (spelling,sister) {len(Au)} acc {(Au.d>0).mean():.3f}")
rowwin=A.groupby('sample_id').d.min()
print(f"   per row truth beats ALL sisters: {(rowwin>0).mean():.3f} of {len(rowwin)} rows")
for t,g in Au.groupby('typ'): print(f"   type {t:6s} n={len(g):4d} acc {(g.d>0).mean():.3f}  margin {q(g.d)}")
print('   margin all:', q(Au.d))
print('   P(d>k) fix-rate on unique pairs:', ' '.join(f'k{k}:{(Au.d>k).mean():.3f}' for k in KS),' | P(d<-k) (confident wrong):',' '.join(f'k{k}:{(Au.d<-k).mean():.3f}' for k in KS))
H=Au[Au.typ!='far']
print('   hard (roman/digit/close) n=%d P(d>k): '%len(H)+' '.join(f'k{k}:{(H.d>k).mean():.3f}' for k in KS)+' | P(d<-k): '+' '.join(f'k{k}:{(H.d<-k).mean():.3f}' for k in KS))
w=Au[Au.d<0].sort_values('d').head(12)
for _,r in w.iterrows(): print(f"     LOSE {r.ko:14s} truth={r.true_en:22s} sister={r.other_en:22s} d={r.d:+.2f}")
# ---- B
B=pd.read_csv(L+'/setB.csv'); B['d']=[lp(t,k)-lp(o,k) for t,k,o in zip(B.true_en,B.ko,B.other_en)]
for k in (2,3):
    b=B[B.k==k]
    print(f"B real truth vs ranker rank-{k} (v29v, {len(b)} rows): acc {(b.d>0).mean():.3f}  margin {q(b.d)}")
    print(f"   break-rate P(l_other - l_truth > kappa): "+' '.join(f'k{x}:{(b.d<-x).mean():.4f}({int((b.d<-x).sum())})' for x in KS))
b2=B[B.k==2]; w=b2[b2.d<0].drop_duplicates(['ko','other_en']).sort_values('d').head(12)
for _,r in w.iterrows(): print(f"     LOSE {r.ko:14s} truth={r.true_en:22s} rank2={r.other_en:22s} d={r.d:+.2f}")
# ---- D
D=pd.read_csv(L+'/setD.csv'); D['d']=[lp(t,k)-lp(o,k) for t,k,o in zip(D.true_en,D.ko,D.other_en)]
print(f"D sister bench (synthetic Hangul) v28i WRONG rows, truth vs wrong pred: n={len(D)} judge prefers truth {(D.d>0).mean():.3f}; P(d>k) "+' '.join(f'k{k}:{(D.d>k).mean():.2f}' for k in KS))
for t,g in D.groupby('type'): print(f"   {t:7s} n={len(g)} acc {(g.d>0).mean():.2f}")
# ---- C
C=pd.read_csv(L+'/setC.csv'); C['lp']=[lp(e,k) for e,k in zip(C.cand_en,C.ko)]
W=C.pivot_table(index='sample_id',columns='k',values=['lp','s'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]
ok=C.assign(ok=C.cand_id==C.true_id).pivot_table(index='sample_id',columns='k',values='ok',aggfunc='first'); ok.columns=[f'ok{c}' for c in ok.columns]
W=W.join(ok.fillna(False).astype(bool)); W['m']=W.s1-W.s2; W['g']=W.lp2-W.lp1
print(f"C domtest (synthetic Hangul, real v29v ranker) rows with m<6: {len(W)}  wrong {int((~W.ok1).sum())}  of which rank2-true {int((~W.ok1&W.ok2).sum())}  rank3-true {int((~W.ok1&W.ok3).sum())}")
print(f"   judge acc on rank1 vs rank2 where one is truth: {((W.g>0)==W.ok2)[W.ok1|W.ok2].mean():.3f} (n={int((W.ok1|W.ok2).sum())})")
print("   rule swap->rank2 if m<tau and l2-l1>kappa : cells = fixed/broken (touched rows)")
print('   tau\\kappa '+''.join(f'{k:>14}' for k in KS))
for t in (0.25,0.5,1,2,3,4,6):
    lo=W.m<t; cells=[]
    for k in KS:
        sw=lo&(W.g>k); cells.append(f"+{int((sw&W.ok2).sum())}/-{int((sw&W.ok1).sum())}({int(lo.sum())})")
    print(f'   tau={t:<5}'+''.join(f'{c:>14}' for c in cells))
# ---- E (scores for its pairs may live in a second file named like the first with pairsE -> sE)
E=pd.read_csv(L+'/setE.csv')
import os
for f in sys.argv[1:]:
    fe=f.replace('s17_','sE17_').replace('s4b_','sE4b_').replace('sens_','sEens_')
    if os.path.exists(fe):
        x=pd.read_csv(fe); LP.update({(e,k):v for e,k,v in zip(x.en,x.ko,x.lp)})
E['d1']=[lp(t,k)-lp(o,k) for t,k,o in zip(E.true_en,E.ko,E.other_en)]; E['d2']=[lp(t,k)-lp(o,k) for t,k,o in zip(E.true_en,E.ko,E.other2_en)]
E=E[~E.true_en.isin(__import__('judge').SHOT_NAMES)] if False else E
print(f"E real rows vs OOF-DE hardest gated negatives: n={E.d1.notna().sum()} acc(neg1) {(E.d1>0).mean():.3f} acc(neg2) {(E.d2>0).mean():.3f}  margin neg1 {q(E.d1.dropna())}")
print('   P(l_neg1 - l_truth > kappa): '+' '.join(f'k{k}:{(E.d1<-k).mean():.4f}({int((E.d1<-k).sum())})' for k in KS))
w=E[E.de_wrong]; print(f"   rows where OOF DE is WRONG (real-Hangul errors): n={len(w)} ({w.true_id.nunique()} vessels) judge picks truth {(w.d1>0).mean():.3f}; P(d>k): "+' '.join(f'k{k}:{(w.d1>k).mean():.2f}' for k in KS))
for _,r in w.drop_duplicates(['ko','other_en']).iterrows(): print(f"     {r.ko:12s} truth={r.true_en:20s} DE-top={r.other_en:20s} d={r.d1:+.2f}")
