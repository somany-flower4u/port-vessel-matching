"""domcurve.py SCORES_CSV [SCORES_E_CSV] : domtest intervention curve by row kind and with a sister guard (rank-1 and rank-2 catalog GT equal)."""
import os
import sys, pandas as pd, numpy as np
L=os.environ.get("SEOKCHEON_WS", ".") + '/agents/llmjudge'
S=pd.read_csv(sys.argv[1]); LP={(e,k):v for e,k,v in zip(S.en,S.ko,S.lp)}
C=pd.read_csv(L+'/setC.csv'); C['lp']=[LP[(e,k)] for e,k in zip(C.cand_en,C.ko)]
kind=pd.read_csv(os.environ.get("SEOKCHEON_WS", ".") + '/work/domtest/truth.csv').set_index('sample_id').kind
dm=pd.read_pickle(L+'/v29v_domtest.pkl').set_index('sample_id')
W=C.pivot_table(index='sample_id',columns='k',values=['lp','s'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]
W=W.join(dm[['ok1','ok2','gt1','gt2']]); W['m']=W.s1-W.s2; W['g']=W.lp2-W.lp1; W['kind']=kind
KS=[0,1,2,3,5,8]
def curve(X,title):
    print(f'-- {title}: rows m<6 {len(X)} wrong {int((~X.ok1).sum())} rank2-true {int((~X.ok1&X.ok2).sum())}; judge acc (one of top2 true) {((X.g>0)==X.ok2)[X.ok1|X.ok2].mean():.3f}')
    for t in (1,2,3,5):
        lo=X.m<t
        print(f'   tau={t:<3}'+''.join(f'{"k"+str(k)+":+"+str(int((lo&(X.g>k)&X.ok2).sum()))+"/-"+str(int((lo&(X.g>k)&X.ok1).sum())):>12}' for k in KS)+f'   (touched {int(lo.sum())})')
curve(W,'domtest all kinds')
curve(W[~W.kind.isin(['abbr','noise'])],'domtest clean/natgt/noflag/noton (no truncation/noise)')
curve(W[W.gt1==W.gt2],'domtest sister guard gt1==gt2')
curve(W[(W.gt1==W.gt2)&~W.kind.isin(['abbr','noise'])],'domtest sister guard + no abbr/noise')
