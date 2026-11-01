"""Domtest set C (synthetic Hangul near-ties, real v29v ranker): fixed/broken counts per judge for
core windows and second bands, with and without the subset guard (rank-2 words strict subset of rank-1's)."""
import re, sys
import numpy as np, pandas as pd
L = sys.argv[1]
words = lambda n: set(re.findall(r"[a-z0-9]+", str(n).lower()))
C = pd.read_csv(f"{L}/setC.csv")
names = C.pivot_table(index="sample_id", columns="k", values="cand_en", aggfunc="first")
for judge in ("s17_ko10", "s4b_ko10", "s4b_ko0"):
    S = pd.read_csv(f"{L}/{judge}.csv")
    LP = {(e, k): v for e, k, v in zip(S.en, S.ko, S.lp)}
    c = C.assign(lp=[LP.get((e, k), np.nan) for e, k in zip(C.cand_en, C.ko)], ok=C.cand_id == C.true_id)
    W = c.pivot_table(index="sample_id", columns="k", values=["lp", "s"], aggfunc="first")
    W.columns = [f"{a}{b}" for a, b in W.columns]
    ok = c.pivot_table(index="sample_id", columns="k", values="ok", aggfunc="first")
    W["ok1"], W["ok2"] = ok[1].fillna(False).astype(bool), ok[2].fillna(False).astype(bool)
    W["m"], W["g"] = W.s1 - W.s2, W.lp2 - W.lp1
    W["sub"] = [words(names.loc[i, 2]) < words(names.loc[i, 1]) for i in W.index]
    print(f"== {judge}: rows m<6 {len(W)}, guard-eligible (rank-2 words subset) {int(W['sub'].sum())}")
    def cell(lo, hi, k, guard):
        sw = (W.m >= lo) & (W.m < hi) & (W.g > k) & (~W["sub"] if guard else True)
        return f"+{int((sw & W.ok2).sum())}/-{int((sw & W.ok1).sum())}"
    for lo, hi in ((0, 2), (0, 3), (2, 3), (3, 4), (3, 5), (3, 6), (4, 6)):
        print(f"   band [{lo},{hi})  " + "  ".join(f"k{k}: {cell(lo, hi, k, False)} | guard {cell(lo, hi, k, True)}" for k in (3, 5, 8, 12, 15)))
