"""de-v2-full track: the scorer track's common.py (load / catalog_meta / fold_of / tie-aware recall) re-pointed at
the current work dir; vm_ko.text comes from the v26b package (identical to v21f's text.py)."""
from __future__ import annotations
import os
import sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd

SP = Path(os.environ.get("SEOKCHEON_WS", ".") + "")
WORK = SP / "work"
PKG = WORK / "v26b" / "model" / "ko"
TRACK = SP / "agents" / "de-v2-full"
for p in (str(WORK), str(PKG)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(PKG))
from vm_ko import text as T  # noqa: E402

NFOLD = 5


def fold_of(vid) -> int:
    return zlib.crc32(str(vid).encode()) % NFOLD


def load():
    return pd.read_csv(WORK / "train.csv"), pd.read_csv(WORK / "vessel_catalog.csv")


def catalog_meta(cat):
    names = cat.vessel_name.fillna("").astype(str).tolist()
    comp = np.array(["".join(T.words(n)) for n in names], dtype=object)
    counts = pd.Series(comp).value_counts()
    dup = pd.Series(comp).map(counts).to_numpy()
    gt = pd.to_numeric(cat.gross_tonnage, errors="coerce").to_numpy(float)
    eligible = (dup == 1) & np.isfinite(gt) & (gt > 0)
    codes = pd.factorize(comp)[0].astype(np.int64)
    id2idx = {v: i for i, v in enumerate(cat.vessel_id.astype(str))}
    return dict(names=names, compact=comp, eligible=eligible, codes=codes, id2idx=id2idx, gt=gt)


def tie_stats(scores, t, codes):
    s = np.nan_to_num(scores.astype(np.float64), nan=-1e18)
    st = s[t]
    other = codes != codes[t]
    return int(((s > st) & other).sum()), int(((s == st) & other).sum())


def recall_at(g, e, k):
    return float(np.clip((k - g) / (e + 1.0), 0.0, 1.0))
