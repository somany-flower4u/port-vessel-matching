"""export_de.py WEIGHTS.npz CAT.npy PKGDIR: ship the full-data Latin DE into PKG/model/en/de (de_weights.npz + de_cat.npz
{emb float16, ids}) and check that the numpy catalog tower reproduces the torch catalog embeddings."""
import os
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
W = os.environ.get("SEOKCHEON_WS", ".") + "/work"
wts, cemb, pkg = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(pkg / "model" / "en"))
from vm_en.dualenc import DualEncoder
cat = pd.read_csv(W + "/vessel_catalog.csv")
ids = cat.vessel_id.astype(str).to_numpy(); names = cat.vessel_name.fillna("").astype(str).tolist()
emb = np.load(cemb)
assert emb.shape == (len(cat), 128) and emb.dtype == np.float16, (emb.shape, emb.dtype)
out = pkg / "model" / "en" / "de"; out.mkdir(parents=True, exist_ok=True)
shutil.copy(wts, out / "de_weights.npz")
np.savez_compressed(out / "de_cat.npz", emb=emb, ids=ids.astype("<U12"))
de = DualEncoder(out / "de_weights.npz", out / "de_cat.npz", ids, names)
rng = np.random.RandomState(0); idx = rng.choice(len(cat), 500, replace=False)
diff = max(float(np.abs(de.l(names[i]) - de.C[i]).max()) for i in idx)
print("numpy l-tower vs torch catalog embedding: max |diff| over 500 rows", round(diff, 5))
print("shipped", sorted(p.name for p in out.iterdir()), [round(p.stat().st_size / 1e6, 2) for p in sorted(out.iterdir())])
