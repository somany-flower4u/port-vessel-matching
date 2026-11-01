"""Latin ranker fit = work/train14.py fit (same PARAMS / 600 rounds / seeds 0-2 / OOF by pair-building fold), but
memory-lean (one pair file, no vstack copy) and with a configurable thread count.
usage: fit_en.py PAIRS.npz MODE(oof|final|both) [SUBDIR] [threads=4]
OOF: 5 rankers (seed 0), each trained on 4 pair-building folds and scored on the 5th; per-kind top-1 errors
(a query without the truth among its candidates counts as an error) -> 'OOF en KIND n= err= cand_miss='."""
import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.environ.get("SEOKCHEON_WS", ".") + "/work")
import train14 as TR
pairs, mode = sys.argv[1], sys.argv[2]
sub = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
PARAMS = dict(TR.PARAMS, num_threads=threads)
if Path(pairs).is_dir():  # unpacked pairs (npz2dir): X memory-mapped, the rest in memory
    d = {k: np.load(Path(pairs) / f"{k}.npy", mmap_mode="r" if k == "X" else None, allow_pickle=True)
         for k in ("X", "y", "qid", "kid", "src", "kinds")}
else:
    d = np.load(pairs, allow_pickle=True)
X, y, qid, kid, src, kinds = d["X"], d["y"], d["qid"], d["kid"], d["src"], list(d["kinds"])
print(pairs, X.shape, kinds, flush=True)
fold = qid // 10_000_000
sfold = src[:, 0] // 10_000_000
t0 = time.time()
if mode in ("oof", "both"):
    res = {}
    import os
    for k in [int(x) for x in os.environ.get("OOF_FOLDS", "0,1,2,3,4").split(",")]:
        tr = np.flatnonzero(fold != k)
        ds = lgb.Dataset(X[tr], y[tr], group=TR.groups(qid[tr]), free_raw_data=True)
        b = lgb.train(dict(PARAMS, seed=0), ds, num_boost_round=600)
        del ds, tr; gc.collect()
        te = np.flatnonzero(fold == k)
        s = b.predict(X[te], num_threads=threads)
        df = pd.DataFrame({"qid": qid[te], "y": y[te], "s": s})
        top = df.sort_values(["qid", "s"], ascending=[True, False]).groupby("qid").head(1)
        okq = set(top.qid[top.y == 1])
        fk = {}
        for q_, ki, has in src[sfold == k]:
            for r_ in (res, fk):
                a = r_.setdefault(kinds[ki], [0, 0, 0]); a[0] += 1; a[1] += q_ not in okq; a[2] += has == 0
        print(f"fold {k} done {time.time() - t0:.0f}s | " + " ".join(f"{n}={e}/{c}" for n, (c, e, m) in fk.items()), flush=True)
    tot = [0, 0, 0]
    for name, (n, e, miss) in res.items():
        print(f"OOF en {name:12s} n={n} err={e} ({e / n:.4f}) cand_miss={miss}", flush=True)
        tot = [tot[0] + n, tot[1] + e, tot[2] + miss]
    print(f"OOF en {'TOTAL':12s} n={tot[0]} err={tot[1]} ({tot[1] / tot[0]:.4f}) cand_miss={tot[2]}", flush=True)
if mode in ("final", "both"):
    sys.path.insert(0, str(sub / "model" / "en"))
    import importlib
    mm = importlib.import_module("vm_en.matcher")
    assert X.shape[1] == len(mm.FEATURES), (X.shape, len(mm.FEATURES))
    ds = lgb.Dataset(X, y, group=TR.groups(qid), free_raw_data=False)
    mdir = sub / "model" / "en"; names = []
    for seed in range(3):
        b = lgb.train(dict(PARAMS, seed=seed), ds, num_boost_round=600)
        b.save_model(str(mdir / f"lgb_seed{seed}.txt")); names.append(f"lgb_seed{seed}.txt")
        print("seed", seed, f"{time.time() - t0:.0f}s", flush=True)
    man = json.loads((mdir / "manifest.json").read_text())
    man.update({"features": list(mm.FEATURES), "models": names, "pairs": [Path(pairs).name], "rounds": 600,
                "params": TR.PARAMS,
                "note": "v27f: v19 Latin recipe (P_TON_ERR=0.1; orig,en_noise,en_heavy,en_abbr,en_abbr,orig@ton,en_noise@ton; "
                        "NO_PRIORS) + Latin char dual encoder features (per-row OOF DE fold crc32 % 5)"})
    (mdir / "manifest.json").write_text(json.dumps(man, indent=1))
    print("final rankers ->", mdir, flush=True)
