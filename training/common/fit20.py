"""fit20.py SUB TRAINPAIRS.npz DOMPAIRS.npz [DOMW] [final]: domestic 5-fold CV (held-out by vessel) + train OOF, then final rankers."""
import sys, json, importlib, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
import train14 as TR
sub, trp, domp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
domw = int(sys.argv[4]) if len(sys.argv) > 4 else 1
do_final = len(sys.argv) > 5 and sys.argv[5] == "final"
sys.path.insert(0, str(sub / "model" / "ko")); mm = importlib.import_module("vm_ko.matcher")
PARAMS = TR.PARAMS
t = np.load(trp, allow_pickle=True); Xt, yt, qt, kidt, srct, tkinds = t["X"], t["y"], t["qid"], t["kid"], t["src"], list(t["kinds"])
d = np.load(domp, allow_pickle=True); Xd, yd, qd, meta, dkinds = d["X"], d["y"], d["qid"], d["meta"], list(d["kinds"])
assert Xt.shape[1] == Xd.shape[1] == len(mm.FEATURES), (Xt.shape, Xd.shape, len(mm.FEATURES))
fold_of = dict(zip(meta[:, 0], meta[:, 1])); fd = np.array([fold_of[q] for q in qd]); ft = qt // 10_000_000
kind_of = dict(zip(meta[:, 0], meta[:, 2])); kd_all = np.array([kind_of[q] for q in qd])
EXCL = [dkinds.index(k) for k in __import__("os").environ.get("EXCLUDE_KINDS", "").split(",") if k in dkinds]
trainable = ~np.isin(kd_all, EXCL)  # kinds evaluated but not trained on

def groups(q):
    ch = np.flatnonzero(np.diff(q)) + 1; return np.diff(np.concatenate([[0], ch, [len(q)]]))

def top1(X, y, q, boosters):
    s = np.mean([b.predict(X, num_threads=8) for b in boosters], axis=0)
    df = pd.DataFrame({"qid": q, "y": y, "s": s})
    top = df.sort_values(["qid", "s"], ascending=[True, False]).groupby("qid").head(1)
    return set(top.qid[top.y == 1])

def stack(Xa, ya, qa, Xb, yb, qb, w):
    X = np.vstack([Xa] + [Xb] * w); y = np.concatenate([ya] + [yb] * w)
    q = np.concatenate([qa] + [qb + 100_000_000 * (j + 1) for j in range(w)])
    return X, y, q

ok_dom, ok_tr = set(), set()
for k in range(5):
    m = (fd != k) & trainable
    X, y, q = stack(Xt[ft != k], yt[ft != k], qt[ft != k], Xd[m], yd[m], qd[m], domw)
    b = lgb.train(dict(PARAMS, seed=0), lgb.Dataset(X, y, group=groups(q)), num_boost_round=600)
    ok_dom |= top1(Xd[fd == k], yd[fd == k], qd[fd == k], [b])
    ok_tr |= top1(Xt[ft == k], yt[ft == k], qt[ft == k], [b])
    print("fold", k, flush=True)
out = []
for ki, kind in enumerate(dkinds):
    m = meta[meta[:, 2] == ki]; out.append(f"{kind}={sum(int(q not in ok_dom) for q in m[:, 0])}/{len(m)}")
tot = sum(int(q not in ok_dom) for q in meta[:, 0])
print(f"DOM held-out total {tot}/{len(meta)} ({tot/len(meta):.3f}) | " + " ".join(out), flush=True)
print("DOM cand miss:", {kind: int(((meta[:, 2] == ki) & (meta[:, 3] == 0)).sum()) for ki, kind in enumerate(dkinds)})
res = {}
for q_, ki, has in srct:
    a = res.setdefault(tkinds[ki], [0, 0, 0]); a[0] += 1; a[1] += q_ not in ok_tr; a[2] += has == 0
for name, (n, e, miss) in res.items():
    print(f"TRAIN OOF {name:10s} n={n} err={e} ({e/n:.4f}) cand_miss={miss}", flush=True)
if do_final:
    X, y, q = stack(Xt, yt, qt, Xd[trainable], yd[trainable], qd[trainable], domw)
    ds = lgb.Dataset(X, y, group=groups(q), free_raw_data=False)
    mdir = sub / "model" / "ko"; names = []
    for s in range(3):
        b = lgb.train(dict(PARAMS, seed=s), ds, num_boost_round=600)
        b.save_model(str(mdir / f"lgb_seed{s}.txt")); names.append(f"lgb_seed{s}.txt"); print("seed", s, flush=True)
    man = json.loads((mdir / "manifest.json").read_text())
    man.update({"features": list(mm.FEATURES), "models": names, "pairs": [Path(trp).name, Path(domp).name], "dom_weight": domw,
                "rounds": 600, "params": PARAMS})
    (mdir / "manifest.json").write_text(json.dumps(man, indent=1)); print("final rankers ->", mdir)
