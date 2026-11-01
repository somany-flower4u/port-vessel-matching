"""v20 Hangul ranker: released recipe pairs (ko18, priors neutralised) + domestic Hangul pairs.
usage: domtrain.py build SUB OUT.npz        (pairs of domgen rows through SUB's Hangul predictor)
       domtrain.py cv SUB DOM.npz           (5-fold by vessel: new ranker vs SUB's rankers on held-out domestic rows)
       domtrain.py final SUB DOM.npz OUTDIR (3 seeds on everything -> OUTDIR/lgb_seed*.txt)"""
from __future__ import annotations
import os
import sys, time, importlib, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import train14 as TR
W = Path(__file__).resolve().parent
KO18 = Path(os.environ.get("SEOKCHEON_WS", ".") + "/work/ko18")
PARAMS = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "lambdarank_truncation_level": 10, "verbose": -1, "num_threads": 8}
G = {}
KINDS = ["clean", "natgt", "igt", "noton", "noflag", "noise", "abbr", "nosp", "rename", "dwt"]


def build_part(part):
    S, cat, train = G["S"], G["cat"], G["train"]
    rows = G["rows"][part::5]
    pred = TR.predictor(S, "ko", cat, train); mm = pred.mm
    t0 = time.time(); Xs, ys, qids, meta = [], [], [], []
    for r in rows.to_dict("records"):
        q = mm.Query(r, pred.ref)
        cand = pred.candidate_set(q)
        truth = pred.cat.id_to_idx[r["target_vessel_id"]]
        y = (cand == truth).astype(np.int8)
        qid = int(r["sample_id"].split("_")[1])
        meta.append((qid, r["fold"], KINDS.index(r["kind"]), int(y.any()), len(cand)))
        if len(cand) == 0:
            continue
        feats = pred.matcher.features(q, cand)
        X = mm.with_relative(feats, np.zeros(len(cand), np.int32))
        rel = X[:, len(mm.ABS_FEATURES):]
        rank_cols = rel[:, 0:2 * len(mm.REL_SOURCES):2]
        keep = (np.nan_to_num(rank_cols, nan=1e9).min(axis=1) < TR.KEEP_RANK) | (y == 1)
        Xs.append(X[keep].astype(np.float32)); ys.append(y[keep]); qids.append(np.full(int(keep.sum()), qid, np.int64))
    print(f"part {part}: {len(rows)} queries, {time.time() - t0:.0f}s", flush=True)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(qids), np.array(meta, dtype=np.int64)


def build(sub: Path, out: Path):
    S = TR.load_script(sub)
    cat = pd.read_csv(W / "vessel_catalog.csv"); train = pd.read_csv(W / "train.csv")
    import domgen
    rows = domgen.make_rows(train, cat, reps=int(__import__('os').environ.get('DOM_REPS', '1')))
    G.update(S=S, cat=cat, train=train, rows=rows)
    with mp.get_context("fork").Pool(5) as pool:
        parts = pool.map(build_part, range(5))
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); qid = np.concatenate([p[2] for p in parts])
    meta = np.vstack([p[3] for p in parts])
    order = np.argsort(qid, kind="stable"); X, y, qid = X[order], y[order], qid[order]
    np.savez_compressed(out, X=X, y=y, qid=qid, meta=meta, kinds=np.array(KINDS))
    rows.to_csv(out.with_suffix(".rows.csv"), index=False)
    print("saved", out, X.shape, "cand_miss", int((meta[:, 3] == 0).sum()), "of", len(meta), flush=True)


def neutralise(X, A):
    col = A.index
    X = X.astype(np.float64, copy=True)
    tons = X[:, col("q_tons")] > 0
    gen = X[:, col("name_llk")] + X[:, col("type_logp")] + np.nan_to_num(X[:, col("flag_logp")], nan=0.0)
    gen = gen + np.where(tons, np.nan_to_num(X[:, col("ton_llk")], nan=-20.0), 0.0)
    X[:, col("gen_score")] = gen
    for c in ("type_rate", "size_logp", "c_log_gt"):
        X[:, col(c)] = np.nan
    return X.astype(np.float32)


def load_ko18(mm):
    A = list(mm.ABS_FEATURES); blocks = []
    for n, f in enumerate(["pairs_v8.npz", "aug_ko_v1.npz", "aug_ko_noise.npz", "aug_ko_noiseh.npz"]):
        d = np.load(KO18 / f, allow_pickle=True)
        assert list(d["features"]) == A, f
        keep = ~d["isref"]
        X = neutralise(d["X"][keep], A) if mm.NO_PRIORS else d["X"][keep]
        qid = d["qid"][keep] + 10_000_000 * (n + 1)
        blocks.append((mm.with_relative(X, qid), d["y"][keep], qid))
    return np.vstack([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]), np.concatenate([b[2] for b in blocks])


def groups(qid):
    ch = np.flatnonzero(np.diff(qid)) + 1
    return np.diff(np.concatenate([[0], ch, [len(qid)]]))


def top1(X, y, qid, booster_or_list):
    s = np.mean([b.predict(X, num_threads=8) for b in (booster_or_list if isinstance(booster_or_list, list) else [booster_or_list])], axis=0)
    df = pd.DataFrame({"qid": qid, "y": y, "s": s})
    top = df.sort_values(["qid", "s"], ascending=[True, False]).groupby("qid").head(1)
    return set(top.qid[top.y == 1])


def report(label, meta, okq, kinds):
    out = []
    for ki, kind in enumerate(kinds):
        m = meta[meta[:, 2] == ki]
        err = sum(int(q not in okq) for q in m[:, 0])
        out.append(f"{kind}={err}/{len(m)}")
    tot = sum(int(q not in okq) for q in meta[:, 0])
    print(f"{label:14s} total {tot}/{len(meta)} ({tot/len(meta):.3f}) | " + " ".join(out), flush=True)


def cv(sub: Path, dom: Path, dom_weight: int = 1):
    S = TR.load_script(sub); mm = importlib.import_module("vm_ko.matcher")
    d = np.load(dom, allow_pickle=True); Xd, yd, qd, meta, kinds = d["X"], d["y"], d["qid"], d["meta"], list(d["kinds"])
    fold_of = dict(zip(meta[:, 0], meta[:, 1])); fd = np.array([fold_of[q] for q in qd])
    X18, y18, q18 = load_ko18(mm)
    old = [lgb.Booster(model_file=str(sub / "model" / "ko" / f)) for f in ["lgb_seed0.txt", "lgb_seed1.txt", "lgb_seed2.txt"]]
    ok_old, ok_new = set(), set()
    for k in range(5):
        tr = fd != k
        Xt = np.vstack([X18] + [Xd[tr]] * dom_weight); yt = np.concatenate([y18] + [yd[tr]] * dom_weight)
        qt = np.concatenate([q18] + [qd[tr] + 100_000_000 * (j + 1) for j in range(dom_weight)])
        ds = lgb.Dataset(Xt, yt, group=groups(qt))
        b = lgb.train(dict(PARAMS, seed=0), ds, num_boost_round=600)
        te = fd == k
        ok_new |= top1(Xd[te], yd[te], qd[te], b)
        ok_old |= top1(Xd[te], yd[te], qd[te], old)
        print("fold", k, "done", flush=True)
    report(f"{sub.name} rankers", meta, ok_old, kinds)
    report("v20 (held-out)", meta, ok_new, kinds)
    print("candidate misses per kind:", {kind: int(((meta[:, 2] == ki) & (meta[:, 3] == 0)).sum()) for ki, kind in enumerate(kinds)})


def final(sub: Path, dom: Path, outdir: Path, dom_weight: int = 1):
    S = TR.load_script(sub); mm = importlib.import_module("vm_ko.matcher")
    d = np.load(dom, allow_pickle=True); Xd, yd, qd = d["X"], d["y"], d["qid"]
    X18, y18, q18 = load_ko18(mm)
    Xt = np.vstack([X18] + [Xd] * dom_weight); yt = np.concatenate([y18] + [yd] * dom_weight)
    qt = np.concatenate([q18] + [qd + 100_000_000 * (j + 1) for j in range(dom_weight)])
    ds = lgb.Dataset(Xt, yt, group=groups(qt), free_raw_data=False)
    outdir.mkdir(parents=True, exist_ok=True)
    for s in range(3):
        b = lgb.train(dict(PARAMS, seed=s), ds, num_boost_round=600)
        b.save_model(str(outdir / f"lgb_seed{s}.txt")); print("seed", s, "done", flush=True)


if __name__ == "__main__":
    what, sub = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, str(sub / "model" / "ko"))
    import domgen  # kogen imports vm_ko.text: must come from SUB's package
    assert Path(importlib.import_module("vm_ko").__file__).resolve().parent.parent == (sub / "model" / "ko").resolve(), importlib.import_module("vm_ko").__file__
    if what == "build":
        build(sub, Path(sys.argv[3]))
    elif what == "cv":
        cv(sub, Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        final(sub, Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1)
