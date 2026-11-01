"""v14 rankers: out-of-fold candidate/feature pairs for train rows and their
perturbed copies, built through the submission's own Predictor (so the
candidate sets are exactly those of inference), then LightGBM lambdarank.

usage: train14.py SUBDIR PKG(ko|en) KINDS OUTNPZ   -> build pairs
       train14.py SUBDIR PKG fit PAIRS.npz[,more]  -> OOF report + final rankers into SUBDIR/model/PKG
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W))
import aug14  # noqa: E402

PARAMS = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "lambdarank_truncation_level": 10, "verbose": -1, "num_threads": 8}
KEEP_RANK = 25  # training rows per query: truth + candidates ranked < KEEP_RANK by any relative score


def load_script(sub: Path):
    spec = importlib.util.spec_from_file_location("sub14", sub / "script.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


def make_folds(train: pd.DataFrame, is_korean, n_folds: int = 5, seed: int = 2026) -> np.ndarray:
    vessels = sorted(train.target_vessel_id.unique())
    ko_vessels = set(train.loc[train.vessel_name.map(is_korean), "target_vessel_id"])
    rng = np.random.RandomState(seed)
    fold_of = {}
    for group in (sorted(ko_vessels), sorted(set(vessels) - ko_vessels)):
        group = list(group)
        rng.shuffle(group)
        for k, v in enumerate(group):
            fold_of[v] = k % n_folds
    return train.target_vessel_id.map(fold_of).to_numpy()


_KO_NAMES = None
TON_KINDS = ("digit", "swap", "drop", "x20", "dwt", "digit", "x20")
P_TON_ERR = float(__import__("os").environ.get("P_TON_ERR", "0"))


def perturb_row(kind: str, r: dict, rng) -> dict:
    """kind = NAMEKIND or NAMEKIND@ton (forced tonnage error)."""
    import tonfx
    name_kind, _, flag = kind.partition("@")
    row = dict(r)
    if name_kind == "ko_letter":
        row["vessel_name"] = aug14.ko_letter_init(r["vessel_name"], rng, r["target_vessel_id"])
    elif name_kind == "ko_rename":  # unrelated Hangul name of another train vessel; tonnage / flag / type unchanged
        global _KO_NAMES
        if _KO_NAMES is None:
            _tr = pd.read_csv(W / "train.csv", encoding="utf-8-sig")
            _KO_NAMES = sorted(set(n for n in _tr.vessel_name if isinstance(n, str) and any("가" <= c <= "힣" for c in n)))
        nm = _KO_NAMES[rng.randint(len(_KO_NAMES))]
        while nm == r["vessel_name"]:
            nm = _KO_NAMES[rng.randint(len(_KO_NAMES))]
        row["vessel_name"] = nm
    else:
        row["vessel_name"] = aug14.make(name_kind, r["vessel_name"], rng)
    if kind == "orig":
        return row
    if flag == "ton" or rng.rand() < P_TON_ERR:
        return tonfx.apply(TON_KINDS[rng.randint(len(TON_KINDS))], row, rng)
    if rng.rand() < 0.12:
        row["gross_tonnage"] = row["international_gross_tonnage"] = np.nan
    if rng.rand() < 0.10 and pd.notna(row["gross_tonnage"]):
        f = rng.uniform(0.95, 0.97) if rng.rand() < 0.5 else rng.uniform(1.03, 1.05)
        row["gross_tonnage"] = round(row["gross_tonnage"] * f)
        if pd.notna(row["international_gross_tonnage"]):
            row["international_gross_tonnage"] = round(row["international_gross_tonnage"] * f)
    if rng.rand() < 0.08:
        row["flag_name"] = row["vessel_type"] = np.nan
    return row


G = {}


def predictor(S, pkg, cat, tr):
    if pkg == "ko":
        a = S.Predictor("vm_ko", S.MODEL / "ko", cat, tr, exclude=S.KO_EXCLUDE, filters=S.KO_FILTERS,
                        exclude_mode=S.KO_EXCLUDE_MODE, alias=S.KO_ALIAS, eligible=S.KO_ELIGIBLE,
                        ton_gate=S.KO_TON_GATE, single_letters=S.KO_SINGLE_LETTERS)
        if getattr(S, "KO_ENSEMBLE", 0) > 0:
            sys.path.insert(0, str(S.MODEL / "ko2"))
            b = S.Predictor("vm_ko2", S.MODEL / "ko2", cat, tr, exclude=S.KO_EXCLUDE, filters=S.KO_FILTERS,
                            exclude_mode=S.KO_EXCLUDE_MODE, alias=S.KO_ALIAS, eligible=S.KO_ELIGIBLE,
                            ton_gate=S.KO_TON_GATE)
            ens = S.EnsemblePredictor(a, b, S.KO_ENSEMBLE)
            ens.mm = a.mm
            return ens
        return a
    return S.Predictor("vm_en", S.MODEL / "en", cat, tr, exclude=S.EN_EXCLUDE, filters=S.EN_FILTERS,
                       tiebreak=S.EN_TIEBREAK, ascii_fold=S.EN_ASCII_FOLD, exclude_mode=S.EN_EXCLUDE_MODE,
                       alias=S.EN_ALIAS, eligible=S.EN_ELIGIBLE, ton_gate=S.EN_TON_GATE)


def build_fold(k: int):
    S, pkg, kinds, cat, train, folds, lang = (G[x] for x in ("S", "pkg", "kinds", "cat", "train", "folds", "lang"))
    t0 = time.time()
    tr = train[folds != k]
    pred = predictor(S, pkg, cat, tr)
    mm = pred.mm
    rows = train[(folds == k) & lang].to_dict("records")
    Xs, ys, qids, kind_ids, srcs = [], [], [], [], []
    qn = 0
    for ki, kind in enumerate(kinds):
        rng = np.random.RandomState(1000 * k + ki)
        for r in rows:
            row = perturb_row(kind, r, rng)
            if (pkg == "ko") != bool(mm.T.is_korean(row["vessel_name"])):
                continue
            q = mm.Query(row, pred.ref)
            cand = pred.candidate_set(q)
            truth = pred.cat.id_to_idx[r["target_vessel_id"]]
            y = (cand == truth).astype(np.int8)
            qid = (k * 10_000_000) + qn
            qn += 1
            if len(cand) == 0:
                Xs.append(np.zeros((0, len(mm.FEATURES)), np.float32)); ys.append(y)
                qids.append(np.zeros(0, np.int64)); kind_ids.append(np.zeros(0, np.int8))
                srcs.append((qid, ki, 0))
                continue
            feats = pred.matcher.features(q, cand)
            X = mm.with_relative(feats, np.zeros(len(cand), np.int32))
            srcs.append((qid, ki, int(y.any())))
            rel = X[:, len(mm.ABS_FEATURES):]
            rank_cols = rel[:, 0:2 * len(mm.REL_SOURCES):2]
            keep = (np.nan_to_num(rank_cols, nan=1e9).min(axis=1) < KEEP_RANK) | (y == 1)
            Xs.append(X[keep].astype(np.float32)); ys.append(y[keep])
            qids.append(np.full(int(keep.sum()), qid, np.int64)); kind_ids.append(np.full(int(keep.sum()), ki, np.int8))
    print(f"fold {k}: {qn} queries, {time.time() - t0:.0f}s", flush=True)
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(qids), np.concatenate(kind_ids),
            np.array(srcs, dtype=np.int64))


def build(sub: Path, pkg: str, kinds: list[str], out: Path) -> None:
    S = load_script(sub)
    cat = pd.read_csv(W / "vessel_catalog.csv", encoding="utf-8-sig")
    train = pd.read_csv(W / "train.csv", encoding="utf-8-sig")
    T = importlib.import_module("vm_ko.text")
    folds = make_folds(train, T.is_korean)
    lang = train.vessel_name.map(T.is_korean).to_numpy()
    if pkg == "en":
        lang = ~lang
    G.update(S=S, pkg=pkg, kinds=kinds, cat=cat, train=train, folds=folds, lang=lang)
    with mp.get_context("fork").Pool(5) as pool:
        parts = pool.map(build_fold, range(5))
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    qid = np.concatenate([p[2] for p in parts]); kid = np.concatenate([p[3] for p in parts])
    src = np.vstack([p[4] for p in parts])
    np.savez_compressed(out, X=X, y=y, qid=qid, kid=kid, src=src, kinds=np.array(kinds))
    print("saved", out, X.shape, flush=True)


def groups(qid: np.ndarray) -> np.ndarray:
    change = np.flatnonzero(np.diff(qid)) + 1
    b = np.concatenate([[0], change, [len(qid)]])
    return np.diff(b)


def fit(sub: Path, pkg: str, pair_files: list[str], seeds: int = 3, rounds: int = 600, oof: bool = True) -> None:
    import lightgbm as lgb
    data = [np.load(f, allow_pickle=True) for f in pair_files]
    X = np.vstack([d["X"] for d in data]); y = np.concatenate([d["y"] for d in data])
    offs = [n * 100_000_000 for n in range(len(data))]
    qid = np.concatenate([d["qid"] + o for d, o in zip(data, offs)])
    kinds_all = [list(d["kinds"]) for d in data]
    kid = np.concatenate([d["kid"] for d in data])
    src = np.vstack([np.column_stack([d["src"][:, 0] + o, d["src"][:, 1], d["src"][:, 2]]) for d, o in zip(data, offs)])
    fold = (qid % 100_000_000) // 10_000_000
    sfold = (src[:, 0] % 100_000_000) // 10_000_000
    kind_names = kinds_all[0]
    if oof:
        res = {}
        for k in [int(x) for x in __import__('os').environ.get('OOF_FOLDS', '0,1,2,3,4').split(',')]:
            tr = fold != k
            ds = lgb.Dataset(X[tr], y[tr], group=groups(qid[tr]))
            b = lgb.train(dict(PARAMS, seed=0), ds, num_boost_round=rounds)
            te = fold == k
            s = b.predict(X[te])
            df = pd.DataFrame({"qid": qid[te], "y": y[te], "s": s, "kid": kid[te]})
            top = df.sort_values(["qid", "s"], ascending=[True, False]).groupby("qid").head(1)
            okq = set(top.qid[top.y == 1])
            for q_, ki, has in src[sfold == k]:
                name = kind_names[ki]
                a = res.setdefault(name, [0, 0, 0])
                a[0] += 1; a[1] += q_ not in okq; a[2] += has == 0
        for name, (n, e, miss) in res.items():
            print(f"OOF {pkg} {name:10s} n={n} err={e} ({e / n:.4f}) cand_miss={miss}", flush=True)
    ds = lgb.Dataset(X, y, group=groups(qid))
    mdir = sub / "model" / pkg
    names = []
    for seed in range(seeds):
        b = lgb.train(dict(PARAMS, seed=seed), ds, num_boost_round=rounds)
        name = f"lgb_seed{seed}.txt"
        b.save_model(str(mdir / name))
        names.append(name)
    man = json.loads((mdir / "manifest.json").read_text())
    man.update({"models": names, "pairs": [Path(f).name for f in pair_files], "rounds": rounds, "params": PARAMS})
    (mdir / "manifest.json").write_text(json.dumps(man, indent=1))
    print("final rankers ->", mdir, flush=True)


if __name__ == "__main__":
    sub, pkg, what = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    if what == "fit":
        fit(sub, pkg, sys.argv[4].split(","), oof=(len(sys.argv) < 6 or sys.argv[5] != "nooof"))
    else:
        build(sub, pkg, what.split(","), Path(sys.argv[4]))
