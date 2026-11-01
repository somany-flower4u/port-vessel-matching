"""Robustness check: perturb held-out queries, rebuild their candidates with
the fold reference and score them with fold models trained on clean pairs."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd







LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _char_edit(s: str, rng) -> str:
    """One random edit anywhere in a name: delete / insert / substitute /
    double / swap a letter (first and last letters included)."""
    pos = [i for i, ch in enumerate(s) if ch.isalpha()]
    if not pos:
        return s
    i = pos[rng.randint(len(pos))]
    op = rng.randint(5)
    if op == 0 and len(pos) > 3:
        return s[:i] + s[i + 1:]
    if op == 1:
        return s[:i] + LETTERS[rng.randint(26)] + s[i:]
    if op == 2:
        return s[:i] + LETTERS[rng.randint(26)] + s[i + 1:]
    if op == 3:
        return s[: i + 1] + s[i] + s[i + 1:]
    if i + 1 < len(s) and s[i + 1].isalpha() and s[i] != s[i + 1]:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def en_noise(name: str, rng) -> str:
    """Broad name noise for robustness (10/19 test names are not explained by
    the train typo model): 1-4 character edits anywhere, sometimes a word
    dropped / swapped / merged / split, or the name cut short."""
    s = str(name).upper()
    ws = s.split()
    r = rng.rand()
    if r < 0.08 and len(ws) >= 2:
        del ws[rng.randint(len(ws))]
        s = " ".join(ws)
    elif r < 0.14 and len(ws) >= 2:
        k = rng.randint(len(ws) - 1)
        ws[k], ws[k + 1] = ws[k + 1], ws[k]
        s = " ".join(ws)
    elif r < 0.22 and len(ws) >= 2:
        k = rng.randint(len(ws) - 1)
        s = " ".join(ws[:k] + [ws[k] + ws[k + 1]] + ws[k + 2:])
    elif r < 0.28:
        long = [k for k, w in enumerate(ws) if len(w) >= 5]
        if long:
            k = long[rng.randint(len(long))]
            j = 2 + rng.randint(len(ws[k]) - 3)
            s = " ".join(ws[:k] + [ws[k][:j], ws[k][j:]] + ws[k + 1:])
    elif r < 0.32 and len(s) > 8:
        s = s[: max(6, int(len(s) * rng.uniform(0.6, 0.9)))].rstrip()
    n_edits = rng.choice([0, 1, 2, 3, 4], p=[0.1, 0.35, 0.3, 0.17, 0.08])
    for _ in range(n_edits):
        s = _char_edit(s, rng)
    return s


_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_VOWEL_SWAP = {"ㅓ": "ㅗ", "ㅗ": "ㅓ", "ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅡ": "ㅜ", "ㅜ": "ㅡ", "ㅕ": "ㅛ", "ㅛ": "ㅕ",
               "ㅏ": "ㅓ", "ㅣ": "ㅢ", "ㅢ": "ㅣ", "ㅚ": "ㅙ", "ㅙ": "ㅚ"}


def _ko_edit(s: str, rng) -> str:
    chars = list(s)
    syl = [i for i, ch in enumerate(chars) if "가" <= ch <= "힣"]
    if not syl:
        return s
    op = rng.randint(6)
    i = syl[rng.randint(len(syl))]
    if op == 0 and len(syl) >= 3:
        del chars[i]
    elif op == 1:
        chars.insert(i, chars[i])
    elif op == 2 and len(syl) >= 2:
        k = rng.randint(len(syl) - 1)
        a, b = syl[k], syl[k + 1]
        if b == a + 1:
            chars[a], chars[b] = chars[b], chars[a]
    elif op == 3:
        code = ord(chars[i]) - 0xAC00
        cho, jung, jong = code // 588, (code % 588) // 28, code % 28
        v = _JUNG[jung]
        if v in _VOWEL_SWAP:
            chars[i] = chr(0xAC00 + cho * 588 + _JUNG.index(_VOWEL_SWAP[v]) * 28 + jong)
    elif op == 4:
        code = ord(chars[i]) - 0xAC00
        cho, jung, jong = code // 588, (code % 588) // 28, code % 28
        jong = 0 if jong else [4, 8, 16, 21][rng.randint(4)]  # drop / add a final consonant (ㄴㄹㅁㅇ)
        chars[i] = chr(0xAC00 + cho * 588 + jung * 28 + jong)
    else:
        if " " in chars and rng.rand() < 0.5:
            chars.remove(" ")
        elif len(syl) >= 3:
            chars.insert(syl[1 + rng.randint(len(syl) - 2)], " ")
    return "".join(chars)


def ko_noise(name: str, rng) -> str:
    """Broad Hangul name noise: 1-3 syllable / jamo / spacing edits, sometimes a word dropped."""
    s = str(name)
    ws = s.split()
    if rng.rand() < 0.08 and len(ws) >= 2:
        del ws[rng.randint(len(ws))]
        s = " ".join(ws)
    for _ in range(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.2])):
        s = _ko_edit(s, rng)
    return s


def en_noise_heavy(name: str, rng) -> str:
    """Heavier non-Hangul noise: the en_noise word-level changes plus 3-6 character edits."""
    s = en_noise(name, rng)
    for _ in range(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2])):
        s = _char_edit(s, rng)
    return s


def ko_noise_heavy(name: str, rng) -> str:
    """Heavier Hangul noise: 2-5 edits, a word dropped more often."""
    s = str(name)
    ws = s.split()
    if rng.rand() < 0.12 and len(ws) >= 2:
        del ws[rng.randint(len(ws))]
        s = " ".join(ws)
    for _ in range(rng.choice([2, 3, 4, 5], p=[0.3, 0.35, 0.25, 0.1])):
        s = _ko_edit(s, rng)
    return s


def typo(word: str, rng) -> str:
    n = len(word)
    op = rng.randint(3)
    if op == 0:
        i = rng.randint(1, n - 1)
        return word[:i] + word[i + 1:]
    if op == 1:
        i = rng.randint(n)
        return word[: i + 1] + word[i] + word[i + 1:]
    i = rng.randint(1, n - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def perturb(train: pd.DataFrame, kind: str, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = train.copy()
    ko = df.vessel_name.map(T.is_korean)
    if kind == "no_tons":
        df["gross_tonnage"] = np.nan
        df["international_gross_tonnage"] = np.nan
    elif kind == "no_flag":
        df["flag_name"] = np.nan
    elif kind == "no_type":
        df["vessel_type"] = np.nan
    elif kind == "no_flag_type":
        df["flag_name"] = np.nan
        df["vessel_type"] = np.nan
    elif kind == "band_tons":
        f = np.where(rng.rand(len(df)) < 0.5, rng.uniform(0.95, 0.97, len(df)), rng.uniform(1.03, 1.05, len(df)))
        df["gross_tonnage"] = np.round(df.gross_tonnage * f)
        df["international_gross_tonnage"] = np.round(df.international_gross_tonnage * f)
    elif kind == "extra_typo":
        names = []
        for name, k in zip(df.vessel_name, ko):
            ws = name.upper().split()
            elig = [i for i, w in enumerate(ws) if len(w) >= 6 and w.isalpha()]
            if k or not elig:
                names.append(name)
                continue
            i = elig[rng.randint(len(elig))]
            ws[i] = typo(ws[i], rng)
            names.append(" ".join(ws))
        df["vessel_name"] = names
    elif kind == "ko_nospace":
        df["vessel_name"] = [n.replace(" ", "") if k else n for n, k in zip(df.vessel_name, ko)]
    elif kind == "no_tons_flag":
        df["gross_tonnage"] = np.nan
        df["international_gross_tonnage"] = np.nan
        df["flag_name"] = np.nan
    elif kind == "big_noise":
        # 10/19 test: tonnage far outside the train range (|log ratio| 0.14 .. 0.47)
        f = np.where(rng.rand(len(df)) < 0.5, rng.uniform(0.6, 0.85, len(df)), rng.uniform(1.15, 1.6, len(df)))
        df["gross_tonnage"] = np.round(df.gross_tonnage * f)
        df["international_gross_tonnage"] = np.round(df.international_gross_tonnage * f)
    elif kind == "clean":
        pass
    elif kind == "en_noise":
        df["vessel_name"] = [n if k else en_noise(n, rng) for n, k in zip(df.vessel_name, ko)]
    elif kind == "ko_noise":
        df["vessel_name"] = [ko_noise(n, rng) if k else n for n, k in zip(df.vessel_name, ko)]
    elif kind == "en_noise_heavy":
        df["vessel_name"] = [n if k else en_noise_heavy(n, rng) for n, k in zip(df.vessel_name, ko)]
    elif kind == "ko_noise_heavy":
        df["vessel_name"] = [ko_noise_heavy(n, rng) if k else n for n, k in zip(df.vessel_name, ko)]
    else:
        raise ValueError(kind)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs_v6.npz")
    ap.add_argument("--kinds", default="no_tons,no_flag,no_type,band_tons,extra_typo,ko_nospace")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--train-set", default="excl")
    args = ap.parse_args()

    d = load(args.pairs)
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    catdf = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    catalog = Catalog(catdf)
    ids = catdf.vessel_id.to_numpy()
    folds = d["folds"]
    cols = list(range(len(FEATURES)))
    row_fold = folds[d["qid"]]
    keep = ~d["isref"] if args.train_set == "excl" else np.ones(len(d["y"]), bool)
    models = {}
    for k in np.unique(folds):
        X, y, qid, _ = view(d, keep & (row_fold != k), cols)
        models[k] = [fit(X, y, qid, PARAMS, 600, s) for s in range(2)]

    rng = np.random.RandomState(0)
    ko_rows = np.flatnonzero(train.vessel_name.map(T.is_korean))
    en_rows = np.flatnonzero(~train.vessel_name.map(T.is_korean))
    rows = np.sort(np.concatenate([ko_rows, rng.choice(en_rows, max(args.sample - len(ko_rows), 0), replace=False)]))
    sub = train.iloc[rows]
    report = {}
    for kind in args.kinds.split(","):
        pert = perturb(sub, kind, seed=1)
        # rebuild with references from the other folds (folds of the full train)
        res = build_fold_subset(train, pert, catalog, folds, rows)
        dd = {"X": res["X"], "y": res["y"], "qid": res["qid"], "cidx": res["cidx"], "isref": res["isref"]}
        ex = ~dd["isref"]
        scores = np.zeros(len(dd["y"]))
        rf = folds[dd["qid"]]
        for k in np.unique(folds):
            m = ex & (rf == k)
            if not m.any():
                continue
            Xe, _, _, _ = view(dd, m, cols)
            scores[m] = np.mean([mdl.predict(Xe, num_threads=8) for mdl in models[k]], axis=0)
        best = top1(scores[ex], dd["qid"][ex], dd["cidx"][ex])
        r, _ = evaluate(pert, best, ids)
        r["missed"] = len(res["missed"])
        report[kind] = r
        print(kind, json.dumps(r), flush=True)
    (ART / "stress.json").write_text(json.dumps(report, indent=1))


def build_fold_subset(full_train, pert, catalog, folds, rows):
    """Like build_pairs.build but references come from the full (clean) train
    minus the query's fold, and queries are the perturbed rows."""
    from vm.matcher import Matcher, Query, Reference
    X, y, qid, cidx, isref, missed = [], [], [], [], [], []
    row_folds = folds[rows]
    for k in np.unique(row_folds):
        ref = Reference(full_train[folds != k], catalog)
        matcher = Matcher(catalog, ref)
        part = pert[row_folds == k]
        for row_id, row in zip(part.index, part.to_dict("records")):
            q = Query(row, ref)
            cand, is_ref = matcher.candidates(q)
            feats = matcher.features(q, cand)
            target = catalog.id_to_idx[row["target_vessel_id"]]
            lab = (cand == target).astype(np.int8)
            if not lab.any():
                missed.append(int(row_id))
            X.append(feats)
            y.append(lab)
            qid.append(np.full(len(cand), row_id, dtype=np.int32))
            cidx.append(cand.astype(np.int32))
            isref.append(is_ref)
    return dict(X=np.concatenate(X), y=np.concatenate(y), qid=np.concatenate(qid),
                cidx=np.concatenate(cidx), isref=np.concatenate(isref), missed=missed)


if __name__ == "__main__":
    main()
