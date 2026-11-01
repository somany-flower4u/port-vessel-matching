"""Catalog index, train-derived reference statistics, candidate retrieval and
pair features.  A query row is always processed on its own: nothing here looks
at other query rows.

Latin port names are compared with the *raw* upper-case catalog words (the
generator copies the catalog spelling and only adds typos); Hangul names are
compared with a normalised reading in which 'No. 7' -> '7' and trailing Roman
numerals -> digits.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import OSA, JaroWinkler, Levenshtein

from . import text as T
from .translit import MAXLEN, Transliterator, encode, ko_units, substring_ids
from .abbr import ABBR_FEATURES, align, span_features, span_keys
from .dualenc import DualEncoder

LOG_BAND = (math.log(0.95), math.log(0.97), math.log(1.03), math.log(1.05))
BAND_TOL = 0.0012  # rounding slack at the band edges (in log space)
TON_GATE = 0.2  # |log(port GT / catalog GT)|; train maximum for the true vessel is 0.10
NO_PRIORS = True  # v19: no type_rate / size_logp / c_log_gt (target-distribution priors)
# v27f: Latin character dual encoder (Latin query tower x Latin catalog tower) as a candidate source and ranker
# features, mirroring the Hangul one in vm_ko (v26b).
DE_DIR = Path(__file__).resolve().parent.parent / "de"  # model/en/de: de_weights.npz + de_cat.npz (full-data model)
DE_FOLD_ENV = "VM_EN_DE_FOLD"          # out-of-fold pair building only: per-fold model K (never trained on that vessel)
DE_FOLD_DIR_ENV = "VM_EN_DE_FOLD_DIR"  # ... whose de_f{K}_weights.npz / de_f{K}_cat.npy live here (not shipped)
DE_K_GLOBAL, DE_K_WINDOW = 60, 30      # fused (name-ratio z + dual-encoder z) top-K over the catalog / the GT window
_DE_MODELS: dict = {}


def de_fold() -> str:
    """'' -> the shipped full-data model; 'K' -> per-fold model K (set by the out-of-fold pair builder)."""
    return os.environ.get(DE_FOLD_ENV, "").strip()


def zscore(s: np.ndarray) -> np.ndarray:
    """z over the whole catalog for one query (NaN -> -10)."""
    s = s.astype(np.float64)
    return np.nan_to_num((s - np.nanmean(s)) / (np.nanstd(s) + 1e-9), nan=-10.0)


def num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def clean_str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def letter_units(text: str) -> list[str]:
    """Transducer units without digits (digits are matched by separate features)."""
    return [u for u in ko_units(text) if u[0] != "D"]


def korean_variants(name: str) -> list[str]:
    out = T.ko_variants(name)
    out += [a for v in out for a in T.ko_acronym_variants(v)]
    return list(dict.fromkeys(out))


def alias_key(name) -> str:
    return "".join(T.words(name))


def eligible_letters(ws: list[str]) -> int:
    """Letters inside words long enough for the synthetic typo generator."""
    return sum(len(w) for w in ws if len(w) >= 6 and not w.isdigit())


def in_band(x):
    a, b, c, d = LOG_BAND
    return ((x >= a - BAND_TOL) & (x <= b + BAND_TOL)) | ((x >= c - BAND_TOL) & (x <= d + BAND_TOL))


def ton_status(r) -> np.ndarray:
    """0 exact, 1 band, 2 other deviation, 3 missing (r = best log ratio)."""
    r = np.asarray(r, dtype=float)
    out = np.full(r.shape, 2, dtype=np.int64)
    out[~np.isfinite(r)] = 3
    fin = np.isfinite(r)
    out[fin & (r == 0)] = 0
    out[fin & (r != 0) & in_band(np.where(fin, r, 0.0))] = 1
    return out


# ---------------------------------------------------------------------------
class Catalog:
    """Static lookup structures over the public vessel catalog."""

    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        if not df.vessel_id.is_unique:
            raise ValueError("catalog vessel_id must be unique")
        self.n = len(df)
        self.ids = df.vessel_id.astype(str).to_numpy()
        self.id_to_idx = {v: i for i, v in enumerate(self.ids)}
        self.names = df.vessel_name.fillna("").astype(str).tolist()
        # raw words: Latin port names are matched against these
        self.rwords = [T.words(n) for n in self.names]
        self.rcompact = ["".join(w) for w in self.rwords]
        self.rkey = [T.en_key(" ".join(w)) for w in self.rwords]
        self.rcons = [T.consonants(k) for k in self.rkey]
        self.rdigits = [tuple(re.findall(r"\d+", " ".join(w))) for w in self.rwords]
        self.elig = np.array([eligible_letters(w) for w in self.rwords], dtype=np.int32)
        # normalised words: Hangul port names are matched against these
        self.words = [T.drop_number_marker(w) for w in self.rwords]
        self.compact = ["".join(w) for w in self.words]
        self.key = [T.en_key(" ".join(w)) for w in self.words]
        self.cons = [T.consonants(k) for k in self.key]
        self.digits = [tuple(re.findall(r"\d+", " ".join(w))) for w in self.words]
        self.gt = pd.to_numeric(df.gross_tonnage, errors="coerce").to_numpy(float)
        self.type = df.vessel_type.fillna("").astype(str).to_numpy()
        self.flag = df.flag_name.fillna("").astype(str).to_numpy()
        # The organisers kept only vessels whose upper-case alphanumeric name is
        # unique in the catalog (no train target has a shared name), and every
        # target has a catalog gross tonnage.
        counts = Counter(self.rcompact)
        self.dup = np.array([counts[c] for c in self.rcompact], dtype=np.int32)
        self.by_compact: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.compact):
            self.by_compact[c].append(i)
        self.word_post: dict[str, list[int]] = defaultdict(list)
        for i, ws in enumerate(self.rwords):
            for w in set(ws):
                self.word_post[w].append(i)
        self.word_df = {w: len(v) for w, v in self.word_post.items()}
        self.by_gt: dict[float, list[int]] = defaultdict(list)
        for i, g in enumerate(self.gt):
            if np.isfinite(g):
                self.by_gt[float(g)].append(i)
        valid = np.flatnonzero(np.isfinite(self.gt) & (self.gt > 0))
        order = np.argsort(self.gt[valid], kind="stable")
        self.gt_sorted_idx = valid[order]
        self.gt_sorted = self.gt[self.gt_sorted_idx]
        safe = np.where(np.isfinite(self.gt) & (self.gt > 0), self.gt, 1.0)
        self.log_gt = np.where(np.isfinite(self.gt) & (self.gt > 0), np.log(safe), np.nan)
        self.type_count = Counter(self.type)
        self.eligible = (self.dup == 1) & np.isfinite(self.gt) & (self.gt > 0)
        self.chars = np.zeros((self.n, MAXLEN), dtype=np.int64)
        self.lens = np.zeros(self.n, dtype=np.int64)
        for i, c in enumerate(self.compact):
            e = encode(re.sub(r"\d", "", c))
            self.chars[i, :len(e)] = e
            self.lens[i] = len(e)

    def gt_window(self, value: float, lo: float, hi: float) -> np.ndarray:
        a = np.searchsorted(self.gt_sorted, value * lo, "left")
        b = np.searchsorted(self.gt_sorted, value * hi, "right")
        return self.gt_sorted_idx[a:b]

    def de(self) -> DualEncoder:
        """The Latin dual encoder of the current DE fold, loaded lazily (cached per process and fold)."""
        fold = de_fold()
        key = (fold, self.n, self.ids[0], self.ids[-1])
        m = _DE_MODELS.get(key)
        if m is None:
            if fold:
                d = Path(os.environ.get(DE_FOLD_DIR_ENV, str(DE_DIR)))
                m = DualEncoder(d / f"de_f{fold}_weights.npz", d / f"de_f{fold}_cat.npy", self.ids, self.names)
            else:
                m = DualEncoder(DE_DIR / "de_weights.npz", DE_DIR / "de_cat.npz", self.ids, self.names)
            _DE_MODELS[key] = m
        return m


# ---------------------------------------------------------------------------
class TonnageModel:
    """P(port tonnage | catalog GT of the true vessel): an exact-match point
    mass plus a histogram density of log ratios (both learned from train)."""

    BINS = np.linspace(-0.12, 0.12, 61)

    def __init__(self, ratios: np.ndarray):
        r = np.asarray(ratios, dtype=float)
        exact = r == 0
        self.p_exact = (exact.sum() + 1.0) / (len(r) + 2.0)
        h = np.histogram(r[~exact], bins=self.BINS)[0].astype(float) + 0.2
        width = self.BINS[1] - self.BINS[0]
        self.density = h / (h.sum() * width)
        self.floor = 0.01

    def llk(self, ratio: np.ndarray, tonnage: float) -> np.ndarray:
        out = np.full(ratio.shape, np.nan)
        ok = np.isfinite(ratio)
        ex = ok & (ratio == 0)
        out[ex] = math.log(self.p_exact)
        ne = ok & (ratio != 0)
        pos = np.searchsorted(self.BINS, ratio[ne]) - 1
        inside = (pos >= 0) & (pos < len(self.density))
        dens = np.where(inside, self.density[np.clip(pos, 0, len(self.density) - 1)], self.floor)
        # probability of this particular integer tonnage ~ density * (1 / tonnage)
        out[ne] = math.log(1.0 - self.p_exact) + np.log(dens) - math.log(tonnage)
        return out


class TypoModel:
    """P(number of legal typo operations) given whether the catalog name has
    words long enough to be typo-ed (learned from train)."""

    def __init__(self, ops: np.ndarray, elig: np.ndarray):
        ops = np.asarray(ops)
        elig = np.asarray(elig)
        self.p = {}
        for has in (False, True):
            sel = ops[(elig > 0) == has]
            counts = {k: float((sel == k).sum()) for k in (0, 1, 2, 9)}
            total = sum(counts.values()) + 4 * 0.5
            self.p[has] = {k: (v + 0.5) / total for k, v in counts.items()}

    def llk(self, ops: np.ndarray, elig: np.ndarray) -> np.ndarray:
        out = np.empty(len(ops))
        for k in range(len(ops)):
            o = int(ops[k]) if ops[k] in (0, 1, 2) else 9
            table = self.p[bool(elig[k] > 0)]
            # an illegal edit means a different (renamed / unrelated) name: the
            # chance of producing exactly this string is tiny
            out[k] = math.log(table[o]) - (o * math.log(3.0 * max(int(elig[k]), 1)) if o < 9 else 12.0)
        return out


UNRELATED_NAME_PENALTY = 12.0  # log-odds against a rename producing this exact string


class ScenarioModel:
    """Joint frequency of (typo class, tonnage status, flag missing) per
    vessel: the generator draws these together (e.g. missing tonnage only
    occurs with exactly one typo).

    Typo classes: 0/1/2 legal operations, 3 near-legal (more edits than the
    generator makes, but at most 3 plain edits), 4 unrelated name (rename).
    An unrelated name additionally pays the probability of producing that
    particular string, so it can never look more plausible than a near miss."""

    def __init__(self, keys: list[tuple]):
        self.counts = Counter(keys)
        self.total = sum(self.counts.values())

    def llk(self, k: np.ndarray, status: np.ndarray, flag_missing: bool) -> np.ndarray:
        cells = 5 * 4 * 2
        out = np.array([math.log((self.counts.get((int(a), int(b), bool(flag_missing)), 0) + 0.5) /
                                 (self.total + 0.5 * cells)) for a, b in zip(k, status)])
        return out - np.where(np.asarray(k) == 4, UNRELATED_NAME_PENALTY, 0.0)


def _best_ratio(g, ig, c) -> float:
    rs = [math.log(x / c) for x in (g, ig) if np.isfinite(x) and x > 0 and np.isfinite(c) and c > 0]
    return min(rs, key=abs) if rs else float("nan")


def _ops_class(ops: np.ndarray, osa: np.ndarray) -> np.ndarray:
    """0/1/2 legal typo operations, 3 near-legal (<= 3 plain edits), 4 unrelated."""
    ops = np.asarray(ops, dtype=float)
    osa = np.asarray(osa, dtype=float)
    return np.where(np.isin(ops, (0, 1, 2)), ops, np.where(osa <= 3, 3, 4)).astype(np.int64)


# ---------------------------------------------------------------------------
class Reference:
    """Statistics learned only from (a subset of) the official train rows."""

    def __init__(self, train: pd.DataFrame, catalog: Catalog):
        self.catalog = catalog
        tgt = train.target_vessel_id.astype(str)
        if not tgt.isin(catalog.id_to_idx).all():
            raise ValueError("train target missing from catalog")
        idx = tgt.map(catalog.id_to_idx).to_numpy()
        self.target_mask = np.zeros(catalog.n, dtype=bool)
        self.target_mask[idx] = True

        calls: dict[str, set] = defaultdict(set)
        for cs, t in zip(train.call_sign, tgt):
            cs = clean_str(cs)
            if cs:
                calls[cs].add(t)
        self.calls = {k: next(iter(v)) for k, v in calls.items() if len(v) == 1}

        aliases: dict[str, set] = defaultdict(set)
        for name, t in zip(train.vessel_name, tgt):
            aliases[alias_key(name)].add(t)
        self.aliases = {k: sorted(v) for k, v in aliases.items() if k}

        # vessel-level unique combinations
        uniq = pd.DataFrame({"t": tgt.to_numpy(), "i": idx,
                             "ptype": train.vessel_type.map(clean_str).to_numpy(),
                             "pflag": train.flag_name.map(clean_str).to_numpy()})
        uniq["ctype"] = catalog.type[uniq.i.to_numpy()]
        uniq["cflag"] = catalog.flag[uniq.i.to_numpy()]
        tp = uniq.drop_duplicates(["t", "ptype"])
        self.type_pair = Counter(zip(tp.ptype, tp.ctype))
        self.type_ctot = Counter(tp.ctype)
        self.type_ptot = Counter(tp.ptype)
        fp = uniq[uniq.pflag != ""].drop_duplicates(["t", "pflag"])
        self.flag_pair = Counter(zip(fp.pflag, fp.cflag))
        self.flag_ctot = Counter(fp.cflag)
        self.flag_ptot = Counter(fp.pflag)
        by_p: dict[str, Counter] = defaultdict(Counter)
        for (p, c), n in self.flag_pair.items():
            by_p[p][c] += n
        self.flag_best = {p: cnt.most_common(1)[0][0] for p, cnt in by_p.items()}
        vt = uniq.drop_duplicates("t")
        self.target_type_count = Counter(vt.ctype)
        # log-GT histogram of targets per port vessel type (size prior)
        self.size_bins = np.linspace(4.0, 13.0, 46)
        lg = catalog.log_gt[tp.i.to_numpy()]
        overall = np.histogram(lg[np.isfinite(lg)], bins=self.size_bins)[0].astype(float) + 0.5
        overall /= overall.sum()
        self.size_hist = {"": np.log(overall)}
        for pt, grp in tp.groupby("ptype"):
            x = catalog.log_gt[grp.i.to_numpy()]
            h = np.histogram(x[np.isfinite(x)], bins=self.size_bins)[0].astype(float)
            h = h + 5.0 * overall
            self.size_hist[pt] = np.log(h / h.sum())

        # per-row generator statistics (vessel-level unique rows)
        korean = train.vessel_name.map(T.is_korean).to_numpy()
        g = pd.to_numeric(train.gross_tonnage, errors="coerce").to_numpy()
        ig = pd.to_numeric(train.international_gross_tonnage, errors="coerce").to_numpy()
        rows = pd.DataFrame({"t": tgt.to_numpy(), "i": idx, "ko": korean, "name": train.vessel_name.to_numpy(),
                             "g": g, "ig": ig, "fm": train.flag_name.map(clean_str).to_numpy() == ""})
        rows = rows.drop_duplicates(["t", "name", "g", "ig", "fm"])
        rows["r"] = [_best_ratio(a, b, catalog.gt[i]) for a, b, i in zip(rows.g, rows.ig, rows.i)]
        self.ton_model = {ko: TonnageModel(rows.r[(rows.ko == ko) & np.isfinite(rows.r)].to_numpy())
                          for ko in (True, False)}
        ops, osa = [], []
        for name, i, ko in zip(rows.name, rows.i, rows.ko):
            if ko:
                ops.append(-1)
                osa.append(-1)
                continue
            qws = T.words(name)
            variants = [qws, T.strip_vessel_prefix(qws)]
            ops.append(min(T.name_ops(v, catalog.rwords[i]) for v in variants))
            osa.append(min(OSA.distance("".join(v), catalog.rcompact[i]) for v in variants))
        rows["ops"] = ops
        rows["osa"] = osa
        en = rows[~rows.ko]
        self.typo_model = TypoModel(en.ops.to_numpy(), catalog.elig[en.i.to_numpy()])
        status = ton_status(rows.r.to_numpy())
        rows["st"] = status
        en = rows[~rows.ko]
        self.scenario = {
            False: ScenarioModel(list(zip(_ops_class(en.ops.to_numpy(), en.osa.to_numpy()), en.st, en.fm))),
            True: ScenarioModel(list(zip(np.zeros(int(rows.ko.sum()), dtype=np.int64),
                                         rows.st[rows.ko], rows.fm[rows.ko]))),
        }

        # Korean word dictionary from aligned (Hangul word, catalog word) pairs.
        self.ko_dict: dict[str, Counter] = defaultdict(Counter)
        kp = pd.DataFrame({"q": train.vessel_name.to_numpy(), "i": idx}).drop_duplicates()
        for q, i in zip(kp.q, kp.i):
            if not T.is_korean(q):
                continue
            cw = catalog.words[i]
            for variant in T.ko_variants(q):
                qw = variant.split()
                if len(qw) == len(cw):
                    for a, b in zip(qw, cw):
                        if T.is_korean(a) and not b.isdigit():
                            self.ko_dict[a][b] += 1
                    break
        pairs = []
        for q, i in zip(kp.q, kp.i):
            if T.is_korean(q):
                pairs.append(([letter_units(v) for v in korean_variants(q)],
                              re.sub(r"\d", "", catalog.compact[i])))
        self.translit = Transliterator().fit(pairs, iters=10) if pairs else None

    # -- probabilities ----------------------------------------------------
    def type_logp(self, ptype: str, ctypes: np.ndarray) -> np.ndarray:
        """log P(port type | catalog type) with add-alpha smoothing."""
        n_types = max(len(self.type_ptot), 1)
        return np.array([math.log((self.type_pair.get((ptype, c), 0) + 0.1) /
                                  (self.type_ctot.get(c, 0) + 0.1 * n_types)) for c in ctypes])

    def type_rate(self, ctypes: np.ndarray) -> np.ndarray:
        """log share of catalog vessels of this type that are dataset targets."""
        cat = self.catalog
        return np.array([math.log((self.target_type_count.get(c, 0) + 0.05) /
                                  (cat.type_count.get(c, 0) + 1.0)) for c in ctypes])

    def flag_logp(self, pflag: str, cflags: np.ndarray) -> np.ndarray:
        if not pflag:
            return np.full(len(cflags), np.nan)
        n_flags = max(len(self.flag_ptot), 1)
        return np.array([math.log((self.flag_pair.get((pflag, c), 0) + 0.1) /
                                  (self.flag_ctot.get(c, 0) + 0.1 * n_flags)) for c in cflags])

    def size_logp(self, ptype: str, log_gt: np.ndarray) -> np.ndarray:
        hist = self.size_hist.get(ptype, self.size_hist[""])
        pos = np.clip(np.searchsorted(self.size_bins, np.nan_to_num(log_gt, nan=0.0)) - 1, 0, len(hist) - 1)
        return np.where(np.isfinite(log_gt), hist[pos], np.nan)

    def translate_ko(self, variant: str) -> str | None:
        """Word-by-word dictionary translation; None if no word is known."""
        out, hit = [], False
        for w in variant.split():
            cnt = self.ko_dict.get(w)
            if cnt:
                out.append(cnt.most_common(1)[0][0])
                hit = True
            else:
                out.append(w)
        return " ".join(out) if hit else None


# ---------------------------------------------------------------------------
class Query:
    """Pre-processed view of one query row."""

    def __init__(self, row: dict, ref: Reference):
        self.name = clean_str(row.get("vessel_name"))
        self.korean = T.is_korean(self.name)
        self.call = clean_str(row.get("call_sign"))
        self.gt = num(row.get("gross_tonnage"))
        self.igt = num(row.get("international_gross_tonnage"))
        self.ptype = clean_str(row.get("vessel_type"))
        self.pflag = clean_str(row.get("flag_name"))
        self.tons = [x for x in dict.fromkeys([self.gt, self.igt]) if x is not None]
        if self.korean:
            base = T.drop_number_marker(T.words(self.name))
            self.word_variants = [base]
            self.compacts = ["".join(base)]
            self.ko_texts = korean_variants(self.name)
            self.units = [letter_units(v) for v in self.ko_texts]
            keys = [T.ko_key(v) for v in self.ko_texts]
            self.dict_texts = []
            for v in self.ko_texts:
                tr = ref.translate_ko(v)
                if tr is not None:
                    self.dict_texts.append(tr)
                    keys.append(T.ko_key(tr))
            self.keys = list(dict.fromkeys(k for k in keys if k))
            self.cons = list(dict.fromkeys(T.consonants(k) for k in self.keys))
            self.romans = list(dict.fromkeys(
                T.romanize(v, s).replace(" ", "") for v in self.ko_texts for s in ("rr", "lw")))
            self.digits = tuple(re.findall(r"\d+", " ".join(self.ko_texts[0].split())))
        else:
            base = T.words(self.name)
            self.word_variants = [base]
            stripped = T.strip_vessel_prefix(base)
            if stripped != base:
                self.word_variants.append(stripped)
            self.compacts = ["".join(w) for w in self.word_variants]
            self.digits = tuple(re.findall(r"\d+", " ".join(self.word_variants[-1])))
            self.ko_texts = self.dict_texts = self.romans = self.units = []
            self.keys = [T.en_key(" ".join(w)) for w in self.word_variants]
            self.cons = [T.consonants(k) for k in self.keys]


# ---------------------------------------------------------------------------
ABS_FEATURES = (
    "is_ko", "q_len", "c_len", "len_diff", "exact", "osa", "lev_sim", "jw", "ratio",
    "token_set", "partial", "ops", "word_eq_frac", "idf_shared", "first_eq", "last_eq", "tail_swap",
    "tail_df",
    "key_sim", "cons_sim", "key_score", "roman_sim", "dict_sim",
    "tl_lp", "tl_llr", "tl_norm", "name_llk", "name_score",
    "q_digits", "c_digits", "digits_eq", "digits_conflict",
    "q_tons", "exact_gt", "exact_igt", "in_band", "near_band", "small_dev",
    "abs_log_min", "log_gt", "log_igt", "ton_llk", "ton_status", "scen_llk",
    "type_logp", "type_rate", "flag_logp", "flag_match", "q_flag_missing",
    "c_log_gt", "size_logp", "gen_score",
)
ABS_FEATURES = ABS_FEATURES + ABBR_FEATURES  # v14
# v27f: Latin dual encoder cosine, its z over the catalog, z(name ratio over the catalog) + emb_z
ABS_FEATURES = ABS_FEATURES + ("emb_cos", "emb_z", "fuse_z")
REL_SOURCES = ("name_score", "key_score", "tl_norm", "tl_llr", "gen_score", "ton_llk", "scen_llk",
               "emb_cos", "fuse_z")
REL_FEATURES = tuple(f"{s}_{k}" for s in REL_SOURCES for k in ("rank", "gap")) + ("ops_min", "n_cand")
FEATURES = ABS_FEATURES + REL_FEATURES


def relative_block(values: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """values [m, len(REL_SOURCES)] for one query -> ranks and gaps, plus the
    best typo count in the set and the set size (query-level context)."""
    m = values.shape[0]
    out = np.full((m, 2 * values.shape[1] + 2), np.nan, dtype=np.float32)
    for k in range(values.shape[1]):
        v = values[:, k]
        if not np.isfinite(v).any():
            continue
        filled = np.where(np.isfinite(v), v, -1e9)
        order = np.argsort(-filled, kind="stable")
        rank = np.empty(m)
        rank[order] = np.arange(m)
        out[:, 2 * k] = rank
        out[:, 2 * k + 1] = np.nanmax(v) - v
    if np.isfinite(ops).any():
        out[:, -2] = np.nanmin(ops)
    out[:, -1] = m
    return out


def with_relative(X_abs: np.ndarray, qid: np.ndarray) -> np.ndarray:
    """Append per-query relative features; rows of one query must be contiguous."""
    src = X_abs[:, [ABS_FEATURES.index(s) for s in REL_SOURCES]]
    ops = X_abs[:, ABS_FEATURES.index("ops")]
    rel = np.empty((len(X_abs), len(REL_FEATURES)), dtype=np.float32)
    change = np.flatnonzero(np.diff(qid)) + 1
    bounds = np.concatenate([[0], change, [len(qid)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rel[a:b] = relative_block(src[a:b], ops[a:b])
    return np.hstack([X_abs, rel])


class Matcher:
    def __init__(self, catalog: Catalog, ref: Reference):
        self.cat = catalog
        self.ref = ref
        self._span_cache: dict[int, list] = {}
        self.span_vocab = self.span_post = self.prefix_post = None
        self._build_prefix_post()

    def translit_scores(self, q: Query, cand: np.ndarray):
        """Best (over Hangul readings) transducer log-prob, LLR and per-letter LLR."""
        tl = self.ref.translit
        m = len(cand)
        nan = np.full(m, np.nan)
        if tl is None or not q.units or m == 0:
            return nan, nan, nan
        chars = self.cat.chars[cand]
        lens = self.cat.lens[cand]
        chars = chars[:, :max(int(lens.max()), 1)]
        ids = substring_ids(chars)
        best = None
        for units in q.units:
            if not units:
                continue
            lp, llr = tl.score(units, chars, ids, lens)
            cur = (lp, llr, llr / np.maximum(lens, 1))
            best = cur if best is None else tuple(np.maximum(a, b) for a, b in zip(best, cur))
        return best if best is not None else (nan, nan, nan)

    # -- v14: abbreviated names ---------------------------------------------
    def _spans(self, i: int):
        sp = self._span_cache.get(i)
        if sp is None:
            sp = span_keys(self.cat.words[i], T._en_word_key)
            self._span_cache[i] = sp
        return sp

    def _build_span_vocab(self) -> None:
        """Phonetic keys of proper word spans (not the whole name) -> ships."""
        post: dict[str, list[int]] = defaultdict(list)
        for i, ws in enumerate(self.cat.words):
            if len(ws) < 2:
                continue
            full = len(ws)
            for key, a, b in span_keys(ws, T._en_word_key):
                if b - a < full and len(key) >= 3 and not key.isdigit():
                    post[key].append(i)
        self.span_vocab = list(post)
        self.span_post = [post[k] for k in self.span_vocab]

    def _build_prefix_post(self) -> None:
        """3-5 letter prefixes of catalog words -> ships (a cut-short word)."""
        post: dict[str, set[int]] = defaultdict(set)
        for i, ws in enumerate(self.cat.words):
            for w in ws:
                if len(w) >= 5 and w.isalpha():
                    for n in (3, 4, 5):
                        if n < len(w):
                            post[w[:n]].add(i)
        self.prefix_post = {k: sorted(v) for k, v in post.items()}

    def _abbr(self, q, cand: np.ndarray, F: dict) -> None:
        m = len(cand)
        cols = {n: np.full(m, np.nan) for n in ABBR_FEATURES}
        cat = self.cat
        qws = [tuple(w) for w in q.word_variants] if not q.korean else []
        for k, i in enumerate(cand):
            cw = tuple(cat.words[i])
            if qws:
                best = min(align(qw, cw) for qw in qws)
                if best[0] < 99:
                    (cols["ab_cost"][k], cols["ab_unq"][k], cols["ab_skip"][k],
                     cols["ab_init"][k], cols["ab_pref"][k]) = best
                    cols["ab_full"][k] = float(best[1] == 0)
            vals = span_features(q.keys, self._spans(int(i)), len(cw), float(F["key_sim"][k]))
            for name, v in zip(ABBR_FEATURES[6:], vals):
                cols[name][k] = v
        F.update(cols)

    # -- v27f: Latin dual encoder ---------------------------------------------
    def name_ratio(self, q: Query) -> np.ndarray:
        """fuzz.ratio (best compact variant) against every raw catalog name, unmasked; cached on the query."""
        if getattr(q, "_ratio", None) is None:
            rs = None
            for c in q.compacts:
                s = process.cdist([c], self.cat.rcompact, scorer=fuzz.ratio, dtype=np.float32, workers=1)[0]
                rs = s if rs is None else np.maximum(rs, s)
            q._ratio = rs if rs is not None else np.zeros(self.cat.n, np.float32)
        return q._ratio

    def de_scores(self, q: Query):
        """(emb_cos, emb_z, fuse_z) over the whole catalog for a Latin query, cached on the query."""
        if getattr(q, "_de", None) is None:
            emb = self.cat.de().scores(T.nfkc_upper(q.name))
            emb_z = zscore(emb)
            q._de = (emb, emb_z, zscore(self.name_ratio(q)) + emb_z)
        return q._de

    # -- retrieval -----------------------------------------------------------
    def candidates(self, q: Query) -> tuple[np.ndarray, np.ndarray]:
        """Candidate catalog indices (eligible, tonnage-gated) and a mask of
        those that are reference (train) targets."""
        cat = self.cat
        pool: set[int] = set()

        def top(scores: np.ndarray, k: int, among: np.ndarray | None = None):
            idx = np.arange(len(scores)) if among is None else among
            s = scores if among is None else scores[among]
            if len(idx) == 0:
                return
            if len(idx) > k:
                idx = idx[np.argpartition(-s, k - 1)[:k]]
            pool.update(int(i) for i in idx)

        near = np.zeros(cat.n, dtype=bool)
        for x in q.tons:
            near[cat.gt_window(x, 0.94, 1.065)] = True
            pool.update(cat.by_gt.get(float(x), [])[:300])
        near_idx = np.flatnonzero(near & cat.eligible)

        if q.korean:
            ks = cs = None
            for key in q.keys:
                s = process.cdist([key], cat.key, scorer=Levenshtein.normalized_similarity,
                                  dtype=np.float32, workers=1)[0]
                ks = s if ks is None else np.maximum(ks, s)
            for key in q.cons:
                s = process.cdist([key], cat.cons, scorer=Levenshtein.normalized_similarity,
                                  dtype=np.float32, workers=1)[0]
                cs = s if cs is None else np.maximum(cs, s)
            score = np.where(cat.eligible, ks + cs, -1.0)
            top(score, 60)
            top(score, 30, near_idx)
            # transducer over the tonnage pool plus the best phonetic-key matches
            broad = np.argpartition(-score, 2000)[:2000]
            broad = broad[score[broad] > 0]
            tl_pool = np.union1d(near_idx, broad)
            if len(tl_pool):
                lp, _, norm = self.translit_scores(q, tl_pool)
                for arr in (norm, lp):
                    a = np.nan_to_num(arr, nan=-1e9)
                    k = min(40, len(a))
                    pool.update(int(i) for i in tl_pool[np.argpartition(-a, k - 1)[:k]])
                if len(near_idx):
                    in_near = np.isin(tl_pool, near_idx)
                    a = np.where(in_near, np.nan_to_num(norm, nan=-1e9), -1e9)
                    k = min(30, int(in_near.sum()))
                    if k:
                        pool.update(int(i) for i in tl_pool[np.argpartition(-a, k - 1)[:k]])
            for text in q.dict_texts:
                pool.update(cat.by_compact.get("".join(T.words(text)), []))
            if self.span_vocab:
                for key in q.keys:
                    if len(key) < 3:
                        continue
                    s = process.cdist([key], self.span_vocab, scorer=Levenshtein.normalized_similarity,
                                      dtype=np.float32, workers=1, score_cutoff=0.75)[0]
                    hits = np.flatnonzero(s >= 0.75)
                    if len(hits) > 40:
                        hits = hits[np.argpartition(-s[hits], 39)[:40]]
                    for h in hits:
                        post = self.span_post[h]
                        if len(post) <= 60:
                            pool.update(post)
                        elif len(near_idx):
                            pa = np.asarray(post)
                            pool.update(int(i) for i in pa[near[pa]][:60])
        else:
            dist = None
            for c in q.compacts:
                d = process.cdist([c], cat.rcompact, scorer=OSA.distance, score_cutoff=4,
                                  dtype=np.int32, workers=1)[0]
                dist = d if dist is None else np.minimum(dist, d)
            dist = np.where(cat.eligible, dist, 99)
            close = np.flatnonzero(dist <= 3)
            if len(close) > 80:
                close = close[np.argsort(dist[close], kind="stable")[:80]]
            pool.update(int(i) for i in close)
            rs = np.where(cat.eligible, self.name_ratio(q), -1.0)
            top(rs, 25)
            top(rs, 40, near_idx)
            # v27f: dual encoder fused with the name ratio (z over the catalog), overall and in the GT window
            fz = np.where(cat.eligible, self.de_scores(q)[2], -1e9)
            top(fz, DE_K_GLOBAL)
            top(fz, DE_K_WINDOW, near_idx)
            for ws in q.word_variants:
                for w in set(ws):
                    if len(w) < 3 or w.isdigit():
                        continue
                    post = cat.word_post.get(w, [])
                    if not post and self.prefix_post is not None and len(w) <= 5:
                        post = self.prefix_post.get(w, [])
                    if not post:
                        continue
                    if len(post) <= 150:
                        pool.update(post)
                    elif len(near_idx):
                        post_arr = np.asarray(post)
                        pool.update(int(i) for i in post_arr[near[post_arr]][:150])
                    else:
                        inits = {x for x in ws if len(x) == 1 and x.isalpha()}
                        if inits:
                            pool.update([i for i in post if inits <= {cw[0] for cw in cat.words[i] if cw}][:150])

        cand = np.fromiter(pool, dtype=np.int64)
        if len(cand):
            keep = cat.eligible[cand]
            if q.tons:
                dev = np.full(len(cand), np.inf)
                for x in q.tons:
                    dev = np.minimum(dev, np.abs(cat.log_gt[cand] - math.log(x)))
                keep &= dev <= TON_GATE
            cand = cand[keep]
            if q.korean and q.digits:
                # A Hangul name with Arabic digits transcribes the hull number
                # as is: every such train pair has identical digits.  (Without
                # digits the number may be spelled out, so no filtering.)
                same = np.array([cat.digits[i] == q.digits for i in cand], dtype=bool)
                if same.any():
                    cand = cand[same]
        cand.sort()
        return cand, self.ref.target_mask[cand]

    # -- features ------------------------------------------------------------
    def features(self, q: Query, cand: np.ndarray) -> np.ndarray:
        """Absolute (candidate-set independent) features, columns = ABS_FEATURES."""
        cat, ref = self.cat, self.ref
        m = len(cand)
        F = {}
        nan = np.full(m, np.nan)
        F["is_ko"] = np.full(m, float(q.korean))
        F["q_len"] = np.full(m, float(len(q.compacts[0])))

        def best(scorer, queries, choices, maximize=True):
            out = None
            for qq in queries:
                s = process.cdist([qq], choices, scorer=scorer, dtype=np.float32, workers=1)[0]
                out = s if out is None else (np.maximum(out, s) if maximize else np.minimum(out, s))
            return out if out is not None else nan.copy()

        if q.korean:
            c_compact = [cat.compact[i] for i in cand]
            F["key_sim"] = best(Levenshtein.normalized_similarity, q.keys, [cat.key[i] for i in cand])
            F["cons_sim"] = best(Levenshtein.normalized_similarity, q.cons, [cat.cons[i] for i in cand])
            for name in ("exact", "osa", "lev_sim", "jw", "ratio", "token_set", "partial",
                         "ops", "word_eq_frac", "idf_shared", "first_eq", "last_eq", "tail_swap",
                         "tail_df"):
                F[name] = nan
            F["roman_sim"] = best(fuzz.ratio, q.romans, c_compact) / 100.0
            if q.dict_texts:
                F["dict_sim"] = best(Levenshtein.normalized_similarity,
                                     [T.ko_key(t) for t in q.dict_texts], [cat.key[i] for i in cand])
            else:
                F["dict_sim"] = nan
            F["tl_lp"], F["tl_llr"], F["tl_norm"] = self.translit_scores(q, cand)
            F["name_llk"] = np.nan_to_num(F["tl_llr"], nan=-50.0)
            F["name_score"] = np.nan_to_num(F["tl_norm"], nan=-50.0)
            c_digits = [cat.digits[i] for i in cand]
            k_class = np.zeros(m, dtype=np.int64)
        else:
            c_compact = [cat.rcompact[i] for i in cand]
            F["key_sim"] = best(Levenshtein.normalized_similarity, q.keys, [cat.rkey[i] for i in cand])
            F["cons_sim"] = best(Levenshtein.normalized_similarity, q.cons, [cat.rcons[i] for i in cand])
            F["exact"] = np.array([float(c in q.compacts) for c in c_compact])
            F["osa"] = best(OSA.distance, q.compacts, c_compact, maximize=False)
            F["lev_sim"] = best(Levenshtein.normalized_similarity, q.compacts, c_compact)
            F["jw"] = best(JaroWinkler.similarity, q.compacts, c_compact)
            F["ratio"] = best(fuzz.ratio, q.compacts, c_compact) / 100.0
            q_spaced = [" ".join(w) for w in q.word_variants]
            c_spaced = [" ".join(cat.rwords[i]) for i in cand]
            F["token_set"] = best(fuzz.token_set_ratio, q_spaced, c_spaced) / 100.0
            F["partial"] = best(fuzz.partial_ratio, q.compacts, c_compact) / 100.0
            ops = np.full(m, 9.0)
            weq, idf, feq, leq, tsw = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
            tdf = np.full(m, np.nan)
            for k, i in enumerate(cand):
                cw = cat.rwords[i]
                for qw in q.word_variants:
                    if F["osa"][k] <= 4:
                        ops[k] = min(ops[k], T.name_ops(qw, cw))
                    shared = set(qw) & set(cw)
                    weq[k] = max(weq[k], len(shared) / max(len(qw), len(cw), 1))
                    idf[k] = max(idf[k], sum(math.log(cat.n / cat.word_df.get(w, 1)) for w in shared))
                    if qw and cw:
                        feq[k] = max(feq[k], float(qw[0] == cw[0]))
                        leq[k] = max(leq[k], float(qw[-1] == cw[-1]))
                        if len(qw) == len(cw) >= 2 and qw[0] != cw[0] and qw[1:] == cw[1:]:
                            tsw[k] = 1.0
                            tdf[k] = min(cat.word_df.get(w, 1) for w in cw[1:])
            F["ops"], F["word_eq_frac"], F["idf_shared"] = ops, weq, idf
            F["first_eq"], F["last_eq"], F["tail_swap"], F["tail_df"] = feq, leq, tsw, tdf
            F["roman_sim"] = F["dict_sim"] = nan
            F["tl_lp"] = F["tl_llr"] = F["tl_norm"] = nan
            F["name_llk"] = ref.typo_model.llk(ops, cat.elig[cand])
            F["name_score"] = F["lev_sim"] + 0.5 * F["jw"] - 0.2 * np.minimum(ops, 3)
            c_digits = [cat.rdigits[i] for i in cand]
            k_class = _ops_class(ops, F["osa"])
        F["c_len"] = np.array([len(c) for c in c_compact], dtype=float)
        F["len_diff"] = F["q_len"] - F["c_len"]
        F["key_score"] = F["key_sim"] + F["cons_sim"]

        F["q_digits"] = np.full(m, float(len(q.digits) > 0))
        F["c_digits"] = np.array([float(len(d) > 0) for d in c_digits])
        F["digits_eq"] = np.array([float(d == q.digits) for d in c_digits])
        F["digits_conflict"] = np.array([float(bool(d) and bool(q.digits) and d != q.digits)
                                         for d in c_digits])

        cgt = cat.gt[cand]
        F["q_tons"] = np.full(m, float(len(q.tons)))
        F["exact_gt"] = np.array([float(q.gt is not None and g == q.gt) for g in cgt])
        F["exact_igt"] = np.array([float(q.igt is not None and g == q.igt) for g in cgt])
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(q.gt / cgt) if q.gt is not None else nan
            lig = np.log(q.igt / cgt) if q.igt is not None else nan
        F["log_gt"], F["log_igt"] = lg, lig
        absmin = np.fmin(np.abs(lg), np.abs(lig))
        F["abs_log_min"] = absmin
        F["in_band"] = (in_band(lg) | in_band(lig)).astype(float)
        F["near_band"] = ((absmin > 0.015) & (absmin < 0.06)).astype(float)
        F["small_dev"] = ((absmin > 0) & (absmin < 0.02)).astype(float)
        tm = ref.ton_model[q.korean]
        llk = nan.copy()
        if q.gt is not None:
            llk = np.fmax(llk, tm.llk(lg, q.gt))
        if q.igt is not None:
            llk = np.fmax(llk, tm.llk(lig, q.igt))
        F["ton_llk"] = llk
        best_r = np.where(np.abs(np.nan_to_num(lg, nan=9.0)) <= np.abs(np.nan_to_num(lig, nan=9.0)), lg, lig)
        status = ton_status(best_r)
        F["ton_status"] = status.astype(float)
        F["scen_llk"] = ref.scenario[q.korean].llk(k_class, status, not q.pflag)

        ctypes = cat.type[cand]
        F["type_logp"] = ref.type_logp(q.ptype, ctypes)
        F["type_rate"] = ref.type_rate(ctypes)
        cflags = cat.flag[cand]
        F["flag_logp"] = ref.flag_logp(q.pflag, cflags)
        best_flag = ref.flag_best.get(q.pflag)
        F["flag_match"] = (np.array([float(f == best_flag) for f in cflags]) if q.pflag else nan)
        F["q_flag_missing"] = np.full(m, float(not q.pflag))
        F["c_log_gt"] = cat.log_gt[cand]
        F["size_logp"] = ref.size_logp(q.ptype, cat.log_gt[cand])
        if NO_PRIORS:
            # no dataset priors: which catalog types / sizes were train targets says
            # nothing about the vessels of the extended evaluation data
            gen = F["name_llk"] + F["type_logp"] + np.nan_to_num(F["flag_logp"], nan=0.0)
            gen = gen + (np.nan_to_num(F["ton_llk"], nan=-20.0) if q.tons else 0.0)
            F["type_rate"] = np.full(m, np.nan)
            F["size_logp"] = np.full(m, np.nan)
            F["c_log_gt"] = np.full(m, np.nan)
        else:
            gen = F["name_llk"] + F["type_logp"] + F["type_rate"] + np.nan_to_num(F["flag_logp"], nan=0.0)
            gen = gen + (np.nan_to_num(F["ton_llk"], nan=-20.0) if q.tons else F["size_logp"])
        if not q.korean:
            gen = gen + F["scen_llk"]
        F["gen_score"] = gen
        self._abbr(q, cand, F)
        if not q.korean and m:
            emb, emb_z, fuse = self.de_scores(q)
            F["emb_cos"], F["emb_z"], F["fuse_z"] = emb[cand], emb_z[cand], fuse[cand]
        else:
            F["emb_cos"] = F["emb_z"] = F["fuse_z"] = np.full(m, np.nan)
        return np.column_stack([np.asarray(F[name], dtype=np.float32) for name in ABS_FEATURES])


def rename_override(X_abs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Scores for one query after the rename rule.

    A Latin port name that no candidate explains with legal typo edits is a
    real rename (the generator never produces it otherwise; in train only the
    13 renamed rows have no legal candidate).  Renames typically keep the
    ship's own name and swap the fleet prefix ('OCEAN WREN' <- 'XY Wren') and
    carry real, unperturbed tonnage.  Prefer such candidates when present."""
    col = {n: i for i, n in enumerate(ABS_FEATURES)}
    if X_abs.shape[0] == 0 or X_abs[0, col["is_ko"]] == 1:
        return scores
    ops = X_abs[:, col["ops"]]
    if np.nanmin(ops) < 9:
        return scores
    tail = (X_abs[:, col["tail_swap"]] == 1) & (np.nan_to_num(X_abs[:, col["tail_df"]], nan=1e9) <= 40)
    dev = X_abs[:, col["abs_log_min"]]
    tons_ok = np.where(np.isfinite(dev), dev <= 0.003, True)
    pick = tail & tons_ok
    if not pick.any():
        return scores
    return np.where(pick, scores + 1000.0, scores)
