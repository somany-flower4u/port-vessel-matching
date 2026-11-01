"""v36js: v29v with an LLM spelling judge (Qwen3-1.7B, bf16) re-ranking Hangul near-ties: when the Hangul ranker's top-2 margin is below 3.0 and the judge's log-probability of the row's Hangul spelling given the rank-2 catalog name exceeds that of rank-1 by more than 3.0 nats, the rank-2 vessel is answered; Latin path unchanged (v28g). Variant of v36a with the subset guard as v36s and the near-tie window narrowed to margin < 2.0.

Base: DACON inference entry point: vessel matching (v29v: Hangul path = v29q (kogen6 character dual encoder with a 2x wider tower — embedding 96, conv width 320 — 200 epochs x 70k synthetic names; v20-recipe rankers); Latin path = v28g (retrained Latin character dual encoder, v19-recipe rankers)).

Each test row is routed by its own vessel_name only (Hangul -> v1 matcher,
otherwise -> v3 matcher) and is then predicted on its own from its own fields
plus the official train.csv (model/) and vessel_catalog.csv (data/):

1. call sign seen in train            -> that train vessel
2. exact train name, tonnage agrees   -> that train vessel
3. otherwise rank catalog candidates with the LightGBM ranker; when the row
   has a call sign that train never saw, train vessels are not candidates
   (every train vessel has exactly one call sign).  A Latin name that no
   tonnage-plausible candidate explains with legal typo edits is a rename
   (see rename_rule).

model/ko and model/en hold the unchanged v1 / v3 matcher packages and rankers.
Rows are spread over worker processes; every row is still computed by the
same deterministic code, so the output does not depend on how rows are split.

v36a judge (model/llm): after the worker pool has finished, each Hangul row
that went through the ranker (not the direct call-sign rule) and whose top-2
margin s[rank1] - s[rank2] < LLM_TAU is scored on its own (batch 1, fixed
10-shot prompt prefix) by model/llm/judge.py: l_k = log p(row's Hangul name |
catalog name of rank k).  If l2 - l1 > LLM_KAPPA the rank-2 vessel is answered.
Without torch / transformers, or on any judge error, the v29v answers stay.
"""
from __future__ import annotations

import os

# Single-threaded numerical libraries: parallelism comes from worker
# processes, and no library thread pool exists when the workers are forked.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
sys.path.insert(0, str(MODEL / "ko"))
sys.path.insert(0, str(MODEL / "en"))

ALIAS_TON_TOL = 0.12  # train maximum |log ratio| for the true vessel is 0.10
EXCLUDE_TRAIN_VESSELS_FOR_UNSEEN_CALL_SIGN = True
# Variant switches (defaults reproduce v5 exactly).
EN_EXCLUDE = True       # non-Hangul rows: drop train vessels for an unseen call sign
EN_FILTERS = True       # non-Hangul rows: unique-name / catalog-GT / tonnage-gate filters
KO_EXCLUDE = True       # Hangul rows: drop train vessels for an unseen call sign
KO_FILTERS = False  # Hangul rows: unique-name / catalog-GT / tonnage-gate filters
PROBE_DUMMY = ""        # leaderboard diagnosis only: "ko" / "en" rows all get one fixed (wrong) vessel
IGNORE_CALL_SIGN = False  # all rows: no call-sign lookup and no train-vessel exclusion
RENAME_ONLY_EXCLUSION = False  # with the exclusion above off: still drop train vessels for a Latin rename
# Per-script overrides (empty / default values keep the switches above in charge).
EN_EXCLUDE_MODE = "never"  # "always" | "rename" | "never": when train vessels leave the candidate set
KO_EXCLUDE_MODE = ""    # "always" | "never"
EN_ALIAS = True         # non-Hangul rows: exact-train-name rule
KO_ALIAS = False  # Hangul rows: exact-train-name rule
EN_ELIGIBLE = True      # non-Hangul rows: only unique-name ships with a catalog GT
KO_ELIGIBLE = True      # Hangul rows: only unique-name ships with a catalog GT
EN_TON_GATE = 99.0  # non-Hangul rows: no tonnage gate (v15: rankers trained with tonnage errors)
KO_TON_GATE = 0.2       # Hangul rows: max |log(port / catalog GT)| of a candidate
EN_TIEBREAK = True     # non-Hangul rows: typo-path likelihood between indistinguishable twins
KO_DIGIT_FILTER = False  # Hangul rows: choose among candidates with the same Arabic digits
KO_SINGLE_LETTERS = False  # Hangul rows: also read standalone letter names ('에스' -> 'S')
EN_RENAME = False  # non-Hangul rows: rename override (v3)
EN_MISSING_TON_EXACT = False  # non-Hangul rows without tonnage: prefer an exact-name candidate
ALIAS_RULE = True       # all rows: exact train name with agreeing tonnage -> that train vessel
CALL_TON_CHECK = False  # all rows: ignore a known call sign whose train vessel's tonnage disagrees
EN_ASCII_FOLD = True  # non-Hangul rows: fold accented / non-Latin letters to ASCII (unidecode)
KO_ASCII_FOLD = False   # Hangul rows: fold their non-Hangul, non-ASCII letters (Hanja, Cyrillic, ...) too
KO_ENSEMBLE = 0.0      # Hangul rows: weight of the second Hangul ranker (model/ko2, v16) in a per-query probability average
NFC_NAMES = True       # all rows: NFC-normalise vessel names before routing (decomposed Hangul is still Hangul)
EN_RENAME2 = False      # non-Hangul rows: generalised rename rule (replaces the v3 rename override)
EN_RENAME3 = False  # non-Hangul rows: tonnage-aware rename rule (replaces the v3 rename override)
RENAME_PLAUSIBLE = True  # with EN_RENAME3: candidates with tonnage outside the train range rank last
RENAME_OWN = True       # with EN_RENAME3: own name kept, fleet prefix swapped / dropped / added
RENAME_UNIQUE = True    # with EN_RENAME3: unrelated name -> the unique exact-tonnage candidate
RENAME_RETRIEVE = False  # with EN_RENAME3: a rename also gets every catalog ship with the same tonnage as candidate
RETRIEVE_IF_NO_EXACT = True  # ... only when no exact-GT, flag-consistent candidate was found
RETRIEVE_FLAG_ONLY = True    # ... and only ships with a consistent flag
EN_SERIES = False       # with EN_RENAME3: a series name shared by sisters -> a sister that left the series
SISTER_TOL = 0.003      # renames carry real tonnage: sisters are within 0.3 %
RARE_DF = 40            # a word in at most this many catalog names is a ship's own name
PLAUSIBLE_DEV = 0.1     # train maximum |log ratio| between port tonnage and the true catalog GT is 0.0915
EXACT_DEV = 0.0005      # renamed train rows: 11 of 13 exact, the rest within 0.05 %

# Standalone Hangul spellings of single Latin letters that are not also common
# Korean name syllables (씨/시 'sea', 이, 오, 지, 유, 아이 are left out).
SINGLE_LETTERS = {"에이": "A", "비": "B", "디": "D", "에프": "F", "에이치": "H", "제이": "J", "케이": "K",
                  "엘": "L", "엠": "M", "엔": "N", "피": "P", "큐": "Q", "알": "R", "에스": "S", "티": "T",
                  "브이": "V", "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z"}
CHUNK = 128
MAX_WORKERS = 6  # evaluation server: 6 vCPU
POOL_TIMEOUT = 900  # seconds; afterwards fall back to sequential prediction
# v36a LLM spelling judge for Hangul near-ties (rank 1 vs rank 2 of the Hangul ranker).
LLM_JUDGE = True
LLM_MODEL_DIR = MODEL / "llm" / "Qwen3-1.7B"
LLM_FORM = "ko10"       # fixed 10-shot prompt of the go/no-go study
LLM_TAU = 2.0           # judge only when s[rank1] - s[rank2] < LLM_TAU
LLM_KAPPA = 3.0         # answer rank 2 when l2 - l1 > LLM_KAPPA (nats)
LLM_SUBSET_GUARD = True  # never swap to a rank-2 whose name words are a strict subset of rank-1's
LLM_MAX_ROWS = 400      # guard: judge at most this many rows (in test order)
LLM_TIME_BUDGET = 600.0  # guard: seconds of judging (model load included); later rows keep rank 1
LLM_DEADLINE = 1500.0   # guard: no judging once the script has run this long (server limit 1800 s)
T_START = time.time()

PREDICTORS: dict[str, "Predictor"] = {}


def read_csv(primary: Path, fallback: Path) -> pd.DataFrame:
    path = primary if primary.exists() else fallback
    return pd.read_csv(path, encoding="utf-8-sig")


def load_package(package: str, ascii_fold: bool = False):
    """Import a matcher package; memoise its per-word phonetic key (a pure
    function of the word) so repeated catalog words are computed once."""
    text = importlib.import_module(f"{package}.text")
    if not hasattr(text._en_word_key, "cache_info"):
        text._en_word_key = functools.lru_cache(maxsize=None)(text._en_word_key)
    if ascii_fold and not getattr(text.nfkc_upper, "ascii_fold", False):
        from unidecode import unidecode
        base = text.nfkc_upper

        def nfkc_upper_ascii(value) -> str:
            s = base(value)
            if s and not s.isascii():
                s = "".join(ch if ch.isascii() or "가" <= ch <= "힣" else unidecode(ch) for ch in s).upper()
            return s

        nfkc_upper_ascii.ascii_fold = True
        text.nfkc_upper = nfkc_upper_ascii
    return importlib.import_module(f"{package}.matcher")


class Predictor:
    """One matcher version (its own package, rankers and reference)."""

    def __init__(self, package: str, model_dir: Path, catalog_df: pd.DataFrame, train_df: pd.DataFrame,
                 exclude: bool = True, filters: bool = True, tiebreak: bool = False,
                 digit_filter: bool = False, single_letters: bool = False, ascii_fold: bool = False,
                 exclude_mode: str = "", alias: bool = True, eligible: bool = True, ton_gate: float = 0.2):
        self.mm = load_package(package, ascii_fold)
        self.exclude, self.tiebreak, self.digit_filter = exclude, tiebreak, digit_filter
        if exclude_mode:
            self.exclude_mode = exclude_mode
        elif not exclude:
            self.exclude_mode = "never"
        elif EXCLUDE_TRAIN_VESSELS_FOR_UNSEEN_CALL_SIGN:
            self.exclude_mode = "always"
        else:
            self.exclude_mode = "rename" if RENAME_ONLY_EXCLUSION else "never"
        self.alias = alias
        self.is_ko = package == "vm_ko"  # the judge info is collected for the Hangul predictor only
        self.ties: list[tuple[int, str, str, float]] = []  # (position in chunk, rank-1 id, rank-2 id, margin)
        self.model_dir = model_dir
        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        if self.manifest["features"] != list(self.mm.FEATURES):
            raise ValueError(f"{package}: model / feature code mismatch")
        self.cat = self.mm.Catalog(catalog_df)
        if not filters:
            self.cat.eligible[:] = True
            self.mm.TON_GATE = 99.0
        else:
            if not eligible:
                self.cat.eligible[:] = True
            self.mm.TON_GATE = ton_gate
        self.ref = self.mm.Reference(train_df, self.cat)
        if single_letters:
            # queries only: the transliterator above was fitted on the original readings
            base = self.mm.korean_variants

            def with_single_letters(name: str) -> list[str]:
                out = base(name)
                extra = []
                for v in out:
                    ws = v.split()
                    if len(ws) >= 2 and any(w in SINGLE_LETTERS for w in ws):
                        extra.append(" ".join(SINGLE_LETTERS.get(w, w) for w in ws))
                return list(dict.fromkeys(out + extra))

            self.mm.korean_variants = with_single_letters
        self.matcher = self.mm.Matcher(self.cat, self.ref)
        self.rename_override = getattr(self.mm, "rename_override", None)
        self._boosters = None

    @property
    def boosters(self):
        # loaded lazily so that LightGBM is first used inside the worker
        if self._boosters is None:
            import lightgbm as lgb
            self._boosters = [lgb.Booster(model_file=str(self.model_dir / f))
                              for f in self.manifest["models"]]
        return self._boosters

    def tons_agree(self, q, vessel_id: str) -> bool:
        g = self.cat.gt[self.cat.id_to_idx[vessel_id]]
        if not q.tons:
            return True
        return bool(np.isfinite(g)) and min(abs(math.log(x / g)) for x in q.tons) <= ALIAS_TON_TOL

    def direct(self, q, row: dict) -> str | None:
        hit = self.ref.calls.get(q.call)
        if hit is not None and hit in self.cat.id_to_idx:
            if not CALL_TON_CHECK or self.tons_agree(q, hit):
                return hit
            q.call = ""  # a reused call sign: match by name and tonnage, without exclusion
        if not ALIAS_RULE or not self.alias:
            return None
        aliases = self.ref.aliases.get(self.mm.alias_key(row.get("vessel_name")), [])
        if len(aliases) == 1 and aliases[0] in self.cat.id_to_idx:
            g = self.cat.gt[self.cat.id_to_idx[aliases[0]]]
            if not q.tons:
                return aliases[0]
            if np.isfinite(g) and min(abs(math.log(x / g)) for x in q.tons) <= ALIAS_TON_TOL:
                return aliases[0]
        return None

    def candidate_set(self, q) -> np.ndarray:
        cand, is_ref = self.matcher.candidates(q)
        if q.call and self.exclude_mode == "always":
            cand = cand[~is_ref]
        if len(cand) == 0:
            cand = self.fallback(q)
        return cand

    def rename_pool(self, q, cand: np.ndarray, feats: np.ndarray):
        """Renamed rows carry the real tonnage, but name retrieval cannot find a
        ship under an unrelated name, and the exact-GT lookup misses a tonnage
        a few tons off (train: 29765 for catalog 29768).  When no plausible
        candidate explains the name with legal typo edits, add every eligible
        catalog ship whose GT is within EXACT_DEV of the port tonnage."""
        if q.korean or not q.tons or len(cand) == 0:
            return cand, feats
        col = self.mm.ABS_FEATURES.index
        dev = feats[:, col("abs_log_min")]
        plausible = np.where(np.isfinite(dev), dev <= PLAUSIBLE_DEV, True) if RENAME_PLAUSIBLE \
            else np.ones(len(cand), bool)
        if np.nanmin(np.where(plausible, feats[:, col("ops")], 9.0)) < 9:
            return cand, feats
        fm = feats[:, col("flag_match")]
        flag_ok = np.where(np.isfinite(fm), fm == 1, True)
        if RETRIEVE_IF_NO_EXACT and ((dev == 0) & flag_ok).any():
            return cand, feats  # the exact-GT lookup already found a consistent ship
        extra: set[int] = set()
        for x in q.tons:
            extra.update(int(i) for i in self.cat.gt_window(x, math.exp(-EXACT_DEV), math.exp(EXACT_DEV)))
        extra -= set(cand.tolist())
        add = np.array(sorted(extra), dtype=np.int64)
        if len(add):
            add = add[self.cat.eligible[add]]
            if q.call and self.exclude_mode in ("always", "rename"):
                add = add[~self.ref.target_mask[add]]
            best_flag = self.ref.flag_best.get(q.pflag) if q.pflag else None
            if RETRIEVE_FLAG_ONLY and best_flag is not None:
                add = add[self.cat.flag[add] == best_flag]
        if not len(add):
            return cand, feats
        cand = np.sort(np.concatenate([cand, add]))
        return cand, self.matcher.features(q, cand)

    def exclude_for_rename(self, q, cand: np.ndarray, feats: np.ndarray):
        """Drop train vessels only when the Latin name is a rename (no
        tonnage-plausible candidate explains it with legal typo edits).  A
        train vessel can then only be the ship through the exact-train-name
        rule, which already ran; a typo-ed train vessel stays a candidate even
        under a call sign train never saw (re-issued call signs)."""
        if q.korean or not q.call or len(cand) == 0:
            return cand, feats
        col = self.mm.ABS_FEATURES.index
        dev = feats[:, col("abs_log_min")]
        plausible = np.where(np.isfinite(dev), dev <= PLAUSIBLE_DEV, True) if RENAME_PLAUSIBLE \
            else np.ones(len(cand), bool)
        if np.nanmin(np.where(plausible, feats[:, col("ops")], 9.0)) < 9:
            return cand, feats
        keep = ~self.ref.target_mask[cand]
        if not keep.any() or keep.all():
            return cand, feats
        return cand[keep], feats[keep]

    def fallback(self, q) -> np.ndarray:
        """Very defensive: name similarity over the whole catalog."""
        from rapidfuzz import fuzz, process
        texts = q.compacts if not q.korean else list(q.keys)
        pool = self.cat.compact if not q.korean else self.cat.key  # as in the v1 / v3 scripts
        best = None
        for t in texts:
            s = process.cdist([t], pool, scorer=fuzz.ratio, dtype=np.float32, workers=1)[0]
            best = s if best is None else np.maximum(best, s)
        if best is None:
            return np.arange(min(10, self.cat.n))
        return np.sort(np.argpartition(-best, 49)[:50])

    def predict_records(self, records: list[dict]) -> list[str]:
        out: list[str] = [""] * len(records)
        self.ties = []
        pending = []
        for pos, row in enumerate(records):
            q = self.mm.Query(row, self.ref)
            if IGNORE_CALL_SIGN:
                q.call = ""
            hit = self.direct(q, row)
            if hit is not None:
                out[pos] = hit
                continue
            cand = self.candidate_set(q)
            feats = self.matcher.features(q, cand)
            if self.exclude_mode == "rename":
                cand, feats = self.exclude_for_rename(q, cand, feats)
            if RENAME_RETRIEVE and EN_RENAME3:
                cand, feats = self.rename_pool(q, cand, feats)
            pending.append((pos, cand, feats, q))
            if len(pending) >= CHUNK:
                self._score(pending, out)
                pending = []
        if pending:
            self._score(pending, out)
        return out

    def _score(self, pending, out) -> None:
        blocks = [self.mm.with_relative(f, np.full(len(c), k, dtype=np.int32))
                  for k, (_, c, f, _) in enumerate(pending)]
        X = np.vstack(blocks)
        scores = np.mean([b.predict(X, num_threads=1) for b in self.boosters], axis=0)
        offset = 0
        for pos, cand, feats, q in pending:
            s = scores[offset: offset + len(cand)]
            offset += len(cand)
            if not np.isfinite(s).all():
                raise ValueError("non-finite model score")
            if EN_RENAME3 and not q.korean:
                s = rename_rule(q, cand, feats, s, self)
            elif EN_RENAME2 and not q.korean:
                s = rename_choice(q, cand, feats, s, self)
            elif self.rename_override is not None and (q.korean or EN_RENAME):
                s = self.rename_override(feats, s)
            if EN_MISSING_TON_EXACT and not q.korean and not q.tons:
                ops = feats[:, self.mm.ABS_FEATURES.index("ops")]
                if (ops == 0).any():
                    s = np.where(ops == 0, s, -np.inf)
            if self.digit_filter and q.korean and q.digits:
                same = np.array([self.cat.digits[i] == q.digits for i in cand], dtype=bool)
                if same.any():
                    s = np.where(same, s, -np.inf)
            order = np.lexsort((cand, -s))  # highest score, ties -> lowest catalog index
            best = order[0]
            if self.is_ko and q.korean and len(order) >= 2:
                self.ties.append((pos, self.cat.ids[cand[order[0]]], self.cat.ids[cand[order[1]]],
                                  float(s[order[0]] - s[order[1]])))
            if self.tiebreak and not q.korean:
                best = twin_tiebreak(self, q, cand, feats, order)
            out[pos] = self.cat.ids[cand[best]]


def rename_choice(q, cand: np.ndarray, X_abs: np.ndarray, scores: np.ndarray, pred: "Predictor") -> np.ndarray:
    """Generalised rename rule for a Latin name that no candidate explains with
    legal typo edits (in train: exactly the 13 renamed rows).  Among sisters
    (catalog GT within 0.3 % of the port tonnage), a rare query word held by
    exactly one sister identifies the ship (port 'OCEAN WREN' <- catalog 'XY Wren'); a query
    word shared by several sisters is a series name, so the ship is a sister
    that left the series (port 'GRAND PARK' <- a sister outside the '... Park' series)."""
    col = {n: i for i, n in enumerate(pred.mm.ABS_FEATURES)}
    if len(cand) == 0 or np.nanmin(X_abs[:, col["ops"]]) < 9:
        return scores
    dev = X_abs[:, col["abs_log_min"]]
    sis = np.flatnonzero(np.where(np.isfinite(dev), dev <= SISTER_TOL, False))
    if len(sis) == 0:
        return scores
    cat = pred.cat
    qset = {w for ws in q.word_variants for w in ws if len(w) >= 3 and w.isalpha()}
    words_of = {i: set(cat.rwords[cand[i]]) for i in sis}
    holders = {w: [i for i in sis if w in words_of[i]] for w in qset}
    unique = {h[0] for w, h in holders.items() if len(h) == 1 and cat.word_df.get(w, 0) <= RARE_DF}
    if len(unique) == 1:
        boost = unique
    elif any(len(h) >= 2 for h in holders.values()):
        boost = {i for i in sis if not (words_of[i] & qset)}
    else:
        return scores
    if not boost:
        return scores
    mask = np.zeros(len(scores), dtype=bool)
    mask[list(boost)] = True
    return np.where(mask, scores + 1000.0, scores)


def rename_rule(q, cand: np.ndarray, X_abs: np.ndarray, scores: np.ndarray, pred: "Predictor") -> np.ndarray:
    """Tonnage-aware rename rule for a Latin port name.

    The 13 renamed train rows are untouched real records: the port tonnage is
    the catalog GT (11 exactly, 2 within 0.05 %), the flag is the usual port
    label of the catalog flag, and no typo is added.  Every other train row is
    explained by legal typo edits with a tonnage within 0.0915 (log) of the
    catalog GT.

    1. A candidate whose tonnage is outside that range cannot be the ship, so
       it drops below every plausible candidate.
    2. If no plausible candidate explains the name with legal typo edits, the
       row is a rename.  Prefer a sister (GT within 0.3 %) that keeps the
       ship's own name -- fleet prefix swapped ('OCEAN WREN' <- 'XY Wren'),
       dropped or added -- with a consistent flag (or, as in v3, a rare own
       name with any flag).
    3. Otherwise, when no sister shares a word with the port name, prefer the
       one exact-tonnage, flag-consistent candidate with the best type
       evidence, if it is unique.

    With RENAME_PLAUSIBLE, RENAME_OWN, RENAME_UNIQUE and EN_SERIES all off
    this is exactly the v3 rename override."""
    col = {n: i for i, n in enumerate(pred.mm.ABS_FEATURES)}
    if len(cand) == 0:
        return scores
    dev = X_abs[:, col["abs_log_min"]]
    known = np.isfinite(dev)
    plausible = np.where(known, dev <= PLAUSIBLE_DEV, True) if RENAME_PLAUSIBLE else np.ones(len(cand), bool)
    if plausible.any() and not plausible.all():
        scores = np.where(plausible, scores, scores - 2000.0)
    ops = X_abs[:, col["ops"]]
    if np.nanmin(np.where(plausible, ops, 9.0)) < 9:
        return scores
    sister = np.where(known, dev <= SISTER_TOL, True)
    exact = known & (dev <= EXACT_DEV)
    fm = X_abs[:, col["flag_match"]]
    flag_ok = np.where(np.isfinite(fm), fm == 1, True)
    tail_swap = X_abs[:, col["tail_swap"]] == 1
    shares = X_abs[:, col["word_eq_frac"]] > 0
    if EN_SERIES and (tail_swap & exact & flag_ok).sum() >= 2:
        # the port name is a series name held by several exact sisters: the ship
        # left the series (port 'GRAND PARK' <- a sister outside the '... Park' series)
        left = exact & flag_ok & ~shares
        if left.any():
            return np.where(left, scores + 1000.0, scores)
    rare = tail_swap & (np.nan_to_num(X_abs[:, col["tail_df"]], nan=1e9) <= RARE_DF)
    pick = sister & rare
    if RENAME_OWN:
        own = tail_swap.copy()
        for k, i in enumerate(cand):
            cw = pred.cat.rwords[i]
            for qw in q.word_variants:
                if (len(cw) >= 2 and cw[1:] == qw) or (len(qw) >= 2 and qw[1:] == cw):
                    own[k] = True
        pick |= sister & own & flag_ok
    if pick.any():
        return np.where(pick, scores + 1000.0, scores)
    if not RENAME_UNIQUE:
        return scores
    cand_exact = exact & flag_ok
    if not cand_exact.any() or (shares & sister).any():
        return scores
    tl = X_abs[:, col["type_logp"]]
    best = cand_exact & (tl >= tl[cand_exact].max() - 0.01)
    if best.sum() == 1:
        return np.where(best, scores + 500.0, scores)
    return scores


def _ops_with_prob(word: str) -> dict[str, float]:
    """Single typo operations on one word with their generator probability:
    op type uniform (1/3), position uniform among the interior positions."""
    n = len(word)
    out: dict[str, float] = {}
    dels = [i for i in range(1, n - 1) if word[i].isalpha()]
    dbls = [i for i in range(1, n - 1) if word[i].isalpha()]
    swps = [i for i in range(1, n - 2) if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]]
    for group, make in ((dels, lambda i: word[:i] + word[i + 1:]),
                        (dbls, lambda i: word[: i + 1] + word[i] + word[i + 1:]),
                        (swps, lambda i: word[:i] + word[i + 1] + word[i] + word[i + 2:])):
        for i in group:
            v = make(i)
            out[v] = out.get(v, 0.0) + 1.0 / (3.0 * len(group))
    return out


@functools.lru_cache(maxsize=100_000)
def _word_path_prob(query_word: str, catalog_word: str) -> float:
    """Probability that one or two generator operations turn the catalog word
    into the query word (summed over all operation paths)."""
    if query_word == catalog_word:
        return 1.0
    first = _ops_with_prob(catalog_word)
    if query_word in first:
        return first[query_word]
    total = 0.0
    for mid, p in first.items():
        total += p * _ops_with_prob(mid).get(query_word, 0.0)
    return total


def path_likelihood(query_words: list[str], catalog_words: list[str]) -> float:
    if len(query_words) != len(catalog_words):
        return 0.0
    elig = max(sum(1 for w in catalog_words if len(w) >= 6 and not w.isdigit()), 1)
    p = 1.0
    for qw, cw in zip(query_words, catalog_words):
        if qw != cw:
            p *= _word_path_prob(qw, cw) / elig
    return p


def twin_tiebreak(pred: "Predictor", q, cand: np.ndarray, feats: np.ndarray, order: np.ndarray) -> int:
    """Among the top candidate and its indistinguishable twins (same catalog
    GT, type, flag and the same number of legal typo operations), prefer the
    one whose name is the likelier generator output; keep the model order
    on equal likelihood."""
    col = pred.mm.ABS_FEATURES.index("ops")
    top = order[0]
    ops_top = feats[top, col]
    if not (0 < ops_top < 9):
        return top
    i0 = cand[top]
    cat = pred.cat
    twins = [j for j in order[:20]
             if feats[j, col] == ops_top and cat.gt[cand[j]] == cat.gt[i0]
             and cat.type[cand[j]] == cat.type[i0] and cat.flag[cand[j]] == cat.flag[i0]]
    if len(twins) < 2:
        return top
    best, best_p = top, -1.0
    for j in twins:  # model order, so equal likelihood keeps the model's choice
        p = max(path_likelihood(qw, cat.rwords[cand[j]]) for qw in q.word_variants)
        if p > best_p * (1 + 1e-9):
            best, best_p = j, p
    return best


class EnsemblePredictor:
    """Two Hangul predictors over one candidate set (the union of theirs):
    the per-query softmax probabilities of the two rankers are averaged."""

    def __init__(self, a: "Predictor", b: "Predictor", weight: float):
        self.a, self.b, self.w = a, b, weight

    @staticmethod
    def _scores(pred: "Predictor", feats: np.ndarray) -> np.ndarray:
        X = pred.mm.with_relative(feats, np.zeros(len(feats), dtype=np.int32))
        s = np.mean([bst.predict(X, num_threads=1) for bst in pred.boosters], axis=0)
        if not np.isfinite(s).all():
            raise ValueError("non-finite model score")
        return s

    @staticmethod
    def _prob(s: np.ndarray) -> np.ndarray:
        e = np.exp(s - s.max())
        return e / e.sum()

    def predict_records(self, records: list[dict]) -> list[str]:
        a, b = self.a, self.b
        out: list[str] = [""] * len(records)
        for pos, row in enumerate(records):
            qa = a.mm.Query(row, a.ref)
            qb = b.mm.Query(row, b.ref)
            hit = a.direct(qa, row)
            if hit is not None:
                out[pos] = hit
                continue
            cand = np.union1d(a.candidate_set(qa), b.candidate_set(qb))
            p = (1 - self.w) * self._prob(self._scores(a, a.matcher.features(qa, cand))) \
                + self.w * self._prob(self._scores(b, b.matcher.features(qb, cand)))
            best = np.lexsort((cand, -p))[0]
            out[pos] = a.cat.ids[cand[best]]
        return out


def _work(task: tuple[str, list[dict]]) -> tuple[list[str], list[tuple[int, str, str, float]]]:
    """-> (answers, near-tie info of the chunk's Hangul ranker rows)."""
    key, records = task
    pred = PREDICTORS[key]
    out = pred.predict_records(records)
    return out, list(getattr(pred, "ties", []))


def _llm_words(name) -> set:
    """Lower-case alphanumeric tokens of a catalog name (subset guard of the LLM judge)."""
    import re as _re
    return set(_re.findall(r"[a-z0-9]+", str(name).lower()))


def llm_judge(test: pd.DataFrame, catalog: pd.DataFrame, preds: np.ndarray, ties: list) -> None:
    """Re-rank Hangul near-ties in place with the LLM spelling judge (see the
    module docstring).  Runs in the main process after the worker pool, so
    torch is never imported before a fork.  Any failure keeps the answers."""
    rows = [t for t in ties if t[3] < LLM_TAU]
    log = lambda m: print(f"[llm-judge] {m}", file=sys.stderr, flush=True)  # noqa: E731
    log(f"Hangul ranker rows {len(ties)}, near-ties (margin < {LLM_TAU}) {len(rows)}")
    if not LLM_JUDGE or not rows:
        return
    if time.time() - T_START > LLM_DEADLINE:
        log("skipped: past the deadline guard")
        return
    if not (LLM_MODEL_DIR / "config.json").exists():
        log(f"skipped: no model at {LLM_MODEL_DIR}")
        return
    t0 = time.time()
    try:
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("v36_llm_judge", str(MODEL / "llm" / "judge.py"))
        jm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jm)
        if not jm.SHOTS:  # public copy: the ten few-shot pairs are rebuilt from the package's train.csv
            jm.SHOTS[:] = jm.shots_from_frames(pd.read_csv(MODEL / "train.csv"), catalog)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            torch.set_num_threads(os.cpu_count() or 1)
        judge = jm.Judge(LLM_MODEL_DIR, LLM_FORM, device)
        log(f"loaded {LLM_MODEL_DIR.name} on {device} in {time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001 - no torch / transformers / weights -> v29v answers
        log(f"skipped: {type(exc).__name__}: {exc}")
        return
    names = dict(zip(catalog.vessel_id.astype(str), catalog.vessel_name))
    judged = swapped = guarded = 0
    t1 = time.time()
    try:
        for i, id1, id2, margin in rows[:LLM_MAX_ROWS]:
            if time.time() - t0 > LLM_TIME_BUDGET or time.time() - T_START > LLM_DEADLINE:
                log("time guard reached; remaining rows keep rank 1")
                break
            ko, en1, en2 = test.vessel_name.iat[i], names.get(id1), names.get(id2)
            if not (isinstance(ko, str) and isinstance(en1, str) and isinstance(en2, str)):
                continue
            if preds[i] != id1:
                continue  # defensive: the answer is not the ranker's rank 1
            if LLM_SUBSET_GUARD and _llm_words(en2) < _llm_words(en1):
                guarded += 1
                continue  # the judge is uninformative here: it always prefers the name without the dropped words
            l1, _ = judge.score(en1, ko)
            l2, _ = judge.score(en2, ko)
            judged += 1
            if l2 - l1 > LLM_KAPPA:
                preds[i] = id2
                swapped += 1
                log(f"swap {test.sample_id.iat[i]} {ko!r}: {en1!r} -> {en2!r} (margin {margin:.2f}, l2-l1 {l2 - l1:+.2f})")
    except Exception as exc:  # noqa: BLE001 - keep what was decided so far (each row independently)
        log(f"stopped: {type(exc).__name__}: {exc}")
    log(f"rows judged {judged}, rows swapped {swapped}, subset-guarded {guarded}, {time.time() - t1:.1f}s scoring")


def n_workers() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, cpus))


def run_tasks(tasks: list[tuple[str, list[dict]]]) -> list[tuple[list[str], list]]:
    workers = n_workers()
    if workers > 1 and len(tasks) > 1 and "fork" in mp.get_all_start_methods():
        try:
            with mp.get_context("fork").Pool(workers) as pool:
                return pool.map_async(_work, tasks, chunksize=1).get(timeout=POOL_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - any pool failure -> sequential
            print(f"parallel prediction failed ({type(exc).__name__}); running sequentially", flush=True)
    return [_work(t) for t in tasks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--output", default=str(HERE / "output" / "submission.csv"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    if NFC_NAMES and "vessel_name" in test.columns:
        import unicodedata
        test["vessel_name"] = test.vessel_name.map(
            lambda v: unicodedata.normalize("NFC", v) if isinstance(v, str) else v)
    if "sample_id" not in test.columns or test.sample_id.isna().any() or not test.sample_id.is_unique:
        raise ValueError("test.csv needs unique, non-null sample_id")
    catalog = read_csv(data_dir / "vessel_catalog.csv", MODEL / "vessel_catalog.csv")
    train = read_csv(MODEL / "train.csv", data_dir / "train.csv")
    for col in ("vessel_name", "call_sign", "gross_tonnage", "international_gross_tonnage",
                "flag_name", "vessel_type"):
        if col not in test.columns:
            test[col] = np.nan

    ko_text = importlib.import_module("vm_ko.text")
    hangul = test.vessel_name.map(ko_text.is_korean).to_numpy(dtype=bool)
    groups = {"ko": np.flatnonzero(hangul), "en": np.flatnonzero(~hangul)}
    preds = np.empty(len(test), dtype=object)
    if PROBE_DUMMY in groups:
        preds[groups[PROBE_DUMMY]] = str(catalog.vessel_id.iloc[0])
        groups[PROBE_DUMMY] = groups[PROBE_DUMMY][:0]
    if len(groups["ko"]):
        PREDICTORS["ko"] = Predictor("vm_ko", MODEL / "ko", catalog, train, exclude=KO_EXCLUDE,
                                     filters=KO_FILTERS, digit_filter=KO_DIGIT_FILTER,
                                     exclude_mode=KO_EXCLUDE_MODE, alias=KO_ALIAS,
                                     eligible=KO_ELIGIBLE, ton_gate=KO_TON_GATE, ascii_fold=KO_ASCII_FOLD,
                                     single_letters=KO_SINGLE_LETTERS)
        if KO_ENSEMBLE > 0:
            sys.path.insert(0, str(MODEL / "ko2"))
            second = Predictor("vm_ko2", MODEL / "ko2", catalog, train, exclude=KO_EXCLUDE,
                               filters=KO_FILTERS, exclude_mode=KO_EXCLUDE_MODE, alias=KO_ALIAS,
                               eligible=KO_ELIGIBLE, ton_gate=KO_TON_GATE)
            PREDICTORS["ko"] = EnsemblePredictor(PREDICTORS["ko"], second, KO_ENSEMBLE)
    if len(groups["en"]):
        PREDICTORS["en"] = Predictor("vm_en", MODEL / "en", catalog, train, exclude=EN_EXCLUDE,
                                     filters=EN_FILTERS, tiebreak=EN_TIEBREAK, ascii_fold=EN_ASCII_FOLD,
                                     exclude_mode=EN_EXCLUDE_MODE, alias=EN_ALIAS,
                                     eligible=EN_ELIGIBLE, ton_gate=EN_TON_GATE)

    records = test.to_dict("records")
    tasks, positions = [], []
    n_chunks = 4 * n_workers()
    for key, idx in groups.items():
        for part in np.array_split(idx, min(len(idx), n_chunks)) if len(idx) else []:
            tasks.append((key, [records[i] for i in part]))
            positions.append(part)
    results = run_tasks(tasks)
    ties = []  # (test row index, rank-1 id, rank-2 id, margin), Hangul ranker rows
    for part, (res, part_ties) in zip(positions, results):
        preds[part] = res
        ties.extend((int(part[j]), a, b, m) for j, a, b, m in part_ties)
    ties.sort()
    try:
        llm_judge(test, catalog, preds, ties)
    except Exception as exc:  # noqa: BLE001 - never fail the submission because of the judge
        print(f"[llm-judge] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    valid = set(catalog.vessel_id.astype(str))
    if any(p not in valid for p in preds):
        raise ValueError("invalid predictions")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test.sample_id, "target_vessel_id": preds}).to_csv(
        out, index=False, encoding="utf-8")
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    main()
