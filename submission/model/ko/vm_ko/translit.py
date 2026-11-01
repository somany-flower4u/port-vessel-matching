"""Learned Korean (jamo) -> Latin (letters) transliteration scorer.

A stochastic transducer: every Korean unit (initial consonant, vowel, final
consonant, digit or Latin letter) emits a Latin substring of length 0..3;
extra Latin letters may be inserted.  Emission weights are fitted by EM on the
official train pairs (Hangul port name, catalog name) only.  Scoring a
candidate is a forward pass, vectorised over many catalog names at once.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import text as T

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CODE = {ch: k + 1 for k, ch in enumerate(ALPHA)}  # 0 = padding
B = len(ALPHA) + 1  # 37
OFF2 = B
OFF3 = OFF2 + B * B
EMPTY = OFF3 + B * B * B
VOCAB = EMPTY + 1
MAXLEN = 40

# Heuristic readings used as a Dirichlet prior for EM (not hard rules).
PRIOR = {
    "Iㄱ": "G K C Q GH CK", "Iㄲ": "K C G KK CK Q", "Iㄴ": "N NN KN", "Iㄷ": "D T DD TH",
    "Iㄸ": "T D TT", "Iㄹ": "R L LL RR _", "Iㅁ": "M MM", "Iㅂ": "B V P BB W", "Iㅃ": "P PP B",
    "Iㅅ": "S C SH TH SS X Z", "Iㅆ": "S SS C SH Z X", "Iㅈ": "J G Z CH DG TS ZH X DZ GE",
    "Iㅉ": "J Z TZ TS CH", "Iㅊ": "CH C TCH TS J Q CZ X", "Iㅋ": "K C CK Q KH CH",
    "Iㅌ": "T TH TT D", "Iㅍ": "P F PH FF PP", "Iㅎ": "H _ WH J",
    "Vㅏ": "A AR U O AH AA E", "Vㅐ": "A E AE AI AY", "Vㅑ": "YA IA A JA", "Vㅒ": "YE IE YA",
    "Vㅓ": "O U E ER A EO OR UR IR EU", "Vㅔ": "E A AI AY EI AE", "Vㅕ": "YO IO YE IE EO YU",
    "Vㅖ": "YE IE E", "Vㅗ": "O OR AU OO AW OH OA U", "Vㅘ": "WA UA OA GUA",
    "Vㅙ": "WAE WE UE", "Vㅚ": "OE WE OI UE", "Vㅛ": "YO IO EO", "Vㅜ": "U OO OU W O EW",
    "Vㅝ": "WO UO WA WOR", "Vㅞ": "WE UE OE", "Vㅟ": "WI UI UE WEE", "Vㅠ": "YU IU U EW YOU",
    "Vㅡ": "_ E U I EU A O", "Vㅢ": "UI I EE EI", "Vㅣ": "I E EE Y IE EA II",
    "Fㄱ": "K G C CK X", "Fㄴ": "N NN NE", "Fㄷ": "T D", "Fㄹ": "L LL R LE RL", "Fㅁ": "M MM ME",
    "Fㅂ": "P B PP", "Fㅅ": "T S D TH", "Fㅆ": "T S", "Fㅇ": "NG N G _", "Fㅈ": "T J",
    "Fㅊ": "T CH", "Fㅋ": "K", "Fㅌ": "T", "Fㅍ": "P", "Fㅎ": "T H",
    # y-vowels after ㅅ/ㅆ/ㅈ/ㅉ/ㅊ: the glide is usually absorbed (쇼 = SHO, 쥬 = JU)
    "Pㅑ": "A YA IA", "Pㅕ": "O E U YO EO", "Pㅛ": "O YO IO OU", "Pㅠ": "U OO IU YU EW",
    "Pㅒ": "E YE A", "Pㅖ": "E YE",
}
PALATAL_CHO = set("ㅅㅆㅈㅉㅊ")
GLIDE_JUNG = set("ㅑㅕㅛㅠㅒㅖ")


def sub_code(s: str) -> int:
    if s == "":
        return EMPTY
    c = [CODE[ch] for ch in s]
    if len(c) == 1:
        return c[0]
    if len(c) == 2:
        return OFF2 + c[0] * B + c[1]
    return OFF3 + c[0] * B * B + c[1] * B + c[2]


def ko_units(text: str) -> list[str]:
    units: list[str] = []
    for word in T.nfkc_upper(text).split():
        if not T.HANGUL_RE.search(word):
            for ch in "".join(T.words(word)):
                if ch in CODE:
                    units.append(("D" if ch.isdigit() else "L") + ch)
            continue
        for a, b, c in T.syllables(word):
            if b is None:
                if a in CODE:
                    units.append(("D" if a.isdigit() else "L") + a)
                continue
            if a != "ㅇ":
                units.append("I" + a)
            units.append(("P" if a in PALATAL_CHO and b in GLIDE_JUNG else "V") + b)
            if c:
                units.append("F" + c)
    return units


def encode(s: str) -> np.ndarray:
    return np.array([CODE[ch] for ch in s if ch in CODE][:MAXLEN], dtype=np.int64)


def substring_ids(chars: np.ndarray) -> np.ndarray:
    """chars: [n, L] int codes (0 = pad) -> ids [n, L, 4] for lengths 0..3
    starting at each position; -1 where the substring runs into padding."""
    n, L = chars.shape
    out = np.full((n, L + 1, 4), -1, dtype=np.int64)
    out[:, :, 0] = EMPTY
    c = np.concatenate([chars, np.zeros((n, 3), dtype=np.int64)], axis=1)
    ok1 = c[:, :L] > 0
    ok2 = ok1 & (c[:, 1:L + 1] > 0)
    ok3 = ok2 & (c[:, 2:L + 2] > 0)
    out[:, :L, 1] = np.where(ok1, c[:, :L], -1)
    out[:, :L, 2] = np.where(ok2, OFF2 + c[:, :L] * B + c[:, 1:L + 1], -1)
    out[:, :L, 3] = np.where(ok3, OFF3 + c[:, :L] * B * B + c[:, 1:L + 1] * B + c[:, 2:L + 2], -1)
    return out


class Transliterator:
    def __init__(self, prior_weight: float = 0.5, backoff: float = 1e-4, ins: float = 0.02):
        self.prior_weight = prior_weight
        self.backoff = backoff
        self.ins_base = ins
        self.units: dict[str, int] = {}
        self.emis = None  # [n_units, VOCAB]
        self.ins = None  # [B]
        self.bg = None  # [B] background letter log-probabilities

    # -- parameters ------------------------------------------------------------
    def _unit_id(self, u: str) -> int:
        if u not in self.units:
            self.units[u] = len(self.units)
        return self.units[u]

    def _prior_counts(self, u: str) -> dict[int, float]:
        out: dict[int, float] = {}
        if u[0] in "LD":
            out[sub_code(u[1])] = 3.0
            return out
        for s in PRIOR.get(u, "").split():
            s = "" if s == "_" else s
            out[sub_code(s)] = out.get(sub_code(s), 0.0) + 1.0
        return out

    def _normalise(self, counts: dict[int, dict[int, float]]) -> None:
        n = len(self.units)
        emis = np.zeros((n, VOCAB), dtype=np.float64)
        for u, uid in self.units.items():
            row = defaultdict(float)
            for k, v in self._prior_counts(u).items():
                row[k] += self.prior_weight * v
            for k, v in counts.get(uid, {}).items():
                row[k] += v
            total = sum(row.values())
            for k, v in row.items():
                emis[uid, k] = (1 - self.backoff) * v / total if total else 0.0
        # back-off mass for any unseen single letter / empty emission
        emis[:, 1:B] += 0.8 * self.backoff / (B - 1)
        emis[:, EMPTY] += 0.2 * self.backoff
        self.emis = emis

    # -- EM ----------------------------------------------------------------------
    def fit(self, pairs: list[tuple[list[list[str]], str]], iters: int = 12) -> "Transliterator":
        """pairs: (alternative unit sequences, Latin target string)."""
        data = []
        for variants, latin in pairs:
            chars = encode(latin)
            if len(chars) == 0:
                continue
            ids = substring_ids(chars[None, :])[0]
            vs = [[self._unit_id(u) for u in v] for v in variants if v]
            if vs:
                data.append((vs, chars, ids))
        for u in list(PRIOR):
            self._unit_id(u)
        for ch in CODE:
            self._unit_id("L" + ch) if not ch.isdigit() else self._unit_id("D" + ch)
        letter_counts = np.ones(B)
        for _, chars, _ in data:
            np.add.at(letter_counts, chars, 1)
        letter_counts[0] = 0
        self.bg = np.log(letter_counts / letter_counts.sum() + 1e-12)
        self.ins = np.full(B, self.ins_base)
        self.ins[0] = 0.0
        self._normalise({})
        for _ in range(iters):
            counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
            ins_counts = np.zeros(B)
            for vs, chars, ids in data:
                best = None
                for v in vs:
                    res = self._forward_backward(v, chars, ids)
                    if res is not None and (best is None or res[0] > best[0]):
                        best = res
                if best is None:
                    continue
                _, em_counts, ic = best
                for (uid, sid), val in em_counts.items():
                    counts[uid][sid] += val
                ins_counts += ic
            self._normalise(counts)
            total_chars = sum(len(c) for _, c, _ in data)
            self.ins = np.clip(ins_counts / max(total_chars, 1), 0.002, 0.05)
            self.ins[0] = 0.0
        return self

    def _forward_backward(self, units: list[int], chars: np.ndarray, ids: np.ndarray):
        m, n = len(units), len(chars)
        emis, ins = self.emis, self.ins
        ins_c = ins[chars]
        alpha = np.zeros((m + 1, n + 1))
        alpha[0, 0] = 1.0
        for j in range(n):
            alpha[0, j + 1] += alpha[0, j] * ins_c[j]
        for i in range(m):
            e = emis[units[i]]
            row = alpha[i]
            nxt = alpha[i + 1]
            for l in range(4):
                sid = ids[: n + 1 - l, l]
                valid = sid >= 0
                w = np.where(valid, e[np.where(valid, sid, 0)], 0.0)
                nxt[l:n + 1] += row[: n + 1 - l] * w
            for j in range(n):
                nxt[j + 1] += nxt[j] * ins_c[j]
        p = alpha[m, n]
        if not p > 0:
            return None
        beta = np.zeros((m + 1, n + 1))
        beta[m, n] = 1.0
        for j in range(n - 1, -1, -1):
            beta[m, j] = ins_c[j] * beta[m, j + 1]
        for i in range(m - 1, -1, -1):
            e = emis[units[i]]
            b = beta[i]
            for l in range(4):
                sid = ids[: n + 1 - l, l]
                valid = sid >= 0
                w = np.where(valid, e[np.where(valid, sid, 0)], 0.0)
                b[: n + 1 - l] += w * beta[i + 1, l:n + 1]
            for j in range(n - 1, -1, -1):
                b[j] += ins_c[j] * b[j + 1]
        em_counts: dict[tuple[int, int], float] = defaultdict(float)
        for i in range(m):
            e = emis[units[i]]
            for l in range(4):
                sid = ids[: n + 1 - l, l]
                valid = sid >= 0
                w = np.where(valid, e[np.where(valid, sid, 0)], 0.0)
                post = alpha[i, : n + 1 - l] * w * beta[i + 1, l:n + 1] / p
                for j in np.flatnonzero(post > 1e-6):
                    em_counts[(units[i], int(sid[j]))] += float(post[j])
        ic = np.zeros(B)
        for i in range(m + 1):
            post = alpha[i, :n] * ins_c * beta[i, 1:n + 1] / p
            np.add.at(ic, chars, post)
        return math.log(p), em_counts, ic

    # -- scoring -------------------------------------------------------------------
    def score(self, units: list[str], chars: np.ndarray, ids: np.ndarray, lengths: np.ndarray):
        """Log P(latin | korean) and length-normalised log-likelihood ratio
        against a letter-unigram background, for a batch of Latin strings.
        chars [n, L], ids [n, L+1, 4], lengths [n]."""
        n, L = chars.shape
        uid = [self.units.get(u) for u in units]
        alpha = np.zeros((n, L + 1))
        alpha[:, 0] = 1.0
        ins_c = self.ins[chars]
        for j in range(L):
            alpha[:, j + 1] += alpha[:, j] * ins_c[:, j]
        scale = np.zeros(n)
        for u in uid:
            nxt = np.zeros_like(alpha)
            if u is None:  # unseen unit: allow it to emit nothing or one letter
                e = np.zeros(VOCAB)
                e[EMPTY] = 0.5
                e[1:B] = 0.5 / (B - 1)
            else:
                e = self.emis[u]
            for l in range(4):
                sid = ids[:, : L + 1 - l, l]
                w = np.where(sid >= 0, e[np.maximum(sid, 0)], 0.0)
                nxt[:, l:] += alpha[:, : L + 1 - l] * w
            for j in range(L):
                nxt[:, j + 1] += nxt[:, j] * ins_c[:, j]
            s = nxt.max(axis=1)
            s = np.where(s > 0, s, 1.0)
            nxt /= s[:, None]
            scale += np.log(s)
            alpha = nxt
        p = alpha[np.arange(n), lengths]
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) + scale, -1e4)
        bg = np.where(chars > 0, self.bg[chars], 0.0).sum(axis=1)
        return logp, logp - bg
