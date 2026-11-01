"""Abbreviated-name evidence (v14).

The 10/19 evaluation rows carry names that the train typo model does not
explain; the task description names abbreviations ("약칭") among the
differences.  Two families of evidence, both per (query, candidate):

* word alignment (Latin names): every query word, in order, matches a
  distinct candidate word exactly, with a small typo, as a prefix or as its
  initial; candidate words may be skipped (a dropped word).
* span keys (both scripts): the phonetic key of the query against the key of
  the best contiguous span of candidate words (a Hangul name that keeps
  only part of the ship's name).
"""
from __future__ import annotations

import functools

from rapidfuzz.distance import OSA, Levenshtein

ABBR_FEATURES = (
    "ab_cost", "ab_unq", "ab_skip", "ab_init", "ab_pref", "ab_full",
    "sp_key_sim", "sp_cons_sim", "sp_cov", "sp_gain", "sp_first", "sp_last",
)
_INF = 99.0
SKIP_COST = 1.0
UNMATCHED_COST = 3.0


@functools.lru_cache(maxsize=500_000)
def word_match(q: str, c: str) -> tuple[float, int]:
    """(cost, kind) of query word q standing for catalog word c.
    kind: 0 exact, 1 typo, 2 prefix, 3 initial, -1 none."""
    if q == c:
        return 0.0, 0
    if len(q) == 1:
        return (1.0, 3) if c[:1] == q and not q.isdigit() else (_INF, -1)
    if q.isdigit() or c.isdigit():
        return _INF, -1
    if len(q) >= 2 and len(q) < len(c) and c.startswith(q):
        return 0.6, 2
    if len(q) >= 4 and len(c) >= 4:
        d = OSA.distance(q, c, score_cutoff=2)
        if d <= 2:
            return 0.4 * d, 1
    if 2 <= len(q) < len(c) and q[0] == c[0] and q.isalpha() and _subsequence(q, c):
        return 0.8, 2  # consonant / clipped abbreviation (INTL <- INTERNATIONAL)
    return _INF, -1


def _subsequence(q: str, c: str) -> bool:
    it = iter(c)
    return all(ch in it for ch in q)


@functools.lru_cache(maxsize=500_000)
def align(qw: tuple[str, ...], cw: tuple[str, ...]) -> tuple[float, int, int, int, int]:
    """Monotone alignment of query words onto candidate words.
    Returns (cost, unmatched query words, skipped candidate words,
    initials used, prefixes used)."""
    n, m = len(qw), len(cw)
    if n == 0 or m == 0 or n > 8 or m > 10:
        return _INF, n, m, 0, 0
    # dp[i][j]: best over first i query words and first j candidate words
    INF = (1e9, 0, 0, 0, 0)
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0.0, 0, 0, 0, 0)
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur[0] >= 1e9:
                continue
            if j < m:  # skip candidate word j
                cand = (cur[0] + SKIP_COST, cur[1], cur[2] + 1, cur[3], cur[4])
                if cand < dp[i][j + 1]:
                    dp[i][j + 1] = cand
            if i < n:  # query word i unmatched
                cand = (cur[0] + UNMATCHED_COST, cur[1] + 1, cur[2], cur[3], cur[4])
                if cand < dp[i + 1][j]:
                    dp[i + 1][j] = cand
            if i < n and j < m:
                cost, kind = word_match(qw[i], cw[j])
                if kind >= 0:
                    cand = (cur[0] + cost, cur[1], cur[2], cur[3] + (kind == 3), cur[4] + (kind == 2))
                    if cand < dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = cand
    return dp[n][m]


def span_keys(words: list[str], word_key) -> list[tuple[str, int, int]]:
    """(key, start, end) for every contiguous span of at most 4 words."""
    keys = [word_key(w) for w in words]
    out = []
    for a in range(len(words)):
        for b in range(a + 1, min(len(words), a + 4) + 1):
            out.append(("".join(keys[a:b]), a, b))
    return out


def consonants(key: str) -> str:
    return "".join(ch for ch in key if ch not in "AEIOUY")


def span_features(qkeys: list[str], spans: list[tuple[str, int, int]], n_words: int,
                  full_sim: float) -> tuple[float, float, float, float, float, float]:
    best = (-1.0, 0.0, 0, 0)
    for qk in qkeys:
        if not qk:
            continue
        for key, a, b in spans:
            if not key:
                continue
            s = Levenshtein.normalized_similarity(qk, key)
            if s > best[0] or (s == best[0] and b - a > best[3] - best[2]):
                best = (s, Levenshtein.normalized_similarity(consonants(qk), consonants(key)), a, b)
    if best[0] < 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    s, cs, a, b = best
    return s, cs, (b - a) / max(n_words, 1), s - full_sim, float(a == 0), float(b == n_words)
