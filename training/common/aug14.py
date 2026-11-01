"""Name perturbations for v14 training copies (train rows only)."""
from __future__ import annotations

import numpy as np

import stress


def _hangul(s: str) -> bool:
    return any("가" <= c <= "힣" for c in s)


# -- Latin abbreviations ------------------------------------------------------
def en_abbr(name: str, rng) -> str:
    ws = str(name).upper().split()
    alpha = [k for k, w in enumerate(ws) if w.isalpha() and len(w) >= 2]
    r = rng.rand()
    if len(ws) >= 2 and alpha:
        if r < 0.18:      # one word -> its initial
            k = alpha[rng.randint(len(alpha))]
            ws[k] = ws[k][0] + ("." if rng.rand() < 0.2 else "")
        elif r < 0.30:    # first word -> initial
            if ws[0].isalpha():
                ws[0] = ws[0][0]
        elif r < 0.50:    # a word dropped
            del ws[rng.randint(len(ws))]
        elif r < 0.62:    # last word dropped
            ws = ws[:-1]
        elif r < 0.74:    # a long word cut short
            long = [k for k in alpha if len(ws[k]) >= 6]
            if long:
                k = long[rng.randint(len(long))]
                ws[k] = ws[k][: 3 + rng.randint(3)]
        elif r < 0.84 and len(ws) >= 3:  # leading words -> acronym
            n = 2 + rng.randint(len(ws) - 2)
            lead = "".join(w[0] for w in ws[:n])
            ws = ([lead] if rng.rand() < 0.7 else list(lead)) + ws[n:]
        elif r < 0.92:    # vowels dropped from a word
            long = [k for k in alpha if len(ws[k]) >= 5]
            if long:
                k = long[rng.randint(len(long))]
                ws[k] = ws[k][0] + "".join(c for c in ws[k][1:] if c not in "AEIOU")
        else:             # two abbreviations
            k = alpha[rng.randint(len(alpha))]
            ws[k] = ws[k][0]
            if len(ws) >= 3:
                del ws[rng.randint(len(ws))]
    else:
        if ws and len(ws[0]) >= 6 and r < 0.5:
            ws[0] = ws[0][: 3 + rng.randint(3)]
        elif ws and len(ws[0]) >= 5:
            ws[0] = ws[0][0] + "".join(c for c in ws[0][1:] if c not in "AEIOU")
    s = " ".join(ws) if ws else str(name).upper()
    if rng.rand() < 0.35:
        s = stress._char_edit(s, rng)
    return s


# -- Hangul abbreviations -----------------------------------------------------
def ko_abbr(name: str, rng) -> str:
    ws = str(name).split()
    r = rng.rand()
    if len(ws) >= 2:
        if r < 0.35:
            ws = ws[1:]
        elif r < 0.65:
            ws = ws[:-1]
        elif r < 0.8 and len(ws) >= 3:
            del ws[1 + rng.randint(len(ws) - 2)]
        else:
            ws = [max(ws, key=len)]
    else:
        w = ws[0] if ws else str(name)
        syl = [c for c in w if "가" <= c <= "힣"]
        if len(syl) >= 5:  # an unspaced name: keep the tail or the head
            cut = 2 + rng.randint(len(w) - 3)
            ws = [w[cut:]] if rng.rand() < 0.5 else [w[:cut]]
        else:
            ws = [w]
    s = " ".join(ws)
    if not _hangul(s):
        s = str(name)
    if rng.rand() < 0.35:
        s = stress._ko_edit(s, rng)
    return s


def make(kind: str, name: str, rng) -> str:
    if kind == "orig":
        return name
    if kind == "en_noise":
        return stress.en_noise(name, rng)
    if kind == "en_heavy":
        return stress.en_noise_heavy(name, rng)
    if kind == "en_abbr":
        return en_abbr(name, rng)
    if kind == "ko_noise":
        return stress.ko_noise(name, rng)
    if kind == "ko_heavy":
        return stress.ko_noise_heavy(name, rng)
    if kind == "ko_abbr":
        return ko_abbr(name, rng)
    if kind == "ko_nosp":
        return str(name).replace(" ", "")
    raise ValueError(kind)


# -- Hangul abbreviations as the Latin string operations would make them -----
def _syl_words(name: str) -> list[str]:
    return str(name).split()


def ko_init(name: str, rng) -> str:
    """One word -> its first character (syllable)."""
    ws = _syl_words(name)
    idx = [k for k, w in enumerate(ws) if len(w) >= 2 and "가" <= w[0] <= "힣"]
    if len(ws) >= 2 and idx:
        k = idx[rng.randint(len(idx))]
        ws[k] = ws[k][0]
        return " ".join(ws)
    return str(name)


def ko_trunc(name: str, rng) -> str:
    """One long word cut to its first 1-3 syllables."""
    ws = _syl_words(name)
    idx = [k for k, w in enumerate(ws) if len(w) >= 3]
    if idx:
        k = idx[rng.randint(len(idx))]
        ws[k] = ws[k][: 1 + rng.randint(min(3, len(ws[k]) - 1))]
        return " ".join(ws)
    return str(name)


def ko_acro(name: str, rng) -> str:
    """Leading words -> their first syllables joined."""
    ws = _syl_words(name)
    if len(ws) >= 2:
        n = 2 + rng.randint(len(ws) - 1) if len(ws) > 2 else 2
        n = min(n, len(ws))
        lead = "".join(w[0] for w in ws[:n])
        rest = ws[n:]
        if not rest and rng.rand() < 0.5:
            return " ".join([lead[:-1], ws[n - 1]])
        return " ".join([lead] + rest)
    return str(name)


def ko_abbr2(name: str, rng) -> str:
    r = rng.rand()
    if r < 0.4:
        s = ko_init(name, rng)
    elif r < 0.75:
        s = ko_trunc(name, rng)
    elif r < 0.9:
        s = ko_acro(name, rng)
    else:
        s = ko_abbr(name, rng)
    if rng.rand() < 0.25:
        s = stress._ko_edit(s, rng)
    return s


_make_v14 = make


def make(kind: str, name: str, rng) -> str:  # noqa: F811
    if kind == "ko_init":
        return ko_init(name, rng)
    if kind == "ko_trunc":
        return ko_trunc(name, rng)
    if kind == "ko_acro":
        return ko_acro(name, rng)
    if kind == "ko_abbr2":
        return ko_abbr2(name, rng)
    return _make_v14(kind, name, rng)


KO_LETTER_NAME = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
                  "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
                  "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유",
                  "X": "엑스", "Y": "와이", "Z": "지"}
_CAT_NAMES = None


def _catalog_name(target: str) -> str:
    global _CAT_NAMES
    if _CAT_NAMES is None:
        import pandas as pd
        from pathlib import Path
        df = pd.read_csv(Path(__file__).resolve().parent / "vessel_catalog.csv", usecols=["vessel_id", "vessel_name"])
        _CAT_NAMES = dict(zip(df.vessel_id, df.vessel_name))
    return str(_CAT_NAMES.get(target, ""))


def ko_letter_init(name: str, rng, target: str = "") -> str:
    """A Hangul word -> the Korean letter name of its English word's initial (M. CROWN -> 엠 크라운)."""
    ws = str(name).split()
    en = [w for w in _catalog_name(target).upper().replace(".", " ").split() if w.isalpha()]
    if len(ws) >= 2 and len(ws) == len(en):
        k = rng.randint(len(ws))
        ws[k] = KO_LETTER_NAME.get(en[k][0], ws[k])
        return " ".join(ws)
    return str(name)
