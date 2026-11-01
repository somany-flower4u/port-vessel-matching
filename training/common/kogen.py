"""Hangul names for catalog vessels (Korean-flagged domestic population).

romanised Korean words  -> Hangul syllables (RR + common informal spellings)
acronyms (KMTC, MS)     -> Korean letter names (케이엠티씨, 엠에스)
English words           -> the Hangul word train uses for them (OCEAN -> 오션)
numbers / No.           -> digits before or after the name, optional 호
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from vm_ko import text as T  # noqa: E402

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
# (romanisation, cost): first entry is Revised Romanisation
CHO_R = {"ㄱ": [("G", 0), ("K", .2)], "ㄲ": [("KK", 0), ("GG", .3)], "ㄴ": [("N", 0)], "ㄷ": [("D", 0), ("T", .2)],
         "ㄸ": [("TT", 0), ("DD", .3)], "ㄹ": [("R", 0), ("L", .2)], "ㅁ": [("M", 0)], "ㅂ": [("B", 0), ("P", .2)],
         "ㅃ": [("PP", 0)], "ㅅ": [("S", 0), ("SH", .2)], "ㅆ": [("SS", 0), ("S", .3)], "ㅇ": [("", 0)],
         "ㅈ": [("J", 0), ("CH", .3)], "ㅉ": [("JJ", 0)], "ㅊ": [("CH", 0)], "ㅋ": [("K", 0)], "ㅌ": [("T", 0)],
         "ㅍ": [("P", 0)], "ㅎ": [("H", 0)]}
JUNG_R = {"ㅏ": [("A", 0)], "ㅐ": [("AE", 0)], "ㅑ": [("YA", 0)], "ㅒ": [("YAE", 0)],
          "ㅓ": [("EO", 0), ("O", .2), ("U", .2)], "ㅔ": [("E", 0)], "ㅕ": [("YEO", 0), ("YO", .2), ("YU", .2)],
          "ㅖ": [("YE", 0)], "ㅗ": [("O", 0), ("OH", .4)], "ㅘ": [("WA", 0)], "ㅙ": [("WAE", 0)],
          "ㅚ": [("OE", 0), ("OI", .3)], "ㅛ": [("YO", 0)], "ㅜ": [("U", 0), ("OO", .2)], "ㅝ": [("WO", 0), ("WEO", .2)],
          "ㅞ": [("WE", 0)], "ㅟ": [("WI", 0)], "ㅠ": [("YU", 0), ("YOU", .3)], "ㅡ": [("EU", 0), ("U", .3)],
          "ㅢ": [("UI", 0), ("EUI", .2)], "ㅣ": [("I", 0), ("EE", .3)]}
JONG_R = {"": [("", 0)], "ㄱ": [("K", 0), ("G", .3), ("CK", .4)], "ㄴ": [("N", 0)], "ㄷ": [("T", 0), ("D", .3)],
          "ㄹ": [("L", 0), ("R", .3)], "ㅁ": [("M", 0)], "ㅂ": [("P", 0), ("B", .3)], "ㅅ": [("T", .2), ("S", .3)],
          "ㅇ": [("NG", 0)]}
SPECIAL = {"WOO": ("우", 0.0), "WU": ("우", 0.2), "YOUNG": ("영", 0.0), "YUNG": ("영", 0.3), "HYUN": ("현", 0.0),
           "SHIN": ("신", 0.0), "KUK": ("국", 0.0), "KOOK": ("국", 0.1), "SUNG": ("성", 0.0), "JUNG": ("정", 0.0),
           "CHUNG": ("정", 0.1), "CHUN": ("천", 0.0), "KUM": ("금", 0.0), "KEUM": ("금", 0.0), "KYUNG": ("경", 0.0),
           "MYUNG": ("명", 0.0), "DUCK": ("덕", 0.0), "DUK": ("덕", 0.1), "SEUNG": ("승", 0.0), "HUNG": ("흥", 0.0),
           "HEUNG": ("흥", 0.0), "KWANG": ("광", 0.0), "WON": ("원", 0.0), "WOON": ("운", 0.1), "SOO": ("수", 0.0),
           "SUH": ("서", 0.1), "LEE": ("이", 0.1), "PUSAN": ("부산", 0.0), "INCHON": ("인천", 0.0),
           "SEOUL": ("서울", 0.0), "KOREA": ("코리아", 0.0), "HO": ("호", 0.0)}
LETTER = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
          "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
          "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유",
          "X": "엑스", "Y": "와이", "Z": "제트"}


def compose(cho: str, jung: str, jong: str) -> str:
    return chr(0xAC00 + CHO.index(cho) * 588 + JUNG.index(jung) * 28 + JONG.index(jong))


class Converter:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame):
        cat = catalog.set_index("vessel_id")
        ko = train[train.vessel_name.str.contains("[가-힣]", na=False)].drop_duplicates("target_vessel_id")
        self.syl_freq = Counter(ch for n in ko.vessel_name for ch in n if "가" <= ch <= "힣")
        # English word -> Hangul word, from train pairs with equal word counts
        en2ko = defaultdict(Counter)
        for name, t in zip(ko.vessel_name, ko.target_vessel_id):
            kw = re.findall(r"[가-힣]+", str(name))
            ew = [w for w in re.findall(r"[A-Z]+", str(cat.loc[t, "vessel_name"]).upper()) if w not in ("NO",)]
            if len(kw) == len(ew):
                for a, b in zip(ew, kw):
                    # keep only phonetically plausible word pairs (misaligned names pair unrelated words)
                    if Levenshtein.normalized_similarity(T.ko_key(b), T.en_key(a)) >= 0.5:
                        en2ko[a][b] += 1
        self.en2ko = {k: v.most_common(1)[0][0] for k, v in en2ko.items()}
        # romanised syllable inventory
        inv: dict[str, tuple[str, float]] = {}
        for ci, c in enumerate(CHO):
            for vi, v in enumerate(JUNG):
                for f, fr in JONG_R.items():
                    syl = compose(c, v, f)
                    prior = -min(0.3, 0.08 * math.log1p(self.syl_freq.get(syl, 0)))
                    for cr, cc in CHO_R[c]:
                        for vr, vc in JUNG_R[v]:
                            for jr, jc in fr:
                                key = cr + vr + jr
                                if not key:
                                    continue
                                cost = 1.0 + cc + vc + jc + prior + (0.3 if syl not in self.syl_freq else 0.0)
                                if key not in inv or cost < inv[key][1]:
                                    inv[key] = (syl, cost)
        for k, (s, cst) in SPECIAL.items():
            inv[k] = (s, cst + 0.6)
        self.inv = inv
        self.maxlen = max(len(k) for k in inv)

    def roman_word(self, w: str):
        """Best segmentation of a romanised Korean word into Hangul syllables (None if impossible)."""
        n = len(w)
        best = [(math.inf, "")] * (n + 1)
        best[0] = (0.0, "")
        for i in range(n):
            if best[i][0] == math.inf:
                continue
            for j in range(i + 1, min(n, i + self.maxlen) + 1):
                hit = self.inv.get(w[i:j])
                if hit:
                    c = best[i][0] + hit[1]
                    if c < best[j][0]:
                        best[j] = (c, best[i][1] + hit[0])
        return best[n][1] if best[n][0] < math.inf and best[n][0] / max(len(best[n][1]), 1) < 1.9 else None

    def word(self, w: str, rng) -> str | None:
        if w in self.en2ko:
            return self.en2ko[w]
        if len(w) <= 4 and not re.search("[AEIOUY]", w):
            return "".join(LETTER[ch] for ch in w)
        if len(w) <= 3 and rng.rand() < 0.5 and w not in SPECIAL:
            return "".join(LETTER[ch] for ch in w)
        return self.roman_word(w)

    def name(self, catalog_name: str, rng) -> str | None:
        s = str(catalog_name).upper().replace("-", " ").replace(".", " ")
        toks = s.split()
        digits = [t for t in toks if t.isdigit()]
        words = [t for t in toks if not t.isdigit() and t not in ("NO",)]
        if not words or len(digits) > 1 or any(not t.isalpha() for t in words):
            return None
        out = []
        for w in words:
            k = self.word(w, rng)
            if k is None:
                return None
            out.append(k)
        if out[-1] == "호" and len(out) >= 2:
            out = out[:-2] + [out[-2] + "호"]
        joined = (" " if rng.rand() < 0.6 else "").join(out)
        if digits:
            d = digits[0]
            r = rng.rand()
            joined = (d + joined) if r < 0.25 else (f"제{d}" + joined) if r < 0.4 else (joined + d) if r < 0.75 else (joined + d + "호")
        elif rng.rand() < 0.15 and not joined.endswith("호"):
            joined += "호"
        return joined
