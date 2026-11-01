"""Name normalisation, typo-legality checks and Korean phonetic keys.

Everything here is a pure function of its arguments (no global state learned
from data), so it is safe to use identically at training and inference time.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

HANGUL_RE = re.compile("[가-힣]")
NON_ALNUM_RE = re.compile(r"[^0-9A-Z가-힣]+")


def is_korean(value) -> bool:
    return isinstance(value, str) and bool(HANGUL_RE.search(value))


def nfkc_upper(value) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).upper()


def words(value) -> list[str]:
    """Upper-case alphanumeric words; every other character separates words."""
    return NON_ALNUM_RE.sub(" ", nfkc_upper(value)).split()


def compact(value) -> str:
    return "".join(words(value))


def strip_vessel_prefix(ws: list[str]) -> list[str]:
    """Drop a leading 'M/V' / 'MV' style prefix that port records sometimes add."""
    if len(ws) >= 3 and ws[0] == "M" and ws[1] == "V":
        return ws[2:]
    if len(ws) >= 2 and ws[0] in ("MV", "MT"):
        return ws[1:]
    return ws


# ---------------------------------------------------------------------------
# Synthetic typo model observed in train: inside words of >= 6 letters the
# generator deletes an interior letter, doubles a letter, or swaps two
# interior neighbours.  Digits, spaces and punctuation are never touched.
# ---------------------------------------------------------------------------

def _one_op(word: str) -> set[str]:
    n = len(word)
    out: set[str] = set()
    for i in range(1, n - 1):
        if word[i].isalpha():
            out.add(word[:i] + word[i + 1:])
    for i in range(n):
        if word[i].isalpha():
            out.add(word[: i + 1] + word[i] + word[i + 1:])
    for i in range(1, n - 2):
        if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]:
            out.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    out.discard(word)
    return out


@lru_cache(maxsize=200_000)
def word_ops(query_word: str, catalog_word: str) -> int:
    """Minimum number (0, 1, 2) of legal typo operations turning the catalog
    word into the query word; 9 when impossible within two operations."""
    if query_word == catalog_word:
        return 0
    if len(catalog_word) < 6 or abs(len(query_word) - len(catalog_word)) > 2:
        return 9
    first = _one_op(catalog_word)
    if query_word in first:
        return 1
    for mid in first:
        if query_word in _one_op(mid):
            return 2
    return 9


def name_ops(query_words: list[str], catalog_words: list[str]) -> int:
    """Total legal typo operations between two word lists of equal length."""
    if len(query_words) != len(catalog_words):
        return 9
    total = 0
    for q, c in zip(query_words, catalog_words):
        total += word_ops(q, c)
        if total >= 9:
            return 9
    return min(total, 9)


# ---------------------------------------------------------------------------
# Korean handling
# ---------------------------------------------------------------------------
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def syllables(text: str):
    """Yield (cho, jung, jong) jamo triples; non-Hangul characters yield
    (char, None, None)."""
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            yield CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]
        else:
            yield ch, None, None


# Revised-Romanisation-like reading, used as one romanised view of a name.
RR_CHO = {"ㄱ": "G", "ㄲ": "KK", "ㄴ": "N", "ㄷ": "D", "ㄸ": "TT", "ㄹ": "R", "ㅁ": "M",
          "ㅂ": "B", "ㅃ": "PP", "ㅅ": "S", "ㅆ": "SS", "ㅇ": "", "ㅈ": "J", "ㅉ": "JJ",
          "ㅊ": "CH", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "H"}
RR_JUNG = {"ㅏ": "A", "ㅐ": "AE", "ㅑ": "YA", "ㅒ": "YAE", "ㅓ": "EO", "ㅔ": "E", "ㅕ": "YEO",
           "ㅖ": "YE", "ㅗ": "O", "ㅘ": "WA", "ㅙ": "WAE", "ㅚ": "OE", "ㅛ": "YO", "ㅜ": "U",
           "ㅝ": "WO", "ㅞ": "WE", "ㅟ": "WI", "ㅠ": "YU", "ㅡ": "EU", "ㅢ": "UI", "ㅣ": "I"}
RR_JONG = {"": "", "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N", "ㄷ": "T",
           "ㄹ": "L", "ㄺ": "K", "ㄻ": "M", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㄿ": "P", "ㅀ": "L",
           "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅅ": "T", "ㅆ": "T", "ㅇ": "NG", "ㅈ": "T", "ㅊ": "T",
           "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "T"}

# Loanword-oriented reading (English words written in Hangul).
LW_CHO = dict(RR_CHO, **{"ㄹ": "L", "ㅂ": "B", "ㅍ": "P", "ㅆ": "S", "ㄲ": "K", "ㄸ": "T",
                         "ㅉ": "J", "ㅃ": "P"})
LW_JUNG = dict(RR_JUNG, **{"ㅓ": "O", "ㅕ": "YO", "ㅡ": "", "ㅐ": "E", "ㅒ": "YE", "ㅚ": "WE",
                           "ㅢ": "I"})
LW_JONG = dict(RR_JONG, **{"ㅅ": "T", "ㄹ": "L"})


def romanize(text: str, style: str = "rr") -> str:
    cho, jung, jong = (RR_CHO, RR_JUNG, RR_JONG) if style == "rr" else (LW_CHO, LW_JUNG, LW_JONG)
    out = []
    for a, b, c in syllables(nfkc_upper(text)):
        if b is None:
            if a.isalnum():
                out.append(a)
            else:
                out.append(" ")
            continue
        out.append(cho[a] + jung[b] + jong[c])
    return " ".join("".join(out).split())


# ---------------------------------------------------------------------------
# Phonetic class keys shared by English and Korean spellings.
# Consonant classes: K T P S J L M N G(=NG) H ; vowels A E I O U ; digits kept.
# ---------------------------------------------------------------------------
KO_CHO_CLASS = {"ㄱ": "K", "ㄲ": "K", "ㅋ": "K", "ㄷ": "T", "ㄸ": "T", "ㅌ": "T", "ㅂ": "P",
                "ㅃ": "P", "ㅍ": "P", "ㅅ": "S", "ㅆ": "S", "ㅈ": "J", "ㅉ": "J", "ㅊ": "J",
                "ㄴ": "N", "ㄹ": "L", "ㅁ": "M", "ㅎ": "H", "ㅇ": ""}
KO_JONG_CLASS = {"": "", "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N",
                 "ㄷ": "T", "ㄹ": "L", "ㄺ": "K", "ㄻ": "M", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L",
                 "ㄿ": "P", "ㅀ": "L", "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅅ": "T", "ㅆ": "T",
                 "ㅇ": "G", "ㅈ": "T", "ㅊ": "T", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "T"}
KO_JUNG_CLASS = {"ㅏ": "A", "ㅐ": "E", "ㅑ": "IA", "ㅒ": "IE", "ㅓ": "O", "ㅔ": "E", "ㅕ": "IO",
                 "ㅖ": "IE", "ㅗ": "O", "ㅘ": "UA", "ㅙ": "UE", "ㅚ": "UE", "ㅛ": "IO", "ㅜ": "U",
                 "ㅝ": "UO", "ㅞ": "UE", "ㅟ": "UI", "ㅠ": "IU", "ㅡ": "", "ㅢ": "I", "ㅣ": "I"}


def ko_key(text: str) -> str:
    """Phonetic class string of a Hangul (or mixed Hangul/Latin) name."""
    out = []
    for word in nfkc_upper(text).split():
        if not HANGUL_RE.search(word):
            out.append("".join(_en_word_key(w) for w in words(word)))
            continue
        for a, b, c in syllables(word):
            if b is None:
                if a.isdigit():
                    out.append(a)
                continue
            out.append(KO_CHO_CLASS[a] + KO_JUNG_CLASS[b] + KO_JONG_CLASS[c])
    return _squeeze("".join(out))


_EN_RULES = [
    ("TCH", "J"), ("SCH", "S"), ("PH", "P"), ("TH", "S"), ("SH", "S"), ("CH", "J"),
    ("CK", "K"), ("GH", ""), ("QU", "KU"), ("X", "KS"), ("Q", "K"), ("NG", "G"),
    ("CE", "SE"), ("CI", "SI"), ("CY", "SI"), ("C", "K"), ("G", "K"),
    ("F", "P"), ("V", "P"), ("B", "P"), ("D", "T"), ("Z", "J"), ("R", "L"),
    ("W", "U"), ("Y", "I"), ("OU", "U"), ("OO", "U"), ("EE", "I"), ("EA", "I"),
]
_VOWELS = set("AEIOUY")


def _en_word_key(word: str) -> str:
    if word.isdigit():
        return word
    # Non-rhotic R (Korean drops it before consonants and at word end) and a
    # silent final E after a consonant.
    w = "".join(ch for k, ch in enumerate(word)
                if not (ch == "R" and k > 0 and (k + 1 == len(word) or word[k + 1] not in _VOWELS)))
    if len(w) > 3 and w.endswith("E") and w[-2] not in _VOWELS:
        w = w[:-1]
    out = []
    i = 0
    while i < len(w):
        for src, dst in _EN_RULES:
            if w.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(w[i])
            i += 1
    return "".join(out)


def en_key(text: str) -> str:
    """Phonetic class string of a Latin-script name, roughly as Korean
    transliteration would render it.  Rules apply within words only."""
    return _squeeze("".join(_en_word_key(w) for w in words(text)))


def _squeeze(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def consonants(key: str) -> str:
    return _squeeze(re.sub("[AEIOU]", "", key))


def digits_of(value) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", nfkc_upper(value)))


ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7",
         "VIII": "8", "IX": "9", "X": "10", "XI": "11", "XII": "12"}
KO_NUMBER_WORDS = {"원": "1", "투": "2", "쓰리": "3", "포": "4", "파이브": "5", "식스": "6",
                   "세븐": "7", "에잇": "8", "에이트": "8", "나인": "9", "텐": "10"}


def drop_number_marker(ws: list[str]) -> list[str]:
    """'NO 7 STAR' -> '7 STAR' (the catalog writes 'No. 7', ports often not) and
    trailing Roman numerals -> digits ('OCEAN STAR II' -> 'OCEAN STAR 2')."""
    out = []
    for k, w in enumerate(ws):
        if w == "NO" and k + 1 < len(ws) and ws[k + 1].isdigit():
            continue
        if w.startswith("NO") and w[2:].isdigit():
            out.append(w[2:])
            continue
        if k > 0 and k == len(ws) - 1 and w in ROMAN:
            out.append(ROMAN[w])
            continue
        out.append(w)
    return out


# Korean spellings of Latin letter names, used to decode acronyms such as
# '케이엠티씨' -> 'KMTC' or '에이치씨' -> 'HC'.
KO_LETTERS = {
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E", "에프": "F", "지": "G",
    "쥐": "G", "에이치": "H", "에치": "H", "아이": "I", "제이": "J", "케이": "K", "엘": "L",
    "엠": "M", "엔": "N", "오": "O", "피": "P", "큐": "Q", "알": "R", "에스": "S", "에쓰": "S",
    "티": "T", "유": "U", "브이": "V", "더블유": "W", "떠블유": "W", "따블유": "W", "엑스": "X",
    "와이": "Y", "제트": "Z", "젯": "Z", "앤": "N",
}
_KO_LETTER_KEYS = sorted(KO_LETTERS, key=len, reverse=True)


def _letter_parses(word: str, limit: int = 4) -> list[str]:
    """All full segmentations of a Hangul word into letter names (bounded)."""
    results: list[str] = []

    def walk(pos: int, acc: str) -> None:
        if len(results) >= limit:
            return
        if pos == len(word):
            results.append(acc)
            return
        for key in _KO_LETTER_KEYS:
            if word.startswith(key, pos):
                walk(pos + len(key), acc + KO_LETTERS[key])

    walk(0, "")
    return results


def ko_acronym_variants(name: str) -> list[str]:
    """Replace spelled-out letter words by their Latin acronym.  Also tries a
    letter-name prefix glued to a following word ('엠에스오션' -> 'MS 오션')."""
    ws = nfkc_upper(name).split()
    options: list[list[str]] = []
    changed = False
    for w in ws:
        opts = [w]
        if HANGUL_RE.search(w):
            for parse in _letter_parses(w):
                if len(parse) >= 2:
                    opts.append(parse)
                    changed = True
            # glued prefix: longest prefix with >= 2 letters
            for cut in range(len(w) - 1, 1, -1):
                head, tail = w[:cut], w[cut:]
                parses = [p for p in _letter_parses(head, 2) if len(p) >= 2]
                if parses and len(tail) >= 2:
                    opts.append(parses[0] + " " + tail)
                    changed = True
                    break
        options.append(opts[:3])
    if not changed:
        return []
    out = []
    # combine: first alternative where available (keeps the count small)
    combo = [o[1] if len(o) > 1 else o[0] for o in options]
    out.append(" ".join(combo))
    return out


_KO_NUM_RE = re.compile(r"제\s*(\d+)\s*호?")
_KO_NUM_SUFFIX_RE = re.compile(r"(\d+)\s*호")


def ko_variants(name: str) -> list[str]:
    """Readings of a Hangul port name with the ship-name decorations removed:
    '제7' / '7호' number markers and a trailing '호' (ship) suffix."""
    base = nfkc_upper(name)
    base = _KO_NUM_RE.sub(r" \1 ", base)
    base = _KO_NUM_SUFFIX_RE.sub(r" \1 ", base)
    ws = base.split()
    if len(ws) >= 2 and ws[-1] in KO_NUMBER_WORDS:
        ws[-1] = KO_NUMBER_WORDS[ws[-1]]
    base = " ".join(ws)
    out = [base]
    stripped = re.sub(r"호$", "", base).strip()
    if stripped != base and len(stripped.replace(" ", "")) >= 2:
        out.append(stripped)
    return out
