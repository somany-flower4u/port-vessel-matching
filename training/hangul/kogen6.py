"""kogen6: kogen5 (domgen-patched kogen + English-loanword / pinyin fallback) with fixes for Chinese and Korean names (round4).

install(catalog) patches kogen.Converter.__init__ (adds the per-word dictionary variant counts), .word and .name
IN THIS PROCESS ONLY (kogen.py / domgen.py / loanword.py / kogen5.py are not edited).  Word decision order:
  0. [k6] Chinese-context names: CHANG SHUN JUN CHUN GU ER -> 창 순 준 춘 구 얼 (장 선 전 천 거 엘 as a 25 % alternative)
  1. roman numerals II..IX            -> 투 / 2, 쓰리 / 3 ...;  [k6] a lone non-initial I -> 아이 / 1 / 원
  2. train dictionary en2ko           -> [k6] every observed spelling, sampled in proportion to its count
                                         (a stray trailing 호 copied from the port name is dropped;
                                         when the hand list disagrees, either spelling 50/50)
  3. hand-spelled loanwords (loanword.EXC)
  4. single letters / vowel-less tokens -> Korean letter names (A 에이, MSC 엠에스씨)
  5. clearly romanised Korean words   -> domgen.patched_word;  [k6] plus sampled Revised-Romanisation sound-change
                                         reversals (Halla 할라/한라, Baengnyeong 뱅녕/백령, Simnyeon 심년/십년)
  6. pinyin words                     -> [k6] pinyin6 reader (샤/샨/샹, 셩, 쉬, 톈/롄 variants);  [k6] routing also takes
                                         any fully pinyin-splittable non-English word in a Chinese-context name, known
                                         Chinese place names (HUNAN, TIANJIN, DALIAN ...) and words with >= 2
                                         distinctively Chinese syllables (BAOSHAN, HUASHAN) or 1 in a one-word name (BOSHI)
  7. English dictionary words seen in >= 3 non-Korean-flag catalog names -> loanword.loan
  8. domgen.patched_word ([k6] + RR sound-change alternatives when the word has RR vowel digraphs EO EU AE OE YEO;
     only the ngn / ngm / mn ones in other words of Korean-flagged names); if it gives None -> loanword.loan
  [k6] RR vowel digraphs never go to the pinyin reader through a mid-word zero-initial syllable (BAENGMA != ba+eng+ma)
A 'Chinese-context' name: its catalog flag is China / Hong Kong, or it has >= 2 words, is not Korean-flagged, every word
splits into pinyin syllables (and is not an English / hand-list word) and at least one word is distinctively Chinese.
Pinyin routing: always for kogen5 pinyin words, CN_PLACES, words with >= 2 strongly Chinese syllables and every word of an
all-pinyin name; sampled (70 %) for distinctively Chinese words in China / Hong Kong flagged names and (50 %) for
one-word names with a strongly Chinese syllable -- otherwise the kogen5 path.
The flag is looked up from the catalog name (install() builds a name -> majority flag map); word() called directly (no
name) is context-free.
name() additionally splits letter-digit tokens (Ace1 -> ACE 1, 25kumjin -> 25 KUMJIN), drops apostrophes / '&' and
[k6] splits hyphenated roman numerals off (Bagan-I -> BAGAN I).
All randomness comes from the rng passed by the caller."""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))

import kogen  # noqa: E402
import domgen  # noqa: E402  (its import patches kogen.Converter.word = domgen.patched_word)
import loanword as LW  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

_PATCHED = domgen.patched_word
_ORIG_INIT = kogen.Converter.__init__
ROMAN_NUM = {"II": ("투", "2"), "III": ("쓰리", "3"), "IV": ("포", "4"), "VI": ("식스", "6"), "VII": ("세븐", "7"),
             "VIII": ("에이트", "8"), "IX": ("나인", "9")}
# common romanised Korean name words that are also English / pinyin-looking strings
KOREAN_WORDS = set("""DAE HAE WOO SAM SUNG YANG DONG HAN KUM KEUM JIN HANA SHIN SIN SEO SUH NAM KANG GANG MIN JOO JU
IL IN HO HWA HYUN YOUNG SEONG SEUNG CHANG CHUNG JUNG JANG GEUM KIM HEUNGA DONGA SAMA PARK LEE CHOI CHO OH YU YOO BONG SAN MI MYUNG
HEUNG HUNG KWANG GWANG BOK BO DO GU GUK KUK KOOK HYUNDAI DAEWOO SAMSUNG HANJIN DONGBANG HEUNGA SINOKOR CHANGMYUNG
SEOUL BUSAN PUSAN INCHEON INCHON ULSAN YEOSU MOKPO JEJU POHANG GUNSAN PYEONGTAEK DAESAN KOREA HANIL HANSUNG""".split())
KOREAN_WORDS |= {"HALLA", "SILLA", "JEOLLA", "SEOLLAL"}  # [k6 change 6] Korean words whose RR spelling hides a sound change

CN_FLAGS = {"Peoples' Republic of China", "Hong Kong"}
KR_FLAG = "South Korea"

_STATS: dict = {}


def install(catalog) -> None:
    """Compute catalog word statistics / name flags and patch kogen.Converter (__init__, word, name)."""
    kr, nonkr = Counter(), Counter()
    flags: dict = defaultdict(Counter)
    for n, f in zip(catalog.vessel_name.astype(str), catalog.flag_name.astype(str)):
        for w in set(_tokens(n)):
            if w.isalpha():
                (kr if f == KR_FLAG else nonkr)[w] += 1
        flags[n.upper().strip()][f] += 1
    _STATS["kr"], _STATS["nonkr"] = kr, nonkr
    _STATS["flag"] = {k: v.most_common(1)[0][0] for k, v in flags.items()}
    kogen.Converter.__init__ = _init
    kogen.Converter.word = word
    kogen.Converter.name = name


# ------------------------------------------------------------------ [k6 change 3] dictionary variants with counts
def _init(self, train, catalog):
    """Original kogen.Converter.__init__, plus self.en2ko_var: word -> Counter of every observed Hangul spelling
    (same pairing / plausibility filter as kogen builds en2ko with; en2ko itself is left unchanged)."""
    _ORIG_INIT(self, train, catalog)
    T = kogen.T  # same key functions kogen uses
    cat = catalog.set_index("vessel_id")
    ko = train[train.vessel_name.str.contains("[가-힣]", na=False)].drop_duplicates("target_vessel_id")
    var = defaultdict(Counter)
    for nm, t in zip(ko.vessel_name, ko.target_vessel_id):
        kw = re.findall(r"[가-힣]+", str(nm))
        ew = [w for w in re.findall(r"[A-Z]+", str(cat.loc[t, "vessel_name"]).upper()) if w not in ("NO",)]
        if len(kw) == len(ew):
            for a, b in zip(ew, kw):
                if Levenshtein.normalized_similarity(T.ko_key(b), T.en_key(a)) >= 0.5:
                    var[a][b] += 1
    self.en2ko_var = {k: v for k, v in var.items() if len(v) > 1}


DICT_VARIANTS = True  # False = kogen5 behaviour (most common spelling only); for ablations


def _dict_word(self, w: str, rng) -> str:
    v = self.en2ko[w]
    c = getattr(self, "en2ko_var", {}).get(w) if DICT_VARIANTS else None
    if c:  # several spellings seen in train (산동 / 샨동, 톈진 / 텐진, 선전 / 센젠): sample by count
        ks = sorted(c)
        p = [c[k] for k in ks]
        v = ks[int(rng.choice(len(ks), p=[x / sum(p) for x in p]))]
    return v


def _tokens(catalog_name: str) -> list[str]:
    s = str(catalog_name).upper()
    # [k6 change 5] hyphenated roman numerals become their own word (Bagan-I -> BAGAN I) before the Heung-A rule
    s = re.sub(r"(?<=[A-Z])-(I{1,3}|IV|VI{0,3}|IX)(?![A-Z])", r" \1", s)
    s = re.sub(r"(?<=[A-Z])-(?=[A-Z](?![A-Z]))", "", s)  # Heung-A -> HEUNGA (흥아)
    s = s.replace("-", " ").replace(".", " ").replace("/", " ").replace("&", " ")
    s = s.replace("'", "").replace("’", "").replace(",", " ").replace("(", " ").replace(")", " ")
    s = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", s)
    return [t for t in s.split() if t]


def _letters(w: str) -> str:
    return "".join(kogen.LETTER[ch] for ch in w)


def is_korean(self, w: str) -> bool:
    if w in kogen.SPECIAL or w in KOREAN_WORDS:
        return True
    k, o = _STATS["kr"].get(w, 0), _STATS["nonkr"].get(w, 0)
    if k >= 3 and k >= 0.3 * (k + o):
        return True
    r = self.roman_word(w)  # one romanised syllable that real Korean port names use
    return bool(r) and len(r) == 1 and self.syl_freq.get(r, 0) >= 2


def english_like(w: str) -> bool:
    return len(w) >= 3 and _STATS["nonkr"].get(w, 0) >= 3 and LW.is_english(w)


# ------------------------------------------------------------------ [k6 change 1] pinyin routing
# Chinese place names that do not carry a 'pinyin mark' (or are read wrongly as Korean / Latin without one)
CN_PLACES = set("""HUNAN HUBEI HENAN HEBEI TIANJIN DALIAN HUASHAN HAINAN YUNNAN ANHUI FUJIAN JIANGSU JIANGXI SHANXI SHAANXI
SICHUAN GANSU JILIN TAISHAN LUSHAN HUAIHE WUHAN NANJING BEIJING TANGSHAN LIANYUNGANG LIANHUA TIANHE TIANLI TIANBAO
TAICANG NANTONG RIZHAO WEIHAI WEIFANG YANTAI DANDONG YINGKOU JINZHOU ZHOUSHAN XIAMEN FUZHOU SHANTOU ZHANJIANG BEIHAI
HAIKOU SANYA HUIZHOU ZHUHAI JIAXING NINGDE PUTIAN QUANZHOU TIANJINXINGANG CAOFEIDIAN JINGTANG HUANGHUA LONGKOU
BAOSHAN BOHAI DONGHAI NANHAI HUANGHAI MINSHENG PENGLAI""".split())
# 'distinctive' syllables (used in China / Hong Kong flagged names and for the all-pinyin test): zh ch sh x q
# initials, ao ou ei ui uo ua ian iao iu ie ong eng ... finals
_DIST_INI = ("zh", "ch", "sh", "x", "q")
_DIST_FIN = {"ao", "ou", "ei", "ui", "uo", "ua", "uai", "uan", "uang", "ian", "iao", "iang", "iu", "ie", "iong", "ong",
             "eng"}
# 'strong' syllables (used without name context): ou / ei / ie / sh+i / sh+a are left out because Japanese romaji
# (HOUEI, SEISHIN, SHIMA) and Arabic / Latin names (NATASHA) are full of them
_STRONG_FIN = {"ao", "ui", "uo", "ua", "uai", "uan", "uang", "ian", "iao", "iang", "iu", "iong", "ong", "eng"}


def _split(w: str):
    sp = LW.pinyin_split(w)
    return sp if sp and len(sp) <= 4 else None


def _english(w: str) -> bool:
    return w in LW.EXC or w in LW.EN_WORDS


def _n_syl(w: str, strong: bool, shi: bool = False) -> int:
    """Number of distinctively (strong=False) or strongly (strong=True) Chinese syllables of a pinyin-splittable word."""
    sp = _split(w)
    if not sp:
        return 0
    n = 0
    for k, s in enumerate(sp):
        ini, fin = LW._PY_SYL[s]
        if strong:  # (shi=True: a final 'shi' of a two-syllable word counts too -- BOSHI, not OSHIMA)
            n += (ini in ("zh", "x", "q") or fin in _STRONG_FIN or (ini == "sh" and fin not in ("i", "a", "e", "u"))
                  or (shi and ini == "sh" and fin == "i" and len(sp) == 2 and k == 1))
        else:
            n += ini in _DIST_INI or fin in _DIST_FIN
    return n


def _py_like(w: str) -> bool:
    return w.isalpha() and _split(w) is not None and not _english(w)


def _cn_word(w: str) -> bool:
    """A word that can stand in an all-pinyin name: pinyin-like and a single syllable, a kogen5 pinyin word, a Chinese
    place name or with a distinctively Chinese syllable (so SERI ANGGUN, KAWA HONGKONG, HELENE JULIE do not qualify)."""
    sp = _split(w)
    if sp and len(sp) == 1 and w.isalpha() and w not in LW.EXC:  # one pinyin syllable, even if also English (SHUN)
        return True
    return (_py_like(w) and (LW.is_pinyin(w) or _n_syl(w, False) >= 1)) or w in CN_PLACES


def _context(words: list, catalog_name: str) -> dict:
    """Name-level context for word():
    cnflag  catalog flag China / Hong Kong (looked up by catalog name)
    allpy   >= 2 words, every word a _cn_word (pinyin-splittable, not English, and one syllable / marked / place /
            distinctive), and either China / Hong Kong flagged or
            (not Korean-flagged and at least one word distinctively Chinese / a single pinyin syllable of >= 3 letters)
    cn      cnflag or allpy  (the 'Chinese-context' of change 4)
    kr      Korean flag;  single  one-word name"""
    flag = _STATS.get("flag", {}).get(str(catalog_name).upper().strip())
    ws = [w for w in words if w.isalpha() and w not in ROMAN_NUM and w != "I"]
    cnflag, kr = flag in CN_FLAGS, flag == KR_FLAG
    allpy = len(ws) >= 2 and all(_cn_word(w) for w in ws) and (cnflag or (
        not kr and any(not _english(w) for w in ws) and any(_n_syl(w, False) or (LW.is_pinyin(w) and len(w) >= 3) for w in ws)))
    return dict(cnflag=cnflag, allpy=allpy, cn=cnflag or allpy, kr=kr, single=len(ws) == 1)


_RR_VOWELS = re.compile("AE|EO|EU|OE")


def _route_pinyin(w: str, ctx) -> float:
    """Probability that w is read by the pinyin reader (1.0 = always; in between = sampled against the kogen5 path)."""
    sp = _split(w)
    if LW.is_pinyin(w) and not (sp and _RR_VOWELS.search(w) and any(LW._PY_SYL[x][0] == "" for x in sp[1:])):
        return 1.0  # kogen5 rule (single syllable, or a 'pinyin mark'), except RR vowel digraphs split as ba+eng
    if not sp:
        return 0.0
    if w in CN_PLACES:
        return 1.0
    if _RR_VOWELS.search(w) and any(LW._PY_SYL[x][0] == "" for x in sp[1:]):  # BAENGMA = ba+eng+ma: RR, not pinyin
        return 0.0
    if _english(w):
        return 0.0
    if len(sp) >= 2 and _n_syl(w, True) >= 2:  # BAOSHAN, HUASHAN (no name context needed)
        return 1.0
    if ctx is None:
        return 0.0
    if ctx["allpy"]:  # the rest of the name is pinyin-like too
        return 1.0
    if re.search("C(?!H)", w):  # Latin c (CANADIAN, BIANCA, ACADIAN): pinyin c- words are rare in ship names
        return 0.0
    if ctx["cnflag"] and _n_syl(w, False) >= 1:  # China / Hong Kong flag (Maersk Shekou); non-Chinese words stay put
        return 0.7
    if ctx["single"] and len(sp) >= 2 and _n_syl(w, True, shi=True) >= 1:  # one-word name: BOSHI, JINHUA, YANTIAN
        return 0.5
    return 0.0


# ------------------------------------------------------------------ [k6 change 2] pinyin reader with port variants
_PY_FIN6 = dict(LW._PY_FIN)
# ian: 톈 / 롄 / 볜 (standard ㅖ) or 텐 / 렌, or 티안 / 리안; the kogen5 ㅕ reading (텬 / 련) is dropped
_PY_FIN6["ian"] = [(["ㅔ"], "ㄴ", .35), (["ㅖ"], "ㄴ", .3), (["ㅣ", "ㅏ"], "ㄴ", .35)]


def pinyin6(word: str, rng=None) -> str | None:
    """loanword.pinyin with extra port-style variants:
    sh + a / an / ang -> 샤 / 샨 / 샹 (35 %) besides 사 / 산 / 상;  sheng -> 셍 / 성 / 셩;  shi -> 시 / 쉬 / 스;
    ian -> 엔 / 옌 / 이안 (톈진, 다롄, 다리안);  xiu stays 슈 / 시우 and shou 서우 / 소우 / 소.
    Unchanged: Feng 펭/펑, Hai 하이, Ning 닝, Tuo 투오/퉈, Ren 렌/런, Rui 루이, Zhen 젠/전, Heng 헹/헝, Yuan 유안/위안, Hui 후이."""
    sp = LW.pinyin_split(re.sub("[^a-z]", "", word.lower()))
    if not sp:
        return None
    out = []
    for s in sp:
        ini, fin = LW._PY_SYL[s]
        if ini in ("j", "q", "x", "y") and fin in LW._V_FIN:
            fin = LW._V_FIN[fin]
        vs = _PY_FIN6[fin]
        if ini == "sh" and fin == "eng":
            vs = [(["ㅔ"], "ㅇ", .5), (["ㅓ"], "ㅇ", .25), (["ㅕ"], "ㅇ", .25)]
        if len(vs) == 1 or rng is None:
            meds, jong, _ = vs[0]
        else:
            k = int(rng.choice(len(vs), p=[v[2] for v in vs]))
            meds, jong, _ = vs[k]
        meds = list(meds)
        cho = LW._PY_ONSET[ini]
        if ini == "y":
            if fin == "i" or fin == "in" or fin == "ing":
                pass
            elif fin == "ou":
                meds = ["ㅠ"] if (rng is None or rng.rand() < 0.6) else ["ㅕ", "ㅜ"]
            elif fin in ("v", "van", "vn", "ve"):
                pass
            else:
                meds[0] = LW.Y_MERGE.get(meds[0], meds[0])
        elif ini == "w":
            if fin == "u":
                pass
            elif fin == "o":
                meds = ["ㅝ"]
            else:
                meds[0] = LW.W_MERGE.get(meds[0], meds[0])
        if ini in ("zh", "ch", "j", "q", "z", "c"):
            meds[0] = LW.Y_COLLAPSE.get(meds[0], meds[0])
            if meds[0] == "ㅟ":
                meds[0] = "ㅜ" if ini in ("zh", "ch") else "ㅟ"
        if ini == "sh" and fin == "i" and rng is not None:
            r = rng.rand()
            meds = ["ㅣ"] if r < 0.5 else ["ㅟ"] if r < 0.8 else ["ㅡ"]  # 시 / 쉬 / 스
        elif ini in ("zh", "ch", "r", "z", "c", "s") and fin == "i" and rng is not None and rng.rand() < 0.3:
            meds = ["ㅡ"]
        if ini == "sh" and fin in ("a", "an", "ang") and rng is not None and rng.rand() < 0.35:
            meds = ["ㅑ"]  # 샤 / 샨 / 샹
        if ini == "x" and fin in LW._I_FIN and meds[0] == "ㅣ" and len(meds) > 1 and rng is not None and rng.rand() < 0.5:
            meds = [LW.Y_MERGE.get(meds[1], meds[1])] + meds[2:]  # xia -> 샤
        syl = [LW._compose(cho, meds[0], jong if len(meds) == 1 else "")]
        for k2, m in enumerate(meds[1:]):
            syl.append(LW._compose("ㅇ", m, jong if k2 == len(meds) - 2 else ""))
        out.append("".join(syl))
    return "".join(out)


# ------------------------------------------------------------------ [k6 change 4] Chinese-context readings
# pinyin reading first (75 %), the Korean-style reading as the alternative (the port convention is unverified)
CN_CTX = {"CHANG": ("창", "장"), "SHUN": ("순", "선"), "JUN": ("준", "전"), "CHUN": ("춘", "천"), "GU": ("구", "거"),
          "ER": ("얼", "엘")}


# ------------------------------------------------------------------ [k6 change 6] RR sound-change reversal
# boundary (coda, next onset) of the literal reading -> alternative (coda, onset) pairs with weights
_RR_ALT = {("ㄹ", "ㄹ"): [(("ㄴ", "ㄹ"), .6), (("ㄹ", "ㄴ"), .4)],                  # ll: 한라 / 설날
           ("ㅇ", "ㄴ"): [(("ㄱ", "ㄹ"), .5), (("ㄱ", "ㄴ"), .3), (("ㅇ", "ㄹ"), .2)],   # ngn: 백령 / 학년 / 종로
           ("ㅇ", "ㅁ"): [(("ㄱ", "ㅁ"), 1.)],                                     # ngm: 백마
           ("ㅁ", "ㄴ"): [(("ㅂ", "ㄴ"), .6), (("ㅂ", "ㄹ"), .25), (("ㅁ", "ㄹ"), .15)],  # mn: 십년 / 협력 / 침로
           ("ㄴ", "ㄴ"): [(("ㄷ", "ㄴ"), 1.)]}                                     # nn after a vowel: ㄷㄴ
_RR_P = {("ㄴ", "ㄴ"): 0.15}  # chance per eligible boundary (default 0.4; ㄷㄴ is rare in names)
_RR_DIGRAPH = re.compile("EO|EU|AE|OE|YEO|WEO|UI$")


def rr_alt(h: str | None, rng, pats=None) -> str | None:
    """Sample a sound-change reversal at each eligible syllable boundary of a literal RR reading (40 % each, nn 15 %)."""
    if not h or len(h) < 2 or rng is None:
        return h
    ch = list(h)
    for i in range(len(ch) - 1):
        a, b = ord(ch[i]) - 0xAC00, ord(ch[i + 1]) - 0xAC00
        if not (0 <= a < 11172 and 0 <= b < 11172):
            continue
        jo, ci = kogen.JONG[a % 28], kogen.CHO[b // 588]
        alts = _RR_ALT.get((jo, ci)) if pats is None or (jo, ci) in pats else None
        if not alts or rng.rand() >= _RR_P.get((jo, ci), 0.4):
            continue
        (nj, nc), _ = alts[int(rng.choice(len(alts), p=[x[1] for x in alts]))] if len(alts) > 1 else alts[0]
        ch[i] = chr(0xAC00 + a - a % 28 + kogen.JONG.index(nj))
        ch[i + 1] = chr(0xAC00 + kogen.CHO.index(nc) * 588 + b % 588)
    return "".join(ch)


_KOREAN_ONLY = {("ㅇ", "ㄴ"), ("ㅇ", "ㅁ"), ("ㅁ", "ㄴ")}  # clusters Latin words rarely produce (ll / nn are left out)


def _korean_reading(self, w: str, rng, pats=None) -> str | None:
    """domgen.patched_word, plus the RR alternatives when it is the syllable reading (not dictionary / letters)."""
    r = _PATCHED(self, w, rng)
    if r is not None and w not in self.en2ko and re.search("[AEIOUY]", w) and r != _letters(w):
        r = rr_alt(r, rng, pats)
    return r


def word(self, w: str, rng) -> str | None:
    ctx = getattr(self, "_k6ctx", None)
    if ctx is not None and ctx["cn"] and w in CN_CTX:  # [k6 change 4]
        a, b = CN_CTX[w]
        return a if rng.rand() < 0.75 else b
    if w in ROMAN_NUM:
        a, d = ROMAN_NUM[w]
        return a if rng.rand() < 0.5 else d
    if w == "I" and ctx is not None and ctx.get("roman_i"):  # [k6 change 5] Victory I, Mohun Bagan-I
        r = rng.rand()
        return "아이" if r < 0.4 else "1" if r < 0.7 else "원"
    if w in self.en2ko:
        v = _dict_word(self, w, rng)  # [k6 change 3]
        if len(v) > 2 and v.endswith("호") and not w.endswith("HO"):
            v = v[:-1]
        if w in LW.EXC and v not in LW.EXC[w] and rng.rand() < 0.5:
            return LW.loan(w, rng)
        return v
    if w in LW.EXC:
        return LW.loan(w, rng)
    if len(w) == 1:
        return _letters(w) if rng.rand() < 0.85 else _PATCHED(self, w, rng)
    if not re.search("[AEIOUY]", w):
        return _letters(w) if len(w) <= 6 else LW.loan(w, rng)
    if is_korean(self, w):
        r = _korean_reading(self, w, rng)  # [k6 change 6]
        return r if r is not None else LW.loan(w, rng)
    rp = _route_pinyin(w, ctx)  # [k6 change 1 + 2]
    if rp >= 1.0 or (rp > 0.0 and rng.rand() < rp):
        return pinyin6(w, rng)
    if english_like(w):
        return LW.loan(w, rng)
    if _RR_DIGRAPH.search(w):  # [k6 change 6] RR-spelled word (BAENGNYEONG, SIMNYEON): all sound-change readings
        r = _korean_reading(self, w, rng)
    elif ctx is not None and ctx["kr"]:  # Korean-flagged name: only the ngn / ngm / mn reversals (BELLA keeps its ll)
        r = _korean_reading(self, w, rng, _KOREAN_ONLY)
    else:
        r = _PATCHED(self, w, rng)
    return r if r is not None else LW.loan(w, rng)


def name(self, catalog_name: str, rng) -> str | None:
    """kogen.Converter.name with the kogen6 tokeniser (same digit / spacing / 호 randomisation) and name context."""
    toks = _tokens(catalog_name)
    digits = [t for t in toks if t.isdigit()]
    words = [t for t in toks if not t.isdigit() and t not in ("NO",)]
    if not words or len(digits) > 1 or any(not t.isalpha() for t in words):
        return None
    ctx = _context(words, catalog_name)
    out = []
    try:
        for i, w in enumerate(words):
            self._k6ctx = dict(ctx, roman_i=(w == "I" and i > 0))
            k = self.word(w, rng)
            if k is None:
                return None
            out.append(k)
    finally:
        self._k6ctx = None
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
