"""Hangul loanword spellings for Latin-script vessel-name words (round4, used by kogen5).

loan(word, rng)    -> Hangul spelling of one word (str) or None (empty / non-letter input)
    1. EXC: hand-written spellings of the most frequent catalog words (first variant preferred; spellings
       the port data uses in the 382 real train Hangul names win, e.g. SUNNY 써니, SPIRIT 스피리트/스피릿)
    2. English words (catalog words found in the system English word list, cached in loan_en.txt):
       rule-based grapheme -> phoneme -> Hangul following 외래어 표기법 (silent e, -tion 션, -ture 처,
       r-coloured vowels er/ar/or -> 어/아/오, ch/sh/th/ph, x -> ks, short-vowel 받침 for p/t/k,
       intervocalic l -> ㄹㄹ, final m/n/ng/l as 받침, other finals with 으/이)
    3. anything else: a 'continental' (Latin-vowel) reading of the same engine (a 아 e 에 i 이 o 오 u 우,
       final r -> 르, fu -> 후, tsu -> 쓰), which is how Japanese / Spanish / Italian names are written.
pinyin(word, rng)  -> port-style Hangul reading of a Chinese pinyin word (1-4 syllables) or None.
Randomness (a few alternative spellings) is driven only by the passed numpy RandomState."""
from __future__ import annotations

import re
from pathlib import Path

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def _compose(c: str, v: str, f: str = "") -> str:
    return chr(0xAC00 + CHO.index(c) * 588 + JUNG.index(v) * 28 + JONG.index(f))


# ---------------------------------------------------------------- exceptions (most frequent catalog words)
_EXC_SRC = """
ACE 에이스 | ADMIRAL 어드마이럴 애드미럴 | ADVANCE 어드밴스 | ADVANTAGE 어드밴티지 | ADVENTURE 어드벤처 | ADVENTURER 어드벤처러
AFRICAN 아프리칸 아프리카 | AIR 에어 | AIRPORT 에어포트 | ALBATROSS 알바트로스 | ALLIANCE 얼라이언스 | ALPHA 알파
AMBER 앰버 엠버 | AMBITION 앰비션 | AMETHYST 애미시스트 | ANGEL 엔젤 앤젤 | AQUA 아쿠아 | AQUAMARINE 아쿠아마린 | ARCTIC 아크틱
ARROW 애로우 에로우 | ASIA 아시아 | ASIAN 아시안 아세안 | ATLANTIC 아틀란틱 애틀랜틱 | ATLAS 아틀라스 | AURORA 오로라 | AZALEA 아젤리아
BALTIC 발틱 | BAY 베이 | BEACH 비치 | BEAUTY 뷰티 | BELL 벨 | BLESS 블레스 | BLESSING 블레싱 | BLOSSOM 블로섬 | BLUE 블루
BOW 바우 보우 | BRAVE 브레이브 | BREEZE 브리즈 | BRIDGE 브리지 브릿지 | BRIGHT 브라이트 | BRILLIANT 브릴리언트 | BULK 벌크
BULKER 벌커 | CAPE 케이프 | CAPTAIN 캡틴 | CARGO 카고 | CARRIER 캐리어 | CARRY 캐리 | CASTLE 캐슬 | CAR 카 | CEMENT 시멘트
CENTURY 센추리 | CHALLENGE 챌린지 | CHALLENGER 챌린저 | CHAMPION 챔피언 | CHEMI 케미 | CHEMICAL 케미칼 케미컬 | CHEMIST 케미스트
CHERRY 체리 | CHIEF 치프 | CHINA 차이나 | CITY 시티 | CLEAN 클린 | CLIPPER 클리퍼 크리퍼 | CLOUD 클라우드 | COAST 코스트 | COASTAL 코스탈
COMET 코멧 | COMMANDER 커맨더 코맨더 | CONCORD 콩코드 | CONDOR 콘도르 콘도 | CONFIDENCE 컨피던스 | CONFIDENT 컨피던트
CONTINENTAL 콘티넨탈 | CORAL 코랄 | COSMOS 코스모스 | COUGAR 쿠거 | COURAGE 커리지 | COURIER 쿠리어 | CREST 크레스트
CROWN 크라운 | CRUISE 크루즈 | CRYSTAL 크리스탈 | DAISY 데이지 | DAWN 돈 던 | DAYLIGHT 데이라이트 | DEEP 딥 | DELIGHT 딜라이트
DELTA 델타 | DEMOCRACY 데모크라시 | DESTINY 데스티니 | DIAMOND 다이아몬드 | DIGNITY 디그니티 | DISCOVERY 디스커버리
DOLPHIN 돌핀 | DOVE 도브 | DRAGON 드라곤 드래곤 | DREAM 드림 | DREAMS 드림스 드림즈 | DRIVE 드라이브 | DUKE 듀크
EAGLE 이글 | EARTH 어스 | EAST 이스트 | EASTERN 이스턴 | ECO 에코 | EIGHT 에이트 | EMERALD 에메랄드 | EMPEROR 엠퍼러
EMPRESS 엠프레스 | ENDEAVOR 인데버 | ENDURANCE 인듀어런스 | ENERGY 에너지 | ENTERPRISE 엔터프라이즈 | EPIC 에픽
ETERNAL 이터널 | ETERNITY 이터니티 | EVER 에버 | EVOLUTION 에볼루션 | EXCELLENCE 엑설런스 | EXPLORER 익스플로러 익스플러어 | EXPRESS 익스프레스
FAITH 페이스 | FAIR 페어 | FALCON 팔콘 | FAMILY 패밀리 | FAST 패스트 | FAVOR 페이버 | FEDERAL 페더럴 | FERRY 페리 | FINE 파인
FIRST 퍼스트 | FLOWER 플라워 | FLYING 플라잉 | FOREST 포레스트 | FOREVER 포에버 | FORTUNE 포춘 포천 | FREEDOM 프리덤
FREEWAY 프리웨이 | FRIENDSHIP 프렌드십 | FRONT 프런트 프론트 | FRONTIER 프런티어 프론티어 | FUTURE 퓨처 | GALAXY 갤럭시
GARNET 가넷 | GAS 가스 | GEM 젬 | GENERAL 제너럴 | GENESIS 제네시스 | GIANT 자이언트 | GLOBAL 글로벌 | GLOBE 글로브 | GLORIOUS 글로리어스
GLORY 글로리 | GOLD 골드 | GOLDEN 골든 | GOOD 굿 | GOODWILL 굿윌 | GRACE 그레이스 | GRAND 그랜드 | GREAT 그레이트 | GREEN 그린
GUARDIAN 가디언 | GULF 걸프 | HAPPY 해피 | HARBOR 하버 | HARBOUR 하버 | HARMONY 하모니 | HARVEST 하베스트 | HAWK 호크
HERO 히어로 | HIGH 하이 | HIGHWAY 하이웨이 | HONOR 오너 아너 | HONOUR 오너 | HOPE 호프 | HORIZON 호라이즌 | HUNTER 헌터
ICE 아이스 | INFINITY 인피니티 | INTEGRITY 인테그리티 | IRIS 아이리스 | ISLAND 아일랜드 | ISLANDER 아일랜더 | ISLE 아일
JADE 제이드 | JASMINE 자스민 재스민 | JEWEL 주얼 | JOURNEY 저니 | JOY 조이 | JUSTICE 저스티스 | KING 킹 | KNIGHT 나이트
LADY 레이디 | LAKE 레이크 | LAND 랜드 | LEADER 리더 | LEGACY 레거시 | LEGEND 레전드 | LIBERTY 리버티 | LIGHT 라이트 | LILY 릴리
LINE 라인 | LINK 링크 | LION 라이온 | LOGISTICS 로지스틱스 | LOTUS 로터스 | LUCK 럭 | LUCKY 럭키 | MAGIC 매직 | MAJESTIC 머제스틱 마제스틱
MAJESTY 마제스티 | MAPLE 메이플 | MARINE 마린 | MARINER 마리너 | MARITIME 마리타임 | MASTER 마스터 마스타 | MELODY 멜로디
MERCHANT 머천트 | MERCURY 머큐리 | MERIDIAN 메리디안 | MERMAID 머메이드 | MIRACLE 미라클 | MISS 미스 | MOON 문
MORNING 모닝 | MOUNT 마운트 | NAVIGATOR 내비게이터 네비게이터 | NEW 뉴 | NOBLE 노블 | NORTH 노스 | NORTHERN 노던
OCEAN 오션 | OCEANIC 오셔닉 | OFFSHORE 오프쇼어 | OIL 오일 | ONE 원 | ONYX 오닉스 | OPAL 오팔 | ORANGE 오렌지
ORCHID 오키드 | ORIENT 오리엔트 | ORIENTAL 오리엔탈 | PALM 팜 | PARADISE 파라다이스 | PARTNER 파트너 | PASSION 패션
PATRIOT 패트리어트 | PEACE 피스 피이스 | PEARL 펄 | PELICAN 펠리칸 | PHOENIX 피닉스 | PINK 핑크 | PIONEER 파이오니어 파이오니아
POINT 포인트 | POLAR 폴라 | PORT 포트 | POST 포스트 | POWER 파워 | PRESTIGE 프레스티지 | PRETTY 프리티 | PRIDE 프라이드
PRIME 프라임 | PRINCE 프린스 | PRINCESS 프린세스 | PROGRESS 프로그레스 | PROMISE 프로미스 | PROSPERITY 프로스페리티 프라스페리티
PROVIDER 프로바이더 | QUEEN 퀸 | QUEST 퀘스트 | RAINBOW 레인보우 | RANGER 레인저 | RAY 레이 | RED 레드 | REEFER 리퍼
RELIANCE 릴라이언스 | RICH 리치 | RISING 라이징 | RIVER 리버 | ROAD 로드 | ROCK 록 락 | ROSE 로즈 | ROSEMARY 로즈마리
ROYAL 로얄 로열 | RUBY 루비 | RUNNER 러너 | SAFETY 세이프티 | SAILOR 세일러 | SAINT 세인트 | SAPPHIRE 사파이어 | SEA 씨 시
SEAS 씨즈 시즈 | SERENITY 세레니티 | SERVICE 서비스 | SEVEN 세븐 | SHINE 샤인 | SHINING 샤이닝 | SHIP 쉽 십 | SHIPPING 쉬핑 시핑
SILVER 실버 | SKY 스카이 | SMILE 스마일 | SOLAR 솔라 | SOUTH 사우스 | SOUTHERN 서던 | SPIRIT 스피리트 스피릿 | SPLENDOR 스플렌더
SPRING 스프링 | STAR 스타 | STARS 스타즈 스타스 | STEEL 스틸 | STREAM 스트림 | SUCCESS 석세스 | SUMMIT 서밋 | SUN 썬 선
SUNFLOWER 썬플라워 선플라워 | SUNNY 써니 서니 | SUNRISE 선라이즈 썬라이즈 | SUNSHINE 선샤인 썬샤인 | SUPER 슈퍼 | SUPREME 슈프림
SWAN 스완 | SWIFT 스위프트 | SYMPHONY 심포니 | TALENT 탤런트 | TANKER 탱커 | THE 더 | TIDE 타이드 | TIGER 타이거 | TOP 탑
TOPAZ 토파즈 | TRADER 트레이더 | TRANSPORT 트랜스포트 | TRANSPORTER 트랜스포터 | TREASURE 트레저 | TRINITY 트리니티
TRIUMPH 트라이엄프 | TRUST 트러스트 | TULIP 튤립 | TWO 투 | UNION 유니온 유니언 | UNITED 유나이티드 | UNITY 유니티
UNIVERSAL 유니버설 | UNIVERSE 유니버스 | VALIANT 밸리언트 | VALOR 밸러 | VANGUARD 뱅가드 | VENTURE 벤처 | VENUS 비너스
VICTOR 빅터 | VICTORIA 빅토리아 | VICTORY 빅토리 | VIKING 바이킹 | VISION 비전 | VOYAGER 보이저 보이져 | WARRIOR 워리어
WATER 워터 | WAVE 웨이브 | WAY 웨이 | WEALTH 웰스 | WEST 웨스트 | WESTERN 웨스턴 | WHITE 화이트 | WIN 윈 | WIND 윈드
WINNER 위너 | WINNING 위닝 | WINGS 윙스 | WISDOM 위즈덤 | WORLD 월드 | YOUNG 영 | ZENITH 제니스 | ZONE 존
HMM 에이치엠엠 | GLOVIS 글로비스 | HYUNDAI 현대 | HANJIN 한진 | PAN 팬 | PANSTAR 팬스타 | PANCON 팬콘 | KMTC 케이엠티씨
SINOKOR 시노코 | POS 포스 | ST 세인트 에스티 | MAERSK 머스크 | COSCO 코스코 | EVERGREEN 에버그린 | MSC 엠에스씨 | ONE 원
HAPAG 하팍 | LLOYD 로이드 | MOBY 모비 | DICK 딕 | TINA 티나 | JULIA 줄리아 | OLIVIA 올리비아 | CECILIE 세실리에 | CELINE 셀린
CAMILLA 카밀라 | CORNELIA 코넬리아 | CORDELIA 코델리아 | GENEVA 제네바 | BRAZIL 브라질 | GOSOK 고속 | NONGHYUP 농협
LONG 롱 | BEACH 비치 | HAVRE 아브르 | LE 르 | ALGECIRAS 알헤시라스 | GDANSK 그단스크 | HAMBURG 함부르크 | COPENHAGEN 코펜하겐
SOUTHAMPTON 사우샘프턴 | STOCKHOLM 스톡홀름 | PETERSBURG 페테르부르크 | TACOMA 타코마 | OAKLAND 오클랜드 | VANCOUVER 밴쿠버
CEBU 세부 | DAVAO 다바오 | DHAKA 다카 | MUNDRA 문드라 | NHAVA 나바 | SHEVA 셰바 | PORTLAND 포틀랜드 | QINGDAO 칭다오
AUCKLAND 오클랜드 | HEDLAND 헤들랜드 | SINGAPORE 싱가포르 | ROTTERDAM 로테르담 | ANTWERP 앤트워프 | MANILA 마닐라
BANGKOK 방콕 | TOKYO 도쿄 | OSAKA 오사카 | KOBE 고베 | NAGOYA 나고야 | YOKOHAMA 요코하마 | TSUSHIMA 쓰시마
LONDON 런던 | PARIS 파리 | AMSTERDAM 암스테르담 | BARCELONA 바르셀로나 | SANTA 산타 | MARIA 마리아 | ANNA 안나
CHEMICARRY 케미캐리 | KOREX 코렉스 | PHAROSTAR 파로스타 | GOLDSTAR 골드스타 | SEAWORLD 씨월드 | SALTLAKE 솔트레이크
SUNCRUISER 썬크루저 | BLUEBELL 블루벨 | CARNATION 카네이션 | BRITANNIA 브리타니아 | TIFFANY 티파니 | BROADWAY 브로드웨이
BRIDGEWATER 브리지워터 | CARAVAN 캐러밴 카라반 | DEMOCRACY 데모크라시 | COMMODORE 코모도 | KOSOK 고속 | PANSY 팬지
LILAC 라일락 | FRANK 프랭크 | BROWN 브라운 | TRITON 트리톤 | PANTHER 팬서 | MARKETER 마케터 | CREATOR 크리에이터
UNIVERSAL 유니버설 | MANN 만 맨 | CLES 클레스 | SPRUCE 스프루스 | FIDES 피데스 | CLIO 클리오 | VESTA 베스타 | FREYA 프레야
BRONZE 브론즈 | MERRY 메리 | BONNY 보니 | PONY 포니 | ZINNA 지나 | DAHLIA 달리아 | CONCERT 콘서트 | CONDUCTOR 컨덕터
CARA 카라 | GENIE 지니 | CALM 캄 | TERESA 테레사 | SARAH 사라 | EVITANIA 에비타니아 | TRITANIA 트리타니아
"""
EXC: dict[str, list[str]] = {}
for _item in _EXC_SRC.replace("\n", "|").split("|"):
    _p = _item.split()
    if len(_p) >= 2:
        EXC.setdefault(_p[0], _p[1:])

_EN_FILE = Path(__file__).resolve().parent / "loan_en.txt"
try:
    EN_WORDS = frozenset(_EN_FILE.read_text().split())
except OSError:  # pragma: no cover - fallback when the cached list is missing
    try:
        EN_WORDS = frozenset(l.strip().upper() for l in open("/usr/share/dict/web2") if l.strip().islower())
    except OSError:
        EN_WORDS = frozenset()


def is_english(w: str) -> bool:
    """English-dictionary word (or a hand-spelled one). Dictionary words ending in a / i / u (MARU, FUJI, SAKURA,
    ALPHA, NOVA) are read with Latin vowels instead: that is right for the foreign ones and the same for the rest."""
    w = w.upper()
    return w in EXC or (w in EN_WORDS and w[-1] not in "AIU")


# ---------------------------------------------------------------- phoneme units -> Hangul
# unit: ("C", sym) | ("V", [medials], short)
CHO_OF = {"p": "ㅍ", "b": "ㅂ", "t": "ㅌ", "d": "ㄷ", "k": "ㅋ", "g": "ㄱ", "f": "ㅍ", "v": "ㅂ", "s": "ㅅ", "ss": "ㅆ",
          "z": "ㅈ", "sh": "ㅅ", "zh": "ㅈ", "ch": "ㅊ", "j": "ㅈ", "th": "ㅅ", "dh": "ㄷ", "m": "ㅁ", "n": "ㄴ",
          "ng": "ㅇ", "l": "ㄹ", "r": "ㄹ", "h": "ㅎ", "ts": "ㅊ", "tt": "ㄸ", "kk": "ㄲ", "pp": "ㅃ", "jj": "ㅉ"}
FINAL_SYL = {"p": "프", "b": "브", "t": "트", "d": "드", "k": "크", "g": "그", "f": "프", "v": "브", "s": "스", "ss": "스",
             "z": "즈", "th": "스", "dh": "드", "ts": "츠", "ch": "치", "j": "지", "zh": "지", "sh": "시", "h": "",
             "r": "", "m": "므", "n": "느", "ng": "응", "l": "르", "tt": "트", "kk": "크", "pp": "프", "jj": "치"}
STOP_JONG = {"p": "ㅂ", "t": "ㅅ", "k": "ㄱ", "pp": "ㅂ", "tt": "ㅅ", "kk": "ㄱ"}
NASAL_JONG = {"m": "ㅁ", "n": "ㄴ", "ng": "ㅇ", "l": "ㄹ"}
W_MERGE = {"ㅏ": "ㅘ", "ㅓ": "ㅝ", "ㅗ": "ㅝ", "ㅣ": "ㅟ", "ㅔ": "ㅞ", "ㅐ": "ㅙ", "ㅜ": "ㅜ", "ㅡ": "ㅜ", "ㅠ": "ㅠ"}
Y_MERGE = {"ㅏ": "ㅑ", "ㅓ": "ㅕ", "ㅗ": "ㅛ", "ㅜ": "ㅠ", "ㅔ": "ㅖ", "ㅐ": "ㅒ", "ㅣ": "ㅣ", "ㅡ": "ㅣ"}
Y_COLLAPSE = {"ㅑ": "ㅏ", "ㅕ": "ㅓ", "ㅛ": "ㅗ", "ㅠ": "ㅜ", "ㅖ": "ㅔ", "ㅒ": "ㅐ"}


def hangulize(units: list, rng=None, cont: bool = False) -> str:
    """Phoneme units -> Hangul syllables (외래어 표기법 style syllabification)."""
    syls: list[list[str]] = []  # [cho, jung, jong]
    onset = None   # consonant waiting for its vowel
    glide = None   # 'w' / 'y' waiting for its vowel
    prev = None    # kind of the previous unit: 'V' / 'C'
    prev_short = False
    n = len(units)

    def push(s: str):
        for ch in s:
            code = ord(ch) - 0xAC00
            syls.append([CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]])

    def add_final(c: str, nxt):
        """c is not followed by a vowel: 받침 or a syllable with 으/이."""
        if c in NASAL_JONG:
            if syls and not syls[-1][2] and (prev == "V" or c != "l" or True):
                syls[-1][2] = NASAL_JONG[c]
                return
            if c in ("m", "n") and syls and syls[-1][2] == "ㄹ":
                syls.append(["ㄹ", "ㅡ", NASAL_JONG[c]])
                return
            push(FINAL_SYL[c])
            return
        if c in STOP_JONG and prev == "V" and prev_short and syls and not syls[-1][2] and (
                nxt is None or (nxt[0] == "C" and nxt[1] not in ("l", "r", "m", "n", "w", "y", "h"))):
            syls[-1][2] = STOP_JONG[c]
            return
        if c == "r":
            if cont and prev == "V":
                push("르")
            return
        if c == "h":
            return
        if c == "w":
            push("우"); return
        if c == "y":
            push("이"); return
        if c == "sh" and nxt is not None:
            push("슈"); return
        push(FINAL_SYL.get(c, ""))

    for i, u in enumerate(units):
        nxt = units[i + 1] if i + 1 < n else None
        if u[0] == "V":
            meds = list(u[1])
            c = onset
            m0 = meds[0]
            if glide == "y" and syls and not onset and syls[-1][1] in ("ㅣ", "ㅢ", "ㅟ") and not syls[-1][2]:
                glide = None  # player, mayor: the y is already in the diphthong
            if glide == "w":
                m0 = W_MERGE.get(m0, m0)
            elif glide == "y":
                m0 = Y_MERGE.get(m0, m0)
            if c == "sh":
                m0 = Y_MERGE.get(m0, m0)
            cho = CHO_OF[c] if c else "ㅇ"
            if c == "ts" and cont and m0 == "ㅜ":
                cho, m0 = "ㅆ", "ㅡ"  # Japanese tsu
            if cho in ("ㅈ", "ㅊ", "ㅉ"):
                m0 = Y_COLLAPSE.get(m0, m0)
            if c == "ng":  # singer: ㅇ받침 + ㅇ onset
                if syls and not syls[-1][2]:
                    syls[-1][2] = "ㅇ"
                cho = "ㅇ"
            syls.append([cho, m0, ""])
            for m in meds[1:]:
                syls.append(["ㅇ", m, ""])
            onset = glide = None
            prev, prev_short = "V", bool(u[2]) and len(meds) == 1
            continue
        c = u[1]
        nv = nxt is not None and nxt[0] == "V"
        if c in ("w", "y"):
            if nv:
                if onset is not None and not (c == "w" and onset in ("k", "g", "h")):
                    add_final(onset, u); prev = "C"; onset = None
                glide = c
            else:
                if onset is not None:
                    add_final(onset, u); prev = "C"; onset = None
                add_final(c, nxt); prev = "C"
            continue
        if onset is not None:  # two consonants in a row: the earlier one closes its syllable
            add_final(onset, u)
            prev = "C"
            onset = None
        gl = nxt is not None and nxt[0] == "C" and nxt[1] in ("w", "y") and i + 2 < n and units[i + 2][0] == "V"
        if nv or (gl and c in ("k", "g", "h")):
            if c == "l" and syls and not syls[-1][2]:
                syls[-1][2] = "ㄹ"  # galaxy 갤럭시, glory 글로리
            onset = c
            continue
        add_final(c, nxt)
        prev = "C"
    if onset is not None:
        add_final(onset, None)
    return "".join(_compose(c if c in CHO else "ㅇ", v, f) for c, v, f in syls)


# ---------------------------------------------------------------- spelling -> units (English & continental)
VOW = set("aeiou")


def _graphemes(w: str, cont: bool):
    """Split a lowercase word into consonant / vowel letter groups [[kind, text]]."""
    n = len(w)

    def isv(j: int) -> bool:
        c = w[j]
        if c in VOW:
            if c == "u" and j > 0 and (w[j - 1] == "q" or (w[j - 1] == "g" and j + 1 < n and w[j + 1] in VOW and not cont)):
                return False  # qu, gu(ard): part of the consonant
            return True
        if c == "y":
            if j == 0:
                return False
            if w[j - 1] in VOW and j + 1 < n and w[j + 1] in VOW:
                return False  # player, voyager: glide
            return True
        if c == "w":
            return j > 0 and w[j - 1] in VOW and (j + 1 >= n or w[j + 1] not in VOW)
        return False

    out = []
    i = 0
    while i < n:
        v = isv(i)
        j = i + 1
        while j < n and isv(j) == v:
            j += 1
        out.append(["V" if v else "C", w[i:j]])
        i = j
    return out


def _choice(rng, opts, p=None):
    if rng is None:
        return opts[0]
    if p is None:
        return opts[rng.randint(len(opts))]
    return opts[int(rng.choice(len(opts), p=p))]


def _cons_units(cl: str, *, rest: str, initial: bool, prevv: str, cont: bool, rng, soft_en: bool = False) -> list:
    """Consonant letter cluster -> consonant symbols. rest: the letters after the cluster."""
    out = []
    i, n = 0, len(cl)
    while i < n:
        r = cl[i:]
        after = cl[i + 1:] + rest  # letters after the current letter
        if r.startswith("tch"):
            out.append("ch"); i += 3; continue
        if r.startswith("sch"):
            out += ["sh"] if cont else ["s", "k"]; i += 3; continue
        if r.startswith("pph"):
            out.append("f"); i += 3; continue
        if r.startswith("ch"):
            a2 = cl[i + 2:] + rest
            if not cont and (a2[:1] in ("r", "l") or (initial and i == 0 and a2[:2] in ("em", "or", "ao", "ol", "ar", "ri", "ro", "ry"))
                             or (i + 2 == n and not rest and prevv == "e")):
                out.append("k")
            else:
                out.append("ch")
            i += 2; continue
        if r.startswith("sh"):
            out.append("sh"); i += 2; continue
        if r.startswith("zh"):
            out.append("j"); i += 2; continue
        if r.startswith("th"):
            a2 = cl[i + 2:] + rest
            if cont:
                out.append("t")
            elif not (initial and i == 0) and a2.startswith("er"):
                out.append("dh")  # northern, mother
            else:
                out.append("th")
            i += 2; continue
        if r.startswith("ph"):
            out.append("f"); i += 2; continue
        if r.startswith("gh"):
            if initial and i == 0:
                out.append("g")
            i += 2; continue
        if r.startswith("ck"):
            out += ["k", "k"] if (rest[:1] in VOW or rest[:1] == "y") and not cont else ["k"]; i += 2; continue
        if r.startswith("ng"):
            a2 = cl[i + 2:] + rest
            if (not cont or soft_en) and a2[:1] in ("e", "i", "y") and prevv[-1:] in ("a", "e") and not a2.startswith("ing"):
                out += ["n", "j"]  # change, orange, challenger, angel
            elif cont and a2[:1] in VOW:
                out += ["n", "g"]  # Japanese / Spanish: gin-ga 긴가, tan-go 탄고
            elif not cont and a2[:1] in VOW | {"l", "r"} and not (a2.startswith("er") or a2.startswith("ing")):
                out += ["ng", "g"]  # tango 탱고, angle 앵글, mango (but singer 싱어)
            else:
                out.append("ng")
            i += 2; continue
        if r.startswith("nk"):
            out += ["ng", "k"]; i += 2; continue
        if r.startswith("dg"):
            out.append("j"); i += 2; continue
        if r.startswith("wh"):
            out += ["h", "w"]; i += 2; continue
        if (r.startswith("wr") or r.startswith("kn") or r.startswith("gn")) and initial and i == 0:
            out.append("r" if r[0] == "w" else "n"); i += 2; continue
        if r.startswith("gn") and i + 2 == n and not rest:
            out.append("n"); i += 2; continue
        if (r.startswith("mb") or r.startswith("mn")) and i + 2 == n and not rest:
            out.append("m"); i += 2; continue
        if not cont and r[:2] in ("lm", "lk") and prevv in ("a", "o") and i == 0:
            i += 1; continue  # palm, walk: silent l
        if r.startswith("qu") or r == "q":
            nx = rest[:1]
            out += ["k"] if (cont and nx in ("e", "i")) else ["k", "w"]
            i += 2 if r.startswith("qu") else 1; continue
        if r.startswith("gu") and not cont:
            out.append("g"); i += 2; continue
        if r.startswith("ts"):
            out.append("ts"); i += 2; continue
        if r[:2] in ("dh", "bh", "kh", "gh", "jh"):
            out.append(r[0]); i += 2; continue
        if r.startswith("cc"):
            out += ["k", "s"] if after[1:2] in ("e", "i", "y") else ["k"]; i += 2; continue
        if r.startswith("sc") and after[1:2] in ("e", "i", "y"):
            out.append("s"); i += 2; continue
        if len(r) >= 2 and r[0] == r[1] and r[0] in "bdfgklmnprstvz":
            c = r[0]
            if cont and c in "tkpnm" and rest[:1] in VOW:  # geminates: 받침 + onset (Hokkaido, Anna)
                out += [c, c]
            else:
                out.append(c)
            i += 2; continue
        c = r[0]
        nx = after[:1]
        if c == "c":
            out.append("s" if nx in ("e", "i", "y") else "k")
        elif c == "g":
            soft = nx in ("e", "i", "y") and not (initial and i == 0 and after[:2] in ("et", "ir", "if", "iv", "ea", "ee", "ey", "ig"))
            out.append("j" if (soft and (not cont or soft_en)) else "g")
        elif c == "x":
            out += (["s"] if cont else ["z"]) if (initial and i == 0) else ["k", "s"]
        elif c == "q":
            out.append("k")
        elif c == "s":
            if not cont and n == 1 and prevv and rest[:1] in VOW | {"y"} and not rest.startswith("u") \
                    and not rest.startswith("io") and (rng is None or rng.rand() < 0.75):
                out.append("z")  # rose, rise, cruise, music
            else:
                out.append("s")
        elif c == "f" and cont and nx == "u":
            out.append("h")  # Japanese fu
        elif c in VOW:
            pass  # the u of qu / gu inside a cluster (vanguard)
        else:
            out.append(c)
        i += 1
    return out


def _vowel(g: str, cx: dict, rng):
    """English vowel letter group -> (medials, short). cx: context flags."""
    pc, nc, last, first, nsyl = cx["prev_c"], cx["nxt_c"], cx["last"], cx["first"], cx["nsyl"]
    after_silent, open_, magic, closed, rcol = cx["after_silent"], cx["open"], cx["magic"], cx["closed"], cx["rcol"]
    after = cx["after"]
    mono = nsyl == 1
    yu = ["ㅜ"] if (pc[-1:] in ("r", "l", "j") or pc.endswith("ch") or pc.endswith("sh")) else ["ㅠ"]
    if len(g) == 1:
        if g == "a":
            if pc.endswith("w") or pc.endswith("qu"):
                if magic:
                    return ["ㅔ", "ㅣ"], False
                return (["ㅓ"] if rcol else _choice(rng, [["ㅓ"], ["ㅏ"]], [.6, .4])), closed
            if rcol:
                return (["ㅔ", "ㅓ"] if after_silent and nc == "r" else ["ㅏ"]), False
            if magic:
                return ["ㅔ", "ㅣ"], False
            if not nc and last:
                return ["ㅏ"], False
            if last and not first and nc in ("l", "n", "nt", "nce", "ns", "m", "nd", "s"):
                return _choice(rng, [["ㅓ"], ["ㅏ"]], [.5, .5]), True
            if closed:
                if first and nsyl >= 2 and not pc and nc[:1] in "dclfpstb":
                    return _choice(rng, [["ㅐ"], ["ㅏ"], ["ㅓ"]], [.45, .3, .25]), True
                return _choice(rng, [["ㅐ"], ["ㅏ"]], [.55, .45]), True
            if nc == "r":  # maria, marine, paradise, caravan
                return _choice(rng, [["ㅏ"], ["ㅐ"]], [.7, .3]), False
            if first and (nsyl == 2 or after.startswith("_shon_")):
                return _choice(rng, [["ㅔ", "ㅣ"], ["ㅐ"], ["ㅏ"]], [.55, .25, .2]), False
            if first:
                return _choice(rng, [["ㅐ"], ["ㅏ"], ["ㅔ", "ㅣ"]], [.45, .4, .15]), False
            return _choice(rng, [["ㅏ"], ["ㅓ"]], [.6, .4]), False
        if g == "e":
            if rcol:
                return (["ㅣ", "ㅓ"] if after_silent and nc == "r" else ["ㅓ"]), False
            if magic or (not nc and last):
                return ["ㅣ"], False
            if last and not first:
                if nc == "n" and pc[-1:] in ("d", "t", "v"):
                    return _choice(rng, [["ㅡ"], ["ㅔ"]], [.7, .3]), True
                if nc in ("nt", "nce", "nse", "ncy"):
                    return _choice(rng, [["ㅓ"], ["ㅔ"]], [.6, .4]), True
                if nc == "t":
                    return _choice(rng, [["ㅔ"], ["ㅣ"]], [.6, .4]), True
            if first and not pc and nc.startswith("x"):
                return ["ㅣ"], True
            if open_ and first:
                return _choice(rng, [["ㅔ"], ["ㅣ"]], [.75, .25]), False
            return ["ㅔ"], closed
        if g == "i":
            if rcol:
                return (["ㅏ", "ㅣ", "ㅓ"] if after_silent and nc == "r" else ["ㅓ"]), False
            if nc.startswith("gh"):
                return ["ㅏ", "ㅣ"], False
            if magic:
                if nsyl > 1 and nc in ("n", "v", "c", "s") and not (nc == "s" and pc[-1:] in ("r", "d")):
                    return ["ㅣ"], True
                if nsyl > 1 and nc == "t" and (rng is None or rng.rand() < 0.5):
                    return ["ㅣ"], True  # elite / favorite (vs dynamite)
                return ["ㅏ", "ㅣ"], False
            if mono and nc in ("ld", "nd"):
                return ["ㅏ", "ㅣ"], False
            if not nc and last:
                return ["ㅣ"], False
            if open_:
                if first and not pc:
                    return _choice(rng, [["ㅏ", "ㅣ"], ["ㅣ"]], [.6, .4]), False
                return _choice(rng, [["ㅣ"], ["ㅏ", "ㅣ"]], [.75, .25] if first else [.85, .15]), False
            return ["ㅣ"], closed
        if g == "o":
            if rcol:
                if pc.endswith("w"):
                    return ["ㅓ"], False
                if after_silent:
                    return _choice(rng, [["ㅗ"], ["ㅗ", "ㅓ"]], [.6, .4]), False
                if last and not first and nc == "r":
                    return _choice(rng, [["ㅓ"], ["ㅗ"]], [.6, .4]), False
                return ["ㅗ"], False
            if last and not first:
                if nc == "m":
                    return _choice(rng, [["ㅓ"], ["ㅗ"]], [.6, .4]), True
                if nc == "n" and pc[-1:] in ("s", "z"):
                    return _choice(rng, [["ㅡ"], ["ㅗ"]], [.5, .5]), True
            if first and nsyl >= 2 and closed and nc[:1] in ("n", "m") and pc in ("c", "k"):
                return _choice(rng, [["ㅗ"], ["ㅓ"]], [.6, .4]), True
            return ["ㅗ"], closed and not magic
        if g == "u":
            if rcol:
                if after_silent and nc == "r":
                    return (["ㅓ"] if pc[-1:] in ("t", "d") or pc in ("ch", "zh") else ["ㅠ", "ㅓ"]), False
                return ["ㅓ"], False
            if magic or (not nc and last):
                return yu, False
            if nc.startswith("ll") or (nc.startswith("sh") and pc[-1:] in ("p", "b", "f")) or (nc == "t" and pc == "p"):
                return ["ㅜ"], True
            if nc == "r" and open_:
                return yu, False
            if closed:
                return ["ㅓ"], True
            if first and not pc:
                return ["ㅠ"], False
            if first:
                return yu, False
            return _choice(rng, [["ㅓ"], yu], [.5, .5]), False
        if g == "y":
            if rcol:
                return ["ㅓ"], False
            if (mono and (not nc or magic)) or magic or (open_ and first and pc):
                return ["ㅏ", "ㅣ"], False
            return ["ㅣ"], closed
        return ["ㅡ"], closed
    # ---- digraphs
    if g in ("ee",):
        if rcol:
            return ["ㅣ", "ㅓ"], False
        return ["ㅣ"], bool(last and nc in ("p", "t", "k") and (rng is None or rng.rand() < 0.5))
    if g == "ea":
        if rcol:
            return (["ㅓ"] if len(nc) > 1 else ["ㅣ", "ㅓ"]), False
        if (nc == "d" and (not after or after == "y")) or nc in ("th", "lth", "v", "lt", "ven") or (nc == "s" and after.startswith("u")):
            return ["ㅔ"], True
        return ["ㅣ"], bool(last and nc in ("p", "t", "k") and (rng is None or rng.rand() < 0.4))
    if g in ("ai", "ay"):
        if rcol:
            return ["ㅔ", "ㅓ"], False
        if g == "ai" and nc == "n" and last and not first and pc[-1:] == "t":
            return ["ㅣ"], True  # captain, mountain
        return ["ㅔ", "ㅣ"], False
    if g in ("ei", "ey"):
        if rcol:
            return ["ㅔ", "ㅓ"], False
        if g == "ey" and last and not nc and not mono:
            return ["ㅣ"], False
        return ["ㅔ", "ㅣ"], False
    if g == "ie":
        if rcol:
            return ["ㅣ", "ㅓ"], False
        if last and not nc:
            return (["ㅏ", "ㅣ"] if mono else ["ㅣ"]), False
        if not mono and nc in ("nt", "nce", "t", "ty"):
            return ["ㅣ", "ㅔ"], False  # orient, client, quiet
        return ["ㅣ"], False
    if g == "oo":
        if rcol:
            return ["ㅗ"], False
        return ["ㅜ"], nc[:1] in ("k", "d", "t")
    if g == "ou":
        if rcol:
            if mono:
                return (["ㅗ"] if pc in ("f", "p", "y", "c") else ["ㅏ", "ㅜ", "ㅓ"]), False
            return ["ㅓ"], False
        if nc == "r" or nc.startswith("ng") or nc.startswith("ch") or nc in ("bl", "pl", "ntr", "gh") or nc.startswith("bl") or nc.startswith("pl"):
            return ["ㅓ"], True
        if nc.startswith("s") and last and not first:
            return ["ㅓ"], True  # glorious
        if not nc and last:
            return ["ㅜ"], False
        if (nc in ("g", "p", "t") and after) or nc == "p":
            return ["ㅜ"], False
        return ["ㅏ", "ㅜ"], False
    if g == "ow":
        if not nc and last:
            return (_choice(rng, [["ㅏ", "ㅜ"], ["ㅗ", "ㅜ"]]) if mono else ["ㅗ", "ㅜ"]), False
        return ["ㅏ", "ㅜ"], False
    if g == "oa":
        return ["ㅗ"], False
    if g in ("oi", "oy"):
        return ["ㅗ", "ㅣ"], False
    if g in ("au", "aw"):
        return ["ㅗ"], False
    if g in ("ew", "eu", "ue", "ieu"):
        return (["ㅜ"] if pc[-1:] in ("r", "l", "j") or pc.endswith("ch") else ["ㅠ"]), False
    if g == "ui":
        return ["ㅜ"], False
    if g in ("uy", "uye"):
        return ["ㅏ", "ㅣ"], False
    if g == "eau":
        return ["ㅗ"], False
    if g in ("ae",):
        return ["ㅔ"], False
    if g == "oe":
        return ["ㅗ"], False
    # ---- hiatus: letter by letter (asia, maria, pioneer, diamond, leo, champion)
    a, b = g[0], g[1:]
    if a == "i":
        m1 = ["ㅏ", "ㅣ"] if first else ["ㅣ"]
    elif a == "e":
        m1 = ["ㅔ"] if not first else _choice(rng, [["ㅔ"], ["ㅣ"]], [.7, .3])
    elif a == "u":
        m1 = yu
    else:
        m1 = {"a": ["ㅏ"], "o": ["ㅗ"], "y": ["ㅏ", "ㅣ"]}.get(a, ["ㅡ"])
    cx2 = dict(cx, prev_c="", first=False)
    if len(b) >= 2:
        m2, s2 = _vowel(b, cx2, rng)
    elif b == "o" and nc == "n" and last:
        m2, s2 = (["ㅓ"] if a == "i" else ["ㅗ"]), True  # champion, lion
    elif b == "a" and last and nc in ("n", "nt", "nce", "l"):
        m2, s2 = _choice(rng, [["ㅓ"], ["ㅏ"]], [.5, .5]), True
    elif b == "u" and last and nc == "s":
        m2, s2 = _choice(rng, [["ㅜ"], ["ㅓ"]], [.5, .5]), True
    elif b == "e" and not nc and last:
        m2, s2 = [], False
    elif b == "y":
        m2, s2 = ["ㅣ"], False
    else:
        m2, s2 = {"a": ["ㅏ"], "e": ["ㅔ"], "i": ["ㅣ"], "o": ["ㅗ"], "u": ["ㅜ"]}.get(b, ["ㅡ"]), closed
    return m1 + m2, s2 and len(m2) == 1


_CONT_V = {"a": "ㅏ", "e": "ㅔ", "i": "ㅣ", "o": "ㅗ", "u": "ㅜ", "y": "ㅣ", "w": "ㅜ"}
_CONT_D = {"ai": ["ㅏ", "ㅣ"], "ei": ["ㅔ", "ㅣ"], "au": ["ㅏ", "ㅜ"], "ou": ["ㅗ", "ㅜ"], "eu": ["ㅔ", "ㅜ"],
           "ae": ["ㅐ"], "oe": ["ㅗ", "ㅔ"], "ie": ["ㅣ", "ㅔ"], "ee": ["ㅣ"], "oo": ["ㅜ"], "ay": ["ㅏ", "ㅣ"],
           "ey": ["ㅔ", "ㅣ"], "oy": ["ㅗ", "ㅣ"], "uy": ["ㅜ", "ㅣ"], "aw": ["ㅏ", "ㅜ"], "ow": ["ㅗ", "ㅜ"], "ew": ["ㅠ"]}


def _cont_vowel(g: str) -> list:
    meds = []
    i = 0
    while i < len(g):
        if g[i:i + 2] in _CONT_D:
            meds += _CONT_D[g[i:i + 2]]; i += 2
        else:
            meds.append(_CONT_V.get(g[i], "ㅡ")); i += 1
    return meds


def _units(w: str, rng, cont: bool, soft: bool = False) -> list:
    gs = _graphemes(w, cont)
    vidx = [k for k, (t, _) in enumerate(gs) if t == "V"]
    if not vidx:
        return []
    silent_e = (not cont) and gs[-1] == ["V", "e"] and len(vidx) > 1
    if not cont and len(gs) >= 2 and gs[-1] == ["V", "e"] and len(vidx) > 1 and gs[-2][1].endswith("r") and len(gs[-2][1]) >= 2 \
            and gs[-2][1][-2] not in VOW:
        silent_e = False  # centre, acre: the e is pronounced (센터)
        gs[-1] = ["V", "_er"]
    pron = [k for k in vidx if not (silent_e and k == len(gs) - 1)]
    nsyl = len(pron)
    # -tion / -sion / -cian / -cean  ->  션 / 전 ; -ture -> 처 ; -sure -> 저
    if not cont:
        for k in range(len(gs) - 2):
            if gs[k][0] == "C" and gs[k + 1][0] == "V" and gs[k + 2][0] == "C" and gs[k + 2][1].startswith("n") \
                    and gs[k + 1][1] in ("io", "ia", "ea") and k + 3 >= len(gs) - (1 if gs[-1][1] in ("e",) else 0) \
                    and gs[k][1][-1:] in ("t", "s", "c"):
                gs[k + 1][1] = "_shon_" + ("z" if gs[k][1] == "s" and k > 0 else "")
    units = []
    vcount = 0
    for k, (t, s) in enumerate(gs):
        prevv = gs[k - 1][1] if k > 0 and gs[k - 1][0] == "V" else ""
        if t == "C":
            rest = "".join(x[1] for x in gs[k + 1:]).replace("_shon_z", "ion").replace("_shon_", "ion").replace("_er", "er")
            cs = _cons_units(s, rest=rest, initial=(k == 0), prevv=prevv, cont=cont, rng=rng, soft_en=soft)
            if not cont and k + 1 < len(gs) and gs[k + 1][1].startswith("_shon_") and cs:
                cs[-1] = "zh" if gs[k + 1][1].endswith("z") and len(s) == 1 else "sh"
            # -ture / -tur+V (century) / -sure
            if not cont and cs and k > 0 and k + 2 < len(gs) and gs[k + 1][1] == "u" and gs[k + 2][1].startswith("r") \
                    and not gs[k + 2][1].startswith("rr"):
                if cs[-1] == "t":
                    cs[-1] = "ch"
                elif cs[-1] in ("s", "z") and len(s) == 1:
                    cs[-1] = "zh"
            units += [("C", c) for c in cs]
            continue
        if silent_e and k == len(gs) - 1:
            continue
        vcount += 1
        if s.startswith("_shon_"):
            units.append(("V", ["ㅓ"], True)); continue
        if s == "_er":
            units.append(("V", ["ㅓ"], False)); continue
        nc = gs[k + 1][1] if k + 1 < len(gs) and gs[k + 1][0] == "C" else ""
        after = gs[k + 2][1] if k + 2 < len(gs) else ""
        after_silent = silent_e and k + 2 == len(gs) - 1
        if cont:
            units.append(("V", _cont_vowel(s), True)); continue
        single = len(nc) == 1 and nc not in ("x",) or nc in ("th", "sh", "ch", "ph")
        open_ = bool(nc) and single and bool(after) and not after_silent
        magic = len(s) == 1 and after_silent and ((single and nc not in ("w", "y", "r")) or nc in ("bl", "pl", "gl", "tl", "dl", "cl", "fl", "kl", "br", "tr"))
        if nc in ("bl", "pl", "gl", "tl", "dl", "cl", "fl", "kl", "br", "tr", "pr", "cr", "gr", "dr", "fr") and not after_silent and after and len(s) == 1:
            open_ = True  # noble-type clusters before a vowel
        rcol = nc.startswith("r") and not nc.startswith("rr") and (len(nc) > 1 or not after or after_silent)
        closed = bool(nc) and not open_ and not magic
        cx = dict(prev_c=gs[k - 1][1] if k > 0 and gs[k - 1][0] == "C" else "", nxt_c=nc, last=(k == pron[-1]),
                  first=(vcount == 1), nsyl=nsyl, after_silent=after_silent, open=open_, magic=magic, closed=closed,
                  rcol=rcol, after=after)
        # vowel + w + vowel (power, flower, jewel): the w is a glide for the next vowel
        if nc == "w" and after and not after_silent:
            m = {"o": ["ㅏ"], "a": ["ㅗ"], "e": (["ㅜ"] if cx["prev_c"][-1:] in ("r", "l", "j") else ["ㅠ"])}.get(s[-1:], None)
            if m and len(s) == 1:
                units.append(("V", m, False)); continue
        m, short = _vowel(s, cx, rng)
        if m:
            units.append(("V", m, short))
    # final s after a voiced sound -> z (stars, dreams)
    if not cont and len(units) >= 2 and units[-1] == ("C", "s") and w.endswith("s") and not w.endswith("ss") \
            and not w.endswith("us") and not w.endswith("is"):
        p = units[-2]
        if (p[0] == "C" and p[1] in ("n", "m", "l", "ng", "d", "b", "g", "v")) or (p[0] == "V" and not p[2]):
            if rng is None or rng.rand() < 0.6:
                units[-1] = ("C", "z")
    return units


def en_hangul(word: str, rng=None) -> str:
    w = re.sub("[^a-z]", "", word.lower())
    return hangulize(_units(w, rng, cont=False), rng) if w else ""


def cont_hangul(word: str, rng=None, soft: bool = False) -> str:
    """Latin-vowel reading; soft=True keeps English soft g (ANGELA 안젤라, GENEVA 제네바)."""
    w = re.sub("[^a-z]", "", word.lower())
    return hangulize(_units(w, rng, cont=True, soft=soft), rng, cont=True) if w else ""



# ---------------------------------------------------------------- pinyin
_PY_INI = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w", ""]
_PY_ONSET = {"b": "ㅂ", "p": "ㅍ", "m": "ㅁ", "f": "ㅍ", "d": "ㄷ", "t": "ㅌ", "n": "ㄴ", "l": "ㄹ", "g": "ㄱ", "k": "ㅋ",
             "h": "ㅎ", "j": "ㅈ", "q": "ㅊ", "x": "ㅅ", "zh": "ㅈ", "ch": "ㅊ", "sh": "ㅅ", "r": "ㄹ", "z": "ㅈ",
             "c": "ㅊ", "s": "ㅅ", "y": "ㅇ", "w": "ㅇ", "": "ㅇ"}
# final -> list of (variant medial list, jong, weight); first = literal port-style reading
_PY_FIN = {
    "a": [(["ㅏ"], "", 1)], "o": [(["ㅗ"], "", 1)], "e": [(["ㅔ"], "", .6), (["ㅓ"], "", .4)],
    "i": [(["ㅣ"], "", 1)], "u": [(["ㅜ"], "", 1)], "v": [(["ㅠ"], "", .6), (["ㅟ"], "", .4)],
    "ai": [(["ㅏ", "ㅣ"], "", 1)], "ei": [(["ㅔ", "ㅣ"], "", 1)], "ao": [(["ㅏ", "ㅗ"], "", 1)],
    "ou": [(["ㅓ", "ㅜ"], "", .5), (["ㅗ", "ㅜ"], "", .3), (["ㅗ"], "", .2)],
    "an": [(["ㅏ"], "ㄴ", 1)], "en": [(["ㅔ"], "ㄴ", .7), (["ㅓ"], "ㄴ", .3)], "ang": [(["ㅏ"], "ㅇ", 1)],
    "eng": [(["ㅔ"], "ㅇ", .7), (["ㅓ"], "ㅇ", .3)], "ong": [(["ㅗ"], "ㅇ", .65), (["ㅜ"], "ㅇ", .35)],
    "er": [(["ㅓ"], "ㄹ", 1)],
    "ia": [(["ㅑ"], "", .5), (["ㅣ", "ㅏ"], "", .5)], "ie": [(["ㅖ"], "", .5), (["ㅣ", "ㅔ"], "", .5)],
    "iao": [(["ㅑ", "ㅗ"], "", .5), (["ㅣ", "ㅏ", "ㅗ"], "", .5)], "iu": [(["ㅠ"], "", .5), (["ㅣ", "ㅜ"], "", .5)],
    "ian": [(["ㅔ"], "ㄴ", .35), (["ㅕ"], "ㄴ", .3), (["ㅣ", "ㅏ"], "ㄴ", .35)], "in": [(["ㅣ"], "ㄴ", 1)],
    "iang": [(["ㅑ"], "ㅇ", .6), (["ㅣ", "ㅏ"], "ㅇ", .4)], "ing": [(["ㅣ"], "ㅇ", 1)],
    "iong": [(["ㅛ"], "ㅇ", .5), (["ㅠ"], "ㅇ", .5)],
    "ua": [(["ㅘ"], "", .5), (["ㅜ", "ㅏ"], "", .5)], "uo": [(["ㅝ"], "", .4), (["ㅜ", "ㅗ"], "", .6)],
    "uai": [(["ㅘ", "ㅣ"], "", 1)], "ui": [(["ㅜ", "ㅣ"], "", .7), (["ㅞ", "ㅣ"], "", .3)],
    "uan": [(["ㅘ"], "ㄴ", .6), (["ㅜ", "ㅏ"], "ㄴ", .4)], "un": [(["ㅜ"], "ㄴ", 1)],
    "uang": [(["ㅘ"], "ㅇ", 1)], "ue": [(["ㅞ"], "", .4), (["ㅠ", "ㅔ"], "", .6)],
    "van": [(["ㅠ", "ㅏ"], "ㄴ", .6), (["ㅟ", "ㅏ"], "ㄴ", .4)], "vn": [(["ㅠ"], "ㄴ", .6), (["ㅟ"], "ㄴ", .4)],
    "ve": [(["ㅠ", "ㅔ"], "", .6), (["ㅞ"], "", .4)],
}
_I_FIN = {"i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong"}
_V_FIN = {"u": "v", "ue": "ve", "uan": "van", "un": "vn"}  # after j q x y
_ZERO_OK = {"a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er"}
_Y_OK = {"a": "ㅑ", "ao": None, "an": None, "ang": None, "e": None, "i": None, "in": None, "ing": None, "ong": None,
         "ou": None, "u": None, "uan": None, "ue": None, "un": None, "o": None}
_W_OK = {"a", "ai", "an", "ang", "ei", "en", "eng", "o", "u"}


def _py_valid(ini: str, fin: str) -> bool:
    if fin not in _PY_FIN and fin not in _V_FIN:
        return False
    if ini == "":
        return fin in _ZERO_OK
    if ini == "y":
        return fin in _Y_OK
    if ini == "w":
        return fin in _W_OK
    if ini in ("j", "q", "x"):
        return fin in _I_FIN or fin in _V_FIN
    if ini in ("zh", "ch", "sh", "r", "z", "c", "s", "g", "k", "h", "f"):
        if fin in _I_FIN and not (fin == "i" and ini in ("zh", "ch", "sh", "r", "z", "c", "s")):
            return False
        if fin in ("ue",):
            return False
    if fin == "o" and ini not in ("b", "p", "m", "f", "l"):
        return False
    if fin in ("ua", "uai", "uang") and ini not in ("g", "k", "h", "zh", "ch", "sh", "r"):
        return False
    if fin == "ong" and ini in ("b", "p", "m", "f", "w"):
        return False
    if fin in ("ue",) and ini not in ("n", "l"):
        return False
    if ini in ("b", "p", "m", "f") and fin in ("ui", "uo", "un", "uan", "ua"):
        return False
    if ini in ("d", "t", "n", "l") and fin in ("ua", "uai", "uang"):
        return False
    if fin == "iong" and ini not in ("j", "q", "x"):
        return False
    return True


_PY_SYL: dict[str, tuple[str, str]] = {}
for _i in _PY_INI:
    for _f in list(_PY_FIN) + list(_V_FIN):
        if _f in ("v", "van", "vn", "ve"):
            continue
        if _py_valid(_i, _f):
            _PY_SYL.setdefault(_i + _f, (_i, _f))


def pinyin_split(w: str):
    """Segment into pinyin syllables (fewest syllables); None if impossible."""
    w = w.lower()
    n = len(w)
    best: list = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(min(n, i + 6), i, -1):
            s = w[i:j]
            if s in _PY_SYL and (best[j] is None or len(best[j]) > len(best[i]) + 1):
                best[j] = best[i] + [s]
    return best[n]


PINYIN_MARK = re.compile(r"X|Q|ZH|IAO|IANG|UANG|IONG|^[BPMFDTNLGKHJQXRZCSYW]?[HI]?(AO|OU|EI|UI|UO)$|ONG|ENG|ANG")


def is_pinyin(w: str) -> bool:
    w = w.upper()
    sp = pinyin_split(w)
    if not sp or len(sp) > 4:
        return False
    if len(sp) == 1:
        return True
    return bool(PINYIN_MARK.search(w))


def pinyin(word: str, rng=None) -> str | None:
    sp = pinyin_split(re.sub("[^a-z]", "", word.lower()))
    if not sp:
        return None
    out = []
    for s in sp:
        ini, fin = _PY_SYL[s]
        if ini in ("j", "q", "x", "y") and fin in _V_FIN:
            fin = _V_FIN[fin]
        vs = _PY_FIN[fin]
        if len(vs) == 1 or rng is None:
            meds, jong, _ = vs[0]
        else:
            k = int(rng.choice(len(vs), p=[v[2] for v in vs]))
            meds, jong, _ = vs[k]
        meds = list(meds)
        cho = _PY_ONSET[ini]
        if ini == "y":
            if fin == "i" or fin == "in" or fin == "ing":
                pass
            elif fin == "ou":
                meds = ["ㅠ"] if (rng is None or rng.rand() < 0.6) else ["ㅕ", "ㅜ"]
            elif fin in ("v", "van", "vn", "ve"):
                pass
            else:
                meds[0] = Y_MERGE.get(meds[0], meds[0])
        elif ini == "w":
            if fin == "u":
                pass
            elif fin == "o":
                meds = ["ㅝ"]
            else:
                meds[0] = W_MERGE.get(meds[0], meds[0])
        if ini in ("zh", "ch", "j", "q", "z", "c"):
            meds[0] = Y_COLLAPSE.get(meds[0], meds[0])
            if meds[0] == "ㅟ":
                meds[0] = "ㅜ" if ini in ("zh", "ch") else "ㅟ"
        if ini in ("x", "sh") and meds[0] in ("ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅖ"):
            pass
        if ini in ("zh", "ch", "sh", "r", "z", "c", "s") and fin == "i" and rng is not None and rng.rand() < 0.3:
            meds = ["ㅡ"]
        if ini == "x" and fin in _I_FIN and meds[0] == "ㅣ" and len(meds) > 1 and rng is not None and rng.rand() < 0.5:
            meds = [Y_MERGE.get(meds[1], meds[1])] + meds[2:]  # xia -> 샤
        syl = [_compose(cho, meds[0], jong if len(meds) == 1 else "")]
        for k2, m in enumerate(meds[1:]):
            syl.append(_compose("ㅇ", m, jong if k2 == len(meds) - 2 else ""))
        out.append("".join(syl))
    return "".join(out)


# ---------------------------------------------------------------- public entry
JAPANESE = re.compile(r"((KY|GY|SH|CH|TS|NY|HY|MY|RY|BY|PY|[KGSZTDNHBPMYRWFJ])?[AEIOU]|N)+")  # romaji-only words keep hard g

def loan(word: str, rng=None) -> str | None:
    """Plausible Hangul loanword spelling of one Latin-script word (None for empty input)."""
    w = re.sub("[^A-Z]", "", str(word).upper())
    if not w:
        return None
    if w in EXC:
        v = EXC[w]
        if len(v) == 1 or rng is None:
            return v[0]
        return v[0] if rng.rand() < 0.7 else v[1 + rng.randint(len(v) - 1)]
    if is_english(w):
        out = en_hangul(w, rng)
    else:
        out = cont_hangul(w, rng, soft=not JAPANESE.fullmatch(w))
    return out or None


if __name__ == "__main__":
    import sys
    import numpy as np
    r = np.random.RandomState(0)
    for a in sys.argv[1:]:
        print(a, loan(a, r), en_hangul(a, None), cont_hangul(a, None), pinyin(a, None))
