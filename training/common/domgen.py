"""Domestic (Korean-flagged, non-train) catalog vessels as Hangul query rows, in several
realistic conditions.  Every vessel gets a 5-fold id so rankers can be held out by vessel."""
from __future__ import annotations
import re, zlib, numpy as np, pandas as pd
import kogen, stress, aug14

PORT_LABEL = {"Pass": "여객선", "Pass/Car F.": "여객선", "RoPax": "여객선", "Pass/Cargo": "여객선", "Pass/Cat": "여객선",
              "Rfr Fish C.": "냉동.냉장선", "Reefer": "냉동.냉장선", "Cement": "기타선", "Aggregates": "기타선",
              "Limestone": "기타선", "Waste": "기타선", "Oceanograph.": "기타선", "Deck Cargo Carr.": "기타선",
              "Tug": "견인용예선", "AHTS": "견인용예선", "Utility": "기타선", "Landing Craft": "기타선",
              "Cable Layer": "기타선", "M. Sulphur": "케미칼 운반선"}
KINDS = ["clean", "natgt", "igt", "noton", "noflag", "noise", "abbr", "nosp", "rename", "dwt"]


def _letters(w: str) -> str:
    return "".join(kogen.LETTER[ch] for ch in w)


def patched_word(self, w: str, rng) -> str | None:
    """kogen.Converter.word without the random letter-spelling of ordinary syllable words."""
    if w in self.en2ko:
        return self.en2ko[w]
    if not re.search("[AEIOUY]", w):
        return _letters(w) if len(w) <= 5 else None
    r = self.roman_word(w)
    if r is None:
        return _letters(w) if len(w) <= 3 else None
    if len(w) <= 2 and w not in kogen.SPECIAL and rng.rand() < 0.1:  # 'SH', 'JE' style initials are sometimes spelled out
        return _letters(w)
    return r


kogen.Converter.word = patched_word


def population(train: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    kc = cat[(cat.flag_name == "South Korea") & ~cat.vessel_id.isin(set(train.target_vessel_id))].copy()
    kc["fold"] = [zlib.crc32(v.encode()) % 5 for v in kc.vessel_id]
    return kc


def make_rows(train: pd.DataFrame, cat: pd.DataFrame, kinds=KINDS, seed=0, reps=1) -> pd.DataFrame:
    cv = kogen.Converter(train, cat)
    m = train.merge(cat[["vessel_id", "vessel_type"]].rename(columns={"vessel_id": "target_vessel_id", "vessel_type": "ct"}),
                    on="target_vessel_id")
    tmap = m.dropna(subset=["vessel_type"]).groupby("ct").vessel_type.agg(lambda x: x.mode().iloc[0]).to_dict()
    kc = population(train, cat)
    rng = np.random.RandomState(seed)
    rows = []
    kc_dwt = dict(zip(kc.vessel_id, pd.to_numeric(kc.deadweight_tonnage, errors="coerce")))
    bases = [b for b in (cv.name(n, np.random.RandomState(7)) for n in kc.vessel_name) if b]  # pool of unrelated Korean-style names
    for rep in range(reps):
        for vid, name, ct, gt, fold in zip(kc.vessel_id, kc.vessel_name, kc.vessel_type, kc.gross_tonnage, kc.fold):
            base = cv.name(name, rng)
            if base is None:
                continue
            gt = float(gt) if pd.notna(gt) and gt > 0 else np.nan
            if not np.isfinite(gt):
                dw = kc_dwt.get(vid, np.nan)
                if np.isfinite(dw) and dw > 0:
                    gt = float(round(dw * rng.uniform(0.35, 0.75)))  # GT-less catalog ship: the record still carries a GT
            for kind in kinds:
                nm = base
                g = ig = gt
                if kind == "natgt" or (kind in ("noise", "abbr", "nosp") and rng.rand() < 0.4):
                    g = float(round(gt * np.exp(rng.uniform(-0.45, -0.03)))) if np.isfinite(gt) else np.nan
                    ig = np.nan if rng.rand() < 0.7 else gt
                elif kind == "igt":
                    g = float(round(gt * np.exp(rng.uniform(-0.45, -0.03)))) if np.isfinite(gt) else np.nan
                    ig = gt
                elif kind == "noton" or (kind in ("noise", "abbr", "nosp", "noflag") and rng.rand() < 0.3):
                    g = ig = np.nan
                if kind == "dwt":  # the record's tonnage field holds the deadweight
                    dw = kc_dwt.get(vid, np.nan)
                    if not (np.isfinite(dw) and dw > 0):
                        continue
                    g = float(dw); ig = np.nan if rng.rand() < 0.6 else float(dw)
                if kind == "rename":  # the record carries an unrelated (former / registered) name: only tonnage, flag and type link it
                    nm = bases[rng.randint(len(bases))]
                    if nm == base:
                        continue
                    g = ig = gt
                    if rng.rand() < 0.3:
                        ig = np.nan
                if kind == "noise":
                    nm = stress.ko_noise(nm, rng) if rng.rand() < 0.7 else stress.ko_noise_heavy(nm, rng)
                elif kind == "abbr":
                    nm = aug14.ko_abbr2(nm, rng)
                elif kind == "nosp":
                    nm = nm.replace(" ", "")
                if not re.search("[가-힣]", nm):
                    continue
                flag = "대한민국" if (kind != "noflag" and rng.rand() < 0.9) else np.nan
                vtype = tmap.get(ct, PORT_LABEL.get(ct, "기타선")) if (kind != "noflag" and rng.rand() < 0.9) else np.nan
                rows.append(dict(sample_id=f"DOM_{len(rows):06d}", vessel_name=nm, call_sign=f"CS_DOM{len(rows):06d}",
                                 gross_tonnage=g, international_gross_tonnage=ig, flag_name=flag, vessel_type=vtype,
                                 arrival_year=2025, target_vessel_id=vid, ctype=ct, kind=kind, fold=int(fold), rep=rep))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tr = pd.read_csv("train.csv"); cat = pd.read_csv("vessel_catalog.csv")
    df = make_rows(tr, cat)
    print(len(df), df.kind.value_counts().to_dict(), df.fold.value_counts().to_dict())
    pd.set_option("display.width", 200)
    c = cat.set_index("vessel_id")
    s = df.sample(40, random_state=1)
    s["cat"] = s.target_vessel_id.map(c.vessel_name)
    print(s[["vessel_name", "cat", "kind", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type"]].to_string())
