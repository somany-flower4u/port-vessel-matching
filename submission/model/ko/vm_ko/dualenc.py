"""v25: numpy-only inference for the character dual encoder (Hangul jamo tower x Latin char tower).

The towers were trained contrastively offline (torch; agents/scorer/dualenc.py, v1 recipe) and exported as
numpy arrays.  A tower is: embedding -> 3 dilated residual Conv1d(k=3) blocks with exact GELU -> [max; mean]
pooling -> linear -> L2 normalisation.  A query row is encoded on its own; the catalog side is precomputed
(float16 on disk, float32 in memory)."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
AZ, DIG = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"
HV = ["<pad>", "_", "<unk>"] + ["I" + c for c in CHO] + ["V" + v for v in JUNG] + ["F" + j for j in JONG[1:]] + \
     ["D" + d for d in DIG] + ["L" + a for a in AZ]
HID = {t: i for i, t in enumerate(HV)}
LV = ["<pad>", "_", "<unk>"] + list(AZ) + list(DIG)
LID = {t: i for i, t in enumerate(LV)}
HMAX, LMAX = 64, 48


def _trim(out: list[int], limit: int) -> list[int]:
    while out and out[-1] == 1:
        out.pop()
    while out and out[0] == 1:
        out.pop(0)
    return out[:limit] or [2]


def h_tokens(s: str) -> list[int]:
    """Hangul side: syllables -> initial / medial / final jamo tokens; digits and Latin letters kept; any other run -> '_'."""
    s = unicodedata.normalize("NFKC", str(s)).upper()
    out: list[int] = []
    for ch in s:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out += [HID["I" + CHO[code // 588]], HID["V" + JUNG[(code % 588) // 28]]]
            if code % 28:
                out.append(HID["F" + JONG[code % 28]])
        elif ch in DIG:
            out.append(HID["D" + ch])
        elif ch in AZ:
            out.append(HID["L" + ch])
        elif not out or out[-1] != 1:
            out.append(1)
    return _trim(out, HMAX)


def l_tokens(s: str) -> list[int]:
    """Latin side: A-Z / 0-9 characters; any other run -> '_'."""
    s = unicodedata.normalize("NFKC", str(s)).upper()
    out: list[int] = []
    for ch in s:
        if ch in LID and ch != "<pad>":
            out.append(LID[ch])
        elif not out or out[-1] != 1:
            out.append(1)
    return _trim(out, LMAX)


_SQRT1_2 = np.float32(1.0 / np.sqrt(2.0))


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7), vectorised."""
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return np.copysign(y, x)


def _gelu(x: np.ndarray) -> np.ndarray:
    """exact (erf) GELU, as torch.nn.functional.gelu"""
    return 0.5 * x * (1.0 + _erf(x * _SQRT1_2))


class Tower:
    """One encoder tower; parameters under prefix 'h.' (Hangul) or 'l.' (Latin)."""

    def __init__(self, z, prefix: str, tokens):
        g = lambda k: np.asarray(z[prefix + k], dtype=np.float32)  # noqa: E731
        self.tokens = tokens
        self.emb = g("emb.weight")
        self.convs = []
        for name, dil in (("c1", 1), ("c2", 2), ("c3", 4)):
            w = g(f"{name}.weight")  # [Cout, Cin, 3]
            self.convs.append(([np.ascontiguousarray(w[:, :, j].T) for j in range(3)], g(f"{name}.bias"), dil))
        self.proj_w = np.ascontiguousarray(g("proj.weight").T)
        self.proj_b = g("proj.bias")

    @staticmethod
    def _conv(x: np.ndarray, conv) -> np.ndarray:
        ws, b, dil = conv
        L = x.shape[0]
        xp = np.zeros((L + 2 * dil, x.shape[1]), dtype=np.float32)
        xp[dil:dil + L] = x
        return xp[0:L] @ ws[0] + xp[dil:dil + L] @ ws[1] + xp[2 * dil:2 * dil + L] @ ws[2] + b

    def __call__(self, text: str) -> np.ndarray:
        x = self.emb[self.tokens(text)]
        h = _gelu(self._conv(x, self.convs[0]))
        h = h + _gelu(self._conv(h, self.convs[1]))
        h = h + _gelu(self._conv(h, self.convs[2]))
        v = np.concatenate([h.max(0), h.mean(0)]) @ self.proj_w + self.proj_b
        return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


class DualEncoder:
    """Hangul query tower + catalog-side embeddings aligned to the given catalog rows."""

    def __init__(self, weights: Path, cat_emb: Path, catalog_ids, catalog_names):
        z = np.load(weights)
        self.h = Tower(z, "h.", h_tokens)
        self.l = Tower(z, "l.", l_tokens) if "l.emb.weight" in z.files else None
        ids = None
        if str(cat_emb).endswith(".npz"):
            d = np.load(cat_emb)
            emb = d["emb"]
            ids = d["ids"] if "ids" in d.files else None
        else:
            emb = np.load(cat_emb, mmap_mode="r")  # float32 per-fold files: shared page cache across workers
        n = len(catalog_ids)
        if ids is not None and not (len(ids) == n and bool(np.all(ids == np.asarray(catalog_ids, dtype=ids.dtype)))):
            # a catalog in another order / with other rows: realign by vessel_id, encode unknown rows
            pos = {v: i for i, v in enumerate(ids.tolist())}
            out = np.zeros((n, emb.shape[1]), dtype=np.float32)
            for i, v in enumerate(catalog_ids):
                j = pos.get(str(v))
                if j is not None:
                    out[i] = emb[j]
                elif self.l is not None:
                    out[i] = self.l(catalog_names[i])
            emb = out
        elif len(emb) != n:
            raise ValueError(f"dual-encoder catalog embeddings have {len(emb)} rows, catalog has {n}")
        self.C = emb if emb.dtype == np.float32 else np.ascontiguousarray(emb, dtype=np.float32)

    def scores(self, name: str) -> np.ndarray:
        """cosine(query name, every catalog name), catalog row order."""
        return self.C @ self.h(name)
