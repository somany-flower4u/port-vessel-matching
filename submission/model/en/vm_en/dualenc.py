"""v27f: numpy-only inference for the Latin character dual encoder (Latin query tower x Latin catalog tower).

Mirror of vm_ko.dualenc (same tower: embedding -> 3 dilated residual Conv1d(k=3) blocks with exact GELU ->
[max; mean] pooling -> linear -> L2 normalisation).  The towers were trained contrastively offline (torch;
agents/latin-de/de_en.py): query tower 'q.' = noisy port name, catalog tower 'l.' = catalog name.  A query row
is encoded on its own; the catalog side is precomputed (float16 on disk, float32 in memory)."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

AZ, DIG = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"
LV = ["<pad>", "_", "<unk>"] + list(AZ) + list(DIG)
LID = {t: i for i, t in enumerate(LV)}
LMAX = 48


def l_tokens(s: str) -> list[int]:
    """A-Z / 0-9 characters; any other run -> '_'."""
    s = unicodedata.normalize("NFKC", str(s)).upper()
    out: list[int] = []
    for ch in s:
        if ch in LID and ch != "<pad>":
            out.append(LID[ch])
        elif not out or out[-1] != 1:
            out.append(1)
    while out and out[-1] == 1:
        out.pop()
    while out and out[0] == 1:
        out.pop(0)
    return out[:LMAX] or [2]


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
    """One encoder tower; parameters under prefix 'q.' (query) or 'l.' (catalog)."""

    def __init__(self, z, prefix: str):
        g = lambda k: np.asarray(z[prefix + k], dtype=np.float32)  # noqa: E731
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
        x = self.emb[l_tokens(text)]
        h = _gelu(self._conv(x, self.convs[0]))
        h = h + _gelu(self._conv(h, self.convs[1]))
        h = h + _gelu(self._conv(h, self.convs[2]))
        v = np.concatenate([h.max(0), h.mean(0)]) @ self.proj_w + self.proj_b
        return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


class DualEncoder:
    """Latin query tower + catalog-side embeddings aligned to the given catalog rows."""

    def __init__(self, weights: Path, cat_emb: Path, catalog_ids, catalog_names):
        z = np.load(weights)
        self.q = Tower(z, "q.")
        self.l = Tower(z, "l.")
        ids = None
        if str(cat_emb).endswith(".npz"):
            d = np.load(cat_emb)
            emb = d["emb"]
            ids = d["ids"] if "ids" in d.files else None
        else:
            emb = np.load(cat_emb, mmap_mode="r")
        n = len(catalog_ids)
        if ids is not None and not (len(ids) == n and bool(np.all(ids == np.asarray(catalog_ids, dtype=ids.dtype)))):
            # a catalog in another order / with other rows: realign by vessel_id, encode unknown rows
            pos = {v: i for i, v in enumerate(ids.tolist())}
            out = np.zeros((n, emb.shape[1]), dtype=np.float32)
            for i, v in enumerate(catalog_ids):
                j = pos.get(str(v))
                out[i] = emb[j] if j is not None else self.l(catalog_names[i])
            emb = out
        elif len(emb) != n:
            raise ValueError(f"dual-encoder catalog embeddings have {len(emb)} rows, catalog has {n}")
        self.C = emb if emb.dtype == np.float32 else np.ascontiguousarray(emb, dtype=np.float32)

    def scores(self, name: str) -> np.ndarray:
        """cosine(query name, every catalog name), catalog row order."""
        return self.C @ self.q(name)
