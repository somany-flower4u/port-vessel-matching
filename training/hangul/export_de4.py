"""Export dual-encoder torch checkpoints to numpy assets (de25/export_de.py with this track's paths; scorer venv: needs torch).
usage: export_de.py fold K TAG FOLDDIR  -> FOLDDIR/de_f{K}_weights.npz (h.* and l.* float32) + FOLDDIR/de_f{K}_cat.npy (float32, memmap-able)
       export_de.py full TAG PKGDIR     -> PKGDIR/model/ko/de/de_weights.npz + de_cat.npz (emb float16 [78,595 x 128], ids)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
import os
OUT = Path(os.environ.get("DE_OUT") or (HERE / "out"))
WORK = Path(os.environ.get("SEOKCHEON_WS", ".") + "/work")


def weights(pt):
    sd = torch.load(pt, map_location="cpu")
    return {k: v.numpy().astype(np.float32) for k, v in sd.items() if k.startswith(("h.", "l."))}


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "fold":
        k, tag, fdir = int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
        fdir.mkdir(parents=True, exist_ok=True)
        np.savez(fdir / f"de_f{k}_weights.npz", **weights(OUT / f"de_f{k}{tag}.pt"))
        c = np.load(OUT / f"de_f{k}{tag}_cat.npy")
        assert c.dtype == np.float16 and c.shape == (78595, 128), (c.dtype, c.shape)
        np.save(fdir / f"de_f{k}_cat.npy", c.astype(np.float32))
        print("fold", k, "exported ->", fdir)
    else:
        tag, pkg = sys.argv[2], Path(sys.argv[3])
        out = pkg / "model" / "ko" / "de"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "de_weights.npz", **weights(OUT / f"de_full{tag}.pt"))
        c = np.load(OUT / f"de_full{tag}_cat.npy")
        assert c.dtype == np.float16 and c.shape == (78595, 128), (c.dtype, c.shape)
        ids = pd.read_csv(WORK / "vessel_catalog.csv").vessel_id.astype(str).to_numpy()
        np.savez_compressed(out / "de_cat.npz", emb=c, ids=ids.astype("U"))
        print("full exported ->", out)
