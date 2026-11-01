"""Latin-Latin character dual encoder (query tower = noisy port name, catalog tower = catalog name).

Mirror of the Hangul DE (agents/de25/dualenc_full.py, v1 recipe: same tower, contrastive InfoNCE both ways,
half of every batch a window of pairs sorted by catalog name), with Latin characters on both sides and two
separate towers ("q." query, "l." catalog).

Held out by vessel: the model for fold F never sees train vessels with crc32(vessel_id) % 5 == F
(FOLD -1 = full model, all train vessels).
Training pairs: distinct (Latin port name, target) of the train Latin rows, REPS copies per epoch, each query
freshly damaged as the Latin ranker's training copies are (orig / stress.en_noise / stress.en_noise_heavy /
aug14.en_abbr).  Optional SYN > 0: + SYN synthetic pairs per epoch from other catalog names (train-style typo +
the same damage); SYN = 0 is the default (train rows only).

de_en2.py = de_en.py + the v2 recipe of agents/de-v2-full/dualenc_v2dom.py: model-mined hard negatives.  At every
epoch in MINE the catalog (l tower) and the epoch's queries (q tower) are encoded, the top NHARD+4 neighbours of each
query are taken, wrong-name (different compact-name code) neighbours first, top NHARD kept; each batch adds one
random mined negative per query as an extra column of the query->catalog loss (same-name clashes masked).  Epochs
after the first MINE epoch that are not MINE epochs reuse the last mined negatives and keep that epoch's pairs (fresh
damage only) so the mined indices line up.  MINE = "none" -> exactly de_en.py.
Outputs go to agents/latin-de/out under de_en.py's names (build_en2.sh copies them from there).

usage: de_en2.py FOLD [epochs=15] [threads=3] [reps=4] [syn=20000] [tag=C] [MINE=2,4,..,epochs-2 | none]
writes out/de_en_{f{F}|full}{tag}_weights.npz (q.* / l.* float32), out/..._cat.npy (float16 [78,595 x 128])
"""
from __future__ import annotations
import os

import json
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = Path(os.environ.get("SEOKCHEON_WS", ".") + "/work")
OUT = Path(__file__).resolve().parent.parent / "latin-de" / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(WORK))
import aug14  # noqa: E402
import stress  # noqa: E402

FOLD = int(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
SYN = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
TAG = sys.argv[6] if len(sys.argv) > 6 else "C"
_m = sys.argv[7] if len(sys.argv) > 7 else ",".join(str(e) for e in range(2, EPOCHS - 1, 2))
MINE_EPOCHS = () if _m in ("none", "") else tuple(int(x) for x in _m.split(","))
NHARD = 8
torch.set_num_threads(THREADS)
SEED = FOLD if FOLD >= 0 else 2026
torch.manual_seed(SEED)

AZ, DIG = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"
LV = ["<pad>", "_", "<unk>"] + list(AZ) + list(DIG)
LID = {t: i for i, t in enumerate(LV)}
LMAX = 48


def l_tokens(s: str) -> list[int]:
    """A-Z / 0-9 characters; any other run -> '_' (identical to vm_ko.dualenc.l_tokens)."""
    s = unicodedata.normalize("NFKC", str(s)).upper()
    out = []
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


def pad(seqs):
    L = max(len(s) for s in seqs)
    a = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return torch.from_numpy(a)


class Tower(nn.Module):
    def __init__(self, V: int, d: int = 64, h: int = 160, out: int = 128):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.c1 = nn.Conv1d(d, h, 3, padding=1)
        self.c2 = nn.Conv1d(h, h, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(h, h, 3, padding=4, dilation=4)
        self.proj = nn.Linear(2 * h, out)

    def forward(self, x):
        mask = (x > 0).float().unsqueeze(1)
        e = self.emb(x).transpose(1, 2)
        h1 = F.gelu(self.c1(e)) * mask
        h2 = h1 + F.gelu(self.c2(h1)) * mask
        h3 = h2 + F.gelu(self.c3(h2)) * mask
        mx = (h3 - 1e4 * (1 - mask)).max(-1).values
        mean = h3.sum(-1) / mask.sum(-1).clamp(min=1)
        return F.normalize(self.proj(torch.cat([mx, mean], -1)), dim=-1)


class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = Tower(len(LV))
        self.l = Tower(len(LV))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.05), dtype=torch.float32))


def encode(tower, toks, bs=2048):
    tower.eval()
    order = np.argsort([len(t) for t in toks])
    out = np.zeros((len(toks), 128), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(toks), bs):
            idx = order[a:a + bs]
            out[idx] = tower(pad([toks[i] for i in idx])).numpy()
    tower.train()
    return out


def damage(s: str, rng) -> str:
    """The Latin ranker's training copies: orig, en_noise, en_heavy, en_abbr x2 (+ orig / en_noise with
    tonnage errors, which do not change the name) -> orig 2/7, noise 2/7, heavy 1/7, abbr 2/7."""
    r = rng.rand()
    if r < 2 / 7:
        return s
    if r < 4 / 7:
        return stress.en_noise(s, rng)
    if r < 5 / 7:
        return stress.en_noise_heavy(s, rng)
    return aug14.en_abbr(s, rng)


def train_typo(name: str, rng) -> str:
    """Train generator style typo on a catalog name: 0-2 interior delete / double / swap in words >= 6 letters."""
    ws = str(name).upper().split()
    for _ in range(rng.choice([0, 1, 2], p=[0.35, 0.4, 0.25])):
        el = [k for k, w in enumerate(ws) if len(w) >= 6 and w.isalpha()]
        if not el:
            break
        k = el[rng.randint(len(el))]
        w = ws[k]
        i = 1 + rng.randint(len(w) - 2)
        op = rng.randint(3)
        if op == 0:
            w = w[:i] + w[i + 1:]
        elif op == 1:
            w = w[:i + 1] + w[i] + w[i + 1:]
        elif i + 1 < len(w) - 1 and w[i] != w[i + 1]:
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        ws[k] = w
    return " ".join(ws)


def fold_of(v) -> int:
    return zlib.crc32(str(v).encode()) % 5


if __name__ == "__main__":
    t0, c0 = time.time(), time.process_time()
    train = pd.read_csv(WORK / "train.csv")
    cat = pd.read_csv(WORK / "vessel_catalog.csv")
    names = cat.vessel_name.fillna("").astype(str).tolist()
    compact = np.array(["".join(re.findall(r"[A-Z0-9]+", unicodedata.normalize("NFKC", n).upper())) for n in names],
                       dtype=object)
    codes = pd.factorize(compact)[0].astype(np.int64)
    id2idx = {v: i for i, v in enumerate(cat.vessel_id.astype(str))}
    en = train[~train.vessel_name.fillna("").str.contains("[가-힣]")].copy()
    en["fold"] = en.target_vessel_id.map(fold_of)
    held = set(train.target_vessel_id[train.target_vessel_id.map(fold_of) == FOLD]) if FOLD >= 0 else set()
    sub = en[~en.target_vessel_id.isin(held)].drop_duplicates(["vessel_name", "target_vessel_id"])
    real = [(str(n), id2idx[t]) for n, t in zip(sub.vessel_name, sub.target_vessel_id)]
    train_ids = set(train.target_vessel_id)
    syn_pool = np.array([i for i, v in enumerate(cat.vessel_id) if v not in held and v not in train_ids
                         and compact[i]], dtype=np.int64)
    cat_tok = [l_tokens(n) for n in names]
    print(f"fold {FOLD}: real pairs {len(real)}, reps {REPS}, syn {SYN} (pool {len(syn_pool)}), mine {MINE_EPOCHS}",
          flush=True)

    model = DualEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.RandomState(SEED)
    B = 384
    total_steps = None
    step = 0
    hard_prev = None
    for ep in range(EPOCHS):
        # v2: between mining epochs keep the same pairs (fresh damage only) so the mined indices line up
        reuse = bool(MINE_EPOCHS) and ep > MINE_EPOCHS[0] and ep not in MINE_EPOCHS and hard_prev is not None
        if not reuse:
            pairs = real * REPS
            if SYN:
                pairs = pairs + [(train_typo(names[i], rng), int(i)) for i in rng.choice(syn_pool, size=SYN, replace=False)]
        qs = [l_tokens(damage(q, rng)) for q, _ in pairs]
        tg = np.array([i for _, i in pairs])
        n = len(pairs)
        if total_steps is None:
            total_steps = EPOCHS * (n // B)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total_steps + 10, pct_start=0.1)
        hard = hard_prev if reuse else None
        if ep in MINE_EPOCHS:
            cemb_m = torch.from_numpy(encode(model.l, cat_tok))
            qemb_m = torch.from_numpy(encode(model.q, qs))
            ccodes = torch.from_numpy(codes)
            tcodes = torch.from_numpy(codes[tg])
            hard = np.zeros((n, NHARD), dtype=np.int64)
            with torch.no_grad():
                for a in range(0, n, 1024):
                    top = torch.topk(qemb_m[a:a + 1024] @ cemb_m.T, NHARD + 4, dim=1).indices  # sorted
                    ok = ccodes[top] != tcodes[a:a + 1024, None]
                    # stable: valid neighbours first, in score order
                    order = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)
                    hard[a:a + len(top)] = torch.gather(top, 1, order)[:, :NHARD].numpy()
            hard_prev = hard
            print(f"  mined hard negatives ep {ep}: wall {time.time() - t0:.0f}s cpu {time.process_time() - c0:.0f}s",
                  flush=True)
        srt = np.argsort(compact[tg], kind="stable")
        perm = rng.permutation(n)
        losses = []
        for b in range(n // B):
            a0 = rng.randint(0, n - B // 2)
            idx = np.concatenate([srt[a0:a0 + B // 2], perm[b * (B // 2):(b + 1) * (B // 2)]])
            _, keep = np.unique(codes[tg[idx]], return_index=True)
            idx = idx[keep]
            qq = model.q(pad([qs[k] for k in idx]))
            ql = model.l(pad([cat_tok[tg[k]] for k in idx]))
            scale = model.logit_scale.exp().clamp(max=100)
            logits = scale * qq @ ql.T
            lab = torch.arange(len(idx))
            if hard is not None:
                hn = np.array([hard[k][rng.randint(NHARD)] for k in idx])
                qn_ = model.l(pad([cat_tok[c] for c in hn]))
                lh = scale * qq @ qn_.T
                clash = torch.from_numpy(codes[hn][None, :] == codes[tg[idx]][:, None])
                lh = lh.masked_fill(clash, -1e4)
                loss = 0.5 * (F.cross_entropy(torch.cat([logits, lh], 1), lab) + F.cross_entropy(logits.T, lab))
            else:
                loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step < total_steps + 9:
                sched.step()
            step += 1
            losses.append(loss.item())
        print(f"ep {ep} pairs {n} loss {np.mean(losses):.3f} scale {model.logit_scale.exp().item():.1f} "
              f"wall {time.time() - t0:.0f}s cpu {time.process_time() - c0:.0f}s", flush=True)
    FN = ("full" if FOLD < 0 else f"f{FOLD}") + TAG
    sd = model.state_dict()
    np.savez(OUT / f"de_en_{FN}_weights.npz",
             **{k: v.numpy().astype(np.float32) for k, v in sd.items() if k.startswith(("q.", "l."))})
    cemb = encode(model.l, cat_tok)
    np.save(OUT / f"de_en_{FN}_cat.npy", cemb.astype(np.float16))
    json.dump({"fold": FOLD, "epochs": EPOCHS, "reps": REPS, "syn": SYN, "real_pairs": len(real),
               "mine": list(MINE_EPOCHS), "nhard": NHARD,
               "train_wall_s": time.time() - t0, "train_cpu_s": time.process_time() - c0, "threads": THREADS},
              open(OUT / f"de_en_{FN}_info.json", "w"))
    print("saved", OUT / f"de_en_{FN}_weights.npz", f"wall {time.time() - t0:.0f}s", flush=True)
