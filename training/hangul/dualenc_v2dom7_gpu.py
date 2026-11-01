"""dualenc_v2dom7_gpu.py = dualenc_v2dom6_gpu.py + TOWER_D / TOWER_H env knobs (default 64/160 = identical).
[de-v2-full track] v2 recipe (scorer/dualenc.py: model-mined hard negatives) + the domestic Korean-flag population.
FOLD -1 = full model (all train Hangul rows); FOLD k = train vessels with crc32(vessel_id) % 5 == k held out.
Domestic population (domgen.population: Korean-flag catalog vessels not in train): vessels with crc32 % 5 == 0 are a
hold-out excluded from EVERY model (never in any pair, never in the synthetic pool); the rest get fresh kogen
(domgen-patched Converter) Hangul names every (non-reuse) epoch, DOMREPS draws per vessel, as extra training pairs.
Everything else is the de25/dualenc_full.py v2 code path (same towers, loss, batches, augmentation, mining).
dom5 (round4 v28h): identical to dualenc_v2dom4.py except that kogen5.install() replaces the domgen word patch
before any pair is built, so the synthetic pool AND the domestic names use the English-loanword / pinyin fallback
(round4/kogen5.py + round4/loanword.py): ~99.9 % of catalog names and all 1,116 domestic vessels are writable (was 49.7 % / 831).
dom6 (round4 v29a): identical to dualenc_v2dom5.py except that kogen6.install() replaces kogen5.install(): pinyin
routing / reader variants, train-dictionary spelling variants sampled by count, Chinese-context readings, roman numeral I,
Revised-Romanisation sound-change readings (round4/kogen6.py).
usage: dualenc_v2dom6.py FOLD [epochs=20] [threads=2] [synthetic pairs per epoch=12000] [tag=v2d] [MINE=2,4,..,18] [DOMREPS=2]
Writes out/de_{full|fK}{tag}.pt and out/de_{full|fK}{tag}_cat.npy (float16 catalog embeddings).
gpu (round4): identical to dualenc_v2dom6.py plus env DEVICE=cpu|cuda|auto (default cpu = the original code path; auto = cuda if
available). On cuda the model, every batch tensor, the mining matmul/top-k and the catalog-embedding pass run on the GPU;
weights are initialised on the CPU (same init as the CPU run) and the state_dict is saved from a CPU copy (same .pt format).
fp32 by default: TF32 is disabled for matmul and cuDNN conv unless DE_TF32=1."""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("SEOKCHEON_WS", ".") + '/agents/de-v2-full')
import common_l as C
from pathlib import Path
import os
OUT = Path(os.environ.get('DE_OUT') or (Path(__file__).resolve().parent / 'out'))
DOM_ALL = os.environ.get('DOM_ALL', '0') == '1'  # round4: include the fold-0 domestic vessels (no domestic hold-out)
OUT.mkdir(exist_ok=True)

FOLD = int(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SYN_PER_EPOCH = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
TAG = sys.argv[5] if len(sys.argv) > 5 else "v2d"
_m = sys.argv[6] if len(sys.argv) > 6 else ",".join(str(e) for e in range(2, EPOCHS, 2))
DOMREPS = int(sys.argv[7]) if len(sys.argv) > 7 else 2
MINE_EPOCHS = () if _m == "none" else tuple(int(x) for x in _m.split(","))
NHARD = 8
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
torch.set_num_threads(THREADS)
SEED = FOLD if FOLD >= 0 else 2026
torch.manual_seed(SEED)
_dev = os.environ.get("DEVICE", "cpu").strip().lower() or "cpu"
if _dev == "auto":
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(_dev)
if DEV.type == "cuda":
    _tf32 = os.environ.get("DE_TF32", "0") == "1"
    torch.backends.cuda.matmul.allow_tf32 = _tf32
    torch.backends.cudnn.allow_tf32 = _tf32
    torch.backends.cudnn.benchmark = False  # variable sequence lengths / batch sizes

import aug14  # noqa: E402
import domgen  # noqa: E402  (patches kogen.Converter.word as in the P2 generator)
import kogen  # noqa: E402
import stress  # noqa: E402
sys.path.append(str(Path(__file__).resolve().parent))
import kogen6  # noqa: E402  (round4: kogen5 + pinyin / dictionary / RR fixes; patches kogen.Converter only in this process)

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIG = "0123456789"
HV = ["<pad>", "_", "<unk>"] + ["I" + c for c in CHO] + ["V" + v for v in JUNG] + ["F" + j for j in JONG[1:]] + \
     ["D" + d for d in DIG] + ["L" + a for a in AZ]
HID = {t: i for i, t in enumerate(HV)}
LV = ["<pad>", "_", "<unk>"] + list(AZ) + list(DIG)
LID = {t: i for i, t in enumerate(LV)}
HMAX, LMAX = 64, 48


def h_tokens(s: str) -> list[int]:
    s = unicodedata.normalize("NFKC", str(s)).upper()
    out = []
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
    while out and out[-1] == 1:
        out.pop()
    while out and out[0] == 1:
        out.pop(0)
    return out[:HMAX] or [2]


def l_tokens(s: str) -> list[int]:
    s = unicodedata.normalize("NFKC", str(s)).upper()
    out = []
    for ch in s:
        if ch in LID and ch not in ("<pad>",):
            out.append(LID[ch])
        elif not out or out[-1] != 1:
            out.append(1)
    while out and out[-1] == 1:
        out.pop()
    while out and out[0] == 1:
        out.pop(0)
    return out[:LMAX] or [2]


def pad(seqs: list[list[int]]) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        _d, _h = int(os.environ.get("TOWER_D", "64")), int(os.environ.get("TOWER_H", "160"))  # v2dom7: capacity knobs (out stays 128)
        self.h = Tower(len(HV), d=_d, h=_h)
        self.l = Tower(len(LV), d=_d, h=_h)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.05), dtype=torch.float32))


def encode(tower: Tower, toks: list[list[int]], bs: int = 2048) -> np.ndarray:
    tower.eval()
    order = np.argsort([len(t) for t in toks])
    out = np.zeros((len(toks), 128), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(toks), bs):
            idx = order[a:a + bs]
            out[idx] = tower(pad([toks[i] for i in idx]).to(DEV)).cpu().numpy()
    tower.train()
    return out


def augment(s: str, rng) -> str:
    r = rng.rand()
    if r < 0.45:
        out = s
    elif r < 0.62:
        out = stress.ko_noise(s, rng)
    elif r < 0.67:
        out = stress.ko_noise_heavy(s, rng)
    elif r < 0.82:
        out = aug14.ko_abbr2(s, rng)
    else:
        out = s.replace(" ", "")
    return out if re.search("[가-힣]", out) else s


if __name__ == "__main__":
    t_start, c_start = time.time(), time.process_time()
    train, cat = C.load()
    kogen6.install(cat)
    meta = C.catalog_meta(cat)
    train["fold"] = train.target_vessel_id.map(C.fold_of)
    sub = train[train.fold != FOLD].reset_index(drop=True)
    held_ids = set(train.target_vessel_id[train.fold == FOLD])
    dom_pop = domgen.population(train, cat)
    dom_ids = set(dom_pop.vessel_id)
    dom_tr = [(n, meta["id2idx"][v]) for n, v, f in zip(dom_pop.vessel_name, dom_pop.vessel_id, dom_pop.fold) if DOM_ALL or f != 0]
    dom_hold = set() if DOM_ALL else set(dom_pop.vessel_id[dom_pop.fold == 0])
    cv = kogen.Converter(sub, cat)
    ko = sub[sub.vessel_name.str.contains("[가-힣]", na=False)].drop_duplicates(["vessel_name", "target_vessel_id"])
    real = [(n, meta["id2idx"][t]) for n, t in zip(ko.vessel_name, ko.target_vessel_id)]
    syn_pool = [i for i, v in enumerate(cat.vessel_id) if v not in held_ids and v not in dom_ids]
    cat_tok = [l_tokens(n) for n in meta["names"]]
    sortkey = np.array(meta["compact"], dtype=object)
    assert not (dom_hold & set(cat.vessel_id[[i for _, i in dom_tr]])) and not (dom_hold & set(cat.vessel_id[syn_pool]))
    print(f"DOM_ALL={DOM_ALL} fold {FOLD}: real pairs {len(real)}, synthetic pool {len(syn_pool)}, domestic train vessels {len(dom_tr)} "
          f"(x{DOMREPS}/epoch), domestic hold-out {len(dom_hold)} (excluded), mine {MINE_EPOCHS}", flush=True)

    model = DualEncoder().to(DEV)  # init on CPU with the seeded generator, then move (no-op on cpu)
    print(f"device {DEV}" + (f" ({torch.cuda.get_device_name(DEV)}, tf32={torch.backends.cudnn.allow_tf32})" if DEV.type == "cuda" else ""), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.RandomState(SEED)
    B = 384
    total_steps = None
    codes = meta["codes"]
    hard_prev = None
    cemb_m = None
    step = 0
    for ep in range(EPOCHS):
        # fresh synthetic Hangul each epoch (Converter is random: spacing, digit placement, letter spellings)
        reuse = bool(MINE_EPOCHS) and ep > MINE_EPOCHS[0] and ep not in MINE_EPOCHS and hard_prev is not None
        if not reuse:
            pairs = []
            for i in rng.choice(syn_pool, size=min(SYN_PER_EPOCH, len(syn_pool)), replace=False):
                h = cv.name(meta["names"][i], rng)
                if h:
                    pairs.append((h, i))
            ndom = 0
            for _ in range(DOMREPS):
                for nm, i in dom_tr:
                    h = cv.name(nm, rng)
                    if h:
                        pairs.append((h, i)); ndom += 1
            pairs += real * 6
        hs = [h_tokens(augment(h, rng)) for h, _ in pairs]
        tg = np.array([i for _, i in pairs])
        n = len(pairs)
        if total_steps is None:
            total_steps = EPOCHS * (n // B)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total_steps + 10, pct_start=0.1)
        hard = hard_prev if reuse else None
        if ep in MINE_EPOCHS:
            cemb_m = torch.from_numpy(encode(model.l, cat_tok)).to(DEV)
            qemb_m = torch.from_numpy(encode(model.h, hs)).to(DEV)
            ccodes = torch.from_numpy(codes).to(DEV)
            tcodes = torch.from_numpy(codes[tg]).to(DEV)
            hard = np.zeros((n, NHARD), dtype=np.int64)
            with torch.no_grad():
                for a in range(0, n, 1024):
                    top = torch.topk(qemb_m[a:a + 1024] @ cemb_m.T, NHARD + 4, dim=1).indices  # sorted
                    ok = ccodes[top] != tcodes[a:a + 1024, None]
                    # stable: valid neighbours first, in score order
                    order = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)
                    hard[a:a + len(top)] = torch.gather(top, 1, order)[:, :NHARD].cpu().numpy()
            hard_prev = hard
            print(f"  mined hard negatives ep {ep}: wall {time.time() - t_start:.0f}s cpu {time.process_time() - c_start:.0f}s", flush=True)
        # half of each batch: a contiguous window of pairs sorted by catalog name (hard in-batch negatives)
        srt = np.argsort(sortkey[tg], kind="stable")
        perm = rng.permutation(n)
        losses = []
        for b in range(n // B):
            a0 = rng.randint(0, n - B // 2)
            idx = np.concatenate([srt[a0:a0 + B // 2], perm[b * (B // 2):(b + 1) * (B // 2)]])
            # one pair per catalog name inside a batch
            _, keep = np.unique(meta["codes"][tg[idx]], return_index=True)
            idx = idx[keep]
            qh = model.h(pad([hs[k] for k in idx]).to(DEV))
            ql = model.l(pad([cat_tok[tg[k]] for k in idx]).to(DEV))
            scale = model.logit_scale.exp().clamp(max=100)
            logits = scale * qh @ ql.T
            lab = torch.arange(len(idx), device=DEV)
            if hard is not None:
                hn = np.array([hard[k][rng.randint(NHARD)] for k in idx])
                qn_ = model.l(pad([cat_tok[c] for c in hn]).to(DEV))
                lh = scale * qh @ qn_.T
                clash = torch.from_numpy(codes[hn][None, :] == codes[tg[idx]][:, None]).to(DEV)
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
            if ep == 0 and b in (4, 19):
                print(f"  step {b} of {n // B}: loss {loss.item():.3f} wall {time.time() - t_start:.0f}s "
                      f"cpu {time.process_time() - c_start:.0f}s", flush=True)
        print(f"ep {ep} pairs {n} (dom {ndom}) loss {np.mean(losses):.3f} scale {model.logit_scale.exp().item():.1f} "
              f"wall {time.time() - t_start:.0f}s cpu {time.process_time() - c_start:.0f}s", flush=True)
    train_cpu = time.process_time() - c_start
    FN = "full" if FOLD < 0 else f"f{FOLD}"
    cemb = encode(model.l, cat_tok)  # (encode before save: no state change; lets the GPU do the catalog pass)
    model.to("cpu")  # no-op on cpu; same state_dict format/dtype as the CPU trainer
    torch.save(model.state_dict(), OUT / f"de_{FN}{TAG}.pt")
    np.save(OUT / f"de_{FN}{TAG}_cat.npy", cemb.astype(np.float16))
    json.dump({"fold": FOLD, "train_cpu_s": train_cpu, "train_wall_s": time.time() - t_start, "threads": THREADS, "device": str(DEV),
               "epochs": EPOCHS, "mine": list(MINE_EPOCHS), "domreps": DOMREPS, "dom_train_vessels": len(dom_tr), "generator": "kogen6"},
              open(OUT / f"de_{FN}{TAG}_timing.json", "w"))
    print("saved", OUT / f"de_{FN}{TAG}.pt", flush=True)
