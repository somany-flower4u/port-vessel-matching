"""LLM spelling judge: log p(Hangul port spelling | Latin catalog name) under a frozen causal LM.

judge.py MODEL_DIR PAIRS_CSV OUT_CSV FORM [--device cpu|cuda] [--threads N] [--limit N]
PAIRS_CSV columns: en, ko.  Each pair is scored on its own (batch 1); the fixed prompt prefix
(instruction + few-shot) is encoded once and its KV cache reused, cropped back after every pair,
so every pair sees exactly the same conditioning.  Only the target tokens (" " + ko + "\n") are scored;
the target token ids are tokenised separately from the prompt, so they are identical across the
candidates of one row and raw sums are comparable."""
import sys, time, argparse, math
import pandas as pd
try:
    import torch
except ImportError:  # prep.py on the Mac only needs SHOTS
    torch = None

import os
# The ten few-shot pairs are real rows of the official train.csv. They are not redistributed in this
# repository: SHOTS is rebuilt from the competition files by sample id (shots_from_frames).
SHOT_SAMPLE_IDS = ["TRAIN_00040", "TRAIN_00132", "TRAIN_00182", "TRAIN_06181", "TRAIN_00914",
                   "TRAIN_01531", "TRAIN_06222", "TRAIN_01768", "TRAIN_01050", "TRAIN_05578"]
def shots_from_frames(train, catalog):
    """(catalog English name, Hangul port spelling) of the SHOT_SAMPLE_IDS rows, in that order."""
    tr = train.set_index("sample_id")
    name = dict(zip(catalog.vessel_id.astype(str), catalog.vessel_name))
    return [(name[str(tr.at[s, "target_vessel_id"])], tr.at[s, "vessel_name"]) for s in SHOT_SAMPLE_IDS]


try:
    _WS = os.environ.get("SEOKCHEON_WS", ".")
    SHOTS = shots_from_frames(pd.read_csv(f"{_WS}/work/train.csv"), pd.read_csv(f"{_WS}/work/vessel_catalog.csv"))
except OSError:  # competition files not present: few-shot forms are unavailable
    SHOTS = []
SHOT_NAMES = {e for e, _ in SHOTS}

def form(name):
    """-> (prefix, mid_template).  Target is always ' ' + ko + '\\n'."""
    if name == "ko0":     # zero-shot Korean
        return "", "선박 영문명: {en}\n한글 표기:"
    if name == "ko10":    # 10-shot Korean
        p = "다음은 외국 선박의 영문 선박명과 한국 항만 입항 기록에 적힌 한글 표기입니다.\n\n"
        p += "".join(f"선박 영문명: {e}\n한글 표기: {k}\n\n" for e, k in SHOTS)
        return p, "선박 영문명: {en}\n한글 표기:"
    if name == "en10":    # 10-shot English framing
        p = "Ship names: the English catalog name and the Korean (Hangul) transliteration written in Korean port records.\n\n"
        p += "".join(f"English: {e}\nKorean: {k}\n\n" for e, k in SHOTS)
        return p, "English: {en}\nKorean:"
    raise ValueError(name)

class Judge:
    def __init__(self, model_dir, form_name, device="cpu", dtype=None, load8=False, offload=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = torch.device(device)
        dtype = dtype or (torch.bfloat16 if self.dev.type == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if load8:  # LLM.int8 weights (bitsandbytes), bf16 activations -- only when bf16 does not fit
            from transformers import BitsAndBytesConfig
            self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0},
                                                          quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
        elif offload:  # bf16, as many layers on the GPU as fit in `offload` GiB, the rest run on the CPU
            self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="auto",
                                                          max_memory={0: f"{offload}GiB", "cpu": "24GiB"}).eval()
        else:
            self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(self.dev).eval()
        self.prefix, self.mid = form(form_name)
        self.pre_ids = self.tok(self.prefix, add_special_tokens=False).input_ids if self.prefix else []
        self.cache = None
        if self.pre_ids:
            with torch.inference_mode():
                o = self.m(input_ids=torch.tensor([self.pre_ids], device=self.dev), use_cache=True)
            self.cache = o.past_key_values; self.Lp = len(self.pre_ids)
            self.pre_last = o.logits[0, -1].float()

    def score(self, en, ko):
      with torch.inference_mode():
        return self._score(en, ko)

    def _score(self, en, ko):
        mid = self.tok(self.mid.format(en=en), add_special_tokens=False).input_ids
        tgt = self.tok(" " + ko + "\n", add_special_tokens=False).input_ids
        ids = torch.tensor([mid + tgt], device=self.dev)
        if self.cache is not None:
            o = self.m(input_ids=ids, past_key_values=self.cache, use_cache=True)
            self.cache.crop(self.Lp)
        else:
            o = self.m(input_ids=ids, use_cache=False)
        lg = o.logits[0, len(mid) - 1: len(mid) + len(tgt) - 1].float().log_softmax(-1)
        t = torch.tensor(tgt, device=self.dev)
        return float(lg.gather(1, t[:, None]).sum()), len(tgt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("pairs"); ap.add_argument("out"); ap.add_argument("form")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=6); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--load8", action="store_true"); ap.add_argument("--offload", type=float, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    P = pd.read_csv(a.pairs)
    if a.limit: P = P.head(a.limit)
    t0 = time.time(); J = Judge(a.model, a.form, a.device, load8=a.load8, offload=a.offload); tl = time.time() - t0
    if a.device == "cuda":
        torch.cuda.synchronize(); print(f"VRAM allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)
    print(f"loaded in {tl:.1f}s; prefix tokens {len(J.pre_ids)}", flush=True)
    res = []; t0 = time.time()
    for i, (en, ko) in enumerate(zip(P.en, P.ko)):
        lp, n = J.score(en, ko); res.append((lp, n))
        if i % 200 == 0: print(i, f"{(time.time()-t0)/(i+1)*1000:.1f} ms/pair", flush=True)
    dt = time.time() - t0
    if a.device == "cuda": print(f"peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    P["lp"] = [r[0] for r in res]; P["ntok"] = [r[1] for r in res]
    P.to_csv(a.out, index=False)
    print(f"done {len(P)} pairs in {dt:.1f}s = {dt/len(P)*1000:.1f} ms/pair (device {a.device}, threads {a.threads})")
