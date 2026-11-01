"""LLM spelling judge: log p(Hangul port spelling | Latin catalog name) under a frozen causal LM.

Self-contained copy of the go/no-go study's scorer (form "ko10": a fixed
instruction + 10 real train example pairs, rebuilt from train.csv by sample
id at run time; see shots_from_frames).  Each pair is scored on its own
(batch 1); the fixed prompt prefix is encoded once and its KV cache reused,
cropped back after every pair, so every pair sees exactly the same
conditioning.  Only the target tokens (" " + ko + "\\n") are scored; they are
tokenised separately from the prompt, so they are identical across the
candidates of one row and raw sums are comparable.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

# The ten few-shot pairs are real rows of the official train.csv. They are not redistributed in this
# repository: SHOTS is rebuilt from the competition files by sample id (shots_from_frames).
SHOT_SAMPLE_IDS = ["TRAIN_00040", "TRAIN_00132", "TRAIN_00182", "TRAIN_06181", "TRAIN_00914",
                   "TRAIN_01531", "TRAIN_06222", "TRAIN_01768", "TRAIN_01050", "TRAIN_05578"]
SHOTS: list = []


def shots_from_frames(train, catalog):
    """(catalog English name, Hangul port spelling) of the SHOT_SAMPLE_IDS rows, in that order."""
    tr = train.set_index("sample_id")
    name = dict(zip(catalog.vessel_id.astype(str), catalog.vessel_name))
    return [(name[str(tr.at[s, "target_vessel_id"])], tr.at[s, "vessel_name"]) for s in SHOT_SAMPLE_IDS]


def form(name):
    """-> (prefix, mid_template).  Target is always ' ' + ko + '\\n'."""
    if name == "ko10":    # 10-shot Korean
        if not SHOTS:
            raise RuntimeError("few-shot pairs not loaded: SHOTS[:] = shots_from_frames(train, catalog)")
        p = "다음은 외국 선박의 영문 선박명과 한국 항만 입항 기록에 적힌 한글 표기입니다.\n\n"
        p += "".join(f"선박 영문명: {e}\n한글 표기: {k}\n\n" for e, k in SHOTS)
        return p, "선박 영문명: {en}\n한글 표기:"
    raise ValueError(name)


class Judge:
    def __init__(self, model_dir, form_name="ko10", device="cpu", dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = torch.device(device)
        dtype = dtype or (torch.bfloat16 if self.dev.type == "cuda" else torch.float32)
        model_dir = str(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype,
                                                      local_files_only=True).to(self.dev).eval()
        self.prefix, self.mid = form(form_name)
        self.pre_ids = self.tok(self.prefix, add_special_tokens=False).input_ids if self.prefix else []
        self.cache = None
        if self.pre_ids:
            with torch.inference_mode():
                o = self.m(input_ids=torch.tensor([self.pre_ids], device=self.dev), use_cache=True)
            self.cache = o.past_key_values
            self.Lp = len(self.pre_ids)

    def score(self, en, ko):
        """-> (sum of target-token log-probabilities, number of target tokens)."""
        with torch.inference_mode():
            return self._score(en, ko)

    def _score(self, en, ko):
        mid = self.tok(self.mid.format(en=en), add_special_tokens=False).input_ids
        tgt = self.tok(" " + ko + "\n", add_special_tokens=False).input_ids
        ids = torch.tensor([mid + tgt], device=self.dev)
        if self.cache is not None:
            try:
                o = self.m(input_ids=ids, past_key_values=self.cache, use_cache=True)
            finally:
                self.cache.crop(self.Lp)
        else:
            o = self.m(input_ids=ids, use_cache=False)
        lg = o.logits[0, len(mid) - 1: len(mid) + len(tgt) - 1].float().log_softmax(-1)
        t = torch.tensor(tgt, device=self.dev)
        return float(lg.gather(1, t[:, None]).sum()), len(tgt)
