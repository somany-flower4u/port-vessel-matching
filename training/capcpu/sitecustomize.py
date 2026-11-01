"""de-v2-full track: process cap (machine crashed under load on 2026-10-27). Active only when VM_PROC_CAP > 0 and this
dir is on PYTHONPATH: multiprocessing context Pool(n) -> Pool(min(n, cap)) and os.cpu_count() -> min(cpu_count, cap).
Results do not change: train14/domtrain/dump seed their RNGs per fold/part, and script.py predicts rows independently."""
import os
_cap = int(os.environ.get("VM_PROC_CAP", "0") or 0)
if _cap > 0:
    _orig_cpu = os.cpu_count
    os.cpu_count = lambda: min(_orig_cpu() or 1, _cap)
    import multiprocessing as _mp
    import multiprocessing.context as _mpc
    _orig_pool = _mpc.BaseContext.Pool

    def _pool(self, processes=None, *a, **k):
        n = processes if processes else (os.cpu_count() or 1)
        return _orig_pool(self, min(n, _cap), *a, **k)
    _mpc.BaseContext.Pool = _pool
    _mp.Pool = _mp.context._default_context.Pool
