"""Lightweight CUDA-event section profiler for the training loop.

Zero overhead when disabled (the default). Enable with ``enable()``; wrap
regions in ``with section('name'):``; call ``report()`` to synchronize and
get per-section stats. Events measure GPU-timeline time between the two
records, so launch-bound idle gaps inside a section are correctly included.
"""

import torch

_ENABLED = False
_records = {}   # name -> list[(start_event, end_event)]


def enable():
    global _ENABLED
    _ENABLED = True


class section:
    __slots__ = ("name", "_s", "_e")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if _ENABLED:
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
            self._s.record()
        return self

    def __exit__(self, *exc):
        if _ENABLED:
            self._e.record()
            _records.setdefault(self.name, []).append((self._s, self._e))
        return False


def report(reset=True):
    """{name: (count, mean_ms, total_ms)} sorted by total, descending."""
    torch.cuda.synchronize()
    out = {}
    for name, evs in _records.items():
        ts = [s.elapsed_time(e) for s, e in evs]
        out[name] = (len(ts), sum(ts) / len(ts), sum(ts))
    if reset:
        _records.clear()
    return dict(sorted(out.items(), key=lambda kv: -kv[1][2]))


def print_report(header=""):
    rep = report()
    total = sum(v[2] for v in rep.values())
    print(f"\n=== section profile {header} (GPU-timeline ms) ===")
    print(f"{'section':28s} {'count':>6s} {'mean':>9s} {'total':>10s} {'share':>7s}")
    for name, (n, mean, tot) in rep.items():
        print(f"{name:28s} {n:6d} {mean:9.2f} {tot:10.0f} {100*tot/total:6.1f}%")
    print(f"{'SUM':28s} {'':6s} {'':9s} {total:10.0f}")
