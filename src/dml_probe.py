"""Probe which op inside v8DetectionLoss.preprocess misbehaves on DirectML."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import dml_compat

dml = dml_compat.install()


def make_batch(nl, nb, seed):
    g = torch.Generator().manual_seed(seed)
    # sorted image indices, several targets per image (dataloader-like)
    counts = torch.randint(1, 6, (nb,), generator=g)
    counts = counts / counts.sum().clamp(min=1) * nl
    counts = counts.round().clamp(min=1).long()
    while int(counts.sum()) < nl:
        counts[0] += 1
    while int(counts.sum()) > nl:
        k = int((counts > 1).nonzero()[0])
        counts[k] -= 1
    idx = torch.repeat_interleave(torch.arange(nb), counts)[:nl]
    return idx.long(), counts.long()


def check(name, fn):
    try:
        ok = fn()
        print(f"  {name}: {'OK' if ok else 'MISMATCH'}", flush=True)
        return ok
    except Exception:
        print(f"  {name}: FAIL", flush=True)
        traceback.print_exc(limit=4)
        return False


def main():
    for nl in (100, 400):
        idx_cpu, _ = make_batch(nl, 16, seed=nl)
        idx = idx_cpu.to(dml)
        print(f"--- nl={nl} ---")

        def t_unique():
            _, a = idx.unique(return_counts=True)
            _, b = idx_cpu.unique(return_counts=True)
            return torch.equal(a.cpu(), b)

        def t_arange():
            return torch.equal(torch.arange(nl, device=dml).cpu(), torch.arange(nl))

        def t_scatter():
            o = torch.zeros(17, dtype=torch.long, device=dml)
            o.scatter_add_(0, idx + 1, torch.ones_like(idx))
            ref = torch.zeros(17, dtype=torch.long)
            ref.scatter_add_(0, idx_cpu + 1, torch.ones_like(idx_cpu))
            return torch.equal(o.cpu(), ref.cumsum(0)) and torch.equal(o.cumsum(0).cpu(), ref.cumsum(0))

        def t_gather():
            ref = torch.zeros(17, dtype=torch.long)
            ref.scatter_add_(0, idx_cpu + 1, torch.ones_like(idx_cpu))
            ref = ref.cumsum(0)
            return torch.equal(ref[idx_cpu], ref.to(dml)[idx].cpu())

        def t_sub():
            ref = torch.zeros(17, dtype=torch.long)
            ref.scatter_add_(0, idx_cpu + 1, torch.ones_like(idx_cpu))
            ref = ref.cumsum(0)
            w = torch.arange(nl, device=dml) - ref.to(dml)[idx]
            return torch.equal(w.cpu(), torch.arange(nl) - ref[idx_cpu])

        check("unique(counts)", t_unique)
        check("arange", t_arange)
        check("scatter_add_", t_scatter)
        check("gather offsets[idx]", t_gather)
        check("arange - offsets[idx]", t_sub)

    print("PROBE_DONE")


if __name__ == "__main__":
    main()
