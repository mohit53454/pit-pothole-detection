"""Time yolov8n forward+backward on this CPU under different precisions/threads.

    .venv-dml/Scripts/python src/bench_cpu.py [imgsz] [batch] [iters]

Prints seconds/iteration for each config so the real training run can use the
fastest one. This is a synthetic-batch benchmark: it isolates the compute cost
from the dataloader, which is what decides the epoch time at 640px.
"""

from __future__ import annotations

import sys
import time

import torch

imgsz = int(sys.argv[1]) if len(sys.argv) > 1 else 640
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 16
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def build():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt").model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def total(out):
    """Sum every float tensor the head produced (accepts dict/list/tensor)."""
    import torch as _t

    if isinstance(out, _t.Tensor):
        return out.float().sum()
    if isinstance(out, dict):
        return sum(total(v) for v in out.values())
    if isinstance(out, (list, tuple)):
        return sum(total(v) for v in out)
    return _t.tensor(0.0)


def run(model, threads, dtype, tag):
    torch.set_num_threads(threads)
    x = torch.randn(batch, 3, imgsz, imgsz)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # warmup
    for _ in range(2):
        loss = total(model(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
    t0 = time.time()
    for _ in range(iters):
        if dtype == torch.bfloat16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = total(model(x))
        else:
            loss = total(model(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = (time.time() - t0) / iters
    print(f"{tag:28s} {dt:6.2f} s/it   -> {dt * (8053 / batch) / 60:5.1f} min/epoch", flush=True)
    return dt


def main():
    print(f"torch {torch.__version__}  imgsz={imgsz} batch={batch} iters={iters}", flush=True)
    results = {}
    for tag, threads, dtype in (
        ("fp32  threads=6", 6, None),
        ("fp32  threads=12", 12, None),
        ("bf16  threads=6", 6, torch.bfloat16),
        ("bf16  threads=12", 12, torch.bfloat16),
    ):
        model = build()
        try:
            results[tag] = run(model, threads, dtype, tag)
        except Exception as e:  # noqa: BLE001 - report, keep benchmarking
            print(f"{tag:28s} FAILED: {type(e).__name__}: {e}", flush=True)
        del model

    if results:
        best = min(results, key=results.get)
        print(f"\nBENCH_DONE best={best} ({results[best]:.2f} s/it)", flush=True)


if __name__ == "__main__":
    main()
