"""Cheap DirectML/CPU smoke test + throughput benchmark.

Shakes out dtype/kernel bugs in the ultralytics loss + validation path before
committing to a multi-hour v2 run, and reports it/s so the faster backend can
be chosen for the real training.

    .venv-dml/Scripts/python src/dml_smoke.py [device] [imgsz] [batch] [val] [workers] [threads]

    val=1 (default) runs the full validation pass; val=0 skips it for pure
    throughput comparison. threads caps torch's intra-op pool - on a 6-core/
    12-thread CPU, 6 compute threads + 4 dataloader workers beats 12 + 0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "directml:0"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    do_val = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    threads = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    if device.startswith("directml"):
        sys.path.insert(0, str(ROOT / "src"))
        import dml_compat

        dml_compat.install()

    import torch

    if threads:
        torch.set_num_threads(threads)

    from ultralytics import YOLO

    tag = f"smoke_{device.replace(':', '_')}_{imgsz}_{batch}_w{workers}_t{threads}"
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    model.train(
        data=str(ROOT / "data" / "yolo2" / "data.yaml"),
        epochs=1,
        imgsz=imgsz,
        batch=batch,
        fraction=0.05,
        seed=42,
        project=str(ROOT / "runs" / "pothole"),
        name=tag,
        exist_ok=True,
        patience=0,
        cache=False,
        workers=workers,
        device=device,
        amp=False,
        plots=False,
        val=do_val,
        verbose=True,
    )
    print(f"\nSMOKE OK device={device} imgsz={imgsz} batch={batch} "
          f"val={do_val} workers={workers} threads={threads or torch.get_num_threads()} "
          f"in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())