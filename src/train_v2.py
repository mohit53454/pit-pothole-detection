"""Train the final YOLOv8n on the v2 dataset (full RDD2022 India + 1,243 real potholes).

Usage:
    .venv-dml/Scripts/python src/train_v2.py [epochs] [imgsz] [device] [resume] [threads]

    resume=resume continues runs/pothole/train_v2/weights/last.pt instead of
    starting over (epoch budget and hyper-parameters come from the checkpoint).

Defaults to CPU. DirectML (AMD iGPU) was measured and rejected: even with the
dml_compat shims it runs the loss incorrectly - box_loss and dfl_loss come out
exactly 0 while cls_loss is ~35x too large, so the model never learns to
localise. It is also no faster (~6.2 s/it vs ~7 s/it on CPU at 640/batch16)
because the broken kernels force constant CPU round-trips. Pass `directml:0`
only to experiment, never for a real run.

Writes best.pt to runs/pothole/train_v2/weights/ and a metrics CSV to
outputs/train_log_v2.csv.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo2" / "data.yaml"
RUNS = ROOT / "runs" / "pothole"
OUT = ROOT / "outputs"
WEIGHTS = RUNS / "train_v2" / "weights"


def keep_awake() -> None:
    """Hold a per-process power request so idle sleep cannot cut the run short."""
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS|ES_SYSTEM_REQUIRED


def main() -> int:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    resume = len(sys.argv) > 4 and sys.argv[4] == "resume"
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 6

    use_dml = device.startswith("directml")
    if use_dml:
        sys.path.insert(0, str(ROOT / "src"))
        import dml_compat

        dml_compat.install()
    else:
        # 6 physical cores; leaving the hyperthreads free avoids fighting the dataloader.
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    import torch

    if not use_dml:
        torch.set_num_threads(threads)

    from ultralytics import YOLO

    OUT.mkdir(exist_ok=True)
    keep_awake()
    last = WEIGHTS / "last.pt"
    if resume:
        if not last.exists():
            print(f"no checkpoint to resume from: {last}", flush=True)
            return 1
        model = YOLO(str(last))
    else:
        model = YOLO("yolov8n.pt")  # COCO-pretrained, auto-downloads
    print(f"training v2: epochs={epochs} imgsz={imgsz} device={device} "
          f"resume={resume} threads={torch.get_num_threads()} data={DATA}", flush=True)
    t0 = time.time()
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        seed=42,
        project=str(RUNS),
        name="train_v2",
        exist_ok=True,
        patience=0,        # fixed budget - no early stopping
        close_mosaic=4,    # default 10 would disable mosaic for most of a short run
        cache=False,
        workers=4,
        device=device,
        amp=False if use_dml else True,
        resume=resume,
        verbose=True,
    )
    elapsed = time.time() - t0

    # per-epoch metrics -> outputs/train_log_v2.csv
    import pandas as pd

    src_csv = RUNS / "train_v2" / "results.csv"
    df = pd.read_csv(src_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "epoch": df["epoch"].astype(int),
            "train_box_loss": df["train/box_loss"].round(4),
            "val_box_loss": df["val/box_loss"].round(4),
            "precision": df["metrics/precision(B)"].round(4),
            "recall": df["metrics/recall(B)"].round(4),
            "map50": df["metrics/mAP50(B)"].round(4),
            "map50_95": df["metrics/mAP50-95(B)"].round(4),
        }
    )
    out.to_csv(OUT / "train_log_v2.csv", index=False)
    print(out.tail(epochs).to_string(index=False))

    best = WEIGHTS / "best.pt"
    print(f"\ntrained {epochs} epochs x imgsz{imgsz} on {device} in {elapsed/60:.1f} min")
    print(f"best weights: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
