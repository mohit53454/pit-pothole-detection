"""Train YOLOv8n on the RDD2022 India subset (CPU-friendly).

Usage:  python src/train.py [epochs] [imgsz]

Transfer learning from COCO-pretrained yolov8n (6 MB). Damage classes map
roughly to crack-like textures the backbone already knows. Saves best.pt into
runs/pothole/ and appends per-epoch metrics to outputs/train_log.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo" / "data.yaml"
RUNS = ROOT / "runs" / "pothole"
OUT = ROOT / "outputs"


def main() -> int:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    weights = sys.argv[3] if len(sys.argv) > 3 else "yolov8n.pt"
    run_name = "train" if weights == "yolov8n.pt" else "train_ext"
    OUT.mkdir(exist_ok=True)

    model = YOLO(weights)  # COCO-pretrained by default, auto-downloads ~6 MB
    t0 = time.time()
    results = model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        seed=42,
        project=str(RUNS),
        name=run_name,
        exist_ok=True,
        patience=0,           # no early stopping — fixed budget
        cache=False,          # keep RAM modest
        workers=2,
        device="cpu",
        verbose=True,
    )
    elapsed = time.time() - t0

    # Persist per-epoch metrics from the trainer's own results.csv
    src_csv = RUNS / run_name / "results.csv"
    csv_path = OUT / ("train_log_ext.csv" if run_name == "train_ext" else "train_log.csv")
    import pandas as pd

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
    out.to_csv(csv_path, index=False)
    print(out.tail(epochs).to_string(index=False))

    best = RUNS / run_name / "weights" / "best.pt"
    print(f"\ntrained {epochs} epochs x imgsz{imgsz} in {elapsed/60:.1f} min")
    print(f"best weights: {best}")
    print(f"metrics csv : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
