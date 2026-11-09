"""DirectML (AMD iGPU) 1-epoch benchmark on the v1 subset — compares vs CPU's 9.8 min.

Usage:  .venv-dml/Scripts/python src/bench_dml2.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dml_compat  # noqa: E402

dml_compat.install()

from ultralytics import YOLO  # noqa: E402


def main() -> int:
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    try:
        model.train(
            data=str(ROOT / "data" / "yolo" / "data.yaml"),  # v1 subset, same as CPU bench
            epochs=1,
            imgsz=640,
            batch=16,
            seed=42,
            project=str(ROOT / "runs" / "pothole"),
            name="train_dml_bench3",
            exist_ok=True,
            patience=0,
            cache=False,
            workers=2,
            device="directml:0",
            amp=False,
            verbose=True,
        )
    except Exception:
        traceback.print_exc()
        print("DML_TRAIN_FAILED", flush=True)
        return 1
    dt = time.time() - t0
    print(f"DML_BENCH_OK {dt / 60:.2f} min for 1 epoch incl validation", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
