"""Final trainer: single-class pothole YOLOv8n on the v3 dataset, CPU.

    .venv-dml/Scripts/python src/train_v3.py [epochs] [imgsz] [mode] [threads]
                                            [data.yaml] [run_name]

    mode=start   (default) fresh run from COCO-pretrained yolov8n.pt
    mode=resume  continue runs/pothole/train_v3/weights/last.pt after an
                 interrupt. All hyper-parameters come from the checkpoint, so
                 the epoch budget is whatever the run was started with.
    mode=extend  fine-tune from the finished run's best.pt into a NEW run
                 directory (train_v3_ext) with a fresh, cosine-decayed
                 schedule. Extending is how accuracy keeps climbing past the
                 first budget; it is not a resume, so the LR restarts gently
                 instead of picking up a mid-cycle value.

Why CPU: DirectML is broken on this stack. Even with the src/dml_compat.py
shims, box_loss and dfl_loss come out exactly 0 while cls_loss is ~35x too
large - the model never learns to localise. It is not faster either (~6.2 s/it
vs ~5.8 s/it at 640/batch16) because the broken kernels force constant CPU
round-trips. 6 threads beat 12 on this 6c/12t CPU (~6.6 vs ~7.0 s/it), and
bf16 autocast is slower still, so plain fp32 with 6 threads is the fast path.

Expected cost on this machine, imgsz=512, batch=16, 3,898 images:
    ~3.7 s/it x 244 it  ~= 15 min/epoch.

Writes runs/pothole/<run>/weights/{best,last}.pt and outputs/train_log_v3.csv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo3" / "data.yaml"
RUNS = ROOT / "runs" / "pothole"
OUT = ROOT / "outputs"
WEIGHTS = RUNS / "train_v3" / "weights"


def keep_awake() -> None:
    """Hold a per-process power request so idle sleep cannot cut the run short."""
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS|ES_SYSTEM_REQUIRED


def write_log(run_name: str) -> None:
    """Copy the ultralytics results table to outputs/ with tidy column names."""
    csv = RUNS / run_name / "results.csv"
    if not csv.exists():
        return
    import pandas as pd

    df = pd.read_csv(csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "epoch": df["epoch"].astype(int),
            "train_box_loss": df["train/box_loss"].round(4),
            "train_cls_loss": df["train/cls_loss"].round(4),
            "val_box_loss": df["val/box_loss"].round(4),
            "precision": df["metrics/precision(B)"].round(4),
            "recall": df["metrics/recall(B)"].round(4),
            "map50": df["metrics/mAP50(B)"].round(4),
            "map50_95": df["metrics/mAP50-95(B)"].round(4),
        }
    )
    out.to_csv(OUT / f"train_log_{run_name}.csv", index=False)
    print(out.to_string(index=False), flush=True)


def main() -> int:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    mode = sys.argv[3] if len(sys.argv) > 3 else "start"
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    # Optional overrides so other datasets (e.g. v4) can reuse this trainer
    # without the v3 run directory being clobbered.
    data = Path(sys.argv[5]) if len(sys.argv) > 5 else DATA
    run_override = sys.argv[6] if len(sys.argv) > 6 else None

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    import torch

    torch.set_num_threads(threads)

    from ultralytics import YOLO

    OUT.mkdir(exist_ok=True)
    keep_awake()

    common = dict(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        seed=42,
        project=str(RUNS),
        exist_ok=True,
        patience=0,          # fixed budget, no early stop
        close_mosaic=2,
        cache=False,
        workers=4,
        device="cpu",
        amp=False,           # bf16 autocast measured slower on this CPU
        save_period=1,       # keep every epoch so an interrupt is survivable
        verbose=True,
    )

    if mode == "resume":
        last = WEIGHTS / "last.pt"
        if not last.exists():
            print(f"nothing to resume from: {last}", flush=True)
            return 1
        print(f"resuming {last}", flush=True)
        # ultralytics reads epochs/imgsz/etc. from the checkpoint on resume;
        # passing them again is what makes resume runs silently misconfigure.
        YOLO(str(last)).train(resume=True)
        run_name = "train_v3"
    elif mode == "extend":
        best = WEIGHTS / "best.pt"
        if not best.exists():
            print(f"nothing to extend from: {best}", flush=True)
            return 1
        run_name = run_override or "train_v3_ext"
        print(f"extending from {best}: +{epochs} epochs at {imgsz}px", flush=True)
        YOLO(str(best)).train(
            **common,
            name=run_name,
            lr0=0.002,        # gentle restart; the head is already sensible
            lrf=0.05,         # long cosine tail
            warmup_epochs=1.0,
            cos_lr=True,
        )
    else:
        if not data.exists():
            print(f"missing dataset config: {data}\nrun src/prepare_v3.py first", flush=True)
            return 1
        run_name = run_override or "train_v3"
        print(f"training {run_name}: epochs={epochs} imgsz={imgsz} batch=16 "
              f"threads={torch.get_num_threads()} data={data}", flush=True)
        YOLO("yolov8n.pt").train(**common, name=run_name)  # COCO-pretrained

    write_log(run_name)

    print(f"\nbest weights: {RUNS / run_name / 'weights' / 'best.pt'}", flush=True)
    print("next: .venv-dml/Scripts/python src/export_onnx.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
