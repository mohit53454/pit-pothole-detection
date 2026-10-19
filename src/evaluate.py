"""Evaluate the trained detector: validation metrics + demo inference.

* Runs ultralytics val on the held-out 200-image split -> outputs/metrics.csv
* Runs inference on 20 unlabeled real test images and saves annotated
  result images to outputs/predictions/ (the "wow" artifacts).
* Writes outputs/metrics_summary.txt with the headline numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo" / "data.yaml"
BEST_CANDIDATES = [
    ROOT / "runs" / "pothole" / "train_ext" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train" / "weights" / "best.pt",
]
BEST = next((p for p in BEST_CANDIDATES if p.exists()), BEST_CANDIDATES[-1])
DEMO = ROOT / "data" / "demo_images"
OUT = ROOT / "outputs"


def main() -> int:
    if not BEST.exists():
        print(f"missing weights: {BEST} — run src/train.py first")
        return 1
    OUT.mkdir(exist_ok=True)

    model = YOLO(str(BEST))

    # 1. Held-out validation
    metrics = model.val(data=str(DATA), imgsz=640, device="cpu", verbose=True)
    per_class = pd.DataFrame(
        {
            "class": list(metrics.names.values()),
            "precision": metrics.box.p.tolist(),
            "recall": metrics.box.r.tolist(),
            "map50": metrics.box.map50.tolist(),
            "map50_95": metrics.box.map.tolist(),
        }
    ).round(4)
    per_class.to_csv(OUT / "metrics.csv", index=False)

    # 2. Demo inference on unseen (unlabeled) test images
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    results = model.predict(
        source=str(DEMO),
        imgsz=640,
        device="cpu",
        conf=0.25,
        save=True,
        save_txt=False,
        project=str(OUT),
        name="predictions",
        exist_ok=True,
    )
    n_boxes = sum(len(r.boxes) for r in results)

    # 3. Headline summary
    summary = (
        f"Validation (200 held-out India images)\n"
        f"  mAP@50      : {metrics.box.map50:.3f}\n"
        f"  mAP@50-95   : {metrics.box.map:.3f}\n"
        f"  precision   : {metrics.box.mp:.3f}\n"
        f"  recall      : {metrics.box.mr:.3f}\n"
        f"\nDemo inference ({len(results)} unseen test images, conf>=0.25)\n"
        f"  damage boxes found: {n_boxes}\n"
    )
    (OUT / "metrics_summary.txt").write_text(summary)
    print(summary)
    print("annotated demo images ->", pred_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
