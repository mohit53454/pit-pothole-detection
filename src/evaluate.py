"""Evaluate a trained detector: held-out metrics + annotated demo inference.

    .venv-dml/Scripts/python src/evaluate.py [weights.pt] [data.yaml] [tag]

Defaults to the best available run (train_v2 -> train_ext -> train) and the
dataset it was trained on. Outputs carry the tag so v1 and v2 results coexist:
outputs/metrics_<tag>.csv, metrics_summary_<tag>.txt, predictions_<tag>/.

Demo inference runs on the 20 unlabeled RDD2022 test photos plus, for v2, a
held-out slice of the real-pothole validation images - the same photos pushed
to the phone to check the app's gallery path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DEMO = ROOT / "data" / "demo_images"

# run name, dataset, output tag
CANDIDATES = [
    ("train_v3_ext", ROOT / "data" / "yolo3" / "data.yaml", "v3ext"),
    ("train_v3", ROOT / "data" / "yolo3" / "data.yaml", "v3"),
    ("train_v2", ROOT / "data" / "yolo2" / "data.yaml", "v2"),
    ("train_ext", ROOT / "data" / "yolo" / "data.yaml", "v1"),
    ("train", ROOT / "data" / "yolo" / "data.yaml", "v1base"),
]


def val_image_count(data_yaml: Path) -> int:
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    val = base / cfg["val"]
    return sum(1 for p in val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def pothole_val_sample(data_yaml: Path, limit: int = 12) -> list[Path]:
    """Held-out real-pothole photos, biggest D40 box first - clear demo targets."""
    val = data_yaml.parent / "images" / "val"
    scored = []
    for img in val.glob("img-*.jpg"):
        lbl = data_yaml.parent / "labels" / "val" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        area = 0.0
        for line in lbl.read_text().splitlines():
            parts = line.split()
            # class 3 in the v2 4-class dataset, class 0 in the v3 pothole set
            if len(parts) == 5 and parts[0] in ("0", "3"):
                area = max(area, float(parts[3]) * float(parts[4]))
        if area > 0:
            scored.append((area, img))
    scored.sort(key=lambda t: -t[0])
    return [p for _, p in scored[:limit]]


def main() -> int:
    run, data, tag = next(
        (c for c in CANDIDATES if (ROOT / "runs" / "pothole" / c[0] / "weights" / "best.pt").exists()),
        CANDIDATES[0],
    )
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "pothole" / run / "weights" / "best.pt"
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else data
    tag = sys.argv[3] if len(sys.argv) > 3 else tag

    if not weights.exists():
        print(f"missing weights: {weights} - train first")
        return 1
    OUT.mkdir(exist_ok=True)
    print(f"evaluating {weights} on {data} (tag={tag})", flush=True)

    model = YOLO(str(weights))

    # 1. Held-out validation
    imgsz = 512 if tag == "v3" else 640
    metrics = model.val(data=str(data), imgsz=imgsz, device="cpu", verbose=True)
    per_class = pd.DataFrame(
        {
            "class": list(metrics.names.values()),
            "precision": metrics.box.p.tolist(),
            "recall": metrics.box.r.tolist(),
            "map50": metrics.box.map50.tolist(),
            "map50_95": metrics.box.map.tolist(),
        }
    ).round(4)
    per_class.to_csv(OUT / f"metrics_{tag}.csv", index=False)

    # 2. Demo inference on unseen photos (flat path list - ultralytics rejects nesting)
    sources = sorted(p for p in DEMO.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if tag in ("v2", "v3"):
        sources += pothole_val_sample(data)
    pred_name = f"predictions_{tag}"
    results = model.predict(
        source=[str(p) for p in sources],
        imgsz=imgsz,
        device="cpu",
        conf=0.25,
        save=True,
        save_txt=False,
        project=str(OUT),
        name=pred_name,
        exist_ok=True,
    )
    n_boxes = sum(len(r.boxes) for r in results)

    # 3. Headline summary
    summary = (
        f"Weights: {weights}\n"
        f"Validation ({val_image_count(data)} held-out images)\n"
        f"  mAP@50      : {metrics.box.map50:.3f}\n"
        f"  mAP@50-95   : {metrics.box.map:.3f}\n"
        f"  precision   : {metrics.box.mp:.3f}\n"
        f"  recall      : {metrics.box.mr:.3f}\n"
        f"\nPer class\n{per_class.to_string(index=False)}\n"
        f"\nDemo inference ({len(results)} unseen images, conf>=0.25)\n"
        f"  damage boxes found: {n_boxes}\n"
    )
    (OUT / f"metrics_summary_{tag}.txt").write_text(summary)
    print(summary)
    print("annotated demo images ->", OUT / pred_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())