"""Precision/recall at every confidence threshold, on the held-out val split.

    .venv-dml/Scripts/python src/threshold_sweep.py [weights.pt] [imgsz]

A single mAP number hides the choice the app actually has to make: lower the
confidence slider and you find more potholes but also invent more. This predicts
the whole val split once at conf=0.001, matches every prediction to ground truth
at IoU 0.50, and then sweeps the threshold in numpy to get the real curve.

Printed:
  * precision / recall / F1 every 0.05 of confidence
  * the highest recall that still holds precision >= 0.95 / 0.90 / 0.80 / 0.70
  * the best-F1 point

"Recall" means: of the potholes a human labelled, how many did the model find.
Raising one costs the other - no detector gets both at 99% on this data, which
is why the trade-off is printed instead of a single headline figure.

Writes outputs/threshold_sweep.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo3" / "data.yaml"
OUT = ROOT / "outputs"
IOU_MATCH = 0.50

WEIGHTS = [
    ROOT / "runs" / "pothole" / "train_v3_ext" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train_v3" / "weights" / "best.pt",
]


def pick_weights(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    for w in WEIGHTS:
        if w.exists():
            return w
    raise SystemExit("no trained weights found; run src/train_v3.py first")


def read_gt(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """YOLO normalised centre/size labels -> pixel xyxy boxes."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv: list[str]) -> int:
    weights = pick_weights(argv)
    imgsz = int(argv[2]) if len(argv) > 2 else 512
    if not DATA.exists():
        raise SystemExit(f"missing dataset config: {DATA}")

    import numpy as np
    import pandas as pd
    from PIL import Image
    from ultralytics import YOLO

    OUT.mkdir(exist_ok=True)
    img_dir = ROOT / "data" / "yolo3" / "images" / "val"
    lbl_dir = ROOT / "data" / "yolo3" / "labels" / "val"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no validation images in {img_dir}")

    print(f"weights : {weights}", flush=True)
    print(f"data    : {len(images)} val images, imgsz={imgsz}", flush=True)

    model = YOLO(str(weights))

    # (confidence, is_true_positive) for every prediction on the whole split.
    scored: list[tuple[float, bool]] = []
    n_gt = 0
    # Per-image presence signals, for the "is there a pothole in this frame" task.
    has_gt: list[bool] = []
    max_conf: list[float] = []

    for i, img_path in enumerate(images, 1):
        with Image.open(img_path) as im:
            w, h = im.size
        gts = read_gt(lbl_dir / (img_path.stem + ".txt"), w, h)
        n_gt += len(gts)
        has_gt.append(len(gts) > 0)

        r = model.predict(str(img_path), imgsz=imgsz, conf=0.001, iou=0.45,
                          max_det=100, device="cpu", verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))

        max_conf.append(float(confs.max()) if len(confs) else 0.0)

        taken = [False] * len(gts)
        for box, c in sorted(zip(boxes, confs), key=lambda t: -t[1]):
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gts):
                if taken[j]:
                    continue
                v = iou(box, g)
                if v > best_iou:
                    best_iou, best_j = v, j
            hit = best_iou >= IOU_MATCH and best_j >= 0
            if hit:
                taken[best_j] = True
            scored.append((float(c), hit))

        if i % 100 == 0:
            print(f"  {i}/{len(images)}", flush=True)

    if n_gt == 0:
        raise SystemExit("no ground-truth boxes found for the val split")

    scored.sort(key=lambda t: -t[0])
    confs = np.array([c for c, _ in scored])
    tps = np.array([1 if t else 0 for _, t in scored])
    # Sweep only where the threshold can change anything.
    grid = np.round(np.arange(0.05, 0.80, 0.05), 2)

    rows = []
    for t in grid:
        sel = confs >= t
        tp = int(tps[sel].sum())
        fp = int(sel.sum() - tp)
        fn = n_gt - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        rows.append({"conf": float(t), "precision": round(p, 4),
                     "recall": round(r_, 4), "f1": round(f1, 4),
                     "tp": tp, "fp": fp, "fn": fn})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "threshold_sweep.csv", index=False)

    print(f"\n  {n_gt} labelled potholes across {len(images)} images")
    print("\n  conf   precision    recall       f1      tp    fp    fn")
    print("  " + "-" * 55)
    for r in rows:
        print(f"  {r['conf']:4.2f}   {r['precision']:9.3f}  {r['recall']:8.3f}"
              f"  {r['f1']:7.3f}  {r['tp']:4d}  {r['fp']:4d}  {r['fn']:4d}")

    print("\n  operating points")
    print("  " + "-" * 60)
    for target in (0.95, 0.90, 0.80, 0.70):
        ok = df[df["precision"] >= target]
        if ok.empty:
            print(f"  precision >= {target:.2f} : not reachable at any threshold")
        else:
            best = ok.loc[ok["recall"].idxmax()]
            print(f"  precision >= {target:.2f} : conf {best['conf']:.2f} -> "
                  f"recall {best['recall']:.3f} ({int(best['tp'])}/{n_gt} found)")

    fb = df.loc[df["f1"].idxmax()]
    print(f"  best F1          : conf {fb['conf']:.2f} -> precision {fb['precision']:.3f}, "
          f"recall {fb['recall']:.3f}")

    # ---- image-level presence -------------------------------------------------
    # A different, much easier question: does this frame contain a pothole at all?
    # This is the task where a high accuracy figure is actually meaningful.
    import numpy as _np

    hg = _np.array(has_gt)
    mc = _np.array(max_conf)
    n_pos, n_neg = int(hg.sum()), int((~hg).sum())
    print(f"\n  image-level presence: {n_pos} with a pothole, {n_neg} without")
    print("\n  conf   accuracy   false-alarm   miss-rate")
    print("  " + "-" * 44)
    best_conf, best_acc = 0.5, -1.0
    for t in _np.arange(0.05, 0.80, 0.05):
        pred_pos = mc >= t
        tp = int((pred_pos & hg).sum())
        fp = int((pred_pos & ~hg).sum())
        fn = int((~pred_pos & hg).sum())
        tn = int((~pred_pos & ~hg).sum())
        acc = (tp + tn) / len(hg)
        far = fp / n_neg if n_neg else 0.0      # clean frames wrongly flagged
        miss = fn / n_pos if n_pos else 0.0     # pothole frames wrongly cleared
        print(f"  {t:4.2f}   {acc:8.3f}   {far:11.3f}   {miss:9.3f}")
        if acc > best_acc:
            best_acc, best_conf = acc, float(t)
    print(f"\n  best image-level accuracy {best_acc:.3f} at conf {best_conf:.2f}")

    print(f"\n  written: {OUT / 'threshold_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
