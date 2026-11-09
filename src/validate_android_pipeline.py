"""Validate the exact inference pipeline the Android app uses.

The app (PotholeDetector.java) does its own letterbox + CHW packing + output
parsing + NMS. A silent mismatch there would produce an APK that runs but
draws wrong boxes, so this reproduces that pipeline byte-for-byte in Python
and checks it against two references:

  A) the same PyTorch weights fed the identical square-letterboxed tensor
     -> isolates ONNX export correctness (expected: near-exact agreement)
  B) ultralytics' own .predict() -> sanity check that detections are real

Usage:
    .venv-dml/Scripts/python src/validate_android_pipeline.py [weights.pt] [n]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Must track PotholeDetector.java: INPUT_SIZE=512, LABELS has one entry.
SIZE = 512
PAD = 114
IOU_THRESH = 0.45
LABELS = ["pothole"]


def letterbox(img: np.ndarray):
    """Mirror PotholeDetector.detect(): centre pad to 640x640 with grey 114."""
    sh, sw = img.shape[:2]
    scale = min(SIZE / sw, SIZE / sh)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    dx = (SIZE - nw) // 2          # Java int division
    dy = (SIZE - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), PAD, dtype=np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def to_chw(canvas: np.ndarray) -> np.ndarray:
    """RGB, /255, planar CHW - same order the Java FloatBuffer is filled in."""
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def parse_output(out: np.ndarray, scale: float, dx: int, dy: int,
                 sw: int, sh: int, conf: float):
    """out is [1,5,5376]; scores are ALREADY sigmoid (ultralytics Detect head)."""
    o = out[0]
    channels, anchors = o.shape
    if channels > anchors:                 # channels-last variant
        o = o.T
        channels, anchors = o.shape
    num_classes = min(len(LABELS), channels - 4)

    boxes_xywh = o[:4, :].T                # (anchors, 4) cx,cy,w,h in 640-space
    cls_scores = o[4:4 + num_classes, :]   # (num_classes, anchors)
    best_c = cls_scores.argmax(axis=0)
    best_s = cls_scores.max(axis=0)
    keep = best_s >= conf

    dets = []
    for a in np.nonzero(keep)[0]:
        cx, cy, w, h = boxes_xywh[a]
        x1 = np.clip((cx - w / 2 - dx) / scale, 0, sw)
        y1 = np.clip((cy - h / 2 - dy) / scale, 0, sh)
        x2 = np.clip((cx + w / 2 - dx) / scale, 0, sw)
        y2 = np.clip((cy + h / 2 - dy) / scale, 0, sh)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        dets.append((float(best_s[a]), int(best_c[a]),
                     np.array([x1, y1, x2, y2], dtype=np.float32)))
    return nms(dets)


def nms(dets):
    dets = sorted(dets, key=lambda d: -d[0])
    suppressed = [False] * len(dets)
    out = []
    for i, cur in enumerate(dets):
        if suppressed[i]:
            continue
        out.append(cur)
        for j in range(i + 1, len(dets)):
            if suppressed[j] or dets[j][1] != cur[1]:
                continue
            if iou(cur[2], dets[j][2]) > IOU_THRESH:
                suppressed[j] = True
    return out


def iou(a, b):
    l, r = max(a[0], b[0]), min(a[2], b[2])
    t, bo = max(a[1], b[1]), min(a[3], b[3])
    if r <= l or bo <= t:
        return 0.0
    inter = (r - l) * (bo - t)
    uni = ((a[2] - a[0]) * (a[3] - a[1])
           + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / uni) if uni > 0 else 0.0


def match(ref, test):
    """Greedy same-class IoU matching; returns (matched, mean IoU, mean |dscore|)."""
    used = set()
    ious, ds = [], []
    for s, c, box in ref:
        best, best_i = 0.0, -1
        for k, (s2, c2, box2) in enumerate(test):
            if k in used or c2 != c:
                continue
            v = iou(box, box2)
            if v > best:
                best, best_i = v, k
        if best_i >= 0 and best >= 0.5:
            used.add(best_i)
            ious.append(best)
            ds.append(abs(s - test[best_i][0]))
    return len(ious), (float(np.mean(ious)) if ious else 0.0), (float(np.mean(ds)) if ds else 0.0)


def main() -> int:
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if weights is None:
        for cand in (ROOT / "runs/pothole/train_v4/weights/best.pt",
                     ROOT / "runs/pothole/train_v3/weights/best.pt",
                     ROOT / "runs/pothole/train_v2/weights/best.pt",
                     ROOT / "runs/pothole/train_v3_ext/weights/best.pt",
                     ROOT / "runs/pothole/train_ext/weights/best.pt",
                     ROOT / "runs/pothole/train/weights/best.pt"):
            if cand.exists():
                weights = cand
                break
    if weights is None or not weights.exists():
        print("no weights found")
        return 1

    onnx_path = weights.with_suffix(".onnx")
    if not onnx_path.exists():
        print(f"missing {onnx_path} - export it first")
        return 1

    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    print(f"weights : {weights}")
    print(f"onnx    : {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    yolo = YOLO(str(weights))
    torch_model = yolo.model.eval().float()

    # Mix of held-out RDD2022 val frames and the real-pothole external images.
    pool = sorted((ROOT / "data/yolo3/images/val").glob("*.jpg"))
    if not pool:
        pool = sorted((ROOT / "data/demo_images").glob("*.jpg"))
    step = max(1, len(pool) // n)
    images = pool[::step][:n]
    if not images:
        print("no validation images found")
        return 1

    conf = 0.25
    tot_ref = tot_test = tot_matched = 0
    all_ious, all_ds = [], []
    max_onnx_pt_gap = 0.0
    print(f"\n{len(images)} images, conf>={conf}\n")
    print(f"{'image':28s} {'PT':>3s} {'ONNX':>5s} {'match':>6s} {'IoU':>6s} {'dScore':>7s} {'maxGap':>9s}")

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        sh, sw = img.shape[:2]
        canvas, scale, dx, dy = letterbox(img)
        tensor = to_chw(canvas)

        onnx_out = sess.run(None, {in_name: tensor})[0]
        with torch.no_grad():
            pt_raw = torch_model(torch.from_numpy(tensor))
        pt_out = (pt_raw[0] if isinstance(pt_raw, (list, tuple)) else pt_raw).numpy()
        max_onnx_pt_gap = max(max_onnx_pt_gap,
                              float(np.abs(onnx_out - pt_out).max()))

        ref = parse_output(pt_out, scale, dx, dy, sw, sh, conf)
        test = parse_output(onnx_out, scale, dx, dy, sw, sh, conf)
        m, mi, md = match(ref, test)
        tot_ref += len(ref)
        tot_test += len(test)
        tot_matched += m
        if m:
            all_ious.append(mi)
            all_ds.append(md)
        print(f"{p.name[:28]:28s} {len(ref):3d} {len(test):5d} {m:6d} "
              f"{mi:6.3f} {md:7.4f} {float(np.abs(onnx_out - pt_out).max()):9.2e}")

    print("\n--- ONNX vs PyTorch (identical input tensor) ---")
    print(f"max abs output gap : {max_onnx_pt_gap:.3e}")
    print(f"boxes PyTorch      : {tot_ref}")
    print(f"boxes ONNX/app     : {tot_test}")
    print(f"matched (IoU>=0.5) : {tot_matched}")
    print(f"mean matched IoU   : {np.mean(all_ious) if all_ious else 0:.4f}")
    print(f"mean score delta   : {np.mean(all_ds) if all_ds else 0:.5f}")

    # Reference B: ultralytics' own predict (different letterbox strategy).
    print("\n--- cross-check vs ultralytics .predict() ---")
    agree = 0
    for p in images[:6]:
        img = cv2.imread(str(p))
        sh, sw = img.shape[:2]
        canvas, scale, dx, dy = letterbox(img)
        app_dets = parse_output(sess.run(None, {in_name: to_chw(canvas)})[0],
                                scale, dx, dy, sw, sh, conf)
        r = yolo.predict(img, imgsz=SIZE, conf=conf, iou=IOU_THRESH, verbose=False)[0]
        ultra = [(float(b.conf), int(b.cls), b.xyxy[0].numpy()) for b in r.boxes]
        m, mi, _ = match(ultra, app_dets)
        agree += m
        print(f"  {p.name[:28]:28s} ultralytics={len(ultra):2d} app={len(app_dets):2d} matched={m:2d} IoU={mi:.3f}")

    ok = max_onnx_pt_gap < 1e-2 and tot_matched == tot_ref and tot_test == tot_ref
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'} "
          f"(export faithful={max_onnx_pt_gap < 1e-2}, "
          f"postprocess identical={tot_matched == tot_ref and tot_test == tot_ref})")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())