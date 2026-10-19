# 🕳️ Pothole Detector — Indian Road Damage Detection (RDD2022)

**Object detection that finds potholes and cracks in Indian road photos — built
for the real complaint every Indian has: broken roads.**

A fine-tuned **YOLOv8n** (COCO → RDD2022-India transfer learning) that draws
boxes around four road-damage types on unseen street photos. A civic body could
use it to triage repair complaints; a citizen could use it as evidence when
filing a pothole report.

```
figshare 13 GB zip ─range-request► India.zip (530 MB) ─► VOC XML ─► YOLO txt
     (remotezip, no full download)                            │
                                                              ▼
                              annotated demo images ◄── YOLOv8n train ◄─ 1,200 imgs
                              + validation metrics                  (CPU, ~50 min)
```

## Run it

```bash
pip install -r requirements.txt
python run.py 8 640      # fetch -> prepare -> train (8 epochs) -> evaluate
```

First run downloads only the **India subset (530 MB)** out of the 13 GB
RDD2022 release using HTTP range requests — no multi-hour download.

## Damage classes (RDD2022 taxonomy)

| Code | Meaning |
|---|---|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator (fatigue) crack |
| D40 | Pothole / patch |

---

## Section-by-section: what is done, by whom, how

### 1. `src/fetch_data.py` — getting 530 MB out of 13 GB
- **What:** Downloads `RDD2022/India.zip` only.
- **Whom:** Data collected by the **CRDDC 2022 consortium** (Japan, India,
  Norway, Czech Republic, China, US road agencies); hosted openly on figshare
  (article 21431547). Fetch code is ours.
- **How:** The figshare release is one zip of per-country zips. `remotezip`
  reads the zip's central directory over HTTP **range requests** and pulls only
  the India member — ~530 MB instead of 13 GB.

### 2. `src/prepare_dataset.py` — VOC → YOLO conversion
- **What:** 7,707 labeled Indian road images (train split of RDD2022) +
  Pascal-VOC XML → ultralytics YOLO format; 90/10 deterministic split,
  capped at 1,200 train / 200 val for CPU-feasible training; 20 *unlabeled*
  test images kept aside as a genuine unseen demo.
- **Whom:** Entirely our code (`xml.etree`, `zipfile`, `random.Random(42)`).
- **How:** each `<object><bndbox>` becomes `class cx cy w h` normalized to 0-1;
  damaged images with zero valid boxes are dropped by the matched-pair filter.

### 3. `src/train.py` — the model
- **What:** Fine-tunes **YOLOv8n** (nano — 3.2 M parameters) on the subset.
- **Whom:** Architecture and pretrained weights by **Ultralytics** (COCO-
  pretrained); the fine-tuning loop and configuration are ours.
- **How:** **Transfer learning** — COCO features (textures, edges) transfer to
  crack detection, so 1,200 images suffice where from-scratch would need
  10-100×. Two-stage schedule at 640 px, batch 16, seed 42, CPU: an 8-epoch
  baseline from COCO weights, then a 16-epoch warm-start continuation from the
  baseline's best checkpoint (`src/train.py <epochs> <imgsz> <weights>`).
  Per-epoch metrics land in `outputs/train_log.csv` / `train_log_ext.csv`.

### Results (final, 200 held-out India images)

| Stage | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| 8-epoch baseline | 0.150 | 0.049 | 0.23 | 0.16 |
| **+16-epoch extension (final)** | **0.240** | **0.095** | **0.62** | **0.25** |

The extension lifted mAP@50 **60%** over the baseline. Demo inference on 20
unseen test photos (app threshold conf ≥ 0.15): 8 damage boxes across 4
images, all classes D20/D40 — crack classes (D00/D10) remain hard at this
CPU-sized budget. See `outputs/metrics.csv`, `metrics_summary.txt`,
`training_curves.png`, `per_class_metrics.png` and `outputs/predictions/`.

### 3b. `app.py` + `static/index.html` — the phone app
- **What:** A mobile web app — snap a road photo, get boxes with damage class
  and confidence. Run `python app.py`, open the printed URL on a phone on the
  same Wi-Fi (`http://<PC-IP>:5000`).
- **Whom:** our code (Flask + vanilla JS canvas rendering); model inference
  stays server-side so the phone needs zero setup.
- **How:** the browser POSTs the photo to `/predict`; the server runs YOLO at
  conf ≥ 0.15 and returns `[{cls, label, conf, xyxy}]`; the canvas draws
  class-colored boxes. Opens with a guaranteed-detectable sample so the demo
  works the second it loads. The app auto-uses the best available weights
  (`train_ext` first, then baseline).

### 4. `src/evaluate.py` — honest scoring + the demo
- **What:** (a) validation metrics on the **held-out 200 images** (never seen
  in training): per-class precision, recall, mAP@50, mAP@50-95;
  (b) inference on the 20 unlabeled test images, saving **annotated result
  images** to `outputs/predictions/`; (c) a headline summary file.
- **Whom:** ultralytics val/predict API; report code is ours.
- **How:** mAP@50 = mean average precision at IoU 0.5 (detection standard);
  mAP@50-95 averages over IoU 0.5→0.95 (strict localization). Demo runs at
  conf ≥ 0.25 to avoid noise boxes.

### 5. `run.py` — one command
Orchestrates fetch → prepare → train → evaluate; accepts `[epochs] [imgsz]`.

---

## Why this is *hard* (not a Kaggle toy)
- **Object detection, not classification** — the model must localize damage in
  cluttered street scenes with wild scale/lighting variation.
- **Domain shift** — RDD2022 spans 6 countries; Indian roads (dust, patchy
  repairs, motorbikes in frame) look different, which is why we train on the
  India subset rather than the pooled data.
- **Class imbalance** — D40 potholes/patches dominate; crack classes are
  rarer, so per-class metrics matter more than overall accuracy.

## Limitations
- Trained on 1,200 of 7,707 available images (CPU budget) — the full dataset
  with a GPU would lift mAP substantially; `MAX_TRAIN` in
  `prepare_dataset.py` is the single knob.
- RDD2022-India images are dashcam-style; phone photos from standing height
  will shift performance (domain gap) — retrain on a few hundred phone photos
  to adapt.
- Not a deployment-grade road inspection system; it is a triage/evidence aid.

## Data & references
- RDD2022 (CRDDC 2022): https://figshare.com/articles/dataset/21431547
- Ultralytics YOLOv8: https://docs.ultralytics.com
- Paper: Arya et al., *RDD2022: An image dataset for hierarchical road damage
  detection*, CRDDC 2022.
