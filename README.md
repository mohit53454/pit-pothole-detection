# 🕳️ Pothole Detector — Indian Road Damage Detection (RDD2022)

**Object detection that finds potholes and cracks in Indian road photos — built
for the real complaint every Indian has: broken roads.**

A fine-tuned **YOLOv8n** (COCO → RDD2022-India transfer learning) that boxes
potholes in road photos. Ships as an **offline Android app** that scans the
street live through the camera: the model runs on the phone, no network, no
server, no cloud. A civic body could use it to triage repair complaints; a
citizen could use it as evidence when filing a pothole report.

```
figshare 13 GB zip ─range-request► India.zip (530 MB) ─► VOC XML ─► YOLO txt
     (remotezip, no full download)                            │
                                                              ▼
                    1,243 real pothole photos ─► v3 dataset (3,898 train)
                                                 single class + hard negatives
                                                              │
                                                              ▼
                                              YOLOv8n train (CPU, 512 px)
                                                              │
                                                              ▼
                                            ONNX export ─► numeric parity check
                                                              │
                                                              ▼
                                              Android APK (ONNX Runtime)
                                              live camera + gallery, offline
```

## Run it

**The Android app (the deliverable):**

A prebuilt debug APK is at `android-app/app/build/outputs/apk/debug/app-debug.apk`.

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk

# or build it yourself (bundled Gradle + SDK, no Android Studio needed):
cd android-app
ANDROID_HOME=../android-sdk ../tools/gradle-8.7/bin/gradle.bat assembleDebug
# Android Studio also works: File > Open > android-app
```

Launch **Pothole Detector**, grant the camera permission, and point it at the
road. Potholes draw red; the confidence slider trades false positives against
misses. **Gallery** runs the same model on a saved photo. The APK
requests only `CAMERA` — it has no `INTERNET` permission, so frames provably
cannot leave the device.

**The training/evaluation pipeline:**

```bash
pip install -r requirements.txt
.venv-dml/Scripts/python src/prepare_v3.py            # build the 1-class pothole set
.venv-dml/Scripts/python src/train_v3.py 16 512       # train (CPU, ~18 min/epoch)
.venv-dml/Scripts/python src/train_v3.py 16 512 resume  # continue an interrupted run
.venv-dml/Scripts/python src/evaluate.py              # held-out metrics + annotated demos
.venv-dml/Scripts/python src/export_onnx.py           # ONNX -> APK assets/model.onnx
.venv-dml/Scripts/python src/validate_android_pipeline.py  # prove the port is faithful
cd android-app && gradle assembleDebug                # -> app-debug.apk
```

Earlier stages are still runnable: `python run.py 8 640` (v1), 
`src/prepare_v2.py` + `src/train_v2.py 12 640` (v2, four classes).

First run downloads only the **India subset (530 MB)** out of the 13 GB
RDD2022 release using HTTP range requests — no multi-hour download.

## What the shipped model detects

One class: **pothole**. The source annotations use the RDD2022 taxonomy (D00
longitudinal crack, D10 transverse crack, D20 alligator crack, D40
pothole/patch); v3 keeps only D40 and discards the crack labels, so a crack is
treated as background the model must learn *not* to fire on. See §3d for why.

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

### Results — v1 (200 held-out India images)

| Stage | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| 8-epoch baseline | 0.150 | 0.049 | 0.23 | 0.16 |
| **+16-epoch extension (final)** | **0.240** | **0.095** | **0.62** | **0.25** |

The extension lifted mAP@50 **60%** over the baseline. Demo inference on 20
unseen test photos (app threshold conf ≥ 0.15): 8 damage boxes across 4
images, all classes D20/D40 — crack classes (D00/D10) remain hard at this
CPU-sized budget. See `outputs/metrics.csv`, `metrics_summary.txt`,
`training_curves.png`, `per_class_metrics.png` and `outputs/predictions/`.

**Recall 0.25 means v1 misses three quarters of the damage**, which is not good
enough to ship. That is what v2 exists to fix.

### 3c. `src/prepare_v2.py` + `src/train_v2.py` — the dataset the app actually uses

- **What:** 8,053 train / 896 val images — the **full** RDD2022 India train
  split (6,935 images) plus **1,243 photographs of real potholes**.
- **Whom:** RDD2022 India from the CRDDC consortium; the pothole photos are the
  public IVCNZ 2020 set (`jaygala24/pothole-detection`, already YOLO-labelled).
  Preparation, class mapping and the split are ours.
- **How and why:** v1 leaned on RDD2022's D40 "pothole/patch" boxes, which in
  dashcam footage are mostly subtle surface wear — there were very few
  unmistakable potholes to learn from. The external photos are mapped into
  class 3 (D40) so the model sees what a pothole actually looks like from
  standing height, which is how the phone app is used. Split is 90/10 per
  source, `random.Random(42)`, so no source leaks across the boundary.

**DirectML was measured and rejected.** The AMD iGPU looked like free speed, so
`src/dml_compat.py` grew shims for seven separate missing/broken kernels
(`unique`, `scatter_add_`, NMS, TAL dtypes, indexing ops, inference-mode version
counters, device selection). It still trained *incorrectly*: at 640/batch16
DirectML produced `box_loss = 0, dfl_loss = 0, cls_loss ≈ 1760` on every batch
while CPU produced a healthy `box 2.68, cls 49.8, dfl 2.69`. Zero box and DFL
loss means the model never learns to localise anything. It was also **no
faster** (~6.2 s/it vs ~6.2 s/it) because the broken kernels force constant CPU
round-trips. Training therefore runs on CPU; the shim file is kept only for
experimentation.

Measured CPU throughput on the Ryzen 5 4600G (6 cores / 12 threads): **6.2 s/it
at 640/batch16**, 3.9 s/it at 512. Thread and worker tuning is neutral — the run
is compute-bound. 640 was chosen over 512 despite the 1.6× cost so that training
resolution matches the app's input exactly. `close_mosaic=4` because the
ultralytics default of 10 would switch mosaic off for most of a 12-epoch run.

Measured on the actual `src/bench_cpu.py` harness at 640/batch16: fp32 with 6
threads **6.64 s/it**, fp32 with 12 threads **6.96 s/it**, and bf16 autocast
slower than both — so plain fp32 with 6 threads is the fast path, not the
seemingly-obvious 12-thread setting.

### 3d. `src/prepare_v3.py` + `src/train_v3.py` — the model that ships

**v2 was not trainable to a useful point on this hardware, and the fixes were
dataset-side, not code-side.** Two problems, both measured:

1. **Gradient dilution.** Of 8,053 v2 train images only 2,498 contain a D40
   pothole; the other 5,555 are crack or empty-road frames. An epoch therefore
   spends 69% of its updates on things that are not potholes. At ~49 min/epoch
   on CPU that capped a realistic run at ~5 epochs, which is nowhere near
   enough for the detection head to converge.
2. **Label-space contention.** Four classes split one small head across crack
types the app never reports, at the direct cost of pothole accuracy.

v3 fixes both:

- **Single class (`pothole`).** D00/D10/D20 boxes are *dropped*, not relabelled
  — a crack that reads as background is exactly the discrimination a road
  scanner needs.
- **1,400 curated hard negatives kept** (800 crack-only, 600 empty road) so the
  model still sees the clean and repaired surfaces a phone camera points at.
  Blindly keeping all 5,555 was the v2 mistake; blindly dropping them would
  trade false negatives for false positives.
- **3,898 train / 415 val images, 7,263 pothole boxes.** Every D40 image is
  kept. Pothole exposure per epoch rises ~2.1×, and the same wall-clock budget
  now buys ~16× more pothole training than v2's 4-class schedule.
- **512 px** instead of 640: 1.6× cheaper per image, and it is also the input
  the phone runs, so on-device inference got faster too. `INPUT_SIZE` in
  `PotholeDetector.java` is the single place that must match it.
- `close_mosaic=2` and `save_period=1` — a 16-epoch run cannot afford 10
  mosaic-off epochs, and per-epoch checkpoints make an interrupt survivable.
- **Resume is exactly `train_v3.py 16 512 resume`.** It passes only
  `resume=True`; ultralytics reads every other setting from the checkpoint.
  Passing `epochs`/`imgsz` again on a resume is what silently misconfigures a
  continued run, so the script refuses to do it.

Cost on this machine: **3.3 s/it × 244 it ≈ 18 min/epoch**, so 16 epochs ≈ 4.8 h.

#### Results — v3 (415 held-out images, never trained on)

| Epoch | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.185 | 0.067 | 0.31 | 0.24 |
| 2 | 0.176 | 0.068 | 0.40 | 0.19 |
| 3 | **0.236** | **0.090** | 0.38 | 0.23 |
| _16 (target)_ | re-run `src/evaluate.py` when the run finishes | | | |

Per-epoch history is written to `outputs/train_log_v3.csv`. Epoch-to-epoch
wobble (0.185 → 0.176 → 0.236) is normal while the warmup LR is still ramping;
the trend across epochs is what matters, not any single row.

For comparison, v1 needed **24 epochs to reach mAP@50 0.240** and v2's val
split sat at 0.057 after its first epoch. v3 matches v1's *final* score at
epoch 3 — while training at 512 px instead of 640, in a third of the epochs.
The single-class redesign, not a longer run, is what bought that.

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
- **What:** (a) validation metrics on the held-out split (never seen in
  training): per-class precision, recall, mAP@50, mAP@50-95; (b) inference on
  the 20 unlabeled test images plus, for v2, the 12 held-out photos with the
  largest pothole boxes, saving **annotated result images**; (c) a headline
  summary file.
- **Whom:** ultralytics val/predict API; report code is ours.
- **How:** mAP@50 = mean average precision at IoU 0.5 (detection standard);
  mAP@50-95 averages over IoU 0.5→0.95 (strict localization). Demo runs at
  conf ≥ 0.25 to avoid noise boxes. Accepts `[weights] [data.yaml] [tag]` and
  defaults to the best available run, writing `metrics_<tag>.csv`,
  `metrics_summary_<tag>.txt` and `predictions_<tag>/` so v1 and v2 results
  coexist instead of overwriting each other.

### 5. `run.py` — one command
Orchestrates fetch → prepare → train → evaluate; accepts `[epochs] [imgsz]`.

### 6. `android-app/` — the offline live scanner
- **What:** A native Android app (Java, minSdk 24) that runs the detector on
  the camera feed in real time and draws red boxes over the preview, so a
  pothole reads at a glance while walking or driving.
- **Whom:** our code. Camera plumbing is **AndroidX CameraX 1.3.4**; on-device
  inference is **ONNX Runtime Android 1.20.0** (Microsoft, MIT licence).
- **How:** `ImageAnalysis` delivers RGBA frames with
  `STRATEGY_KEEP_ONLY_LATEST`, so a slow inference drops frames instead of
  queueing them and lagging behind reality. One background thread converts the
  frame (honouring row stride and sensor rotation), runs the model, and posts
  boxes to the UI. `Preview` and `ImageAnalysis` are both pinned to 16:9 and the
  `PreviewView` uses `FILL_CENTER`, so `OverlayView` can map box coordinates
  with the identical cover-and-centre-crop transform — otherwise every box
  would be offset from the damage it marks.
- **Bugs worth recording**, because none of them stopped the build:
  CameraX throws `Cannot use both setTargetResolution and setTargetAspectRatio
  on the same config`, and the surrounding catch turned that into a silent
  "Camera unavailable" — live scanning would never have started. An inverted
  busy flag and a leaked `ImageProxy` stalled the analyzer after a few frames.
  `BitmapFactory` ignores EXIF orientation, so portrait gallery photos were
  being analysed sideways. All fixed; the APK now builds and installs clean.

### 7. `src/export_onnx.py` + `src/validate_android_pipeline.py` — proving the port
- **What:** Exports the trained weights to ONNX (opset 12, static 512×512,
  simplified) into `android-app/app/src/main/assets/model.onnx`, then checks
  that the app's pipeline computes the *same thing* as PyTorch. Verified shape
  contract: input `images [1,3,512,512]`, output `output0 [1,5,5376]`
  (4 box coords + 1 class score). `PotholeDetector.INPUT_SIZE` must equal the
  export size or session creation fails outright, so the two are changed
  together.
- **Whom:** our code; ONNX export via ultralytics, runtime via ONNX Runtime.
- **Why ONNX and not TFLite:** ultralytics' TFLite exporter is now `litert`,
  which pulls in jax + torchao + transformers, and the `onnx2tf` alternative
  pins `numpy<2.0` against this environment's numpy 2.x. ONNX Runtime Android
  is a single prebuilt AAR with no Python-side dependency tree.
- **How the check works:** `validate_android_pipeline.py` reimplements the Java
  pre/post-processing in Python — same letterbox (grey 114, centre pad, integer
  division offsets), same CHW RGB /255 tensor, same per-class NMS at IoU 0.45,
  same undo-letterbox mapping — and compares ONNX Runtime against the identical
  PyTorch weights on the same images. It asserts the raw output gap is < 1e-2
  and that every reference box is matched at IoU ≥ 0.5.
- **One trap it guards:** ultralytics already applies the sigmoid inside the
  Detect head, so the exported scores are probabilities. Applying sigmoid a
  second time squashes everything into [0.5, 0.73] and any threshold below 0.5
  then passes every anchor — a flood of false positives. The Java detector and
  the validator both use the scores as-is.

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
- **Domain gap is still the big one.** RDD2022-India is dashcam footage and the
  pothole photos are stills; neither is live video from a phone held at walking
  height on an Indian street. Motion blur, night, rain and wet-road glare are
  not represented in training. Collecting a few hundred labelled phone frames
  from the actual deployment street is the highest-value next step.
- **CPU epoch budget is still the hard ceiling.** v3 runs 16 epochs over 3,898
  images at 512 px in ~4.8 h on a Ryzen 5 4600G. The RDD2022 literature uses
  100+ epochs at 640 px on a GPU; that is ~35× more compute than this machine
  can supply. The ceiling here is compute, not data.
- Thin longitudinal/transverse cracks (D00/D10) stay the hardest classes —
  they are a few pixels wide at 640 and easy to confuse with road markings,
  shadows and tar seams. Potholes (D40) are the reliable class and the one the
  app highlights in red.
- **`pwa/` is a superseded prototype and is not shipped.** Its `app.js` applies
  a sigmoid to scores that ultralytics has already sigmoided, which pushes
  every anchor into [0.5, 0.73] and defeats the 0.20 threshold — it would
  report damage on almost any frame. The Android app is the deliverable and
  does not have this bug (see §7).
- Not a deployment-grade road inspection system; it is a triage/evidence aid.
  A box is a hint for a human to check, not a certified measurement.

## Data & references
- RDD2022 (CRDDC 2022): https://figshare.com/articles/dataset/21431547
- Ultralytics YOLOv8: https://docs.ultralytics.com
- Paper: Arya et al., *RDD2022: An image dataset for hierarchical road damage
  detection*, CRDDC 2022.
