"""Prepare RDD2022 India subset for YOLO training.

Source: RDD2022.zip (figshare id 21431547) -> nested RDD2022/India.zip, already
downloaded to data/RDD2022/India.zip by fetch_data.py.

Steps:
1. Extract India/train images + Pascal-VOC xml labels (7,707 images).
2. Convert VOC -> YOLO txt (class_id cx cy w h, normalized).
   RDD2022 damage classes: D00 longitudinal, D10 transverse,
   D20 alligator, D40 patch/pothole.
3. Deterministic 90/10 train/val split (seed 42), capped at MAX_TRAIN/MAX_VAL
   for CPU-feasible training.
4. Write ultralytics data.yaml and a few unlabeled test images for demos.
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "RDD2022" / "India.zip"
RAW = ROOT / "data" / "raw"
YOLO_DIR = ROOT / "data" / "yolo"
DEMO_DIR = ROOT / "data" / "demo_images"

CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_DESC = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole/patch",
}
MAX_TRAIN = 1200   # CPU-friendly subset
MAX_VAL = 200
MAX_DEMO = 20
SEED = 42


def extract_train() -> None:
    if (RAW / "train" / "images").exists():
        print("raw train images already extracted")
        return
    print("extracting train images + xml labels (one-time, ~450 MB)...")
    (RAW / "train").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        members = [
            n for n in z.namelist()
            if n.startswith("India/train/") and not n.endswith("/")
        ]
        for n in members:
            dest = RAW / n.removeprefix("India/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"extracted {len(members)} files")


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """One line per damage box: class cx cy w h (all normalized 0-1)."""
    lines: list[str] = []
    root = ET.parse(xml_path).getroot()
    for obj in root.iter("object"):
        name = obj.findtext("name")
        if name not in CLASSES:
            continue
        b = obj.find("bndbox")
        xmin = float(b.findtext("xmin")); xmax = float(b.findtext("xmax"))
        ymin = float(b.findtext("ymin")); ymax = float(b.findtext("ymax"))
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def build() -> None:
    images = sorted((RAW / "train" / "images").glob("*.jpg"))
    xmls = {p.stem: p for p in (RAW / "train" / "annotations" / "xmls").glob("*.xml")}
    pairs = [(img, xmls[img.stem]) for img in images if img.stem in xmls]
    print(f"matched image/xml pairs: {len(pairs)}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    train_pairs = pairs[:MAX_TRAIN]
    val_pairs = pairs[MAX_TRAIN:MAX_TRAIN + MAX_VAL]

    for split, subset in (("train", train_pairs), ("val", val_pairs)):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, xml in subset:
            size = ET.parse(xml).getroot().find("size")
            w = int(size.findtext("width")); h = int(size.findtext("height"))
            lines = voc_to_yolo(xml, w, h)
            shutil.copy2(img, YOLO_DIR / "images" / split / img.name)
            (YOLO_DIR / "labels" / split / (img.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        print(f"{split}: {len(subset)} images")

    yaml_text = (
        "path: " + str(YOLO_DIR).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (YOLO_DIR / "data.yaml").write_text(yaml_text)

    # unlabeled test images for the inference demo
    with zipfile.ZipFile(ZIP) as z:
        demos = [n for n in z.namelist()
                 if n.startswith("India/test/images/") and n.endswith(".jpg")]
        random.Random(SEED).shuffle(demos)
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        for n in demos[:MAX_DEMO]:
            with z.open(n) as src, open(DEMO_DIR / Path(n).name, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"demo images: {MAX_DEMO} -> {DEMO_DIR}")


if __name__ == "__main__":
    extract_train()
    build()
    print("dataset ready")
