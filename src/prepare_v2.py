"""Prepare the v2 dataset: full RDD2022 India + external real-pothole images.

Why v2: the v1 subset (1,200 imgs) leaned on RDD2022 D40 "pothole/patch"
boxes that are mostly subtle dashcam wear — no clearly visible potholes.
v2 adds 1,243 web-sourced photos of *actual potholes* (jaygala24/pothole-
detection, IVCNZ 2020, YOLO format) mapped into class 3 (D40 pothole).

Sources
  A) data/raw/train  : 7,707 RDD2022 India images + VOC xml (already extracted)
  B) data/external   : Pothole.Dataset.IVCNZ.zip -> 1,243 img/txt YOLO pairs

Output: data/yolo2/ (images/{train,val}, labels/{train,val}, data.yaml)
Split: deterministic 90/10 per source, seed 42.
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXTERNAL_ZIP = ROOT / "data" / "external" / "Pothole.Dataset.IVCNZ.zip"
EXTERNAL_DIR = ROOT / "data" / "external" / "pothole"
YOLO_DIR = ROOT / "data" / "yolo2"

CLASSES = ["D00", "D10", "D20", "D40"]
SEED = 42
POThOLE_CLASS = 3          # external single-class potholes -> D40
EXTERNAL_TRAIN_FRAC = 0.9


def extract_external() -> None:
    if EXTERNAL_DIR.exists() and any(EXTERNAL_DIR.glob("*.jpg")):
        print("external pothole images already extracted")
        return
    print("extracting external pothole dataset...")
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EXTERNAL_ZIP) as z:
        for n in z.namelist():
            if n.endswith((".jpg", ".txt")):
                with z.open(n) as src, open(EXTERNAL_DIR / Path(n).name, "wb") as out:
                    shutil.copyfileobj(src, out)
    print(f"extracted {len(list(EXTERNAL_DIR.glob('*.jpg')))} images")


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
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


def place(src: Path, lines: list[str], split: str) -> None:
    shutil.copy2(src, YOLO_DIR / "images" / split / src.name)
    (YOLO_DIR / "labels" / split / (src.stem + ".txt")).write_text(
        "\n".join(lines) + ("\n" if lines else "")
    )


def build() -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    # ---- Source A: full RDD2022 India train split ----
    images = sorted((RAW / "train" / "images").glob("*.jpg"))
    xmls = {p.stem: p for p in (RAW / "train" / "annotations" / "xmls").glob("*.xml")}
    pairs = [(img, xmls[img.stem]) for img in images if img.stem in xmls]
    print(f"RDD2022 matched pairs: {len(pairs)}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * 0.9)
    for split, subset in (("train", pairs[:n_train]), ("val", pairs[n_train:])):
        for img, xml in subset:
            size = ET.parse(xml).getroot().find("size")
            w = int(size.findtext("width")); h = int(size.findtext("height"))
            place(img, voc_to_yolo(xml, w, h), split)
        print(f"RDD2022 {split}: {len(subset)}")

    # ---- Source B: external real potholes (single class 0 -> D40 = 3) ----
    ext_imgs = sorted(EXTERNAL_DIR.glob("*.jpg"))
    print(f"external pothole images: {len(ext_imgs)}")
    rng2 = random.Random(SEED)
    rng2.shuffle(ext_imgs)
    n_ext_train = int(len(ext_imgs) * EXTERNAL_TRAIN_FRAC)
    for split, subset in (("train", ext_imgs[:n_ext_train]),
                          ("val", ext_imgs[n_ext_train:])):
        for img in subset:
            txt = img.with_suffix(".txt")
            lines: list[str] = []
            if txt.exists():
                for row in txt.read_text().splitlines():
                    parts = row.split()
                    if len(parts) == 5:
                        # class 0 (pothole) -> class 3 (D40)
                        lines.append(f"{POThOLE_CLASS} " + " ".join(parts[1:]))
            if not lines:
                continue
            place(img, lines, split)
        print(f"external {split}: {len(subset)}")

    # ---- data.yaml ----
    yaml_text = (
        "path: " + str(YOLO_DIR).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (YOLO_DIR / "data.yaml").write_text(yaml_text)

    # ---- per-class box counts (sanity) ----
    counts = {c: 0 for c in CLASSES}
    for lbl in (YOLO_DIR / "labels").rglob("*.txt"):
        for row in lbl.read_text().splitlines():
            parts = row.split()
            if parts:
                counts[CLASSES[int(parts[0])]] += 1
    print("box counts:", counts)
    print(f"data.yaml -> {YOLO_DIR / 'data.yaml'}")


if __name__ == "__main__":
    extract_external()
    build()
    print("v2 dataset ready")
