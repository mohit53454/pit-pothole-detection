"""Build the v4 dataset: v3 road-damage frames + real-world pothole photos.

v3 was trained only on RDD2022 road-damage frames, which is why the phone misses
real potholes: the external set (1,243 phone-style photos, 4,076 boxes) is the
same class the app actually sees but was never in training. v4 unions the two.

    .venv-dml/Scripts/python src/prepare_v4.py

Writes data/v4/{images,labels}/{train,val} plus data/v4/data.yaml.

External images are copied as ext_<name> so they cannot collide with the
RDD2022 filenames already in data/yolo3. The val split keeps every v3 val frame
(so v3 and v4 numbers stay comparable) and adds a deterministic 20% holdout of
the external set, so the val metric finally reflects real potholes too.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Y3 = ROOT / "data" / "yolo3"
EXT = ROOT / "data" / "external" / "pothole"
OUT = ROOT / "data" / "v4"

SEED = 42
EXT_VAL_FRACTION = 0.20
PREFIX = "ext_"


def stage(src_img: Path, src_lbl: Path, split: str, name: str) -> int:
    """Copy one image+label pair into the v4 split. Returns boxes copied."""
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    shutil.copyfile(src_img, img_dir / name)
    boxes = 0
    if src_lbl.exists():
        text = src_lbl.read_text(encoding="utf-8", errors="replace").strip()
        boxes = sum(1 for line in text.splitlines() if line.strip())
        (lbl_dir / (Path(name).stem + ".txt")).write_text(
            text + ("\n" if text else ""), encoding="utf-8"
        )
    else:
        # Negative sample: an empty label file teaches "no pothole here".
        (lbl_dir / (Path(name).stem + ".txt")).write_text("", encoding="utf-8")
    return boxes


def main() -> int:
    if not (Y3 / "images" / "train").is_dir():
        print(f"missing v3 dataset at {Y3}")
        return 1
    if not EXT.is_dir():
        print(f"missing external pothole set at {EXT}")
        return 1

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": [0, 0], "val": [0, 0]}  # [images, boxes]

    # 1. v3 frames, splits preserved verbatim.
    for split in ("train", "val"):
        for img in sorted((Y3 / "images" / split).iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = Y3 / "labels" / split / (img.stem + ".txt")
            stats[split][1] += stage(img, lbl, split, img.name)
            stats[split][0] += 1
    print(f"v3  train={stats['train'][0]} val={stats['val'][0]}")

    # 2. External pothole photos, deterministic holdout for val.
    ext_imgs = sorted(p for p in EXT.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(SEED)
    idx = list(range(len(ext_imgs)))
    rng.shuffle(idx)
    n_val = int(round(len(ext_imgs) * EXT_VAL_FRACTION))
    val_set = set(idx[:n_val])

    ext_counts = {"train": [0, 0], "val": [0, 0]}
    for i, img in enumerate(ext_imgs):
        split = "val" if i in val_set else "train"
        name = PREFIX + img.name
        ext_counts[split][1] += stage(img, img.with_suffix(".txt"), split, name)
        ext_counts[split][0] += 1
    stats["train"][0] += ext_counts["train"][0]
    stats["train"][1] += ext_counts["train"][1]
    stats["val"][0] += ext_counts["val"][0]
    stats["val"][1] += ext_counts["val"][1]
    print(f"ext train={ext_counts['train'][0]} val={ext_counts['val'][0]}")

    (OUT / "data.yaml").write_text(
        "\n".join([
            f"path: {OUT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: pothole",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"\nv4  train={stats['train'][0]} images / {stats['train'][1]} boxes")
    print(f"v4  val  ={stats['val'][0]} images / {stats['val'][1]} boxes")
    print(f"config -> {OUT / 'data.yaml'}")
    print("\nnext: .venv-dml/Scripts/python src/train_v3.py 20 512 start 6 "
          f"{OUT / 'data.yaml'} train_v4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
