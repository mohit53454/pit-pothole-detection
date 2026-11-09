"""Build the v3 dataset: single-class POTHOLE detector, sized for a 4-hour CPU run.

Why v3 replaces v2
  v2 has 8,053 train images but only 2,498 of them contain a D40 pothole, so a
  v2 epoch spends 69% of its gradient on crack/background images. On this
  machine an epoch costs ~49 min at 640px, which caps a practical run at ~5
  epochs - nowhere near enough for the head to learn potholes well.

  v3 keeps every D40 image, keeps a deliberately hard negative set (cracks and
  clean road - what a phone camera actually sees), and collapses the label
  space to one class. That roughly doubles pothole gradient density per epoch
  and lets the run use 512px, so the same wall-clock budget buys an order of
  magnitude more pothole training.

Label mapping
  D00/D10/D20 boxes are DROPPED from the kept images: a crack re-labelled as
  "not a pothole" is exactly the discrimination the app needs.

Inputs   data/yolo2/{images,labels}/{train,val}
Output   data/yolo3/{images,labels}/{train,val}, data.yaml (1 class: pothole)

Split sizes are deterministic (seed 42) so reruns are reproducible.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "yolo2"
DST = ROOT / "data" / "yolo3"

CLASSES = ["pothole"]
SEED = 42

# Kept negatives, per split: cracks are genuinely hard negatives (a worn patch
# must not fire as a pothole); blanks teach "clean road = nothing".
NEG_SPLIT = {"train": (800, 600), "val": (80, 60)}  # (crack-only, empty)


def read_labels(p: Path) -> list[list[str]]:
    return [r.split() for r in p.read_text().splitlines() if r.strip()]


def write_out(img: Path, rows: list[list[str]], split: str) -> None:
    """Hardlink the image when possible (same volume) and emit the label file."""
    dst_img = DST / "images" / split / img.name
    if not dst_img.exists():
        try:
            os.link(img, dst_img)
        except OSError:  # different volume / unsupported fs
            shutil.copy2(img, dst_img)
    lines = [" ".join(r) for r in rows]
    (DST / "labels" / split / (img.stem + ".txt")).write_text(
        "\n".join(lines) + ("\n" if lines else "")
    )


def build_split(split: str, rng: random.Random) -> None:
    img_dir = SRC / "images" / split
    lbl_dir = SRC / "labels" / split

    pos: list[Path] = []
    crack: list[Path] = []
    empty: list[Path] = []
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / (img.stem + ".txt")
        rows = read_labels(lbl) if lbl.exists() else []
        cls = {int(r[0]) for r in rows}
        if 3 in cls:
            pos.append(img)
        elif not rows:
            empty.append(img)
        else:
            crack.append(img)

    n_crack, n_empty = NEG_SPLIT[split]
    rng.shuffle(crack)
    rng.shuffle(empty)
    keep_crack = crack[:n_crack]
    keep_empty = empty[:n_empty]

    for img in pos:
        rows = read_labels(lbl_dir / (img.stem + ".txt"))
        # class 3 -> class 0 (the only class); drop everything else
        keep = [["0", *r[1:]] for r in rows if r[0] == "3" and len(r) == 5]
        if keep:
            write_out(img, keep, split)
    for img in keep_crack + keep_empty:
        write_out(img, [], split)

    total = len(pos) + len(keep_crack) + len(keep_empty)
    print(f"{split}: pothole={len(pos)} crack_neg={len(keep_crack)} "
          f"empty_neg={len(keep_empty)} -> {total} images")


def main() -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DST / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    for split in ("train", "val"):
        build_split(split, rng)

    yaml_text = (
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: pothole\n"
    )
    (DST / "data.yaml").write_text(yaml_text)

    n_boxes = sum(
        len(read_labels(p)) for p in (DST / "labels").rglob("*.txt")
    )
    print(f"pothole boxes: {n_boxes}")
    print(f"data.yaml -> {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
