"""End-to-end: fetch -> prepare -> train -> evaluate.

Usage: python run.py [epochs] [imgsz]   (defaults: 8, 640)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src import evaluate, fetch_data, prepare_dataset, train  # noqa: E402


def main() -> int:
    for step in (fetch_data.main, prepare_dataset.extract_train, prepare_dataset.build):
        rc = step()
        if rc:
            return rc
    sys.argv = [
        "train",
        sys.argv[1] if len(sys.argv) > 1 else "8",
        sys.argv[2] if len(sys.argv) > 2 else "640",
    ]
    if train.main() != 0:
        return 1
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main())
