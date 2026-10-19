"""Publish figures: training curves + per-class detection metrics.

Reads outputs/train_log.csv and outputs/metrics.csv, writes:
  outputs/training_curves.png   (loss + mAP per epoch)
  outputs/per_class_metrics.png (precision/recall/mAP50 per damage class)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> int:
    # Prefer the extended run's log once it exists
    log_path = OUT / "train_log_ext.csv"
    if not log_path.exists():
        log_path = OUT / "train_log.csv"
    log = pd.read_csv(log_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(log["epoch"], log["train_box_loss"], label="train box loss")
    ax1.plot(log["epoch"], log["val_box_loss"], label="val box loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")

    ax2.plot(log["epoch"], log["map50"], label="mAP@50")
    ax2.plot(log["epoch"], log["map50_95"], label="mAP@50-95")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("mAP"); ax2.legend(); ax2.set_title("Accuracy")
    fig.suptitle(f"YOLOv8n fine-tune on RDD2022-India ({log_path.stem})")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=130)
    plt.close(fig)

    pc = pd.read_csv(OUT / "metrics.csv")
    x = range(len(pc))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.25 for i in x], pc["precision"], width=0.25, label="precision")
    ax.bar([i for i in x], pc["recall"], width=0.25, label="recall")
    ax.bar([i + 0.25 for i in x], pc["map50"], width=0.25, label="mAP@50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pc["class"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Per-class detection metrics (200 held-out images)")
    fig.tight_layout()
    fig.savefig(OUT / "per_class_metrics.png", dpi=130)
    plt.close(fig)

    print("wrote training_curves.png + per_class_metrics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
