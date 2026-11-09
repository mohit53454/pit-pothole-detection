"""Resume the interrupted v4 training run from weights/last.pt.

Ultralytics resume restores model, EMA, optimizer state, LR schedule and epoch
counter from the checkpoint, so the run continues at epoch 12 rather than
restarting the schedule.
"""
import os
import sys

from ultralytics import YOLO


def main() -> int:
    ckpt = os.path.join("runs", "pothole", "train_v4", "weights", "last.pt")
    if not os.path.exists(ckpt):
        print("Missing checkpoint: %s" % ckpt)
        return 1

    print("Resuming from %s" % ckpt, flush=True)
    model = YOLO(ckpt)
    model.train(resume=True)
    print("RESUME_FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
