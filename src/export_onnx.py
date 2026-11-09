"""Export the best trained weights to ONNX and deploy them into the Android app.

The app runs ONNX Runtime on-device (see PotholeDetector.java). Windows cannot
use ultralytics' LiteRT/TFLite exporter - litert-torch drags in jax + torchao +
transformers, and the onnx2tf route pins numpy<2.0 which breaks this env - so
ONNX is the deployment format.

    .venv-dml/Scripts/python src/export_onnx.py [weights.pt] [imgsz]

defaults: runs/pothole/train_v3/weights/best.pt at imgsz=512 (the INPUT_SIZE
baked into PotholeDetector.java - change both together or the app will fail to
create the session).

Copies the result to android-app/app/src/main/assets/model.onnx, which is the
exact asset MainActivity loads. Rebuild the APK afterwards.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "android-app" / "app" / "src" / "main" / "assets"

CANDIDATES = [
    # newest/strongest run first; train_v3_ext is last on purpose - it was an
    # abandoned fine-tune and scores well below train_v3/train_v4.
    ROOT / "runs" / "pothole" / "train_v4" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train_v3" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train_v2" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train_v3_ext" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train_ext" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train" / "weights" / "best.pt",
]


def main() -> int:
    if len(sys.argv) > 1:
        weights = Path(sys.argv[1])
    else:
        weights = next((p for p in CANDIDATES if p.exists()), None)
    if weights is None or not weights.exists():
        print("no weights found - train first")
        return 1
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    from ultralytics import YOLO

    print(f"exporting {weights} at imgsz={imgsz}")
    model = YOLO(str(weights))
    out = Path(model.export(format="onnx", imgsz=imgsz, opset=12,
                            simplify=True, dynamic=False, device="cpu"))

    ASSETS.mkdir(parents=True, exist_ok=True)
    dest = ASSETS / "model.onnx"
    shutil.copyfile(out, dest)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    print(f"\ndeployed -> {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    for i in sess.get_inputs():
        print(f"  input  {i.name} {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  output {o.name} {o.shape} {o.type}")
    print("\nNext: rebuild the APK, then run src/validate_android_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())