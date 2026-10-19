"""Pothole Detector — phone-friendly web app.

Run:  python app.py     then on your phone (same Wi-Fi) open the printed URL.

Serves a mobile camera UI, runs YOLOv8n inference server-side, returns boxes.
Uses the best available weights (extended run if present, else baseline).
"""

from __future__ import annotations

import io
import socket
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DEMO = ROOT / "data" / "demo_images"

CANDIDATES = [
    ROOT / "runs" / "pothole" / "train_ext" / "weights" / "best.pt",
    ROOT / "runs" / "pothole" / "train" / "weights" / "best.pt",
]
WEIGHTS = next((p for p in CANDIDATES if p.exists()), None)
if WEIGHTS is None:
    raise SystemExit("no weights found — run training first (python run.py)")

CLASSES = ["D00 longitudinal crack", "D10 transverse crack",
           "D20 alligator crack", "D40 pothole/patch"]
CONF = 0.15  # demo-friendly threshold; UI displays each box's confidence

app = Flask(__name__)
model = YOLO(str(WEIGHTS))


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/sample/<name>")
def sample(name: str):
    return send_from_directory(DEMO, name)


@app.get("/samples")
def samples():
    return jsonify(sorted(p.name for p in DEMO.glob("*.jpg")))


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image"}), 400
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    result = model.predict(img, imgsz=640, conf=CONF, verbose=False)[0]
    boxes = [
        {
            "cls": int(b.cls),
            "label": CLASSES[int(b.cls)],
            "conf": round(float(b.conf), 3),
            "xyxy": [round(float(v), 1) for v in b.xyxy.flatten().tolist()],
        }
        for b in result.boxes
    ]
    return jsonify({"w": img.width, "h": img.height, "boxes": boxes})


if __name__ == "__main__":
    ip = lan_ip()
    print("\n" + "=" * 52)
    print(f"  Pothole Detector is live")
    print(f"  On this PC : http://localhost:5000")
    print(f"  On your phone (same Wi-Fi): http://{ip}:5000")
    print(f"  weights    : {WEIGHTS}")
    print("=" * 52 + "\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
