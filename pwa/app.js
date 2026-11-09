/* Live on-device pothole detection: YOLOv8n (ONNX) via onnxruntime-web.
   Everything runs in the browser — camera frames never leave the phone. */

const SIZE = 416;                 // model input (matches exported ONNX)
const CONF = 0.20;
const IOU = 0.45;
const CLASSES = ["D00 longitudinal", "D10 transverse", "D20 alligator", "D40 pothole"];
const COLORS = ["#4fc3f7", "#ffd54f", "#81c784", "#e57373"];

const video = document.getElementById("video");
const snap = document.getElementById("snap");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const hud = document.getElementById("hud");
const fpsEl = document.getElementById("fps");
const countEl = document.getElementById("count");
const camBtn = document.getElementById("camBtn");
const photoBtn = document.getElementById("photoBtn");
const fileInput = document.getElementById("filebtn");

let session = null;
let busy = false;
let running = false;          // live loop on
let stream = null;
let fpsWindow = [];
let mode = null;              // 'live' | 'photo'
let lastBoxes = [];

/* ---------- model ---------- */
async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "ort/";
    ort.env.wasm.numThreads = 1;   // no cross-origin isolation on Pages
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
    });
    statusEl.textContent = "✅ Model ready — start the camera, aim at the road.";
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js");
  } catch (e) {
    statusEl.textContent = "❌ Model failed to load: " + e.message;
    console.error(e);
  }
}

/* ---------- image -> tensor ---------- */
function frameToTensor(src, sw, sh) {
  // stretch the source into SIZE x SIZE (mapping back is a simple scale)
  const c = document.createElement("canvas");
  c.width = SIZE; c.height = SIZE;
  const cx = c.getContext("2d");
  cx.drawImage(src, 0, 0, SIZE, SIZE);
  const { data } = cx.getImageData(0, 0, SIZE, SIZE);
  const out = new Float32Array(3 * SIZE * SIZE);   // CHW
  const px = SIZE * SIZE;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = data[i] / 255;
    out[px + p] = data[i + 1] / 255;
    out[2 * px + p] = data[i + 2] / 255;
  }
  return new ort.Tensor("float32", out, [1, 3, SIZE, SIZE]);
}

/* ---------- postprocess: [1,8,N] -> boxes in source pixels ---------- */
function postprocess(output, dispW, dispH) {
  const t = output.dims;                 // [1, 8, N]
  const N = t[2];
  const d = output.data;
  const sx = dispW / SIZE, sy = dispH / SIZE;
  const raw = [];
  const sig = (x) => 1 / (1 + Math.exp(-x));
  for (let i = 0; i < N; i++) {
    let best = 0, bestScore = 0;
    for (let k = 0; k < 4; k++) {
      const s = sig(d[(4 + k) * N + i]);
      if (s > bestScore) { bestScore = s; best = k; }
    }
    if (bestScore < CONF) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    raw.push({
      cls: best, conf: bestScore,
      x1: (cx - w / 2) * sx, y1: (cy - h / 2) * sy,
      x2: (cx + w / 2) * sx, y2: (cy + h / 2) * sy,
    });
  }
  return nms(raw);
}

function nms(boxes) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of boxes) {
    let ok = true;
    for (const k of keep) {
      const ix = Math.max(0, Math.min(b.x2, k.x2) - Math.max(b.x1, k.x1));
      const iy = Math.max(0, Math.min(b.y2, k.y2) - Math.max(b.y1, k.y1));
      const inter = ix * iy;
      const a1 = (b.x2 - b.x1) * (b.y2 - b.y1);
      const a2 = (k.x2 - k.x1) * (k.y2 - k.y1);
      if (inter / (a1 + a2 - inter) > IOU) { ok = false; break; }
    }
    if (ok) keep.push(b);
  }
  return keep;
}

/* ---------- draw ---------- */
function draw(boxes, w, h) {
  overlay.width = w; overlay.height = h;
  octx.clearRect(0, 0, w, h);
  octx.lineWidth = Math.max(3, w / 220);
  octx.font = `bold ${Math.max(13, w / 34)}px system-ui`;
  for (const b of boxes) {
    const color = COLORS[b.cls];
    octx.strokeStyle = color;
    octx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const tag = `${b.cls} ${(b.conf * 100).toFixed(0)}%`
      .replace(/\d\b/, (m) => m) // cls index kept short
      .replace(/^(\d+)/, (m) => CLASSES[+m].split(" ")[0]);
    const label = `${CLASSES[b.cls].split(" ")[0]} ${(b.conf * 100).toFixed(0)}%`;
    const tw = octx.measureText(label).width + 12;
    const th = octx.lineWidth * 6;
    const ly = Math.max(0, b.y1 - th);
    octx.fillStyle = color;
    octx.fillRect(b.x1, ly, tw, th);
    octx.fillStyle = "#000";
    octx.fillText(label, b.x1 + 6, ly + th * 0.75);
  }
}

function updateHud(boxes) {
  countEl.textContent = `${boxes.length} box${boxes.length === 1 ? "" : "es"}`;
  const now = performance.now();
  fpsWindow.push(now);
  while (fpsWindow.length && now - fpsWindow[0] > 1000) fpsWindow.shift();
  fpsEl.textContent = `${fpsWindow.length} fps`;
}

/* ---------- inference on one frame ---------- */
async function inferOnce(src, dispW, dispH) {
  if (!session || busy) return;
  busy = true;
  try {
    const tensor = frameToTensor(src, dispW, dispH);
    const out = (await session.run({ images: tensor }));
    const key = Object.keys(out)[0];
    const boxes = postprocess(out[key], dispW, dispH);
    lastBoxes = boxes;
    draw(boxes, dispW, dispH);
    updateHud(boxes);
  } finally {
    busy = false;
  }
}

/* ---------- live loop ---------- */
async function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    await inferOnce(video, video.videoWidth, video.videoHeight);
  }
  requestAnimationFrame(loop);
}

/* ---------- camera ---------- */
camBtn.onclick = async () => {
  if (running) { stopCamera(); return; }
  if (!session) { statusEl.textContent = "Model still loading…"; return; }
  try {
    statusEl.textContent = "Requesting camera…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    video.style.display = "block";
    snap.style.display = "none";
    hud.style.display = "block";
    camBtn.textContent = "⏹ Stop Camera";
    camBtn.classList.add("sec");
    statusEl.textContent = "🔴 Live — aim at the street. Works with zero network.";
    running = true; mode = "live";
    requestAnimationFrame(loop);
  } catch (e) {
    statusEl.textContent = "Camera unavailable (" + e.name + ") — use 🖼️ Test Photo instead.";
  }
};

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.style.display = "none";
  hud.style.display = "none";
  octx.clearRect(0, 0, overlay.width, overlay.height);
  camBtn.textContent = "📷 Start Live Camera";
  camBtn.classList.remove("sec");
  statusEl.textContent = "Camera stopped.";
}

/* ---------- photo mode (fallback + no-camera testing) ---------- */
photoBtn.onclick = () => fileInput.click();
fileInput.onchange = async () => {
  const f = fileInput.files[0];
  if (!f) return;
  if (!session) { statusEl.textContent = "Model still loading…"; return; }
  if (running) stopCamera();
  const url = URL.createObjectURL(f);
  snap.src = url;
  await snap.decode();
  snap.style.display = "block";
  hud.style.display = "block";
  mode = "photo";
  statusEl.textContent = "Analyzing photo…";
  await inferOnce(snap, snap.naturalWidth, snap.naturalHeight);
  statusEl.textContent = lastBoxes.length
    ? `✅ ${lastBoxes.length} damage area(s) found in your photo.`
    : "ℹ️ No damage detected — try another angle/light.";
};

loadModel();
