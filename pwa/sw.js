/* Offline-first service worker: after the first visit, the whole app
   (UI + ORT runtime + 12 MB model) is cached and runs with zero network. */
const CACHE = "pothole-live-v1";
const ASSETS = [
  "./",
  "index.html",
  "app.js",
  "manifest.webmanifest",
  "model.onnx",
  "ort/ort.min.js",
  "ort/ort-wasm-simd.wasm",
  "ort/ort-wasm.wasm",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then(
      (hit) => hit || fetch(e.request).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return resp;
      })
    )
  );
});
