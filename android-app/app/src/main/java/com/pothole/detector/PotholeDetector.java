package com.pothole.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * YOLOv8 pothole detector running 100% on-device via ONNX Runtime.
 *
 * Input  "images"  : float32 [1,3,512,512], RGB /255, letterboxed with pad 114.
 * Output "output0" : float32 [1,5,5376] = 4 box coords (cx,cy,w,h in 512-space)
 *                    + 1 class probability. Ultralytics already applies the
 *                    sigmoid inside the Detect head, so scores are used as-is.
 *
 * Single class on purpose: the v3 model is trained on potholes only (plus
 * crack/clean-road hard negatives). Splitting the label space across crack
 * types cost most of the pothole accuracy for no benefit to this app.
 */
public class PotholeDetector implements AutoCloseable {

    public static class Detection {
        public final RectF box;    // source-bitmap coordinates
        public final int classId;
        public final String label;
        public final float score;

        public Detection(RectF box, int classId, String label, float score) {
            this.box = box;
            this.classId = classId;
            this.label = label;
            this.score = score;
        }
    }

    public static final String[] LABELS = {"pothole"};
    public static final String[] SHORT_LABELS = {"POTHOLE"};
    public static final int[] COLORS = {
            0xFFFF3B30    // pothole - red
    };
    public static final int POTHOLE_CLASS = 0;

    /** Must match the imgsz the ONNX file was exported at (src/export_onnx.py). */
    private static final int INPUT_SIZE = 512;
    private static final float IOU_THRESH = 0.45f;
    private static final int PAD = 114;
    private static final String TAG = "PIT";

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    /** Output layout, resolved once at construction. */
    private final int channels;
    private final int anchors;
    private final boolean chFirst;
    private final int numClasses;

    /** Reused across frames so live scanning does not churn the allocator. */
    private final FloatBuffer inputBuf;
    private final Bitmap letterbox;
    private final Canvas letterboxCanvas;
    private final int[] pixels;
    private final float[] scores;

    /**
     * Live default. The old 0.25 discarded roughly two thirds of real potholes
     * before tracking even began. 0.15 costs precision, but the tracker's
     * CONFIRM_HITS gate is what rejects one-frame false positives now, so weak
     * candidates are filtered downstream instead of thrown away here.
     */
    private volatile float confThresh = 0.15f;

    public static byte[] loadModelBytes(Context ctx, String asset) throws IOException {
        try (InputStream in = ctx.getAssets().open(asset);
             ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
            return bos.toByteArray();
        }
    }

    public PotholeDetector(Context ctx, String asset) throws IOException, OrtException {
        byte[] model = loadModelBytes(ctx, asset);
        env = OrtEnvironment.getEnvironment();

        // Graph optimisation and intra-op threading are the two settings that
        // have failed to create a session on device. A bare session is slower but
        // fully usable, so never let a tuning knob be the reason nothing detects.
        OrtSession created = null;
        Throwable tunedErr = null;
        try {
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setIntraOpNumThreads(Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors())));
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            try {
                created = env.createSession(model, opts);
            } finally {
                opts.close();
            }
        } catch (Throwable t) {
            tunedErr = t;
            Log.e(TAG, "tuned ONNX session failed, retrying with defaults", t);
        }
        if (created == null) {
            try {
                created = env.createSession(model, new OrtSession.SessionOptions());
            } catch (Throwable plainErr) {
                Log.e(TAG, "plain ONNX session failed too", plainErr);
                IOException io = new IOException(
                        "ONNX session could not be created. tuned=" + tunedErr
                        + " plain=" + plainErr);
                io.initCause(plainErr);
                throw io;
            }
        }
        session = created;

        inputName = session.getInputNames().iterator().next();

        // getOutputInfo() is a Map<String, NodeInfo>, not a List. Indexing it with
        // 0 compiles but always yields null, which is what crashed the app with
        // "ValueInfo NodeInfo.getInfo() on a null object reference". Take the
        // first *value* instead so the output name never has to be hard-coded.
        java.util.Map<String, ai.onnxruntime.NodeInfo> outInfo = session.getOutputInfo();
        if (outInfo.isEmpty()) {
            throw new IOException("model exposes no outputs");
        }
        long[] shape = ((TensorInfo) outInfo.values().iterator().next().getInfo()).getShape();
        int d1 = shape.length > 1 ? (int) shape[1] : 5;
        int d2 = shape.length > 2 ? (int) shape[2] : 5376;
        chFirst = d1 < d2;
        channels = chFirst ? d1 : d2;
        anchors = chFirst ? d2 : d1;
        numClasses = Math.min(LABELS.length, Math.max(1, channels - 4));

        inputBuf = FloatBuffer.allocate(INPUT_SIZE * INPUT_SIZE * 3);
        letterbox = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        letterboxCanvas = new Canvas(letterbox);
        pixels = new int[INPUT_SIZE * INPUT_SIZE];
        scores = new float[anchors * channels];
    }

    public void setConfThresh(float v) {
        confThresh = Math.max(0.05f, Math.min(0.9f, v));
    }

    public float getConfThresh() {
        return confThresh;
    }

    /** Runs detection; returned boxes are in src's coordinate space. */
    public List<Detection> detect(Bitmap src) throws OrtException {
        final int sw = src.getWidth(), sh = src.getHeight();
        if (sw < 2 || sh < 2) return Collections.emptyList();

        final float scale = Math.min(INPUT_SIZE / (float) sw, INPUT_SIZE / (float) sh);
        final int nw = Math.max(1, Math.round(sw * scale));
        final int nh = Math.max(1, Math.round(sh * scale));
        final int dx = (INPUT_SIZE - nw) / 2;
        final int dy = (INPUT_SIZE - nh) / 2;

        letterboxCanvas.drawColor(Color.rgb(PAD, PAD, PAD));
        Bitmap scaled = Bitmap.createScaledBitmap(src, nw, nh, true);
        letterboxCanvas.drawBitmap(scaled, dx, dy, null);
        if (scaled != src) scaled.recycle();
        letterbox.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // CHW, RGB, normalised to 0..1. Three passes over the pixel array so
        // no per-frame temporary buffers are allocated.
        final int plane = INPUT_SIZE * INPUT_SIZE;
        inputBuf.rewind();
        for (int i = 0; i < plane; i++) inputBuf.put(((pixels[i] >> 16) & 0xFF) / 255f);
        for (int i = 0; i < plane; i++) inputBuf.put(((pixels[i] >> 8) & 0xFF) / 255f);
        for (int i = 0; i < plane; i++) inputBuf.put((pixels[i] & 0xFF) / 255f);
        inputBuf.rewind();

        final float thresh = confThresh;
        List<Detection> cands = new ArrayList<>();

        try (OnnxTensor input = OnnxTensor.createTensor(env, inputBuf,
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {

            OnnxTensor out = (OnnxTensor) result.get(0);
            FloatBuffer fb = out.getFloatBuffer();
            fb.rewind();
            fb.get(scores, 0, Math.min(scores.length, fb.remaining()));

            for (int a = 0; a < anchors; a++) {
                float best = 0f;
                int bestC = -1;
                for (int k = 0; k < numClasses; k++) {
                    int idx = chFirst ? (4 + k) * anchors + a : a * channels + 4 + k;
                    float s = scores[idx];
                    if (s > best) {
                        best = s;
                        bestC = k;
                    }
                }
                if (bestC < 0 || best < thresh) continue;

                int i0 = chFirst ? a : a * channels;
                float cx = scores[i0], cy = scores[i0 + 1];
                float w = scores[i0 + 2], h = scores[i0 + 3];

                // undo letterbox -> source coordinates
                float sx1 = clamp((cx - w / 2f - dx) / scale, 0, sw);
                float sy1 = clamp((cy - h / 2f - dy) / scale, 0, sh);
                float sx2 = clamp((cx + w / 2f - dx) / scale, 0, sw);
                float sy2 = clamp((cy + h / 2f - dy) / scale, 0, sh);
                if (sx2 - sx1 < 2 || sy2 - sy1 < 2) continue;

                cands.add(new Detection(new RectF(sx1, sy1, sx2, sy2), bestC, LABELS[bestC], best));
            }
        }
        return nms(cands);
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static List<Detection> nms(List<Detection> dets) {
        dets.sort((a, b) -> Float.compare(b.score, a.score));
        boolean[] suppressed = new boolean[dets.size()];
        List<Detection> out = new ArrayList<>();
        for (int i = 0; i < dets.size(); i++) {
            if (suppressed[i]) continue;
            Detection cur = dets.get(i);
            out.add(cur);
            for (int j = i + 1; j < dets.size(); j++) {
                Detection o = dets.get(j);
                if (suppressed[j] || o.classId != cur.classId) continue;
                if (iou(cur.box, o.box) > IOU_THRESH) suppressed[j] = true;
            }
        }
        return out;
    }

    private static float iou(RectF a, RectF b) {
        float l = Math.max(a.left, b.left), r = Math.min(a.right, b.right);
        float t = Math.max(a.top, b.top), bo = Math.min(a.bottom, b.bottom);
        if (r <= l || bo <= t) return 0f;
        float inter = (r - l) * (bo - t);
        float uni = a.width() * a.height() + b.width() * b.height() - inter;
        return uni <= 0 ? 0 : inter / uni;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
            // nothing useful to do at teardown
        }
    }
}