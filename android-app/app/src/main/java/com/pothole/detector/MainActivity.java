package com.pothole.detector;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.InputStream;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Live camera pothole detection — everything runs on-device, no network. */
public class MainActivity extends ComponentActivity {

    private static final int REQ_CAMERA = 100;
    private static final int REQ_GALLERY = 101;
    private static final String MODEL_ASSET = "model.onnx";

    private PreviewView previewView;
    private OverlayView overlay;
    private TextView status;
    private TextView fpsView;
    private TextView confValue;
    private SeekBar confSeek;

    private PotholeDetector detector;
    /** Non-null when the ONNX model could not be loaded; shown instead of status. */
    private String loadError;
    /** Fraction of frame height below which a pothole counts as "right ahead". */
    private static final float HORIZON = 0.55f;
    private final PotholeTracker tracker = new PotholeTracker();
    private boolean useFront = false;
    private boolean bound = false;

    private ExecutorService ioExec;
    private final android.os.Handler mainHandler = new Handler(Looper.getMainLooper());
    private final AtomicBoolean busy = new AtomicBoolean(false);

    private long lastFrameNs = 0L;
    private float fps = 0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.camera_preview);
        overlay = findViewById(R.id.overlay);
        status = findViewById(R.id.status);
        fpsView = findViewById(R.id.fps);
        confValue = findViewById(R.id.conf_value);
        confSeek = findViewById(R.id.conf_seek);
        Button btnGallery = findViewById(R.id.btn_gallery);
        Button btnFlip = findViewById(R.id.btn_flip);

        // PreviewView crops to fill; the overlay must use the same mapping.
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);

        ioExec = Executors.newSingleThreadExecutor();

        try {
            detector = new PotholeDetector(this, MODEL_ASSET);
        } catch (Throwable e) {
            // Throwable, not Exception: a native ONNX library that will not load
            // surfaces as UnsatisfiedLinkError, which an Exception catch misses.
            detector = null;
            loadError = "Model failed to load: " + describe(e);
            status.setText(loadError);
            btnGallery.setEnabled(false);
            btnFlip.setEnabled(false);
            confSeek.setEnabled(false);
            // Custom pill backgrounds carry no disabled state, so dim the
            // controls explicitly or they still look tappable.
            btnGallery.setAlpha(0.35f);
            btnFlip.setAlpha(0.35f);
            confSeek.setAlpha(0.35f);
        }

        // SeekBar 0..85 -> confidence 0.05..0.90
        confSeek.setMax(85);
        confSeek.setProgress(detector != null
                ? Math.round((detector.getConfThresh() - 0.05f) * 100f) : 20);
        updateConfLabel(confSeek.getProgress());
        confSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar sb, int p, boolean fromUser) {
                updateConfLabel(p);
                if (detector != null) detector.setConfThresh(0.05f + p / 100f);
            }
            @Override public void onStartTrackingTouch(SeekBar sb) { }
            @Override public void onStopTrackingTouch(SeekBar sb) { }
        });

        btnGallery.setOnClickListener(v -> pickImage());
        btnFlip.setOnClickListener(v -> {
            useFront = !useFront;
            // Rebind even when the previous attempt failed. Gating on `bound`
            // stranded the user on a dead preview if one camera was unavailable,
            // because Flip could never try again.
            startCamera();
        });

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            status.setText(R.string.permission_rationale);
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    private void updateConfLabel(int progress) {
        confValue.setText(String.format(java.util.Locale.US, "%.2f", 0.05f + progress / 100f));
    }

    /** One-line, on-screen summary of a load failure (no room for a stack trace). */
    private static String describe(Throwable t) {
        String msg = t.getMessage();
        String cls = t.getClass().getSimpleName();
        return (msg == null || msg.isEmpty()) ? cls : cls + ": " + msg;
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] grantResults) {
        super.onRequestPermissionsResult(req, perms, grantResults);
        if (req == REQ_CAMERA && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            status.setText(R.string.status_no_camera);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> fut = ProcessCameraProvider.getInstance(this);
        fut.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = fut.get();

                Preview preview = new Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Same 16:9 aspect as the preview so overlay mapping stays exact.
                // CameraX forbids combining aspect ratio with target resolution.
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                analysis.setAnalyzer(ioExec, this::analyze);

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this,
                        useFront ? CameraSelector.DEFAULT_FRONT_CAMERA
                                 : CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, analysis);
                bound = true;
                // Fresh ids for a different camera or a re-bind.
                tracker.reset();
                overlay.clear();
                // Do not overwrite the model-load error: with no detector this
                // line is the only thing telling the user why nothing detects.
                status.setText(detector != null
                        ? getString(R.string.status_scanning)
                        : loadError);
            } catch (Exception e) {
                bound = false;
                status.setText(R.string.status_no_camera);
            }
        }, mainHandler::post);
    }

    private void analyze(ImageProxy proxy) {
        // compareAndSet: only the idle->busy transition may run inference.
        if (detector == null || !busy.compareAndSet(false, true)) {
            proxy.close();
            return;
        }
        Bitmap bmp = null;
        try {
            bmp = rgbaToBitmap(proxy);
            int rot = proxy.getImageInfo().getRotationDegrees();
            if (rot != 0) {
                Matrix m = new Matrix();
                m.postRotate(rot);
                Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
                bmp.recycle();
                bmp = rotated;
            }

            final List<PotholeDetector.Detection> dets = detector.detect(bmp);
            final int w = bmp.getWidth(), h = bmp.getHeight();
            final List<PotholeTracker.Track> tracks = tracker.update(dets, w, h);

            // Count only tracks the tracker has confirmed, so a one-frame false
            // positive never reaches the status line.
            final int potholes = PotholeTracker.confirmedCount(tracks);
            final PotholeTracker.Track close = PotholeTracker.nearestAhead(tracks, h, HORIZON);
            final String txt;
            if (tracks.isEmpty()) {
                txt = getString(R.string.status_scanning);
            } else if (potholes == 0) {
                txt = "Confirming...";
            } else {
                txt = potholes + " pothole" + (potholes == 1 ? "" : "s")
                      + (close != null ? "  \u26a0 ahead #" + close.id : "");
            }

            final long now = System.nanoTime();
            if (lastFrameNs != 0L) {
                float inst = 1e9f / (now - lastFrameNs);
                fps = fps == 0f ? inst : fps * 0.85f + inst * 0.15f;
            }
            lastFrameNs = now;
            final String fpsTxt = String.format(java.util.Locale.US, "%.1f fps", fps);

            mainHandler.post(() -> {
                overlay.setTracks(tracks, w, h);
                status.setText(txt);
                fpsView.setText(fpsTxt);
            });
        } catch (Throwable t) {
            mainHandler.post(() -> status.setText("Inference error: " + t.getMessage()));
        } finally {
            // Always release the camera buffer, or CameraX stalls after a few frames.
            proxy.close();
            if (bmp != null) bmp.recycle();
            busy.set(false);
        }
    }

    /** RGBA_8888 ImageProxy -> ARGB bitmap (stride-safe). */
    private Bitmap rgbaToBitmap(ImageProxy proxy) {
        ImageProxy.PlaneProxy plane = proxy.getPlanes()[0];
        java.nio.ByteBuffer buf = plane.getBuffer().duplicate();
        buf.rewind();
        int w = proxy.getWidth(), h = proxy.getHeight();
        int rowStride = plane.getRowStride();
        int pixStride = plane.getPixelStride();

        java.nio.ByteBuffer pixels = java.nio.ByteBuffer.allocate(w * h * 4);
        if (pixStride == 4) {
            byte[] row = new byte[w * 4];
            for (int y = 0; y < h; y++) {
                buf.position(y * rowStride);
                buf.get(row, 0, w * 4);
                pixels.put(row);
            }
        } else {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int off = y * rowStride + x * pixStride;
                    pixels.put(buf.get(off));
                    pixels.put(buf.get(off + 1));
                    pixels.put(buf.get(off + 2));
                    pixels.put(buf.get(off + 3));
                }
            }
        }
        pixels.rewind();
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        bmp.copyPixelsFromBuffer(pixels);
        return bmp;
    }

    // ---------------- gallery ----------------

    private void pickImage() {
        Intent i = new Intent(Intent.ACTION_GET_CONTENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("image/*");
        try {
            startActivityForResult(i, REQ_GALLERY);
        } catch (android.content.ActivityNotFoundException e) {
            // No document picker on the device. Report it instead of crashing,
            // which would be indistinguishable from a dead button.
            status.setText("No gallery app available");
        }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req != REQ_GALLERY || res != RESULT_OK || data == null || data.getData() == null) return;
        final Uri uri = data.getData();
        ioExec.execute(() -> {
            if (detector == null) return;
            Bitmap bmp = decodeUri(uri);
            if (bmp == null) return;
            try {
                final List<PotholeDetector.Detection> dets = detector.detect(bmp);
                int potholes = 0;
                for (PotholeDetector.Detection d : dets) {
                    if (d.classId == PotholeDetector.POTHOLE_CLASS) potholes++;
                }
                final int p = potholes;

                Bitmap annotated = bmp.copy(Bitmap.Config.ARGB_8888, true);
                drawDetections(new Canvas(annotated), dets,
                        Math.max(3f, bmp.getWidth() * 0.004f));
                final Bitmap result = annotated;
                mainHandler.post(() -> {
                    ImageView iv = new ImageView(this);
                    iv.setAdjustViewBounds(true);
                    iv.setScaleType(ImageView.ScaleType.FIT_CENTER);
                    iv.setPadding(8, 8, 8, 8);
                    iv.setImageBitmap(result);
                    String title = dets.isEmpty()
                            ? "No damage found"
                            : dets.size() + " finding" + (dets.size() == 1 ? "" : "s")
                              + " — " + p + " pothole" + (p == 1 ? "" : "s");
                    new AlertDialog.Builder(this)
                            .setTitle(title)
                            .setView(iv)
                            .setPositiveButton("OK", (d, which) -> result.recycle())
                            .setOnCancelListener(d -> result.recycle())
                            .show();
                });
            } catch (Throwable t) {
                mainHandler.post(() -> status.setText("Inference error: " + t.getMessage()));
            } finally {
                bmp.recycle();
            }
        });
    }

    private Bitmap decodeUri(Uri uri) {
        try {
            BitmapFactory.Options o = new BitmapFactory.Options();
            o.inJustDecodeBounds = true;
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                BitmapFactory.decodeStream(in, null, o);
            }
            int sample = 1;
            while (Math.max(o.outWidth, o.outHeight) / (sample * 2) >= 1280) sample *= 2;
            BitmapFactory.Options o2 = new BitmapFactory.Options();
            o2.inSampleSize = sample;
            Bitmap bmp;
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                bmp = BitmapFactory.decodeStream(in, null, o2);
            }
            return bmp == null ? null : applyExifRotation(uri, bmp);
        } catch (Exception e) {
            return null;
        }
    }

    /** Camera JPEGs carry rotation in EXIF, which BitmapFactory ignores. */
    private Bitmap applyExifRotation(Uri uri, Bitmap bmp) {
        int orientation;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            orientation = new ExifInterface(in).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        } catch (Exception e) {
            return bmp;
        }
        Matrix m = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90: m.postRotate(90); break;
            case ExifInterface.ORIENTATION_ROTATE_180: m.postRotate(180); break;
            case ExifInterface.ORIENTATION_ROTATE_270: m.postRotate(270); break;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: m.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL: m.postScale(1, -1); break;
            case ExifInterface.ORIENTATION_TRANSPOSE: m.postRotate(90); m.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_TRANSVERSE: m.postRotate(270); m.postScale(-1, 1); break;
            default: return bmp;
        }
        Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
        if (rotated != bmp) bmp.recycle();
        return rotated;
    }

    static void drawDetections(Canvas cv, List<PotholeDetector.Detection> dets, float stroke) {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        float textSize = Math.max(14f, cv.getWidth() * 0.03f);
        p.setTextSize(textSize);
        for (PotholeDetector.Detection d : dets) {
            int color = PotholeDetector.COLORS[d.classId];
            p.setColor(color);
            p.setStyle(Paint.Style.STROKE);
            p.setStrokeWidth(stroke);
            cv.drawRect(d.box, p);

            String t = d.label + " " + Math.round(d.score * 100) + "%";
            float tw = p.measureText(t);
            float th = textSize * 1.4f;
            float top = Math.max(0, d.box.top - th);
            p.setStyle(Paint.Style.FILL);
            p.setColor(color);
            cv.drawRect(d.box.left, top, d.box.left + tw + 12, top + th, p);
            p.setColor(Color.BLACK);
            cv.drawText(t, d.box.left + 6, top + th - textSize * 0.3f, p);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Closing on ioExec's single thread keeps it behind any in-flight run().
        if (ioExec != null) {
            ioExec.execute(() -> {
                if (detector != null) detector.close();
            });
            ioExec.shutdown();
        } else if (detector != null) {
            detector.close();
        }
    }
}