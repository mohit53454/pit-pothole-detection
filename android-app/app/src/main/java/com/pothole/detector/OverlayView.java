package com.pothole.detector;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/** Draws tracked potholes on top of the camera preview (FILL_CENTER mapping). */
public class OverlayView extends View {

    /** Tracks dimmer than this are still being confirmed. */
    private static final int ALPHA_UNCONFIRMED = 130;
    private static final int ALPHA_COASTING = 100;

    private List<PotholeTracker.Track> tracks = new ArrayList<>();
    private int bmpW = 1, bmpH = 1;

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF tmp = new RectF();
    private final DashPathEffect dashed = new DashPathEffect(new float[]{14f, 10f}, 0f);

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
    }

    public void setTracks(List<PotholeTracker.Track> t, int w, int h) {
        tracks = t;
        bmpW = Math.max(1, w);
        bmpH = Math.max(1, h);
        postInvalidateOnAnimation();
    }

    public void clear() {
        tracks = new ArrayList<>();
        postInvalidateOnAnimation();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (tracks.isEmpty()) return;

        float vw = getWidth(), vh = getHeight();
        // PreviewView uses FILL_CENTER: scale to cover, then centre-crop.
        float s = Math.max(vw / bmpW, vh / bmpH);
        float ox = (vw - bmpW * s) / 2f;
        float oy = (vh - bmpH * s) / 2f;

        float stroke = Math.max(3f, vw * 0.006f);
        float textSize = Math.max(13f, vw * 0.036f);
        textPaint.setTextSize(textSize);

        for (PotholeTracker.Track t : tracks) {
            int color = PotholeDetector.COLORS[t.classId];

            tmp.set(t.box.left * s + ox, t.box.top * s + oy,
                    t.box.right * s + ox, t.box.bottom * s + oy);
            // Clamp so a box running off-screen still draws its visible edge.
            tmp.intersect(0f, 0f, vw, vh);
            if (tmp.isEmpty()) continue;

            int alpha = t.confirmed
                    ? (t.fresh ? 255 : ALPHA_COASTING)
                    : ALPHA_UNCONFIRMED;

            boxPaint.setColor(color);
            boxPaint.setAlpha(alpha);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(stroke);
            // Dashed = coasting on a missed frame, so the box is not read as a fresh hit.
            boxPaint.setPathEffect(t.fresh ? null : dashed);
            canvas.drawRect(tmp, boxPaint);
            boxPaint.setPathEffect(null);

            String label = t.confirmed
                    ? "#" + t.id + " " + Math.round(t.score * 100) + "%"
                    : "...";
            float tw = textPaint.measureText(label);
            float th = textSize * 1.4f;
            float top = Math.max(0, tmp.top - th);
            float right = Math.min(vw, tmp.left + tw + 12);

            textPaint.setStyle(Paint.Style.FILL);
            textPaint.setColor(color);
            textPaint.setAlpha(alpha);
            canvas.drawRect(tmp.left, top, right, top + th, textPaint);
            textPaint.setColor(Color.BLACK);
            textPaint.setAlpha(alpha);
            canvas.drawText(label, tmp.left + 6, top + th - textSize * 0.3f, textPaint);
        }
    }
}
