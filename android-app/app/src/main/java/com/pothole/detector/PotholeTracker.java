package com.pothole.detector;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Frame-to-frame tracker for live camera detection.
 *
 * <p>A raw detector calls every pothole a new object on every frame, so boxes
 * jitter and blink as confidence crosses the threshold. This assigns each
 * pothole a stable id, averages its box over time, and lets it coast for a few
 * frames when a single detection is missed. That is what makes the overlay read
 * as live tracking rather than a strobe.
 */
public class PotholeTracker {

    /** One pothole being followed across frames. */
    public static final class Track {
        public int id;
        /** Smoothed box in source-bitmap pixels. */
        public final RectF box = new RectF();
        /** Smoothed confidence in 0..1. */
        public float score;
        /** Consecutive frames this track was matched. */
        public int hits;
        /** Consecutive frames it was not matched. */
        public int missed;
        /** Seen in enough frames to trust; unconfirmed tracks draw faint. */
        public boolean confirmed;
        /** Matched on the most recent frame (false while coasting). */
        public boolean fresh;
        public int classId;

        public float width() { return box.width(); }
        public float height() { return box.height(); }
    }

    /**
     * Minimum IoU for a detection to be considered the same pothole.
     *
     * <p>0.30 was too strict for a moving phone: at ~5 fps of inference the box
     * of a pothole you are driving towards shifts far enough between frames to
     * fall under that overlap, so association failed, hits reset to 1 every
     * frame, and nothing ever reached CONFIRM_HITS. That is what made the app
     * look like it detected nothing while the vehicle was moving.
     */
    private static final float IOU_MATCH = 0.20f;
    /** Weight of the newest box in the moving average (higher = less lag). */
    private static final float BOX_SMOOTH = 0.55f;
    private static final float SCORE_SMOOTH = 0.45f;
    /**
     * Matches needed before a track is drawn as confirmed.
     *
     * <p>Two is enough now that the detector runs at a lower confidence: a
     * single-frame false positive still cannot confirm, but a real pothole only
     * needs to survive one frame-to-frame association instead of two.
     */
    private static final int CONFIRM_HITS = 2;
    /** Frames a track survives without a matching detection. */
    private static final int MAX_MISSED = 8;
    /** Per-frame confidence decay while coasting. */
    private static final float COAST_DECAY = 0.90f;

    private final List<Track> tracks = new ArrayList<>();
    private int nextId = 1;
    private int frameW = -1, frameH = -1;

    /** Drop every track. Called on camera flip, model reload, or frame-size change. */
    public void reset() {
        tracks.clear();
        nextId = 1;
    }

    public int getFrameWidth() { return frameW; }
    public int getFrameHeight() { return frameH; }

    /**
     * Consume one frame of detections and return the live tracks.
     *
     * @param dets detections in source-bitmap pixel coordinates
     * @param w    source bitmap width
     * @param h    source bitmap height
     * @return tracks to draw, highest confidence first
     */
    public List<Track> update(List<PotholeDetector.Detection> dets, int w, int h) {
        // A resolution or rotation change invalidates every box we hold.
        if (w != frameW || h != frameH) {
            reset();
            frameW = w;
            frameH = h;
        }
        for (Track t : tracks) t.fresh = false;

        // Greedy best-IoU association: strongest overlap claims first.
        final int nt = tracks.size(), nd = dets.size();
        List<float[]> pairs = new ArrayList<>();
        for (int ti = 0; ti < nt; ti++) {
            for (int di = 0; di < nd; di++) {
                float iou = iou(tracks.get(ti).box, dets.get(di).box);
                if (iou >= IOU_MATCH) pairs.add(new float[]{iou, ti, di});
            }
        }
        Collections.sort(pairs, (a, b) -> Float.compare(b[0], a[0]));

        boolean[] trackUsed = new boolean[nt];
        boolean[] detUsed = new boolean[nd];
        for (float[] p : pairs) {
            int ti = (int) p[1], di = (int) p[2];
            if (trackUsed[ti] || detUsed[di]) continue;
            trackUsed[ti] = true;
            detUsed[di] = true;

            Track t = tracks.get(ti);
            PotholeDetector.Detection d = dets.get(di);

            t.box.left   += (d.box.left   - t.box.left)   * BOX_SMOOTH;
            t.box.top    += (d.box.top    - t.box.top)    * BOX_SMOOTH;
            t.box.right  += (d.box.right  - t.box.right)  * BOX_SMOOTH;
            t.box.bottom += (d.box.bottom - t.box.bottom) * BOX_SMOOTH;
            t.score      += (d.score - t.score) * SCORE_SMOOTH;

            t.hits++;
            t.missed = 0;
            t.fresh = true;
            t.classId = d.classId;
            t.confirmed = t.hits >= CONFIRM_HITS;
        }

        // Unmatched tracks coast, then expire.
        for (int ti = nt - 1; ti >= 0; ti--) {
            if (trackUsed[ti]) continue;
            Track t = tracks.get(ti);
            t.missed++;
            t.score *= COAST_DECAY;
            if (t.missed > MAX_MISSED) tracks.remove(ti);
        }

        // Unmatched detections start new tracks.
        for (int di = 0; di < nd; di++) {
            if (detUsed[di]) continue;
            PotholeDetector.Detection d = dets.get(di);
            Track t = new Track();
            t.id = nextId++;
            t.box.set(d.box);
            t.score = d.score;
            t.hits = 1;
            t.classId = d.classId;
            t.fresh = true;
            t.confirmed = CONFIRM_HITS <= 1;
            tracks.add(t);
        }

        List<Track> out = new ArrayList<>(tracks);
        Collections.sort(out, (a, b) -> Float.compare(b.score, a.score));
        return out;
    }

    /**
     * How many tracks are considered real right now.
     *
     * <p>Coasting tracks still count. Requiring {@code fresh} made a confirmed
     * pothole vanish from the count on any single missed frame, so the status
     * line strobed between "N potholes" and "Confirming..." while the pothole
     * was plainly still in view. MAX_MISSED already bounds staleness.
     */
    public static int confirmedCount(List<Track> ts) {
        int n = 0;
        for (Track t : ts) if (t.confirmed) n++;
        return n;
    }

    /**
     * The pothole closest to the vehicle: the confirmed track reaching lowest in
     * the frame, ignoring anything above {@code horizonFrac} (far road ahead).
     *
     * @return that track, or null when nothing is both confirmed and near
     */
    public static Track nearestAhead(List<Track> ts, int frameH, float horizonFrac) {
        Track best = null;
        for (Track t : ts) {
            if (!t.confirmed) continue;
            if (t.box.bottom < horizonFrac * frameH) continue;
            if (best == null || t.box.bottom > best.box.bottom) best = t;
        }
        return best;
    }

    private static float iou(RectF a, RectF b) {
        float ix = Math.max(0f, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        float iy = Math.max(0f, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        float inter = ix * iy;
        if (inter <= 0f) return 0f;
        float union = a.width() * a.height() + b.width() * b.height() - inter;
        return union <= 0f ? 0f : inter / union;
    }
}
