#!/usr/bin/env bash
# Unattended finishing chain for the pothole detector.
#
#   1. wait for the stage-1 trainer to write outputs/train_log_v3.csv
#   2. finalise: evaluate -> export ONNX -> parity check -> rebuild APK
#   3. extend training from best.pt for another 16 epochs
#   4. finalise again with the extended weights
#
# Every stage is idempotent, so re-running the script after an interruption is
# safe. Progress lands in outputs/autopilot.log.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY=".venv-dml/Scripts/python.exe"
LOG="outputs/autopilot.log"
SDK="$ROOT/android-sdk"
GRADLE="$ROOT/tools/gradle-8.7/bin/gradle.bat"

mkdir -p outputs
say() { echo "[$(date '+%H:%M:%S')] $*" >>"$LOG"; }

finalise() {
    local tag="$1"
    say "== finalise ($tag): evaluate =="
    "$PY" src/evaluate.py >>"$LOG" 2>&1
    say "== finalise ($tag): export onnx =="
    if ! "$PY" src/export_onnx.py >>"$LOG" 2>&1; then
        say "EXPORT FAILED ($tag) - leaving the APK alone"
        return 1
    fi
    say "== finalise ($tag): android parity check =="
    "$PY" src/validate_android_pipeline.py >>"$LOG" 2>&1
    say "== finalise ($tag): build apk =="
    ( cd android-app && ANDROID_HOME="$SDK" "$GRADLE" assembleDebug --console=plain ) >>"$LOG" 2>&1
    if [ -f android-app/app/build/outputs/apk/debug/app-debug.apk ]; then
        say "APK READY ($tag): $(du -h android-app/app/build/outputs/apk/debug/app-debug.apk | cut -f1)"
    else
        say "APK BUILD FAILED ($tag)"
    fi
}

# ---- 1. wait for stage 1 ---------------------------------------------------
if [ ! -f outputs/train_log_v3.csv ]; then
    say "waiting for stage-1 training to finish"
    for _ in $(seq 1 720); do          # 720 x 30s = 6 h ceiling
        [ -f outputs/train_log_v3.csv ] && break
        sleep 30
    done
    if [ ! -f outputs/train_log_v3.csv ]; then
        say "stage-1 never finished; aborting"
        exit 1
    fi
fi
say "stage-1 complete"

# ---- 2. ship the stage-1 model --------------------------------------------
finalise "stage1"

# ---- 3. extend -------------------------------------------------------------
say "== stage 2: extend training +16 epochs =="
"$PY" src/train_v3.py 16 512 extend >>"$LOG" 2>&1
if [ -f runs/pothole/train_v3_ext/weights/best.pt ]; then
    say "stage-2 complete"
    # ---- 4. ship the extended model ---------------------------------------
    finalise "stage2"
else
    say "stage-2 produced no weights; keeping the stage-1 APK"
fi

say "autopilot done"
