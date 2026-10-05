#!/bin/bash
# GPU 0 chain (encdec seed424 training already finished 2026-09-02 13:26):
#   1. inference of the encdec seed424 model (testsplit seed424 + test2025 remap)
#   2. train seed424 / alpha 1.0
#   3. inference of the new alpha1 model (testsplit seed424 + test2025 remap)
set -u
cd /home/basile/workdir/vona_legacy_window_split
export CUDA_VISIBLE_DEVICES=0
PY=/home/basile/miniconda3/envs/tesi_rev/bin/python

run() {
  echo "[queue-gpu0] $(date '+%F %T') >>> $1"
  "$PY" -u scripts/run.py --config "configs/$1"
  local rc=$?
  echo "[queue-gpu0] $(date '+%F %T') <<< $1 (exit $rc)"
  return $rc
}

# --- inference of the already-trained encdec seed424 model (tolerate failure) ---
run inference_testsplit_legacy_encdec_seed424.yaml || true
run inference_test2025_legacy_encdec_seed424_remap.yaml || true

# --- new training: seed 424, alpha 1.0 (abort chain if it fails) ---
if ! run train_seed424_alpha1.yaml; then
  echo "[queue-gpu0] $(date '+%F %T') TRAIN FAILED - aborting inference"
  exit 1
fi

# --- inference of the new alpha1 model ---
run inference_testsplit_seed424_alpha1.yaml || true
run inference_test2025_seed424_alpha1_remap.yaml || true

echo "[queue-gpu0] $(date '+%F %T') ALL DONE"
