#!/bin/bash
# GPU 1 chain (resnet seed424 training already finished 2026-09-02 13:48):
#   1. inference of the resnet seed424 model (testsplit seed424 + test2025 remap)
#   2. train seed424 / alpha 0.0
#   3. inference of the new alpha0 model (testsplit seed424 + test2025 remap)
set -u
cd /home/basile/workdir/vona_legacy_window_split
export CUDA_VISIBLE_DEVICES=1
PY=/home/basile/miniconda3/envs/tesi_rev/bin/python

run() {
  echo "[queue-gpu1] $(date '+%F %T') >>> $1"
  "$PY" -u scripts/run.py --config "configs/$1"
  local rc=$?
  echo "[queue-gpu1] $(date '+%F %T') <<< $1 (exit $rc)"
  return $rc
}

# --- inference of the already-trained resnet seed424 model (tolerate failure) ---
run inference_testsplit_legacy_resnet_seed424.yaml || true
run inference_test2025_legacy_resnet_seed424_remap.yaml || true

# --- new training: seed 424, alpha 0.0 (abort chain if it fails) ---
if ! run train_seed424_alpha0.yaml; then
  echo "[queue-gpu1] $(date '+%F %T') TRAIN FAILED - aborting inference"
  exit 1
fi

# --- inference of the new alpha0 model ---
run inference_testsplit_seed424_alpha0.yaml || true
run inference_test2025_seed424_alpha0_remap.yaml || true

echo "[queue-gpu1] $(date '+%F %T') ALL DONE"
