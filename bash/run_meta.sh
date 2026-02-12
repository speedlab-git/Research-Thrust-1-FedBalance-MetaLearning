#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_meta_fed_ucfcrime_r50.sh
# Runs: meta_fed_ucfcrime_resnet50_wandb_trainfrac.py
# ============================================================

# ---- EDIT THESE PATHS ----
TRAIN_DIR="/mnt/beegfs/home/mdzarifhossa2025/mdhossa/NSF/ucf-crime-dataset/Train"
TEST_DIR="/mnt/beegfs/home/mdzarifhossa2025/mdhossa/NSF/ucf-crime-dataset/Test"

# ---- EXPERIMENT CONFIG ----
NUM_CLIENTS=5
ROUNDS=100
BATCH_SIZE=64
IMG_SIZE=224

# Meta-learning hyperparams
OUTER_LR=1e-3     # server/meta update lr
INNER_LR=1e-2     # client adaptation lr
INNER_STEPS=5
SUPPORT_RATIO=0.5

# Dataset balancing controls
NORMAL_CLASS_NAME="NormalVideos"
NORMAL_CAP=80000
TRAIN_FRAC=0.4

# W&B logging
USE_WANDB=1
WANDB_PROJECT="ucfcrime-meta-fed"
WANDB_RUN_NAME="meta_fed_r50_iid_20pct"

# Save root folder
SAVE_ROOT="meta_fed"

# Optional: freeze backbone (set to 1 to enable)
FREEZE_BACKBONE=0

# Workers
NUM_WORKERS=2

# ---- BUILD CMD ----
CMD=(python fcrimnet_meta.py
  --train_dir "${TRAIN_DIR}"
  --test_dir "${TEST_DIR}"
  --img_size "${IMG_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --num_clients "${NUM_CLIENTS}"
  --rounds "${ROUNDS}"
  --outer_lr "${OUTER_LR}"
  --inner_lr "${INNER_LR}"
  --inner_steps "${INNER_STEPS}"
  --support_ratio "${SUPPORT_RATIO}"
  --normal_class_name "${NORMAL_CLASS_NAME}"
  --normal_cap "${NORMAL_CAP}"
  --train_frac "${TRAIN_FRAC}"
  --num_workers "${NUM_WORKERS}"
  --save_root "${SAVE_ROOT}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
)

if [[ "${USE_WANDB}" -eq 1 ]]; then
  CMD+=(--use_wandb)
fi

if [[ "${FREEZE_BACKBONE}" -eq 1 ]]; then
  CMD+=(--freeze_backbone)
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"
