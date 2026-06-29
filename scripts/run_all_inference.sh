#!/bin/bash
# Runs UBFC-rPPG first (smaller, validates pipeline), then MMPD.
# Sequential, not parallel - respects Colab/Kaggle session/GPU memory limits.
set -e

CONFIGS=(
  "configs/infer_configs/UBFC_DEEPPHYS.yaml"
  "configs/infer_configs/UBFC_PHYSNET.yaml"
  "configs/infer_configs/UBFC_EFFICIENTPHYS.yaml"
  "configs/infer_configs/MMPD_DEEPPHYS.yaml"
  "configs/infer_configs/MMPD_PHYSNET.yaml"
  "configs/infer_configs/MMPD_EFFICIENTPHYS.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running $cfg ==="
  if [ ! -f "$cfg" ]; then
    echo "WARNING: $cfg not found yet - skipping. Create this config before the full run."
    continue
  fi
  python rPPG-Toolbox/main.py --config_file "$cfg"
  echo "=== Done: $cfg ==="
done

echo "All configured runs complete."
