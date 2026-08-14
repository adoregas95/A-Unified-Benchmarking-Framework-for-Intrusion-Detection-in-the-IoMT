#!/bin/bash
#SBATCH --job-name=preproc-task19
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=9-00:00:00
#SBATCH --output=logs/preprocessing/task_19_%j.out
#SBATCH --error=logs/preprocessing/task_19_%j.err
#
# ===========================================================================
# SLURM Job: Preprocess CICIoMT2024 — Task 19 (19-Class Individual Attacks)
# ===========================================================================
#
# Preprocessing pipeline:
#   1. Load ALL CSVs (train + test files merged)
#   2. Stratified 80/20 re-split (proportional class representation)
#   3. Clean Inf/NaN → RobustScaler → Cumulative MI feature selection (90%)
#   4. SMOTEENN (SMOTE oversampling + Edited Nearest Neighbors cleanup)
#   5. Cache to NPZ
#
# GPU partition: 512GB RAM to handle SMOTEENN's k-NN queries on large data.
# Task 19 is the largest (19 classes after SMOTE = most synthetic samples).
# 9-day walltime: SMOTEENN boundary cleanup is O(n²) — generous time.
#
# Usage:
#   cd ~/dissertation
#   mkdir -p logs/preprocessing
#   sbatch scripts/slurm_preprocess_task19.sh
#
# ===========================================================================

set -euo pipefail

# --- Environment setup ---
export PATH="/mmfs1/cm/shared/apps_local/python/3.11/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# --- Navigate to project root ---
cd ~/dissertation
mkdir -p logs/preprocessing

# --- Provenance ---
echo "============================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Python: $(which python3) ($(python3 --version))"
echo "Working dir: $(pwd)"
echo "Task: 19 (19-Class Individual Attacks)"
echo "============================================="

# Install required packages if not already present
pip install --user scikit-learn imbalanced-learn pyyaml numpy pandas 2>&1 | tail -1

# --- Run preprocessing for Task 19 only ---
python3 scripts/run_preprocessing.py --dataset CICIoMT2024 --task 19 --verbose --force

echo ""
echo "============================================="
echo "Job finished: $(date)"
echo "============================================="

# Show what was cached
echo ""
echo "Cached files:"
find preprocessing/cache -name "*.npz" -exec ls -lh {} \; 2>/dev/null || echo "  (no cache files found)"
