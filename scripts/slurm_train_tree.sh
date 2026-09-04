#!/bin/bash
#SBATCH --job-name=iomt-tree
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=9-00:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/training/tree_array_%A_%a.out
#SBATCH --error=logs/training/tree_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu
#
# ===========================================================================
# SLURM Array Job: Train all tree-based models on CICIoMT2024
# ===========================================================================
#
# Submits 12 independent jobs as a SLURM array:
#   4 models (RandomForest, XGBoost, LightGBM, CatBoost) × 3 tasks (2, 6, 19)
#
# Each job:
#   1. Loads preprocessed NPZ cache (must exist — run preprocessing tasks first)
#   2. Runs 50 Optuna HPO trials (TPE sampler) on the validation split
#   3. Retrains best config on train+val combined
#   4. Evaluates on held-out test set
#   5. Saves: results.json, model.pkl, confusion_matrix.png, all_results.csv
#
# Array index mapping:
#   Index | Model        | Task
#   ------+--------------+------
#     0   | RandomForest |  2
#     1   | RandomForest |  6
#     2   | RandomForest | 19
#     3   | XGBoost      |  2
#     4   | XGBoost      |  6
#     5   | XGBoost      | 19
#     6   | LightGBM     |  2
#     7   | LightGBM     |  6
#     8   | LightGBM     | 19
#     9   | CatBoost     |  2
#    10   | CatBoost     |  6
#    11   | CatBoost     | 19
#
# Memory: 256GB. The preprocessed data (NPZ) after SMOTEENN expansion can
# be large, especially for the 19-class task. 256GB gives comfortable headroom.
#
# CPUs: 16 cores. RandomForest and LightGBM use n_jobs=-1 (all cores).
# XGBoost also parallelizes internally. CatBoost uses its own threading.
# 16 cores gives strong parallel speedup for ensemble training.
#
# Time: 9 days max. HPO runs 50 trials; each trial trains a full model.
# The 19-class task on large SMOTEENN-expanded data may need multiple days.
#
# Log organization: SLURM writes initial logs to logs/tree_array_*.out,
# then the script copies them to logs/{ModelName}/task_{N}.out after
# determining which model/task this array index maps to.
#
# Prerequisites:
#   cd ~/dissertation
#   # Preprocessing must complete first:
#   sbatch scripts/slurm_preprocess_task2.sh
#   sbatch scripts/slurm_preprocess_task6.sh
#   sbatch scripts/slurm_preprocess_task19.sh
#
# Usage:
#   sbatch scripts/slurm_train_tree.sh
#
# To monitor:
#   squeue -u $USER
#   tail -f logs/training/{ModelName}/task_{N}.out
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
mkdir -p logs/training results

# --------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to (model, task)
# --------------------------------------------------------------------------
MODELS=("RandomForest" "XGBoost" "LightGBM" "CatBoost")
TASKS=(2 6 19)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

MODEL=${MODELS[$MODEL_IDX]}
TASK=${TASKS[$TASK_IDX]}

# --- Create model-specific log directory and redirect output ---
LOG_DIR="logs/training/${MODEL}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/task_${TASK}.out"

# Redirect all subsequent stdout and stderr to the organized log file
# (also keeps writing to the SLURM default log via tee)
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Provenance ---
echo "============================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Array index: $SLURM_ARRAY_TASK_ID"
echo "Python: $(which python3) ($(python3 --version))"
echo "============================================="

# Dependencies: pre-installed via one-time `pip install --user` on login node.
# See README or SLURM script header for the install command.

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Dataset: CICIoMT2024"
echo "Log file: $LOG_FILE"
echo "============================================="

# --------------------------------------------------------------------------
# Run the training pipeline
# --------------------------------------------------------------------------
python3 scripts/run_single.py \
    --model "$MODEL" \
    --dataset CICIoMT2024 \
    --task "$TASK" \
    --verbose

echo ""
echo "============================================="
echo "Job finished: $(date)"
echo "============================================="
