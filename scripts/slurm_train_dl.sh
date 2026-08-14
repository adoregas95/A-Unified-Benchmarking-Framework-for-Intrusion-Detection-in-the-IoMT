#!/bin/bash
#SBATCH --job-name=iomt-dl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=9-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/training/dl_array_%A_%a.out
#SBATCH --error=logs/training/dl_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oswald.adohinzin@trojans.dsu.edu
#
# ===========================================================================
# SLURM Array Job: Train deep learning models on CICIoMT2024
# ===========================================================================
#
# Submits 6 independent jobs as a SLURM array:
#   2 models (CNN1D, BiLSTM) × 3 tasks (2, 6, 19)
#
# Each job:
#   1. Loads preprocessed NPZ cache (must exist)
#   2. Runs 50 Optuna HPO trials with early stopping per trial
#   3. Retrains best config on train+val combined
#   4. Evaluates on held-out test set
#   5. Saves: results.json, model checkpoint, confusion_matrix.png
#
# Array index mapping:
#   Index | Model  | Task
#   ------+--------+------
#     0   | CNN1D  |  2
#     1   | CNN1D  |  6
#     2   | CNN1D  | 19
#     3   | BiLSTM |  2
#     4   | BiLSTM |  6
#     5   | BiLSTM | 19
#
# GPU: 1× A100 80GB. PyTorch models use CUDA for training and inference.
# Mixed precision (AMP) is enabled automatically when CUDA is available.
#
# Memory: 256GB system RAM. Loading SMOTEENN-expanded NPZ files into
# tensors plus model overhead requires substantial system memory.
#
# CPUs: 8 cores for DataLoader workers and data loading.
#
# Time: 9 days max. 50 HPO trials × up to 50 epochs each, with early
# stopping. The 19-class task is the most expensive.
#
# Prerequisites:
#   Preprocessing must complete first for all 3 tasks.
#
# Usage:
#   mkdir -p logs/training
#   sbatch scripts/slurm_train_dl.sh
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
MODELS=("CNN1D" "BiLSTM")
TASKS=(2 6 19)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

MODEL=${MODELS[$MODEL_IDX]}
TASK=${TASKS[$TASK_IDX]}

# --- Create model-specific log directory and redirect output ---
LOG_DIR="logs/training/${MODEL}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/task_${TASK}.out"

exec > >(tee -a "$LOG_FILE") 2>&1

# --- Provenance ---
echo "============================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Array index: $SLURM_ARRAY_TASK_ID"
echo "Python: $(which python3) ($(python3 --version))"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "============================================="

# Dependencies: pre-installed via one-time `pip install --user` on login node.
# PyTorch with CUDA comes from the cluster module system.
# Temporarily relax strict mode: module/conda activation scripts may
# reference unbound variables and return non-zero exit codes.
set +eu
module load pytorch/2.2.0
set -eu

# Verify CUDA-enabled PyTorch is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after module load!'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

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
