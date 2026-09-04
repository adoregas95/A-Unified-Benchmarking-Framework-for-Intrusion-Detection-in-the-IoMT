#!/bin/bash
#SBATCH --job-name=iomt-transformer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=9-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/training/transformer_array_%A_%a.out
#SBATCH --error=logs/training/transformer_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu
#
# ===========================================================================
# SLURM Array Job: Train transformer models on CICIoMT2024
# ===========================================================================
#
# Submits 6 independent jobs as a SLURM array:
#   2 models (FTTransformer, SAINT) × 3 tasks (2, 6, 19)
#
# Each job:
#   1. Loads preprocessed NPZ cache (must exist)
#   2. Runs 100 Optuna HPO trials (more than DL — larger search space)
#   3. Retrains best config on train+val combined
#   4. Evaluates on held-out test set
#   5. Saves: results.json, model checkpoint, confusion_matrix.png
#
# Array index mapping:
#   Index | Model         | Task
#   ------+---------------+------
#     0   | FTTransformer |  2
#     1   | FTTransformer |  6
#     2   | FTTransformer | 19
#     3   | SAINT         |  2
#     4   | SAINT         |  6
#     5   | SAINT         | 19
#
# IMPORTANT: Transformers receive ALL 76 features (no MI feature selection).
# This is handled at preprocessing time via config.yaml transformers_use_all.
# The preprocessed NPZ for transformers uses the full feature set.
#
# GPU: 1× A100 80GB. Transformer self-attention is memory-intensive;
# SAINT's intersample attention additionally scales with batch size.
# Mixed precision (AMP) is enabled automatically.
#
# Memory: 256GB system RAM.
#
# HPO budget: 100 trials (vs 50 for tree-based and DL). Transformers have
# a larger and more sensitive hyperparameter space (embed_dim × num_heads ×
# depth × two dropout rates × learning_rate × batch_size).
#
# Time: 9 days max. SAINT is the most expensive model due to intersample
# attention scaling quadratically with batch size.
#
# Prerequisites:
#   Preprocessing must complete first for all 3 tasks.
#
# Usage:
#   mkdir -p logs/training
#   sbatch scripts/slurm_train_transformer.sh
#
# ===========================================================================

set -euo pipefail

# --- Environment setup ---
# --- Portability: adjust these for your cluster ----------------------------
# PYTHON_BIN  Directory containing python3.11+. The value below is the one used
#             for the original runs on the SDSU HPC cluster. Override it by
#             exporting PYTHON_BIN, or replace this with `module load python/3.11`.
# PROJECT_ROOT Directory where you cloned this repository.
# Also review the #SBATCH partition, memory, GPU and walltime requests above,
# which are specific to the cluster these jobs were run on.
PYTHON_BIN="${PYTHON_BIN:-/mmfs1/cm/shared/apps_local/python/3.11/bin}"
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dissertation}"
# ---------------------------------------------------------------------------
export PATH="$PYTHON_BIN:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# --- Navigate to project root ---
cd "$PROJECT_ROOT"
mkdir -p logs/training results

# --------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to (model, task)
# --------------------------------------------------------------------------
MODELS=("FTTransformer" "SAINT")
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
