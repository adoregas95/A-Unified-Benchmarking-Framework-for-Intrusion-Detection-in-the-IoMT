#!/bin/bash
#SBATCH --job-name=iomt-xds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --array=0-47
#SBATCH --output=logs/cross_dataset/xds_array_%A_%a.out
#SBATCH --error=logs/cross_dataset/xds_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu
#
# ===========================================================================
# SLURM Array Job: Cross-Dataset Generalization (Zero-Shot)
# ===========================================================================
#
# Evaluates all 8 models on 3 cross-dataset targets × 2 tasks = 48 jobs.
#
# Array index mapping (48 jobs total):
#   model_idx = index / 6        (0-7: 8 models)
#   target_idx = (index % 6) / 2 (0-2: 3 targets)
#   task_idx = index % 2          (0-1: tasks 2 and 6)
#
#   Index | Model        | Target            | Task
#   ------+--------------+-------------------+------
#     0   | RandomForest | CIC-BoT-IoT       |  2
#     1   | RandomForest | CIC-BoT-IoT       |  6
#     2   | RandomForest | CIC-IoT-DIAD-2024 |  2
#     3   | RandomForest | CIC-IoT-DIAD-2024 |  6
#     4   | RandomForest | CIC-ToN-IoT       |  2
#     5   | RandomForest | CIC-ToN-IoT       |  6
#     6   | XGBoost      | CIC-BoT-IoT       |  2
#     ...
#    47   | SAINT        | CIC-ToN-IoT       |  6
#
# Time: 2 days. Cross-dataset is evaluation-only (no training), so much
# faster than training jobs. Main cost is data loading and preprocessing.
#
# Note: GPU is requested for all jobs because DL/Transformer models (CNN1D,
# BiLSTM, FTTransformer, SAINT) need CUDA for inference. Tree-based models
# (RandomForest, XGBoost, LightGBM, CatBoost) don't use the GPU but finish
# quickly, so the allocation overhead is minimal.
#
# Prerequisites:
#   - All training jobs must have completed on CICIoMT2024
#   - Preprocessing cache must exist for primary dataset
#   - Cross-dataset CSVs must exist in data/ directories
#
# Usage:
#   mkdir -p logs/cross_dataset
#   sbatch scripts/slurm_cross_dataset.sh
#
# ===========================================================================

set -euo pipefail

export PATH="/mmfs1/cm/shared/apps_local/python/3.11/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd ~/dissertation
mkdir -p logs/cross_dataset results

# --------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to (model, target, task)
# --------------------------------------------------------------------------
MODELS=("RandomForest" "XGBoost" "LightGBM" "CatBoost" "CNN1D" "BiLSTM" "FTTransformer" "SAINT")
TARGETS=("CIC-BoT-IoT" "CIC-IoT-DIAD-2024" "CIC-ToN-IoT")
TASKS=(2 6)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % 6 ))
TARGET_IDX=$(( REMAINDER / 2 ))
TASK_IDX=$(( REMAINDER % 2 ))

MODEL=${MODELS[$MODEL_IDX]}
TARGET=${TARGETS[$TARGET_IDX]}
TASK=${TASKS[$TASK_IDX]}

LOG_DIR="logs/cross_dataset/${MODEL}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TARGET}_task_${TASK}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
echo "============================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Array index: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Target: $TARGET"
echo "Task: $TASK"
echo "Transfer: zero-shot (no retraining)"
echo "Python: $(python3 --version 2>&1)"
echo "============================================="

# --------------------------------------------------------------------------
# Load PyTorch for DL/Transformer models (CNN1D, BiLSTM, FTTransformer, SAINT)
# --------------------------------------------------------------------------
DL_MODELS=("CNN1D" "BiLSTM" "FTTransformer" "SAINT")
NEEDS_GPU=false
for dm in "${DL_MODELS[@]}"; do
    if [ "$MODEL" == "$dm" ]; then
        NEEDS_GPU=true
        break
    fi
done

if [ "$NEEDS_GPU" = true ]; then
    echo "DL/Transformer model detected — loading PyTorch module..."
    set +eu
    module load pytorch/2.2.0
    set -eu
    echo ""
    nvidia-smi
    echo ""
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available after module load!'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
    echo ""
else
    echo "Tree-based model — no PyTorch module needed."
    echo ""
fi

# --------------------------------------------------------------------------
# Run cross-dataset evaluation
# --------------------------------------------------------------------------
CHECKPOINT="results/CICIoMT2024/task_${TASK}/${MODEL}/model.pkl"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Training must complete before cross-dataset evaluation."
    exit 1
fi

python3 scripts/run_cross_dataset.py \
    --model "$MODEL" \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --target "$TARGET" \
    --verbose

echo ""
echo "============================================="
echo "Job finished: $(date)"
echo "============================================="
