#!/bin/bash
#SBATCH --job-name=iomt-xai
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=9-00:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/explainability/xai_array_%A_%a.out
#SBATCH --error=logs/explainability/xai_array_%A_%a.err
#
# ===========================================================================
# SLURM Array Job: Explainability Analysis (Stage 1 + Stage 2)
# ===========================================================================
#
# Stage 1 (indices 0-23): Lightweight SHAP global for ALL 8 models × 3 tasks
#   Array index mapping:
#     Index | Model         | Task | Stage
#     ------+---------------+------+-------
#       0   | RandomForest  |  2   |   1
#       1   | RandomForest  |  6   |   1
#       2   | RandomForest  | 19   |   1
#       3   | XGBoost       |  2   |   1
#       ...  (pattern: model_idx = index / 3, task_idx = index % 3)
#      23   | SAINT         | 19   |   1
#
# Stage 2 runs separately AFTER Stage 1 + model selection determines the
# top model per family. Use slurm_explainability_stage2.sh for that.
#
# Prerequisites:
#   - All training jobs must have completed (model checkpoints must exist)
#   - Preprocessing cache must exist
#   - Dependencies installed BEFORE submitting (run once on login node):
#       pip install --user 'traitlets>=5.14' 'IPython>=8.18' shap lime
#
# Usage:
#   mkdir -p logs/explainability
#   # Install deps ONCE on login node (not inside the job):
#   pip install --user 'traitlets>=5.14' 'IPython>=8.18' shap lime
#   sbatch scripts/slurm_explainability.sh
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
mkdir -p logs/explainability results

# --------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to (model, task)
# --------------------------------------------------------------------------
MODELS=("RandomForest" "XGBoost" "LightGBM" "CatBoost" "CNN1D" "BiLSTM" "FTTransformer" "SAINT")
TASKS=(2 6 19)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

MODEL=${MODELS[$MODEL_IDX]}
TASK=${TASKS[$TASK_IDX]}

# --- Create model-specific log directory and redirect output ---
LOG_DIR="logs/explainability/${MODEL}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage1_task_${TASK}.out"

exec > >(tee -a "$LOG_FILE") 2>&1

# --- Provenance ---
echo "============================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array index: $SLURM_ARRAY_TASK_ID"
echo "Python: $(which python3) ($(python3 --version))"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "============================================="

# Verify SHAP is importable (deps must be pre-installed on login node)
python3 -c "import shap; print(f'SHAP {shap.__version__} OK')" || {
    echo "ERROR: SHAP not importable. Run on login node first:"
    echo "  pip install --user 'traitlets>=5.14' 'IPython>=8.18' shap lime"
    exit 1
}

echo "Model: $MODEL"
echo "Task: $TASK"
echo "Stage: 1 (SHAP Global)"
echo "============================================="

# --------------------------------------------------------------------------
# Run Stage 1 SHAP analysis
# --------------------------------------------------------------------------
python3 -c "
import sys, os, logging
sys.path.insert(0, os.path.expanduser('~/dissertation'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
)

from training.train import get_model_instance, load_config
from preprocessing.preprocessing import load_preprocessed_data
from evaluation.explainability.shap_analysis import run_shap_analysis

config = load_config()
cache_dir = os.path.join(os.path.expanduser('~/dissertation'), config['preprocessing']['cache_dir'])
random_state = config['preprocessing']['random_state']

model_name = '$MODEL'
task = $TASK

# Load preprocessed data
X_train, _, _, _, X_test, y_test, p_meta = load_preprocessed_data(
    cache_dir, 'CICIoMT2024', task
)
feature_names = p_meta.get('feature_names', [f'f{i}' for i in range(X_test.shape[1])])
input_dim = X_test.shape[1]
n_classes = len(p_meta.get('label_classes', []))

# Load model
checkpoint_path = os.path.join(
    os.path.expanduser('~/dissertation'),
    'results', 'CICIoMT2024', f'task_{task}', model_name, 'model.pkl',
)
model = get_model_instance(model_name, random_state, input_dim=input_dim, n_classes=n_classes)
model.load_checkpoint(checkpoint_path)

# Run Stage 1 SHAP
xai_dir = os.path.join(
    os.path.expanduser('~/dissertation'),
    'results', 'CICIoMT2024', f'task_{task}', model_name, 'explainability',
)
result = run_shap_analysis(
    model=model, model_name=model_name,
    X_test=X_test, feature_names=feature_names,
    output_dir=xai_dir, stage=1, random_state=random_state,
)
print(f'Top 5 features: {result[\"global_importance\"][:5]}')
print(f'Computation time: {result[\"computation_time_seconds\"]:.1f}s')
"

echo ""
echo "============================================="
echo "Job finished: $(date)"
echo "============================================="
