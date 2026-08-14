#!/bin/bash
#SBATCH --job-name=iomt-xai2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=9-00:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/explainability/xai_stage2_%A_%a.out
#SBATCH --error=logs/explainability/xai_stage2_%A_%a.err
#
# ===========================================================================
# SLURM Array Job: Explainability Stage 2 (Full SHAP + LIME)
# ===========================================================================
#
# Applied to ALL 8 models × 3 tasks = 24 jobs.
# Same model ordering as Stage 1 for consistency.
#
# Array index mapping (identical to Stage 1):
#   Index | Model         | Task
#   ------+---------------+------
#     0   | RandomForest  |  2
#     1   | RandomForest  |  6
#     2   | RandomForest  | 19
#     3   | XGBoost       |  2
#     ...  (model_idx = index / 3, task_idx = index % 3)
#    23   | SAINT         | 19
#
# Prerequisites:
#   - Stage 1 explainability must have completed for all 24 jobs
#   - Dependencies installed BEFORE submitting (run once on login node):
#       pip install --user 'traitlets>=5.14' 'IPython>=8.18' shap lime
#
# Usage:
#   mkdir -p logs/explainability
#   sbatch scripts/slurm_explainability_stage2.sh
#
# ===========================================================================

set -euo pipefail

# --- Environment setup ---
export PATH="/mmfs1/cm/shared/apps_local/python/3.11/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

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
LOG_FILE="${LOG_DIR}/stage2_task_${TASK}.out"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array index: $SLURM_ARRAY_TASK_ID"
echo "Python: $(which python3) ($(python3 --version))"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Stage: 2 (Full SHAP + LIME + Cross-Validation)"
echo "============================================="

# Verify SHAP and LIME are importable
python3 -c "import shap; import lime.lime_tabular; print(f'SHAP {shap.__version__}, LIME OK')" || {
    echo "ERROR: SHAP/LIME not importable. Run on login node first:"
    echo "  pip install --user 'traitlets>=5.14' 'IPython>=8.18' shap lime"
    exit 1
}

# --------------------------------------------------------------------------
# Run Stage 2: Full SHAP + LIME with cross-validation
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
from evaluation.explainability.lime_analysis import run_lime_analysis, generate_consistency_report

config = load_config()
cache_dir = os.path.join(os.path.expanduser('~/dissertation'), config['preprocessing']['cache_dir'])
random_state = config['preprocessing']['random_state']

model_name = '$MODEL'
task = $TASK

# Load data
X_train, _, _, _, X_test, y_test, p_meta = load_preprocessed_data(
    cache_dir, 'CICIoMT2024', task
)
feature_names = p_meta.get('feature_names', [f'f{i}' for i in range(X_test.shape[1])])
class_names = p_meta.get('label_classes', [str(i) for i in range(len(set(y_test)))])
input_dim = X_test.shape[1]
n_classes = len(class_names)

# Load model
checkpoint_path = os.path.join(
    os.path.expanduser('~/dissertation'),
    'results', 'CICIoMT2024', f'task_{task}', model_name, 'model.pkl',
)
model = get_model_instance(model_name, random_state, input_dim=input_dim, n_classes=n_classes)
model.load_checkpoint(checkpoint_path)

xai_dir = os.path.join(
    os.path.expanduser('~/dissertation'),
    'results', 'CICIoMT2024', f'task_{task}', model_name, 'explainability',
)

# Stage 2 SHAP (2000 samples, faithfulness + stability + detailed plots)
print('Running Stage 2 SHAP...')
shap_result = run_shap_analysis(
    model=model, model_name=model_name,
    X_test=X_test, feature_names=feature_names,
    output_dir=xai_dir, stage=2,
    y_test=y_test, class_names=class_names,
    random_state=random_state,
)
print(f'SHAP: faithfulness={shap_result[\"faithfulness_score\"]:.4f}, '
      f'stability={shap_result[\"stability_score\"]:.4f}')

# Align LIME data with the SHAP subsample so indices match
subsample_idx = shap_result.get('subsample_idx')
if subsample_idx is not None:
    import numpy as _np
    X_test_sub = X_test[subsample_idx]
    y_test_sub = y_test[subsample_idx]
    print(f'LIME will use SHAP subsample ({len(subsample_idx)} instances) for index alignment')
else:
    X_test_sub = X_test
    y_test_sub = y_test

# Stage 2 LIME with SHAP cross-validation
print('Running Stage 2 LIME...')
lime_result = run_lime_analysis(
    model=model, model_name=model_name,
    X_train=X_train, X_test=X_test_sub, y_test=y_test_sub,
    feature_names=feature_names, class_names=class_names,
    output_dir=xai_dir,
    shap_values=shap_result.get('shap_values'),
    random_state=random_state,
)

# Consistency report
generate_consistency_report(lime_result, model_name, xai_dir)

kl5 = lime_result['mean_consistency_l5']
kl10 = lime_result['mean_consistency_l10']
print(f'LIME: mean consistency kappa_l5={kl5 if kl5 is not None else \"N/A\"}, '
      f'kappa_l10={kl10 if kl10 is not None else \"N/A\"}')
print(f'Flagged instances: {lime_result[\"n_flagged\"]}')
"

echo ""
echo "============================================="
echo "Job finished: $(date)"
echo "============================================="
