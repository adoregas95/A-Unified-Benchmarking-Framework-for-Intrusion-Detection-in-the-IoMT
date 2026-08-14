# Explainability Phase — Cluster Instructions

**Created:** July 6, 2026
**Phase:** Two-stage XAI analysis (SHAP + LIME)
**Prerequisites:** All 24 training jobs completed, all model checkpoints present

---

## Overview

The explainability phase runs in two stages:

1. **Stage 1** — Lightweight SHAP global feature importance for ALL 8 models × 3 tasks (24 jobs). Produces faithfulness and stability scores for each model.
2. **Model selection** — Use Stage 1 XAI scores + performance + efficiency to pick the best model per family (tree, DL, transformer). This is a quick interactive step.
3. **Stage 2** — Full SHAP + LIME + cross-validation for the 3 selected models × 3 tasks (9 jobs). Produces detailed plots, instance-level explanations, and SHAP-LIME consistency reports.

---

## Step 1: Install dependencies

SSH into the cluster and run:

```bash
cd ~/dissertation
pip install --user shap lime
```

Verify:

```bash
python3 -c "import shap; print('SHAP', shap.__version__)"
python3 -c "import lime; print('LIME OK')"
```

---

## Step 2: Create log directory

```bash
mkdir -p ~/dissertation/logs/explainability
```

---

## Step 3: Submit Stage 1 (24 jobs)

```bash
cd ~/dissertation
sbatch scripts/slurm_explainability.sh
```

This submits an array job with indices 0-23 (8 models × 3 tasks). Each job:
- Loads the model checkpoint from `results/CICIoMT2024/task_{task}/{model}/model.pkl`
- Runs SHAP analysis with 500 samples (lightweight)
- Saves results to `results/CICIoMT2024/task_{task}/{model}/explainability/`
- Uses the appropriate SHAP explainer per family:
  - Tree-based (RF, XGBoost, LightGBM, CatBoost) → TreeSHAP
  - Deep learning (CNN1D, BiLSTM) → DeepSHAP
  - Transformers (FTTransformer, SAINT) → KernelSHAP

**Resource allocation:** 1 GPU, 8 CPUs, 256GB RAM, 9-day walltime per job.

**Expected runtime:** Tree-based models should finish in minutes. DL models in tens of minutes. KernelSHAP (transformers) may take several hours due to the model-agnostic approximation.

### Monitor progress

```bash
# Check job status
squeue -u $USER --name=iomt-xai

# Watch a specific job's output
tail -f logs/explainability/XGBoost/stage1_task_2.out

# Check how many have completed
ls results/CICIoMT2024/task_*/*/explainability/shap_stage1_*.json 2>/dev/null | wc -l
# (should reach 24 when all done)
```

---

## Step 4: Verify Stage 1 completion

Once all 24 jobs finish, verify every model produced output:

```bash
cd ~/dissertation
for task in 2 6 19; do
  echo "=== Task $task ==="
  for model in RandomForest XGBoost LightGBM CatBoost CNN1D BiLSTM FTTransformer SAINT; do
    if ls results/CICIoMT2024/task_${task}/${model}/explainability/shap_stage1_* 1>/dev/null 2>&1; then
      echo "  $model: OK"
    else
      echo "  $model: MISSING"
    fi
  done
done
```

If any say MISSING, check the error logs:

```bash
cat logs/explainability/{MODEL}/stage1_task_{TASK}.out
cat logs/explainability/xai_array_{JOBID}_{ARRAYID}.err
```

---

## Step 5: Run model selection (interactive)

This determines which 3 models go into Stage 2. Run interactively on the cluster (or in a short SLURM job):

```bash
cd ~/dissertation
python3 << 'EOF'
import sys, os, json, logging
sys.path.insert(0, os.path.expanduser('~/dissertation'))

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')

import numpy as np
import pandas as pd
from evaluation.model_selection import compute_final_scores, select_best_per_family

models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
          "CNN1D", "BiLSTM", "FTTransformer", "SAINT"]
tasks = [2, 6, 19]

print("=" * 70)
print("MODEL SELECTION: Performance + Efficiency + Stage 1 XAI")
print("=" * 70)

for task in tasks:
    print(f"\n{'='*70}")
    print(f"TASK {task}")
    print(f"{'='*70}")

    rows = []
    faith_scores = []
    stab_scores = []

    for m in models:
        result_path = os.path.expanduser(
            f'~/dissertation/results/CICIoMT2024/task_{task}/{m}/results.json')
        d = json.load(open(result_path))

        row = {'model': m}
        row['f1_weighted'] = d['test_metrics']['f1_weighted']

        # Efficiency metrics
        eff = d.get('efficiency', {})
        row['inference_latency_ms_per_sample'] = eff.get('inference_latency_ms_per_sample', -1)
        row['peak_memory_mb_inference'] = eff.get('peak_memory_mb_inference', -1)
        row['model_parameter_count'] = eff.get('model_parameter_count', -1)
        row['energy_joules_per_sample'] = eff.get('energy_joules_per_sample', -1)
        rows.append(row)

        # XAI scores from Stage 1
        xai_dir = os.path.expanduser(
            f'~/dissertation/results/CICIoMT2024/task_{task}/{m}/explainability/')
        xai_files = [f for f in os.listdir(xai_dir) if f.startswith('shap_stage1') and f.endswith('.json')]
        if xai_files:
            xai = json.load(open(os.path.join(xai_dir, xai_files[0])))
            faith_scores.append(xai.get('faithfulness_score', 0.0))
            stab_scores.append(xai.get('stability_score', 0.0))
        else:
            faith_scores.append(0.0)
            stab_scores.append(0.0)

    df = pd.DataFrame(rows)
    ranked = compute_final_scores(
        df,
        performance_metric='f1_weighted',
        faithfulness_scores=np.array(faith_scores),
        stability_scores=np.array(stab_scores),
    )

    print("\nRanked models:")
    for _, row in ranked.iterrows():
        print(f"  #{int(row['rank'])} {row['model']:16s} "
              f"final={row['final_score']:.4f} "
              f"(perf={row['perf_score']:.4f} eff={row['eff_score']:.4f} xai={row['xai_score']:.4f})")

    best = select_best_per_family(ranked)
    print(f"\nBest per family: {best}")

print("\n" + "=" * 70)
print("UPDATE slurm_explainability_stage2.sh WITH THESE VALUES:")
print("=" * 70)
print("Look at the 'Best per family' output above.")
print("If the same model wins across all tasks, use it.")
print("If different models win on different tasks, pick the one that")
print("wins on the majority of tasks, or the one with the highest mean score.")
EOF
```

This will print the ranked models per task and the best model per family. Note the winners.

---

## Step 6: Update Stage 2 script

Edit the Stage 2 SLURM script with the actual best models:

```bash
nano scripts/slurm_explainability_stage2.sh
```

Find these lines near the top (around line 57-59):

```bash
BEST_TREE="XGBoost"          # Edit after model selection
BEST_DL="CNN1D"              # Edit after model selection
BEST_TRANSFORMER="FTTransformer"  # Edit after model selection
```

Replace with the actual best models from Step 5. For example, if the selection says
XGBoost / BiLSTM / SAINT, change to:

```bash
BEST_TREE="XGBoost"
BEST_DL="BiLSTM"
BEST_TRANSFORMER="SAINT"
```

Save and exit (Ctrl+O, Enter, Ctrl+X in nano).

---

## Step 7: Submit Stage 2 (9 jobs)

```bash
cd ~/dissertation
sbatch scripts/slurm_explainability_stage2.sh
```

This submits an array job with indices 0-8 (3 families × 3 tasks). Each job:
- Runs full SHAP with 2000 samples (detailed analysis)
- Generates force/waterfall/dependence plots
- Runs LIME analysis
- Computes SHAP-LIME cross-validation (kappa at ℓ=5 and ℓ=10)
- Generates consistency report

**Expected runtime:** Longer than Stage 1 due to 4x more samples and LIME. KernelSHAP
models may take many hours.

### Monitor progress

```bash
squeue -u $USER --name=iomt-xai2
tail -f logs/explainability/{MODEL}/stage2_task_{TASK}.out
```

---

## Step 8: Verify Stage 2 completion

```bash
cd ~/dissertation
# Read the best models from the script
source <(grep 'BEST_' scripts/slurm_explainability_stage2.sh | head -3)

for task in 2 6 19; do
  echo "=== Task $task ==="
  for model in "$BEST_TREE" "$BEST_DL" "$BEST_TRANSFORMER"; do
    xai_dir="results/CICIoMT2024/task_${task}/${model}/explainability"
    echo "  $model:"
    echo "    SHAP Stage 2: $(ls ${xai_dir}/shap_stage2_* 2>/dev/null | wc -l) files"
    echo "    LIME:          $(ls ${xai_dir}/lime_* 2>/dev/null | wc -l) files"
    echo "    Consistency:   $(ls ${xai_dir}/consistency_* 2>/dev/null | wc -l) files"
    echo "    Plots:         $(ls ${xai_dir}/*.png 2>/dev/null | wc -l) files"
  done
done
```

---

## Step 9: Copy results to local machine

After both stages complete, download the explainability results:

```bash
# From your local machine (not the cluster):
scp -r cluster:~/dissertation/results/CICIoMT2024/task_*/*/explainability/ \
    ~/Downloads/dissertation/results/  # adjust path as needed
```

Or sync the entire results directory:

```bash
rsync -avz cluster:~/dissertation/results/ ~/Downloads/dissertation/results/
```

---

## What each stage produces

### Stage 1 output (per model, per task):
- `shap_stage1_results.json` — global feature importance ranking, faithfulness score, stability score, computation time
- `shap_global_importance.png` — bar plot of top features

### Stage 2 output (per selected model, per task):
- `shap_stage2_results.json` — detailed SHAP values, class-level importance, faithfulness, stability
- `shap_summary_plot.png` — beeswarm/violin plot
- `shap_dependence_*.png` — dependence plots for top features
- `shap_waterfall_*.png` — instance-level waterfall plots
- `lime_results.json` — LIME explanations, cross-validation kappa
- `lime_explanation_*.png` — instance-level LIME plots
- `consistency_report.json` — SHAP-LIME agreement analysis

---

## Troubleshooting

**SHAP import error:** Run `pip install --user shap lime` on the cluster.

**Out of memory:** KernelSHAP on transformers (SAINT especially with 61M params) may
need more RAM. If OOM occurs, you can reduce `KERNEL_SHAP_BACKGROUND_SIZE` in
`evaluation/explainability/shap_analysis.py` from 100 to 50.

**DeepSHAP fails on BiLSTM/CNN1D:** The code falls back to KernelSHAP automatically.
Check the log for "Falling back to KernelSHAP" messages.

**Job timeout (9 days):** KernelSHAP on SAINT Task 19 may be the longest. If it times
out, the Stage 1 script uses 500 samples which should be fast enough. Stage 2 with
2000 samples is more at risk — you can reduce `STAGE2_MAX_SAMPLES` in
`evaluation/explainability/shap_analysis.py` if needed.

---

## Preliminary model selection (performance-only, pre-XAI)

Based on f1_weighted alone (before XAI and efficiency are factored in):

| Family | Best Model | Mean F1 Weighted |
|--------|-----------|-----------------|
| Tree-based | XGBoost | 0.9786 |
| Deep learning | BiLSTM | 0.9772 |
| Transformers | SAINT | 0.9786 |

These are placeholders. The actual Stage 2 models will be determined by the full
composite scoring (Step 5) after Stage 1 XAI completes.
