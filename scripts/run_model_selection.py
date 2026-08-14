#!/usr/bin/env python3
"""
MCDM Model Selection Runner — Per-Task Rankings.

Collects performance, efficiency, and explainability data from results.json
and Stage 2 explainability logs, then runs the composite MCDM scoring.

Usage (on cluster login node — no GPU needed):
    cd ~/dissertation
    python3 scripts/run_model_selection.py

Output:
    results/model_selection_results.json  — structured results for all tasks
    Console output with rankings, sensitivity analysis, and best-per-family
"""

import json
import os
import re
import sys
import glob
import logging

import numpy as np
import pandas as pd

# --- Project setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.model_selection import (
    compute_final_scores,
    select_best_per_family,
    run_sensitivity_analysis,
    DEFAULT_WEIGHTS,
    PERFORMANCE_METRICS,
    EFFICIENCY_METRICS,
)

logging.basicConfig(level=logging.WARNING)  # Suppress library chatter
logger = logging.getLogger(__name__)

# --- Configuration ---
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "CICIoMT2024")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "explainability")
MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
          "CNN1D", "BiLSTM", "FTTransformer", "SAINT"]
TASKS = [2, 6, 19]
TASK_LABELS = {2: "Binary (2-class)", 6: "6-class", 19: "19-class"}


# =========================================================================
# Data collection
# =========================================================================

def load_results_json(model: str, task: int) -> dict:
    """Load performance and efficiency metrics from results.json."""
    path = os.path.join(RESULTS_DIR, f"task_{task}", model, "results.json")
    with open(path) as f:
        data = json.load(f)

    row = {"model": model, "task": task}

    # Performance metrics
    tm = data.get("test_metrics", {})
    for metric in PERFORMANCE_METRICS:
        row[metric] = tm.get(metric, 0.0)
    # Extra metrics for display
    row["accuracy"] = tm.get("accuracy", 0.0)
    row["f1_macro"] = tm.get("f1_macro", 0.0)

    # Efficiency metrics
    eff = data.get("efficiency", {})
    for metric in EFFICIENCY_METRICS:
        row[metric] = eff.get(metric, -1.0)
    row["training_time_s"] = eff.get("training_time_seconds", -1.0)

    return row


def parse_xai_from_logs(model: str, task: int) -> dict:
    """Extract faithfulness, stability, and LIME kappa from Stage 2 logs."""
    result = {"faithfulness": None, "stability": None,
              "kappa_l5": None, "kappa_l10": None, "n_flagged": None}

    # Try per-model log first
    log_path = os.path.join(LOGS_DIR, model, f"stage2_task_{task}.out")
    if not os.path.exists(log_path):
        # Fall back to SLURM array log
        model_idx = MODELS.index(model)
        task_idx = TASKS.index(task)
        array_id = model_idx * 3 + task_idx
        matches = sorted(glob.glob(
            os.path.join(LOGS_DIR, f"xai_stage2_*_{array_id}.out")
        ))
        if matches:
            log_path = matches[-1]
        else:
            return result

    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"SHAP:\s*faithfulness=([-\d.]+),\s*stability=([-\d.]+)", line
            )
            if m:
                result["faithfulness"] = float(m.group(1))
                result["stability"] = float(m.group(2))
            m = re.search(r"kappa_l5=([-\d.eE+]+)", line)
            if m:
                result["kappa_l5"] = float(m.group(1))
            m = re.search(r"kappa_l10=([-\d.eE+]+)", line)
            if m:
                result["kappa_l10"] = float(m.group(1))
            m = re.search(r"Flagged instances:\s*(\d+)", line)
            if m:
                result["n_flagged"] = int(m.group(1))

    return result


# =========================================================================
# Main
# =========================================================================

def main():
    all_task_results = {}
    all_data_rows = []

    print("=" * 90)
    print("MCDM MODEL SELECTION — Per-Task Rankings")
    print(f"Performance composite: {', '.join(PERFORMANCE_METRICS)}")
    print(f"Efficiency composite:  {', '.join(EFFICIENCY_METRICS)}")
    print(f"Explainability composite: faithfulness, stability")
    print(f"Weights: Perf={DEFAULT_WEIGHTS['performance']:.0%}, "
          f"Eff={DEFAULT_WEIGHTS['efficiency']:.0%}, "
          f"XAI={DEFAULT_WEIGHTS['explainability']:.0%}")
    print("=" * 90)

    for task in TASKS:
        print(f"\n{'=' * 90}")
        print(f"TASK {task} ({TASK_LABELS.get(task, '')})")
        print(f"{'=' * 90}")

        rows = []
        faith_list, stab_list = [], []

        for model in MODELS:
            row = load_results_json(model, task)
            xai = parse_xai_from_logs(model, task)
            row.update(xai)
            faith_list.append(xai["faithfulness"] if xai["faithfulness"] is not None else 0.0)
            stab_list.append(xai["stability"] if xai["stability"] is not None else 0.0)
            rows.append(row)
            all_data_rows.append(row)

        df = pd.DataFrame(rows)
        faith_arr = np.array(faith_list)
        stab_arr = np.array(stab_list)

        # --- Raw data table ---
        print(f"\nRaw metrics:")
        print(f"  {'Model':<16} {'F1_w':>7} {'Rec_w':>7} {'MCC':>7} "
              f"{'Latency':>10} {'Mem_MB':>9} {'Params':>11} "
              f"{'Faith':>8} {'Stab':>7} {'κ_l5':>7} {'Flag':>5}")
        print(f"  {'-' * 107}")
        for _, r in df.iterrows():
            lat = f"{r['inference_latency_ms_per_sample']:.6f}" if r['inference_latency_ms_per_sample'] >= 0 else "N/A"
            mem = f"{r['peak_memory_mb_inference']:.1f}" if r['peak_memory_mb_inference'] >= 0 else "N/A"
            par = f"{int(r['model_parameter_count']):,}" if r['model_parameter_count'] >= 0 else "N/A"
            fa = f"{r['faithfulness']:.4f}" if r['faithfulness'] is not None else "N/A"
            st = f"{r['stability']:.4f}" if r['stability'] is not None else "N/A"
            k5 = f"{r['kappa_l5']:.3f}" if r['kappa_l5'] is not None else "N/A"
            nf = str(r['n_flagged']) if r['n_flagged'] is not None else "N/A"
            print(f"  {r['model']:<16} {r['f1_weighted']:>7.4f} {r['recall_weighted']:>7.4f} "
                  f"{r['mcc']:>7.4f} {lat:>10} {mem:>9} {par:>11} "
                  f"{fa:>8} {st:>7} {k5:>7} {nf:>5}")

        # --- MCDM rankings ---
        ranked_df = compute_final_scores(df, faith_arr, stab_arr)

        print(f"\n  MCDM Rankings (Task {task}):")
        print(f"  {'Rank':>4} {'Model':<16} {'Final':>8} {'Perf':>8} {'Eff':>8} {'XAI':>8}")
        print(f"  {'-' * 56}")
        for _, r in ranked_df.iterrows():
            print(f"  {int(r['rank']):>4} {r['model']:<16} {r['final_score']:>8.4f} "
                  f"{r['perf_score']:>8.4f} {r['eff_score']:>8.4f} {r['xai_score']:>8.4f}")

        # --- Best per family ---
        best = select_best_per_family(ranked_df)
        print(f"\n  Best per family:")
        for fam in ["tree_based", "deep_learning", "transformers"]:
            if fam in best:
                print(f"    {fam}: {best[fam]}")

        # --- Sensitivity analysis ---
        sens_df = run_sensitivity_analysis(df, faith_arr, stab_arr)
        baseline_top = ranked_df.sort_values("rank").iloc[0]["model"]
        n_rank_stable = sens_df["ranking_matches_baseline"].sum()
        n_top_stable = (sens_df["top_model"] == baseline_top).sum()
        n_total = len(sens_df)
        print(f"\n  Sensitivity analysis:")
        print(f"    Top model stable in {n_top_stable}/{n_total} scenarios")
        print(f"    Full ranking stable in {n_rank_stable}/{n_total} scenarios")
        for _, s in sens_df.iterrows():
            rank_match = "SAME" if s["ranking_matches_baseline"] else "CHANGED"
            top_flag = "" if s["top_model"] == baseline_top else " *NEW TOP*"
            print(f"    {s['scenario']:12s}  "
                  f"w=({s['w_perf']:.2f}/{s['w_eff']:.2f}/{s['w_xai']:.2f})  "
                  f"top={s['top_model']:<16}  [{rank_match}]{top_flag}")

        # Store for JSON
        all_task_results[f"task_{task}"] = {
            "task_label": TASK_LABELS.get(task, f"task_{task}"),
            "rankings": ranked_df[
                ["model", "rank", "final_score", "perf_score", "eff_score", "xai_score"]
            ].to_dict(orient="records"),
            "best_per_family": best,
            "sensitivity": {
                "n_top_model_stable": int(n_top_stable),
                "n_full_ranking_stable": int(n_rank_stable),
                "n_total": int(n_total),
                "scenarios": sens_df.to_dict(orient="records"),
            },
        }

    # --- Save structured output ---
    output = {
        "description": "MCDM Model Selection — Per-Task Rankings",
        "weights": DEFAULT_WEIGHTS,
        "performance_metrics": PERFORMANCE_METRICS,
        "efficiency_metrics": EFFICIENCY_METRICS,
        "explainability_metrics": ["faithfulness", "stability"],
        "tasks": all_task_results,
        "raw_data": all_data_rows,
    }

    out_path = os.path.join(PROJECT_ROOT, "results", "model_selection_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{'=' * 90}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
