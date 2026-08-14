#!/usr/bin/env python3
"""
Generate final report from all experimental results.

Reads the master CSV files (all_results.csv, cross_dataset_results.csv)
and produces:
    1. Per-task leaderboards (ranked by primary metric)
    2. Cross-model comparison bar charts
    3. Efficiency vs performance scatter plots
    4. Cross-dataset generalization heatmaps
    5. Multi-criteria model selection summary
    6. Combined LaTeX-ready tables

Usage:
    # Generate full report after all experiments:
    python scripts/generate_report.py

    # Custom input/output:
    python scripts/generate_report.py --results-dir results --output-dir reports

    # Generate only specific sections:
    python scripts/generate_report.py --sections leaderboards plots
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    "RandomForest", "XGBoost", "LightGBM", "CatBoost",
    "CNN1D", "BiLSTM", "FTTransformer", "SAINT",
]
MODEL_FAMILY = {
    "RandomForest": "Tree-Based", "XGBoost": "Tree-Based",
    "LightGBM": "Tree-Based", "CatBoost": "Tree-Based",
    "CNN1D": "Deep Learning", "BiLSTM": "Deep Learning",
    "FTTransformer": "Transformer", "SAINT": "Transformer",
}
FAMILY_COLORS = {
    "Tree-Based": "#2196F3",
    "Deep Learning": "#FF9800",
    "Transformer": "#4CAF50",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def compile_results_from_json(results_dir: str) -> pd.DataFrame:
    """Compile results from individual results.json files into a single DataFrame.

    Scans results/CICIoMT2024/task_*/*/results.json for all 24 model-task
    combinations. Extracts test metrics and efficiency fields, building a
    DataFrame that matches the all_results.csv schema expected by existing
    functions. Saves the compiled CSV for future use.

    Args:
        results_dir: Root results directory containing CICIoMT2024/ subfolder.

    Returns:
        DataFrame with columns: model, task, f1_weighted, f1_macro, accuracy,
        mcc, precision_weighted, recall_weighted, inference_latency_ms,
        peak_memory_mb_inference, model_parameter_count, energy_joules_per_sample,
        training_time_s.
    """
    pattern = os.path.join(results_dir, "CICIoMT2024", "task_*", "*", "results.json")
    json_files = glob.glob(pattern)

    if not json_files:
        logger.error(f"No results.json files found matching: {pattern}")
        return pd.DataFrame()

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {jf}: {e}")
            continue

        # Extract task number and model name from path
        parts = Path(jf).parts
        # Expect: .../CICIoMT2024/task_N/ModelName/results.json
        model_name = parts[-2]
        task_dir = parts[-3]  # e.g., "task_2"
        try:
            task_num = int(task_dir.split("_")[1])
        except (IndexError, ValueError):
            logger.warning(f"Cannot parse task number from {task_dir}")
            continue

        # Extract test metrics
        test_metrics = data.get("test_metrics", {})
        efficiency = data.get("efficiency", {})

        row = {
            "model": model_name,
            "task": task_num,
            "f1_weighted": test_metrics.get("f1_weighted"),
            "f1_macro": test_metrics.get("f1_macro"),
            "accuracy": test_metrics.get("accuracy"),
            "mcc": test_metrics.get("mcc"),
            "precision_weighted": test_metrics.get("precision_weighted"),
            "recall_weighted": test_metrics.get("recall_weighted"),
            "inference_latency_ms": efficiency.get(
                "inference_latency_ms_per_sample",
                test_metrics.get("inference_latency_ms"),
            ),
            "peak_memory_mb_inference": efficiency.get(
                "peak_memory_mb_inference",
                test_metrics.get("peak_memory_mb_inference"),
            ),
            "model_parameter_count": efficiency.get(
                "model_parameter_count",
                data.get("model_parameter_count"),
            ),
            "energy_joules_per_sample": efficiency.get("energy_joules_per_sample"),
            "training_time_s": efficiency.get(
                "training_time_seconds",
                efficiency.get("training_time_s",
                               data.get("training_time_s")),
            ),
        }
        rows.append(row)

    if not rows:
        logger.error("No valid results extracted from JSON files")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Compiled {len(df)} results from {len(json_files)} JSON files")

    # Save compiled CSV for future use
    csv_path = os.path.join(results_dir, "all_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Saved compiled results to {csv_path}")

    return df


def load_primary_results(results_dir: str) -> pd.DataFrame:
    """Load primary training results from all_results.csv.

    Falls back to compiling results from individual results.json files
    if the CSV does not exist.
    """
    csv_path = os.path.join(results_dir, "all_results.csv")
    if not os.path.exists(csv_path):
        logger.warning(
            f"Primary CSV not found: {csv_path}. "
            "Attempting to compile from individual results.json files..."
        )
        return compile_results_from_json(results_dir)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} primary results from {csv_path}")
    return df


def load_cross_dataset_results(results_dir: str) -> pd.DataFrame:
    """Load cross-dataset results."""
    csv_path = os.path.join(results_dir, "cross_dataset", "cross_dataset_results.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"Cross-dataset results not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} cross-dataset results from {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------

def generate_leaderboards(
    df: pd.DataFrame,
    output_dir: str,
) -> Dict[int, pd.DataFrame]:
    """Generate per-task leaderboard tables ranked by primary metric.

    For each task, creates a table with all 8 models ranked by weighted F1,
    showing all metrics in a dissertation-ready format.

    Args:
        df: Primary results DataFrame.
        output_dir: Where to save CSV and LaTeX tables.

    Returns:
        Dict mapping task → leaderboard DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    leaderboards = {}

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task].copy()

        # Define columns to display
        display_cols = [
            "model", "f1_weighted", "f1_macro", "accuracy", "mcc",
            "precision_weighted", "recall_weighted",
            "inference_latency_ms", "peak_memory_mb_inference",
            "model_parameter_count",
        ]
        available = [c for c in display_cols if c in task_df.columns]
        leaderboard = task_df[available].copy()

        # Sort by weighted F1 (primary metric for CICIoMT2024)
        leaderboard = leaderboard.sort_values("f1_weighted", ascending=False)
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

        # Add family column
        leaderboard.insert(
            1, "family",
            leaderboard["model"].map(MODEL_FAMILY),
        )

        leaderboards[task] = leaderboard

        # Save CSV
        csv_path = os.path.join(output_dir, f"leaderboard_task{task}.csv")
        leaderboard.to_csv(csv_path, index=False, float_format="%.4f")
        logger.info(f"Leaderboard saved: {csv_path}")

        # Save LaTeX
        latex_path = os.path.join(output_dir, f"leaderboard_task{task}.tex")
        _save_latex_table(leaderboard, latex_path, f"Task {task} Leaderboard")

        # Log top 3
        logger.info(f"\nTask {task} — Top 3:")
        for _, row in leaderboard.head(3).iterrows():
            logger.info(
                f"  #{int(row['rank'])} {row['model']:20s} "
                f"F1(w)={row['f1_weighted']:.4f} "
                f"F1(m)={row.get('f1_macro', 0):.4f} "
                f"MCC={row.get('mcc', 0):.4f}"
            )

    return leaderboards


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def generate_comparison_plots(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate cross-model comparison bar charts.

    Creates grouped bar charts comparing all 8 models on key metrics
    for each task.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = [
        ("f1_weighted", "Weighted F1"),
        ("f1_macro", "Macro F1"),
        ("mcc", "MCC"),
        ("accuracy", "Accuracy"),
    ]

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task].copy()

        # Order models consistently
        task_df["model"] = pd.Categorical(
            task_df["model"], categories=MODEL_ORDER, ordered=True,
        )
        task_df = task_df.sort_values("model")

        for metric, label in metrics_to_plot:
            if metric not in task_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = [FAMILY_COLORS.get(MODEL_FAMILY.get(m, ""), "#999999")
                      for m in task_df["model"]]
            bars = ax.bar(task_df["model"], task_df[metric], color=colors)

            ax.set_ylabel(label)
            ax.set_title(f"Task {task} — {label} by Model")
            ax.set_ylim(bottom=max(0, task_df[metric].min() - 0.05))
            plt.xticks(rotation=30, ha="right")

            # Add value labels on bars
            for bar, val in zip(bars, task_df[metric]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

            # Legend for families
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()
            ]
            ax.legend(handles=legend_elements, loc="lower right")

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"task{task}_{metric}_comparison.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot saved: {plot_path}")


def generate_efficiency_plots(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate efficiency vs performance scatter plots.

    X-axis: inference latency (log scale) or model size
    Y-axis: weighted F1
    Color: model family
    Size: model parameter count
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    efficiency_metrics = [
        ("inference_latency_ms", "Inference Latency (ms/sample)", True),
        ("peak_memory_mb_inference", "Peak Memory Inference (MB)", True),
        ("model_parameter_count", "Model Parameters", True),
    ]

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task].copy()

        for eff_col, eff_label, use_log in efficiency_metrics:
            if eff_col not in task_df.columns or "f1_weighted" not in task_df.columns:
                continue

            # Drop rows with missing efficiency data
            plot_df = task_df.dropna(subset=[eff_col, "f1_weighted"])
            if plot_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 7))

            for family, color in FAMILY_COLORS.items():
                family_df = plot_df[plot_df["model"].map(MODEL_FAMILY) == family]
                if family_df.empty:
                    continue

                ax.scatter(
                    family_df[eff_col],
                    family_df["f1_weighted"],
                    c=color,
                    label=family,
                    s=100,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=0.5,
                )

                # Label each point with model name
                for _, row in family_df.iterrows():
                    ax.annotate(
                        row["model"],
                        (row[eff_col], row["f1_weighted"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )

            if use_log and plot_df[eff_col].min() > 0:
                ax.set_xscale("log")

            ax.set_xlabel(eff_label)
            ax.set_ylabel("Weighted F1")
            ax.set_title(f"Task {task} — Efficiency vs Performance")
            ax.legend()
            plt.tight_layout()

            plot_path = os.path.join(
                output_dir,
                f"task{task}_efficiency_{eff_col}.png",
            )
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Efficiency plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Cross-dataset heatmaps
# ---------------------------------------------------------------------------

def generate_cross_dataset_heatmap(
    cd_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate heatmaps of cross-dataset F1(macro) performance.

    Rows: models, Columns: target datasets
    Values: macro F1 score (color-mapped)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    if cd_df.empty:
        logger.warning("No cross-dataset results — skipping heatmap")
        return

    for task in sorted(cd_df["task"].unique()):
        task_df = cd_df[cd_df["task"] == task]

        # Pivot: rows=model, cols=target_dataset, values=f1_macro
        pivot = task_df.pivot_table(
            index="model",
            columns="target_dataset",
            values="f1_macro",
            aggfunc="first",
        )

        # Reorder rows to standard model order
        present_models = [m for m in MODEL_ORDER if m in pivot.index]
        pivot = pivot.loc[present_models]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"Cross-Dataset Generalization — Task {task} (Macro F1)")
        ax.set_ylabel("Model (trained on CICIoMT2024)")
        ax.set_xlabel("Target Dataset")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"cross_dataset_heatmap_task{task}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Cross-dataset heatmap saved: {plot_path}")

        # Also save as CSV
        csv_path = os.path.join(output_dir, f"cross_dataset_pivot_task{task}.csv")
        pivot.to_csv(csv_path, float_format="%.4f")


# ---------------------------------------------------------------------------
# MCDM composite score chart
# ---------------------------------------------------------------------------

def _apply_plot_style():
    """Apply consistent matplotlib style for dissertation figures."""
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")


def generate_mcdm_chart(results_dir: str, output_dir: str) -> None:
    """Generate MCDM composite score grouped bar charts for each task.

    Reads model_selection_results.json and creates grouped bar charts
    showing perf_score, eff_score, and xai_score side by side for each
    model, ordered by final_score rank. A line with markers overlays the
    final composite score.

    Args:
        results_dir: Root results directory.
        output_dir: Where to save the chart PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    ms_path = os.path.join(results_dir, "model_selection_results.json")
    if not os.path.exists(ms_path):
        logger.warning(f"Model selection results not found: {ms_path}")
        return

    with open(ms_path, "r") as f:
        ms_data = json.load(f)

    # model_selection_results.json may have per-task keys or a combined structure
    # Try task-keyed structure first, fall back to raw_data
    task_results = {}
    if "tasks" in ms_data:
        task_results = ms_data["tasks"]
    elif "raw_data" in ms_data:
        # raw_data is a list of records; group by task
        raw = ms_data["raw_data"]
        if isinstance(raw, list):
            for rec in raw:
                t = rec.get("task")
                if t is not None:
                    task_results.setdefault(str(t), []).append(rec)
        elif isinstance(raw, dict):
            task_results = raw
    else:
        # Top-level might be task keys directly
        for key in ms_data:
            if key.startswith("task") or key.isdigit():
                task_results[key] = ms_data[key]

    if not task_results:
        logger.warning("Could not parse task data from model_selection_results.json")
        return

    for task_key, task_data in task_results.items():
        # Normalize task number
        task_num = str(task_key).replace("task_", "").replace("task", "")

        # Convert to DataFrame if list of dicts
        if isinstance(task_data, list):
            tdf = pd.DataFrame(task_data)
        elif isinstance(task_data, dict) and "models" in task_data:
            tdf = pd.DataFrame(task_data["models"])
        elif isinstance(task_data, dict) and "rankings" in task_data:
            tdf = pd.DataFrame(task_data["rankings"])
        else:
            logger.warning(f"Unexpected structure for task {task_key}")
            continue

        # Ensure required columns exist
        score_cols = ["perf_score", "eff_score", "xai_score", "final_score"]
        missing = [c for c in score_cols if c not in tdf.columns]
        if missing:
            logger.warning(
                f"Task {task_key}: missing columns {missing}. "
                f"Available: {list(tdf.columns)}"
            )
            continue

        # Sort by final_score descending (rank 1 = highest)
        tdf = tdf.sort_values("final_score", ascending=False).reset_index(drop=True)
        models = tdf["model"].tolist()

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(models))
        width = 0.22

        components = [
            ("perf_score", "Performance (60%)", "#2196F3"),
            ("eff_score", "Efficiency (25%)", "#FF9800"),
            ("xai_score", "Explainability (15%)", "#4CAF50"),
        ]

        for i, (col, label, color) in enumerate(components):
            edge_colors = [
                FAMILY_COLORS.get(MODEL_FAMILY.get(m, ""), "#999999")
                for m in models
            ]
            bars = ax.bar(
                x + i * width, tdf[col], width,
                label=label, color=color, alpha=0.75,
                edgecolor=edge_colors, linewidth=1.5,
            )

        # Overlay final composite score as line with markers
        ax.plot(
            x + width, tdf["final_score"],
            "ko-", markersize=8, linewidth=2, label="Final Composite Score",
            zorder=5,
        )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"MCDM Composite Scores — Task {task_num}", fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"mcdm_composite_task{task_num}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"MCDM chart saved: {plot_path}")


# ---------------------------------------------------------------------------
# Novel class absorption heatmap
# ---------------------------------------------------------------------------

def generate_novel_class_heatmap(results_dir: str, output_dir: str) -> None:
    """Generate heatmaps showing how novel attack classes are absorbed.

    For each external target dataset, aggregates the novel class absorption
    maps across all 8 models and creates a heatmap where rows are novel
    attack families and columns are the 6 known CICIoMT2024 classes.
    Each row is normalized to 100%.

    Args:
        results_dir: Root results directory.
        output_dir: Where to save the heatmap PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    known_classes = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]

    # Scan for novel_analysis.json files
    # cross_dataset.py saves as {model}_{dataset}_novel_analysis.json inside
    # results/cross_dataset/{Dataset}/task_6/{Model}/
    pattern = os.path.join(
        results_dir, "cross_dataset", "*", "task_6", "*", "*_novel_analysis.json",
    )
    json_files = glob.glob(pattern)

    if not json_files:
        logger.warning(f"No novel_analysis.json files found: {pattern}")
        return

    # Group by target dataset
    dataset_absorptions: Dict[str, Dict[str, Dict[str, int]]] = {}
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {jf}: {e}")
            continue

        parts = Path(jf).parts
        # .../cross_dataset/{DatasetName}/task_6/{Model}/{model}_{dataset}_novel_analysis.json
        dataset_name = parts[-4]

        absorption_map = data.get("absorption_map", {})
        if not absorption_map:
            continue

        if dataset_name not in dataset_absorptions:
            dataset_absorptions[dataset_name] = {}

        for novel_class, mapping in absorption_map.items():
            if novel_class not in dataset_absorptions[dataset_name]:
                dataset_absorptions[dataset_name][novel_class] = {}
            for known_class, count in mapping.items():
                prev = dataset_absorptions[dataset_name][novel_class].get(
                    known_class, 0,
                )
                dataset_absorptions[dataset_name][novel_class][known_class] = (
                    prev + count
                )

    for dataset_name, novel_map in dataset_absorptions.items():
        if not novel_map:
            logger.info(
                f"No novel classes found for {dataset_name} — skipping heatmap"
            )
            continue

        # Build matrix: rows = novel classes, cols = known classes
        novel_classes = sorted(novel_map.keys())
        matrix = np.zeros((len(novel_classes), len(known_classes)))

        for i, nc in enumerate(novel_classes):
            for j, kc in enumerate(known_classes):
                # absorption_map uses string indices ("0","1",...) as keys
                matrix[i, j] = novel_map[nc].get(str(j), 0)

        # Normalize each row to 100%
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        matrix_pct = (matrix / row_sums) * 100

        fig, ax = plt.subplots(
            figsize=(max(10, len(known_classes) * 1.5), max(6, len(novel_classes) * 0.8))
        )
        sns.heatmap(
            matrix_pct,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            xticklabels=known_classes,
            yticklabels=novel_classes,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Absorption (%)"},
        )
        ax.set_title(
            f"Novel Class Absorption — {dataset_name}", fontsize=14,
        )
        ax.set_xlabel("Known CICIoMT2024 Class", fontsize=12)
        ax.set_ylabel("Novel Attack Family", fontsize=12)
        plt.tight_layout()

        safe_name = dataset_name.replace(" ", "_")
        plot_path = os.path.join(
            output_dir, f"novel_class_absorption_{safe_name}.png",
        )
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Novel class absorption heatmap saved: {plot_path}")


# ---------------------------------------------------------------------------
# Per-class F1 heatmap (Task 19)
# ---------------------------------------------------------------------------

def generate_perclass_f1_heatmap(results_dir: str, output_dir: str) -> None:
    """Generate a heatmap of per-class F1 scores for the 19-class task.

    Reads per_class_f1 from each model's results.json for task 19 and
    creates a heatmap: rows = 19 sorted class names, columns = 8 models
    in MODEL_ORDER. Uses RdYlGn colormap (red=low, green=high).

    Args:
        results_dir: Root results directory.
        output_dir: Where to save the heatmap PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _apply_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(
        results_dir, "CICIoMT2024", "task_19", "*", "results.json",
    )
    json_files = glob.glob(pattern)

    if not json_files:
        logger.warning(f"No task 19 results.json files found: {pattern}")
        return

    model_f1s: Dict[str, Dict[str, float]] = {}
    all_classes = set()

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {jf}: {e}")
            continue

        model_name = Path(jf).parts[-2]
        per_class = data.get("test_metrics", {}).get("per_class_f1", {})
        if not per_class:
            # Try alternate location
            per_class = data.get("per_class_f1", {})
        if per_class:
            model_f1s[model_name] = per_class
            all_classes.update(per_class.keys())

    if not model_f1s:
        logger.warning("No per-class F1 data found for task 19")
        return

    # Build matrix
    class_names = sorted(all_classes)
    models_present = [m for m in MODEL_ORDER if m in model_f1s]

    matrix = np.zeros((len(class_names), len(models_present)))
    for j, model in enumerate(models_present):
        for i, cls in enumerate(class_names):
            matrix[i, j] = model_f1s[model].get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=models_present,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Per-Class F1 Scores — Task 19 (19-class)", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "perclass_f1_task19.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Per-class F1 heatmap saved: {plot_path}")


# ---------------------------------------------------------------------------
# MCDM sensitivity analysis chart
# ---------------------------------------------------------------------------

def generate_sensitivity_chart(results_dir: str, output_dir: str) -> None:
    """Generate sensitivity analysis charts for MCDM model selection.

    For each task, creates a grouped bar chart showing the final composite
    score of the top model under each of the 7 weight scenarios (baseline,
    perf+/-10%, effi+/-10%, expl+/-10%). A horizontal dashed line marks the
    baseline score, and the baseline bar is colored distinctly.

    Args:
        results_dir: Root results directory.
        output_dir: Where to save the chart PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    ms_path = os.path.join(results_dir, "model_selection_results.json")
    if not os.path.exists(ms_path):
        logger.warning(f"Model selection results not found: {ms_path}")
        return

    with open(ms_path, "r") as f:
        ms_data = json.load(f)

    # Sensitivity data lives inside each task entry:
    # ms_data["tasks"]["task_N"]["sensitivity"]["scenarios"] -> list of dicts
    tasks_data = ms_data.get("tasks", {})
    if not tasks_data:
        logger.warning("No tasks data found in model_selection_results.json")
        return

    found_any = False
    for task_key, task_entry in tasks_data.items():
        sens_block = task_entry.get("sensitivity", {})
        scenarios_list = sens_block.get("scenarios", [])
        if not scenarios_list:
            continue
        found_any = True

        task_num = str(task_key).replace("task_", "").replace("task", "")

        scenario_order = [
            "baseline",
            "perf+10%", "perf-10%",
            "effi+10%", "effi-10%",
            "expl+10%", "expl-10%",
        ]

        # Parse scenarios list into dicts keyed by scenario name
        scenario_scores = {}
        scenario_top_models = {}
        for sdata in scenarios_list:
            name = sdata.get("scenario", "")
            score = sdata.get("top_score", 0)
            top_model = sdata.get("top_model", "?")
            scenario_scores[name] = score
            scenario_top_models[name] = top_model

        # Reorder scenarios to match expected order, keeping any extras
        ordered_scenarios = [s for s in scenario_order if s in scenario_scores]
        extras = [s for s in scenario_scores if s not in scenario_order]
        ordered_scenarios.extend(sorted(extras))

        if not ordered_scenarios:
            continue

        scores = [scenario_scores[s] for s in ordered_scenarios]
        top_models = [scenario_top_models[s] for s in ordered_scenarios]

        # Count how many scenarios have the same top model as baseline
        baseline_model = scenario_top_models.get("baseline", top_models[0])
        same_top_count = sum(1 for m in top_models if m == baseline_model)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = []
        for s in ordered_scenarios:
            if s == "baseline":
                colors.append("#2196F3")  # distinct blue for baseline
            else:
                colors.append("#90CAF9")  # lighter blue for variants

        bars = ax.bar(range(len(ordered_scenarios)), scores, color=colors)

        # Baseline dashed line
        baseline_score = scenario_scores.get("baseline", scores[0])
        ax.axhline(
            y=baseline_score, color="#D32F2F", linestyle="--",
            linewidth=1.5, label=f"Baseline ({baseline_score:.4f})",
        )

        ax.set_xlabel("Weight Scenario", fontsize=12)
        ax.set_ylabel("Final Composite Score", fontsize=12)
        ax.set_title(
            f"MCDM Sensitivity Analysis — Task {task_num}", fontsize=14,
        )
        ax.set_xticks(range(len(ordered_scenarios)))
        ax.set_xticklabels(ordered_scenarios, rotation=35, ha="right")

        # Annotation for robustness
        ax.annotate(
            f"{same_top_count}/{len(ordered_scenarios)} scenarios: same top model",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        ax.legend(loc="lower right")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"sensitivity_task{task_num}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Sensitivity chart saved: {plot_path}")

    if not found_any:
        logger.warning("No sensitivity analysis data found in any task entry")


# ---------------------------------------------------------------------------
# MCC comparison across tasks
# ---------------------------------------------------------------------------

def generate_mcc_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Generate a grouped bar chart comparing MCC across models and tasks.

    X-axis: models in MODEL_ORDER, grouped bars for each task
    (Task 2, Task 6, Task 19). Y-axis: MCC values. Bars colored by task.

    Args:
        df: Primary results DataFrame.
        output_dir: Where to save the chart PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    if "mcc" not in df.columns:
        logger.warning("MCC column not found in results — skipping MCC comparison")
        return

    tasks = sorted(df["task"].unique())
    task_colors = {2: "#2196F3", 6: "#FF9800", 19: "#4CAF50"}
    # Fallback colors for unexpected task numbers
    default_palette = ["#9C27B0", "#F44336", "#009688", "#795548"]

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, ax = plt.subplots(figsize=(14, 7))

    n_tasks = len(tasks)
    width = 0.8 / n_tasks
    x = np.arange(len(models_present))

    for i, task in enumerate(tasks):
        task_df = df[df["task"] == task]
        mcc_vals = []
        for model in models_present:
            model_row = task_df[task_df["model"] == model]
            if not model_row.empty:
                mcc_vals.append(model_row["mcc"].values[0])
            else:
                mcc_vals.append(0.0)

        color = task_colors.get(task, default_palette[i % len(default_palette)])
        ax.bar(
            x + i * width - (n_tasks - 1) * width / 2,
            mcc_vals, width,
            label=f"Task {task}",
            color=color, alpha=0.85,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_title("MCC Comparison Across Tasks", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models_present, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "mcc_comparison_all_tasks.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"MCC comparison chart saved: {plot_path}")


# ---------------------------------------------------------------------------
# Collect dissertation figures
# ---------------------------------------------------------------------------

def collect_dissertation_figures(results_dir: str, output_dir: str) -> None:
    """Copy all figures needed for the dissertation into organized subdirectories.

    Creates chapter5/ for the 7 main figures and appendices/ for supplementary
    figures including all confusion matrices, SHAP plots, and additional charts.

    Args:
        results_dir: Root results directory.
        output_dir: Root output directory (reports/).
    """
    os.makedirs(os.path.join(output_dir, "chapter5"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "appendices"), exist_ok=True)

    copied = 0
    skipped = 0

    def _copy(src: str, dst_subdir: str, dst_name: Optional[str] = None) -> None:
        nonlocal copied, skipped
        if dst_name is None:
            dst_name = os.path.basename(src)
        dst = os.path.join(output_dir, dst_subdir, dst_name)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {src} -> {dst}")
            copied += 1
        else:
            logger.warning(f"  Not found (skipped): {src}")
            skipped += 1

    # ----- Chapter 5: Main figures -----
    logger.info("Collecting Chapter 5 figures...")
    plots_dir = os.path.join(output_dir, "plots")

    # MCDM composite chart (task 6 is the representative multi-class task)
    _copy(
        os.path.join(plots_dir, "mcdm_composite_task6.png"),
        "chapter5",
    )

    # Cross-dataset heatmaps for task 2 and task 6
    _copy(
        os.path.join(plots_dir, "cross_dataset_heatmap_task2.png"),
        "chapter5",
    )
    _copy(
        os.path.join(plots_dir, "cross_dataset_heatmap_task6.png"),
        "chapter5",
    )

    # Novel class absorption for CIC-ToN-IoT (chapter 5)
    _copy(
        os.path.join(plots_dir, "novel_class_absorption_CIC-ToN-IoT.png"),
        "chapter5",
    )

    # Novel class absorption for CIC-BoT-IoT and CIC-IoT-DIAD-2024 (appendices)
    _copy(
        os.path.join(plots_dir, "novel_class_absorption_CIC-BoT-IoT.png"),
        "appendices",
    )
    _copy(
        os.path.join(plots_dir, "novel_class_absorption_CIC-IoT-DIAD-2024.png"),
        "appendices",
    )

    # XGBoost SHAP bar plot (task 6)
    _copy(
        os.path.join(
            results_dir, "CICIoMT2024", "task_6", "XGBoost",
            "explainability", "XGBoost_shap_global_bar.png",
        ),
        "chapter5",
    )

    # XGBoost SHAP summary/beeswarm (task 6)
    _copy(
        os.path.join(
            results_dir, "CICIoMT2024", "task_6", "XGBoost",
            "explainability", "XGBoost_shap_summary.png",
        ),
        "chapter5",
    )

    # XGBoost confusion matrix (task 19) - renamed for LaTeX consistency
    _copy(
        os.path.join(
            results_dir, "CICIoMT2024", "task_19", "XGBoost",
            "confusion_matrix.png",
        ),
        "chapter5",
        "cm_task19_XGBoost.png",
    )

    # Efficiency scatter plot (task 19)
    _copy(
        os.path.join(
            plots_dir, "task19_efficiency_inference_latency_ms.png",
        ),
        "chapter5",
    )

    # ----- Appendices: Supplementary figures -----
    logger.info("Collecting Appendix figures...")

    # Per-class F1 heatmap
    _copy(
        os.path.join(plots_dir, "perclass_f1_task19.png"),
        "appendices",
    )

    # Sensitivity charts
    for task in [2, 6, 19]:
        _copy(
            os.path.join(plots_dir, f"sensitivity_task{task}.png"),
            "appendices",
        )

    # MCC comparison
    _copy(
        os.path.join(plots_dir, "mcc_comparison_all_tasks.png"),
        "appendices",
    )

    # MCDM composite charts for tasks 2 and 19 (task 6 is in chapter5)
    _copy(
        os.path.join(plots_dir, "mcdm_composite_task2.png"),
        "appendices",
    )
    _copy(
        os.path.join(plots_dir, "mcdm_composite_task19.png"),
        "appendices",
    )

    # All 24 CICIoMT2024 confusion matrices (3 tasks x 8 models)
    for task in [2, 6, 19]:
        for model in MODEL_ORDER:
            _copy(
                os.path.join(
                    results_dir, "CICIoMT2024", f"task_{task}",
                    model, "confusion_matrix.png",
                ),
                "appendices",
                f"cm_task{task}_{model}.png",
            )

    # All 48 cross-dataset confusion matrices (3 datasets x 2 tasks x 8 models)
    # cross_dataset.py saves as {model}_{dataset}_task{N}_confusion_matrix.png
    # inside results/cross_dataset/{Dataset}/task_{N}/{Model}/
    cd_pattern = os.path.join(
        results_dir, "cross_dataset", "*", "task_*", "*", "*_confusion_matrix.png",
    )
    for cm_path in sorted(glob.glob(cd_pattern)):
        parts = Path(cm_path).parts
        dataset = parts[-4]
        task_dir = parts[-3]
        model = parts[-2]
        safe_dataset = dataset.replace(" ", "_")
        _copy(
            cm_path,
            "appendices",
            f"cm_xd_{safe_dataset}_{task_dir}_{model}.png",
        )

    # Selected SHAP plots (all models, task 6)
    for model in MODEL_ORDER:
        shap_dir = os.path.join(
            results_dir, "CICIoMT2024", "task_6", model, "explainability",
        )
        for shap_file in [
            f"{model}_shap_global_bar.png",
            f"{model}_shap_summary.png",
        ]:
            _copy(
                os.path.join(shap_dir, shap_file),
                "appendices",
                f"shap_task6_{shap_file}",
            )

    logger.info(
        f"\nDissertation figure collection complete: "
        f"{copied} copied, {skipped} not found"
    )


# ---------------------------------------------------------------------------
# Model selection summary
# ---------------------------------------------------------------------------

def generate_model_selection_summary(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate multi-criteria model selection summary.

    Runs the 60/25/15 composite scoring and saves ranked tables.
    """
    from evaluation.model_selection import compute_final_scores, select_best_per_family

    os.makedirs(output_dir, exist_ok=True)

    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task].copy()

        try:
            ranked = compute_final_scores(task_df)
            best = select_best_per_family(ranked)

            # Save ranked table
            cols_to_save = [
                "rank", "model", "final_score", "perf_score",
                "eff_score", "xai_score", "f1_weighted", "f1_macro", "mcc",
            ]
            available = [c for c in cols_to_save if c in ranked.columns]
            save_df = ranked[available]

            csv_path = os.path.join(output_dir, f"model_selection_task{task}.csv")
            save_df.to_csv(csv_path, index=False, float_format="%.4f")
            logger.info(f"Model selection saved: {csv_path}")

            # Log best per family
            logger.info(f"\nTask {task} — Best per family:")
            for family, model in best.items():
                row = ranked[ranked["model"] == model].iloc[0]
                logger.info(
                    f"  {family:15s} → {model:20s} "
                    f"(final={row['final_score']:.4f})"
                )
        except Exception as e:
            logger.error(f"Model selection for task {task} failed: {e}")


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _save_latex_table(
    df: pd.DataFrame,
    output_path: str,
    caption: str,
) -> None:
    """Save a DataFrame as a LaTeX table."""
    try:
        latex = df.to_latex(
            index=False,
            float_format="%.4f",
            caption=caption,
            label=f"tab:{Path(output_path).stem}",
            column_format="l" * len(df.columns),
        )
        with open(output_path, "w") as f:
            f.write(latex)
    except Exception as e:
        logger.warning(f"LaTeX export failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate final report from experimental results.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results CSVs (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save generated reports (default: reports)",
    )
    parser.add_argument(
        "--sections",
        type=str,
        nargs="+",
        default=["leaderboards", "plots", "efficiency", "cross_dataset",
                 "model_selection", "mcdm_chart", "novel_class",
                 "perclass_f1", "sensitivity", "mcc_comparison", "collect"],
        help="Which report sections to generate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results_dir = os.path.join(project_root, args.results_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    # Load data
    df = load_primary_results(results_dir)
    cd_df = load_cross_dataset_results(results_dir)

    if df.empty:
        logger.error("No primary results found. Run training first.")
        sys.exit(1)

    # Generate sections
    if "leaderboards" in args.sections:
        logger.info("\n--- Generating leaderboards ---")
        generate_leaderboards(df, os.path.join(output_dir, "leaderboards"))

    if "plots" in args.sections:
        logger.info("\n--- Generating comparison plots ---")
        generate_comparison_plots(df, os.path.join(output_dir, "plots"))

    if "efficiency" in args.sections:
        logger.info("\n--- Generating efficiency plots ---")
        generate_efficiency_plots(df, os.path.join(output_dir, "plots"))

    if "cross_dataset" in args.sections and not cd_df.empty:
        logger.info("\n--- Generating cross-dataset heatmaps ---")
        generate_cross_dataset_heatmap(cd_df, os.path.join(output_dir, "plots"))

    if "model_selection" in args.sections:
        logger.info("\n--- Generating model selection summary ---")
        generate_model_selection_summary(df, os.path.join(output_dir, "model_selection"))

    if "mcdm_chart" in args.sections:
        logger.info("\n--- Generating MCDM composite charts ---")
        generate_mcdm_chart(results_dir, os.path.join(output_dir, "plots"))

    if "novel_class" in args.sections:
        logger.info("\n--- Generating novel class absorption heatmaps ---")
        generate_novel_class_heatmap(results_dir, os.path.join(output_dir, "plots"))

    if "perclass_f1" in args.sections:
        logger.info("\n--- Generating per-class F1 heatmap ---")
        generate_perclass_f1_heatmap(results_dir, os.path.join(output_dir, "plots"))

    if "sensitivity" in args.sections:
        logger.info("\n--- Generating sensitivity analysis charts ---")
        generate_sensitivity_chart(results_dir, os.path.join(output_dir, "plots"))

    if "mcc_comparison" in args.sections:
        logger.info("\n--- Generating MCC comparison chart ---")
        generate_mcc_comparison(df, os.path.join(output_dir, "plots"))

    if "collect" in args.sections:
        logger.info("\n--- Collecting dissertation figures ---")
        collect_dissertation_figures(results_dir, output_dir)

    logger.info(f"\nReport generation complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
