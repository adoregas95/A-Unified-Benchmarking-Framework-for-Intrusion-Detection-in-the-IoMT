#!/usr/bin/env python3
"""
Master orchestration script for the IoMT IDS Benchmarking Framework.

Generates and runs all (model × task) combinations for:
    1. Primary training (CICIoMT2024)
    2. Cross-dataset generalization (3 targets × applicable tasks)
    3. Explainability analysis (Stage 1 for all, Stage 2 for top per family)

On HPC, this script is NOT the primary way to submit jobs. Instead, use the
SLURM array scripts (slurm_train_tree.sh, slurm_train_dl.sh, etc.) which
submit jobs as SLURM arrays for parallel execution. This script is useful for:
    - Local development/testing (run a subset sequentially)
    - Orchestrating the full pipeline after all training completes
    - Generating the final job matrix for verification

Usage:
    # List all jobs without running (dry-run):
    python scripts/run_all.py --dry-run

    # Run all primary training sequentially:
    python scripts/run_all.py --stage training

    # Run cross-dataset evaluation (after training):
    python scripts/run_all.py --stage cross_dataset

    # Run explainability (after training):
    python scripts/run_all.py --stage explainability

    # Run everything in sequence:
    python scripts/run_all.py --stage all

    # Run only specific models/tasks:
    python scripts/run_all.py --stage training --models XGBoost LightGBM --tasks 2 6
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.train import (
    run_training_pipeline,
    load_config,
    merge_result_csvs,
    MODEL_REGISTRY,
    get_model_instance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# All model names, grouped by family
# ---------------------------------------------------------------------------
ALL_MODELS = list(MODEL_REGISTRY.keys())
TREE_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DL_MODELS = ["CNN1D", "BiLSTM"]
TRANSFORMER_MODELS = ["FTTransformer", "SAINT"]

PRIMARY_TASKS = [2, 6, 19]
CROSS_DATASET_TASKS = [2, 6]  # Task 19 is not applicable
CROSS_DATASET_TARGETS = ["CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT"]


def generate_training_jobs(
    models: Optional[List[str]] = None,
    tasks: Optional[List[int]] = None,
) -> List[Tuple[str, str, int]]:
    """Generate all (model, dataset, task) training combinations.

    Args:
        models: Subset of models to include. Defaults to all 8.
        tasks: Subset of tasks. Defaults to [2, 6, 19].

    Returns:
        List of (model_name, dataset, task) tuples.
    """
    models = models or ALL_MODELS
    tasks = tasks or PRIMARY_TASKS

    jobs = []
    for model in models:
        for task in tasks:
            jobs.append((model, "CICIoMT2024", task))

    return jobs


def generate_cross_dataset_jobs(
    models: Optional[List[str]] = None,
    tasks: Optional[List[int]] = None,
    targets: Optional[List[str]] = None,
) -> List[Tuple[str, str, int]]:
    """Generate all cross-dataset evaluation combinations.

    Args:
        models: Subset of models. Defaults to all 8.
        tasks: Subset of tasks. Defaults to [2, 6].
        targets: Subset of targets. Defaults to all 3.

    Returns:
        List of (model_name, target_dataset, task) tuples.
    """
    models = models or ALL_MODELS
    tasks = tasks or CROSS_DATASET_TASKS
    targets = targets or CROSS_DATASET_TARGETS

    jobs = []
    for model in models:
        for target in targets:
            for task in tasks:
                jobs.append((model, target, task))

    return jobs


def run_training_stage(
    config: Dict[str, Any],
    models: Optional[List[str]] = None,
    tasks: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run all primary training jobs sequentially.

    Args:
        config: Parsed config.yaml.
        models: Model subset.
        tasks: Task subset.
        dry_run: If True, list jobs without executing.

    Returns:
        Dict mapping job_id to result dict.
    """
    jobs = generate_training_jobs(models, tasks)
    output_dir = config.get("output", {}).get("results_dir", "results")

    logger.info(f"Training stage: {len(jobs)} jobs")
    for i, (model, dataset, task) in enumerate(jobs):
        logger.info(f"  [{i+1:2d}] {model} / {dataset} / task={task}")

    if dry_run:
        return {}

    results = {}
    for i, (model, dataset, task) in enumerate(jobs):
        job_id = f"{model}_{dataset}_task{task}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Job {i+1}/{len(jobs)}: {job_id}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        try:
            result = run_training_pipeline(
                model_name=model,
                dataset=dataset,
                task=task,
                config=config,
                output_dir=output_dir,
            )
            elapsed = time.time() - t0
            result["wall_time_seconds"] = elapsed
            results[job_id] = result
            logger.info(
                f"  DONE: F1(w)={result['test_metrics']['f1_weighted']:.4f} "
                f"in {elapsed:.0f}s"
            )
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  FAILED after {elapsed:.0f}s: {e}", exc_info=True)
            results[job_id] = {"error": str(e), "wall_time_seconds": elapsed}

    return results


def run_cross_dataset_stage(
    config: Dict[str, Any],
    models: Optional[List[str]] = None,
    tasks: Optional[List[int]] = None,
    targets: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run all cross-dataset evaluation jobs.

    Loads model checkpoints from training results and evaluates them
    on cross-dataset targets.

    Args:
        config: Parsed config.yaml.
        models: Model subset.
        tasks: Task subset (only 2 and 6 applicable).
        targets: Target dataset subset.
        dry_run: If True, list jobs without executing.

    Returns:
        Dict mapping job_id to result dict.
    """
    from evaluation.cross_dataset import run_cross_dataset_evaluation

    jobs = generate_cross_dataset_jobs(models, tasks, targets)
    output_dir = config.get("output", {}).get("results_dir", "results")
    cache_dir = os.path.join(project_root, config["preprocessing"]["cache_dir"])

    logger.info(f"Cross-dataset stage: {len(jobs)} jobs")
    for i, (model, target, task) in enumerate(jobs):
        logger.info(f"  [{i+1:2d}] {model} → {target} / task={task}")

    if dry_run:
        return {}

    # Fail-fast: verify all required checkpoints exist before starting
    missing_checkpoints = []
    for model_name, target, task in jobs:
        checkpoint_path = os.path.join(
            project_root, output_dir, "CICIoMT2024",
            f"task_{task}", model_name, "model.pkl",
        )
        if not os.path.exists(checkpoint_path):
            missing_checkpoints.append(
                f"  {model_name}/task_{task}: {checkpoint_path}"
            )

    if missing_checkpoints:
        msg = (
            f"Cannot run cross-dataset evaluation: "
            f"{len(missing_checkpoints)} model checkpoint(s) missing.\n"
            + "\n".join(missing_checkpoints)
            + "\n\nRun primary training first to generate these checkpoints."
        )
        raise FileNotFoundError(msg)

    results = {}
    random_state = config["preprocessing"]["random_state"]

    for i, (model_name, target, task) in enumerate(jobs):
        job_id = f"{model_name}_{target}_task{task}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-dataset {i+1}/{len(jobs)}: {job_id}")
        logger.info(f"{'='*60}")

        # Load checkpoint from primary training
        checkpoint_path = os.path.join(
            project_root, output_dir, "CICIoMT2024",
            f"task_{task}", model_name, "model.pkl",
        )

        try:
            # Determine dimensions from primary preprocessed data
            from preprocessing.preprocessing import load_preprocessed_data
            _, _, _, _, X_test_p, _, p_meta = load_preprocessed_data(
                cache_dir, "CICIoMT2024", task
            )
            input_dim = X_test_p.shape[1]
            n_classes = len(p_meta.get("label_classes", []))

            model = get_model_instance(
                model_name, random_state,
                input_dim=input_dim, n_classes=n_classes,
            )
            model.load_checkpoint(checkpoint_path)

            t0 = time.time()
            result = run_cross_dataset_evaluation(
                model=model,
                model_name=model_name,
                target_dataset=target,
                task=task,
                primary_cache_dir=cache_dir,
                output_dir=os.path.join(project_root, output_dir),
            )
            elapsed = time.time() - t0
            result["wall_time_seconds"] = elapsed
            results[job_id] = result
            logger.info(
                f"  DONE: F1(macro)={result['test_metrics']['f1_macro']:.4f} "
                f"in {elapsed:.0f}s"
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            results[job_id] = {"error": str(e)}

    return results


def run_explainability_stage(
    config: Dict[str, Any],
    models: Optional[List[str]] = None,
    tasks: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run explainability analysis (Stage 1 + Stage 2).

    Stage 1: Lightweight SHAP for all models.
    Stage 2: Full SHAP + LIME for top model per family (after model selection).

    Args:
        config: Parsed config.yaml.
        models: Model subset (Stage 1 only).
        tasks: Task subset.
        dry_run: If True, list jobs without executing.

    Returns:
        Dict mapping job_id to result dict.
    """
    from evaluation.explainability.shap_analysis import run_shap_analysis
    from evaluation.explainability.lime_analysis import (
        run_lime_analysis,
        generate_consistency_report,
    )
    from evaluation.model_selection import compute_final_scores, select_best_per_family

    models = models or ALL_MODELS
    tasks = tasks or PRIMARY_TASKS
    output_dir = config.get("output", {}).get("results_dir", "results")
    cache_dir = os.path.join(project_root, config["preprocessing"]["cache_dir"])
    random_state = config["preprocessing"]["random_state"]

    # Stage 1: global SHAP for all models × tasks
    stage1_jobs = [(m, t) for m in models for t in tasks]
    logger.info(f"Explainability Stage 1: {len(stage1_jobs)} jobs (SHAP global)")
    for i, (model, task) in enumerate(stage1_jobs):
        logger.info(f"  [{i+1:2d}] Stage 1 SHAP: {model} / task={task}")

    if dry_run:
        # Also show Stage 2 plan
        logger.info(f"Explainability Stage 2: top model per family × {len(tasks)} tasks")
        logger.info("  (models determined after Stage 1 + model selection)")
        return {}

    results = {}

    # --- Stage 1 ---
    for i, (model_name, task) in enumerate(stage1_jobs):
        job_id = f"shap_stage1_{model_name}_task{task}"
        logger.info(f"\nStage 1 [{i+1}/{len(stage1_jobs)}]: {model_name} / task={task}")

        checkpoint_path = os.path.join(
            project_root, output_dir, "CICIoMT2024",
            f"task_{task}", model_name, "model.pkl",
        )
        if not os.path.exists(checkpoint_path):
            logger.error(f"  Checkpoint not found: {checkpoint_path}")
            results[job_id] = {"error": "Checkpoint not found"}
            continue

        try:
            from preprocessing.preprocessing import load_preprocessed_data
            X_train, _, _, _, X_test, y_test, p_meta = load_preprocessed_data(
                cache_dir, "CICIoMT2024", task
            )
            feature_names = p_meta.get("feature_names", [f"f{i}" for i in range(X_test.shape[1])])
            input_dim = X_test.shape[1]
            n_classes = len(p_meta.get("label_classes", []))

            model = get_model_instance(
                model_name, random_state,
                input_dim=input_dim, n_classes=n_classes,
            )
            model.load_checkpoint(checkpoint_path)

            xai_dir = os.path.join(
                project_root, output_dir, "CICIoMT2024",
                f"task_{task}", model_name, "explainability",
            )

            t0 = time.time()
            result = run_shap_analysis(
                model=model,
                model_name=model_name,
                X_test=X_test,
                feature_names=feature_names,
                output_dir=xai_dir,
                stage=1,
                random_state=random_state,
            )
            elapsed = time.time() - t0
            result["wall_time_seconds"] = elapsed
            results[job_id] = result
            logger.info(
                f"  DONE: {elapsed:.0f}s, top feature = {result['global_importance'][0][0]}"
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            results[job_id] = {"error": str(e)}

    # --- Stage 2: run for top model per family per task ---
    for task in tasks:
        logger.info(f"\n--- Stage 2 for task={task} ---")

        # Load all_results.csv to determine best model per family.
        # This file must be created by merge_result_csvs() after training.
        master_csv = os.path.join(project_root, output_dir, "all_results.csv")
        if not os.path.exists(master_csv):
            raise FileNotFoundError(
                f"Master CSV not found: {master_csv}. "
                f"Run merge_result_csvs() after all training jobs complete, "
                f"then re-run the explainability stage."
            )

        import pandas as pd
        all_results_df = pd.read_csv(master_csv)
        task_df = all_results_df[all_results_df["task"] == task].copy()

        if task_df.empty:
            logger.warning(f"  No results for task={task}. Skipping Stage 2.")
            continue

        try:
            ranked_df = compute_final_scores(task_df, performance_metric="f1_weighted")
            best_per_family = select_best_per_family(ranked_df)
        except Exception as e:
            logger.error(f"  Model selection failed: {e}")
            continue

        for family, model_name in best_per_family.items():
            job_id = f"shap_lime_stage2_{model_name}_task{task}"
            logger.info(f"  Stage 2: {model_name} (best {family}) / task={task}")

            checkpoint_path = os.path.join(
                project_root, output_dir, "CICIoMT2024",
                f"task_{task}", model_name, "model.pkl",
            )
            if not os.path.exists(checkpoint_path):
                logger.error(f"    Checkpoint not found")
                results[job_id] = {"error": "Checkpoint not found"}
                continue

            try:
                X_train, _, _, _, X_test, y_test, p_meta = load_preprocessed_data(
                    cache_dir, "CICIoMT2024", task
                )
                feature_names = p_meta.get(
                    "feature_names",
                    [f"f{i}" for i in range(X_test.shape[1])],
                )
                class_names = p_meta.get("label_classes", [])
                input_dim = X_test.shape[1]
                n_classes = len(class_names)

                model = get_model_instance(
                    model_name, random_state,
                    input_dim=input_dim, n_classes=n_classes,
                )
                model.load_checkpoint(checkpoint_path)

                xai_dir = os.path.join(
                    project_root, output_dir, "CICIoMT2024",
                    f"task_{task}", model_name, "explainability",
                )

                # Stage 2 SHAP
                t0 = time.time()
                shap_result = run_shap_analysis(
                    model=model,
                    model_name=model_name,
                    X_test=X_test,
                    feature_names=feature_names,
                    output_dir=xai_dir,
                    stage=2,
                    y_test=y_test,
                    class_names=class_names,
                    random_state=random_state,
                )

                # Stage 2 LIME with SHAP cross-validation
                lime_result = run_lime_analysis(
                    model=model,
                    model_name=model_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    feature_names=feature_names,
                    class_names=class_names,
                    output_dir=xai_dir,
                    shap_values=shap_result.get("shap_values"),
                    random_state=random_state,
                )

                # Generate consistency report
                generate_consistency_report(lime_result, model_name, xai_dir)

                elapsed = time.time() - t0
                results[job_id] = {
                    "shap": {k: v for k, v in shap_result.items() if k != "shap_values"},
                    "lime": {k: v for k, v in lime_result.items()
                             if k not in ("instance_explanations",)},
                    "wall_time_seconds": elapsed,
                }
                logger.info(
                    f"    DONE: {elapsed:.0f}s, faithfulness={shap_result.get('faithfulness_score', 'N/A')}"
                )
            except Exception as e:
                logger.error(f"    FAILED: {e}", exc_info=True)
                results[job_id] = {"error": str(e)}

    return results


def print_job_matrix(
    models: Optional[List[str]] = None,
    tasks: Optional[List[int]] = None,
) -> None:
    """Print the full job matrix for verification."""
    models = models or ALL_MODELS
    tasks = tasks or PRIMARY_TASKS

    print("\n" + "=" * 70)
    print("JOB MATRIX — IoMT IDS Benchmarking Framework")
    print("=" * 70)

    print(f"\n--- Primary Training (CICIoMT2024) ---")
    training_jobs = generate_training_jobs(models, tasks)
    print(f"Total: {len(training_jobs)} jobs")
    for m, d, t in training_jobs:
        print(f"  {m:20s} / task={t}")

    cd_tasks = [t for t in tasks if t in CROSS_DATASET_TASKS]
    if cd_tasks:
        print(f"\n--- Cross-Dataset Generalization ---")
        cd_jobs = generate_cross_dataset_jobs(models, cd_tasks)
        print(f"Total: {len(cd_jobs)} jobs")
        for m, target, t in cd_jobs:
            print(f"  {m:20s} → {target:25s} / task={t}")

    print(f"\n--- Explainability ---")
    stage1_count = len(models) * len(tasks)
    stage2_count = 3 * len(tasks)  # 3 families × tasks
    print(f"Stage 1 (SHAP global): {stage1_count} jobs (all models × all tasks)")
    print(f"Stage 2 (SHAP+LIME):   {stage2_count} jobs (top per family × all tasks)")

    grand_total = len(training_jobs) + len(cd_jobs) + stage1_count + stage2_count
    print(f"\nGrand total: ~{grand_total} jobs")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Master orchestration for the IoMT IDS Framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["training", "cross_dataset", "explainability", "all"],
        help="Which stage(s) to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_MODELS,
        help="Specific models to run (default: all 8)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=None,
        choices=[2, 6, 19],
        help="Specific tasks to run (default: all applicable)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: tiny data, few epochs, debug/ output tree",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all jobs without executing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/run_all.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    config = load_config(args.config)

    # Apply debug overrides if requested
    if args.debug:
        from config.debug import apply_debug_overrides, ensure_debug_dirs
        config = apply_debug_overrides(config)
        ensure_debug_dirs()

    if args.dry_run:
        print_job_matrix(args.models, args.tasks)

    stages = (
        ["training", "cross_dataset", "explainability"]
        if args.stage == "all"
        else [args.stage]
    )

    total_start = time.time()
    all_results = {}

    for stage in stages:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Stage: {stage.upper()}")
        logger.info(f"{'#'*60}")

        if stage == "training":
            results = run_training_stage(
                config, args.models, args.tasks, args.dry_run,
            )
            # After training completes, merge per-job CSVs into all_results.csv
            if not args.dry_run and results:
                output_dir = config.get("output", {}).get("results_dir", "results")
                merge_result_csvs(os.path.join(project_root, output_dir))
        elif stage == "cross_dataset":
            results = run_cross_dataset_stage(
                config, args.models, args.tasks, dry_run=args.dry_run,
            )
        elif stage == "explainability":
            results = run_explainability_stage(
                config, args.models, args.tasks, args.dry_run,
            )
        else:
            results = {}

        all_results[stage] = results

    total_elapsed = time.time() - total_start

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL STAGES COMPLETE — {total_elapsed:.0f}s total")
    logger.info(f"{'='*60}")
    for stage, results in all_results.items():
        n_ok = sum(1 for r in results.values() if "error" not in r)
        n_fail = sum(1 for r in results.values() if "error" in r)
        logger.info(f"  {stage}: {n_ok} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
