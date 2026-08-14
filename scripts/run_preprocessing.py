#!/usr/bin/env python3
"""
Standalone preprocessing runner for the IoMT IDS Framework.

Runs the preprocessing pipeline for a specific (dataset, task) combination
and caches the result to disk. Feature selection uses cumulative MI
thresholding (default 90%) — the threshold is adaptive per task, so each
(dataset, task) pair needs only one preprocessing run.

Designed to be run on the HPC cluster via SLURM before model training begins.

Usage:
    # Single run:
    python scripts/run_preprocessing.py --dataset CICIoMT2024 --task 2

    # Run ALL primary dataset + task combinations (3 runs):
    python scripts/run_preprocessing.py --all-primary

    # Run with custom MI threshold:
    python scripts/run_preprocessing.py --dataset CICIoMT2024 --task 6 --mi-threshold 0.95

    # Skip feature selection entirely (use all 76 features):
    python scripts/run_preprocessing.py --dataset CICIoMT2024 --task 2 --mi-threshold all

    # Dry run (show what would be processed without actually running):
    python scripts/run_preprocessing.py --all-primary --dry-run
"""

import argparse
import logging
import os
import sys
import time
import yaml

# Add the project root to Python path so we can import our modules.
# This handles both running from project root and from scripts/ directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from preprocessing.preprocessing import preprocess_pipeline


def load_config(config_path: str = None) -> dict:
    """
    Load the project configuration from config.yaml.

    Args:
        config_path: Path to config.yaml. If None, looks in the default
                     location (project_root/config/config.yaml).

    Returns:
        Parsed YAML config as a dict.
    """
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_all_primary_combinations(config: dict) -> list:
    """
    Generate all (task,) combinations for the primary dataset.

    With cumulative MI thresholding, the threshold is fixed per config
    (default 0.90), so each task only needs one preprocessing run.
    3 tasks = 3 runs.

    Returns:
        List of task_id integers.
    """
    return [t["id"] for t in config["tasks"].values()]


def run_single_preprocessing(
    dataset: str,
    task: int,
    mi_threshold,
    config: dict,
    data_root: str = None,
    force: bool = False,
    debug_mode: bool = False,
) -> dict:
    """
    Run preprocessing for a single (dataset, task) combination.

    Args:
        dataset: Dataset name (e.g., "CICIoMT2024")
        task: Classification task (2, 6, or 19)
        mi_threshold: MI threshold (float like 0.90, or "all" for no selection)
        config: Parsed config.yaml dict
        data_root: Path to data/ directory (defaults to project_root/data)
        force: If True, reprocess even if cache exists

    Returns:
        Result dict from preprocess_pipeline (or summary if cache hit)
    """
    # Read preprocessing settings from config
    scaling = config["preprocessing"]["scaling"]
    imbalance = config["preprocessing"]["imbalance_method"]
    random_state = config["preprocessing"]["random_state"]
    cache_dir = os.path.join(project_root, config["preprocessing"]["cache_dir"])
    val_ratio = config["hpo"]["validation_split"]

    if data_root is None:
        data_root = os.path.join(project_root, "data")

    # Run the full preprocessing pipeline
    logging.info(f"{'='*60}")
    logging.info(f"Processing: {dataset} | task={task} | mi_threshold={mi_threshold}")
    logging.info(f"  Scaling: {scaling}")
    logging.info(f"  Imbalance: {imbalance}")
    logging.info(f"  Val ratio: {val_ratio}")
    logging.info(f"  Data root: {data_root}")
    logging.info(f"  Cache dir: {cache_dir}")
    logging.info(f"{'='*60}")

    start_time = time.time()

    result = preprocess_pipeline(
        dataset=dataset,
        task=task,
        data_root=data_root,
        scaling=scaling,
        imbalance_method=imbalance,
        val_ratio=val_ratio,
        mi_threshold=mi_threshold,
        cache_dir=cache_dir,
        random_state=random_state,
        force_reprocess=force,
        debug_mode=debug_mode,
    )

    elapsed = time.time() - start_time

    # Log results including how many features were selected
    n_features = result["X_train"].shape[1]
    logging.info(
        f"DONE: {dataset}/task_{task} in {elapsed:.1f}s — "
        f"{n_features} features retained"
    )
    logging.info(
        f"  X_train: {result['X_train'].shape}, "
        f"X_val: {result['X_val'].shape}, "
        f"X_test: {result['X_test'].shape}"
    )
    logging.info(
        f"  Classes: {len(result['label_encoder'].classes_)} classes"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline and cache results to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess CICIoMT2024 task 2 (uses default 90% MI threshold):
  python scripts/run_preprocessing.py --dataset CICIoMT2024 --task 2

  # Run ALL 3 primary dataset task combinations:
  python scripts/run_preprocessing.py --all-primary

  # Custom MI threshold (keep features covering 95% of total MI):
  python scripts/run_preprocessing.py --dataset CICIoMT2024 --task 2 --mi-threshold 0.95

  # Skip feature selection (use all 76 features):
  python scripts/run_preprocessing.py --dataset CICIoMT2024 --task 2 --mi-threshold all

  # Force reprocessing (ignore existing cache):
  python scripts/run_preprocessing.py --all-primary --force

  # See what would be processed without running:
  python scripts/run_preprocessing.py --all-primary --dry-run
        """
    )

    # Mode: single run or all combinations
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all-primary",
        action="store_true",
        help="Run all 3 tasks for CICIoMT2024 (binary, 6-class, 19-class)",
    )
    mode_group.add_argument(
        "--dataset",
        type=str,
        choices=["CICIoMT2024", "CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT"],
        help="Dataset to preprocess (use with --task)",
    )

    # Single-run parameters
    parser.add_argument("--task", type=int, choices=[2, 6, 19],
                        help="Classification task")
    parser.add_argument(
        "--mi-threshold", type=str, default=None,
        help=(
            "MI cumulative threshold (e.g., 0.90 for 90%%) or 'all' to skip "
            "feature selection. Defaults to config.yaml value."
        ),
    )

    # Optional overrides
    parser.add_argument("--scaling", type=str, default=None,
                        help="Override scaling method (robust/minmax/standard)")
    parser.add_argument("--imbalance", type=str, default=None,
                        help="Override imbalance method (smotetomek/smote/none)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override path to data/ directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")

    # Control flags
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: subsample data, use debug/ output tree")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if cache exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Apply debug overrides if requested
    if args.debug:
        from config.debug import apply_debug_overrides, ensure_debug_dirs
        config = apply_debug_overrides(config)
        ensure_debug_dirs()

    # Apply any command-line overrides to the config
    if args.scaling:
        config["preprocessing"]["scaling"] = args.scaling
    if args.imbalance:
        config["preprocessing"]["imbalance_method"] = args.imbalance

    # Determine MI threshold: command-line > config.yaml default
    if args.mi_threshold is not None:
        if args.mi_threshold == "all":
            mi_threshold = "all"
        else:
            try:
                mi_threshold = float(args.mi_threshold)
            except ValueError:
                parser.error(
                    f"--mi-threshold must be a number (0-1) or 'all', "
                    f"got '{args.mi_threshold}'"
                )
    else:
        # Use config.yaml default
        mi_threshold = config["feature_selection"]["mi_threshold"]

    # Determine which combinations to run
    if args.all_primary:
        # All 3 tasks for CICIoMT2024
        task_ids = get_all_primary_combinations(config)
        combinations = [("CICIoMT2024", task) for task in task_ids]
    else:
        # Single run — validate required args
        if args.task is None:
            parser.error("--task is required when using --dataset")
        combinations = [(args.dataset, args.task)]

    # Show plan
    logging.info(f"Preprocessing plan: {len(combinations)} combination(s)")
    logging.info(f"MI threshold: {mi_threshold}")
    for i, (ds, task) in enumerate(combinations, 1):
        logging.info(f"  [{i:2d}] {ds} | task={task}")

    if args.dry_run:
        logging.info("DRY RUN — exiting without processing.")
        return

    # Execute all combinations
    total_start = time.time()
    results_summary = []

    for i, (ds, task) in enumerate(combinations, 1):
        logging.info(f"\n{'#'*60}")
        logging.info(f"# Job {i}/{len(combinations)}")
        logging.info(f"{'#'*60}")

        try:
            result = run_single_preprocessing(
                dataset=ds,
                task=task,
                mi_threshold=mi_threshold,
                config=config,
                data_root=args.data_root,
                force=args.force,
                debug_mode=args.debug,
            )
            n_feat = result["X_train"].shape[1]
            results_summary.append((ds, task, n_feat, "success"))

        except Exception as e:
            logging.error(f"FAILED: {ds}/task_{task} — {e}")
            results_summary.append((ds, task, 0, f"FAILED: {e}"))

    # Final summary
    total_elapsed = time.time() - total_start
    logging.info(f"\n{'='*60}")
    logging.info(f"PREPROCESSING COMPLETE — {total_elapsed:.1f}s total")
    logging.info(f"{'='*60}")

    for ds, task, n_feat, status in results_summary:
        icon = "OK" if status == "success" else "!!"
        logging.info(
            f"  [{icon}] {ds}/task_{task} — {status} "
            f"({n_feat} features)" if n_feat > 0 else
            f"  [{icon}] {ds}/task_{task} — {status}"
        )


if __name__ == "__main__":
    main()
