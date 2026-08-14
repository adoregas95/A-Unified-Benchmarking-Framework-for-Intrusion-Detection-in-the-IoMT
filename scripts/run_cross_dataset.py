#!/usr/bin/env python3
"""
Run zero-shot cross-dataset generalization evaluation.

CLI entry point that loads a trained model checkpoint and evaluates it
on cross-dataset targets without any retraining.

Usage:
    # Evaluate XGBoost (trained on CICIoMT2024 task 2) on all targets:
    python scripts/run_cross_dataset.py \
        --model XGBoost --task 2 \
        --checkpoint results/CICIoMT2024/task_2/XGBoost/model.pkl

    # Evaluate on a single target:
    python scripts/run_cross_dataset.py \
        --model RandomForest --task 6 \
        --checkpoint results/CICIoMT2024/task_6/RandomForest/model.pkl \
        --target CIC-ToN-IoT

    # Dry run (show configuration):
    python scripts/run_cross_dataset.py \
        --model XGBoost --task 2 \
        --checkpoint results/CICIoMT2024/task_2/XGBoost/model.pkl \
        --dry-run
"""

import argparse
import logging
import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.train import get_model_instance, load_config, MODEL_REGISTRY
from evaluation.cross_dataset import (
    run_cross_dataset_evaluation,
    run_all_cross_dataset,
)

ALL_TARGETS = ["CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on cross-dataset targets (zero-shot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All targets, binary classification:
  python scripts/run_cross_dataset.py --model XGBoost --task 2 \\
      --checkpoint results/CICIoMT2024/task_2/XGBoost/model.pkl

  # Single target, family classification:
  python scripts/run_cross_dataset.py --model RandomForest --task 6 \\
      --checkpoint results/CICIoMT2024/task_6/RandomForest/model.pkl \\
      --target CIC-ToN-IoT
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name (must match the trained checkpoint)",
    )
    parser.add_argument(
        "--task",
        type=int,
        required=True,
        choices=[2, 6],
        help="Classification task (2=binary, 6=families). Task 19 not applicable.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (model.pkl from training pipeline)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=ALL_TARGETS,
        help="Single target dataset. If omitted, evaluates on all three targets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root output directory (default: results)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: use debug/ output tree",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Apply debug overrides if requested
    if args.debug:
        from config.debug import apply_debug_overrides, ensure_debug_dirs
        config = apply_debug_overrides(config)
        ensure_debug_dirs()
        if args.output_dir == "results":
            args.output_dir = config["output"]["results_dir"]
    cache_dir = os.path.join(project_root, config["preprocessing"]["cache_dir"])

    targets = [args.target] if args.target else ALL_TARGETS

    # Print run info
    logging.info("=" * 60)
    logging.info("Cross-Dataset Generalization Evaluation")
    logging.info(f"  Model:      {args.model}")
    logging.info(f"  Task:       {args.task}")
    logging.info(f"  Checkpoint: {args.checkpoint}")
    logging.info(f"  Targets:    {targets}")
    logging.info(f"  Output:     {args.output_dir}")
    logging.info(f"  Transfer:   zero-shot (no retraining)")
    logging.info("=" * 60)

    if args.dry_run:
        logging.info("DRY RUN — exiting without evaluation.")
        return

    # Load trained model from checkpoint
    logging.info(f"Loading model checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Determine input_dim and n_classes from primary preprocessed data
    from preprocessing.preprocessing import load_preprocessed_data
    _, _, _, _, X_test_primary, _, p_meta = load_preprocessed_data(
        cache_dir, "CICIoMT2024", args.task
    )
    input_dim = X_test_primary.shape[1]
    n_classes = p_meta.get("n_classes", len(p_meta.get("label_classes", [])))

    model = get_model_instance(
        args.model,
        random_state=config["preprocessing"]["random_state"],
        input_dim=input_dim,
        n_classes=n_classes,
    )
    model.load_checkpoint(args.checkpoint)

    # Run evaluation
    total_start = time.time()

    try:
        if args.target:
            # Single target
            result = run_cross_dataset_evaluation(
                model=model,
                model_name=args.model,
                target_dataset=args.target,
                task=args.task,
                primary_cache_dir=cache_dir,
                output_dir=args.output_dir,
            )
            results = {args.target: result}
        else:
            # All targets
            results = run_all_cross_dataset(
                model=model,
                model_name=args.model,
                task=args.task,
                primary_cache_dir=cache_dir,
                output_dir=args.output_dir,
                target_datasets=targets,
            )

        total_elapsed = time.time() - total_start
        logging.info(f"Total wall time: {total_elapsed:.1f}s")

        # Print summary
        for ds, res in results.items():
            if "error" in res:
                logging.info(f"  {ds}: FAILED — {res['error']}")
            else:
                m = res["test_metrics"]
                logging.info(
                    f"  {ds}: F1(macro)={m['f1_macro']:.4f}, "
                    f"MCC={m['mcc']:.4f}"
                )

    except Exception as e:
        logging.error(f"FAILED: {args.model} cross-dataset task_{args.task}")
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
