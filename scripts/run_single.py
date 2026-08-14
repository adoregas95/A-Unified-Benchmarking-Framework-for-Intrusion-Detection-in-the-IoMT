#!/usr/bin/env python3
"""
Run a single (model, dataset, task) training pipeline.

CLI entry point that wraps training.train.run_training_pipeline().
Designed to be called from SLURM job scripts.

Usage:
    # Train RandomForest on CICIoMT2024 binary classification:
    python scripts/run_single.py --model RandomForest --dataset CICIoMT2024 --task 2

    # Train XGBoost on 6-class with custom output directory:
    python scripts/run_single.py --model XGBoost --dataset CICIoMT2024 --task 6 \
        --output-dir results/experiment_v2

    # Dry run (show what would be executed without running):
    python scripts/run_single.py --model LightGBM --dataset CICIoMT2024 --task 19 --dry-run
"""

import argparse
import logging
import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.train import run_training_pipeline, load_config, MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Train a single model on a specific dataset and task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tree-based models:
  python scripts/run_single.py --model RandomForest --dataset CICIoMT2024 --task 2
  python scripts/run_single.py --model XGBoost --dataset CICIoMT2024 --task 6
  python scripts/run_single.py --model LightGBM --dataset CICIoMT2024 --task 19
  python scripts/run_single.py --model CatBoost --dataset CICIoMT2024 --task 2

  # Show what would run without executing:
  python scripts/run_single.py --model RandomForest --dataset CICIoMT2024 --task 2 --dry-run
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CICIoMT2024", "CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--task",
        type=int,
        required=True,
        choices=[2, 6, 19],
        help="Classification task (2=binary, 6=families, 19=individual)",
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
        help="Debug mode: use debug/ output tree with reduced HPO/epochs",
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
        # Override output_dir to debug tree (unless user explicitly set it)
        if args.output_dir == "results":
            args.output_dir = config["output"]["results_dir"]

    # Print run info
    logging.info("=" * 60)
    logging.info(f"Model:   {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Task:    {args.task}")
    logging.info(f"Output:  {args.output_dir}")
    logging.info(f"Config:  {args.config or 'config/config.yaml'}")
    logging.info("=" * 60)

    if args.dry_run:
        logging.info("DRY RUN — exiting without training.")
        return

    # Run the pipeline
    total_start = time.time()

    try:
        results = run_training_pipeline(
            model_name=args.model,
            dataset=args.dataset,
            task=args.task,
            config=config,
            output_dir=args.output_dir,
        )

        total_elapsed = time.time() - total_start
        logging.info(f"Total wall time: {total_elapsed:.1f}s")
        logging.info(
            f"Test F1(weighted): {results['test_metrics']['f1_weighted']:.4f}"
        )
        logging.info(
            f"Test Accuracy:     {results['test_metrics']['accuracy']:.4f}"
        )

    except Exception as e:
        logging.error(f"FAILED: {args.model}/{args.dataset}/task_{args.task}")
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
