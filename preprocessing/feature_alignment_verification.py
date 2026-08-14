#!/usr/bin/env python3
"""
Feature Alignment Verification Across All 4 Datasets
=====================================================
Oswald Adohinzin — Dakota State University
Dissertation: A Unified Benchmarking Framework for IDS in the IoMT

This script verifies that all four datasets share the same 76 ML-usable
CICFlowMeter features, enabling cross-dataset generalization experiments.

Datasets:
  1. CICIoMT2024      — Re-extracted from PCAP with Java CICFlowMeter v4.0
  2. CIC-BoT-IoT      — CICFlowMeter version (abbreviated column names)
  3. CIC-IoT-DIAD-2024 — CICFlowMeter extraction (full column names)
  4. CIC-ToN-IoT      — CICFlowMeter version (abbreviated column names)

Usage:
  python feature_alignment_verification.py [--data-root /path/to/data]
"""

import os
import sys
import csv
import glob
import argparse
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────────────────────
# Column name mapping: abbreviated (BoT-IoT, ToN-IoT) → full (IoMT, DIAD)
# ─────────────────────────────────────────────────────────────────────────────

ABBREVIATED_TO_FULL = {
    'Tot Fwd Pkts': 'Total Fwd Packet',
    'Tot Bwd Pkts': 'Total Bwd packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packet',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packet',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Fwd Header Len': 'Fwd Header Length',
    'Bwd Header Len': 'Bwd Header Length',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Pkt Len Min': 'Packet Length Min',
    'Pkt Len Max': 'Packet Length Max',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std',
    'Pkt Len Var': 'Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'CWE Flag Count': 'CWR Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count',
    'Pkt Size Avg': 'Average Packet Size',
    'Fwd Seg Size Avg': 'Fwd Segment Size Avg',
    'Bwd Seg Size Avg': 'Bwd Segment Size Avg',
    'Fwd Byts/b Avg': 'Fwd Bytes/Bulk Avg',
    'Fwd Pkts/b Avg': 'Fwd Packet/Bulk Avg',
    'Fwd Blk Rate Avg': 'Fwd Bulk Rate Avg',
    'Bwd Byts/b Avg': 'Bwd Bytes/Bulk Avg',
    'Bwd Pkts/b Avg': 'Bwd Packet/Bulk Avg',
    'Bwd Blk Rate Avg': 'Bwd Bulk Rate Avg',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes',
    'Subflow Bwd Pkts': 'Subflow Bwd Packets',
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'FWD Init Win Bytes',
    'Init Bwd Win Byts': 'Bwd Init Win Bytes',
}

FLOW_ID_COLS = {'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp'}
LABEL_COLS = {'Label', 'Class', 'Sub-Class', 'Attack'}


def read_header(csv_path):
    """Read just the header row of a CSV file."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return next(reader)


def count_rows(csv_path):
    """Count data rows (excluding header) in a CSV file."""
    count = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for _ in reader:
            count += 1
    return count


def get_ml_features(columns):
    """Extract ML-usable feature names from a column list, applying name standardization."""
    features = []
    for col in columns:
        col = col.strip()
        if col in FLOW_ID_COLS or col in LABEL_COLS:
            continue
        # Standardize abbreviated names
        standardized = ABBREVIATED_TO_FULL.get(col, col)
        features.append(standardized)
    return features


def verify_dataset(name, csv_path, is_directory=False):
    """Verify a dataset and return its ML features."""
    print(f"\n{'─' * 60}")
    print(f"  Dataset: {name}")
    print(f"{'─' * 60}")

    if is_directory:
        files = glob.glob(os.path.join(csv_path, "**", "*.csv"), recursive=True)
        # Exclude feature description files
        files = [f for f in files if 'Features' not in os.path.basename(f)]
        if not files:
            print(f"  ⚠ No CSV files found in {csv_path}")
            return None
        print(f"  Files found: {len(files)}")

        # Read header from first file
        header = read_header(files[0])
        print(f"  Total columns: {len(header)}")
        print(f"  Sample file: {os.path.basename(files[0])}")

        # Verify all files have same columns
        mismatches = []
        for f in files[1:]:
            h = read_header(f)
            if h != header:
                mismatches.append(os.path.basename(f))
        if mismatches:
            print(f"  ⚠ Column mismatches in {len(mismatches)} files!")
        else:
            print(f"  Column consistency: ALL {len(files)} FILES MATCH ✓")

    else:
        if not os.path.exists(csv_path):
            print(f"  ⚠ File not found: {csv_path}")
            return None
        header = read_header(csv_path)
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
        print(f"  Total columns: {len(header)}")

    # Get ML features
    raw_features = [c.strip() for c in header if c.strip() not in FLOW_ID_COLS and c.strip() not in LABEL_COLS]
    ml_features = get_ml_features(header)

    # Count columns needing rename
    renamed = sum(1 for c in raw_features if ABBREVIATED_TO_FULL.get(c, c) != c)

    print(f"  Flow ID columns: {sum(1 for c in header if c.strip() in FLOW_ID_COLS)}")
    print(f"  Label columns: {sum(1 for c in header if c.strip() in LABEL_COLS)}")
    print(f"  ML features: {len(ml_features)}")
    if renamed > 0:
        print(f"  Columns needing rename: {renamed} (abbreviated → full names)")
    else:
        print(f"  Column names: Already in standard format ✓")

    return ml_features


def main():
    parser = argparse.ArgumentParser(description="Feature alignment verification")
    parser.add_argument("--data-root", default=None,
                        help="Root data directory (default: ../data relative to this script)")
    args = parser.parse_args()

    if args.data_root:
        data_root = args.data_root
    else:
        data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    data_root = os.path.abspath(data_root)
    print("=" * 60)
    print("  FEATURE ALIGNMENT VERIFICATION")
    print("  IoMT IDS Unified Benchmarking Framework")
    print("=" * 60)
    print(f"\n  Data root: {data_root}")

    # ─── Verify each dataset ───
    results = OrderedDict()

    # 1. CICIoMT2024
    ciciomt_dir = os.path.join(data_root, "CICIoMT2024", "CSV")
    results["CICIoMT2024"] = verify_dataset("CICIoMT2024", ciciomt_dir, is_directory=True)

    # 2. CIC-BoT-IoT
    botiot_csv = os.path.join(data_root, "CIC-BoT-IoT", "data", "CIC-BoT-IoT.csv")
    results["CIC-BoT-IoT"] = verify_dataset("CIC-BoT-IoT", botiot_csv, is_directory=False)

    # 3. CIC-IoT-DIAD-2024
    diad_dir = os.path.join(data_root, "CIC-IoT-DIAD-2024")
    results["CIC-IoT-DIAD-2024"] = verify_dataset("CIC-IoT-DIAD-2024", diad_dir, is_directory=True)

    # 4. CIC-ToN-IoT
    toniot_csv = os.path.join(data_root, "CIC-ToN-IoT", "data", "CIC-ToN-IoT.csv")
    results["CIC-ToN-IoT"] = verify_dataset("CIC-ToN-IoT", toniot_csv, is_directory=False)

    # ─── Cross-dataset alignment check ───
    print(f"\n{'=' * 60}")
    print("  CROSS-DATASET ALIGNMENT")
    print(f"{'=' * 60}")

    available = {k: v for k, v in results.items() if v is not None}
    if len(available) < 2:
        print("\n  ⚠ Need at least 2 datasets for alignment check.")
        return

    # Use CICIoMT2024 as reference
    ref_name = "CICIoMT2024"
    if ref_name not in available:
        ref_name = list(available.keys())[0]
    ref_features = available[ref_name]

    all_match = True
    for name, features in available.items():
        if name == ref_name:
            continue
        if features == ref_features:
            print(f"\n  {name} vs {ref_name}: PERFECT MATCH ✓")
            print(f"    All {len(ref_features)} ML features aligned.")
        else:
            all_match = False
            # Find differences
            ref_set = set(ref_features)
            other_set = set(features)
            only_ref = ref_set - other_set
            only_other = other_set - ref_set
            print(f"\n  {name} vs {ref_name}: MISMATCH ✗")
            if only_ref:
                print(f"    Only in {ref_name}: {only_ref}")
            if only_other:
                print(f"    Only in {name}: {only_other}")

    # ─── Final Summary ───
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  Datasets verified: {len(available)} / 4")
    print(f"  Reference features: {len(ref_features)} ML-usable CICFlowMeter features")
    print(f"  Feature list (standardized names):")
    for i, feat in enumerate(ref_features, 1):
        print(f"    {i:2d}. {feat}")

    if all_match:
        print(f"\n  ✓ ALL DATASETS ALIGNED — {len(ref_features)} shared ML features")
        print(f"  ✓ Cross-dataset generalization experiments can proceed.")
    else:
        print(f"\n  ✗ ALIGNMENT ISSUES DETECTED — see details above.")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
