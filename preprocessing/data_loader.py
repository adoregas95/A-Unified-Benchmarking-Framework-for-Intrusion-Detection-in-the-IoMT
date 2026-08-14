"""
Dataset-agnostic data loader for the IoMT IDS Framework.

Supports: CICIoMT2024, CIC-BoT-IoT, CIC-IoT-DIAD-2024, CIC-ToN-IoT
All four datasets use CICFlowMeter extraction (76 ML-usable features).
This loader handles label mapping, column name standardization, and
feature alignment across all four datasets.

Usage:
    from preprocessing.data_loader import load_dataset
    X_train, y_train, X_test, y_test, metadata = load_dataset(
        dataset="CICIoMT2024", task=19
    )
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


# =============================================================================
# Column Name Standardization
# =============================================================================
# CIC-BoT-IoT and CIC-ToN-IoT use abbreviated column names.
# CICIoMT2024 and CIC-IoT-DIAD-2024 use full names.
# This mapping renames abbreviated → full to standardize all datasets.

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


# =============================================================================
# Columns to drop (flow identifiers, not ML features)
# =============================================================================

FLOW_ID_COLS = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
    'Protocol', 'Timestamp',
]

# Label-related columns added by various datasets (dropped before returning features)
LABEL_COLS = ['Label', 'Class', 'Sub-Class', 'Attack', 'label']


# =============================================================================
# Label Mapping: CICIoMT2024
# =============================================================================

# The Sub-Class column in CICIoMT2024 CSVs contains the 19-class labels directly.
# These mappings handle task-level aggregation.

def _map_ciciomt_label(sub_class: str, task: int, class_col: str = "") -> str:
    """Map a CICIoMT2024 label to the appropriate task granularity.

    The CICFlowMeter-extracted CSVs have both 'Class' (6 families) and
    'Sub-Class' (15 unique names) columns. However, Sub-Class alone is
    NOT sufficient for 19-class classification because the 4 flood types
    (ICMP-Flood, SYN-Flood, TCP-Flood, UDP-Flood) are shared between
    the DDoS and DoS families. For example, both DDoS-ICMP and DoS-ICMP
    have Sub-Class="ICMP-Flood" — only the Class column distinguishes them.

    For task=19, we combine Class + Sub-Class to reconstruct the full
    19-class labels that match the DPKT version of the dataset used in
    the conference paper (e.g., "DDoS-ICMP-Flood", "DoS-ICMP-Flood").

    Args:
        sub_class: Value from the 'Sub-Class' column.
        task: Classification task (2=binary, 6=families, 19=individual).
        class_col: Value from the 'Class' column (6 attack families).

    Returns:
        Mapped label string.
    """
    if task == 2:
        return "Benign" if sub_class == "Benign" else "Attack"

    elif task == 6:
        # Use the dataset's own 'Class' column directly — it already
        # contains the correct 6-class grouping.
        if class_col:
            return class_col
        return sub_class  # fallback

    else:  # task == 19
        # Combine Class + Sub-Class to get the full 19-class label.
        # This is necessary because flood attacks share Sub-Class names
        # across DDoS and DoS families.
        if class_col and class_col in ("DDoS", "DoS"):
            # "DDoS" + "ICMP-Flood" → "DDoS-ICMP-Flood"
            # "DoS"  + "ICMP-Flood" → "DoS-ICMP-Flood"
            return f"{class_col}-{sub_class}"
        elif class_col and class_col == "MQTT":
            # "MQTT" + "DDoS-Connect-Flood" → "MQTT-DDoS-Connect-Flood"
            return f"{class_col}-{sub_class}"
        elif class_col and class_col == "Recon":
            # "Recon" + "OS-Scan" → "Recon-OS-Scan"
            return f"{class_col}-{sub_class}"
        else:
            # Benign stays "Benign", Spoofing sub-classes keep their name
            return sub_class


# =============================================================================
# Label Mapping: Cross-Dataset (for generalization experiments)
# =============================================================================

# CIC-BoT-IoT: Attack column has family names
BOTIOT_FAMILY_MAP = {
    'Benign': 'Benign',
    'DDoS': 'DDoS',
    'DoS': 'DoS',
    'Reconnaissance': 'Recon',
    'Theft': 'Theft',  # novel — not in CICIoMT2024
}

# CIC-ToN-IoT: Attack column has lowercase family names
TONIOT_FAMILY_MAP = {
    'Benign': 'Benign',
    'ddos': 'DDoS',
    'dos': 'DoS',
    'scanning': 'Recon',
    'injection': 'Injection',     # novel
    'mitm': 'MITM',               # novel
    'password': 'Password',       # novel
    'ransomware': 'Ransomware',   # novel
    'xss': 'XSS',                 # novel
    'backdoor': 'Backdoor',       # novel
}

# CIC-IoT-DIAD-2024: Labels derived from folder structure
DIAD_FOLDER_FAMILY_MAP = {
    'Benign': 'Benign',
    'DDoS': 'DDoS',
    'DoS': 'DoS',
    'Recon': 'Recon',
    'Spoofing': 'Spoofing',
    'BruteForce': 'BruteForce',   # novel
    'Mirai': 'Mirai',             # novel
    'Web-Based': 'Web-Based',     # novel
}


def _map_cross_dataset_label(family: str, task: int) -> str:
    """Map a cross-dataset family label to the task granularity."""
    if task == 2:
        return "Benign" if family == "Benign" else "Attack"
    else:
        # For task 6 (and task 19 on cross-dataset), return the family label
        return family


# =============================================================================
# Dataset Loaders
# =============================================================================

def _load_ciciomt2024(data_dir: str, split: str, task: int) -> pd.DataFrame:
    """
    Load CICIoMT2024 CICFlowMeter-extracted CSV files.

    Files are in data_dir/CSV/ with naming convention:
        {AttackType}_{split}.pcap_Flow.csv
    Each CSV has built-in Label, Class, Sub-Class columns.
    """
    csv_dir = os.path.join(data_dir, "CSV")
    # Match files for this split: *_train.pcap_Flow.csv or *_test.pcap_Flow.csv
    pattern = os.path.join(csv_dir, f"*_{split}.pcap_Flow.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found matching {pattern}. "
            f"Expected CICFlowMeter-extracted CSVs in {csv_dir}/"
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Validate required columns
        if "Sub-Class" not in df.columns:
            raise ValueError(
                f"CSV file {f} missing 'Sub-Class' column. "
                "Expected CICFlowMeter-extracted CSV with Label/Class/Sub-Class."
            )
        # For task=6 and task=19, we need BOTH the Class and Sub-Class columns:
        #   - task=6: Class column has the correct 6-family grouping directly
        #   - task=19: Class + Sub-Class must be combined because flood attacks
        #     (ICMP-Flood, SYN-Flood, TCP-Flood, UDP-Flood) share the same
        #     Sub-Class name across DDoS and DoS families
        # For task=2 (binary), Sub-Class alone suffices (everything → Attack/Benign)
        if "Class" in df.columns:
            df["label"] = df.apply(
                lambda row: _map_ciciomt_label(
                    row["Sub-Class"], task, class_col=row["Class"]
                ),
                axis=1,
            )
        else:
            df["label"] = df["Sub-Class"].apply(
                lambda x: _map_ciciomt_label(x, task)
            )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Drop flow identifiers and original label columns (keep our mapped 'label')
    cols_to_drop = [c for c in FLOW_ID_COLS + LABEL_COLS if c in combined.columns
                    and c != "label"]
    combined = combined.drop(columns=cols_to_drop)

    return combined


def _load_cic_bot_iot(data_dir: str, task: int) -> pd.DataFrame:
    """
    Load CIC-BoT-IoT dataset.

    Single CSV file: data/CIC-BoT-IoT.csv
    Columns use abbreviated CICFlowMeter names → rename to full names.
    Label column is binary (0/1), Attack column has family names.
    """
    csv_path = os.path.join(data_dir, "data", "CIC-BoT-IoT.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CIC-BoT-IoT data not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Rename abbreviated columns to full names
    df = df.rename(columns=ABBREVIATED_TO_FULL)

    # Drop rows with NaN in the Attack column (unmappable rows)
    n_before = len(df)
    df = df.dropna(subset=["Attack"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"CIC-BoT-IoT: dropped {n_dropped} rows with NaN Attack labels "
            f"({n_dropped/n_before:.4%} of {n_before} total)"
        )

    # Map labels
    family = df["Attack"].map(BOTIOT_FAMILY_MAP).fillna(df["Attack"])
    df["label"] = family.apply(lambda x: _map_cross_dataset_label(x, task))

    # Drop flow identifiers and original label columns
    cols_to_drop = [c for c in FLOW_ID_COLS + LABEL_COLS if c in df.columns
                    and c != "label"]
    df = df.drop(columns=cols_to_drop)

    # Force all remaining columns to numeric (some CIC-BoT-IoT columns have
    # mixed types due to malformed CSV rows where label values bleed into
    # feature columns). Coerce non-numeric values to NaN.
    feature_cols = [c for c in df.columns if c != "label"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _load_cic_iot_diad(data_dir: str, task: int) -> pd.DataFrame:
    """
    Load CIC-IoT-DIAD-2024 dataset.

    Organized by attack family subdirectories. Column names match CICIoMT2024.
    Label column contains 'NeedManualLabel' — derive from folder structure.

    Note: 5 files in DoS/DoS-TCP_Flood/ are missing their header row.
    We detect this and insert the standard header automatically.
    """
    # Standard header for CIC-IoT-DIAD-2024 (same as CICIoMT2024 minus Class/Sub-Class)
    DIAD_STANDARD_HEADER = [
        'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
        'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
        'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
        'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg',
        'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
        'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets',
        'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts',
        'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
        'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label',
    ]

    all_dfs = []
    for family_folder, family_label in DIAD_FOLDER_FAMILY_MAP.items():
        family_dir = os.path.join(data_dir, family_folder)
        if not os.path.isdir(family_dir):
            continue

        # Recursively find all CSVs under this family folder
        csv_files = glob.glob(os.path.join(family_dir, "**", "*.csv"), recursive=True)
        for csv_path in csv_files:
            # Detect headerless files (first column value looks like an IP/flow ID)
            with open(csv_path, 'r') as peek:
                first_line = peek.readline().strip()
            if not first_line.startswith('Flow ID'):
                # Missing header — read without header and assign standard column names
                df = pd.read_csv(csv_path, header=None, names=DIAD_STANDARD_HEADER)
            else:
                df = pd.read_csv(csv_path)
            df["label"] = _map_cross_dataset_label(family_label, task)
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop flow identifiers and original label columns
    cols_to_drop = [c for c in FLOW_ID_COLS + LABEL_COLS if c in combined.columns
                    and c != "label"]
    combined = combined.drop(columns=cols_to_drop)

    return combined


def _load_cic_ton_iot(data_dir: str, task: int) -> pd.DataFrame:
    """
    Load CIC-ToN-IoT dataset (CICFlowMeter version).

    Single CSV file: data/CIC-ToN-IoT.csv
    Columns use abbreviated CICFlowMeter names → rename to full names.
    Label column is binary (0/1), Attack column has lowercase family names.
    """
    csv_path = os.path.join(data_dir, "data", "CIC-ToN-IoT.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CIC-ToN-IoT data not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename abbreviated columns to full names
    df = df.rename(columns=ABBREVIATED_TO_FULL)

    # Map labels
    family = df["Attack"].map(TONIOT_FAMILY_MAP).fillna(df["Attack"])
    df["label"] = family.apply(lambda x: _map_cross_dataset_label(x, task))

    # Drop flow identifiers and original label columns
    cols_to_drop = [c for c in FLOW_ID_COLS + LABEL_COLS if c in df.columns
                    and c != "label"]
    df = df.drop(columns=cols_to_drop)

    return df


# =============================================================================
# Public API
# =============================================================================

def load_dataset(
    dataset: str,
    task: int,
    data_root: str = None,
    split_ratio: float = 0.8,
    cross_dataset_mode: str = "full_test",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load a dataset and return train/test splits with labels.

    For the PRIMARY dataset (CICIoMT2024):
        Loads ALL CSV files (both *_train and *_test), merges them, and
        performs a stratified re-split to ensure proportional class
        representation. The original PCAP-file-level split allocated 80%
        of files to training but produced a 96.2/3.8 sample-level split
        with extreme per-class imbalance in the test set (e.g., 14 Benign
        samples out of 254K test, DDoS-TCP-Flood with only 11 test
        samples out of 861K total). The stratified re-split fixes this.

    For CROSS-DATASET targets (CIC-BoT-IoT, CIC-IoT-DIAD-2024, CIC-ToN-IoT):
        Two modes are available via cross_dataset_mode:

        - "full_test" (default): The ENTIRE dataset is returned as the test set,
          and the training set is returned EMPTY (0 rows). This is the correct
          mode for cross-dataset generalization experiments, where models trained
          on CICIoMT2024 are evaluated on the full cross-dataset target. There is
          no train/test contamination risk because the model was never exposed to
          any data from these datasets during training.

        - "split": A stratified 80/20 split is performed, creating train/test
          subsets. This mode would be used if you wanted to train a model directly
          on a cross-dataset target (e.g., for a within-dataset baseline comparison)
          or for potential future fine-tuning experiments.

    Args:
        dataset: One of "CICIoMT2024", "CIC-BoT-IoT", "CIC-IoT-DIAD-2024",
                 "CIC-ToN-IoT"
        task: Classification task (2=binary, 6=families, 19=individual attacks)
        data_root: Root directory containing dataset folders (defaults to
                   project's data/ directory)
        split_ratio: Train/test split ratio when cross_dataset_mode="split"
                     (default 0.8 = 80% train, 20% test). Ignored for
                     CICIoMT2024 and when cross_dataset_mode="full_test".
        cross_dataset_mode: How to handle cross-dataset targets:
                     "full_test" = entire dataset as test (for generalization)
                     "split" = stratified train/test split (for within-dataset use)

    Returns:
        X_train: Training features (DataFrame, may be empty for "full_test" mode)
        y_train: Training labels (Series)
        X_test: Test features (DataFrame)
        y_test: Test labels (Series)
        metadata: Dict with feature_names, label_encoder, n_classes,
                  class_names, train_samples, test_samples, mode
    """
    # Default to the project's data/ directory if no root is specified
    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )

    if dataset == "CICIoMT2024":
        data_dir = os.path.join(data_root, "CICIoMT2024")

        # Load ALL CICIoMT2024 data (both train and test files) and merge,
        # then perform a stratified re-split to ensure proportional class
        # representation in both sets.
        #
        # Why re-split: The original PCAP-file-level 80/20 split produces a
        # 96.2/3.8 sample-level split with extreme class imbalance in the
        # test set (e.g., 14 Benign samples out of 254K, DDoS-TCP-Flood
        # with only 11 test samples out of 861K). This is because the
        # authors allocated 80% of *files* (not samples) to training, and
        # file sizes vary by orders of magnitude. The paper does not
        # document any session-level or temporal separation between train
        # and test files — it was an arbitrary file allocation.
        # See data_exploration/01_train_test_distribution.py for analysis.
        train_df = _load_ciciomt2024(data_dir, "train", task)
        test_df = _load_ciciomt2024(data_dir, "test", task)

        # Merge and re-split with stratification
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        train_df, test_df = _stratified_split(full_df, split_ratio)

    elif dataset in ("CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT"):
        # Cross-dataset targets: load the full dataset first, then decide
        # how to split based on cross_dataset_mode
        if dataset == "CIC-BoT-IoT":
            data_dir = os.path.join(data_root, "CIC-BoT-IoT")
            full_df = _load_cic_bot_iot(data_dir, task)
        elif dataset == "CIC-IoT-DIAD-2024":
            data_dir = os.path.join(data_root, "CIC-IoT-DIAD-2024")
            full_df = _load_cic_iot_diad(data_dir, task)
        else:  # CIC-ToN-IoT
            data_dir = os.path.join(data_root, "CIC-ToN-IoT")
            full_df = _load_cic_ton_iot(data_dir, task)

        if cross_dataset_mode == "full_test":
            # Cross-dataset generalization: entire dataset becomes the test set.
            # Training set is empty because the model was trained on CICIoMT2024.
            # This is the standard mode for evaluating cross-dataset transfer.
            test_df = full_df
            # Create an empty DataFrame with matching columns for consistency
            train_df = full_df.iloc[:0].copy()
        elif cross_dataset_mode == "split":
            # Within-dataset mode: create a train/test split for cases where
            # you want to train directly on this dataset (e.g., baselines)
            train_df, test_df = _stratified_split(full_df, split_ratio)
        else:
            raise ValueError(
                f"Unknown cross_dataset_mode: '{cross_dataset_mode}'. "
                f"Use 'full_test' or 'split'."
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Separate features (X) from labels (y)
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # Get the list of feature column names (should be 76 for all datasets)
    feature_names = X_test.columns.tolist() if len(X_train) == 0 else X_train.columns.tolist()

    # Encode all unique labels across both splits for consistent mapping
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]) if len(y_train) > 0 else y_test)
    n_classes = len(le.classes_)

    # Build metadata dict for downstream use
    metadata = {
        "dataset": dataset,
        "task": task,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "label_encoder": le,
        "class_names": list(le.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cross_dataset_mode": cross_dataset_mode if dataset != "CICIoMT2024" else "stratified_resplit",
    }

    return X_train, y_train, X_test, y_test, metadata


def _stratified_split(
    df: pd.DataFrame, train_ratio: float = 0.8, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split for datasets without predefined splits.

    Used when cross_dataset_mode="split" — creates a random but reproducible
    split that preserves class proportions. Not used for the primary
    cross-dataset generalization experiments (which use full_test mode).

    Args:
        df: Full dataset DataFrame with a "label" column
        train_ratio: Fraction of data to use for training (default 0.8)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        train_df, test_df: Split DataFrames with reset indices
    """
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, train_size=train_ratio, stratify=df["label"],
        random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
