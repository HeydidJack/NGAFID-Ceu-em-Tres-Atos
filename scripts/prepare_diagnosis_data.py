#!/usr/bin/env python3
"""
NGAFID Diagnosis Dataset Preparation Script
==========================================

Unified script for preprocessing NGAFID aviation maintenance data into
5-fold stratified cross-validation sets for end-to-end diagnosis tasks.

Supports two dataset modes:
  - subset: 2-day subset (2days.tar.gz) from Zenodo, ~1.1GB raw
  - full: Complete flight records (all_flight.tar.gz), ~4.3GB raw

Usage:
  python scripts/prepare_diagnosis_data.py --dataset subset --seed 350234
  python scripts/prepare_diagnosis_data.py --dataset full --seed 350234

Arguments:
  --dataset {subset,full}  Select dataset to process (required)
  --seed INT              Random seed for 5-fold split (default: 350234)
  --output DIR            Output directory (default: ../ProcessedData)

Outputs:
  ProcessedData/
  ├── ProcessedData_Diagnosis_DeD_2days/      (if subset selected)
  │   ├── diagnosis_fold{0-4}_2048.pkl       # 5-fold data, 2048 timesteps
  │   ├── fault_mapping_config.pkl           # Dynamic label mapping
  │   ├── fault_mapping_table.txt            # Human-readable label table
  │   └── random_seed.txt
  └── ProcessedData_Diagnosis_DeD_full/       (if full selected)
      └── [same structure]

Data Flow:
  Raw CSV/Parquet → Dynamic Label Mapping → NaN Handling → Resampling (2048)
                  → Stratified 5-Fold Split → Pickle Serialization
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pyarrow.parquet as pq

# Sensor channel definitions (23 variables matching NGAFID standard)
ITEM_MEANINGS = [
    "volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow",
    "E1 OilT", "E1 OilP", "E1 RPM", "E1 CHT1", "E1 CHT2", "E1 CHT3",
    "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4", "OAT",
    "IAS", "VSpd", "NormAc", "AltMSL"
]

HEALTHY_LABEL = "flights recorded after maintenance"


class DynamicFaultMapper:
    """
    Dynamic fault-to-ID mapper assigning integer labels based on empirical
    dataset distribution. Guarantees consistent ID mapping across runs.
    """

    def __init__(self, healthy_id=0):
        self.healthy_id = healthy_id
        self.fault_to_id = {}
        self.num_classes = None

    def fit(self, label_dict):
        """Analyze dataset statistics and assign IDs by frequency (descending)."""
        fault_counts = {}
        for fault_type, indices in label_dict.items():
            if fault_type != HEALTHY_LABEL:
                fault_counts[fault_type] = len(indices)

        # Sort by sample count (majority classes get lower IDs)
        sorted_faults = sorted(fault_counts.items(), key=lambda x: x[1], reverse=True)

        current_id = 1
        for fault_type, count in sorted_faults:
            self.fault_to_id[fault_type] = current_id
            current_id += 1

        self.num_classes = len(self.fault_to_id) + 1  # +1 for healthy class
        return self

    def get_id(self, fault_type):
        """Retrieve numeric ID for fault string."""
        return self.healthy_id if fault_type == HEALTHY_LABEL else self.fault_to_id.get(fault_type)

    def onehot(self, fault_type):
        """Generate one-hot encoded vector for fault type."""
        vec = np.zeros(self.num_classes)
        vec[self.get_id(fault_type)] = 1.0
        return vec

    def get_config(self):
        """Export mapping configuration for reproducibility."""
        return {
            'healthy_id': self.healthy_id,
            'healthy_label': HEALTHY_LABEL,
            'fault_to_id': self.fault_to_id,
            'num_classes': self.num_classes
        }


def load_subset_data(base_dir):
    """Load 2-day subset from pickled Python objects."""
    subset_dir = Path(base_dir) / "Datasets" / "2days" / "2days"
    header_df = pd.read_csv(subset_dir / "flight_header.csv")

    with open(subset_dir / "flight_data.pkl", "rb") as f:
        flight_data = pickle.load(f)

    return flight_data, header_df


def load_full_data(base_dir):
    """Load complete dataset from partitioned Parquet files."""
    full_dir = Path(base_dir) / "Datasets" / "all_flights" / "all_flights" / "one_parq"
    header_path = Path(base_dir) / "Datasets" / "one_parq" / "flight_header.csv"

    header_df = pd.read_csv(header_path)
    flight_dict = {}

    # Process 401 partitioned Parquet files (memory-efficient streaming)
    for pid in range(401):
        pq_path = full_dir / f"part.{pid}.parquet"
        if not pq_path.exists():
            continue

        table = pq.read_table(pq_path)
        df = table.to_pandas().iloc[:, :23]  # Select sensor channels only

        # Group by flight ID and convert to numpy
        for fid, group in df.groupby('Master Index'):
            flight_dict[fid] = group.values.astype(np.float32)

    return flight_dict, header_df


def extract_labels(header_path):
    """Parse flight header CSV to create fault-type-to-flight-indices mapping."""
    df = pd.read_csv(header_path)
    label_map = {}

    for _, row in df.iterrows():
        # -1 indicates post-maintenance (healthy) flight
        label = HEALTHY_LABEL if row['number_flights_before'] == -1 else row['label']

        if label not in label_map:
            label_map[label] = []
        label_map[label].append(row['Master Index'])

    return label_map


def forward_fill(arr):
    """Impute missing values using forward-fill strategy."""
    arr = np.array(arr, copy=True)
    n_samples, n_features = arr.shape

    for col in range(n_features):
        if np.isnan(arr[0, col]):
            arr[0, col] = 0.0
        for i in range(1, n_samples):
            if np.isnan(arr[i, col]):
                arr[i, col] = arr[i - 1, col]
    return arr


def resample_sequence(seq, target_len=2048, degree=3):
    """
    Resample variable-length flight sequences to fixed length using
    polynomial interpolation (cubic by default).
    """
    orig_len = len(seq)
    x_old = np.linspace(0, orig_len - 1, orig_len)
    x_new = np.linspace(0, orig_len - 1, target_len)

    resampled = np.zeros((target_len, seq.shape[1]))
    kind = {3: 'cubic', 2: 'quadratic', 1: 'linear'}[degree]

    for i in range(seq.shape[1]):
        interp_fn = interp1d(x_old, seq[:, i], kind=kind)
        resampled[:, i] = interp_fn(x_new)

    return resampled


def filter_and_label(label_dict, flight_data, mapper, min_len=60, nan_thresh=0.1):
    """
    Quality control pipeline:
    1. Filter sequences shorter than min_len (seconds)
    2. Reject flights with >10% NaN values in any sensor
    3. Assign one-hot encoded labels via dynamic mapper
    """
    train_data, raw_lengths, labels = [], [], []
    stats = {'total': 0, 'rejected': 0}

    for fault_type, flight_ids in label_dict.items():
        for fid in flight_ids:
            try:
                seq = flight_data.get(fid)
                if seq is None:
                    stats['rejected'] += 1
                    continue

                # Length validation
                if len(seq) < min_len:
                    stats['rejected'] += 1
                    continue

                # NaN ratio validation
                nan_ratio = np.isnan(seq).sum(axis=0) / len(seq)
                if (nan_ratio > nan_thresh).any():
                    stats['rejected'] += 1
                    continue

                train_data.append(seq)
                raw_lengths.append(len(seq))
                labels.append(mapper.onehot(fault_type))
                stats['total'] += 1

            except Exception as e:
                stats['rejected'] += 1
                continue

    return train_data, raw_lengths, labels, stats


def generate_folds(data_list, label_list, length_list, seed):
    """
    Generate stratified 5-fold splits ensuring class distribution
    preservation across train/validation sets.
    """
    # Preprocess: imputation + resampling to 2048 timesteps
    proc_data = []
    for seq in data_list:
        clean = forward_fill(seq)
        proc_data.append(resample_sequence(clean, 2048))

    X = np.array(proc_data)
    y = np.array(label_list).argmax(axis=1)  # Convert one-hot to class indices

    # Stratified split to handle class imbalance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(X)), y)):
        fold_data = {
            'index': val_idx.tolist(),
            'rawsl': [length_list[i] for i in val_idx],
            'train': X[val_idx].tolist(),
            'label': np.array(label_list)[val_idx].tolist()
        }
        folds.append(fold_data)

    return folds


def save_outputs(folds, output_dir, label_dict, seed, mapper):
    """Serialize processed data and metadata."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save fold data
    for idx, fold in enumerate(folds):
        path = Path(output_dir) / f"diagnosis_fold{idx}_2048.pkl"
        with open(path, 'wb') as f:
            pickle.dump(fold, f)

    # Save mapping configuration
    config_path = Path(output_dir) / "fault_mapping_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(mapper.get_config(), f)

    # Save human-readable mapping table
    table_path = Path(output_dir) / "fault_mapping_table.txt"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("NGAFID Dynamic Fault Mapping\n")
        f.write(f"Healthy ID: {mapper.healthy_id} -> {HEALTHY_LABEL}\n")
        f.write(f"Total Classes: {mapper.num_classes}\n\n")
        for fault, fid in mapper.fault_to_id.items():
            count = len(label_dict.get(fault, []))
            f.write(f"ID {fid:2d}: {fault:50s} (n={count:5d})\n")

    # Save seed
    with open(Path(output_dir) / "random_seed.txt", 'w') as f:
        f.write(str(seed))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NGAFID diagnosis datasets with 5-fold CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_diagnosis_data.py --dataset subset --seed 350234
  python prepare_diagnosis_data.py --dataset full --output ./ProcessedData
        """
    )
    parser.add_argument('--dataset', required=True, choices=['subset', 'full'],
                        help='Select dataset variant: subset (2days) or full (all_flights)')
    parser.add_argument('--seed', type=int, default=350234,
                        help='Random seed for stratified split (default: 350234)')
    parser.add_argument('--output', default='./ProcessedData',
                        help='Output directory path (default: ./ProcessedData)')

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent  # Repository root

    # Dataset-specific configuration
    if args.dataset == 'subset':
        print("Processing 2-day subset (DiagnosisDeDData)...")
        flight_data, header_df = load_subset_data(base_dir)
        header_path = base_dir / "Datasets" / "2days" / "2days" / "flight_header.csv"
        output_name = "ProcessedData_Diagnosis_DeD_2days"
    else:
        print("Processing full dataset (DiagnosisDeDFullData)...")
        flight_data, header_df = load_full_data(base_dir)
        header_path = base_dir / "Datasets" / "one_parq" / "flight_header.csv"
        output_name = "ProcessedData_Diagnosis_DeD_full"

    output_path = Path(args.output) / output_name

    # Label extraction and dynamic mapping
    label_dict = extract_labels(header_path)
    print(f"Found {len(label_dict)} fault categories")

    mapper = DynamicFaultMapper()
    mapper.fit(label_dict)
    print(f"Dynamic mapping: {mapper.num_classes} total classes")

    # Data filtering and labeling
    train_data, raw_lengths, labels, stats = filter_and_label(
        label_dict, flight_data, mapper
    )
    print(f"Accepted: {stats['total']}, Rejected: {stats['rejected']}")

    # Generate stratified folds
    print(f"Generating 5-fold stratified split (seed={args.seed})...")
    folds = generate_folds(train_data, labels, raw_lengths, args.seed)

    # Save outputs
    print(f"Saving to {output_path}...")
    save_outputs(folds, output_path, label_dict, args.seed, mapper)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()