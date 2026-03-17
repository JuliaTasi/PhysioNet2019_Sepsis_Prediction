#!/usr/bin/env python3
"""
run_baseline.py
===============
CLI script to run different imputation baselines and evaluate downstream sepsis prediction.

Usage:
    # Config-only                                                                                                                                     
    python3 src/experiments/run_baseline.py --config configs/baseline1.yaml                                                                           
                                                                                                                                                    
    # Config + CLI override                                                                                                                           
    python3 src/experiments/run_baseline.py --config configs/baseline3.yaml --top_k 10                                                                
                                                                                                                                                    
    # CLI-only (still works)                                                                                                                          
    python3 src/experiments/run_baseline.py --baseline 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from definitions import ROOT_DIR
from src.data.dataset import TimeSeriesDataset
from src.omni.functions import load_pickle
from src.features.derived_features import shock_index, partial_sofa
from src.features.rolling import RollingStatistic
from src.external.evaluate_sepsis_score import compute_prediction_utility


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline imputation experiments for sepsis prediction.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override config values.")
    parser.add_argument("--baseline", type=str, choices=["1", "2-1", "2-2", "3-1", "3-2", "4"],
                        help="Which baseline to run (1/2-1/2-2/3-1/3-2/4)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to experiment data (relative to project root)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results and models (relative to project root)")
    parser.add_argument("--external_ratio", type=float, default=None,
                        help="Ratio of external data")
    parser.add_argument("--downsample_rate", type=int, default=None,
                        help="Downsample interval in hours")
    parser.add_argument("--impute_method", type=str, default=None,
                        help="Imputation method for baseline 2")
    parser.add_argument("--similarity_metric", type=str, default=None,
                        help="Similarity metric for baseline 3")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Number of similar patients for weighted avg (baseline 3)")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters for baseline 3 (default 20)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Defaults
    defaults = dict(
        baseline=None,
        data_path="data/experiment",
        output_dir="results",
        external_ratio=0.5,
        downsample_rate=8,
        impute_method="saits",
        similarity_metric="soft_dtw",
        top_k=5,
        n_clusters=50,
        seed=42,
    )

    # Load YAML config if provided
    if args.config is not None:
        config_path = args.config if os.path.isabs(args.config) \
            else os.path.join(ROOT_DIR, args.config)
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f) or {}
        defaults.update(yaml_cfg)

    # CLI args override config/defaults (only if explicitly provided)
    for key, default_val in defaults.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val

    # Write merged values back to args namespace
    for key, val in defaults.items():
        setattr(args, key, val)

    if args.baseline is None:
        parser.error("--baseline is required (either via CLI or config file)")

    return args


def resolve_path(path):
    """Resolve a path: absolute stays absolute, relative is joined with ROOT_DIR."""
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_expert_features(dataset):
    """Add derived features and rolling statistics to a dataset."""
    dataset['ShockIndex'] = shock_index(dataset)
    dataset['PartialSOFA'] = partial_sofa(dataset)
    dataset['MaxShockIndex'] = RollingStatistic(statistic='max', window_length=5).transform(dataset['SBP'])
    dataset['MinHR'] = RollingStatistic(statistic='min', window_length=8).transform(dataset['HR'])
    return dataset


def save_results(output_dir, baseline, metrics, clf=None, imputation_model=None, args=None):
    """Save performance metrics, RF classifier, and optional imputation model."""
    output_dir = resolve_path(output_dir)
    baseline_dir = os.path.join(output_dir, f"baseline_{baseline}_full")
    os.makedirs(baseline_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics as JSON
    metrics_path = os.path.join(baseline_dir, f"metrics_{timestamp}.json")
    record = {
        "baseline": baseline,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    if args is not None:
        record["config"] = {k: v for k, v in vars(args).items() if k != "config"}
    with open(metrics_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save RF classifier
    if clf is not None:
        clf_path = os.path.join(baseline_dir, f"rf_model_{timestamp}.pkl")
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f, protocol=4)
        print(f"RF model saved to {clf_path}")

    # Save imputation model (baseline-specific)
    if imputation_model is not None:
        imp_path = os.path.join(baseline_dir, f"imputation_model_{timestamp}.pkl")
        with open(imp_path, 'wb') as f:
            pickle.dump(imputation_model, f, protocol=4)
        print(f"Imputation model saved to {imp_path}")

    return baseline_dir


def compute_normalized_utility(labels_per_patient, preds_per_patient):
    """Compute the official PhysioNet 2019 normalized utility score.

    Args:
        labels_per_patient: list of 1D arrays, one per patient.
        preds_per_patient: list of 1D arrays (binary), one per patient.

    Returns:
        Normalized utility score (float).
    """
    observed = np.zeros(len(labels_per_patient))
    best = np.zeros(len(labels_per_patient))
    inaction = np.zeros(len(labels_per_patient))

    dt_early, dt_optimal, dt_late = -12, -6, 3

    for k, (lab, pred) in enumerate(zip(labels_per_patient, preds_per_patient)):
        lab = np.asarray(lab, dtype=float)
        pred = np.asarray(pred, dtype=float)
        n = len(lab)

        best_pred = np.zeros(n)
        if np.any(lab):
            t_sepsis = np.argmax(lab) - dt_optimal
            best_pred[max(0, t_sepsis + dt_early):min(t_sepsis + dt_late + 1, n)] = 1

        observed[k] = compute_prediction_utility(lab, pred, check_errors=False)
        best[k] = compute_prediction_utility(lab, best_pred, check_errors=False)
        inaction[k] = compute_prediction_utility(lab, np.zeros(n), check_errors=False)

    total_observed = np.sum(observed)
    total_best = np.sum(best)
    total_inaction = np.sum(inaction)

    if total_best - total_inaction == 0:
        return 0.0
    return (total_observed - total_inaction) / (total_best - total_inaction)


def train_and_evaluate(train_dataset, train_labels, test_dataset, test_labels, seed=42):
    """Shared downstream pipeline: convert to ML format, fill NaN, train RF, evaluate.

    Returns (clf, metrics_dict) so callers can save them.
    """
    X_train = train_dataset.to_ml()
    X_train[torch.isnan(X_train)] = -1000

    X_test = test_dataset.to_ml()
    X_test[torch.isnan(X_test)] = -1000

    clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    clf.fit(X_train.numpy(), np.array(train_labels))

    probas = clf.predict_proba(X_test.numpy())
    preds = (probas[:, 1] > 0.5).astype(int)

    y_test = np.array(test_labels)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas[:, 1])

    # Split predictions/labels back to per-patient for utility score
    lengths = test_dataset.lengths
    labels_per_patient = []
    preds_per_patient = []
    idx = 0
    for l in lengths:
        labels_per_patient.append(y_test[idx:idx + l])
        preds_per_patient.append(preds[idx:idx + l])
        idx += l

    utility = compute_normalized_utility(labels_per_patient, preds_per_patient)

    print(f"Accuracy: {acc * 100:.3f}%")
    print(f"AUC:      {auc:.3f}")
    print(f"Utility:  {utility:.4f}")

    metrics = {"accuracy": acc, "auc": auc, "utility": utility}
    return clf, metrics


def load_data(data_dir):
    """Load train internal, train external, and test datasets + labels."""
    train_int = TimeSeriesDataset().load(data_dir + '/train_internal_full.tsd')
    labels_train_int = load_pickle(data_dir + '/labels_train_internal_full.pickle')
    test = TimeSeriesDataset().load(data_dir + '/test_full.tsd')
    labels_test = load_pickle(data_dir + '/labels_test_full.pickle')
    return train_int, labels_train_int, test, labels_test


def upsample_to_1hr(dataset, downsample_rate=8):
    """Upsample a coarse dataset to 1hr resolution by inserting NaN rows.

    For each patient with L timesteps at `downsample_rate`-hr intervals,
    create a new sequence of length (L-1)*downsample_rate + 1 where
    original observations are placed at every `downsample_rate` steps
    and intermediate steps are NaN.
    """
    data_list = dataset.to_list()
    upsampled = []
    new_lengths = []
    for patient_tensor in data_list:
        L, C = patient_tensor.shape
        new_L = (L - 1) * downsample_rate + 1
        new_tensor = torch.full((new_L, C), float('nan'))
        for t in range(L):
            new_tensor[t * downsample_rate] = patient_tensor[t]
        upsampled.append(new_tensor)
        new_lengths.append(new_L)

    new_dataset = TimeSeriesDataset(upsampled, list(dataset.columns))
    new_dataset.lengths = new_lengths
    return new_dataset


def upsample_labels(labels, original_lengths, downsample_rate):
    """Expand per-timepoint labels to match upsampled 1hr resolution.

    Original labels correspond to coarse timepoints. For each patient,
    repeat each label `downsample_rate` times (last segment may be shorter).
    """
    labels = np.array(labels)
    idx = 0
    new_labels = []
    for L in original_lengths:
        patient_labels = labels[idx:idx + L]
        new_L = (L - 1) * downsample_rate + 1
        expanded = np.full(new_L, float('nan'))
        for t in range(L):
            start = t * downsample_rate
            end = min(start + downsample_rate, new_L)
            expanded[start:end] = patient_labels[t]
        new_labels.append(expanded)
        idx += L
    return np.concatenate(new_labels)


# ============================================================================
# Baseline 1: No imputation
# ============================================================================
def run_baseline_1(args):
    print("=" * 60)
    print("Baseline 1: No imputation (coarse 8hr data)")
    print("=" * 60)
    data_dir = resolve_path(args.data_path)

    train_ds, train_labels, test_ds, test_labels = load_data(data_dir)

    print(f"Train: {len(train_ds)} patients, Test: {len(test_ds)} patients")
    train_ds = add_expert_features(train_ds)
    test_ds = add_expert_features(test_ds)

    clf, metrics = train_and_evaluate(train_ds, train_labels, test_ds, test_labels, seed=args.seed)
    save_results(args.output_dir, baseline=1, metrics=metrics, clf=clf, args=args)


# ============================================================================
# Baseline 2: SAITS imputation (pypots)
# ============================================================================
def _find_latest_saits_model(saits_dir):
    """Find the latest trained SAITS model under the saving directory."""
    saits_dir = resolve_path(saits_dir)
    if not os.path.isdir(saits_dir):
        return None
    # Each run creates a timestamped subdirectory containing SAITS.pypots
    subdirs = sorted(
        [d for d in os.listdir(saits_dir)
         if os.path.isfile(os.path.join(saits_dir, d, "SAITS.pypots"))],
    )
    if not subdirs:
        return None
    return os.path.join(saits_dir, subdirs[-1])


def run_baseline_2_1(args):
    print("=" * 60)
    print("Baseline 2-1: Linear interpolation imputation (per patient)")
    print("=" * 60)

    data_dir = resolve_path(args.data_path)
    train_ds, train_labels, test_ds, test_labels = load_data(data_dir)
    print(f"Train: {len(train_ds)} patients, Test: {len(test_ds)} patients")

    # Upsample to 1hr resolution
    print("Upsampling train to 1hr resolution...")
    train_up = upsample_to_1hr(train_ds, downsample_rate=args.downsample_rate)
    print("Upsampling test to 1hr resolution...")
    test_up = upsample_to_1hr(test_ds, downsample_rate=args.downsample_rate)

    def linear_impute_dataset(dataset):
        """Per-patient, per-column linear interpolation. Edges extrapolated flat by np.interp."""
        data_list = dataset.to_list()
        for i in range(len(data_list)):
            arr = data_list[i].numpy()  # [L, C]
            for c in range(arr.shape[1]):
                col = arr[:, c]
                nans = np.isnan(col)
                if nans.all() or (~nans).all():
                    continue
                idx = np.where(~nans)[0]
                arr[:, c] = np.interp(np.arange(len(col)), idx, col[idx])
            data_list[i] = torch.tensor(arr, dtype=torch.float32)
        dataset_out = TimeSeriesDataset(data_list, list(dataset.columns))
        dataset_out.lengths = dataset.lengths
        return dataset_out

    print("Imputing train data with linear interpolation...")
    train_up = linear_impute_dataset(train_up)
    print("Imputing test data with linear interpolation...")
    test_up = linear_impute_dataset(test_up)

    # Expert features + evaluate
    train_up = add_expert_features(train_up)
    test_up = add_expert_features(test_up)

    train_labels_up = upsample_labels(train_labels, train_ds.lengths, args.downsample_rate)
    test_labels_up = upsample_labels(test_labels, test_ds.lengths, args.downsample_rate)

    clf, metrics = train_and_evaluate(train_up, train_labels_up, test_up, test_labels_up, seed=args.seed)
    save_results(args.output_dir, baseline="2-1", metrics=metrics, clf=clf, args=args)


def run_baseline_2_2(args):
    print("=" * 60)
    print(f"Baseline 2-2: {args.impute_method.upper()} imputation")
    print("=" * 60)
    from pypots.imputation import SAITS
    from pygrinder import mcar

    data_dir = resolve_path(args.data_path)

    train_ds, train_labels, test_ds, test_labels = load_data(data_dir)
    print(f"Train: {len(train_ds)} patients, Test: {len(test_ds)} patients")

    # Upsample to 1hr resolution
    print("Upsampling train to 1hr resolution...")
    train_up = upsample_to_1hr(train_ds, downsample_rate=args.downsample_rate)
    print("Upsampling test to 1hr resolution...")
    test_up = upsample_to_1hr(test_ds, downsample_rate=args.downsample_rate)

    # Prepare data for SAITS: expects dict with 'X' as [N, L, C] numpy array
    # Pad train and test to the same sequence length so SAITS can impute both
    train_np = train_up.data.numpy()
    test_np = test_up.data.numpy()
    n_features = train_np.shape[2]
    train_L = train_np.shape[1]
    test_L = test_np.shape[1]
    max_L = max(train_L, test_L)

    if train_L < max_L:
        pad = np.full((train_np.shape[0], max_L - train_L, n_features), np.nan)
        train_np = np.concatenate([train_np, pad], axis=1)
    if test_L < max_L:
        pad = np.full((test_np.shape[0], max_L - test_L, n_features), np.nan)
        test_np = np.concatenate([test_np, pad], axis=1)

    # Check for existing trained model
    saits_save_dir = os.path.join(args.output_dir, "baseline_2", "saits")
    existing_model_dir = _find_latest_saits_model(saits_save_dir)

    if existing_model_dir is not None:
        model_path = os.path.join(existing_model_dir, "SAITS.pypots")
        print(f"Found existing SAITS model: {model_path}")
        print("Loading pre-trained model (skipping training)...")
        saits = SAITS(
            n_steps=max_L,
            n_features=n_features,
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            epochs=50,
            patience=5,
            batch_size=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        saits.load(model_path)
    else:
        # 1. Train/val split for SAITS training
        print("No existing model found. Training from scratch...")
        print("Splitting train into train/val for SAITS (80/20)...")
        saits_train, saits_val = train_test_split(train_np, test_size=0.2, random_state=args.seed)
        print(f"SAITS train: {saits_train.shape[0]}, SAITS val: {saits_val.shape[0]}")
        # 2. For validation set, keep original as X_ori and create artificially masked version
        saits_val_ori = saits_val.copy()

        # Artificially mask some additional observed values (e.g., 10% more)
        saits_val_masked = mcar(saits_val_ori, p=0.1)

        train_set = {"X": saits_train}
        val_set = {
            "X": saits_val_masked,      # Data with artificial + natural missing values
            "X_ori": saits_val_ori      # Original data (with only natural missing values)
        }

        print("Training SAITS model...")
        saits = SAITS(
            n_steps=max_L,
            n_features=n_features,
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            epochs=50,
            patience=5,
            batch_size=16,
            saving_path=resolve_path(os.path.join(args.output_dir, "baseline_2", "saits")),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"SAITS device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        saits.fit(train_set, val_set)

    # Impute train
    print("Imputing train data...")
    train_imputed = saits.impute({"X": train_np})
    if isinstance(train_imputed, dict):
        train_imputed = train_imputed['imputation']
    train_up.data = torch.tensor(train_imputed[:, :train_L, :], dtype=torch.float32)

    # Impute test
    print("Imputing test data...")
    test_imputed = saits.impute({"X": test_np})
    if isinstance(test_imputed, dict):
        test_imputed = test_imputed['imputation']
    test_up.data = torch.tensor(test_imputed[:, :test_L, :], dtype=torch.float32)

    # Expert features + evaluate
    train_up = add_expert_features(train_up)
    test_up = add_expert_features(test_up)

    # Expand labels to match upsampled timepoints
    train_labels_up = upsample_labels(train_labels, train_ds.lengths, args.downsample_rate)
    test_labels_up = upsample_labels(test_labels, test_ds.lengths, args.downsample_rate)

    clf, metrics = train_and_evaluate(train_up, train_labels_up, test_up, test_labels_up, seed=args.seed)
    save_results(args.output_dir, baseline="2-2", metrics=metrics, clf=clf,
                 imputation_model=saits, args=args)


# ============================================================================
# Baseline 3-1: Statistical-feature Euclidean similarity imputation
# ============================================================================
def run_baseline_3_1(args):
    print("=" * 60)
    print(f"Baseline 3-1: Euclidean feature-space similarity imputation (top_k={args.top_k})")
    print("=" * 60)
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import euclidean

    n_clusters = getattr(args, 'n_clusters', 20)
    data_dir = resolve_path(args.data_path)

    train_ds, train_labels, test_ds, test_labels = load_data(data_dir)
    train_ext = TimeSeriesDataset().load(data_dir + '/train_external.tsd')

    print(f"Train internal: {len(train_ds)} patients")
    print(f"Train external: {len(train_ext)} patients")
    print(f"Test:           {len(test_ds)} patients")

    # Get shared columns between internal and external
    shared_cols = [c for c in train_ds.columns if c in train_ext.columns]
    print(f"Shared columns: {len(shared_cols)}")

    # Extract external patients as list of numpy arrays (shared cols + full)
    ext_list = train_ext.subset(shared_cols).to_list()
    ext_np = [np.nan_to_num(t.numpy(), nan=0.0) for t in ext_list]
    ext_full_list = train_ext.to_list()
    ext_full_np = [t.numpy() for t in ext_full_list]

    # --- Extract statistical features for clustering ---
    def extract_features(ts_list):
        """Compute per-channel statistics (mean, std, min, max, length) as a
        fixed-length feature vector regardless of time-series length."""
        n_cols = ts_list[0].shape[1] if len(ts_list) > 0 else 0
        # 4 stats per channel + 1 length feature
        feat_dim = n_cols * 4 + 1
        features = np.zeros((len(ts_list), feat_dim), dtype=np.float32)
        for i, ts in enumerate(ts_list):
            ts_clean = np.nan_to_num(ts, nan=0.0)
            if len(ts_clean) == 0:
                continue
            offset = 0
            features[i, offset:offset + n_cols] = ts_clean.mean(axis=0)
            offset += n_cols
            features[i, offset:offset + n_cols] = ts_clean.std(axis=0)
            offset += n_cols
            features[i, offset:offset + n_cols] = ts_clean.min(axis=0)
            offset += n_cols
            features[i, offset:offset + n_cols] = ts_clean.max(axis=0)
            offset += n_cols
            features[i, offset] = len(ts_clean)
        return features

    features_ext = extract_features(ext_np)

    # --- Cluster external patients with MiniBatchKMeans ---
    print(f"Clustering {len(ext_np)} external patients into {n_clusters} clusters...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_ext)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=args.seed)
    ext_cluster_labels = kmeans.fit_predict(features_scaled)

    # Build per-cluster index of external patients
    cluster_to_ext_idx = {c: [] for c in range(n_clusters)}
    for idx, cl in enumerate(ext_cluster_labels):
        cluster_to_ext_idx[cl].append(idx)
    print("Cluster sizes:", {c: len(v) for c, v in cluster_to_ext_idx.items()})

    def impute_with_external(internal_ds, downsample_rate, top_k):
        """For each internal patient, assign to nearest cluster then find top-K within that cluster."""
        int_shared = internal_ds.subset(shared_cols)
        int_list = int_shared.to_list()

        # Upsample internal to 1hr
        upsampled_ds = upsample_to_1hr(internal_ds, downsample_rate)
        up_data = upsampled_ds.to_list()

        col_idx_map = {c: upsampled_ds.columns.index(c) for c in shared_cols}
        ext_col_idx_map = {c: train_ext.columns.index(c) for c in shared_cols}

        for i in range(len(int_list)):
            if i % 20 == 0:
                print(f"  Processing patient {i+1}/{len(int_list)}...")

            patient_obs = int_list[i].numpy()  # [L_coarse, C_shared]

            valid_mask = ~np.isnan(patient_obs).all(axis=1)
            patient_valid = patient_obs[valid_mask]
            if len(patient_valid) == 0:
                continue

            patient_clean = np.nan_to_num(patient_valid, nan=0.0)

            # Extract features, scale, and predict cluster
            patient_feat = extract_features([patient_clean])
            patient_feat_scaled = scaler.transform(patient_feat)
            cluster_id = kmeans.predict(patient_feat_scaled)[0]
            candidate_indices = cluster_to_ext_idx[cluster_id]

            if len(candidate_indices) == 0:
                continue

            # Compute Euclidean distance in feature space within the assigned cluster
            patient_vec = patient_feat_scaled[0]
            distances = np.array([
                euclidean(patient_vec, features_scaled[ext_idx])
                for ext_idx in candidate_indices
            ])

            n_select = min(top_k, len(candidate_indices))
            local_top_k = np.argsort(distances)[:n_select]
            top_k_idx = [candidate_indices[li] for li in local_top_k]
            top_k_dists = distances[local_top_k]

            # Compute weights (inverse distance)
            top_k_dists = np.maximum(top_k_dists, 1e-8)
            weights = 1.0 / top_k_dists
            weights = weights / weights.sum()

            # Get upsampled patient tensor
            patient_up = up_data[i].numpy()  # [L_up, C_all]
            new_L = patient_up.shape[0]

            for k_i, ext_idx in enumerate(top_k_idx):
                ext_full = ext_full_np[ext_idx]  # [L_ext, C_ext]

                for c in shared_cols:
                    up_col = col_idx_map[c]
                    ext_col = ext_col_idx_map[c]
                    for t in range(min(new_L, len(ext_full))):
                        if np.isnan(patient_up[t, up_col]) and not np.isnan(ext_full[t, ext_col]):
                            if k_i == 0:
                                patient_up[t, up_col] = 0.0
                            patient_up[t, up_col] += weights[k_i] * ext_full[t, ext_col]

            up_data[i] = torch.tensor(patient_up, dtype=torch.float32)

        # Rebuild dataset
        new_lengths = [t.shape[0] for t in up_data]
        new_ds = TimeSeriesDataset(up_data, list(upsampled_ds.columns))
        new_ds.lengths = new_lengths
        return new_ds

    print("Imputing train internal with external data...")
    train_imputed = impute_with_external(train_ds, args.downsample_rate, args.top_k)

    print("Imputing test with external data...")
    test_imputed = impute_with_external(test_ds, args.downsample_rate, args.top_k)

    # Expert features
    train_imputed = add_expert_features(train_imputed)
    test_imputed = add_expert_features(test_imputed)

    # Expand labels
    train_labels_up = upsample_labels(train_labels, train_ds.lengths, args.downsample_rate)
    test_labels_up = upsample_labels(test_labels, test_ds.lengths, args.downsample_rate)

    clf, metrics = train_and_evaluate(train_imputed, train_labels_up, test_imputed, test_labels_up, seed=args.seed)
    save_results(args.output_dir, baseline="3-1", metrics=metrics, clf=clf, args=args)


# ============================================================================
# Baseline 3-2: Pearson-correlation similarity imputation (coarse scale)
# ============================================================================
def run_baseline_3_2(args):
    print("=" * 60)
    print(f"Baseline 3-2: Pearson-correlation similarity imputation (top_k={args.top_k})")
    print("=" * 60)

    data_dir = resolve_path(args.data_path)
    downsample_rate = args.downsample_rate
    top_k = args.top_k

    train_ds, train_labels, test_ds, test_labels = load_data(data_dir)
    train_ext = TimeSeriesDataset().load(data_dir + '/train_external.tsd')

    print(f"Train internal: {len(train_ds)} patients")
    print(f"Train external: {len(train_ext)} patients")
    print(f"Test:           {len(test_ds)} patients")

    # Get shared columns between internal and external
    shared_cols = [c for c in train_ds.columns if c in train_ext.columns]
    print(f"Shared columns: {len(shared_cols)}")

    # External patients: keep NaNs for mask-based correlation
    ext_shared_list = train_ext.subset(shared_cols).to_list()
    ext_shared_np = [t.numpy() for t in ext_shared_list]       # raw with NaNs
    ext_full_list = train_ext.to_list()
    ext_full_np = [t.numpy() for t in ext_full_list]

    # Downsample external to coarse scale (take every downsample_rate-th row)
    ext_coarse = []
    for ts in ext_shared_np:
        ext_coarse.append(ts[::downsample_rate])  # [L_ext_coarse, C_shared]
    print(f"Downsampled external to coarse scale (rate={downsample_rate})")

    def pearson_corr_distance(a, b):
        """Pearson correlation distance between two 2D arrays.
        Only uses positions where both a and b are non-NaN."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        valid = ~(np.isnan(a_flat) | np.isnan(b_flat))
        if valid.sum() < 2:
            return float('inf')
        a_v = a_flat[valid]
        b_v = b_flat[valid]
        corr = np.corrcoef(a_v, b_v)[0, 1]
        if np.isnan(corr):
            return float('inf')
        return 1.0 - corr  # distance: 0 = perfect correlation, 2 = perfect anti-correlation

    def impute_with_external(internal_ds, downsample_rate, top_k):
        """For each internal patient, find top-K external by Pearson correlation at coarse scale."""
        int_shared = internal_ds.subset(shared_cols)
        int_list = int_shared.to_list()

        # Upsample internal to 1hr
        upsampled_ds = upsample_to_1hr(internal_ds, downsample_rate)
        up_data = upsampled_ds.to_list()

        col_idx_map = {c: upsampled_ds.columns.index(c) for c in shared_cols}
        ext_col_idx_map = {c: train_ext.columns.index(c) for c in shared_cols}

        for i in range(len(int_list)):
            if i % 20 == 0:
                print(f"  Processing patient {i+1}/{len(int_list)}...")

            patient_obs = int_list[i].numpy()  # [L_coarse, C_shared], has NaNs

            # Check patient has some valid data
            if np.isnan(patient_obs).all():
                continue

            # Compute Pearson correlation distance to all external (at coarse scale)
            distances = np.empty(len(ext_coarse))
            for j, ec in enumerate(ext_coarse):
                # Align lengths: use min of both
                L_min = min(len(patient_obs), len(ec))
                distances[j] = pearson_corr_distance(patient_obs[:L_min], ec[:L_min])

            # Select top-k most correlated
            n_select = min(top_k, len(ext_coarse))
            top_k_local = np.argsort(distances)[:n_select]
            top_k_dists = distances[top_k_local]

            # Skip if no valid correlations found
            if top_k_dists[0] == float('inf'):
                continue

            # Compute weights (inverse distance)
            top_k_dists = np.maximum(top_k_dists, 1e-8)
            weights = 1.0 / top_k_dists
            weights = weights / weights.sum()

            # Get upsampled patient tensor
            patient_up = up_data[i].numpy()  # [L_up, C_all]
            new_L = patient_up.shape[0]

            for k_i, ext_idx in enumerate(top_k_local):
                ext_full = ext_full_np[ext_idx]  # [L_ext, C_ext]

                for c in shared_cols:
                    up_col = col_idx_map[c]
                    ext_col = ext_col_idx_map[c]
                    for t in range(min(new_L, len(ext_full))):
                        if np.isnan(patient_up[t, up_col]) and not np.isnan(ext_full[t, ext_col]):
                            if k_i == 0:
                                patient_up[t, up_col] = 0.0
                            patient_up[t, up_col] += weights[k_i] * ext_full[t, ext_col]

            up_data[i] = torch.tensor(patient_up, dtype=torch.float32)

        # Rebuild dataset
        new_lengths = [t.shape[0] for t in up_data]
        new_ds = TimeSeriesDataset(up_data, list(upsampled_ds.columns))
        new_ds.lengths = new_lengths
        return new_ds

    print("Imputing train internal with external data...")
    train_imputed = impute_with_external(train_ds, downsample_rate, top_k)

    print("Imputing test with external data...")
    test_imputed = impute_with_external(test_ds, downsample_rate, top_k)

    # Expert features
    train_imputed = add_expert_features(train_imputed)
    test_imputed = add_expert_features(test_imputed)

    # Expand labels
    train_labels_up = upsample_labels(train_labels, train_ds.lengths, downsample_rate)
    test_labels_up = upsample_labels(test_labels, test_ds.lengths, downsample_rate)

    clf, metrics = train_and_evaluate(train_imputed, train_labels_up, test_imputed, test_labels_up, seed=args.seed)
    save_results(args.output_dir, baseline="3-2", metrics=metrics, clf=clf, args=args)



# ============================================================================
# Baseline 4: TBD
# ============================================================================
def run_baseline_4(args):
    print("Baseline 4: Not implemented yet")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    runners = {
        "1": run_baseline_1,
        "2-1": run_baseline_2_1,
        "2-2": run_baseline_2_2,
        "3-1": run_baseline_3_1,
        "3-2": run_baseline_3_2,
        "4": run_baseline_4,
    }

    runners[str(args.baseline)](args)


if __name__ == "__main__":
    main()
