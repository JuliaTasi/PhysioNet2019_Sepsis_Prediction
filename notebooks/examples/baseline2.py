#!/usr/bin/env python
# coding: utf-8
"""Baseline 2 — Granularity enhancement to 1hr resolution without external data.

Upsample train_internal and test from 8hr to 1hr by inserting NaN at hourly
timestamps. Train SAITS on the upsampled internal data. Impute both splits.
Downstream classifier: LGBMClassifier, fixed train/test split.
"""

import sys
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# =============================================================================
# Tee: mirror all print() output to a log file
# =============================================================================
class _Tee:
    encoding = sys.__stdout__.encoding
    errors   = sys.__stdout__.errors
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

_log_dir = os.path.join(os.path.dirname(__file__), '../../results')
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, 'baseline2_output.txt')
_log_file = open(_log_path, 'w')
sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"Logging output to: {os.path.abspath(_log_path)}\n")
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from pygrinder import mcar
from pypots.imputation import SAITS

from definitions import DATA_DIR
from src.data.dataset import TimeSeriesDataset
from src.omni.functions import load_pickle
from src.external.evaluate_sepsis_score import compute_prediction_utility

# =============================================================================
# Hyperparameters (printed for reproducibility)
# =============================================================================
SEED           = 42
DOWNSAMPLE_RATE = 8   # original coarse-grained resolution used during downsampling

SAITS_PARAMS = dict(
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
)

print("=" * 60)
print("Baseline 2: SAITS imputation at 1hr resolution (no external)")
print("=" * 60)
print(f"Seed:             {SEED}")
print(f"Downsample rate:  {DOWNSAMPLE_RATE}hr")
print(f"SAITS parameters: {SAITS_PARAMS}")
print(f"Device:           {'cpu'}")
print()

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = DATA_DIR + '/splited'


# =============================================================================
# Helper: upsample a coarse TSD to 1hr by inserting NaN rows
# =============================================================================
def upsample_to_1hr(dataset, rate=DOWNSAMPLE_RATE):
    """Upsample each patient's coarse (rate-hr) time series to 1hr by inserting NaN rows.

    Original observations are placed at every `rate`-th step; intermediate
    steps are filled with NaN.  Returns a new TimeSeriesDataset.
    """
    data_list   = dataset.to_list()
    upsampled   = []
    new_lengths = []
    for patient in data_list:
        L, C    = patient.shape
        new_L   = (L - 1) * rate + 1
        new_t   = torch.full((new_L, C), float('nan'))
        for t in range(L):
            new_t[t * rate] = patient[t]
        upsampled.append(new_t)
        new_lengths.append(new_L)
    new_ds         = TimeSeriesDataset(upsampled, list(dataset.columns))
    new_ds.lengths = new_lengths
    return new_ds


# =============================================================================
# Helper: expand per-timepoint labels to match upsampled 1hr lengths
# =============================================================================
def upsample_labels(labels, original_lengths, rate=DOWNSAMPLE_RATE):
    """Expand flat label array from coarse to 1hr resolution.

    Each coarse label is repeated for `rate` consecutive 1hr timesteps
    (last patient may be shorter).
    """
    labels = np.array(labels)
    idx    = 0
    out    = []
    for L in original_lengths:
        patient_labels = labels[idx:idx + L]
        new_L  = (L - 1) * rate + 1
        expanded = np.full(new_L, float('nan'))
        for t in range(L):
            start          = t * rate
            end            = min(start + rate, new_L)
            expanded[start:end] = patient_labels[t]
        out.append(expanded)
        idx += L
    return np.concatenate(out)


# =============================================================================
# Step 1: Load Data
# =============================================================================
train_internal = TimeSeriesDataset().load(DATA_PATH + '/train_internal.tsd')
test           = TimeSeriesDataset().load(DATA_PATH + '/test.tsd')

labels_train_internal = load_pickle(DATA_PATH + '/labels_train_internal.pickle')
labels_test           = load_pickle(DATA_PATH + '/labels_test.pickle')

print(f"Train internal: {len(train_internal)} patients")
print(f"Test:           {len(test)} patients")

# Save original lengths (coarse) for label expansion
orig_lengths_train = list(train_internal.lengths)
orig_lengths_test  = list(test.lengths)

# =============================================================================
# Step 2: SAITS Imputation — Baseline 2
# 1. Upsample each patient's coarse time series to 1hr by inserting NaN rows.
# 2. Train saits_1hr_internal on the upsampled train_internal.
# 3. Impute both upsampled splits with saits_1hr_internal.
# =============================================================================

# Step 2.1 — Upsample to 1hr resolution
print("\nUpsampling train_internal to 1hr resolution...")
train_up = upsample_to_1hr(train_internal)
print("Upsampling test to 1hr resolution...")
test_up  = upsample_to_1hr(test)

print(f"Train upsampled max length: {train_up.data.shape[1]}")
print(f"Test  upsampled max length: {test_up.data.shape[1]}")

# Step 2.2 — Prepare numpy arrays [N, L, C] with NaN for missing values
train_np = train_up.data.numpy()
test_np  = test_up.data.numpy()

n_features = train_np.shape[2]
train_L    = train_np.shape[1]
test_L     = test_np.shape[1]
max_L      = max(train_L, test_L)

if train_L < max_L:
    pad = np.full((train_np.shape[0], max_L - train_L, n_features), np.nan)
    train_np = np.concatenate([train_np, pad], axis=1)
if test_L < max_L:
    pad = np.full((test_np.shape[0], max_L - test_L, n_features), np.nan)
    test_np = np.concatenate([test_np, pad], axis=1)

# Save original 1hr upsampled test array (NaN = missing/inserted) for imputation MSE
test_np_orig = test_np.copy()

print(f"\nSequence length after padding: {max_L}")
print(f"Number of features:            {n_features}")

# Step 2.3 — Train/val split for SAITS (80/20)
saits_train, saits_val = train_test_split(train_np, test_size=0.2, random_state=SEED)
saits_val_ori    = saits_val.copy()
saits_val_masked = mcar(saits_val_ori, p=0.1)

train_set = {"X": saits_train}
val_set   = {"X": saits_val_masked, "X_ori": saits_val_ori}

# Train saits_1hr_internal on upsampled train_internal
print("\nTraining saits_1hr_internal on upsampled train_internal...")
saits_1hr_internal = SAITS(
    n_steps=max_L,
    n_features=n_features,
    device="cpu",
    **SAITS_PARAMS,
)
saits_1hr_internal.fit(train_set, val_set)

# Step 2.4 — Impute both splits
print("Imputing upsampled train_internal...")
train_imputed = saits_1hr_internal.impute({"X": train_np})
if isinstance(train_imputed, dict):
    train_imputed = train_imputed['imputation']
train_up.data = torch.tensor(train_imputed[:, :train_L, :], dtype=torch.float32)

print("Imputing upsampled test...")
test_imputed = saits_1hr_internal.impute({"X": test_np})
if isinstance(test_imputed, dict):
    test_imputed = test_imputed['imputation']
test_imputed_crop = test_imputed[:, :test_L, :]    # [N_test, test_L, C]
test_up.data = torch.tensor(test_imputed_crop, dtype=torch.float32)

# Imputation MSE: MSE at originally observed (non-NaN) positions in the 1hr test set.
# Observed positions = original 8hr timestamps (every DOWNSAMPLE_RATE-th step);
# inserted intermediate NaN rows are excluded automatically by the mask.
test_orig_crop = test_np_orig[:, :test_L, :]       # [N_test, test_L, C]
obs_mask = ~np.isnan(test_orig_crop)
imputation_mse = float(np.mean(
    (test_orig_crop[obs_mask] - test_imputed_crop[obs_mask]) ** 2
))

# =============================================================================
# Step 6: Train LGBMClassifier
# Fixed split: train on imputed 1hr train_internal, evaluate on imputed 1hr test.
# =============================================================================
X_train = train_up.to_ml()
X_train[torch.isnan(X_train)] = -1000
X_test  = test_up.to_ml()
X_test[torch.isnan(X_test)]   = -1000

# Expand labels to match 1hr upsampled lengths
y_train = upsample_labels(labels_train_internal, orig_lengths_train)
y_test  = upsample_labels(labels_test, orig_lengths_test)

assert len(X_train) == len(y_train), "Train size mismatch after label expansion"
assert len(X_test)  == len(y_test),  "Test size mismatch after label expansion"

print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")

clf = LGBMClassifier(random_state=SEED, n_jobs=-1)
clf.fit(X_train.numpy(), y_train)

probas = clf.predict_proba(X_test.numpy())
preds  = (probas[:, 1] > 0.5).astype(int)

# =============================================================================
# Step 7: Evaluate — Accuracy, AUC, PhysioNet 2019 Utility Score
# =============================================================================
acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probas[:, 1])


def compute_normalized_utility(labels_per_patient, preds_per_patient):
    """Compute normalized PhysioNet 2019 utility score."""
    observed = np.zeros(len(labels_per_patient))
    best     = np.zeros(len(labels_per_patient))
    inaction = np.zeros(len(labels_per_patient))

    dt_optimal = -6

    for k, (lab, pred) in enumerate(zip(labels_per_patient, preds_per_patient)):
        lab  = np.asarray(lab, dtype=float)
        pred = np.asarray(pred, dtype=float)
        n    = len(lab)

        best_pred = np.zeros(n)
        if np.any(lab):
            t_sepsis = np.argmax(lab) - dt_optimal
            best_pred[max(0, t_sepsis - 12):min(t_sepsis + 3 + 1, n)] = 1

        observed[k] = compute_prediction_utility(lab, pred, check_errors=False)
        best[k]     = compute_prediction_utility(lab, best_pred, check_errors=False)
        inaction[k] = compute_prediction_utility(lab, np.zeros(n), check_errors=False)

    total_obs      = np.sum(observed)
    total_best     = np.sum(best)
    total_inaction = np.sum(inaction)

    if total_best - total_inaction == 0:
        return 0.0
    return (total_obs - total_inaction) / (total_best - total_inaction)


# Split flat predictions back to per-patient arrays (1hr lengths)
labels_per_patient = []
preds_per_patient  = []
idx = 0
for l in test_up.lengths:
    labels_per_patient.append(y_test[idx:idx + l])
    preds_per_patient.append(preds[idx:idx + l])
    idx += l

utility = compute_normalized_utility(labels_per_patient, preds_per_patient)

print(f"\nBaseline 2 | Accuracy: {acc * 100:.3f}% | AUC: {auc:.3f} | Utility: {utility:.4f} | Imputation MSE: {imputation_mse:.6f}")
