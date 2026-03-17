#!/usr/bin/env python
# coding: utf-8
"""Baseline 1 — SAITS Imputation at original coarse-grained (8hr) resolution.

No granularity change. Missing values filled by SAITS trained on train_internal.
Downstream classifier: LGBMClassifier, fixed train/test split.
"""

import sys
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import torch

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
_log_path = os.path.join(_log_dir, 'baseline1_output.txt')
_log_file = open(_log_path, 'w')
sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"Logging output to: {os.path.abspath(_log_path)}\n")
from sklearn.metrics import accuracy_score, roc_auc_score
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
SEED = 42
DOWNSAMPLE_RATE = 8          # original coarse-grained resolution (hours)

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
print("Baseline 1: SAITS imputation at 8hr coarse-grained resolution")
print("=" * 60)
print(f"Seed:             {SEED}")
print(f"SAITS parameters: {SAITS_PARAMS}")
print(f"Device:           {'cpu'}")
print()

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = DATA_DIR + '/splited'

# =============================================================================
# Step 1: Load Data
# =============================================================================
train_internal = TimeSeriesDataset().load(DATA_PATH + '/train_internal.tsd')
test           = TimeSeriesDataset().load(DATA_PATH + '/test.tsd')

labels_train_internal = load_pickle(DATA_PATH + '/labels_train_internal.pickle')
labels_test           = load_pickle(DATA_PATH + '/labels_test.pickle')

print(f"Train internal: {len(train_internal)} patients")
print(f"Test:           {len(test)} patients")

# =============================================================================
# Step 2: SAITS Imputation — Baseline 1
# Keep data at original 8hr resolution (no upsampling).
# Train saits_8hr on train_internal; impute train and test.
# =============================================================================

# Prepare numpy arrays [N, L, C] with NaN for missing values
train_np = train_internal.data.numpy()   # [N_train, L_train, C]
test_np  = test.data.numpy()             # [N_test,  L_test,  C]

n_features = train_np.shape[2]
train_L    = train_np.shape[1]
test_L     = test_np.shape[1]
max_L      = max(train_L, test_L)

# Pad to the same sequence length so SAITS can handle both splits
if train_L < max_L:
    pad = np.full((train_np.shape[0], max_L - train_L, n_features), np.nan)
    train_np = np.concatenate([train_np, pad], axis=1)
if test_L < max_L:
    pad = np.full((test_np.shape[0], max_L - test_L, n_features), np.nan)
    test_np = np.concatenate([test_np, pad], axis=1)

# Save original test array (NaN = missing) for imputation MSE computation
test_np_orig = test_np.copy()

print(f"\nSequence length after padding: {max_L}")
print(f"Number of features:            {n_features}")

# 80/20 train/val split for SAITS training
saits_train, saits_val = train_test_split(train_np, test_size=0.2, random_state=SEED)
saits_val_ori    = saits_val.copy()
saits_val_masked = mcar(saits_val_ori, p=0.1)   # add 10% artificial masking for validation

train_set = {"X": saits_train}
val_set   = {"X": saits_val_masked, "X_ori": saits_val_ori}

# Train saits_8hr on train_internal (NaN positions are the missing values)
print("\nTraining saits_8hr on train_internal...")
saits_8hr = SAITS(
    n_steps=max_L,
    n_features=n_features,
    device="cpu",
    **SAITS_PARAMS,
)
saits_8hr.fit(train_set, val_set)

# Impute train_internal with saits_8hr
print("Imputing train_internal with saits_8hr...")
train_imputed = saits_8hr.impute({"X": train_np})
if isinstance(train_imputed, dict):
    train_imputed = train_imputed['imputation']
train_internal.data = torch.tensor(
    train_imputed[:, :train_L, :], dtype=torch.float32
)

# Impute test with saits_8hr
print("Imputing test with saits_8hr...")
test_imputed = saits_8hr.impute({"X": test_np})
if isinstance(test_imputed, dict):
    test_imputed = test_imputed['imputation']
test_imputed_crop = test_imputed[:, :test_L, :]   # [N_test, test_L, C]
test.data = torch.tensor(test_imputed_crop, dtype=torch.float32)

# Imputation MSE: MSE at originally observed (non-NaN) positions in the test set
test_orig_crop = test_np_orig[:, :test_L, :]      # [N_test, test_L, C]
obs_mask = ~np.isnan(test_orig_crop)
imputation_mse = float(np.mean(
    (test_orig_crop[obs_mask] - test_imputed_crop[obs_mask]) ** 2
))

# =============================================================================
# Step 6: Train LGBMClassifier
# Fixed split: train on train_internal, evaluate on test.
# =============================================================================
X_train = train_internal.to_ml()
X_train[torch.isnan(X_train)] = -1000
X_test  = test.to_ml()
X_test[torch.isnan(X_test)]   = -1000

y_train = np.array(labels_train_internal)
y_test  = np.array(labels_test)

assert len(X_train) == len(y_train), "Train size mismatch"
assert len(X_test)  == len(y_test),  "Test size mismatch"

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


# Split flat predictions back to per-patient arrays
labels_per_patient = []
preds_per_patient  = []
idx = 0
for l in test.lengths:
    labels_per_patient.append(y_test[idx:idx + l])
    preds_per_patient.append(preds[idx:idx + l])
    idx += l

utility = compute_normalized_utility(labels_per_patient, preds_per_patient)

print(f"\nBaseline 1 | Accuracy: {acc * 100:.3f}% | AUC: {auc:.3f} | Utility: {utility:.4f} | Imputation MSE: {imputation_mse:.6f}")
