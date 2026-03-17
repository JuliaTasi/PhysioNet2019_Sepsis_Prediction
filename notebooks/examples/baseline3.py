#!/usr/bin/env python
# coding: utf-8
"""Baseline 3 — Granularity enhancement to 1hr with naive Pearson similarity-based external fusion.

Two independent SAITS models are trained:
  - saits_1hr_internal: on train_internal upsampled to 1hr
  - saits_1hr_external: on train_external (already at 1hr)

For each internal/test patient, the top-5 most similar external patients
(by Pearson correlation at the original 8hr scale) are identified, their
1hr imputed trajectories are averaged, and fused 50/50 with the patient's
own imputed trajectory.

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
_log_path = os.path.join(_log_dir, 'baseline3_output.txt')
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
SEED            = 42
DOWNSAMPLE_RATE = 8   # original coarse-grained resolution used during downsampling
TOP_K           = 5   # number of similar external patients for fusion
FUSION_WEIGHT   = 0.5 # equal weighting: 0.5 internal + 0.5 external

# Shared SAITS parameters (same for both models to keep comparison fair)
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
print("Baseline 3: SAITS at 1hr + Pearson similarity external fusion")
print("=" * 60)
print(f"Seed:             {SEED}")
print(f"Downsample rate:  {DOWNSAMPLE_RATE}hr")
print(f"Top-K:            {TOP_K}")
print(f"Fusion weight:    {FUSION_WEIGHT} internal / {1.0 - FUSION_WEIGHT} external")
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
    """Upsample each patient's coarse (rate-hr) series to 1hr by inserting NaN rows.

    Original observations placed at every `rate`-th step; intermediate steps NaN.
    """
    data_list   = dataset.to_list()
    upsampled   = []
    new_lengths = []
    for patient in data_list:
        L, C  = patient.shape
        new_L = (L - 1) * rate + 1
        new_t = torch.full((new_L, C), float('nan'))
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
    """Expand flat label array from coarse to 1hr resolution."""
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
# Helper: Pearson correlation between two 2D arrays (NaN-safe, flattened)
# =============================================================================
def pearson_corr(a, b):
    """Pearson correlation between two arrays; only uses positions where both are non-NaN.

    Returns the correlation coefficient, or np.nan if fewer than 2 valid pairs.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    valid  = ~(np.isnan(a_flat) | np.isnan(b_flat))
    if valid.sum() < 2:
        return np.nan
    a_v = a_flat[valid]
    b_v = b_flat[valid]
    corr = np.corrcoef(a_v, b_v)[0, 1]
    return corr


# =============================================================================
# Helper: train a SAITS model given a numpy array [N, L_max, C] with NaN
# =============================================================================
def train_saits(data_np, seed, tag=""):
    """Split 80/20, add 10% artificial masking to val, fit SAITS. Returns fitted model."""
    n_features = data_np.shape[2]
    max_L      = data_np.shape[1]

    saits_train, saits_val = train_test_split(data_np, test_size=0.2, random_state=seed)
    saits_val_ori    = saits_val.copy()
    saits_val_masked = mcar(saits_val_ori, p=0.1)

    train_set = {"X": saits_train}
    val_set   = {"X": saits_val_masked, "X_ori": saits_val_ori}

    print(f"Training {tag}: n_steps={max_L}, n_features={n_features}, "
          f"train={saits_train.shape[0]}, val={saits_val.shape[0]}")
    model = SAITS(
        n_steps=max_L,
        n_features=n_features,
        device="cpu",
        **SAITS_PARAMS,
    )
    model.fit(train_set, val_set)
    return model


# =============================================================================
# Helper: apply SAITS imputation and return imputed numpy array
# =============================================================================
def impute_with_saits(model, data_np, original_L):
    """Impute data_np with a fitted SAITS model; crop back to original_L timesteps."""
    result = model.impute({"X": data_np})
    if isinstance(result, dict):
        result = result['imputation']
    return result[:, :original_L, :]   # [N, original_L, C]


# =============================================================================
# Step 1: Load Data
# =============================================================================
train_internal = TimeSeriesDataset().load(DATA_PATH + '/train_internal.tsd')
train_external = TimeSeriesDataset().load(DATA_PATH + '/train_external.tsd')
test           = TimeSeriesDataset().load(DATA_PATH + '/test.tsd')

labels_train_internal = load_pickle(DATA_PATH + '/labels_train_internal.pickle')
labels_test           = load_pickle(DATA_PATH + '/labels_test.pickle')

print(f"Train internal: {len(train_internal)} patients")
print(f"Train external: {len(train_external)} patients")
print(f"Test:           {len(test)} patients")

# Save original coarse lengths for label expansion
orig_lengths_train = list(train_internal.lengths)
orig_lengths_test  = list(test.lengths)

# =============================================================================
# Step 2: SAITS Imputation — Baseline 3
# =============================================================================

# ----------------------------------------------------------------------------
# Step 2a — Upsample coarse sets to 1hr; external is already at 1hr
# ----------------------------------------------------------------------------
print("\nUpsampling train_internal to 1hr resolution...")
train_internal_up = upsample_to_1hr(train_internal)
print("Upsampling test to 1hr resolution...")
test_up           = upsample_to_1hr(test)
# train_external is already at 1hr — use as-is

print(f"train_internal_up max length: {train_internal_up.data.shape[1]}")
print(f"test_up          max length: {test_up.data.shape[1]}")
print(f"train_external   max length: {train_external.data.shape[1]}")

# ----------------------------------------------------------------------------
# Step 2a — Train two independent SAITS models (no shared weights)
# ----------------------------------------------------------------------------

# Model 1: saits_1hr_internal — trained on upsampled train_internal
int_train_np = train_internal_up.data.numpy()   # [N_int, L_int, C]
int_test_np  = test_up.data.numpy()             # [N_test, L_test, C]
int_L_train  = int_train_np.shape[1]
int_L_test   = int_test_np.shape[1]
int_max_L    = max(int_L_train, int_L_test)

if int_L_train < int_max_L:
    pad = np.full((int_train_np.shape[0], int_max_L - int_L_train, int_train_np.shape[2]), np.nan)
    int_train_np = np.concatenate([int_train_np, pad], axis=1)
if int_L_test < int_max_L:
    pad = np.full((int_test_np.shape[0], int_max_L - int_L_test, int_test_np.shape[2]), np.nan)
    int_test_np = np.concatenate([int_test_np, pad], axis=1)

print()
saits_1hr_internal = train_saits(int_train_np, seed=SEED, tag="saits_1hr_internal")

# Model 2: saits_1hr_external — trained on train_external (already at 1hr)
ext_np   = train_external.data.numpy()          # [N_ext, L_ext, C]
ext_max_L = ext_np.shape[1]

print()
saits_1hr_external = train_saits(ext_np, seed=SEED, tag="saits_1hr_external")

# ----------------------------------------------------------------------------
# Step 2b — Produce imputed outputs
# ----------------------------------------------------------------------------
print("\nImputing upsampled train_internal with saits_1hr_internal...")
imputed_internal_train_np = impute_with_saits(saits_1hr_internal, int_train_np, int_L_train)
# shape: [N_int, L_int, C]

print("Imputing upsampled test with saits_1hr_internal...")
imputed_internal_test_np  = impute_with_saits(saits_1hr_internal, int_test_np, int_L_test)
# shape: [N_test, L_test, C]

print("Imputing train_external with saits_1hr_external...")
imputed_external_np = impute_with_saits(saits_1hr_external, ext_np, ext_max_L)
# shape: [N_ext, L_ext, C]

# ----------------------------------------------------------------------------
# Step 2c — Patient similarity matching via Pearson correlation
# Compare internal/test patient's original 8hr observations against
# train_external downsampled to 8hr scale.
# ----------------------------------------------------------------------------
print("\nPre-computing external coarse representations (every 8th row)...")
# External at coarse scale for similarity comparison
ext_list_orig = train_external.to_list()          # list of [L_ext_i, C] tensors
ext_coarse    = [t.numpy()[::DOWNSAMPLE_RATE] for t in ext_list_orig]
# imputed_external_np: [N_ext, L_ext_max, C] — we will access per patient

ext_lengths = list(train_external.lengths)


def find_top_k_external(patient_coarse_np, top_k=TOP_K):
    """Return indices and Pearson correlations of top-K most similar external patients.

    patient_coarse_np: [L_coarse, C] numpy array with NaN for missing values.
    Comparison done at coarse timestamps using all channels.
    External patients are compared at their own 8th-row representation.
    """
    if np.isnan(patient_coarse_np).all():
        return [], []

    correlations = np.full(len(ext_coarse), np.nan)
    for j, ec in enumerate(ext_coarse):
        L_min = min(len(patient_coarse_np), len(ec))
        corr  = pearson_corr(patient_coarse_np[:L_min], ec[:L_min])
        correlations[j] = corr

    # Sort by correlation descending; skip NaN
    valid_mask = ~np.isnan(correlations)
    if valid_mask.sum() == 0:
        return [], []

    ranked    = np.where(valid_mask)[0]
    ranked    = ranked[np.argsort(-correlations[ranked])]
    top_idx   = ranked[:top_k].tolist()
    top_corrs = correlations[top_idx].tolist()
    return top_idx, top_corrs


# ----------------------------------------------------------------------------
# Step 2d — Naive fusion: 0.5 * internal + 0.5 * average(top-K external)
# ----------------------------------------------------------------------------
def fuse_patient(imputed_internal, imputed_ext_list, top_k_idx):
    """Fuse one patient's internal imputed trajectory with top-K external trajectories.

    imputed_internal:  [L_int, C] numpy array (fully imputed)
    imputed_ext_list:  list of [L_ext_i, C] numpy arrays (fully imputed external)
    top_k_idx:         list of indices into imputed_ext_list

    Returns fused array of shape [L_int, C].
    """
    L_int, C = imputed_internal.shape
    fused     = imputed_internal.copy()

    if not top_k_idx:
        return fused

    for t in range(L_int):
        ext_vals = []
        for ej in top_k_idx:
            ext_traj = imputed_ext_list[ej]
            if t < len(ext_traj):
                ext_vals.append(ext_traj[t])   # [C]
        if ext_vals:
            ext_mean     = np.mean(np.stack(ext_vals, axis=0), axis=0)  # [C]
            fused[t]     = (1.0 - FUSION_WEIGHT) * imputed_internal[t] + FUSION_WEIGHT * ext_mean

    return fused


# Build list of per-patient imputed external trajectories (valid lengths)
imputed_external_list = [
    imputed_external_np[i, :ext_lengths[i], :]
    for i in range(len(ext_lengths))
]

# Internal coarse observations for similarity (before upsampling; raw NaN values)
int_coarse_list  = train_internal.to_list()   # [L_coarse_i, C] per patient
test_coarse_list = test.to_list()

print(f"\nFusing train_internal patients (N={len(int_coarse_list)})...")
fused_train_list   = []
fused_train_lengths = []
for i, patient in enumerate(int_coarse_list):
    if i % 100 == 0:
        print(f"  train patient {i + 1}/{len(int_coarse_list)}")
    patient_coarse_np = patient.numpy()            # [L_coarse_i, C] with NaN
    top_k_idx, _      = find_top_k_external(patient_coarse_np)
    L_int_i           = int_L_train                # max-padded length used — use real length
    real_L            = train_internal_up.lengths[i]
    imputed_int_i     = imputed_internal_train_np[i, :real_L, :]  # [L_1hr_i, C]
    fused_i           = fuse_patient(imputed_int_i, imputed_external_list, top_k_idx)
    fused_train_list.append(torch.tensor(fused_i, dtype=torch.float32))
    fused_train_lengths.append(real_L)

# Save original 1hr upsampled test array before imputation for MSE computation
# Observed positions = every DOWNSAMPLE_RATE-th step; NaN elsewhere.
int_test_np_orig = int_test_np[:, :int_L_test, :].copy()   # [N_test, int_L_test, C]

print(f"\nFusing test patients (N={len(test_coarse_list)})...")
fused_test_list   = []
fused_test_lengths = []
for i, patient in enumerate(test_coarse_list):
    if i % 100 == 0:
        print(f"  test patient {i + 1}/{len(test_coarse_list)}")
    patient_coarse_np = patient.numpy()
    top_k_idx, _      = find_top_k_external(patient_coarse_np)
    real_L            = test_up.lengths[i]
    imputed_int_i     = imputed_internal_test_np[i, :real_L, :]   # [L_1hr_i, C]
    fused_i           = fuse_patient(imputed_int_i, imputed_external_list, top_k_idx)
    fused_test_list.append(torch.tensor(fused_i, dtype=torch.float32))
    fused_test_lengths.append(real_L)

# Imputation MSE: compare fused 1hr test trajectories to original 8hr observations.
# For each patient, original observations are at timestep indices [0, 8, 16, ...] in 1hr space.
mse_vals = []
for i, (orig_coarse, fused_1hr) in enumerate(zip(test_coarse_list, fused_test_list)):
    orig_np  = orig_coarse.numpy()   # [L_coarse, C] — original 8hr values with NaN for missing
    fused_np = fused_1hr.numpy()     # [L_1hr, C]    — final fused imputed trajectory
    for t in range(orig_np.shape[0]):
        t_1hr = t * DOWNSAMPLE_RATE
        if t_1hr >= fused_np.shape[0]:
            break
        obs_mask = ~np.isnan(orig_np[t])
        if obs_mask.any():
            diff = orig_np[t, obs_mask] - fused_np[t_1hr, obs_mask]
            mse_vals.extend((diff ** 2).tolist())
imputation_mse = float(np.mean(mse_vals)) if mse_vals else float('nan')

# Rebuild TimeSeriesDatasets from fused trajectories
train_fused_ds         = TimeSeriesDataset(fused_train_list, list(train_internal_up.columns))
train_fused_ds.lengths = fused_train_lengths

test_fused_ds         = TimeSeriesDataset(fused_test_list, list(test_up.columns))
test_fused_ds.lengths = fused_test_lengths

# =============================================================================
# Step 6: Train LGBMClassifier
# Fixed split: train on fused 1hr train_internal, evaluate on fused 1hr test.
# =============================================================================
X_train = train_fused_ds.to_ml()
X_train[torch.isnan(X_train)] = -1000
X_test  = test_fused_ds.to_ml()
X_test[torch.isnan(X_test)]   = -1000

# Expand binary labels to match 1hr upsampled lengths
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
for l in test_fused_ds.lengths:
    labels_per_patient.append(y_test[idx:idx + l])
    preds_per_patient.append(preds[idx:idx + l])
    idx += l

utility = compute_normalized_utility(labels_per_patient, preds_per_patient)

print(f"\nBaseline 3 | Accuracy: {acc * 100:.3f}% | AUC: {auc:.3f} | Utility: {utility:.4f} | Imputation MSE: {imputation_mse:.6f}")
