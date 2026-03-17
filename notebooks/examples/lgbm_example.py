#!/usr/bin/env python
# coding: utf-8
# converted from ".ipynb", should based on this code
"""Prediction of Sepsis from ICU Data using Random Forest."""

import sys
sys.path.insert(0, '../../')

import torch
from definitions import *
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.features.derived_features import shock_index, partial_sofa
from src.features.rolling import RollingStatistic
# from src.features.signatures.augmentations import apply_augmentation_list
# from src.features.signatures.compute import RollingSignature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, roc_auc_score

# =============================================================================
# Step 1: Load Data
# =============================================================================
df = load_pickle(DATA_DIR + '/raw/df.pickle')
labels_binary = load_pickle(DATA_DIR + '/processed/labels/binary.pickle')
labels_utility = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')

# Load the TimeSeriesDataset (ragged time-series stored as padded tensor [N, L_max, C])
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# =============================================================================
# Step 2: Handle Missing Data (forward fill)
# =============================================================================
dataset.data = torch_ffill(dataset.data)

# =============================================================================
# Step 3: Expert Features
# =============================================================================
dataset['ShockIndex'] = shock_index(dataset)
dataset['PartialSOFA'] = partial_sofa(dataset)

# =============================================================================
# Step 4: Rolling Statistics
# =============================================================================
dataset['MaxShockIndex'] = RollingStatistic(statistic='max', window_length=5).transform(dataset['SBP'])
dataset['MinHR'] = RollingStatistic(statistic='min', window_length=8).transform(dataset['HR'])

# =============================================================================
# Step 5: Rolling Signatures (requires signatory; optional)
# =============================================================================
# rolling_signature = RollingSignature(window=6, depth=3, logsig=True, aug_list=['addtime'])
# signatures = rolling_signature.transform(dataset[['PartialSOFA', 'ShockIndex']])
# dataset.add_features(signatures)

# rolling_signature = RollingSignature(window=10, depth=3, logsig=True, aug_list=['addtime', 'penoff'])
# signatures = rolling_signature.transform(dataset[['PartialSOFA', 'HR']])
# dataset.add_features(signatures)

# =============================================================================
# Step 6: Train a Random Forest
# =============================================================================
X = dataset.to_ml()
y = labels_binary
assert len(X) == len(y)

X[torch.isnan(X)] = -1000

cv = list(KFold(5).split(X))
clf = RandomForestClassifier()
probas = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')

# =============================================================================
# Step 7: Evaluate
# =============================================================================
preds = (probas[:, 1] > 0.5).astype(int)
acc = accuracy_score(y, preds)
auc = roc_auc_score(y, probas[:, 1])

print('Accuracy: {:.3f}% \nAUC: {:.3f}'.format(acc * 100, auc))
