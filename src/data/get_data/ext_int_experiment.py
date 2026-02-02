#!/usr/bin/env python
# coding: utf-8
"""
Experiment: External vs Internal dataset for imputation study.

Splits data into:
  - Train external (50% of train): full 1hr resolution (fine-grained)
  - Train internal (50% of train): downsampled to 8hr intervals (coarse-grained)
  - Test: downsampled to 8hr intervals (coarse-grained)

Train/test split is 70/30, stratified by patient-level sepsis label.

Saves binary labels and TimeSeriesDataset (.tsd) for each subset.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import shutil
import numpy as np
import torch
from tqdm import tqdm
from definitions import *
from sklearn.model_selection import train_test_split
from src.data.dataset import TimeSeriesDataset

# =============================================================================
# Step 1: Load Data
# =============================================================================
print('Loading data...')
df = load_pickle(DATA_DIR + '/raw/df.pickle')
overall_labels = load_pickle(DATA_DIR + '/processed/labels/overall_labels.pickle')

patient_ids = df['id'].unique()
overall_labels = np.array(overall_labels)  # shape [N_patients]

print(f'Total patients: {len(patient_ids)}')
print(f'Sepsis positive: {overall_labels.sum():.0f} ({overall_labels.mean()*100:.1f}%)')

# =============================================================================
# Step 2: Stratified Train/Test Split by Patient ID (70/30)
# =============================================================================
train_ids, test_ids = train_test_split(
    patient_ids,
    test_size=0.3,
    random_state=42,
    stratify=overall_labels,
)

# =============================================================================
# Step 3: Split Training into External/Internal (50/50, stratified)
# =============================================================================
train_labels = overall_labels[train_ids]

external_ids, internal_ids = train_test_split(
    train_ids,
    test_size=0.5,
    random_state=42,
    stratify=train_labels,
)

# =============================================================================
# Step 4: Build DataFrames and Downsample
# =============================================================================
def downsample_by_patient(dataframe, interval=8):
    """Take every `interval`-th row per patient."""
    return dataframe.groupby('id', group_keys=False).apply(lambda x: x.iloc[::interval])

# External: full resolution
df_train_external = df[df['id'].isin(external_ids)].copy()

# Internal: downsampled to 8hr
df_train_internal = downsample_by_patient(df[df['id'].isin(internal_ids)])

# Test: downsampled to 8hr
df_test = downsample_by_patient(df[df['id'].isin(test_ids)])

# =============================================================================
# Step 5: Extract Binary Labels & Build TimeSeriesDatasets
# =============================================================================
def df_to_binary_labels(dataframe):
    """Extract per-timepoint SepsisLabel as a tensor."""
    return torch.Tensor(dataframe['SepsisLabel'].values)

def df_to_timeseries_dataset(dataframe):
    """Convert a DataFrame (with id, time, SepsisLabel, features) to a TimeSeriesDataset."""
    df_copy = dataframe.copy()
    df_copy.drop(['time', 'SepsisLabel'], axis=1, inplace=True)
    columns = list(df_copy.drop(['id'], axis=1).columns)

    tensor_data = []
    for pid in tqdm(df_copy['id'].unique()):
        data = df_copy[df_copy['id'] == pid].drop('id', axis=1)
        tensor_data.append(torch.Tensor(data.values.astype(float)))

    return TimeSeriesDataset(data=tensor_data, columns=columns)

print('\nBuilding labels and datasets...')
labels_train_external = df_to_binary_labels(df_train_external)
labels_train_internal = df_to_binary_labels(df_train_internal)
labels_test = df_to_binary_labels(df_test)

print('  Converting train_external to TimeSeriesDataset...')
tsd_train_external = df_to_timeseries_dataset(df_train_external)

print('  Converting train_internal to TimeSeriesDataset...')
tsd_train_internal = df_to_timeseries_dataset(df_train_internal)

print('  Converting test to TimeSeriesDataset...')
tsd_test = df_to_timeseries_dataset(df_test)

# =============================================================================
# Step 6: Clean old files and Save
# =============================================================================
output_dir = DATA_DIR + '/processed/experiment'

# Delete old experiment directory to avoid stale files
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Save binary labels
save_pickle(labels_train_external, output_dir + '/labels_train_external.pickle')
save_pickle(labels_train_internal, output_dir + '/labels_train_internal.pickle')
save_pickle(labels_test, output_dir + '/labels_test.pickle')

# Save TimeSeriesDatasets
tsd_train_external.save(output_dir + '/train_external.tsd')
tsd_train_internal.save(output_dir + '/train_internal.tsd')
tsd_test.save(output_dir + '/test.tsd')

# =============================================================================
# Summary
# =============================================================================
def print_summary(name, dataframe, labels, tsd):
    n_patients = dataframe['id'].nunique()
    n_timepoints = len(dataframe)
    sepsis_rate = labels.mean().item() * 100
    print(f'  {name:25s} | patients: {n_patients:6d} | timepoints: {n_timepoints:8d} | sepsis%: {sepsis_rate:.2f}% | tsd shape: {list(tsd.data.size())}')

print('\n=== Dataset Summary ===')
print_summary('Train External (1hr)', df_train_external, labels_train_external, tsd_train_external)
print_summary('Train Internal (8hr)', df_train_internal, labels_train_internal, tsd_train_internal)
print_summary('Test (8hr)', df_test, labels_test, tsd_test)
print(f'\nSaved to: {output_dir}')
