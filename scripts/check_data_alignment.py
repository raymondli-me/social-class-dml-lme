#!/usr/bin/env python3
"""Check data alignment between essays and social class labels"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")

# Check essay dataset
print("=== Essay Dataset ===")
essays = pd.read_csv(BASE_DIR / "data/essay_dataset.csv")
print(f"Shape: {essays.shape}")
print(f"Columns: {essays.columns.tolist()}")
print(f"First few TIDs: {essays['TID'].head().tolist()}")
print(f"Unique TIDs: {len(essays['TID'].unique())}")

# Check social class labels
print("\n=== Social Class Labels ===")
sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
print(f"Shape: {sc_labels.shape}")
print(f"Columns: {sc_labels.columns.tolist()}")
print(f"First few TIDs: {sc_labels['TID'].head().tolist()}")
print(f"Unique TIDs: {len(sc_labels['TID'].unique())}")

# Check overlap
print("\n=== Overlap Analysis ===")
essays_tids = set(essays['TID'])
sc_tids = set(sc_labels['TID'])
overlap = essays_tids & sc_tids
print(f"Essays with SC labels: {len(overlap)}")
print(f"Essays without SC labels: {len(essays_tids - sc_tids)}")
print(f"SC labels without essays: {len(sc_tids - essays_tids)}")

# Try merging
print("\n=== Merge Test ===")
merged = essays.merge(sc_labels, on='TID', how='inner')
print(f"Merged shape: {merged.shape}")
print(f"SC distribution in merged data:")
print(merged['sc11'].value_counts().sort_index())

# Check AI ratings file
print("\n=== AI Ratings File ===")
ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
print(f"Shape: {ai_ratings.shape}")
print(f"Columns: {ai_ratings.columns.tolist()}")
print(f"Unique essay_ids: {len(ai_ratings['essay_id'].unique())}")
print(f"First few essay_ids: {ai_ratings['essay_id'].head().tolist()}")

# Check if essay_id matches TID format
print("\n=== ID Format Check ===")
print(f"Essay TID format: {essays['TID'].iloc[0]}")
print(f"SC TID format: {sc_labels['TID'].iloc[0]}")
print(f"AI essay_id format: {ai_ratings['essay_id'].iloc[0]}")

# Check the actual analysis script to see how it loads data
print("\n=== Checking Original Script's Data Loading ===")
# The issue: the script renames TID to id, but uses essay_id for AI ratings
# This might cause a mismatch