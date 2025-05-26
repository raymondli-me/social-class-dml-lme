#!/usr/bin/env python3
"""
Compare NV-Embed full 4,096 dims vs PCA 200 dims
Using exact same data pipeline as analyze_nvembed_complete.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import sys

print("="*80)
print("NV-EMBED: FULL 4,096 DIMENSIONS vs PCA 200")
print("="*80)

# Use the exact paths from the working script
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

# File paths (from working script)
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_FILE = DATA_DIR / "essays_9513.csv"  # This file has the sc11 column
AI_RATINGS_FILE = DATA_DIR / "vllm_outputs/all_results_526x50_20250524_120949.csv"

# Load data exactly as in the working script
print("\n=== Loading Data ===")

# Load essays
essays = pd.read_csv(ESSAYS_FILE)
print(f"✓ Essays loaded: {len(essays)} (should be 9,513)")

# Load social class labels from essays_9513.csv
sc_data = pd.read_csv(SC_FILE)
# Extract just the columns we need
sc_labels = sc_data[['TID', 'sc11']].copy()
print(f"✓ SC labels loaded: {len(sc_labels)}")

# Merge
df = essays.merge(sc_labels, on='TID', how='inner')
print(f"✓ After merge: {len(df)} essays with SC labels")

# Load AI ratings
ai_ratings = pd.read_csv(AI_RATINGS_FILE)
print(f"✓ AI ratings loaded: {len(ai_ratings)}")

# Average AI ratings per essay
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']

# Final merge
df = df.merge(ai_avg, on='TID', how='inner')
print(f"✓ Final dataset: {len(df)} essays with all data")

# Load NV-Embed embeddings
print("\n=== Loading NV-Embed Embeddings ===")
embeddings = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"✓ Embeddings shape: {embeddings.shape}")

# Verify and align data
print("\n=== Verifying Data Alignment ===")
df_aligned = df[df['TID'].isin(essay_ids)].copy()
df_aligned['essay_idx'] = df_aligned['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df_aligned = df_aligned.sort_values('essay_idx').reset_index(drop=True)

# Get aligned features and targets
X_full = embeddings[df_aligned['essay_idx'].values]
y_ai = df_aligned['ai_average'].values
y_sc = df_aligned['sc11'].values

print(f"✓ Aligned data: {len(df_aligned)} essays")
print(f"✓ X shape: {X_full.shape}")

# Standardize
print("\n=== Standardizing Features ===")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Configure XGBoost
xgb_params = {
    'n_estimators': 50,  # Same as in original analysis
    'max_depth': 6,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'  # For faster high-dim processing
}

print("\n" + "="*70)
print("RESULTS:")
print("="*70)

# 1. Full 4,096 dimensions
print("\n1. FULL 4,096 DIMENSIONS (XGBoost):")
print("   Running 5-fold cross-validation...")

xgb_model = xgb.XGBRegressor(**xgb_params)

scores_ai_full = cross_val_score(xgb_model, X_full_scaled, y_ai, cv=5, scoring='r2')
scores_sc_full = cross_val_score(xgb_model, X_full_scaled, y_sc, cv=5, scoring='r2')

ai_r2_full = scores_ai_full.mean()
sc_r2_full = scores_sc_full.mean()

print(f"   AI Ratings:   R² = {ai_r2_full:.4f} (±{scores_ai_full.std():.4f})")
print(f"   Social Class: R² = {sc_r2_full:.4f} (±{scores_sc_full.std():.4f})")
print(f"   Gap: {(ai_r2_full - sc_r2_full) * 100:.1f}%")

# 2. PCA 200 dimensions
print("\n2. PCA 200 DIMENSIONS (XGBoost):")
print("   Applying PCA...")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)
explained_var = pca.explained_variance_ratio_.sum()
print(f"   Explained variance: {explained_var:.1%}")

print("   Running 5-fold cross-validation...")
scores_ai_pca = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_pca = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')

ai_r2_pca = scores_ai_pca.mean()
sc_r2_pca = scores_sc_pca.mean()

print(f"   AI Ratings:   R² = {ai_r2_pca:.4f} (±{scores_ai_pca.std():.4f})")
print(f"   Social Class: R² = {sc_r2_pca:.4f} (±{scores_sc_pca.std():.4f})")
print(f"   Gap: {(ai_r2_pca - sc_r2_pca) * 100:.1f}%")

# 3. Compare with checkpoint results
print("\n3. CHECKPOINT RESULTS (for verification):")
print("   From analyze_nvembed_complete.py:")
print("   AI Ratings:   R² = 0.5972")
print("   Social Class: R² = 0.0733")

# Test intermediate PCA dimensions
print("\n4. TESTING INTERMEDIATE PCA DIMENSIONS:")
for n_comp in [500, 1000]:
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_pca_test = pca_test.fit_transform(X_full_scaled)
    var_explained = pca_test.explained_variance_ratio_.sum()
    
    scores_ai = cross_val_score(xgb_model, X_pca_test, y_ai, cv=5, scoring='r2')
    scores_sc = cross_val_score(xgb_model, X_pca_test, y_sc, cv=5, scoring='r2')
    
    print(f"\n   PCA {n_comp} (explains {var_explained:.1%} variance):")
    print(f"   AI R² = {scores_ai.mean():.4f}, SC R² = {scores_sc.mean():.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

# Calculate improvements
ai_improvement = ((ai_r2_full - ai_r2_pca) / ai_r2_pca * 100)
sc_improvement = ((sc_r2_full - sc_r2_pca) / sc_r2_pca * 100)

print(f"\nFull 4,096 dims vs PCA 200:")
print(f"- AI prediction:  {ai_r2_full:.4f} vs {ai_r2_pca:.4f} ({ai_improvement:+.1f}%)")
print(f"- SC prediction:  {sc_r2_full:.4f} vs {sc_r2_pca:.4f} ({sc_improvement:+.1f}%)")

if ai_improvement > 10:
    print("\n🚨 MAJOR FINDING: Full dimensions significantly outperform PCA!")
    print("   PCA is losing critical information for this task.")
    print("   Consider using more PCA components or full dimensions.")
elif ai_improvement > 0:
    print("\n✅ Full dimensions slightly outperform PCA.")
    print("   The improvement may not justify the computational cost.")
else:
    print("\n❌ PCA performs as well or better than full dimensions.")
    print("   PCA is effectively removing noise from the embeddings.")

print("\nKey insight: With only 72.1% variance retained in PCA,")
print("we may be losing important signal for social class detection.")
print("="*70)