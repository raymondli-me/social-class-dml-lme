#!/usr/bin/env python3
"""
Test NV-Embed with full 4,096 dimensions vs PCA
Quick and simple comparison
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

print("="*80)
print("NV-EMBED: FULL DIMENSIONS vs PCA COMPARISON")
print("="*80)

# First, let's just load the already-prepared data from checkpoints
CHECKPOINT_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")

# Load full embeddings
print("\nLoading NV-Embed embeddings...")
embeddings_full = np.load(CHECKPOINT_DIR / "nvembed_checkpoints/nvembed_embeddings.npy")
print(f"Full embeddings shape: {embeddings_full.shape}")

# Load the DML checkpoint data which has aligned targets
print("\nLoading aligned target data from DML checkpoint...")
import pickle

# Try loading from DML checkpoints first
dml_data = pickle.load(open(CHECKPOINT_DIR / "dml_checkpoints/pca_features.pkl", 'rb'))

# Extract targets
y_ai = dml_data['ai_ratings']
y_sc = dml_data['social_class']
n_samples = len(y_ai)

print(f"Targets shape: AI={y_ai.shape}, SC={y_sc.shape}")

# Make sure embeddings match
X_full = embeddings_full[:n_samples]  # Trim to match

# Standardize
print("\nStandardizing features...")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# XGBoost config
xgb_model = xgb.XGBRegressor(
    n_estimators=50,  # Match original
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)

# 1. Full dimensions
print("\n1. FULL 4,096 DIMENSIONS:")
print("   Running XGBoost...")

# Use smaller CV for speed
cv_folds = 3

scores_ai_full = cross_val_score(xgb_model, X_full_scaled, y_ai, cv=cv_folds, scoring='r2')
scores_sc_full = cross_val_score(xgb_model, X_full_scaled, y_sc, cv=cv_folds, scoring='r2')

print(f"   AI Ratings R²: {scores_ai_full.mean():.4f}")
print(f"   Social Class R²: {scores_sc_full.mean():.4f}")
print(f"   Gap: {(scores_ai_full.mean() - scores_sc_full.mean()) * 100:.1f}%")

# 2. PCA 200
print("\n2. PCA 200 DIMENSIONS:")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

scores_ai_pca = cross_val_score(xgb_model, X_pca, y_ai, cv=cv_folds, scoring='r2')
scores_sc_pca = cross_val_score(xgb_model, X_pca, y_sc, cv=cv_folds, scoring='r2')

print(f"   AI Ratings R²: {scores_ai_pca.mean():.4f}")
print(f"   Social Class R²: {scores_sc_pca.mean():.4f}")
print(f"   Gap: {(scores_sc_pca.mean() - scores_sc_pca.mean()) * 100:.1f}%")

# Quick test of intermediate dimensions
print("\n3. QUICK TEST - PCA 1000:")
pca1000 = PCA(n_components=1000, random_state=42)
X_pca1000 = pca1000.fit_transform(X_full_scaled)
print(f"   Explained variance: {pca1000.explained_variance_ratio_.sum():.1%}")

scores_ai_1000 = cross_val_score(xgb_model, X_pca1000, y_ai, cv=cv_folds, scoring='r2')
print(f"   AI Ratings R²: {scores_ai_1000.mean():.4f}")

# Summary
print("\n" + "="*80)
print("COMPARISON SUMMARY:")
print("="*80)

print(f"\nDimensions | AI R² | SC R² | Gap")
print(f"-----------|-------|-------|------")
print(f"Full 4,096 | {scores_ai_full.mean():.3f} | {scores_sc_full.mean():.3f} | {(scores_ai_full.mean()-scores_sc_full.mean())*100:.0f}%")
print(f"PCA 1000   | {scores_ai_1000.mean():.3f} | ----- | ---")
print(f"PCA 200    | {scores_ai_pca.mean():.3f} | {scores_sc_pca.mean():.3f} | {(scores_ai_pca.mean()-scores_sc_pca.mean())*100:.0f}%")
print(f"Original*  | 0.597 | 0.073 | 52%")
print("\n*Original = from checkpoint analysis")

improvement = ((scores_ai_full.mean() - scores_ai_pca.mean()) / scores_ai_pca.mean() * 100)
print(f"\nFull dims vs PCA 200 improvement: {improvement:+.1f}%")

if improvement > 10:
    print("\n🚨 SIGNIFICANT: Full dimensions are MUCH better!")
    print("   The 72% variance retention loses important information.")
elif improvement > 0:
    print("\n✅ Full dimensions are slightly better.")
else:
    print("\n❌ PCA is as good or better (noise reduction).")