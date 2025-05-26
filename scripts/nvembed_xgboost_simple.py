#!/usr/bin/env python3
"""
Simple test: XGBoost on full NV-Embed dimensions vs PCA
"""

import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

print("="*80)
print("COMPARING FULL 4,096 DIMS vs PCA 200 DIMS")
print("="*80)

# Load the saved analysis results to get aligned data
CHECKPOINT_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_checkpoints")

# Load full embeddings
print("\nLoading full embeddings...")
embeddings_full = np.load(CHECKPOINT_DIR / "nvembed_embeddings.npy")
print(f"Full embeddings shape: {embeddings_full.shape}")

# Load PCA features and get the targets from there
print("\nLoading PCA features to get aligned targets...")
with open(CHECKPOINT_DIR / "nvembed_pca_200_features.pkl", 'rb') as f:
    pca_data = pickle.load(f)

X_pca = pca_data['features']
# We need to load the targets separately
# Let's use the saved analysis results
with open(CHECKPOINT_DIR / "nvembed_analysis_results.pkl", 'rb') as f:
    results = pickle.load(f)

print(f"PCA features shape: {X_pca.shape}")
print(f"AI ratings shape: {y_ai.shape}")
print(f"Social class shape: {y_sc.shape}")

# The embeddings should already be aligned since we used them to create PCA
X_full = embeddings_full[:len(y_ai)]  # Just in case

# Standardize full embeddings
print("\nStandardizing full embeddings...")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Configure XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)

# Test 1: Full dimensions
print("\n1. FULL 4,096 DIMENSIONS:")
scores_ai_full = cross_val_score(xgb_model, X_full_scaled, y_ai, cv=5, scoring='r2')
scores_sc_full = cross_val_score(xgb_model, X_full_scaled, y_sc, cv=5, scoring='r2')

print(f"   AI Ratings R²: {scores_ai_full.mean():.4f} (±{scores_ai_full.std():.4f})")
print(f"   Social Class R²: {scores_sc_full.mean():.4f} (±{scores_sc_full.std():.4f})")
print(f"   Gap: {(scores_ai_full.mean() - scores_sc_full.mean()) * 100:.1f}%")

# Test 2: PCA 200 (for comparison)
print("\n2. PCA 200 DIMENSIONS (for verification):")
scores_ai_pca = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_pca = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')

print(f"   AI Ratings R²: {scores_ai_pca.mean():.4f} (±{scores_ai_pca.std():.4f})")
print(f"   Social Class R²: {scores_sc_pca.mean():.4f} (±{scores_sc_pca.std():.4f})")
print(f"   Gap: {(scores_ai_pca.mean() - scores_sc_pca.mean()) * 100:.1f}%")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"\nUsing FULL dimensions vs PCA:")
print(f"AI prediction improvement: {((scores_ai_full.mean() - scores_ai_pca.mean()) / scores_ai_pca.mean() * 100):.1f}%")
print(f"SC prediction improvement: {((scores_sc_full.mean() - scores_sc_pca.mean()) / scores_sc_pca.mean() * 100):.1f}%")

if scores_ai_full.mean() > scores_ai_pca.mean():
    print("\n✅ Full dimensions perform BETTER than PCA!")
    print("   PCA is losing important information.")
else:
    print("\n❌ PCA performs as well or better than full dimensions.")
    print("   PCA is effectively reducing noise.")

print("\nPrevious analysis results (from checkpoint):")
print("- PCA 200: AI R² = 0.597, SC R² = 0.073")
print("- This should match our PCA verification above.")