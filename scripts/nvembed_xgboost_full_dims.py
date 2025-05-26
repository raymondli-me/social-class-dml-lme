#!/usr/bin/env python3
"""
Test NV-Embed-v2 with XGBoost on FULL 4,096 dimensions (no PCA)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

print("="*80)
print("NV-EMBED-V2 ANALYSIS WITH FULL 4,096 DIMENSIONS")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "nvembed_checkpoints"

# Load embeddings
print("\n=== Loading Full NV-Embed Embeddings ===")
embeddings_full = np.load(CHECKPOINT_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(CHECKPOINT_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"✓ Embeddings shape: {embeddings_full.shape}")

# Load other data
print("\n=== Loading Target Data ===")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
sc_labels = pd.read_csv(DATA_DIR / "asc_9513_sc11.csv")
ai_ratings = pd.read_csv(DATA_DIR / "asc_9513_llm_ratings.csv")

# Merge and align
df = essays[['TID', 'original']]
df = df.merge(sc_labels[['TID', 'sc11']], on='TID')
df = df[df['TID'].isin(essay_ids)]

# Get AI ratings average
ai_avg = ai_ratings.groupby('TID')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID')

# Ensure alignment
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

print(f"✓ Final dataset: {len(df)} essays")

# Prepare data
X_full = embeddings_full[df['essay_idx'].values]
y_ai = df['ai_average'].values
y_sc = df['sc11'].values

print(f"\n=== Data Shapes ===")
print(f"X (full): {X_full.shape}")
print(f"y_ai: {y_ai.shape}")
print(f"y_sc: {y_sc.shape}")

# Standardize features
print("\n=== Standardizing Features ===")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Test XGBoost with different numbers of dimensions
print("\n=== XGBoost Performance Comparison ===")

# 1. Full 4,096 dimensions
print("\n1. FULL 4,096 DIMENSIONS:")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# AI ratings prediction
scores_ai_full = cross_val_score(xgb_model, X_full_scaled, y_ai, cv=5, scoring='r2')
print(f"   AI Ratings R²: {scores_ai_full.mean():.4f} (±{scores_ai_full.std():.4f})")

# Social class prediction
scores_sc_full = cross_val_score(xgb_model, X_full_scaled, y_sc, cv=5, scoring='r2')
print(f"   Social Class R²: {scores_sc_full.mean():.4f} (±{scores_sc_full.std():.4f})")
print(f"   Gap: {(scores_ai_full.mean() - scores_sc_full.mean()) * 100:.1f}%")

# 2. Compare with PCA results
print("\n2. PCA 200 DIMENSIONS (from previous analysis):")
print("   AI Ratings R²: 0.5972")
print("   Social Class R²: 0.0733")
print("   Gap: 52.4%")

# 3. Test intermediate dimensions
print("\n3. TESTING INTERMEDIATE DIMENSIONS:")
from sklearn.decomposition import PCA

for n_components in [500, 1000, 2000]:
    print(f"\n   PCA {n_components} dimensions:")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_full_scaled)
    
    # AI ratings
    scores_ai = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
    print(f"   AI Ratings R²: {scores_ai.mean():.4f}")
    
    # Social class
    scores_sc = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')
    print(f"   Social Class R²: {scores_sc.mean():.4f}")
    print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# Save results
results = {
    'full_4096': {
        'ai_r2': scores_ai_full.mean(),
        'ai_r2_std': scores_ai_full.std(),
        'sc_r2': scores_sc_full.mean(),
        'sc_r2_std': scores_sc_full.std(),
        'gap': scores_ai_full.mean() - scores_sc_full.mean()
    }
}

with open(CHECKPOINT_DIR / 'nvembed_full_dims_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "="*80)
print("SUMMARY:")
print(f"Using full 4,096 dimensions vs 200 PCA components:")
print(f"- AI prediction: {scores_ai_full.mean():.4f} vs 0.5972")
print(f"- SC prediction: {scores_sc_full.mean():.4f} vs 0.0733")
print("="*80)