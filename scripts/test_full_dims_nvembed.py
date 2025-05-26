#!/usr/bin/env python3
"""
Test NV-Embed with full 4,096 dimensions using the same data pipeline
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import pickle

print("="*80)
print("NV-EMBED: FULL DIMENSIONS vs PCA COMPARISON")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

# Load embeddings
print("\nLoading NV-Embed embeddings...")
embeddings_full = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"Embeddings shape: {embeddings_full.shape}")

# Load essay data and ratings
print("\nLoading essay data...")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")

# Load ratings from the VLLM output
vllm_results = pd.read_csv(DATA_DIR / "vllm_outputs/all_results_526x50_20250524_120949.csv")

# Get social class from essays (it's in the 'actual_sc' column)
sc_data = essays[['TID', 'actual_sc']].copy()
sc_data.columns = ['TID', 'sc11']

# Get AI ratings (average across 2 prompts as in original analysis)
# Filter for the 2 prompts used in other analyses
prompt_ids = [49, 50]  # Based on the checkpoint documentation
ai_data = vllm_results[vllm_results['prompt_id'].isin(prompt_ids)]
ai_avg = ai_data.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']

# Merge everything
df = essays[['TID']]
df = df.merge(sc_data, on='TID')
df = df.merge(ai_avg, on='TID')

# Ensure we only use essays that have embeddings
df = df[df['TID'].isin(essay_ids)]

# Align embeddings with dataframe
print("\nAligning data...")
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

X_full = embeddings_full[df['essay_idx'].values]
y_ai = df['ai_average'].values
y_sc = df['sc11'].values

print(f"\nFinal shapes:")
print(f"X_full: {X_full.shape}")
print(f"y_ai: {y_ai.shape}")
print(f"y_sc: {y_sc.shape}")

# Standardize
print("\nStandardizing features...")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'  # Faster for high-dimensional data
)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)

# 1. Full dimensions
print("\n1. FULL 4,096 DIMENSIONS:")
print("   Running 5-fold CV...")
scores_ai_full = cross_val_score(xgb_model, X_full_scaled, y_ai, cv=5, scoring='r2', n_jobs=1)
scores_sc_full = cross_val_score(xgb_model, X_full_scaled, y_sc, cv=5, scoring='r2', n_jobs=1)

print(f"   AI Ratings R²: {scores_ai_full.mean():.4f} (±{scores_ai_full.std():.4f})")
print(f"   Social Class R²: {scores_sc_full.mean():.4f} (±{scores_sc_full.std():.4f})")
print(f"   Gap: {(scores_ai_full.mean() - scores_sc_full.mean()) * 100:.1f}%")

# 2. PCA 200
print("\n2. PCA 200 DIMENSIONS:")
print("   Applying PCA...")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

print("   Running 5-fold CV...")
scores_ai_pca = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2', n_jobs=1)
scores_sc_pca = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2', n_jobs=1)

print(f"   AI Ratings R²: {scores_ai_pca.mean():.4f} (±{scores_ai_pca.std():.4f})")
print(f"   Social Class R²: {scores_sc_pca.mean():.4f} (±{scores_sc_pca.std():.4f})")
print(f"   Gap: {(scores_ai_pca.mean() - scores_sc_pca.mean()) * 100:.1f}%")

# 3. Test more PCA dimensions
print("\n3. TESTING OTHER PCA DIMENSIONS:")
for n_comp in [500, 1000, 2000]:
    print(f"\n   PCA {n_comp}:")
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_pca_test = pca_test.fit_transform(X_full_scaled)
    print(f"   Explained variance: {pca_test.explained_variance_ratio_.sum():.1%}")
    
    scores_ai = cross_val_score(xgb_model, X_pca_test, y_ai, cv=5, scoring='r2', n_jobs=1)
    print(f"   AI R²: {scores_ai.mean():.4f}")

# Summary
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
improvement_ai = ((scores_ai_full.mean() - scores_ai_pca.mean()) / scores_ai_pca.mean() * 100)
improvement_sc = ((scores_sc_full.mean() - scores_sc_pca.mean()) / scores_sc_pca.mean() * 100)

print(f"\nFull dims vs PCA 200:")
print(f"- AI prediction: {scores_ai_full.mean():.4f} vs {scores_ai_pca.mean():.4f} ({improvement_ai:+.1f}%)")
print(f"- SC prediction: {scores_sc_full.mean():.4f} vs {scores_sc_pca.mean():.4f} ({improvement_sc:+.1f}%)")

if improvement_ai > 5:
    print("\n🚨 SIGNIFICANT FINDING: Full dimensions substantially outperform PCA!")
    print("   The 72.1% variance retention is losing critical information.")
elif improvement_ai > 0:
    print("\n✅ Full dimensions slightly outperform PCA.")
else:
    print("\n❌ PCA performs as well or better - it's effectively denoising.")

# Save results
results = {
    'full_4096': {
        'ai_r2_mean': scores_ai_full.mean(),
        'ai_r2_std': scores_ai_full.std(),
        'sc_r2_mean': scores_sc_full.mean(),
        'sc_r2_std': scores_sc_full.std(),
    },
    'pca_200': {
        'ai_r2_mean': scores_ai_pca.mean(),
        'ai_r2_std': scores_ai_pca.std(),
        'sc_r2_mean': scores_sc_pca.mean(),
        'sc_r2_std': scores_sc_pca.std(),
    }
}

with open(NVEMBED_DIR / 'full_dims_comparison.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n✓ Results saved to nvembed_checkpoints/full_dims_comparison.pkl")