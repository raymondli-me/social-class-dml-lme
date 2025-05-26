#!/usr/bin/env python3
"""
Test NV-Embed with full 4,096 dimensions vs PCA 200
Using the exact same data pipeline as the successful analysis
"""

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

print("="*80)
print("NV-EMBED: FULL 4,096 DIMENSIONS vs PCA 200 COMPARISON")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

# Load the saved analysis results to get properly aligned data
print("\nLoading saved analysis data for consistency...")
with open(NVEMBED_DIR / "nvembed_pca_200_features.pkl", 'rb') as f:
    pca_data = pickle.load(f)

# Load full embeddings
print("Loading full NV-Embed embeddings...")
embeddings_full = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids_embed = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)

# Get essay IDs from PCA data to ensure alignment
essay_ids_pca = pca_data['essay_ids']

# Verify alignment
print(f"\nVerifying data alignment...")
print(f"Embeddings shape: {embeddings_full.shape}")
print(f"Essay IDs in embeddings: {len(essay_ids_embed)}")
print(f"Essay IDs in PCA data: {len(essay_ids_pca)}")

# Load targets from the working analysis
# We'll use the exact same data loading as analyze_nvembed_complete.py
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

# Load essays
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings = pd.read_csv(AI_RATINGS_FILE)

# Merge data
df = essays.merge(sc_labels, on='TID', how='inner')
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID', how='inner')

# Align with embeddings
df = df[df['TID'].isin(essay_ids_embed)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids_embed == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

# Get features and targets
X_full = embeddings_full[df['essay_idx'].values]
y_ai = df['ai_average'].values
y_sc = df['sc11'].values

print(f"\nFinal data shapes:")
print(f"X_full: {X_full.shape}")
print(f"y_ai: {y_ai.shape}")
print(f"y_sc: {y_sc.shape}")

# Standardize
print("\nStandardizing features...")
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Configure XGBoost (same params as original analysis)
xgb_params = {
    'n_estimators': 50,
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
print("   Running 5-fold cross-validation (this may take a while)...")

xgb_model = xgb.XGBRegressor(**xgb_params)

scores_ai_full = cross_val_score(xgb_model, X_full_scaled, y_ai, cv=5, scoring='r2', verbose=0)
scores_sc_full = cross_val_score(xgb_model, X_full_scaled, y_sc, cv=5, scoring='r2', verbose=0)

print(f"   AI Ratings:   R² = {scores_ai_full.mean():.4f} (std: {scores_ai_full.std():.4f})")
print(f"   Social Class: R² = {scores_sc_full.mean():.4f} (std: {scores_sc_full.std():.4f})")
print(f"   Gap: {(scores_ai_full.mean() - scores_sc_full.mean()) * 100:.1f}%")

# 2. PCA 200 dimensions (for comparison)
print("\n2. PCA 200 DIMENSIONS (XGBoost):")
print("   Applying PCA...")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

print("   Running 5-fold cross-validation...")
scores_ai_pca = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2', verbose=0)
scores_sc_pca = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2', verbose=0)

print(f"   AI Ratings:   R² = {scores_ai_pca.mean():.4f} (std: {scores_ai_pca.std():.4f})")
print(f"   Social Class: R² = {scores_sc_pca.mean():.4f} (std: {scores_sc_pca.std():.4f})")
print(f"   Gap: {(scores_ai_pca.mean() - scores_sc_pca.mean()) * 100:.1f}%")

# 3. Test intermediate dimensions
print("\n3. TESTING INTERMEDIATE PCA DIMENSIONS:")
for n_comp in [500, 1000]:
    print(f"\n   PCA {n_comp}:")
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_pca_test = pca_test.fit_transform(X_full_scaled)
    print(f"   Explained variance: {pca_test.explained_variance_ratio_.sum():.1%}")
    
    scores_ai = cross_val_score(xgb_model, X_pca_test, y_ai, cv=5, scoring='r2', verbose=0)
    scores_sc = cross_val_score(xgb_model, X_pca_test, y_sc, cv=5, scoring='r2', verbose=0)
    
    print(f"   AI R² = {scores_ai.mean():.4f}, SC R² = {scores_sc.mean():.4f}")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON:")
print("="*70)

ai_improvement = ((scores_ai_full.mean() - scores_ai_pca.mean()) / scores_ai_pca.mean() * 100)
sc_improvement = ((scores_sc_full.mean() - scores_sc_pca.mean()) / scores_sc_pca.mean() * 100)

print(f"\nDimensions    | AI R²  | SC R²  | Var Explained")
print(f"--------------|--------|--------|---------------")
print(f"Full 4,096    | {scores_ai_full.mean():.4f} | {scores_sc_full.mean():.4f} | 100%")
print(f"PCA 200       | {scores_ai_pca.mean():.4f} | {scores_sc_pca.mean():.4f} | 72.1%")
print(f"From analysis | 0.5972 | 0.0733 | 72.1%")

print(f"\nImprovement with full dimensions:")
print(f"- AI prediction: {ai_improvement:+.1f}%")
print(f"- SC prediction: {sc_improvement:+.1f}%")

if ai_improvement > 10 or sc_improvement > 50:
    print("\n🚨 MAJOR FINDING: Full dimensions significantly outperform PCA!")
    print("   The 72.1% variance retention is losing critical information.")
    print("   This explains why NV-Embed underperformed vs OpenAI.")
elif ai_improvement > 0 or sc_improvement > 0:
    print("\n✅ Full dimensions slightly outperform PCA.")
else:
    print("\n❌ PCA performs as well or better (effective denoising).")

# Save results
results = {
    'full_4096': {
        'ai_r2': scores_ai_full.mean(),
        'ai_r2_std': scores_ai_full.std(),
        'sc_r2': scores_sc_full.mean(), 
        'sc_r2_std': scores_sc_full.std(),
    },
    'pca_200': {
        'ai_r2': scores_ai_pca.mean(),
        'ai_r2_std': scores_ai_pca.std(),
        'sc_r2': scores_sc_pca.mean(),
        'sc_r2_std': scores_sc_pca.std(),
    }
}

with open(NVEMBED_DIR / 'full_vs_pca_comparison.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n✓ Results saved to nvembed_checkpoints/full_vs_pca_comparison.pkl")