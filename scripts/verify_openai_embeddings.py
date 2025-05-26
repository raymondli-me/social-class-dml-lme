#!/usr/bin/env python3
"""
Verify OpenAI embeddings using the EXACT same methodology as other models
To ensure fair comparison
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from econml.dml import LinearDML

print("="*80)
print("VERIFYING OPENAI EMBEDDINGS WITH CONSISTENT METHODOLOGY")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
OPENAI_DIR = BASE_DIR / "openai_checkpoints"

# Check if OpenAI embeddings exist
print("\nChecking for OpenAI embeddings...")
embeddings_file = OPENAI_DIR / "openai_embeddings.npy"
pca_file = OPENAI_DIR / "pca_200_features.pkl"

if embeddings_file.exists():
    print("✓ Found raw embeddings file")
    openai_embeddings = np.load(embeddings_file)
    print(f"  Shape: {openai_embeddings.shape}")
else:
    print("❌ Raw embeddings not found")

if pca_file.exists():
    print("✓ Found PCA features file")
    with open(pca_file, 'rb') as f:
        pca_data = pickle.load(f)
    print(f"  Keys: {list(pca_data.keys())}")
else:
    print("❌ PCA features not found")

# Load data using EXACT same approach as other analyses
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

print("\nLoading data (same as other analyses)...")
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings = pd.read_csv(AI_RATINGS_FILE)

# Merge exactly as in other analyses
df = essays.merge(sc_labels, on='TID', how='inner')
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID', how='inner')

# Generate demographics with same seed
np.random.seed(42)
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))

print(f"Final dataset: {len(df)} essays")

# Get targets
y_ai = df['ai_average'].values
y_sc = df['sc11'].values
W = df[['age', 'female', 'education_level_numeric']].values

print(f"\nTarget shapes: y_ai={y_ai.shape}, y_sc={y_sc.shape}")

# Try to load OpenAI PCA features
if pca_file.exists():
    with open(pca_file, 'rb') as f:
        pca_data = pickle.load(f)
    
    # Extract features (try different keys)
    if 'features' in pca_data:
        X_pca = pca_data['features']
    elif 'X' in pca_data:
        X_pca = pca_data['X']
    elif 'features_pca' in pca_data:
        X_pca = pca_data['features_pca']
    else:
        print(f"Available keys: {list(pca_data.keys())}")
        raise KeyError("Cannot find PCA features in saved file")
    
    print(f"\nLoaded PCA features: {X_pca.shape}")
    
    # Ensure alignment
    if X_pca.shape[0] != len(df):
        print(f"⚠️  Shape mismatch! PCA has {X_pca.shape[0]} samples, data has {len(df)}")
        # Try to align if essay IDs are available
        if 'essay_ids' in pca_data:
            essay_ids = pca_data['essay_ids']
            # Create alignment
            df_aligned = df[df['TID'].isin(essay_ids)]
            if len(df_aligned) == len(essay_ids):
                print(f"✓ Successfully aligned {len(df_aligned)} essays")
                y_ai = df_aligned['ai_average'].values
                y_sc = df_aligned['sc11'].values
                W = df_aligned[['age', 'female', 'education_level_numeric']].values
            else:
                print("❌ Could not align data")
else:
    print("❌ No PCA file found - cannot proceed")
    exit()

# Run EXACT same evaluation as other models
print("\n" + "="*60)
print("EVALUATION WITH CONSISTENT METHODOLOGY")
print("="*60)

# 1. Linear Model
print("\nLinear Model (5-fold CV):")
linear = LinearRegression()
scores_ai_linear = cross_val_score(linear, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_linear = cross_val_score(linear, X_pca, y_sc, cv=5, scoring='r2')
print(f"  AI R² = {scores_ai_linear.mean():.3f} (±{scores_ai_linear.std():.3f})")
print(f"  SC R² = {scores_sc_linear.mean():.3f} (±{scores_sc_linear.std():.3f})")

# 2. XGBoost Model (same hyperparameters as other analyses)
print("\nXGBoost Model (5-fold CV):")
xgb_model = xgb.XGBRegressor(
    n_estimators=50,  # Same as NV-Embed analysis
    max_depth=6,      # Same as NV-Embed analysis
    random_state=42,
    n_jobs=-1
)
scores_ai_xgb = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_xgb = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')
print(f"  AI R² = {scores_ai_xgb.mean():.3f} (±{scores_ai_xgb.std():.3f})")
print(f"  SC R² = {scores_sc_xgb.mean():.3f} (±{scores_sc_xgb.std():.3f})")
print(f"  Gap = {(scores_ai_xgb.mean() - scores_sc_xgb.mean()) * 100:.1f}%")

# 3. DML Analysis (same as other models)
print("\nDML Analysis (XGBoost first stage):")
dml = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    discrete_treatment=False,
    cv=5,
    random_state=42
)

dml.fit(y_ai, y_sc, X=X_pca, W=W)
theta = dml.coef_[0]
print(f"  θ = {theta:.4f}")

# Compare with claimed results
print("\n" + "="*60)
print("COMPARISON WITH CLAIMED RESULTS")
print("="*60)

print("\nClaimed OpenAI results:")
print("  AI R² = 0.923")
print("  SC R² = 0.537")
print("  θ = 0.0527")

print("\nActual results with consistent methodology:")
print(f"  AI R² = {scores_ai_xgb.mean():.3f}")
print(f"  SC R² = {scores_sc_xgb.mean():.3f}")
print(f"  θ = {theta:.4f}")

if abs(scores_ai_xgb.mean() - 0.923) > 0.05:
    print("\n⚠️  SIGNIFICANT DISCREPANCY DETECTED!")
    print("   The original analysis may have:")
    print("   - Used training R² instead of CV R²")
    print("   - Used different hyperparameters")
    print("   - Had data alignment issues")
else:
    print("\n✅ Results are consistent with claims")

# Also test with original hyperparameters if different
print("\n" + "="*60)
print("TESTING WITH ORIGINAL HYPERPARAMETERS")
print("="*60)

print("\nXGBoost with n_estimators=100, max_depth=5 (from original):")
xgb_orig = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
scores_ai_orig = cross_val_score(xgb_orig, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_orig = cross_val_score(xgb_orig, X_pca, y_sc, cv=5, scoring='r2')
print(f"  AI R² = {scores_ai_orig.mean():.3f}")
print(f"  SC R² = {scores_sc_orig.mean():.3f}")

print("\n" + "="*60)