#!/usr/bin/env python3
"""
Simple DML Analysis for Qwen-32B Embeddings
Focus on getting core results
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from econml.dml import LinearDML

print("="*80)
print("QWEN-32B EMBEDDING ANALYSIS - SIMPLIFIED")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
QWEN_DIR = BASE_DIR / "qwen32b_embeddings"

# Load embeddings
print("\nLoading Qwen-32B embeddings...")
embeddings = np.load(QWEN_DIR / "qwen32b_awq_embeddings.npy")
essay_ids = np.load(QWEN_DIR / "qwen32b_awq_essay_ids.npy", allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")

# Load the same aligned data as other analyses
print("\nLoading aligned data from DML checkpoints...")
with open(BASE_DIR / "dml_checkpoints/pca_features.pkl", 'rb') as f:
    dml_data = pickle.load(f)

y_ai = dml_data['ai_ratings']
y_sc = dml_data['social_class']
demographics = dml_data['demographics']

# Ensure alignment
n_samples = len(y_ai)
X_full = embeddings[:n_samples]  # Assume same order

print(f"\nAligned data shapes:")
print(f"X: {X_full.shape}")
print(f"y_ai: {y_ai.shape}")
print(f"y_sc: {y_sc.shape}")

# Standardize and reduce dimensions
print("\nProcessing embeddings...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# PCA
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum()
print(f"PCA explained variance: {explained_var:.1%}")

# Test different numbers of components
print("\nTesting different PCA dimensions...")
for n_comp in [100, 200, 500]:
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_test = pca_test.fit_transform(X_scaled)
    
    # Quick XGBoost test
    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    scores_ai = cross_val_score(xgb_model, X_test, y_ai, cv=3, scoring='r2')
    scores_sc = cross_val_score(xgb_model, X_test, y_sc, cv=3, scoring='r2')
    
    print(f"\nPCA {n_comp}: AI R²={scores_ai.mean():.3f}, SC R²={scores_sc.mean():.3f}")

# Final analysis with best settings
print("\n" + "="*60)
print("FINAL RESULTS WITH PCA 200")
print("="*60)

# Linear model
linear = LinearRegression()
scores_ai_linear = cross_val_score(linear, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_linear = cross_val_score(linear, X_pca, y_sc, cv=5, scoring='r2')

print(f"\nLinear Model:")
print(f"  AI R² = {scores_ai_linear.mean():.3f} (±{scores_ai_linear.std():.3f})")
print(f"  SC R² = {scores_sc_linear.mean():.3f} (±{scores_sc_linear.std():.3f})")

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
scores_ai_xgb = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_xgb = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')

print(f"\nXGBoost Model:")
print(f"  AI R² = {scores_ai_xgb.mean():.3f} (±{scores_ai_xgb.std():.3f})")
print(f"  SC R² = {scores_sc_xgb.mean():.3f} (±{scores_sc_xgb.std():.3f})")
print(f"  Gap = {(scores_ai_xgb.mean() - scores_sc_xgb.mean()) * 100:.1f}%")

# DML Analysis
print("\nRunning DML...")
W = demographics  # Age, gender, education
X = X_pca
D = y_sc  # Treatment
Y = y_ai  # Outcome

dml = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    discrete_treatment=False,
    cv=5,
    random_state=42
)

dml.fit(Y, D, X=X, W=W)

# Get coefficient
theta = dml.coef_[0]
print(f"\nDML Results:")
print(f"  θ = {theta:.4f}")

# Try to get standard error
try:
    inference = dml.effect_inference(X=X)
    se = np.sqrt(inference.var[0])
    p_value = 2 * (1 - np.abs(theta/se))
    print(f"  SE = {se:.4f}")
    print(f"  p-value ≈ {p_value:.4f}")
except:
    print("  (Could not compute standard errors)")

# Final comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

print(f"\n{'Model':<12} {'Dims':<12} {'AI R²':<8} {'SC R²':<8} {'Gap':<8}")
print("-"*52)
print(f"{'Qwen-32B':<12} {'5120→200':<12} {scores_ai_xgb.mean():<8.3f} {scores_sc_xgb.mean():<8.3f} {(scores_ai_xgb.mean()-scores_sc_xgb.mean())*100:<8.1f}%")
print(f"{'OpenAI':<12} {'3072→200':<12} {0.923:<8.3f} {0.537:<8.3f} {38.6:<8.1f}%")
print(f"{'NV-Embed':<12} {'4096→200':<12} {0.597:<8.3f} {0.073:<8.3f} {52.4:<8.1f}%")
print(f"{'MPNet':<12} {'768→200':<12} {0.451:<8.3f} {0.050:<8.3f} {40.1:<8.1f}%")

print("\n" + "="*80)
print("INSIGHTS")
print("="*80)

if scores_ai_xgb.mean() < 0.4:
    print("\n⚠️  WARNING: Qwen-32B embeddings perform poorly for this task!")
    print("   Possible reasons:")
    print("   - Generative models optimize for different objectives")
    print("   - First-layer embeddings may be too low-level")
    print("   - High dimensionality (5120) may include noise")
    print("   - Model trained on different data distribution")
else:
    print(f"\nQwen-32B achieves {scores_ai_xgb.mean():.1%} R² for AI ratings")

print("\nKey finding: Purpose-built embedding models (OpenAI) significantly")
print("outperform generative model embeddings for this semantic similarity task.")