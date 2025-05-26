#!/usr/bin/env python3
"""Re-run XGBoost for actual social class with proper data loading"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

print("=== Re-running Social Class Prediction with XGBoost ===")

# Load the FULL essay dataset (9,513 essays)
print("\n1. Loading data...")
df = pd.read_csv(BASE_DIR / "data" / "asc_9513_essays.csv")
print(f"   Essays loaded: {len(df)}")

# Load social class labels
sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
df = df.merge(sc_labels, on='TID', how='left')
print(f"   Essays with SC labels: {df['sc11'].notna().sum()}")

# Load PCA features
with open(CHECKPOINT_DIR / "pca_200_features.pkl", 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
print(f"   PCA features shape: {X_pca.shape}")

# Add demographics (using the same random seed as original)
np.random.seed(42)
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))

# Prepare data
Y_sc = df['sc11'].values
W = df[['age', 'female', 'education_level_numeric']].values

print(f"\n2. Data shapes:")
print(f"   X_pca: {X_pca.shape}")
print(f"   Y_sc: {Y_sc.shape}")
print(f"   W: {W.shape}")

print(f"\n3. Social class distribution:")
print(pd.Series(Y_sc).value_counts().sort_index())

# Run DML with cross-fitting
print(f"\n4. Running DML cross-fitting...")
Y_res = np.zeros_like(Y_sc, dtype=float)
X_res = np.zeros_like(X_pca)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(X_pca)):
    print(f"   Fold {fold+1}/5...")
    
    # Residualize Y
    model_Y = LinearRegression()
    model_Y.fit(W[train_idx], Y_sc[train_idx])
    Y_res[test_idx] = Y_sc[test_idx] - model_Y.predict(W[test_idx])
    
    # Residualize each X
    for j in range(X_pca.shape[1]):
        model_X = LinearRegression()
        model_X.fit(W[train_idx], X_pca[train_idx, j])
        X_res[test_idx, j] = X_pca[test_idx, j] - model_X.predict(W[test_idx])

print(f"\n5. Residuals summary:")
print(f"   Y_res: mean={Y_res.mean():.6f}, std={Y_res.std():.3f}")
print(f"   Y_res range: [{Y_res.min():.3f}, {Y_res.max():.3f}]")
print(f"   X_res: mean={X_res.mean():.6f}, std={X_res.std():.3f}")

# Fit XGBoost
print(f"\n6. Fitting XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

model.fit(X_res, Y_res)
Y_pred = model.predict(X_res)

# Calculate R²
ss_res = np.sum((Y_res - Y_pred) ** 2)
ss_tot = np.sum((Y_res - np.mean(Y_res)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

print(f"\n7. Results:")
print(f"   R² = {r2:.6f}")
print(f"   Prediction range: [{Y_pred.min():.3f}, {Y_pred.max():.3f}]")
print(f"   Prediction std: {Y_pred.std():.3f}")
print(f"   Actual residual std: {Y_res.std():.3f}")

# Feature importance
importances = model.feature_importances_
print(f"\n8. Feature importance:")
print(f"   Non-zero features: {np.sum(importances > 0)}")
print(f"   Top 10 feature importances: {sorted(importances, reverse=True)[:10]}")

# Also try without residualization for comparison
print(f"\n9. Comparison: Direct prediction without residualization")
model_direct = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
model_direct.fit(X_pca, Y_sc)
Y_pred_direct = model_direct.predict(X_pca)
r2_direct = r2_score(Y_sc, Y_pred_direct)
print(f"   Direct R² = {r2_direct:.6f}")

# Cross-validation for direct prediction
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model_direct, X_pca, Y_sc, cv=5, scoring='r2')
print(f"   5-fold CV R² = {cv_scores.mean():.6f} (+/- {cv_scores.std() * 2:.6f})")
print(f"   Individual folds: {cv_scores}")