#!/usr/bin/env python3
"""Verify the social class prediction results and check for potential issues"""

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

def check_existing_results():
    """Check the existing DML results in detail"""
    print("=== Checking Existing Results ===")
    
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        results = pickle.load(f)
    
    # Check actual_SC results
    sc_results = results['actual_SC']
    
    for model_name, res in sc_results.items():
        print(f"\n{model_name}:")
        print(f"  R² = {res['r2']:.6f}")
        
        if 'Y_res' in res and 'model' in res:
            Y_res = res['Y_res']
            X_res = res['X_res']
            model = res['model']
            
            # Recalculate predictions
            Y_pred = model.predict(X_res)
            
            # Manual R² calculation
            ss_res = np.sum((Y_res - Y_pred) ** 2)
            ss_tot = np.sum((Y_res - np.mean(Y_res)) ** 2)
            r2_manual = 1 - (ss_res / ss_tot)
            
            # sklearn R² 
            r2_sklearn = r2_score(Y_res, Y_pred)
            
            print(f"  Manual R² = {r2_manual:.6f}")
            print(f"  Sklearn R² = {r2_sklearn:.6f}")
            print(f"  Y_res range: [{Y_res.min():.2f}, {Y_res.max():.2f}]")
            print(f"  Y_pred range: [{Y_pred.min():.2f}, {Y_pred.max():.2f}]")
            print(f"  Y_res std: {Y_res.std():.3f}")
            print(f"  Y_pred std: {Y_pred.std():.3f}")

def rerun_xgboost_social_class():
    """Re-run just XGBoost for social class with diagnostics"""
    print("\n\n=== Re-running XGBoost for Social Class ===")
    
    # Load data
    data_path = BASE_DIR / "data/essay_dataset.csv"
    df = pd.read_csv(data_path)
    df = df.rename(columns={'TID': 'id', 'original': 'response'})
    
    # Load social class labels
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    df = df.merge(sc_labels, on='TID', how='left')
    df = df.rename(columns={'sc11': 'self_perceived_social_class'})
    
    # Add demographics (dummy for now)
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    # Load PCA features
    with open(CHECKPOINT_DIR / "pca_200_features.pkl", 'rb') as f:
        pca_data = pickle.load(f)
        X_pca = pca_data['features']
    
    # Prepare data
    Y_sc = df['self_perceived_social_class'].values
    W = df[['age', 'female', 'education_level_numeric']].values
    
    print(f"\nData shapes:")
    print(f"  X_pca: {X_pca.shape}")
    print(f"  Y_sc: {Y_sc.shape}")
    print(f"  W: {W.shape}")
    print(f"\nY_sc distribution:")
    print(pd.Series(Y_sc).value_counts().sort_index())
    
    # Run DML with cross-fitting
    Y_res = np.zeros_like(Y_sc, dtype=float)
    X_res = np.zeros_like(X_pca)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nCross-fitting...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_pca)):
        print(f"  Fold {fold+1}: train={len(train_idx)}, test={len(test_idx)}")
        
        # Residualize Y
        model_Y = LinearRegression()
        model_Y.fit(W[train_idx], Y_sc[train_idx])
        Y_res[test_idx] = Y_sc[test_idx] - model_Y.predict(W[test_idx])
        
        # Residualize each X
        for j in range(X_pca.shape[1]):
            model_X = LinearRegression()
            model_X.fit(W[train_idx], X_pca[train_idx, j])
            X_res[test_idx, j] = X_pca[test_idx, j] - model_X.predict(W[test_idx])
    
    print("\nResiduals summary:")
    print(f"  Y_res: mean={Y_res.mean():.3f}, std={Y_res.std():.3f}")
    print(f"  X_res: mean={X_res.mean():.3f}, std={X_res.std():.3f}")
    
    # Try different XGBoost configurations
    configs = [
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 50, 'max_depth': 10, 'learning_rate': 0.3},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n\nXGBoost Config {i+1}: {config}")
        
        model = xgb.XGBRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        # Fit model
        model.fit(X_res, Y_res, verbose=False)
        Y_pred = model.predict(X_res)
        
        # Calculate R²
        ss_res = np.sum((Y_res - Y_pred) ** 2)
        ss_tot = np.sum((Y_res - np.mean(Y_res)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Also check if model is actually learning
        r2_sklearn = r2_score(Y_res, Y_pred)
        
        print(f"  R² (manual): {r2:.6f}")
        print(f"  R² (sklearn): {r2_sklearn:.6f}")
        print(f"  Prediction range: [{Y_pred.min():.3f}, {Y_pred.max():.3f}]")
        print(f"  Prediction std: {Y_pred.std():.3f}")
        print(f"  Actual std: {Y_res.std():.3f}")
        
        # Check if predictions are constant
        if Y_pred.std() < 0.001:
            print("  WARNING: Predictions are nearly constant!")
        
        # Feature importance
        importances = model.feature_importances_
        print(f"  Non-zero features: {np.sum(importances > 0)}")
        print(f"  Top 5 feature importances: {sorted(importances, reverse=True)[:5]}")

if __name__ == "__main__":
    check_existing_results()
    rerun_xgboost_social_class()