#!/usr/bin/env python3
"""Diagnose why social class prediction is failing"""

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

def diagnose_residualization():
    """Check what's happening with the residualization"""
    print("=== Diagnosing Residualization Issue ===")
    
    # Load the DML results to get the residualized values
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        results = pickle.load(f)
    
    # Get XGBoost results for both outcomes
    ai_xgb = results['AI_ratings']['XGBoost']
    sc_xgb = results['actual_SC']['XGBoost']
    
    print("\nAI Ratings (working well):")
    print(f"  Y_res shape: {ai_xgb['Y_res'].shape}")
    print(f"  Y_res mean: {ai_xgb['Y_res'].mean():.6f}")
    print(f"  Y_res std: {ai_xgb['Y_res'].std():.6f}")
    print(f"  Y_res range: [{ai_xgb['Y_res'].min():.3f}, {ai_xgb['Y_res'].max():.3f}]")
    
    print("\nActual Social Class (failing):")
    print(f"  Y_res shape: {sc_xgb['Y_res'].shape}")
    print(f"  Y_res mean: {sc_xgb['Y_res'].mean():.6f}")
    print(f"  Y_res std: {sc_xgb['Y_res'].std():.6f}")
    print(f"  Y_res range: [{sc_xgb['Y_res'].min():.3f}, {sc_xgb['Y_res'].max():.3f}]")
    
    # Check unique values
    print(f"\n  Unique Y_res values: {len(np.unique(sc_xgb['Y_res']))}")
    print(f"  First 20 Y_res values: {sc_xgb['Y_res'][:20]}")

def test_simple_prediction():
    """Test if we can predict social class without residualization"""
    print("\n\n=== Testing Direct Prediction (No Residualization) ===")
    
    # Load PCA features
    with open(CHECKPOINT_DIR / "pca_200_features.pkl", 'rb') as f:
        pca_data = pickle.load(f)
        X_pca = pca_data['features']
    
    # Load the data properly
    # First load essays
    data_path = BASE_DIR / "data/essay_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Load social class labels
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    
    # Merge using the correct column names
    df = df.merge(sc_labels, on='TID', how='left')
    Y_sc = df['sc11'].values
    
    print(f"\nSocial class distribution:")
    print(pd.Series(Y_sc).value_counts().sort_index())
    print(f"\nY shape: {Y_sc.shape}")
    print(f"X_pca shape: {X_pca.shape}")
    
    # Simple train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, Y_sc, test_size=0.2, random_state=42
    )
    
    # Try different models without residualization
    models = {
        'Linear': LinearRegression(),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # R² scores
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        print(f"  Train R²: {r2_train:.4f}")
        print(f"  Test R²: {r2_test:.4f}")
        print(f"  Prediction range: [{y_pred_test.min():.2f}, {y_pred_test.max():.2f}]")
        print(f"  Actual range: [{y_test.min():.2f}, {y_test.max():.2f}]")

def check_demographics_impact():
    """Check how demographics relate to social class"""
    print("\n\n=== Checking Demographics Impact ===")
    
    # Load the full dataset with AI ratings
    ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    
    # Load essays and social class
    df = pd.read_csv(BASE_DIR / "data/essay_dataset.csv")
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    df = df.merge(sc_labels, on='TID', how='left')
    
    # Add dummy demographics
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    # Check correlation with social class
    W = df[['age', 'female', 'education_level_numeric']].values
    Y_sc = df['sc11'].values
    
    # Fit linear model to see how much demographics explain
    model = LinearRegression()
    model.fit(W, Y_sc)
    Y_pred = model.predict(W)
    r2 = r2_score(Y_sc, Y_pred)
    
    print(f"Demographics → Social Class R²: {r2:.4f}")
    print(f"Coefficients: age={model.coef_[0]:.3f}, female={model.coef_[1]:.3f}, education={model.coef_[2]:.3f}")
    
    # Check residuals
    Y_res = Y_sc - Y_pred
    print(f"\nResiduals after demographics:")
    print(f"  Mean: {Y_res.mean():.6f}")
    print(f"  Std: {Y_res.std():.6f}")
    print(f"  Range: [{Y_res.min():.3f}, {Y_res.max():.3f}]")

if __name__ == "__main__":
    diagnose_residualization()
    test_simple_prediction()
    check_demographics_impact()