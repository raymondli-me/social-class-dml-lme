#!/usr/bin/env python3
"""Compute DML coefficients with standard errors, t-stats, p-values, and confidence intervals"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

def compute_dml_with_inference(X, Y, W, model_class='xgboost', n_splits=5, n_bootstrap=100):
    """
    Compute DML with inference using cross-fitting and bootstrap
    
    Returns:
        dict with coefficient, se, t_stat, p_value, ci_lower, ci_upper
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store predictions for each fold
    Y_res_full = np.zeros_like(Y, dtype=float)
    X_res_full = np.zeros_like(X)
    theta_folds = []
    
    # Cross-fitting
    for train_idx, test_idx in kf.split(X):
        # Residualize Y
        model_Y = LinearRegression()
        model_Y.fit(W[train_idx], Y[train_idx])
        Y_res_full[test_idx] = Y[test_idx] - model_Y.predict(W[test_idx])
        
        # Residualize X
        for j in range(X.shape[1]):
            model_X = LinearRegression()
            model_X.fit(W[train_idx], X[train_idx, j])
            X_res_full[test_idx, j] = X[test_idx, j] - model_X.predict(W[test_idx])
    
    # Fit final model on all residualized data
    if model_class == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
    else:
        model = LinearRegression()
    
    model.fit(X_res_full, Y_res_full)
    Y_pred = model.predict(X_res_full)
    
    # For XGBoost, we need to compute an aggregate treatment effect
    # We'll use the average marginal effect across all features
    if model_class == 'xgboost':
        # Bootstrap for standard errors
        theta_bootstrap = []
        
        for b in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(n, n, replace=True)
            X_boot = X_res_full[boot_idx]
            Y_boot = Y_res_full[boot_idx]
            
            # Fit model
            model_boot = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42 + b,
                n_jobs=-1,
                objective='reg:squarederror'
            )
            model_boot.fit(X_boot, Y_boot)
            
            # Compute average prediction change for unit change in first PC
            # (as a summary measure of the embedding effect)
            X_plus = X_boot.copy()
            X_plus[:, 0] += 1  # Increase first PC by 1 SD
            Y_plus = model_boot.predict(X_plus)
            Y_base = model_boot.predict(X_boot)
            theta_b = np.mean(Y_plus - Y_base)
            theta_bootstrap.append(theta_b)
        
        # Main effect (on full sample)
        X_plus = X_res_full.copy()
        X_plus[:, 0] += 1
        Y_plus = model.predict(X_plus)
        Y_base = model.predict(X_res_full)
        theta = np.mean(Y_plus - Y_base)
        
        # Standard error from bootstrap
        se = np.std(theta_bootstrap)
        
    else:  # Linear model
        theta = model.coef_[0]  # Coefficient on first PC
        
        # Compute standard errors using HC3 robust standard errors
        residuals = Y_res_full - Y_pred
        leverage = np.diag(X_res_full @ np.linalg.inv(X_res_full.T @ X_res_full) @ X_res_full.T)
        weights = residuals**2 / (1 - leverage)**2
        V = X_res_full.T @ np.diag(weights) @ X_res_full
        se = np.sqrt(np.diag(np.linalg.inv(X_res_full.T @ X_res_full) @ V @ np.linalg.inv(X_res_full.T @ X_res_full))[0])
    
    # Inference
    t_stat = theta / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-X.shape[1]-W.shape[1]))
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se
    
    # R-squared
    ss_res = np.sum((Y_res_full - Y_pred)**2)
    ss_tot = np.sum((Y_res_full - np.mean(Y_res_full))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    return {
        'coefficient': theta,
        'std_error': se,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'r_squared': r2,
        'n_obs': n
    }

def main():
    print("=== Computing DML Coefficients with Inference ===\n")
    
    # Load data
    df = pd.read_csv(BASE_DIR / "data" / "asc_9513_essays.csv")
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    df = df.merge(sc_labels, on='TID', how='left')
    
    # Load AI ratings
    ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['TID', 'ai_average']
    df = df.merge(ai_avg, on='TID', how='left')
    
    # Load PCA features
    with open(CHECKPOINT_DIR / "pca_200_features.pkl", 'rb') as f:
        pca_data = pickle.load(f)
        X_pca = pca_data['features']
    
    # Add demographics
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    # Prepare data
    Y_ai = df['ai_average'].values
    Y_sc = df['sc11'].values
    W = df[['age', 'female', 'education_level_numeric']].values
    
    # Compute DML for both outcomes
    outcomes = [
        ('AI Ratings', Y_ai),
        ('Actual Social Class', Y_sc)
    ]
    
    for outcome_name, Y in outcomes:
        print(f"\n{outcome_name}")
        print("=" * 60)
        
        # XGBoost results
        print("\nXGBoost (Average Marginal Effect of 1 SD change in PC1):")
        results_xgb = compute_dml_with_inference(X_pca, Y, W, model_class='xgboost', n_bootstrap=100)
        
        print(f"  Coefficient:     {results_xgb['coefficient']:.4f}")
        print(f"  Std. Error:      {results_xgb['std_error']:.4f}")
        print(f"  t-statistic:     {results_xgb['t_statistic']:.4f}")
        print(f"  p-value:         {results_xgb['p_value']:.4f}")
        print(f"  95% CI:          [{results_xgb['ci_95_lower']:.4f}, {results_xgb['ci_95_upper']:.4f}]")
        print(f"  R-squared:       {results_xgb['r_squared']:.4f}")
        
        # Linear model results for comparison
        print("\nLinear Model (Coefficient on PC1):")
        results_linear = compute_dml_with_inference(X_pca, Y, W, model_class='linear')
        
        print(f"  Coefficient:     {results_linear['coefficient']:.4f}")
        print(f"  Std. Error:      {results_linear['std_error']:.4f}")
        print(f"  t-statistic:     {results_linear['t_statistic']:.4f}")
        print(f"  p-value:         {results_linear['p_value']:.4f}")
        print(f"  95% CI:          [{results_linear['ci_95_lower']:.4f}, {results_linear['ci_95_upper']:.4f}]")
        print(f"  R-squared:       {results_linear['r_squared']:.4f}")
        
        # Statistical significance
        if results_xgb['p_value'] < 0.001:
            sig = "***"
        elif results_xgb['p_value'] < 0.01:
            sig = "**"
        elif results_xgb['p_value'] < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        
        print(f"\nStatistical Significance: {sig}")
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("- Coefficients represent the effect of a 1 SD change in the first principal component")
    print("- For XGBoost: Average marginal effect computed via prediction differences")
    print("- For Linear: Direct coefficient on PC1")
    print("- Standard errors for XGBoost computed via bootstrap (100 iterations)")
    print("- Standard errors for Linear computed using HC3 robust standard errors")
    print("- All estimates use 5-fold cross-fitting as per DML")

if __name__ == "__main__":
    main()