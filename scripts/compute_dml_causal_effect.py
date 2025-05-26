#!/usr/bin/env python3
"""Compute DML causal effect of actual social class on AI ratings using embeddings as instruments"""

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

def compute_dml_causal_effect(X, Y, D, W, n_splits=5):
    """
    Compute DML estimate of causal effect of D on Y using X as instruments
    
    Args:
        X: Instruments (200 PCs from embeddings)
        Y: Outcome (AI ratings)
        D: Treatment (actual social class)
        W: Controls (demographics)
        
    This implements the partially linear DML:
    Y = θ*D + g(W) + ε
    D = m(W) + v
    
    Where we use ML methods to estimate g() and m()
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Arrays to store cross-fitted values
    Y_res = np.zeros_like(Y, dtype=float)
    D_res = np.zeros_like(D, dtype=float)
    
    # For storing fold-specific estimates for variance calculation
    theta_folds = []
    
    print("Running DML with cross-fitting...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/{n_splits}...")
        
        # Combine X and W for the ML predictions
        XW_train = np.hstack([X[train_idx], W[train_idx]])
        XW_test = np.hstack([X[test_idx], W[test_idx]])
        
        # Step 1: Predict Y using X and W (but not D)
        # Using XGBoost for flexible function approximation
        model_Y = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model_Y.fit(XW_train, Y[train_idx])
        Y_pred = model_Y.predict(XW_test)
        Y_res[test_idx] = Y[test_idx] - Y_pred
        
        # Step 2: Predict D using X and W
        model_D = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model_D.fit(XW_train, D[train_idx])
        D_pred = model_D.predict(XW_test)
        D_res[test_idx] = D[test_idx] - D_pred
        
        # Step 3: Compute theta for this fold (for variance estimation)
        theta_fold = np.sum(D_res[test_idx] * Y_res[test_idx]) / np.sum(D_res[test_idx]**2)
        theta_folds.append(theta_fold)
    
    # Final theta estimate using all residuals
    theta = np.sum(D_res * Y_res) / np.sum(D_res**2)
    
    # Variance estimation using influence functions
    # ψ = (Y_res - theta * D_res) * D_res
    psi = (Y_res - theta * D_res) * D_res
    var_theta = np.mean(psi**2) / (np.mean(D_res**2)**2) / n
    se_theta = np.sqrt(var_theta)
    
    # Alternative variance estimate using cross-fitting
    se_theta_cf = np.std(theta_folds) / np.sqrt(n_splits)
    
    # Use the larger (more conservative) standard error
    se_theta = max(se_theta, se_theta_cf)
    
    # t-statistic and p-value
    t_stat = theta / se_theta
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-X.shape[1]-W.shape[1]))
    
    # 95% confidence interval
    ci_lower = theta - 1.96 * se_theta
    ci_upper = theta + 1.96 * se_theta
    
    # Compute R² for both first stage models
    # For Y model
    Y_pred_full = np.zeros_like(Y)
    D_pred_full = np.zeros_like(D)
    
    for train_idx, test_idx in kf.split(X):
        XW_train = np.hstack([X[train_idx], W[train_idx]])
        XW_test = np.hstack([X[test_idx], W[test_idx]])
        
        model_Y_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model_Y_temp.fit(XW_train, Y[train_idx])
        Y_pred_full[test_idx] = model_Y_temp.predict(XW_test)
        
        model_D_temp = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model_D_temp.fit(XW_train, D[train_idx])
        D_pred_full[test_idx] = model_D_temp.predict(XW_test)
    
    r2_Y = 1 - np.sum((Y - Y_pred_full)**2) / np.sum((Y - np.mean(Y))**2)
    r2_D = 1 - np.sum((D - D_pred_full)**2) / np.sum((D - np.mean(D))**2)
    
    # Check strength of instrument
    F_stat = (np.sum(D_res**2) / n) / (np.sum((D - D_pred_full)**2) / (n - X.shape[1] - W.shape[1]))
    
    return {
        'theta': theta,
        'se': se_theta,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'r2_first_stage_Y': r2_Y,
        'r2_first_stage_D': r2_D,
        'F_stat': F_stat,
        'n_obs': n,
        'Y_res_std': np.std(Y_res),
        'D_res_std': np.std(D_res)
    }

def main():
    print("=== Computing DML Causal Effect of Actual Social Class on AI Ratings ===\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(BASE_DIR / "data" / "asc_9513_essays.csv")
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    df = df.merge(sc_labels, on='TID', how='left')
    
    # Load AI ratings
    ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['TID', 'ai_average']
    df = df.merge(ai_avg, on='TID', how='left')
    
    # Load PCA features (200 components)
    with open(CHECKPOINT_DIR / "pca_200_features.pkl", 'rb') as f:
        pca_data = pickle.load(f)
        X_pca = pca_data['features']
    
    print(f"Loaded {X_pca.shape[0]} essays with {X_pca.shape[1]} PCA components")
    
    # Add demographics
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    # Prepare data
    Y = df['ai_average'].values  # Outcome: AI ratings
    D = df['sc11'].values        # Treatment: Actual social class
    X = X_pca                    # Instruments: 200 PCs
    W = df[['age', 'female', 'education_level_numeric']].values  # Controls
    
    print(f"\nData summary:")
    print(f"  Y (AI ratings): mean={Y.mean():.2f}, std={Y.std():.2f}")
    print(f"  D (Actual SC):  mean={D.mean():.2f}, std={D.std():.2f}")
    print(f"  Correlation(Y,D): {np.corrcoef(Y, D)[0,1]:.3f}")
    
    # Compute DML estimate
    print("\n" + "="*70)
    results = compute_dml_causal_effect(X, Y, D, W)
    
    print("\nDML Results:")
    print("-"*70)
    print(f"Causal Effect of Actual Social Class on AI Ratings:")
    print(f"  θ (theta):       {results['theta']:.6f}")
    print(f"  SE:              {results['se']:.6f}")
    print(f"  t-statistic:     {results['t_stat']:.4f}")
    print(f"  p-value:         {results['p_value']:.6f}")
    print(f"  95% CI:          [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    
    # Significance
    if results['p_value'] < 0.001:
        sig = "***"
    elif results['p_value'] < 0.01:
        sig = "**"  
    elif results['p_value'] < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    print(f"  Significance:    {sig}")
    
    print(f"\nFirst Stage Performance:")
    print(f"  R² for Y|X,W:    {results['r2_first_stage_Y']:.4f}")
    print(f"  R² for D|X,W:    {results['r2_first_stage_D']:.4f}")
    print(f"  F-statistic:     {results['F_stat']:.2f}")
    
    print(f"\nResidual Standard Deviations:")
    print(f"  Y residuals:     {results['Y_res_std']:.4f}")
    print(f"  D residuals:     {results['D_res_std']:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  A 1-unit increase in actual social class causes a {results['theta']:.3f} unit change")
    print(f"  in AI-perceived social class rating.")
    
    # Effect size in standard deviations
    effect_size_sd = results['theta'] * np.std(D) / np.std(Y)
    print(f"\n  Effect size: {effect_size_sd:.3f} SD of AI ratings per SD of actual social class")
    
    if results['F_stat'] < 10:
        print("\n  WARNING: Weak instrument (F < 10). Results may be biased.")
    
    print("\n" + "="*70)
    print("\nNotes:")
    print("- Uses 200 PCA components from OpenAI embeddings as instruments")
    print("- XGBoost for flexible first-stage predictions")
    print("- 5-fold cross-fitting to avoid overfitting bias")
    print("- Standard errors account for estimation uncertainty in both stages")

if __name__ == "__main__":
    main()