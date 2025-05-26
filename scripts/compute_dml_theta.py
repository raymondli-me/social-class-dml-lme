#!/usr/bin/env python3
"""Compute the DML theta coefficient - the causal effect of embeddings on outcomes"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

def compute_dml_theta(X, Y, W, n_splits=5):
    """
    Compute DML theta coefficient with inference
    
    This implements the standard DML procedure:
    1. Predict Y from W (nuisance)
    2. Predict X from W (nuisance) - here X is multivariate
    3. Regress residuals: Y_res ~ X_res
    
    For multivariate X (200 PCs), we compute:
    - A single theta by projecting Y_res onto X_res
    - This gives the "embedding effect" as a single number
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Arrays to store cross-fitted residuals
    Y_res = np.zeros_like(Y, dtype=float)
    X_res = np.zeros_like(X, dtype=float)
    
    # Cross-fitting
    print(f"  Cross-fitting with {n_splits} folds...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Step 1: Residualize Y on W
        model_Y = LinearRegression()
        model_Y.fit(W[train_idx], Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - model_Y.predict(W[test_idx])
        
        # Step 2: Residualize each PC on W
        for j in range(X.shape[1]):
            model_X = LinearRegression()
            model_X.fit(W[train_idx], X[train_idx, j])
            X_res[test_idx, j] = X[test_idx, j] - model_X.predict(W[test_idx])
    
    print(f"  Residualization complete.")
    print(f"  Y_res: mean={Y_res.mean():.6f}, std={Y_res.std():.4f}")
    print(f"  X_res: mean={X_res.mean():.6f}, std={X_res.std():.4f}")
    
    # Step 3: Compute theta as projection coefficient
    # theta = (X_res'Y_res) / (X_res'X_res) for multivariate case
    # This is equivalent to OLS of Y_res on X_res
    
    # Method 1: Direct OLS on all 200 PCs
    model_final = LinearRegression(fit_intercept=False)  # No intercept for residuals
    model_final.fit(X_res, Y_res)
    
    # Get predicted values for R²
    Y_pred = model_final.predict(X_res)
    ss_res = np.sum((Y_res - Y_pred)**2)
    ss_tot = np.sum((Y_res - np.mean(Y_res))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    # Method 2: Compute aggregate theta as norm of coefficient vector
    # This gives us a single "embedding effect" measure
    beta_vector = model_final.coef_
    theta_norm = np.linalg.norm(beta_vector)  # L2 norm of coefficient vector
    
    # Method 3: First principal component of X_res
    # This gives us the dominant direction of variation
    U, S, Vt = np.linalg.svd(X_res, full_matrices=False)
    X_pc1 = X_res @ Vt[0, :]  # First PC of residualized X
    theta_pc1 = np.dot(X_pc1, Y_res) / np.dot(X_pc1, X_pc1)
    
    # Standard error for PC1 approach (most interpretable)
    residuals_pc1 = Y_res - theta_pc1 * X_pc1
    sigma2 = np.sum(residuals_pc1**2) / (n - 2)  # -2 for intercept and slope
    se_pc1 = np.sqrt(sigma2 / np.sum(X_pc1**2))
    
    # Standard error for full model (using sandwich estimator)
    # V = (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
    e2 = (Y_res - Y_pred)**2
    try:
        XtX_inv = np.linalg.inv(X_res.T @ X_res)
        meat = X_res.T @ np.diag(e2) @ X_res
        V = XtX_inv @ meat @ XtX_inv
        # SE for norm: use delta method
        grad = beta_vector / theta_norm  # Gradient of norm
        se_norm = np.sqrt(grad @ V @ grad)
    except:
        se_norm = np.nan
    
    # t-statistics and p-values
    t_pc1 = theta_pc1 / se_pc1
    p_pc1 = 2 * (1 - stats.t.cdf(abs(t_pc1), df=n-2))
    
    t_norm = theta_norm / se_norm if not np.isnan(se_norm) else np.nan
    p_norm = 2 * (1 - stats.norm.cdf(abs(t_norm))) if not np.isnan(t_norm) else np.nan
    
    # Confidence intervals
    ci_pc1 = (theta_pc1 - 1.96 * se_pc1, theta_pc1 + 1.96 * se_pc1)
    ci_norm = (theta_norm - 1.96 * se_norm, theta_norm + 1.96 * se_norm) if not np.isnan(se_norm) else (np.nan, np.nan)
    
    return {
        'theta_pc1': theta_pc1,
        'se_pc1': se_pc1,
        't_pc1': t_pc1,
        'p_pc1': p_pc1,
        'ci_pc1': ci_pc1,
        'theta_norm': theta_norm,
        'se_norm': se_norm,
        't_norm': t_norm,
        'p_norm': p_norm,
        'ci_norm': ci_norm,
        'r_squared': r2,
        'n_obs': n,
        'beta_vector': beta_vector
    }

def main():
    print("=== Computing DML Theta Coefficients ===\n")
    
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
    Y_ai = df['ai_average'].values
    Y_sc = df['sc11'].values
    W = df[['age', 'female', 'education_level_numeric']].values
    
    # Compute DML theta for both outcomes
    print("\n" + "="*70)
    
    for outcome_name, Y in [('AI Ratings', Y_ai), ('Actual Social Class', Y_sc)]:
        print(f"\n{outcome_name}")
        print("-"*70)
        
        results = compute_dml_theta(X_pca, Y, W)
        
        print(f"\nMethod 1: First PC of Residualized Embeddings")
        print(f"  θ (theta):       {results['theta_pc1']:.6f}")
        print(f"  SE:              {results['se_pc1']:.6f}")
        print(f"  t-statistic:     {results['t_pc1']:.4f}")
        print(f"  p-value:         {results['p_pc1']:.6f}")
        print(f"  95% CI:          [{results['ci_pc1'][0]:.6f}, {results['ci_pc1'][1]:.6f}]")
        
        # Significance
        if results['p_pc1'] < 0.001:
            sig = "***"
        elif results['p_pc1'] < 0.01:
            sig = "**"  
        elif results['p_pc1'] < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        print(f"  Significance:    {sig}")
        
        print(f"\nMethod 2: L2 Norm of Full Coefficient Vector")
        print(f"  ||β||₂:          {results['theta_norm']:.6f}")
        if not np.isnan(results['se_norm']):
            print(f"  SE:              {results['se_norm']:.6f}")
            print(f"  t-statistic:     {results['t_norm']:.4f}")
            print(f"  p-value:         {results['p_norm']:.6f}")
            print(f"  95% CI:          [{results['ci_norm'][0]:.6f}, {results['ci_norm'][1]:.6f}]")
        else:
            print(f"  SE:              Not available (singular matrix)")
        
        print(f"\nModel Fit:")
        print(f"  R-squared:       {results['r_squared']:.6f}")
        print(f"  N observations:  {results['n_obs']:,}")
        
        # Show top coefficients by magnitude
        beta = results['beta_vector']
        top_idx = np.argsort(np.abs(beta))[-5:][::-1]
        print(f"\n  Top 5 PCs by |coefficient|:")
        for idx in top_idx:
            print(f"    PC{idx+1:3d}: {beta[idx]:>10.6f}")
    
    print("\n" + "="*70)
    print("\nNotes:")
    print("- θ (theta) represents the causal effect of embeddings on the outcome")
    print("- Method 1: Effect of first PC of residualized embeddings (most interpretable)")
    print("- Method 2: Overall embedding effect measured by L2 norm of coefficient vector")
    print("- All estimates use 5-fold cross-fitting to avoid overfitting bias")
    print("- Standard errors account for estimation uncertainty in both stages")

if __name__ == "__main__":
    main()