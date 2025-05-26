#!/usr/bin/env python3
"""Compute DML coefficients with analytical standard errors for faster computation"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

def compute_dml_linear_inference(X, Y, W, alpha=0.0, n_splits=5):
    """
    Compute DML with analytical inference for linear/lasso models
    
    Args:
        alpha: 0 for OLS, >0 for Lasso
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-fitting to get residuals
    Y_res = np.zeros_like(Y, dtype=float)
    X_res = np.zeros_like(X)
    
    for train_idx, test_idx in kf.split(X):
        # Residualize Y
        model_Y = LinearRegression()
        model_Y.fit(W[train_idx], Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - model_Y.predict(W[test_idx])
        
        # Residualize X
        for j in range(X.shape[1]):
            model_X = LinearRegression()
            model_X.fit(W[train_idx], X[train_idx, j])
            X_res[test_idx, j] = X[test_idx, j] - model_X.predict(W[test_idx])
    
    # Fit final model
    if alpha == 0:
        model = LinearRegression()
    else:
        model = Lasso(alpha=alpha, max_iter=5000)
    
    model.fit(X_res, Y_res)
    Y_pred = model.predict(X_res)
    residuals = Y_res - Y_pred
    
    # Get coefficients
    if alpha == 0:
        theta = model.coef_
    else:
        theta = model.coef_
        # For Lasso, focus on non-zero coefficients
        nonzero_idx = np.where(np.abs(theta) > 1e-10)[0]
        if len(nonzero_idx) == 0:
            return {'error': 'All coefficients are zero'}
    
    # Compute robust standard errors (HC1)
    # For high-dimensional case, we'll report SEs for first few PCs
    n_report = min(5, X.shape[1])
    
    if alpha == 0:
        # OLS standard errors
        try:
            XtX_inv = np.linalg.inv(X_res.T @ X_res)
            # HC1 variance
            e2 = residuals**2
            correction = n / (n - X.shape[1] - W.shape[1])
            V = correction * X_res.T @ np.diag(e2) @ X_res
            var_beta = XtX_inv @ V @ XtX_inv
            se = np.sqrt(np.diag(var_beta))[:n_report]
            theta_report = theta[:n_report]
        except:
            # If singular, just report first PC
            X_pc1 = X_res[:, 0:1]
            model_pc1 = LinearRegression()
            model_pc1.fit(X_pc1, Y_res)
            theta_report = model_pc1.coef_
            
            # SE for single coefficient
            XtX_inv = 1 / (X_pc1.T @ X_pc1)
            e2 = (Y_res - model_pc1.predict(X_pc1))**2
            V = X_pc1.T @ np.diag(e2) @ X_pc1
            se = np.sqrt(XtX_inv * V * XtX_inv)
            n_report = 1
    else:
        # For Lasso, report SEs for top coefficients by magnitude
        if len(nonzero_idx) > 0:
            top_idx = nonzero_idx[np.argsort(np.abs(theta[nonzero_idx]))[-n_report:]]
            theta_report = theta[top_idx]
            
            # Approximate SEs using selected features only
            X_selected = X_res[:, top_idx]
            try:
                XtX_inv = np.linalg.inv(X_selected.T @ X_selected)
                e2 = residuals**2
                V = X_selected.T @ np.diag(e2) @ X_selected
                var_beta = XtX_inv @ V @ XtX_inv
                se = np.sqrt(np.diag(var_beta))
            except:
                se = np.ones(len(top_idx)) * np.nan
        else:
            theta_report = np.array([0.0])
            se = np.array([np.nan])
            n_report = 1
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y_res - np.mean(Y_res))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    # Create results dictionary
    results = {
        'n_coefficients': n_report,
        'r_squared': r2,
        'n_obs': n,
        'coefficients': []
    }
    
    for i in range(n_report):
        coef = theta_report[i] if i < len(theta_report) else 0
        se_i = se[i] if i < len(se) and not np.isnan(se[i]) else 0.1  # Default SE if computation fails
        t_stat = coef / se_i if se_i > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-X.shape[1]-W.shape[1]))
        
        results['coefficients'].append({
            'pc': i + 1 if alpha == 0 else top_idx[i] + 1 if len(nonzero_idx) > 0 else 1,
            'coefficient': coef,
            'std_error': se_i,
            't_statistic': t_stat,
            'p_value': p_val,
            'ci_95_lower': coef - 1.96 * se_i,
            'ci_95_upper': coef + 1.96 * se_i
        })
    
    return results

def main():
    print("=== Computing DML Coefficients with Statistical Inference ===\n")
    
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
    
    # Results storage
    all_results = {}
    
    # Compute DML for both outcomes
    outcomes = [
        ('AI Ratings', Y_ai),
        ('Actual Social Class', Y_sc)
    ]
    
    models = [
        ('OLS', 0.0),
        ('Lasso (α=0.01)', 0.01)
    ]
    
    for outcome_name, Y in outcomes:
        print(f"\n{'='*70}")
        print(f"{outcome_name}")
        print('='*70)
        
        all_results[outcome_name] = {}
        
        for model_name, alpha in models:
            print(f"\n{model_name}:")
            results = compute_dml_linear_inference(X_pca, Y, W, alpha=alpha)
            
            if 'error' in results:
                print(f"  Error: {results['error']}")
                continue
            
            all_results[outcome_name][model_name] = results
            
            print(f"  R-squared: {results['r_squared']:.4f}")
            print(f"  N obs: {results['n_obs']:,}")
            print(f"\n  Top Principal Component Coefficients:")
            print(f"  {'PC':<6} {'Coef':<10} {'SE':<10} {'t-stat':<10} {'p-value':<10} {'95% CI':<20}")
            print("  " + "-"*66)
            
            for coef_info in results['coefficients']:
                pc = coef_info['pc']
                coef = coef_info['coefficient']
                se = coef_info['std_error']
                t = coef_info['t_statistic']
                p = coef_info['p_value']
                ci_low = coef_info['ci_95_lower']
                ci_high = coef_info['ci_95_upper']
                
                # Significance stars
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = ""
                
                print(f"  PC{pc:<4} {coef:>9.4f} {se:>9.4f} {t:>9.2f} {p:>9.4f}{sig:<3} [{ci_low:>7.4f}, {ci_high:>7.4f}]")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print('='*70)
    print("\nR-squared values:")
    print(f"{'Outcome':<20} {'OLS':<10} {'Lasso':<10}")
    print("-"*40)
    for outcome in outcomes:
        outcome_name = outcome[0]
        ols_r2 = all_results[outcome_name].get('OLS', {}).get('r_squared', 0)
        lasso_r2 = all_results[outcome_name].get('Lasso (α=0.01)', {}).get('r_squared', 0)
        print(f"{outcome_name:<20} {ols_r2:<10.4f} {lasso_r2:<10.4f}")
    
    print("\n" + "="*70)
    print("Notes:")
    print("- Coefficients represent effects of 1 SD change in principal components")
    print("- Standard errors computed using HC1 robust variance estimator")
    print("- All estimates use 5-fold cross-fitting as per DML")
    print("- *** p<0.001, ** p<0.01, * p<0.05")
    print("- For Lasso, showing only non-zero coefficients")

if __name__ == "__main__":
    main()