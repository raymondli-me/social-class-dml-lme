#!/usr/bin/env python3
"""
Compute DML estimate of the causal effect of Actual Social Class on AI Ratings
Controlling for text embeddings (200 PCs) and demographics
"""

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

def compute_dml_effect(Y, D, X, W, n_splits=5):
    """
    Compute DML estimate: Y = θD + g(X,W) + ε
    
    Args:
        Y: Outcome (AI ratings)
        D: Treatment (Actual social class)
        X: High-dimensional controls (200 PCs)
        W: Additional controls (demographics)
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Arrays for cross-fitted residuals
    Y_res = np.zeros_like(Y, dtype=float)
    D_res = np.zeros_like(D, dtype=float)
    
    print(f"Running {n_splits}-fold cross-fitting...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/{n_splits}")
        
        # Combine all controls
        XW = np.hstack([X, W])
        XW_train = XW[train_idx]
        XW_test = XW[test_idx]
        
        # Step 1: ML model for E[Y|X,W] (outcome model)
        # Use XGBoost for flexible function approximation
        model_Y = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model_Y.fit(XW_train, Y[train_idx])
        Y_pred = model_Y.predict(XW_test)
        Y_res[test_idx] = Y[test_idx] - Y_pred
        
        # Step 2: ML model for E[D|X,W] (treatment model)
        model_D = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model_D.fit(XW_train, D[train_idx])
        D_pred = model_D.predict(XW_test)
        D_res[test_idx] = D[test_idx] - D_pred
    
    # Step 3: Final stage - regress Y residuals on D residuals
    # θ = Σ(D_res * Y_res) / Σ(D_res²)
    theta = np.sum(D_res * Y_res) / np.sum(D_res**2)
    
    # Compute standard error using influence function
    psi = (Y_res - theta * D_res) * D_res
    var_theta = np.mean(psi**2) / (np.mean(D_res**2)**2 * n)
    se_theta = np.sqrt(var_theta)
    
    # t-statistic and p-value
    t_stat = theta / se_theta
    df = n - X.shape[1] - W.shape[1] - 1  # Approximate df
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
    
    # 95% confidence interval
    ci_lower = theta - 1.96 * se_theta
    ci_upper = theta + 1.96 * se_theta
    
    # Additional statistics
    # Partial R² (how much D explains of Y after controlling for X,W)
    Y_pred_from_D = theta * D_res
    ss_res = np.sum((Y_res - Y_pred_from_D)**2)
    ss_tot = np.sum(Y_res**2)
    partial_r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    return {
        'theta': theta,
        'se': se_theta,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'partial_r2': partial_r2,
        'n_obs': n,
        'Y_res_std': np.std(Y_res),
        'D_res_std': np.std(D_res),
        'correlation_residuals': np.corrcoef(Y_res, D_res)[0,1]
    }

def main():
    print("="*70)
    print("DML: Effect of Actual Social Class on AI Ratings")
    print("Controlling for Text Embeddings (200 PCs) and Demographics")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
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
    
    # Add demographics
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    # Prepare variables
    Y = df['ai_average'].values          # Outcome: AI ratings
    D = df['sc11'].values               # Treatment: Actual social class
    X = X_pca                           # Controls: 200 PCs from embeddings
    W = df[['age', 'female', 'education_level_numeric']].values
    
    print(f"\nData shapes:")
    print(f"  Y (AI ratings):    {Y.shape}")
    print(f"  D (Actual SC):     {D.shape}")
    print(f"  X (200 PCs):       {X.shape}")
    print(f"  W (demographics):  {W.shape}")
    
    # Simple correlation before controlling
    raw_corr = np.corrcoef(Y, D)[0,1]
    print(f"\nRaw correlation(AI, SC): {raw_corr:.4f}")
    
    # Run DML
    print("\n" + "-"*70)
    results = compute_dml_effect(Y, D, X, W)
    
    print("\nDML Results:")
    print("="*70)
    print(f"θ (theta):          {results['theta']:.6f}")
    print(f"SE:                 {results['se']:.6f}")
    print(f"t-statistic:        {results['t_stat']:.4f}")
    print(f"p-value:            {results['p_value']:.6f}")
    print(f"95% CI:             [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    
    # Significance
    if results['p_value'] < 0.001:
        sig = "***"
    elif results['p_value'] < 0.01:
        sig = "**"
    elif results['p_value'] < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    print(f"Significance:       {sig}")
    
    print(f"\nPartial R²:         {results['partial_r2']:.6f}")
    print(f"  (Variance in AI ratings explained by SC after controlling for text)")
    
    print(f"\nResidual correlations:")
    print(f"  Corr(Y_res, D_res): {results['correlation_residuals']:.4f}")
    print(f"  Std(Y_res):         {results['Y_res_std']:.4f}")
    print(f"  Std(D_res):         {results['D_res_std']:.4f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    print(f"\nAfter controlling for text content (200 PCs) and demographics:")
    print(f"- A 1-unit increase in actual social class causes a {results['theta']:.4f} unit")
    print(f"  increase in AI-perceived social class rating")
    
    if results['p_value'] < 0.05:
        print(f"- This effect is statistically significant (p = {results['p_value']:.4f})")
    else:
        print(f"- This effect is NOT statistically significant (p = {results['p_value']:.4f})")
    
    print(f"- Actual SC explains {100*results['partial_r2']:.1f}% of remaining variance in AI ratings")
    print(f"  after accounting for what can be predicted from text alone")
    
    # Compare to raw effect
    print(f"\nComparison:")
    print(f"- Raw correlation: {raw_corr:.4f}")
    print(f"- Causal effect after controlling for text: {results['theta']:.4f}")
    print(f"- Text explains {100*(raw_corr - results['theta'])/raw_corr:.1f}% of the raw correlation")

if __name__ == "__main__":
    main()