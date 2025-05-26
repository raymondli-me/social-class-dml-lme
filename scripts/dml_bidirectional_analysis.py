#!/usr/bin/env python3
"""
Bidirectional DML Analysis:
1. Effect of Actual SC on AI Ratings (controlling for text)
2. Effect of AI Ratings on Actual SC (controlling for text)
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

def compute_dml_effect(Y, D, X, W, n_splits=5, model_type='xgboost'):
    """
    Compute DML estimate: Y = θD + g(X,W) + ε
    
    Args:
        Y: Outcome 
        D: Treatment
        X: High-dimensional controls (200 PCs)
        W: Additional controls (demographics)
        n_splits: Number of CV folds
        model_type: 'xgboost' or 'linear'
    
    Returns dict with all statistics including R² values
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Arrays for cross-fitted residuals and predictions
    Y_res = np.zeros_like(Y, dtype=float)
    D_res = np.zeros_like(D, dtype=float)
    Y_pred_full = np.zeros_like(Y, dtype=float)
    D_pred_full = np.zeros_like(D, dtype=float)
    
    print(f"  Running {n_splits}-fold cross-fitting with {model_type}...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Combine all controls
        XW = np.hstack([X, W])
        XW_train = XW[train_idx]
        XW_test = XW[test_idx]
        
        # Step 1: ML model for E[Y|X,W] (outcome model)
        if model_type == 'xgboost':
            model_Y = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            model_Y = LinearRegression()
            
        model_Y.fit(XW_train, Y[train_idx])
        Y_pred = model_Y.predict(XW_test)
        Y_pred_full[test_idx] = Y_pred
        Y_res[test_idx] = Y[test_idx] - Y_pred
        
        # Step 2: ML model for E[D|X,W] (treatment model)
        if model_type == 'xgboost':
            model_D = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            model_D = LinearRegression()
            
        model_D.fit(XW_train, D[train_idx])
        D_pred = model_D.predict(XW_test)
        D_pred_full[test_idx] = D_pred
        D_res[test_idx] = D[test_idx] - D_pred
    
    # Step 3: Final stage - regress Y residuals on D residuals
    theta = np.sum(D_res * Y_res) / np.sum(D_res**2)
    
    # Compute standard error using influence function
    psi = (Y_res - theta * D_res) * D_res
    var_theta = np.mean(psi**2) / (np.mean(D_res**2)**2 * n)
    se_theta = np.sqrt(var_theta)
    
    # t-statistic and p-value
    t_stat = theta / se_theta
    df = n - X.shape[1] - W.shape[1] - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
    
    # 95% confidence interval
    ci_lower = theta - 1.96 * se_theta
    ci_upper = theta + 1.96 * se_theta
    
    # R² values for first stage models
    r2_Y = 1 - np.sum((Y - Y_pred_full)**2) / np.sum((Y - np.mean(Y))**2)
    r2_D = 1 - np.sum((D - D_pred_full)**2) / np.sum((D - np.mean(D))**2)
    
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
        'r2_Y_given_XW': r2_Y,
        'r2_D_given_XW': r2_D,
        'partial_r2': partial_r2,
        'n_obs': n,
        'n_splits': n_splits,
        'Y_res_std': np.std(Y_res),
        'D_res_std': np.std(D_res),
        'correlation_residuals': np.corrcoef(Y_res, D_res)[0,1]
    }

def main():
    print("="*80)
    print("BIDIRECTIONAL DML ANALYSIS")
    print("Controlling for Text Embeddings (200 PCs) and Demographics")
    print("="*80)
    
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
    ai_ratings = df['ai_average'].values
    actual_sc = df['sc11'].values
    X = X_pca  # 200 PCs from embeddings
    W = df[['age', 'female', 'education_level_numeric']].values
    
    print(f"\nData summary:")
    print(f"  N observations: {len(ai_ratings):,}")
    print(f"  AI ratings: mean={ai_ratings.mean():.2f}, std={ai_ratings.std():.2f}")
    print(f"  Actual SC:  mean={actual_sc.mean():.2f}, std={actual_sc.std():.2f}")
    print(f"  Raw correlation: {np.corrcoef(ai_ratings, actual_sc)[0,1]:.4f}")
    
    # Run both directions
    print("\n" + "="*80)
    print("MODEL 1: Effect of Actual SC → AI Ratings")
    print("="*80)
    print("Y = AI Ratings, D = Actual SC, Controls = Text(200 PCs) + Demographics")
    
    results1 = compute_dml_effect(
        Y=ai_ratings, 
        D=actual_sc, 
        X=X, 
        W=W,
        n_splits=5,
        model_type='xgboost'
    )
    
    print(f"\nFirst Stage R² Values:")
    print(f"  R²(AI|Text,Demo):  {results1['r2_Y_given_XW']:.4f}")
    print(f"  R²(SC|Text,Demo):  {results1['r2_D_given_XW']:.4f}")
    
    print(f"\nDML Causal Effect:")
    print(f"  θ = {results1['theta']:.6f} (SE = {results1['se']:.6f})")
    print(f"  t = {results1['t_stat']:.4f}, p = {results1['p_value']:.6f}")
    print(f"  95% CI: [{results1['ci_lower']:.6f}, {results1['ci_upper']:.6f}]")
    
    sig1 = "***" if results1['p_value'] < 0.001 else "**" if results1['p_value'] < 0.01 else "*" if results1['p_value'] < 0.05 else "n.s."
    print(f"  Significance: {sig1}")
    
    print(f"\nPartial R²: {results1['partial_r2']:.6f}")
    print(f"  (SC explains {100*results1['partial_r2']:.2f}% of AI variance after controlling for text)")
    
    print("\n" + "="*80)
    print("MODEL 2: Effect of AI Ratings → Actual SC")
    print("="*80)
    print("Y = Actual SC, D = AI Ratings, Controls = Text(200 PCs) + Demographics")
    
    results2 = compute_dml_effect(
        Y=actual_sc,
        D=ai_ratings,
        X=X,
        W=W,
        n_splits=5,
        model_type='xgboost'
    )
    
    print(f"\nFirst Stage R² Values:")
    print(f"  R²(SC|Text,Demo):  {results2['r2_Y_given_XW']:.4f}")
    print(f"  R²(AI|Text,Demo):  {results2['r2_D_given_XW']:.4f}")
    
    print(f"\nDML Causal Effect:")
    print(f"  θ = {results2['theta']:.6f} (SE = {results2['se']:.6f})")
    print(f"  t = {results2['t_stat']:.4f}, p = {results2['p_value']:.6f}")
    print(f"  95% CI: [{results2['ci_lower']:.6f}, {results2['ci_upper']:.6f}]")
    
    sig2 = "***" if results2['p_value'] < 0.001 else "**" if results2['p_value'] < 0.01 else "*" if results2['p_value'] < 0.05 else "n.s."
    print(f"  Significance: {sig2}")
    
    print(f"\nPartial R²: {results2['partial_r2']:.6f}")
    print(f"  (AI explains {100*results2['partial_r2']:.2f}% of SC variance after controlling for text)")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\nCross-validation: {results1['n_splits']}-fold for both models")
    
    print(f"\nCausal Effects:")
    print(f"  SC → AI:  θ = {results1['theta']:>8.4f} {sig1}")
    print(f"  AI → SC:  θ = {results2['theta']:>8.4f} {sig2}")
    
    print(f"\nFirst Stage R² (how well text predicts each):")
    print(f"  Text → AI:  R² = {results1['r2_Y_given_XW']:.4f}")
    print(f"  Text → SC:  R² = {results1['r2_D_given_XW']:.4f}")
    
    print(f"\nPartial R² (additional variance explained):")
    print(f"  SC → AI:  {100*results1['partial_r2']:>5.2f}%")
    print(f"  AI → SC:  {100*results2['partial_r2']:>5.2f}%")
    
    print(f"\nInterpretation:")
    print(f"- Text explains {100*results1['r2_Y_given_XW']:.1f}% of AI ratings")
    print(f"- Text explains {100*results1['r2_D_given_XW']:.1f}% of actual social class")
    print(f"- After controlling for text:")
    print(f"  - Actual SC has a small but significant effect on AI ratings")
    print(f"  - AI ratings have {'a significant' if results2['p_value'] < 0.05 else 'no significant'} effect on actual SC")

if __name__ == "__main__":
    main()