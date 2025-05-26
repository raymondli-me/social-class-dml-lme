#!/usr/bin/env python3
"""Compute the relationship between actual social class and AI ratings using DML"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

def main():
    print("=== Analyzing Relationship Between Actual Social Class and AI Ratings ===\n")
    
    # Load the data
    df = pd.read_csv(BASE_DIR / "data" / "asc_9513_essays.csv")
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    df = df.merge(sc_labels, on='TID', how='left')
    
    # Load AI ratings
    ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['TID', 'ai_average']
    df = df.merge(ai_avg, on='TID', how='left')
    
    Y_ai = df['ai_average'].values
    Y_sc = df['sc11'].values
    
    # Load the DML results
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        dml_results = pickle.load(f)
    
    # Get the XGBoost predictions from embeddings
    model_ai = dml_results['AI_ratings']['XGBoost']['model']
    model_sc = dml_results['actual_SC']['XGBoost']['model']
    X_res = dml_results['AI_ratings']['XGBoost']['X_res']  # Same for both
    
    # Get predictions from embeddings
    Y_ai_pred_from_X = model_ai.predict(X_res)
    Y_sc_pred_from_X = model_sc.predict(X_res)
    
    # Get residuals (already computed in DML)
    Y_ai_res = dml_results['AI_ratings']['XGBoost']['Y_res']
    Y_sc_res = dml_results['actual_SC']['XGBoost']['Y_res']
    
    print("1. Raw Relationship (No Controls)")
    print("-" * 50)
    
    # Simple regression
    model_raw = LinearRegression()
    model_raw.fit(Y_sc.reshape(-1, 1), Y_ai)
    theta_raw = model_raw.coef_[0]
    Y_ai_pred_raw = model_raw.predict(Y_sc.reshape(-1, 1))
    
    # R² and correlation
    r2_raw = 1 - np.sum((Y_ai - Y_ai_pred_raw)**2) / np.sum((Y_ai - np.mean(Y_ai))**2)
    corr_raw = np.corrcoef(Y_ai, Y_sc)[0,1]
    
    print(f"  θ (slope):        {theta_raw:.4f}")
    print(f"  R²:               {r2_raw:.4f}")
    print(f"  Correlation:      {corr_raw:.4f}")
    print(f"  Interpretation:   1 unit ↑ in SC → {theta_raw:.3f} unit ↑ in AI rating")
    
    print("\n2. Controlling for Text (What's Left After Embeddings)")
    print("-" * 50)
    
    # This uses the residuals after removing what embeddings can predict
    if np.std(Y_sc_res) > 1e-6:  # Check if there's variation
        model_resid = LinearRegression()
        model_resid.fit(Y_sc_res.reshape(-1, 1), Y_ai_res)
        theta_resid = model_resid.coef_[0]
        
        # Standard error
        n = len(Y_ai_res)
        Y_pred_resid = model_resid.predict(Y_sc_res.reshape(-1, 1))
        residuals = Y_ai_res - Y_pred_resid
        sigma2 = np.sum(residuals**2) / (n - 2)
        X_var = np.sum((Y_sc_res - np.mean(Y_sc_res))**2)
        se_resid = np.sqrt(sigma2 / X_var) if X_var > 0 else np.inf
        
        t_stat = theta_resid / se_resid
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        
        r2_resid = 1 - np.sum(residuals**2) / np.sum((Y_ai_res - np.mean(Y_ai_res))**2)
        corr_resid = np.corrcoef(Y_ai_res, Y_sc_res)[0,1]
        
        print(f"  θ_residual:       {theta_resid:.6f}")
        print(f"  SE:               {se_resid:.6f}")
        print(f"  t-stat:           {t_stat:.4f}")
        print(f"  p-value:          {p_value:.6f}")
        print(f"  R²:               {r2_resid:.6f}")
        print(f"  Correlation:      {corr_resid:.6f}")
    else:
        print("  WARNING: No variation in social class after controlling for embeddings!")
        print(f"  Std(Y_sc_res) = {np.std(Y_sc_res):.2e}")
    
    print("\n3. Decomposition: How Much Do Embeddings Explain?")
    print("-" * 50)
    
    # Variance decomposition
    var_ai_total = np.var(Y_ai)
    var_ai_explained_by_X = np.var(Y_ai_pred_from_X)
    var_ai_residual = np.var(Y_ai_res)
    
    var_sc_total = np.var(Y_sc)
    var_sc_explained_by_X = np.var(Y_sc_pred_from_X)
    var_sc_residual = np.var(Y_sc_res)
    
    print(f"  AI Ratings:")
    print(f"    Total variance:        {var_ai_total:.4f}")
    print(f"    Explained by text:     {var_ai_explained_by_X:.4f} ({100*var_ai_explained_by_X/var_ai_total:.1f}%)")
    print(f"    Residual variance:     {var_ai_residual:.4f} ({100*var_ai_residual/var_ai_total:.1f}%)")
    
    print(f"\n  Actual Social Class:")
    print(f"    Total variance:        {var_sc_total:.4f}")
    print(f"    Explained by text:     {var_sc_explained_by_X:.4f} ({100*var_sc_explained_by_X/var_sc_total:.1f}%)")
    print(f"    Residual variance:     {var_sc_residual:.4f} ({100*var_sc_residual/var_sc_total:.1f}%)")
    
    print("\n4. Mediation Analysis")
    print("-" * 50)
    
    # How much of the SC→AI relationship is mediated by text?
    # Total effect = Direct effect + Indirect effect (through text)
    
    # Predict AI from SC and embeddings jointly
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Full model: AI ~ SC + Embeddings
    X_full = np.column_stack([Y_sc, X_scaled[:, :10]])  # Use first 10 PCs for tractability
    model_full = LinearRegression()
    model_full.fit(X_full, Y_ai)
    
    theta_direct = model_full.coef_[0]  # Direct effect controlling for embeddings
    
    print(f"  Total effect (no controls):    {theta_raw:.4f}")
    print(f"  Direct effect (control embed):  {theta_direct:.4f}")
    print(f"  Indirect effect (via text):     {theta_raw - theta_direct:.4f}")
    print(f"  % Mediated by text:             {100*(theta_raw - theta_direct)/theta_raw:.1f}%")
    
    # Statistical test for direct effect
    n = len(Y_ai)
    Y_pred_full = model_full.predict(X_full)
    residuals_full = Y_ai - Y_pred_full
    
    # Compute standard error for SC coefficient
    XtX_inv = np.linalg.inv(X_full.T @ X_full)
    sigma2_full = np.sum(residuals_full**2) / (n - X_full.shape[1])
    se_direct = np.sqrt(sigma2_full * XtX_inv[0, 0])
    t_direct = theta_direct / se_direct
    p_direct = 2 * (1 - stats.t.cdf(abs(t_direct), df=n-X_full.shape[1]))
    
    print(f"\n  Direct effect inference:")
    print(f"    SE:      {se_direct:.4f}")
    print(f"    t-stat:  {t_direct:.4f}")
    print(f"    p-value: {p_direct:.6f}")
    
    sig = "***" if p_direct < 0.001 else "**" if p_direct < 0.01 else "*" if p_direct < 0.05 else "n.s."
    print(f"    Significance: {sig}")
    
    print("\n" + "="*70)
    print("Summary:")
    print(f"- Raw correlation between actual SC and AI ratings: {corr_raw:.3f}")
    print(f"- Text embeddings explain {100*var_ai_explained_by_X/var_ai_total:.1f}% of AI rating variance")
    print(f"- Text embeddings explain {100*var_sc_explained_by_X/var_sc_total:.1f}% of actual SC variance")
    print(f"- About {100*(theta_raw - theta_direct)/theta_raw:.1f}% of the SC→AI relationship is mediated by text")

if __name__ == "__main__":
    main()