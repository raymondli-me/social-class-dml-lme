#!/usr/bin/env python3
"""Compute DML theta using pre-computed residuals from existing analysis"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

def main():
    print("=== Computing DML Theta: Effect of Actual Social Class on AI Ratings ===\n")
    
    # Load the pre-computed DML results
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        dml_results = pickle.load(f)
    
    # Get the residuals from XGBoost models
    # Y_res: AI ratings residuals after controlling for demographics
    # D_res: Actual social class residuals after controlling for demographics
    
    Y_res = dml_results['AI_ratings']['XGBoost']['Y_res']  # AI ratings residuals
    D_res = dml_results['actual_SC']['XGBoost']['Y_res']   # Actual SC residuals
    
    # The DML estimator: θ = Σ(D_res * Y_res) / Σ(D_res²)
    # This is the coefficient from regressing Y_res on D_res
    theta = np.sum(D_res * Y_res) / np.sum(D_res**2)
    
    # Compute standard error using influence function
    n = len(Y_res)
    psi = (Y_res - theta * D_res) * D_res  # Influence function
    var_theta = np.mean(psi**2) / (np.mean(D_res**2)**2) / n
    se_theta = np.sqrt(var_theta)
    
    # t-statistic and p-value
    t_stat = theta / se_theta
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # 95% confidence interval
    ci_lower = theta - 1.96 * se_theta
    ci_upper = theta + 1.96 * se_theta
    
    # Additional statistics
    # R² from regressing Y_res on D_res
    Y_pred = theta * D_res
    ss_res = np.sum((Y_res - Y_pred)**2)
    ss_tot = np.sum((Y_res - np.mean(Y_res))**2)
    r2 = 1 - ss_res/ss_tot
    
    # Correlation between residuals
    corr = np.corrcoef(Y_res, D_res)[0,1]
    
    print("Data Summary:")
    print(f"  N observations:           {n:,}")
    print(f"  Y_res (AI) std:          {np.std(Y_res):.4f}")
    print(f"  D_res (SC) std:          {np.std(D_res):.4f}")
    print(f"  Correlation(Y_res,D_res): {corr:.4f}")
    
    print("\n" + "="*70)
    print("DML Estimate: Effect of Actual Social Class on AI Ratings")
    print("="*70)
    
    print(f"\nθ (theta):        {theta:.6f}")
    print(f"SE:               {se_theta:.6f}")
    print(f"t-statistic:      {t_stat:.4f}")
    print(f"p-value:          {p_value:.6f}")
    print(f"95% CI:           [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Significance
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"  
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    print(f"Significance:     {sig}")
    
    print(f"\nR² (Y_res|D_res): {r2:.6f}")
    
    print("\nInterpretation:")
    print(f"After controlling for demographics and using text embeddings,")
    print(f"a 1-unit increase in actual social class causes a {theta:.3f} unit")
    print(f"increase in AI-perceived social class rating.")
    
    # Also check the reverse: Effect of AI ratings on actual social class
    print("\n" + "-"*70)
    print("Reverse Check: Effect of AI Ratings on Actual Social Class")
    print("-"*70)
    
    # Swap Y and D
    theta_reverse = np.sum(Y_res * D_res) / np.sum(Y_res**2)
    psi_reverse = (D_res - theta_reverse * Y_res) * Y_res
    var_theta_reverse = np.mean(psi_reverse**2) / (np.mean(Y_res**2)**2) / n
    se_theta_reverse = np.sqrt(var_theta_reverse)
    t_stat_reverse = theta_reverse / se_theta_reverse
    p_value_reverse = 2 * (1 - stats.t.cdf(abs(t_stat_reverse), df=n-1))
    
    print(f"\nθ_reverse:        {theta_reverse:.6f}")
    print(f"SE:               {se_theta_reverse:.6f}")
    print(f"t-statistic:      {t_stat_reverse:.4f}")
    print(f"p-value:          {p_value_reverse:.6f}")
    
    print("\n" + "="*70)
    print("Notes:")
    print("- Uses residuals from XGBoost models with 200 PCA components")
    print("- Both outcomes are residualized on demographics (age, gender, education)")
    print("- Standard errors computed using influence functions")
    print("- This is the 'partially linear' DML estimator")

if __name__ == "__main__":
    main()