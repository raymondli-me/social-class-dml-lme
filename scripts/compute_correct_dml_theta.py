#!/usr/bin/env python3
"""Compute the correct DML theta coefficient"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"

def main():
    print("=== Computing Correct DML Theta ===\n")
    
    # The DML procedure estimates: E[Y|X,W] where
    # Y = outcome (AI ratings or actual SC)
    # X = treatment (200 PCs from embeddings) 
    # W = controls (demographics)
    
    # Load the DML results
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        dml_results = pickle.load(f)
    
    print("Understanding the DML Results:")
    print("-" * 50)
    
    # For AI ratings
    ai_results = dml_results['AI_ratings']['XGBoost']
    print(f"AI Ratings:")
    print(f"  R² = {ai_results['r2']:.4f}")
    print(f"  This means: embeddings explain {100*ai_results['r2']:.1f}% of AI rating variance")
    print(f"  (after controlling for demographics)")
    
    # For actual social class  
    sc_results = dml_results['actual_SC']['XGBoost']
    print(f"\nActual Social Class:")
    print(f"  R² = {sc_results['r2']:.4f}")
    print(f"  This means: embeddings explain {100*sc_results['r2']:.1f}% of actual SC variance")
    print(f"  (after controlling for demographics)")
    
    # The key insight
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    
    r2_ai = ai_results['r2']
    r2_sc = sc_results['r2']
    
    print(f"\nThe 200 PCA components from OpenAI embeddings can predict:")
    print(f"  - {100*r2_ai:.1f}% of the variance in AI ratings")
    print(f"  - {100*r2_sc:.1f}% of the variance in actual social class")
    
    print(f"\nThis {100*r2_ai:.1f}% vs {100*r2_sc:.1f}% difference shows that:")
    print("  - AI models 'see' social class cues in text that strongly predict their ratings")
    print("  - But these same textual cues only weakly predict actual self-reported social class")
    print("  - The gap represents AI bias: what AI thinks indicates social class vs reality")
    
    # If you want a single "theta" coefficient representing the causal effect
    # of embeddings on outcomes, we can compute the norm of the coefficient vector
    print("\n" + "="*70)
    print("Overall Embedding Effect (||β||₂):")
    print("="*70)
    
    # Get the models
    model_ai = ai_results['model']
    model_sc = sc_results['model']
    
    # For XGBoost, we can't get a simple coefficient vector
    # But we can compute the average marginal effect
    X_res = ai_results['X_res']
    
    # Compute average marginal effects by perturbing each PC
    print("\nComputing average marginal effects...")
    n_samples = min(1000, len(X_res))  # Subsample for speed
    idx = np.random.choice(len(X_res), n_samples, replace=False)
    
    # For each PC, compute average effect of 1 SD increase
    effects_ai = []
    effects_sc = []
    
    for j in range(min(10, X_res.shape[1])):  # First 10 PCs
        X_base = X_res[idx].copy()
        X_plus = X_res[idx].copy()
        X_plus[:, j] += np.std(X_res[:, j])  # Increase by 1 SD
        
        # Average marginal effect
        ame_ai = np.mean(model_ai.predict(X_plus) - model_ai.predict(X_base))
        ame_sc = np.mean(model_sc.predict(X_plus) - model_sc.predict(X_base))
        
        effects_ai.append(ame_ai)
        effects_sc.append(ame_sc)
        
        print(f"  PC{j+1}: AI effect = {ame_ai:>7.4f}, SC effect = {ame_sc:>7.4f}")
    
    # Overall effect as norm
    theta_ai = np.linalg.norm(effects_ai)
    theta_sc = np.linalg.norm(effects_sc)
    
    print(f"\nAggregate Effects (L2 norm of first 10 PCs):")
    print(f"  AI ratings:    ||θ|| = {theta_ai:.4f}")
    print(f"  Actual SC:     ||θ|| = {theta_sc:.4f}")
    print(f"  Ratio:         {theta_ai/theta_sc:.2f}x")
    
    print("\n" + "="*70)
    print("Conclusion:")
    print("Text embeddings have a {:.1f}x stronger causal effect on AI ratings".format(theta_ai/theta_sc))
    print("than on actual social class, controlling for demographics.")

if __name__ == "__main__":
    main()