#!/usr/bin/env python3
"""
Pre-compute cross-fitted metrics for v13 visualization
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from scipy import stats

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'

print("=== Computing Cross-fitted Metrics ===")

# Load essays and social class
print("Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Load DML analysis results
with open(OUTPUT_DIR / 'dml_pc_analysis_results_with_umap.pkl', 'rb') as f:
    dml_results = pickle.load(f)
    data_with_pcs = dml_results['data_with_pcs']

# Load full PCA data
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']

# Ensure alignment
essays_df = essays_df[essays_df['essay_id'].isin(pca_data['essay_ids'])].copy()
essays_df = essays_df.set_index('essay_id').loc[pca_data['essay_ids']].reset_index()

# Drop rows with NaN values
mask = ~(essays_df['sc11'].isna() | essays_df['ai_rating'].isna())
essays_df = essays_df[mask].reset_index(drop=True)
X_pca = X_pca[mask]

# Clean data
Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Function to compute cross-fitted metrics
def compute_crossfitted_metrics(X, Y_treatment, Y_outcome, n_folds=5):
    """Compute cross-fitted R² and DML estimates with proper k-fold cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    r2_scores_cf = []
    residuals_treatment_cf = np.zeros_like(Y_treatment)
    residuals_outcome_cf = np.zeros_like(Y_outcome)
    
    print(f"    Running {n_folds}-fold cross-validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"      Fold {fold_idx + 1}/{n_folds}...")
        X_train, X_test = X[train_idx], X[test_idx]
        Y_treatment_train, Y_treatment_test = Y_treatment[train_idx], Y_treatment[test_idx]
        Y_outcome_train, Y_outcome_test = Y_outcome[train_idx], Y_outcome[test_idx]
        
        # Train models on fold
        model_outcome = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model_outcome.fit(X_train, Y_outcome_train)
        
        model_treatment = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model_treatment.fit(X_train, Y_treatment_train)
        
        # Predict on test fold (cross-fitted)
        pred_outcome = model_outcome.predict(X_test)
        pred_treatment = model_treatment.predict(X_test)
        
        # Calculate residuals for DML
        residuals_treatment_cf[test_idx] = Y_treatment_test - pred_treatment
        residuals_outcome_cf[test_idx] = Y_outcome_test - pred_outcome
        
        # R² on test fold
        r2_scores_cf.append(r2_score(Y_outcome_test, pred_outcome))
    
    # Train on full data for non-cross-fitted R²
    print("    Training on full data...")
    model_full = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    model_full.fit(X, Y_outcome)
    predictions_full = model_full.predict(X)
    r2_full = r2_score(Y_outcome, predictions_full)
    
    # Cross-fitted R²
    r2_cf = np.mean(r2_scores_cf)
    
    # DML theta estimate (cross-fitted)
    theta_cf = np.sum(residuals_treatment_cf * residuals_outcome_cf) / np.sum(residuals_treatment_cf ** 2)
    
    # Standard error for theta
    n = len(Y_outcome)
    se_cf = np.sqrt(np.sum((residuals_outcome_cf - theta_cf * residuals_treatment_cf) ** 2) / 
                    (n * np.sum(residuals_treatment_cf ** 2)))
    
    # Confidence interval
    ci_cf = (theta_cf - 1.96 * se_cf, theta_cf + 1.96 * se_cf)
    
    # P-value (two-tailed t-test)
    t_stat = theta_cf / se_cf
    pval_cf = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1] - 1))
    
    return {
        'r2_full': r2_full,
        'r2_cf': r2_cf,
        'theta_cf': theta_cf,
        'se_cf': se_cf,
        'ci_cf': ci_cf,
        'pval_cf': pval_cf,
        'model': model_full
    }

# Compute metrics for all 200 PCs - bidirectional
print("\nComputing cross-fitted metrics for all 200 PCs...")
print("  SC → AI direction...")
metrics_sc_to_ai_200 = compute_crossfitted_metrics(X_pca, Y_sc, Y_ai)

print("  AI → SC direction...")
metrics_ai_to_sc_200 = compute_crossfitted_metrics(X_pca, Y_ai, Y_sc)

# Compute metrics for top 5 PCs - bidirectional
X_top5 = data_with_pcs[['PC0', 'PC2', 'PC5', 'PC13', 'PC46']].values
X_top5 = X_top5[mask]

print("\nComputing cross-fitted metrics for top 5 PCs...")
print("  SC → AI direction...")
metrics_sc_to_ai_top5 = compute_crossfitted_metrics(X_top5, Y_sc, Y_ai)

print("  AI → SC direction...")
metrics_ai_to_sc_top5 = compute_crossfitted_metrics(X_top5, Y_ai, Y_sc)

# Save all metrics
all_metrics = {
    'sc_to_ai_200': metrics_sc_to_ai_200,
    'ai_to_sc_200': metrics_ai_to_sc_200,
    'sc_to_ai_top5': metrics_sc_to_ai_top5,
    'ai_to_sc_top5': metrics_ai_to_sc_top5
}

output_path = OUTPUT_DIR / 'crossfitted_metrics_v13.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(all_metrics, f)

print(f"\nMetrics saved to: {output_path}")

# Print summary
print("\n=== RESULTS SUMMARY ===")
print("\nAll 200 PCs:")
print(f"  SC → AI: R²={metrics_sc_to_ai_200['r2_full']:.3f} (full), R²={metrics_sc_to_ai_200['r2_cf']:.3f} (CF), θ={metrics_sc_to_ai_200['theta_cf']:.3f}±{metrics_sc_to_ai_200['se_cf']:.3f}")
print(f"  AI → SC: R²={metrics_ai_to_sc_200['r2_full']:.3f} (full), R²={metrics_ai_to_sc_200['r2_cf']:.3f} (CF), θ={metrics_ai_to_sc_200['theta_cf']:.3f}±{metrics_ai_to_sc_200['se_cf']:.3f}")

print("\nTop 5 PCs:")
print(f"  SC → AI: R²={metrics_sc_to_ai_top5['r2_full']:.3f} (full), R²={metrics_sc_to_ai_top5['r2_cf']:.3f} (CF), θ={metrics_sc_to_ai_top5['theta_cf']:.3f}±{metrics_sc_to_ai_top5['se_cf']:.3f}")
print(f"  AI → SC: R²={metrics_ai_to_sc_top5['r2_full']:.3f} (full), R²={metrics_ai_to_sc_top5['r2_cf']:.3f} (CF), θ={metrics_ai_to_sc_top5['theta_cf']:.3f}±{metrics_ai_to_sc_top5['se_cf']:.3f}")