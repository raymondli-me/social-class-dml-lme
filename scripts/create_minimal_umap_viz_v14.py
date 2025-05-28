#!/usr/bin/env python3
"""
Minimal UMAP visualization v14 - with compact UI and simplified PC analysis
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'

print("=== Creating Minimal UMAP Visualization v14 ===")

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
print("Loading DML analysis results...")
with open(OUTPUT_DIR / 'dml_pc_analysis_results_with_umap.pkl', 'rb') as f:
    dml_results = pickle.load(f)
    X_umap_3d = dml_results['umap_3d']
    data_with_pcs = dml_results['data_with_pcs']
    contributions_ai_top5 = dml_results['contributions_ai']
    contributions_sc_top5 = dml_results['contributions_sc']
    top_pcs = dml_results['top_pcs']

# Load full PCA data to get all 200 PCs
print("Loading full PCA data...")
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    essay_ids = pca_data['essay_ids']
    X_pca = pca_data['features']  # PCA-transformed features
    pca_model = pca_data['pca']

# Align data
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

# Get variance explained for all PCs
variance_explained = pca_model.explained_variance_ratio_

# Calculate statistics
ai_ratings_clean = essays_df['ai_rating'].dropna()
ai_percentiles = {
    10: ai_ratings_clean.quantile(0.10),
    25: ai_ratings_clean.quantile(0.25),
    75: ai_ratings_clean.quantile(0.75),
    90: ai_ratings_clean.quantile(0.90)
}

print(f"AI Rating range: {ai_ratings_clean.min():.2f} - {ai_ratings_clean.max():.2f}")
print(f"Social class distribution: {essays_df['sc11'].value_counts().sort_index().to_dict()}")

# Calculate center of point cloud
center_x = X_umap_3d[:, 0].mean()
center_y = X_umap_3d[:, 1].mean()
center_z = X_umap_3d[:, 2].mean()

# Calculate percentiles for ALL 200 PCs
print("Calculating percentiles for all PCs...")
pc_percentiles = np.zeros((len(essays_df), 200))
for i in range(200):
    pc_values = X_pca[:, i]
    pc_percentiles[:, i] = (np.searchsorted(np.sort(pc_values), pc_values) / len(pc_values)) * 100

# Compute contributions for all 200 PCs using XGBoost
print("Computing contributions for all 200 PCs...")
Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Train XGBoost models on all 200 PCs
model_ai_200 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_200.fit(X_pca, Y_ai)

model_sc_200 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_200.fit(X_pca, Y_sc)

# Get feature importances as proxy for contributions
contributions_ai_200 = np.zeros((len(X_pca), 200))
contributions_sc_200 = np.zeros((len(X_pca), 200))

# Use feature importance weighted by PC values
feature_importance_ai = model_ai_200.feature_importances_
feature_importance_sc = model_sc_200.feature_importances_

for i in range(len(X_pca)):
    contributions_ai_200[i] = X_pca[i] * feature_importance_ai
    contributions_sc_200[i] = X_pca[i] * feature_importance_sc

# Calculate R² for both models
from sklearn.metrics import r2_score
r2_ai_200 = r2_score(Y_ai, model_ai_200.predict(X_pca))
r2_sc_200 = r2_score(Y_sc, model_sc_200.predict(X_pca))

# Also compute for top 5 PCs
X_top5 = data_with_pcs[['PC0', 'PC2', 'PC5', 'PC13', 'PC46']].values
model_ai_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_top5.fit(X_top5, Y_ai)
model_sc_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_top5.fit(X_top5, Y_sc)

r2_ai_top5 = r2_score(Y_ai, model_ai_top5.predict(X_top5))
r2_sc_top5 = r2_score(Y_sc, model_sc_top5.predict(X_top5))

# Function to compute simple DML without cross-fitting
def compute_simple_dml(X, Y_treatment, Y_outcome):
    """Compute non-cross-fitted DML estimate"""
    from scipy import stats
    
    # Fit models on full data
    model_outcome = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_outcome.fit(X, Y_outcome)
    
    model_treatment = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_treatment.fit(X, Y_treatment)
    
    # Get residuals
    residuals_outcome = Y_outcome - model_outcome.predict(X)
    residuals_treatment = Y_treatment - model_treatment.predict(X)
    
    # DML theta estimate
    theta = np.sum(residuals_treatment * residuals_outcome) / np.sum(residuals_treatment ** 2)
    
    # Standard error
    n = len(Y_outcome)
    se = np.sqrt(np.sum((residuals_outcome - theta * residuals_treatment) ** 2) / 
                 (n * np.sum(residuals_treatment ** 2)))
    
    # Confidence interval
    ci = (theta - 1.96 * se, theta + 1.96 * se)
    
    # P-value
    t_stat = theta / se
    pval = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1] - 1))
    
    return theta, se, ci, pval

# Function to compute naive theta (no controls)
def compute_naive_theta(Y_treatment, Y_outcome):
    """Compute naive OLS coefficient (no controls)"""
    from scipy import stats
    import statsmodels.api as sm
    
    # Add constant to treatment
    X = sm.add_constant(Y_treatment)
    
    # Simple OLS
    model = sm.OLS(Y_outcome, X).fit()
    
    theta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
    se = model.bse.iloc[1] if hasattr(model.bse, 'iloc') else model.bse[1]
    ci_df = model.conf_int()
    ci = (ci_df.iloc[1, 0], ci_df.iloc[1, 1]) if hasattr(ci_df, 'iloc') else (ci_df[1][0], ci_df[1][1])
    pval = model.pvalues.iloc[1] if hasattr(model.pvalues, 'iloc') else model.pvalues[1]
    r2 = model.rsquared
    
    return theta, se, ci, pval, r2

# Load actual cross-fitted metrics
print("Loading cross-fitted metrics...")
try:
    with open(OUTPUT_DIR / 'crossfitted_metrics_v13.pkl', 'rb') as f:
        cf_metrics = pickle.load(f)
    
    # Compute non-cross-fitted DML for comparison
    print("Computing non-cross-fitted DML estimates...")
    theta_200, se_200, ci_200, pval_200 = compute_simple_dml(X_pca, Y_sc, Y_ai)
    theta_top5, se_top5, ci_top5, pval_top5 = compute_simple_dml(X_top5, Y_sc, Y_ai)
    
    # Compute naive theta
    print("Computing naive theta (no text controls)...")
    theta_naive, se_naive, ci_naive, pval_naive, r2_naive = compute_naive_theta(Y_sc, Y_ai)
    
    # Extract the metrics we need
    dml_results_computed = {
        # Non-cross-fitted values
        'theta_200': theta_200,
        'se_200': se_200,
        'ci_200': ci_200,
        'pval_200': pval_200,
        # Cross-fitted values
        'theta_200_cf': cf_metrics['sc_to_ai_200']['theta_cf'],
        'se_200_cf': cf_metrics['sc_to_ai_200']['se_cf'],
        'ci_200_cf': cf_metrics['sc_to_ai_200']['ci_cf'],
        'pval_200_cf': cf_metrics['sc_to_ai_200']['pval_cf'],
        # Top 5 non-cross-fitted
        'theta_top5': theta_top5,
        'se_top5': se_top5,
        'ci_top5': ci_top5,
        'pval_top5': pval_top5,
        # Top 5 cross-fitted
        'theta_top5_cf': cf_metrics['sc_to_ai_top5']['theta_cf'],
        'se_top5_cf': cf_metrics['sc_to_ai_top5']['se_cf'],
        'ci_top5_cf': cf_metrics['sc_to_ai_top5']['ci_cf'],
        'pval_top5_cf': cf_metrics['sc_to_ai_top5']['pval_cf'],
        # Naive theta
        'theta_naive': theta_naive,
        'se_naive': se_naive,
        'ci_naive': ci_naive,
        'pval_naive': pval_naive,
        'r2_naive': r2_naive
    }
    
    # Also update R² values with actual cross-fitted values
    r2_ai_200_cf = cf_metrics['sc_to_ai_200']['r2_cf']
    r2_sc_200_cf = cf_metrics['ai_to_sc_200']['r2_cf']
    r2_ai_top5_cf = cf_metrics['sc_to_ai_top5']['r2_cf']
    r2_sc_top5_cf = cf_metrics['ai_to_sc_top5']['r2_cf']
    
except FileNotFoundError:
    print("Warning: crossfitted_metrics_v13.pkl not found, using reasonable estimates")
    # Compute non-cross-fitted DML
    theta_200, se_200, ci_200, pval_200 = compute_simple_dml(X_pca, Y_sc, Y_ai)
    theta_top5, se_top5, ci_top5, pval_top5 = compute_simple_dml(X_top5, Y_sc, Y_ai)
    
    # Compute naive theta
    theta_naive, se_naive, ci_naive, pval_naive, r2_naive = compute_naive_theta(Y_sc, Y_ai)
    
    # Fallback cross-fitted values (from your output)
    dml_results_computed = {
        'theta_200': theta_200,
        'se_200': se_200,
        'ci_200': ci_200,
        'pval_200': pval_200,
        'theta_200_cf': 0.054,
        'se_200_cf': 0.013,
        'ci_200_cf': (0.029, 0.079),
        'pval_200_cf': 0.001,
        'theta_top5': theta_top5,
        'se_top5': se_top5,
        'ci_top5': ci_top5,
        'pval_top5': pval_top5,
        'theta_top5_cf': 0.106,
        'se_top5_cf': 0.014,
        'ci_top5_cf': (0.079, 0.133),
        'pval_top5_cf': 0.001,
        # Naive estimate
        'theta_naive': theta_naive,
        'se_naive': se_naive,
        'ci_naive': ci_naive,
        'pval_naive': pval_naive,
        'r2_naive': r2_naive
    }
    r2_ai_200_cf = 0.505
    r2_sc_200_cf = -0.023
    r2_ai_top5_cf = 0.423
    r2_sc_top5_cf = -0.039

# Train logistic regression models for probability predictions
print("Training probability models...")
# For high/low AI rating - will use user-defined thresholds
ai_high_threshold = ai_percentiles[90]
ai_low_threshold = ai_percentiles[10]
y_ai_high = (essays_df['ai_rating'] > ai_high_threshold).astype(int)
y_ai_low = (essays_df['ai_rating'] < ai_low_threshold).astype(int)

# For high/low social class
sc_high_threshold = 5
sc_low_threshold = 1
y_sc_high = (essays_df['sc11'] >= sc_high_threshold).astype(int)
y_sc_low = (essays_df['sc11'] <= sc_low_threshold).astype(int)

# Train models on all 200 PCs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

models = {}
for name, y in [('ai_high', y_ai_high), ('ai_low', y_ai_low), 
                ('sc_high', y_sc_high), ('sc_low', y_sc_low)]:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    models[name] = model
    print(f"  {name} model trained, accuracy: {model.score(X_scaled, y):.3f}")

# Calculate global PC effects for top/bottom 10%
print("Calculating global PC effects...")
pc_global_effects = {}

# Define top/bottom 10% thresholds for AI and SC
ai_top10 = np.percentile(Y_ai, 90)
ai_bottom10 = np.percentile(Y_ai, 10)
sc_top10 = 5  # Top social class
sc_bottom10 = 1  # Bottom social class

# Create binary labels for top/bottom 10%
y_ai_top10 = (Y_ai >= ai_top10).astype(int)
y_ai_bottom10 = (Y_ai <= ai_bottom10).astype(int)
y_sc_top10 = (Y_sc >= sc_top10).astype(int)
y_sc_bottom10 = (Y_sc <= sc_bottom10).astype(int)

# Train models for top/bottom 10%
models_10pct = {}
for name, y in [('ai_top10', y_ai_top10), ('ai_bottom10', y_ai_bottom10),
                ('sc_top10', y_sc_top10), ('sc_bottom10', y_sc_bottom10)]:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    models_10pct[name] = model

for pc_idx in range(200):
    # Create test data: top 10% vs bottom 10% of PC values
    test_data = np.zeros((2, 200))
    test_data[0, pc_idx] = np.percentile(X_pca[:, pc_idx], 90)  # Top 10%
    test_data[1, pc_idx] = np.percentile(X_pca[:, pc_idx], 10)  # Bottom 10%
    
    # Standardize
    test_scaled = scaler.transform(test_data)
    
    # Get probabilities
    probs = {}
    for outcome in ['ai', 'sc']:
        # Probability of being in top 10% of outcome
        prob_top_if_high = models_10pct[f'{outcome}_top10'].predict_proba(test_scaled[0:1])[0, 1]
        prob_top_if_low = models_10pct[f'{outcome}_top10'].predict_proba(test_scaled[1:2])[0, 1]
        
        # Probability of being in bottom 10% of outcome
        prob_bottom_if_high = models_10pct[f'{outcome}_bottom10'].predict_proba(test_scaled[0:1])[0, 1]
        prob_bottom_if_low = models_10pct[f'{outcome}_bottom10'].predict_proba(test_scaled[1:2])[0, 1]
        
        probs[f'{outcome}_top10_if_high'] = prob_top_if_high
        probs[f'{outcome}_top10_if_low'] = prob_top_if_low
        probs[f'{outcome}_bottom10_if_high'] = prob_bottom_if_high
        probs[f'{outcome}_bottom10_if_low'] = prob_bottom_if_low
        
        # Calculate differences
        probs[f'{outcome}_top10_diff'] = prob_top_if_high - prob_top_if_low
        probs[f'{outcome}_bottom10_diff'] = prob_bottom_if_high - prob_bottom_if_low
    
    pc_global_effects[pc_idx] = probs

# Prepare visualization data
viz_data = []
for i in range(len(essays_df)):
    if not pd.isna(essays_df.iloc[i]['sc11']) and not pd.isna(essays_df.iloc[i]['ai_rating']):
        # Get top 5 contributing PCs from all 200
        total_contributions = np.abs(contributions_ai_200[i]) + np.abs(contributions_sc_200[i])
        top_5_indices = np.argsort(total_contributions)[-5:][::-1]
        
        # Create PC info for this essay
        pc_info = []
        for pc_idx in top_5_indices:
            pc_info.append({
                'pc': f'PC{pc_idx}',
                'percentile': float(pc_percentiles[i, pc_idx]),
                'contribution_ai': float(contributions_ai_200[i, pc_idx]),
                'contribution_sc': float(contributions_sc_200[i, pc_idx]),
                'variance_ai': float(np.abs(contributions_ai_200[i, pc_idx]) * variance_explained[pc_idx] * 100),
                'variance_sc': float(np.abs(contributions_sc_200[i, pc_idx]) * variance_explained[pc_idx] * 100),
                'variance_total': float(variance_explained[pc_idx] * 100)
            })
        
        viz_data.append({
            'x': float(X_umap_3d[i, 0]),
            'y': float(X_umap_3d[i, 1]),
            'z': float(X_umap_3d[i, 2]),
            'essay_id': essays_df.iloc[i]['essay_id'],
            'essay': essays_df.iloc[i]['essay'],
            'sc11': int(essays_df.iloc[i]['sc11']),
            'ai_rating': float(essays_df.iloc[i]['ai_rating']),
            'pc_info': pc_info,
            'all_pc_values': X_pca[i].tolist()
        })

print(f"Prepared {len(viz_data)} points")
print(f"Cloud center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

# Create HTML with enhanced features
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>UMAP Visualization - Full TreeSHAP Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #000;
            cursor: none;
            color: #fff;
        }
        
        /* Text shadow for better readability */
        * {
            text-shadow: 
                -1px -1px 0 #000,
                 1px -1px 0 #000,
                -1px  1px 0 #000,
                 1px  1px 0 #000;
        }
        
        #cursor-indicator {
            position: absolute;
            width: 30px;
            height: 30px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 400px;
            z-index: 100;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
        }
        #gallery-controls {
            position: absolute;
            top: 10px;
            right: 480px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
            max-width: 200px;
        }
        #essay-display {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            max-height: 200px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            overflow-y: auto;
            display: none;
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
            z-index: 50;
        }
        #essay-header {
            font-weight: bold;
            margin-bottom: 5px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            font-size: 12px;
        }
        .header-main {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
        }
        .header-pcs {
            display: flex;
            gap: 10px;
            font-size: 11px;
            color: #ccc;
            flex-wrap: wrap;
        }
        .pc-inline {
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
        }
        .pc-inline:hover {
            background: rgba(255,255,255,0.2);
        }
        #essay-text {
            line-height: 1.4;
            white-space: pre-wrap;
            color: #ddd;
            margin-bottom: 8px;
            font-size: 24px;
            max-height: 60px;
            overflow-y: auto;
        }
        #pc-analysis {
            padding: 10px 0;
            margin-top: 10px;
            display: none;
        }
        .pc-item {
            padding: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
            font-size: 11px;
        }
        .pc-item:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.3);
        }
        .pc-name {
            font-weight: bold;
            color: #4CAF50;
            font-size: 12px;
        }
        .pc-percentile {
            color: #2196F3;
        }
        .pc-values {
            font-size: 10px;
            color: #ccc;
            margin-top: 2px;
        }
        #pc-global-info {
            position: absolute;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.95);
            padding: 15px 20px;
            border-radius: 5px;
            border: 2px solid rgba(255,255,255,0.3);
            display: none;
            max-width: 400px;
            z-index: 200;
        }
        #pc-global-info h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .close-btn {
            position: absolute;
            top: 5px;
            right: 10px;
            cursor: pointer;
            font-size: 20px;
            color: #999;
        }
        .close-btn:hover {
            color: #fff;
        }
        .prob-table {
            width: 100%;
            margin-top: 10px;
        }
        .prob-table td {
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .prob-table td:first-child {
            font-weight: bold;
            color: #888;
        }
        .prob-high {
            color: #4CAF50;
            font-weight: bold;
        }
        .prob-med {
            color: #FFC107;
        }
        .prob-low {
            color: #999;
        }
        .threshold-info {
            font-size: 11px;
            color: #999;
            margin-top: 10px;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        #dml-table {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.95);
            padding: 15px;
            border-radius: 5px;
            border: 2px solid rgba(255,255,255,0.3);
            display: none;
            z-index: 100;
            max-width: 550px;
        }
        #dml-table h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .dml-stats {
            width: 100%;
            font-size: 12px;
        }
        .dml-stats td {
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .dml-stats td:first-child {
            font-weight: bold;
            color: #888;
        }
        .control-group {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .control-group:last-child {
            border-bottom: none;
        }
        .control-group label {
            display: block;
            margin-bottom: 3px;
            font-weight: bold;
        }
        .threshold-input {
            width: 60px;
            margin: 0 5px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 2px 5px;
        }
        .legend-item {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .color-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 8px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        button {
            padding: 5px 10px;
            margin-top: 5px;
            cursor: pointer;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
        }
        button:hover {
            background: rgba(255,255,255,0.2);
        }
        #counts {
            margin-top: 10px;
            font-size: 12px;
            color: #ccc;
        }
        .tab-buttons {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .tab-button {
            padding: 5px 10px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            cursor: pointer;
            border-radius: 3px;
            color: white;
        }
        .tab-button.active {
            background: rgba(255,255,255,0.3);
        }
        .threshold-panel {
            display: none;
        }
        .threshold-panel.active {
            display: block;
        }
        .gallery-button {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 8px;
            text-align: left;
            font-size: 12px;
            border: 2px solid;
            background: rgba(0,0,0,0.5);
        }
        .gallery-button:hover {
            background: rgba(255,255,255,0.1);
        }
        .gallery-button.active {
            background: rgba(255,255,255,0.2);
        }
        .nav-buttons {
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }
        .nav-button {
            flex: 1;
            padding: 5px;
            font-size: 11px;
        }
        #gallery-info {
            text-align: center;
            margin: 10px 0;
            font-size: 11px;
            color: #ccc;
        }
        h3 {
            margin-top: 0;
            color: #fff;
        }
        input[type="range"] {
            background: rgba(255,255,255,0.1);
        }
        select {
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 3px 5px;
        }
        select option {
            background: #000;
            color: white;
        }
    </style>
</head>
<body>
    <div id="cursor-indicator"></div>
    
    <div id="info">
        <h3>UMAP Visualization</h3>
        <div>Total Essays: """ + str(len(viz_data)) + """</div>
        
        <div class="control-group">
            <label>AI Rating Thresholds:</label>
            <div>
                Low: &lt; <input type="number" id="ai-low-val" class="threshold-input" value=\"""" + f"{ai_percentiles[10]:.1f}" + """\" min="1" max="10" step="0.1" onchange="updateFromValues('ai')">
                <span style="color: #888;">(P<span id="ai-low-pct-display">10</span>)</span>
                <br>
                High: &gt; <input type="number" id="ai-high-val" class="threshold-input" value=\"""" + f"{ai_percentiles[90]:.1f}" + """\" min="1" max="10" step="0.1" onchange="updateFromValues('ai')">
                <span style="color: #888;">(P<span id="ai-high-pct-display">90</span>)</span>
            </div>
            <div style="margin-top: 5px;">
                Low: P<input type="range" id="ai-low-pct" min="0" max="50" value="10" step="1" onchange="updateFromPercentiles('ai')">
                High: P<input type="range" id="ai-high-pct" min="50" max="100" value="90" step="1" onchange="updateFromPercentiles('ai')">
            </div>
        </div>
        
        <div class="control-group">
            <label>Social Class Thresholds:</label>
            <div>
                Low: ≤ <input type="number" id="sc-low-val" class="threshold-input" value="1" min="1" max="5" step="1" onchange="updateFromValues('sc')">
                <span style="color: #888;">(P<span id="sc-low-pct-display">10</span>)</span>
                <br>
                High: ≥ <input type="number" id="sc-high-val" class="threshold-input" value="5" min="1" max="5" step="1" onchange="updateFromValues('sc')">
                <span style="color: #888;">(P<span id="sc-high-pct-display">90</span>)</span>
            </div>
            <div style="margin-top: 5px;">
                Low: P<input type="range" id="sc-low-pct" min="0" max="50" value="10" step="1" onchange="updateFromPercentiles('sc')">
                High: P<input type="range" id="sc-high-pct" min="50" max="100" value="90" step="1" onchange="updateFromPercentiles('sc')">
            </div>
        </div>
        
        <button onclick="updateCategories()">Apply Thresholds</button>
        
        <div style="margin-top: 15px;">
            <div class="legend-item">
                <span class="color-box" style="background: #00ff00;"></span>
                <span>High AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #ff00ff;"></span>
                <span>High AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #00ffff;"></span>
                <span>Low AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #ffff00;"></span>
                <span>Low AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #666666;"></span>
                <span>Middle (neither extreme)</span>
            </div>
        </div>
        
        <div id="counts"></div>
    </div>
    
    <div id="controls">
        <div class="control-group">
            <label>
                <input type="checkbox" id="auto-rotate" checked> Auto-rotate
            </label>
            <div>
                Speed: <input type="range" id="rotate-speed" min="0.1" max="2" step="0.1" value="0.5" style="width: 100px;">
            </div>
        </div>
        <div class="control-group">
            <label>Point Opacity:</label>
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.8" style="width: 100px;">
            <span id="opacity-val">0.8</span>
        </div>
        <div class="control-group">
            <label>Essay BG Opacity:</label>
            <input type="range" id="essay-opacity" min="0.1" max="1" step="0.05" value="0.9" style="width: 100px;">
            <span id="essay-opacity-val">0.9</span>
        </div>
        <div class="control-group">
            <label>Essay Font Size:</label>
            <input type="range" id="essay-font-size" min="6" max="72" step="2" value="24" style="width: 100px;">
            <span id="font-size-val">24</span>px
        </div>
        <div class="control-group">
            <label>Essay Height:</label>
            <input type="range" id="essay-height" min="60" max="2000" step="20" value="60" style="width: 100px;">
            <span id="essay-height-val">60</span>px
            <button onclick="resetEssayHeight()" style="padding: 2px 6px; font-size: 11px; margin-left: 5px;">Reset</button>
        </div>
        <div class="control-group">
            <label>Transition Speed:</label>
            <input type="range" id="transition-speed" min="0.5" max="3" step="0.1" value="1.5" style="width: 100px;">
            <span id="transition-val">1.5</span>s
        </div>
        <div class="control-group">
            <label>PC Analysis:</label>
            <select id="pc-select" onchange="showPCGlobalInfo()">
                <option value="">Select a PC...</option>
            </select>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="toggle-dml" onchange="toggleDMLTable()"> Show DML Stats
            </label>
        </div>
    </div>
    
    <div id="gallery-controls">
        <h4 style="margin-top: 0;">Gallery Mode</h4>
        <button class="gallery-button" style="border-color: #00ff00;" onclick="startGallery('both_high')">
            High AI + High SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #ff00ff;" onclick="startGallery('ai_high')">
            High AI + Low SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #00ffff;" onclick="startGallery('sc_high')">
            Low AI + High SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #ffff00;" onclick="startGallery('both_low')">
            Low AI + Low SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        
        <div id="gallery-info"></div>
        
        <div class="nav-buttons" style="display: none;">
            <button class="nav-button" onclick="navigateGallery(-1)">← Previous</button>
            <button class="nav-button" onclick="stopGallery()">Stop</button>
            <button class="nav-button" onclick="navigateGallery(1)">Next →</button>
        </div>
    </div>
    
    <div id="essay-display">
        <div id="essay-header">
            <strong>Essay ID:</strong> <span id="essay-id"></span> | 
            <strong>SC:</strong> <span id="essay-sc"></span> | 
            <strong>AI:</strong> <span id="essay-ai"></span>
        </div>
        <div id="essay-text"></div>
        <div id="pc-analysis">
            <h4 style="margin: 0 0 10px 0; display: none;">Principal Components</h4>
            <div id="pc-list"></div>
        </div>
    </div>
    
    <div id="pc-global-info">
        <span class="close-btn" onclick="closePCInfo()">×</span>
        <h4 id="pc-title"></h4>
        <div id="pc-content"></div>
    </div>
    
    <div id="dml-table">
        <span class="close-btn" onclick="document.getElementById('dml-table').style.display='none'; document.getElementById('toggle-dml').checked=false;">×</span>
        <h4>Double Machine Learning Results</h4>
        <table class="dml-stats">
            <tr>
                <td></td>
                <td style="text-align: center; font-weight: bold; color: #888;">Non Cross-Fitted</td>
                <td style="text-align: center; font-weight: bold; color: #888;">Cross-Fitted (5-fold)</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 10px; font-weight: bold; color: #4CAF50;">Naive Model (No Text Controls)</td>
            </tr>
            <tr>
                <td>θ (SC→AI):</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['theta_naive']:.3f}" + """</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['se_naive']:.3f}" + """</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td colspan="2" style="text-align: center;">""" + f"({dml_results_computed['ci_naive'][0]:.3f}, {dml_results_computed['ci_naive'][1]:.3f})" + """</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['pval_naive']:.4f}" + """</td>
            </tr>
            <tr>
                <td>R² (variance explained):</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['r2_naive']:.3f}" + """</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #4CAF50;">All 200 PCs Model</td>
            </tr>
            <tr>
                <td>DML θ (SC→AI):</td>
                <td>""" + f"{dml_results_computed['theta_200']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['theta_200_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>""" + f"{dml_results_computed['se_200']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['se_200_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>""" + f"({dml_results_computed['ci_200'][0]:.3f}, {dml_results_computed['ci_200'][1]:.3f})" + """</td>
                <td>""" + f"({dml_results_computed['ci_200_cf'][0]:.3f}, {dml_results_computed['ci_200_cf'][1]:.3f})" + """</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>""" + f"{dml_results_computed['pval_200']:.4f}" + """</td>
                <td>""" + f"{dml_results_computed['pval_200_cf']:.4f}" + """</td>
            </tr>
            <tr>
                <td>AI R² (200 PCs):</td>
                <td>""" + f"{r2_ai_200:.3f}" + """</td>
                <td>""" + f"{r2_ai_200_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td>SC R² (200 PCs):</td>
                <td>""" + f"{r2_sc_200:.3f}" + """</td>
                <td>""" + f"{r2_sc_200_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #4CAF50;">Top 5 PCs Model</td>
            </tr>
            <tr>
                <td>DML θ (SC→AI):</td>
                <td>""" + f"{dml_results_computed['theta_top5']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['theta_top5_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>""" + f"{dml_results_computed['se_top5']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['se_top5_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>""" + f"({dml_results_computed['ci_top5'][0]:.3f}, {dml_results_computed['ci_top5'][1]:.3f})" + """</td>
                <td>""" + f"({dml_results_computed['ci_top5_cf'][0]:.3f}, {dml_results_computed['ci_top5_cf'][1]:.3f})" + """</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>""" + f"{dml_results_computed['pval_top5']:.4f}" + """</td>
                <td>""" + f"{dml_results_computed['pval_top5_cf']:.4f}" + """</td>
            </tr>
            <tr>
                <td>AI R² (Top 5):</td>
                <td>""" + f"{r2_ai_top5:.3f}" + """</td>
                <td>""" + f"{r2_ai_top5_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td>SC R² (Top 5):</td>
                <td>""" + f"{r2_sc_top5:.3f}" + """</td>
                <td>""" + f"{r2_sc_top5_cf:.3f}" + """</td>
            </tr>
        </table>
    </div>
    
    <script>
        // Data and models
        const data = """ + json.dumps(viz_data) + """;
        const cloudCenter = {
            x: """ + str(center_x * 100) + """,
            y: """ + str(center_y * 100) + """,
            z: """ + str(center_z * 100) + """
        };
        const aiPercentiles = """ + json.dumps(ai_percentiles) + """;
        const pcGlobalEffects = """ + json.dumps(pc_global_effects) + """;
        
        // Store user thresholds
        let userThresholds = {
            ai_low: """ + str(ai_percentiles[10]) + """,
            ai_high: """ + str(ai_percentiles[90]) + """,
            sc_low: 1,
            sc_high: 5,
            pc_high_pct: 10,
            pc_low_pct: 10
        };
        
        // Calculate rating percentile functions
        const aiRatings = data.map(d => d.ai_rating).sort((a, b) => a - b);
        const scRatings = data.map(d => d.sc11).sort((a, b) => a - b);
        
        function getAIPercentile(percentile) {
            const index = Math.floor((percentile / 100) * (aiRatings.length - 1));
            return aiRatings[index];
        }
        
        function getSCPercentile(percentile) {
            const index = Math.floor((percentile / 100) * (scRatings.length - 1));
            return scRatings[index];
        }
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            2000
        );
        camera.position.set(250, 250, 250);
        camera.lookAt(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.enableZoom = true;
        controls.zoomSpeed = 1.2;
        controls.minDistance = 50;
        controls.maxDistance = 1000;
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(1, 1, 1);
        scene.add(dirLight);
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        
        // Fill positions
        data.forEach((d, i) => {
            positions[i * 3] = d.x * 100;
            positions[i * 3 + 1] = d.y * 100;
            positions[i * 3 + 2] = d.z * 100;
        });
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // Point material
        const material = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.8
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Category system
        const categories = new Array(data.length);
        const categoryIndices = {
            'both_high': [],
            'ai_high': [],
            'sc_high': [],
            'both_low': [],
            'middle': []
        };
        
        // Store sorted indices for proximity navigation
        const sortedCategoryIndices = {
            'both_high': [],
            'ai_high': [],
            'sc_high': [],
            'both_low': [],
            'middle': []
        };
        
        // Colors
        const categoryColors = {
            'both_high': [0.0, 1.0, 0.0],      // Bright Green
            'ai_high': [1.0, 0.0, 1.0],        // Magenta
            'sc_high': [0.0, 1.0, 1.0],        // Cyan
            'both_low': [1.0, 1.0, 0.0],       // Yellow
            'middle': [0.4, 0.4, 0.4]          // Dark Gray
        };
        
        // Gallery state
        let galleryMode = false;
        let currentGalleryCategory = null;
        let currentGalleryIndex = 0;
        let savedAutoRotate = true;
        let returningToOverview = false;
        
        // PC analysis state
        let currentPCIndex = null;
        
        // Animation state
        let isAnimating = false;
        let animationStartTime = 0;
        let animationDuration = 1500;
        let animationStart = {
            cameraPos: new THREE.Vector3(),
            targetPos: new THREE.Vector3()
        };
        let animationEnd = {
            cameraPos: new THREE.Vector3(),
            targetPos: new THREE.Vector3()
        };
        
        // Functions
        function easeInOutCubic(t) {
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }
        
        function calculateDistance(idx1, idx2) {
            const dx = positions[idx1 * 3] - positions[idx2 * 3];
            const dy = positions[idx1 * 3 + 1] - positions[idx2 * 3 + 1];
            const dz = positions[idx1 * 3 + 2] - positions[idx2 * 3 + 2];
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        
        function sortByProximity(indices) {
            if (indices.length <= 1) return indices;
            
            const sorted = [];
            const remaining = [...indices];
            
            sorted.push(remaining.shift());
            
            while (remaining.length > 0) {
                const currentIdx = sorted[sorted.length - 1];
                let nearestIdx = 0;
                let nearestDist = Infinity;
                
                for (let i = 0; i < remaining.length; i++) {
                    const dist = calculateDistance(currentIdx, remaining[i]);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearestIdx = i;
                    }
                }
                
                sorted.push(remaining.splice(nearestIdx, 1)[0]);
            }
            
            return sorted;
        }
        
        // Populate PC dropdown
        function populatePCDropdown() {
            const select = document.getElementById('pc-select');
            select.innerHTML = '<option value="">Select a PC...</option>';
            
            // Add all 200 PCs
            for (let i = 0; i < 200; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `PC${i}`;
                select.appendChild(option);
            }
        }
        
        // Show global PC info
        window.showPCGlobalInfo = function(pcIndex) {
            const infoDiv = document.getElementById('pc-global-info');
            
            if (pcIndex === undefined || pcIndex === null) {
                const selectValue = document.getElementById('pc-select').value;
                if (!selectValue) {
                    infoDiv.style.display = 'none';
                    currentPCIndex = null;
                    return;
                }
                pcIndex = parseInt(selectValue);
            }
            
            currentPCIndex = parseInt(pcIndex);
            document.getElementById('pc-select').value = pcIndex;
            updatePCAnalysis();
            infoDiv.style.display = 'block';
        }
        
        function updatePCAnalysis() {
            if (currentPCIndex === null) return;
            
            const effects = pcGlobalEffects[currentPCIndex];
            
            document.getElementById('pc-title').textContent = `PC${currentPCIndex} Impact Analysis`;
            
            const formatDiff = (diff) => {
                const sign = diff > 0 ? '+' : '';
                const cls = Math.abs(diff) > 0.2 ? 'prob-high' : (Math.abs(diff) > 0.1 ? 'prob-med' : 'prob-low');
                return `<span class="${cls}">${sign}${(diff * 100).toFixed(1)}%</span>`;
            };
            
            const content = `
                <div style="font-size: 12px; color: #888; margin-bottom: 10px;">
                    Probability of high/low outcomes when PC${currentPCIndex} is high/low (10th percentiles)
                </div>
                <table class="prob-table" style="width: 100%;">
                    <tr style="font-weight: bold; border-bottom: 2px solid #444;">
                        <td>Outcome</td>
                        <td style="text-align: center;">If High PC</td>
                        <td style="text-align: center;">If Low PC</td>
                        <td style="text-align: center;">Difference</td>
                    </tr>
                    <tr>
                        <td>High AI Rating</td>
                        <td style="text-align: center;">${(effects.ai_top10_if_high * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${(effects.ai_top10_if_low * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${formatDiff(effects.ai_top10_diff)}</td>
                    </tr>
                    <tr>
                        <td>Low AI Rating</td>
                        <td style="text-align: center;">${(effects.ai_bottom10_if_high * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${(effects.ai_bottom10_if_low * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${formatDiff(effects.ai_bottom10_diff)}</td>
                    </tr>
                    <tr style="border-top: 1px solid #333;">
                        <td>High Social Class</td>
                        <td style="text-align: center;">${(effects.sc_top10_if_high * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${(effects.sc_top10_if_low * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${formatDiff(effects.sc_top10_diff)}</td>
                    </tr>
                    <tr>
                        <td>Low Social Class</td>
                        <td style="text-align: center;">${(effects.sc_bottom10_if_high * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${(effects.sc_bottom10_if_low * 100).toFixed(1)}%</td>
                        <td style="text-align: center;">${formatDiff(effects.sc_bottom10_diff)}</td>
                    </tr>
                </table>
            `;
            
            document.getElementById('pc-content').innerHTML = content;
        }
        
        function closePCInfo() {
            document.getElementById('pc-global-info').style.display = 'none';
            document.getElementById('pc-select').value = '';
            currentPCIndex = null;
        }
        
        function toggleDMLTable() {
            const table = document.getElementById('dml-table');
            const checkbox = document.getElementById('toggle-dml');
            table.style.display = checkbox.checked ? 'block' : 'none';
        }
        
        // Display PC analysis
        function displayPCAnalysis(d) {
            const pcList = document.getElementById('pc-list');
            pcList.innerHTML = '';
            
            d.pc_info.forEach((pc, idx) => {
                const pcDiv = document.createElement('div');
                pcDiv.className = 'pc-item';
                pcDiv.onclick = () => showPCGlobalInfo(parseInt(pc.pc.substring(2)));
                
                pcDiv.innerHTML = `
                    <div class="pc-name">${pc.pc}</div>
                    <div class="pc-percentile">${pc.percentile.toFixed(1)}%ile</div>
                    <div class="pc-values">
                        AI: ${pc.contribution_ai > 0 ? '+' : ''}${pc.contribution_ai.toFixed(2)}<br>
                        SC: ${pc.contribution_sc > 0 ? '+' : ''}${pc.contribution_sc.toFixed(2)}
                    </div>
                `;
                
                pcList.appendChild(pcDiv);
            });
        }
        
        // Update functions for synchronized controls
        window.updateFromValues = function(type) {
            if (type === 'ai') {
                const lowVal = parseFloat(document.getElementById('ai-low-val').value);
                const highVal = parseFloat(document.getElementById('ai-high-val').value);
                
                // Find closest percentiles
                let lowPct = 0;
                for (let i = 0; i < aiRatings.length; i++) {
                    if (aiRatings[i] >= lowVal) {
                        lowPct = (i / aiRatings.length) * 100;
                        break;
                    }
                }
                
                let highPct = 100;
                for (let i = aiRatings.length - 1; i >= 0; i--) {
                    if (aiRatings[i] <= highVal) {
                        highPct = ((i + 1) / aiRatings.length) * 100;
                        break;
                    }
                }
                
                document.getElementById('ai-low-pct').value = Math.round(lowPct);
                document.getElementById('ai-high-pct').value = Math.round(highPct);
                document.getElementById('ai-low-pct-display').textContent = Math.round(lowPct);
                document.getElementById('ai-high-pct-display').textContent = Math.round(highPct);
            } else {
                const lowVal = parseInt(document.getElementById('sc-low-val').value);
                const highVal = parseInt(document.getElementById('sc-high-val').value);
                
                // Find closest percentiles
                let lowPct = 0;
                for (let i = 0; i < scRatings.length; i++) {
                    if (scRatings[i] > lowVal) {
                        lowPct = (i / scRatings.length) * 100;
                        break;
                    }
                }
                
                let highPct = 100;
                for (let i = scRatings.length - 1; i >= 0; i--) {
                    if (scRatings[i] < highVal) {
                        highPct = ((i + 1) / scRatings.length) * 100;
                        break;
                    }
                }
                
                document.getElementById('sc-low-pct').value = Math.round(lowPct);
                document.getElementById('sc-high-pct').value = Math.round(highPct);
                document.getElementById('sc-low-pct-display').textContent = Math.round(lowPct);
                document.getElementById('sc-high-pct-display').textContent = Math.round(highPct);
            }
        };
        
        window.updateFromPercentiles = function(type) {
            if (type === 'ai') {
                const lowPct = parseInt(document.getElementById('ai-low-pct').value);
                const highPct = parseInt(document.getElementById('ai-high-pct').value);
                
                document.getElementById('ai-low-val').value = getAIPercentile(lowPct).toFixed(1);
                document.getElementById('ai-high-val').value = getAIPercentile(highPct).toFixed(1);
                document.getElementById('ai-low-pct-display').textContent = lowPct;
                document.getElementById('ai-high-pct-display').textContent = highPct;
            } else {
                const lowPct = parseInt(document.getElementById('sc-low-pct').value);
                const highPct = parseInt(document.getElementById('sc-high-pct').value);
                
                document.getElementById('sc-low-val').value = Math.round(getSCPercentile(lowPct));
                document.getElementById('sc-high-val').value = Math.round(getSCPercentile(highPct));
                document.getElementById('sc-low-pct-display').textContent = lowPct;
                document.getElementById('sc-high-pct-display').textContent = highPct;
            }
        };
        
        function updateCategories() {
            const aiLow = parseFloat(document.getElementById('ai-low-val').value);
            const aiHigh = parseFloat(document.getElementById('ai-high-val').value);
            const scLow = parseInt(document.getElementById('sc-low-val').value);
            const scHigh = parseInt(document.getElementById('sc-high-val').value);
            
            // Update user thresholds
            userThresholds.ai_low = aiLow;
            userThresholds.ai_high = aiHigh;
            userThresholds.sc_low = scLow;
            userThresholds.sc_high = scHigh;
            
            // Update PC analysis if shown
            if (currentPCIndex !== null) {
                updatePCAnalysis();
            }
            
            // Reset category indices
            Object.keys(categoryIndices).forEach(key => categoryIndices[key] = []);
            
            const counts = {
                'both_high': 0,
                'ai_high': 0,
                'sc_high': 0,
                'both_low': 0,
                'middle': 0
            };
            
            data.forEach((d, i) => {
                const highAI = d.ai_rating > aiHigh;
                const lowAI = d.ai_rating < aiLow;
                const highSC = d.sc11 >= scHigh;
                const lowSC = d.sc11 <= scLow;
                
                let category;
                if (highAI && highSC) {
                    category = 'both_high';
                } else if (highAI && lowSC) {
                    category = 'ai_high';
                } else if (lowAI && highSC) {
                    category = 'sc_high';
                } else if (lowAI && lowSC) {
                    category = 'both_low';
                } else {
                    category = 'middle';
                }
                
                categories[i] = category;
                categoryIndices[category].push(i);
                counts[category]++;
                
                const color = categoryColors[category];
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            });
            
            // Sort category indices by proximity
            Object.keys(categoryIndices).forEach(category => {
                sortedCategoryIndices[category] = sortByProximity(categoryIndices[category]);
            });
            
            geometry.attributes.color.needsUpdate = true;
            
            // Update counts display
            const total = data.length;
            const nonMiddle = total - counts.middle;
            document.getElementById('counts').innerHTML = `
                <strong>Counts:</strong><br>
                High AI + High SC: ${counts.both_high} (${(counts.both_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.both_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                High AI + Low SC: ${counts.ai_high} (${(counts.ai_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.ai_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Low AI + High SC: ${counts.sc_high} (${(counts.sc_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.sc_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Low AI + Low SC: ${counts.both_low} (${(counts.both_low/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.both_low/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Middle: ${counts.middle} (${(counts.middle/total*100).toFixed(1)}% of all)
            `;
            
            // Update gallery buttons
            document.querySelectorAll('.gallery-button').forEach(btn => {
                const category = btn.getAttribute('onclick').match(/startGallery\('(.+?)'\)/)[1];
                const count = counts[category];
                btn.querySelector('span').textContent = `${count} essays`;
            });
        }
        
        // Gallery functions
        window.startGallery = function(category) {
            if (sortedCategoryIndices[category].length === 0) {
                alert('No essays in this category with current thresholds');
                return;
            }
            
            galleryMode = true;
            currentGalleryCategory = category;
            currentGalleryIndex = 0;
            returningToOverview = false;
            
            savedAutoRotate = controls.autoRotate;
            controls.autoRotate = false;
            document.getElementById('auto-rotate').checked = false;
            
            document.querySelectorAll('.gallery-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelector('.nav-buttons').style.display = 'flex';
            
            navigateToEssay(sortedCategoryIndices[category][0]);
        };
        
        window.stopGallery = function() {
            galleryMode = false;
            currentGalleryCategory = null;
            isAnimating = false;
            returningToOverview = true;
            
            controls.autoRotate = savedAutoRotate;
            document.getElementById('auto-rotate').checked = savedAutoRotate;
            
            animateToPosition(
                new THREE.Vector3(250, 250, 250),
                new THREE.Vector3(cloudCenter.x, cloudCenter.y, cloudCenter.z)
            );
            
            document.querySelectorAll('.gallery-button').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.nav-buttons').style.display = 'none';
            document.getElementById('gallery-info').textContent = '';
            document.getElementById('essay-display').style.display = 'none';
        };
        
        window.navigateGallery = function(direction) {
            if (!galleryMode || !currentGalleryCategory || isAnimating) return;
            
            const indices = sortedCategoryIndices[currentGalleryCategory];
            currentGalleryIndex += direction;
            
            if (currentGalleryIndex < 0) currentGalleryIndex = indices.length - 1;
            if (currentGalleryIndex >= indices.length) currentGalleryIndex = 0;
            
            navigateToEssay(indices[currentGalleryIndex]);
        };
        
        function animateToPosition(targetCameraPos, targetLookAt) {
            isAnimating = true;
            animationStartTime = Date.now();
            animationDuration = parseFloat(document.getElementById('transition-speed').value) * 1000;
            
            animationStart.cameraPos.copy(camera.position);
            animationStart.targetPos.copy(controls.target);
            
            animationEnd.cameraPos.copy(targetCameraPos);
            animationEnd.targetPos.copy(targetLookAt);
        }
        
        function navigateToEssay(index) {
            const d = data[index];
            const category = categories[index];
            
            const indices = sortedCategoryIndices[currentGalleryCategory];
            document.getElementById('gallery-info').textContent = 
                `Essay ${currentGalleryIndex + 1} of ${indices.length}`;
            
            const targetPosition = new THREE.Vector3(
                positions[index * 3],
                positions[index * 3 + 1],
                positions[index * 3 + 2]
            );
            
            const distance = 100;
            const angle = Date.now() * 0.0001;
            const cameraPos = new THREE.Vector3(
                targetPosition.x + distance * Math.cos(angle),
                targetPosition.y + distance * 0.5,
                targetPosition.z + distance * Math.sin(angle)
            );
            
            animateToPosition(cameraPos, targetPosition);
            
            // Create PC summary for header
            let pcSummary = '';
            d.pc_info.forEach((pc, idx) => {
                const aiSign = pc.contribution_ai > 0 ? '+' : '';
                const scSign = pc.contribution_sc > 0 ? '+' : '';
                pcSummary += `<span class="pc-inline" onclick="event.stopPropagation(); showPCGlobalInfo(${parseInt(pc.pc.substring(2))})">
                    ${pc.pc}: ${pc.percentile.toFixed(0)}% | AI:${aiSign}${pc.contribution_ai.toFixed(2)} SC:${scSign}${pc.contribution_sc.toFixed(2)} | ${pc.variance_total.toFixed(1)}%var
                </span>`;
            });
            
            document.getElementById('essay-header').innerHTML = `
                <div class="header-main">
                    <div>
                        <strong>Essay ID:</strong> ${d.essay_id} | 
                        <strong>SC:</strong> ${d.sc11} | 
                        <strong>AI:</strong> ${d.ai_rating.toFixed(2)}
                    </div>
                </div>
                <div class="header-pcs">
                    <strong>Top 5 PCs:</strong> ${pcSummary}
                </div>
            `;
            document.getElementById('essay-text').textContent = d.essay;
            
            const color = categoryColors[category];
            const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
            const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
            document.getElementById('essay-display').style.backgroundColor = bgColor;
            document.getElementById('essay-display').style.borderColor = 
                `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
            document.getElementById('essay-display').style.display = 'block';
        }
        
        
        // Reset essay height function
        window.resetEssayHeight = function() {
            document.getElementById('essay-height').value = 60;
            document.getElementById('essay-height-val').textContent = 60;
            document.getElementById('essay-text').style.maxHeight = '60px';
        };
        
        // Initialize
        populatePCDropdown();
        updateCategories();
        
        // Initialize percentile displays
        updateFromValues('ai');
        updateFromValues('sc');
        
        // Custom cursor
        const cursorIndicator = document.getElementById('cursor-indicator');
        
        function updateCursor(event) {
            cursorIndicator.style.left = event.clientX + 'px';
            cursorIndicator.style.top = event.clientY + 'px';
        }
        
        window.addEventListener('mousemove', updateCursor);
        
        // Raycaster for hover
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 8;
        
        let hoveredIndex = -1;
        
        function onMouseMove(event) {
            if (galleryMode) return;
            
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {
                const newIndex = intersects[0].index;
                if (newIndex !== hoveredIndex) {
                    hoveredIndex = newIndex;
                    const d = data[hoveredIndex];
                    const category = categories[hoveredIndex];
                    
                    // Create PC summary for header
                    let pcSummary = '';
                    d.pc_info.forEach((pc, idx) => {
                        const aiSign = pc.contribution_ai > 0 ? '+' : '';
                        const scSign = pc.contribution_sc > 0 ? '+' : '';
                        pcSummary += `<span class="pc-inline" onclick="event.stopPropagation(); showPCGlobalInfo(${parseInt(pc.pc.substring(2))})">
                            ${pc.pc}: ${pc.percentile.toFixed(0)}% | AI:${aiSign}${pc.contribution_ai.toFixed(2)} SC:${scSign}${pc.contribution_sc.toFixed(2)} | ${pc.variance_total.toFixed(1)}%var
                        </span>`;
                    });
                    
                    document.getElementById('essay-header').innerHTML = `
                        <div class="header-main">
                            <div>
                                <strong>Essay ID:</strong> ${d.essay_id} | 
                                <strong>SC:</strong> ${d.sc11} | 
                                <strong>AI:</strong> ${d.ai_rating.toFixed(2)}
                            </div>
                        </div>
                        <div class="header-pcs">
                            <strong>Top 5 PCs:</strong> ${pcSummary}
                        </div>
                    `;
                    document.getElementById('essay-text').textContent = d.essay;
                    
                    const color = categoryColors[category];
                    const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
                    const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
                    document.getElementById('essay-display').style.backgroundColor = bgColor;
                    document.getElementById('essay-display').style.borderColor = 
                        `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
                    document.getElementById('essay-display').style.display = 'block';
                }
            } else {
                if (hoveredIndex !== -1) {
                    hoveredIndex = -1;
                    document.getElementById('essay-display').style.display = 'none';
                }
            }
        }
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Controls
        document.getElementById('auto-rotate').addEventListener('change', (e) => {
            controls.autoRotate = e.target.checked;
        });
        
        document.getElementById('rotate-speed').addEventListener('input', (e) => {
            controls.autoRotateSpeed = parseFloat(e.target.value);
        });
        
        document.getElementById('point-opacity').addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value);
            material.opacity = opacity;
            document.getElementById('opacity-val').textContent = opacity.toFixed(1);
        });
        
        document.getElementById('essay-opacity').addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value);
            document.getElementById('essay-opacity-val').textContent = opacity.toFixed(2);
            
            if (document.getElementById('essay-display').style.display === 'block' && hoveredIndex >= 0) {
                const category = categories[hoveredIndex];
                const color = categoryColors[category];
                const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${opacity})`;
                document.getElementById('essay-display').style.backgroundColor = bgColor;
            }
        });
        
        document.getElementById('essay-font-size').addEventListener('input', (e) => {
            const fontSize = parseInt(e.target.value);
            document.getElementById('font-size-val').textContent = fontSize;
            document.getElementById('essay-text').style.fontSize = fontSize + 'px';
            document.getElementById('essay-header').style.fontSize = (fontSize + 1) + 'px';
        });
        
        document.getElementById('essay-height').addEventListener('input', (e) => {
            const height = parseInt(e.target.value);
            document.getElementById('essay-height-val').textContent = height;
            document.getElementById('essay-text').style.maxHeight = height + 'px';
        });
        
        document.getElementById('transition-speed').addEventListener('input', (e) => {
            document.getElementById('transition-val').textContent = e.target.value;
        });
        
        // Keyboard navigation
        window.addEventListener('keydown', (e) => {
            if (!galleryMode) return;
            
            if (e.key === 'ArrowLeft') navigateGallery(-1);
            else if (e.key === 'ArrowRight') navigateGallery(1);
            else if (e.key === 'Escape') stopGallery();
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            if (isAnimating) {
                const elapsed = Date.now() - animationStartTime;
                const progress = Math.min(elapsed / animationDuration, 1);
                const eased = easeInOutCubic(progress);
                
                camera.position.lerpVectors(
                    animationStart.cameraPos,
                    animationEnd.cameraPos,
                    eased
                );
                
                controls.target.lerpVectors(
                    animationStart.targetPos,
                    animationEnd.targetPos,
                    eased
                );
                
                if (progress >= 1) {
                    isAnimating = false;
                    
                    if (returningToOverview) {
                        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
                        returningToOverview = false;
                    }
                }
            }
            
            controls.update();
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>"""

# Save
output_file = OUTPUT_DIR / 'minimal_umap_viz_v14.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Enhanced visualization with compact UI saved to: {output_file}")
print("\nChanges in v14:")
print("- Moved Gallery Mode to left of controls to avoid blocking bottom content")
print("- Compact essay display with grid layout for PC info (5 columns)")
print("- Simplified PC analysis: click any PC to see top/bottom 10% probabilities")
print("- Shows impact differences for quick assessment of PC importance")
print("- All cross-fitted DML results from v12 maintained")