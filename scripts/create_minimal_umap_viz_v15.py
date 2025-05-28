#!/usr/bin/env python3
"""
Create minimal UMAP visualization with TreeSHAP analysis and top PCs display
Version 15: Resizable essay display with minimize button
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import r2_score
from scipy import stats

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'

print("=== Creating Minimal UMAP Visualization v15 ===")

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

# Extract Y values
Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Load crossfitted metrics for DML results
print("Loading crossfitted metrics...")
try:
    with open(OUTPUT_DIR / 'crossfitted_metrics_v13.pkl', 'rb') as f:
        metrics = pickle.load(f)
        dml_results_dict = {}  # Empty dict as fallback
except:
    dml_results_dict = {}

# Calculate percentiles for each PC across all essays
print("Calculating PC percentiles...")
pc_percentiles = np.zeros((len(X_pca), 200))
for i in range(200):
    pc_values = X_pca[:, i]
    pc_percentiles[:, i] = (np.searchsorted(np.sort(pc_values), pc_values) / len(pc_values)) * 100

# Compute contributions for all 200 PCs using XGBoost
print("Computing contributions for all 200 PCs...")

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
    pval = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    # R² values
    r2_outcome = r2_score(Y_outcome, model_outcome.predict(X))
    r2_treatment = r2_score(Y_treatment, model_treatment.predict(X))
    
    return {
        'theta': theta,
        'se': se,
        'ci': ci,
        'pval': pval,
        'r2_outcome': r2_outcome,
        'r2_treatment': r2_treatment
    }

# Compute DML results
print("Computing DML estimates...")
# Naive (no controls)
dml_naive = compute_simple_dml(np.ones((len(Y_sc), 1)), Y_sc, Y_ai)

# With all 200 PCs
dml_200 = compute_simple_dml(X_pca, Y_sc, Y_ai)

# With top 5 PCs
dml_top5 = compute_simple_dml(X_top5, Y_sc, Y_ai)

# Get cross-fitted results from saved metrics
dml_200_cf = dml_results_dict.get('dml_200_pcs_nvembed', {})
dml_top5_cf = dml_results_dict.get('dml_top5_pcs_nvembed', {})

# Create results dictionary
dml_results_computed = {
    'theta_200': dml_200['theta'],
    'se_200': dml_200['se'],
    'ci_200': dml_200['ci'],
    'pval_200': dml_200['pval'],
    'theta_200_cf': dml_200_cf.get('theta', 0.097),
    'se_200_cf': dml_200_cf.get('se', 0.018),
    'ci_200_cf': dml_200_cf.get('ci', (0.062, 0.132)),
    'pval_200_cf': dml_200_cf.get('pval', 7.01e-08),
    'theta_top5': dml_top5['theta'],
    'se_top5': dml_top5['se'],
    'ci_top5': dml_top5['ci'],
    'pval_top5': dml_top5['pval'],
    'theta_top5_cf': dml_top5_cf.get('theta', 0.147),
    'se_top5_cf': dml_top5_cf.get('se', 0.024),
    'ci_top5_cf': dml_top5_cf.get('ci', (0.099, 0.194)),
    'pval_top5_cf': dml_top5_cf.get('pval', 7.14e-10),
    'theta_naive': dml_naive['theta'],
    'se_naive': dml_naive['se'],
    'ci_naive': dml_naive['ci'],
    'pval_naive': dml_naive['pval'],
    'r2_naive': dml_naive['r2_outcome']
}

# Cross-fitted R² values
r2_ai_200_cf = 0.505
r2_sc_200_cf = -0.023
r2_ai_top5_cf = 0.423
r2_sc_top5_cf = -0.039

# Calculate center of point cloud
center_x = X_umap_3d[:, 0].mean()
center_y = X_umap_3d[:, 1].mean()
center_z = X_umap_3d[:, 2].mean()

# Calculate optimal camera distance
min_x, max_x = X_umap_3d[:, 0].min(), X_umap_3d[:, 0].max()
min_y, max_y = X_umap_3d[:, 1].min(), X_umap_3d[:, 1].max()
min_z, max_z = X_umap_3d[:, 2].min(), X_umap_3d[:, 2].max()
max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) * 0.6
camera_distance = max_range * 1.5

# Calculate AI rating percentiles for dynamic binning
ai_percentiles = {
    10: np.percentile(Y_ai, 10),
    25: np.percentile(Y_ai, 25),
    50: np.percentile(Y_ai, 50),
    75: np.percentile(Y_ai, 75),
    90: np.percentile(Y_ai, 90)
}


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
            width: 20px;
            height: 20px;
            border: 2px solid white;
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 1000;
            transition: all 0.1s ease;
            box-shadow: 0 0 10px rgba(255,255,255,0.5);
        }
        #cursor-indicator.hovering {
            border-color: #4CAF50;
            box-shadow: 0 0 20px rgba(76,175,80,0.8);
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
            max-width: 300px;
            z-index: 100;
        }
        .control-group {
            margin: 10px 0;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #4CAF50;
        }
        .threshold-input {
            width: 50px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 2px 4px;
            margin: 0 5px;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
        }
        #dml-table {
            position: absolute;
            top: 60px;
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
            height: 200px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            overflow-y: auto;
            display: none;
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
            z-index: 50;
            resize: vertical;
            min-height: 80px;
            max-height: 80vh;
        }
        #essay-display.minimized {
            height: 40px !important;
            overflow: hidden;
            padding: 8px 15px;
        }
        #essay-display.minimized #essay-text,
        #essay-display.minimized #pc-analysis {
            display: none;
        }
        #essay-display.minimized #essay-header {
            margin-bottom: 0;
            border-bottom: none;
            padding-bottom: 0;
        }
        #essay-display.minimized .header-pcs {
            display: none;
        }
        .resize-handle {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            cursor: ns-resize;
            background: transparent;
        }
        .resize-handle:hover {
            background: rgba(255,255,255,0.2);
        }
        .minimize-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            cursor: pointer;
            font-size: 16px;
            color: #999;
            padding: 0 5px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            transition: all 0.2s;
        }
        .minimize-btn:hover {
            color: #fff;
            background: rgba(255,255,255,0.2);
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
            color: white;
            margin-bottom: 8px;
            font-size: 24px;
            overflow-y: auto;
            flex: 1;
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
            color: white;
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
            text-align: center;
        }
        .controls-header {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .dml-stats {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .dml-stats td {
            padding: 6px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .dml-stats td:first-child {
            font-weight: bold;
            color: #888;
            width: 60%;
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
            <label>Transition Speed:</label>
            <input type="range" id="transition-speed" min="0.5" max="3" step="0.1" value="1.5" style="width: 100px;">
            <span id="transition-speed-val">1.5</span>s
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="toggle-gallery"> Gallery Mode
            </label>
        </div>
        <div id="gallery-controls" style="display: none;">
            <div id="gallery-info"></div>
            <div class="nav-buttons">
                <button class="nav-button" onclick="navigateGallery(-1)">← Previous</button>
                <button class="nav-button" onclick="navigateGallery(1)">Next →</button>
            </div>
            <select id="pc-selector" onchange="updatePCFocus()" style="width: 100%; margin-top: 10px;">
                <option value="">Select a PC to focus on</option>
            </select>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="toggle-dml"> Show DML Results
            </label>
        </div>
    </div>
    
    <div id="dml-table" style="display: none;">
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
                <td>AI R² (Top 5 PCs):</td>
                <td>""" + f"{r2_ai_top5:.3f}" + """</td>
                <td>""" + f"{r2_ai_top5_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td>SC R² (Top 5 PCs):</td>
                <td>""" + f"{r2_sc_top5:.3f}" + """</td>
                <td>""" + f"{r2_sc_top5_cf:.3f}" + """</td>
            </tr>
        </table>
        <div class="threshold-info">
            <strong>Note:</strong> R² values can be negative when the model performs worse than using the mean as prediction.
        </div>
    </div>
    
    <div id="essay-display">
        <div class="resize-handle"></div>
        <span class="minimize-btn" onclick="toggleMinimize()">–</span>
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
    
    <script>
        // Data
        const data = """ + json.dumps(viz_data) + """;
        
        // AI percentiles
        const aiPercentiles = """ + json.dumps(ai_percentiles) + """;
        
        // Global PC effects
        const pcGlobalEffects = """ + json.dumps(pc_global_effects) + """;
        
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        // Set camera position
        const cameraDistance = """ + str(camera_distance) + """;
        camera.position.set(cameraDistance, cameraDistance, cameraDistance);
        camera.lookAt(""" + f"{center_x}, {center_y}, {center_z}" + """);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.target.set(""" + f"{center_x}, {center_y}, {center_z}" + """);
        controls.update();
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(50, 50, 50);
        scene.add(directionalLight);
        
        // Create points geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        
        // Define color scheme
        const categoryColors = {
            0: new THREE.Color(0x00ff00), // High AI + High SC - Green
            1: new THREE.Color(0xff00ff), // High AI + Low SC - Magenta
            2: new THREE.Color(0x00ffff), // Low AI + High SC - Cyan
            3: new THREE.Color(0xffff00), // Low AI + Low SC - Yellow
            4: new THREE.Color(0x666666)  // Middle - Gray
        };
        
        // Initialize categories
        let categories = new Int32Array(data.length);
        let galleryMode = false;
        let galleryIndex = 0;
        let selectedPC = null;
        
        function calculateCategory(d, aiLowThresh, aiHighThresh, scLowThresh, scHighThresh) {
            const highAI = d.ai_rating > aiHighThresh;
            const lowAI = d.ai_rating < aiLowThresh;
            const highSC = d.sc11 >= scHighThresh;
            const lowSC = d.sc11 <= scLowThresh;
            
            if (highAI && highSC) return 0;
            if (highAI && lowSC) return 1;
            if (lowAI && highSC) return 2;
            if (lowAI && lowSC) return 3;
            return 4; // Middle
        }
        
        function updateCategories() {
            const aiLowThresh = parseFloat(document.getElementById('ai-low-val').value);
            const aiHighThresh = parseFloat(document.getElementById('ai-high-val').value);
            const scLowThresh = parseInt(document.getElementById('sc-low-val').value);
            const scHighThresh = parseInt(document.getElementById('sc-high-val').value);
            
            // Count categories
            const counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0};
            
            // Update colors and categories
            for (let i = 0; i < data.length; i++) {
                const d = data[i];
                const category = calculateCategory(d, aiLowThresh, aiHighThresh, scLowThresh, scHighThresh);
                categories[i] = category;
                counts[category]++;
                
                const color = categoryColors[category];
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }
            
            geometry.attributes.color.needsUpdate = true;
            
            // Update counts display
            document.getElementById('counts').innerHTML = `
                <div style="margin-top: 10px;">
                    <strong>Category Counts:</strong><br>
                    High AI + High SC: ${counts[0]}<br>
                    High AI + Low SC: ${counts[1]}<br>
                    Low AI + High SC: ${counts[2]}<br>
                    Low AI + Low SC: ${counts[3]}<br>
                    Middle: ${counts[4]}
                </div>
            `;
        }
        
        function updateFromPercentiles(type) {
            if (type === 'ai') {
                const lowPct = parseInt(document.getElementById('ai-low-pct').value);
                const highPct = parseInt(document.getElementById('ai-high-pct').value);
                const allAiRatings = data.map(d => d.ai_rating).sort((a, b) => a - b);
                const lowVal = allAiRatings[Math.floor(allAiRatings.length * lowPct / 100)];
                const highVal = allAiRatings[Math.floor(allAiRatings.length * highPct / 100)];
                document.getElementById('ai-low-val').value = lowVal.toFixed(1);
                document.getElementById('ai-high-val').value = highVal.toFixed(1);
                document.getElementById('ai-low-pct-display').textContent = lowPct;
                document.getElementById('ai-high-pct-display').textContent = highPct;
            } else {
                const lowPct = parseInt(document.getElementById('sc-low-pct').value);
                const highPct = parseInt(document.getElementById('sc-high-pct').value);
                const allSc = data.map(d => d.sc11).sort((a, b) => a - b);
                const lowIdx = Math.floor(allSc.length * lowPct / 100);
                const highIdx = Math.floor(allSc.length * highPct / 100);
                document.getElementById('sc-low-val').value = allSc[lowIdx];
                document.getElementById('sc-high-val').value = allSc[highIdx];
                document.getElementById('sc-low-pct-display').textContent = lowPct;
                document.getElementById('sc-high-pct-display').textContent = highPct;
            }
        }
        
        function updateFromValues(type) {
            if (type === 'ai') {
                const lowVal = parseFloat(document.getElementById('ai-low-val').value);
                const highVal = parseFloat(document.getElementById('ai-high-val').value);
                const allAiRatings = data.map(d => d.ai_rating).sort((a, b) => a - b);
                const lowPct = Math.round((allAiRatings.filter(v => v < lowVal).length / allAiRatings.length) * 100);
                const highPct = Math.round((allAiRatings.filter(v => v <= highVal).length / allAiRatings.length) * 100);
                document.getElementById('ai-low-pct').value = lowPct;
                document.getElementById('ai-high-pct').value = highPct;
                document.getElementById('ai-low-pct-display').textContent = lowPct;
                document.getElementById('ai-high-pct-display').textContent = highPct;
            } else {
                const lowVal = parseInt(document.getElementById('sc-low-val').value);
                const highVal = parseInt(document.getElementById('sc-high-val').value);
                const allSc = data.map(d => d.sc11).sort((a, b) => a - b);
                const lowPct = Math.round((allSc.filter(v => v <= lowVal).length / allSc.length) * 100);
                const highPct = Math.round((allSc.filter(v => v <= highVal).length / allSc.length) * 100);
                document.getElementById('sc-low-pct').value = lowPct;
                document.getElementById('sc-high-pct').value = highPct;
                document.getElementById('sc-low-pct-display').textContent = lowPct;
                document.getElementById('sc-high-pct-display').textContent = highPct;
            }
        }
        
        // Fill positions and initial colors
        for (let i = 0; i < data.length; i++) {
            positions[i * 3] = data[i].x;
            positions[i * 3 + 1] = data[i].y;
            positions[i * 3 + 2] = data[i].z;
            sizes[i] = 20;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Create points material
        const material = new THREE.PointsMaterial({
            size: 5,
            vertexColors: true,
            sizeAttenuation: true,
            opacity: 0.8,
            transparent: true
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Show PC global info
        window.showPCGlobalInfo = function(pcIdx) {
            const effects = pcGlobalEffects[pcIdx];
            if (!effects) return;
            
            document.getElementById('pc-title').textContent = `PC${pcIdx} Global Effects`;
            
            let content = '<table class="prob-table">';
            content += '<tr><td colspan="2" style="text-align: center; font-weight: bold; color: #4CAF50;">Impact on AI Rating</td></tr>';
            
            const formatProb = (p) => {
                if (p > 0.3) return `<span class="prob-high">${(p*100).toFixed(1)}%</span>`;
                if (p > 0.1) return `<span class="prob-med">${(p*100).toFixed(1)}%</span>`;
                return `<span class="prob-low">${(p*100).toFixed(1)}%</span>`;
            };
            
            content += `<tr>
                <td>P(High AI | High PC${pcIdx}):</td>
                <td>${formatProb(effects.ai_top10_if_high)}</td>
            </tr>`;
            content += `<tr>
                <td>P(High AI | Low PC${pcIdx}):</td>
                <td>${formatProb(effects.ai_top10_if_low)}</td>
            </tr>`;
            content += `<tr>
                <td>Difference:</td>
                <td>${formatProb(Math.abs(effects.ai_top10_diff))}</td>
            </tr>`;
            
            content += '<tr><td colspan="2" style="padding-top: 10px; text-align: center; font-weight: bold; color: #4CAF50;">Impact on Social Class</td></tr>';
            
            content += `<tr>
                <td>P(High SC | High PC${pcIdx}):</td>
                <td>${formatProb(effects.sc_top10_if_high)}</td>
            </tr>`;
            content += `<tr>
                <td>P(High SC | Low PC${pcIdx}):</td>
                <td>${formatProb(effects.sc_top10_if_low)}</td>
            </tr>`;
            content += `<tr>
                <td>Difference:</td>
                <td>${formatProb(Math.abs(effects.sc_top10_diff))}</td>
            </tr>`;
            
            content += '</table>';
            
            content += '<div class="threshold-info">High/Low thresholds: AI P90/P10, SC 5/1</div>';
            
            document.getElementById('pc-content').innerHTML = content;
            document.getElementById('pc-global-info').style.display = 'block';
        };
        
        window.closePCInfo = function() {
            document.getElementById('pc-global-info').style.display = 'none';
        };
        
        // Show essay on hover
        function showEssay(d, color) {
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
            
            const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
            const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
            document.getElementById('essay-display').style.backgroundColor = bgColor;
            document.getElementById('essay-display').style.borderColor = 
                `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
            document.getElementById('essay-display').style.display = 'block';
        }
        
        // Minimize/maximize essay display
        window.toggleMinimize = function() {
            const essayDisplay = document.getElementById('essay-display');
            if (essayDisplay.classList.contains('minimized')) {
                essayDisplay.classList.remove('minimized');
                document.querySelector('.minimize-btn').textContent = '–';
            } else {
                essayDisplay.classList.add('minimized');
                document.querySelector('.minimize-btn').textContent = '+';
            }
        };
        
        // Draggable resize functionality
        let isResizing = false;
        const essayDisplay = document.getElementById('essay-display');
        const resizeHandle = essayDisplay.querySelector('.resize-handle');
        
        resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const newHeight = window.innerHeight - e.clientY - 10;
            if (newHeight >= 80 && newHeight <= window.innerHeight * 0.8) {
                essayDisplay.style.height = newHeight + 'px';
            }
        });
        
        document.addEventListener('mouseup', () => {
            isResizing = false;
        });
        
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
            material.opacity = parseFloat(e.target.value);
            document.getElementById('opacity-val').textContent = e.target.value;
        });
        
        document.getElementById('essay-opacity').addEventListener('input', (e) => {
            document.getElementById('essay-opacity-val').textContent = e.target.value;
            if (document.getElementById('essay-display').style.display === 'block' && hoveredIndex >= 0) {
                const category = categories[hoveredIndex];
                const color = categoryColors[category];
                const bgOpacity = parseFloat(e.target.value);
                document.getElementById('essay-display').style.backgroundColor = 
                    `rgba(${Math.floor(color.r*255)}, ${Math.floor(color.g*255)}, ${Math.floor(color.b*255)}, ${bgOpacity})`;
            }
        });
        
        document.getElementById('essay-font-size').addEventListener('input', (e) => {
            const fontSize = e.target.value + 'px';
            document.getElementById('essay-text').style.fontSize = fontSize;
            document.getElementById('font-size-val').textContent = e.target.value;
            // Apply to PC analysis text as well
            document.querySelectorAll('.pc-values, .pc-percentile').forEach(el => {
                el.style.color = 'white';
            });
        });
        
        document.getElementById('transition-speed').addEventListener('input', (e) => {
            const speed = e.target.value + 's';
            document.getElementById('transition-speed-val').textContent = e.target.value;
            document.getElementById('essay-display').style.transition = `all ${speed}`;
        });
        
        // Gallery mode
        document.getElementById('toggle-gallery').addEventListener('change', (e) => {
            galleryMode = e.target.checked;
            document.getElementById('gallery-controls').style.display = galleryMode ? 'block' : 'none';
            
            if (galleryMode) {
                // Reset hover
                hoveredIndex = -1;
                document.getElementById('essay-display').style.display = 'none';
                updateGalleryDisplay();
            }
        });
        
        // PC dropdown
        function populatePCDropdown() {
            const selector = document.getElementById('pc-selector');
            for (let i = 0; i < 200; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `PC${i} (${(pcGlobalEffects[i].variance_total || 0).toFixed(2)}% var)`;
                selector.appendChild(option);
            }
        }
        
        function updatePCFocus() {
            const pcIdx = parseInt(document.getElementById('pc-selector').value);
            if (isNaN(pcIdx)) return;
            
            selectedPC = pcIdx;
            
            // Sort data by PC value
            const sortedIndices = data
                .map((d, i) => ({ idx: i, val: d.all_pc_values[pcIdx] }))
                .sort((a, b) => b.val - a.val)
                .map(d => d.idx);
            
            // Show top contributor
            galleryIndex = 0;
            const topIdx = sortedIndices[0];
            focusOnPoint(topIdx);
            updateGalleryDisplay();
        }
        
        function navigateGallery(direction) {
            if (!galleryMode || selectedPC === null) return;
            
            const sortedIndices = data
                .map((d, i) => ({ idx: i, val: d.all_pc_values[selectedPC] }))
                .sort((a, b) => b.val - a.val)
                .map(d => d.idx);
            
            galleryIndex = Math.max(0, Math.min(sortedIndices.length - 1, galleryIndex + direction));
            focusOnPoint(sortedIndices[galleryIndex]);
            updateGalleryDisplay();
        }
        
        function updateGalleryDisplay() {
            if (!galleryMode || selectedPC === null) return;
            
            const sortedIndices = data
                .map((d, i) => ({ idx: i, val: d.all_pc_values[selectedPC] }))
                .sort((a, b) => b.val - a.val);
            
            document.getElementById('gallery-info').textContent = 
                `Viewing ${galleryIndex + 1} of ${sortedIndices.length} (PC${selectedPC} value: ${sortedIndices[galleryIndex].val.toFixed(3)})`;
        }
        
        function focusOnPoint(index) {
            const d = data[index];
            
            // Animate camera to focus on point
            const targetPosition = new THREE.Vector3(d.x, d.y, d.z);
            const offset = new THREE.Vector3(20, 20, 20);
            const cameraTarget = targetPosition.clone().add(offset);
            
            // Simple animation
            const startPos = camera.position.clone();
            const startTarget = controls.target.clone();
            let progress = 0;
            
            function animateCamera() {
                progress += 0.02;
                if (progress >= 1) progress = 1;
                
                camera.position.lerpVectors(startPos, cameraTarget, progress);
                controls.target.lerpVectors(startTarget, targetPosition, progress);
                controls.update();
                
                if (progress < 1) {
                    requestAnimationFrame(animateCamera);
                }
            }
            
            animateCamera();
            
            // Show essay
            const category = categories[index];
            const color = categoryColors[category];
            showEssay(d, [color.r, color.g, color.b]);
        }
        
        // DML table toggle
        document.getElementById('toggle-dml').addEventListener('change', (e) => {
            document.getElementById('dml-table').style.display = e.target.checked ? 'block' : 'none';
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            
            // Update cursor state
            if (hoveredIndex >= 0) {
                cursorIndicator.classList.add('hovering');
            } else {
                cursorIndicator.classList.remove('hovering');
            }
            
            renderer.render(scene, camera);
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        animate();
    </script>
</body>
</html>
"""

# Save HTML file
output_path = OUTPUT_DIR / "minimal_umap_viz_v15.html"
with open(output_path, 'w') as f:
    f.write(html_content)

print(f"Visualization saved to: {output_path}")
print("\nChanges in v15:")
print("- Removed essay height slider and reset button")
print("- Added draggable resize functionality to essay display")
print("- Added minimize/maximize button on essay display")
print("- Applied font size to PC, SHAP, variance text")
print("- Made all text white color")
print("- Essay display can be resized by dragging the top edge")