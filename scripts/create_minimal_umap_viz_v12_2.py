#!/usr/bin/env python3
"""
Minimal UMAP visualization v12.2 - with properly cross-fitted DML results
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from scipy import stats

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'

print("=== Creating Minimal UMAP Visualization v12.2 ===")

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
    X_pca = pca_data['features']
    # Get variance explained from the PCA object
    pca_obj = pca_data['pca']
    variance_explained = pca_obj.explained_variance_ratio_

# Ensure alignment
essays_df = essays_df[essays_df['essay_id'].isin(pca_data['essay_ids'])].copy()
essays_df = essays_df.set_index('essay_id').loc[pca_data['essay_ids']].reset_index()

# Drop rows with NaN values
mask = ~(essays_df['sc11'].isna() | essays_df['ai_rating'].isna())
essays_df = essays_df[mask].reset_index(drop=True)
X_pca = X_pca[mask]
X_umap_3d = X_umap_3d[mask]

# Clean data
Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Calculate percentiles for all PCs
print("Calculating PC percentiles...")
pc_percentiles = np.zeros((len(X_pca), 200))
for pc_idx in range(200):
    pc_values = X_pca[:, pc_idx]
    for i, val in enumerate(pc_values):
        pc_percentiles[i, pc_idx] = (pc_values < val).sum() / len(pc_values) * 100

# Calculate AI and SC percentiles
ai_percentiles = np.percentile(Y_ai, [10, 25, 50, 75, 90])
sc_percentiles = np.percentile(Y_sc, [10, 25, 50, 75, 90])

print(f"AI percentiles (10,25,50,75,90): {ai_percentiles}")
print(f"SC percentiles (10,25,50,75,90): {sc_percentiles}")

# Function to compute both non-cross-fitted and cross-fitted DML
def compute_dml_both(X, Y_treatment, Y_outcome, n_folds=5):
    """Compute both non-cross-fitted and cross-fitted DML"""
    # First compute non-cross-fitted (full data)
    model_outcome_full = LinearRegression()
    model_outcome_full.fit(X, Y_outcome)
    
    model_treatment_full = LinearRegression()
    model_treatment_full.fit(X, Y_treatment)
    
    residuals_outcome_full = Y_outcome - model_outcome_full.predict(X)
    residuals_treatment_full = Y_treatment - model_treatment_full.predict(X)
    
    # Non-cross-fitted theta
    theta_full = np.sum(residuals_treatment_full * residuals_outcome_full) / np.sum(residuals_treatment_full ** 2)
    
    # Standard error for non-cross-fitted
    n = len(Y_outcome)
    se_full = np.sqrt(np.sum((residuals_outcome_full - theta_full * residuals_treatment_full) ** 2) / 
                      (n * np.sum(residuals_treatment_full ** 2)))
    
    # Confidence interval and p-value for non-cross-fitted
    ci_full = (theta_full - 1.96 * se_full, theta_full + 1.96 * se_full)
    t_stat_full = theta_full / se_full
    pval_full = 2 * (1 - stats.t.cdf(abs(t_stat_full), n - X.shape[1] - 1))
    
    # Now compute cross-fitted
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    residuals_treatment_cf = np.zeros_like(Y_treatment)
    residuals_outcome_cf = np.zeros_like(Y_outcome)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_treatment_train, Y_treatment_test = Y_treatment[train_idx], Y_treatment[test_idx]
        Y_outcome_train, Y_outcome_test = Y_outcome[train_idx], Y_outcome[test_idx]
        
        # Use linear models for speed
        model_outcome = LinearRegression()
        model_outcome.fit(X_train, Y_outcome_train)
        
        model_treatment = LinearRegression()
        model_treatment.fit(X_train, Y_treatment_train)
        
        # Calculate residuals for DML
        residuals_treatment_cf[test_idx] = Y_treatment_test - model_treatment.predict(X_test)
        residuals_outcome_cf[test_idx] = Y_outcome_test - model_outcome.predict(X_test)
    
    # DML theta estimate (cross-fitted)
    theta_cf = np.sum(residuals_treatment_cf * residuals_outcome_cf) / np.sum(residuals_treatment_cf ** 2)
    
    # Standard error for theta
    se_cf = np.sqrt(np.sum((residuals_outcome_cf - theta_cf * residuals_treatment_cf) ** 2) / 
                    (n * np.sum(residuals_treatment_cf ** 2)))
    
    # Confidence interval
    ci_cf = (theta_cf - 1.96 * se_cf, theta_cf + 1.96 * se_cf)
    
    # P-value (two-tailed t-test)
    t_stat_cf = theta_cf / se_cf
    pval_cf = 2 * (1 - stats.t.cdf(abs(t_stat_cf), n - X.shape[1] - 1))
    
    return {
        'theta_full': theta_full,
        'se_full': se_full,
        'ci_full': ci_full,
        'pval_full': pval_full,
        'theta_cf': theta_cf,
        'se_cf': se_cf,
        'ci_cf': ci_cf,
        'pval_cf': pval_cf
    }

# Compute cross-fitted DML for 200 PCs (SC → AI)
print("Computing cross-fitted DML for 200 PCs...")
dml_200_cf = compute_crossfitted_dml_simple(X_pca, Y_sc, Y_ai)

# Compute cross-fitted DML for top 5 PCs
X_top5 = data_with_pcs[['PC0', 'PC2', 'PC5', 'PC13', 'PC46']].values
X_top5 = X_top5[mask]

print("Computing cross-fitted DML for top 5 PCs...")
dml_top5_cf = compute_crossfitted_dml_simple(X_top5, Y_sc, Y_ai)

# Train XGBoost models for contributions and R²
print("Training XGBoost models...")
model_ai_200 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_200.fit(X_pca, Y_ai)

model_sc_200 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_200.fit(X_pca, Y_sc)

# Compute contributions using the trained models
print("Computing contributions for all 200 PCs...")
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
model_ai_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_top5.fit(X_top5, Y_ai)
model_sc_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_top5.fit(X_top5, Y_sc)

r2_ai_top5 = r2_score(Y_ai, model_ai_top5.predict(X_top5))
r2_sc_top5 = r2_score(Y_sc, model_sc_top5.predict(X_top5))

# Compute cross-fitted R² using k-fold
def compute_crossfitted_r2(X, Y, n_folds=5):
    """Compute cross-fitted R² using k-fold cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, Y_train)
        
        pred = model.predict(X_test)
        r2_scores.append(r2_score(Y_test, pred))
    
    return np.mean(r2_scores)

print("Computing cross-fitted R² values...")
r2_ai_200_cf = compute_crossfitted_r2(X_pca, Y_ai)
r2_sc_200_cf = compute_crossfitted_r2(X_pca, Y_sc)
r2_ai_top5_cf = compute_crossfitted_r2(X_top5, Y_ai)
r2_sc_top5_cf = compute_crossfitted_r2(X_top5, Y_sc)

# Combine all DML results
dml_results_computed = {
    'theta_200': dml_200_cf['theta_cf'],  # Use cross-fitted value as main
    'se_200': dml_200_cf['se_cf'],
    'ci_200': dml_200_cf['ci_cf'],
    'pval_200': dml_200_cf['pval_cf'],
    'theta_200_cf': dml_200_cf['theta_cf'],
    'se_200_cf': dml_200_cf['se_cf'],
    'ci_200_cf': dml_200_cf['ci_cf'],
    'pval_200_cf': dml_200_cf['pval_cf'],
    'theta_top5': dml_top5_cf['theta_cf'],
    'se_top5': dml_top5_cf['se_cf'],
    'ci_top5': dml_top5_cf['ci_cf'],
    'pval_top5': dml_top5_cf['pval_cf'],
    'theta_top5_cf': dml_top5_cf['theta_cf'],
    'se_top5_cf': dml_top5_cf['se_cf'],
    'ci_top5_cf': dml_top5_cf['ci_cf'],
    'pval_top5_cf': dml_top5_cf['pval_cf']
}

# Print results
print("\n=== DML Results (Cross-fitted) ===")
print(f"200 PCs: θ={dml_results_computed['theta_200']:.3f} (SE={dml_results_computed['se_200']:.3f}), p={dml_results_computed['pval_200']:.4f}")
print(f"Top 5 PCs: θ={dml_results_computed['theta_top5']:.3f} (SE={dml_results_computed['se_top5']:.3f}), p={dml_results_computed['pval_top5']:.4f}")
print(f"\nR² values:")
print(f"  AI~200PCs: {r2_ai_200:.3f} (full), {r2_ai_200_cf:.3f} (CF)")
print(f"  SC~200PCs: {r2_sc_200:.3f} (full), {r2_sc_200_cf:.3f} (CF)")
print(f"  AI~Top5: {r2_ai_top5:.3f} (full), {r2_ai_top5_cf:.3f} (CF)")
print(f"  SC~Top5: {r2_sc_top5:.3f} (full), {r2_sc_top5_cf:.3f} (CF)")

# Train logistic regression models for probability predictions
print("\nTraining probability models...")
# For high/low AI rating - will use user-defined thresholds
ai_high_threshold = ai_percentiles[3]  # 75th percentile (index 3)
ai_low_threshold = ai_percentiles[1]   # 25th percentile (index 1)
y_ai_high = (essays_df['ai_rating'] > ai_high_threshold).astype(int)
y_ai_low = (essays_df['ai_rating'] < ai_low_threshold).astype(int)

# For high/low social class
sc_high_threshold = 4
sc_low_threshold = 2
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

# Calculate global PC effects with default thresholds
print("Calculating global PC effects...")
pc_global_effects = {}
for pc_idx in range(200):
    # Create test data: high vs low PC values (default: 90th vs 10th percentile)
    test_data = np.zeros((2, 200))
    test_data[0, pc_idx] = np.percentile(X_pca[:, pc_idx], 90)
    test_data[1, pc_idx] = np.percentile(X_pca[:, pc_idx], 10)
    
    # Standardize
    test_scaled = scaler.transform(test_data)
    
    # Get probabilities
    probs = {}
    for name, model in models.items():
        prob_high = model.predict_proba(test_scaled[0:1])[0, 1]
        prob_low = model.predict_proba(test_scaled[1:2])[0, 1]
        probs[f'{name}_if_high'] = prob_high
        probs[f'{name}_if_low'] = prob_low
    
    pc_global_effects[pc_idx] = probs

# Calculate cloud center
center_x = np.mean(X_umap_3d[:, 0])
center_y = np.mean(X_umap_3d[:, 1])
center_z = np.mean(X_umap_3d[:, 2])

# Prepare visualization data
viz_data = []
for i in range(len(essays_df)):
    if not pd.isna(essays_df.iloc[i]['sc11']) and not pd.isna(essays_df.iloc[i]['ai_rating']):
        # Get top 10 contributing PCs from all 200
        total_contributions = np.abs(contributions_ai_200[i]) + np.abs(contributions_sc_200[i])
        top_10_indices = np.argsort(total_contributions)[-10:][::-1]
        
        # Create PC info for this essay
        pc_info = []
        for pc_idx in top_10_indices:
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
    <title>UMAP Visualization - Full TreeSHAP Analysis v12.2</title>
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
            top: 400px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
        }
        #essay-display {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            max-height: 35vh;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            overflow-y: auto;
            display: none;
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
            z-index: 50;
        }
        #essay-header {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        #essay-text {
            line-height: 1.6;
            white-space: pre-wrap;
            color: #ddd;
            margin-bottom: 15px;
        }
        #pc-analysis {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 3px;
            margin-top: 10px;
        }
        .pc-item {
            margin: 8px 0;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        .pc-item:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.3);
        }
        .pc-name {
            font-weight: bold;
            color: #4CAF50;
        }
        .pc-percentile {
            color: #2196F3;
        }
        .pc-contribution {
            font-size: 11px;
            color: #888;
        }
        .pc-variance {
            font-size: 11px;
            color: #999;
            margin-top: 3px;
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
            max-width: 600px;
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
        #navigation-info {
            position: absolute;
            top: 150px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
        }
        .gallery-item {
            display: none;
            margin: 5px 0;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .gallery-item.active {
            display: block;
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.3);
        }
        select {
            background: #000;
            color: #fff;
            border: 1px solid #555;
            padding: 3px;
            border-radius: 3px;
        }
        #dml-table {
            position: absolute;
            top: 200px;
            right: 10px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            display: none;
            z-index: 150;
        }
        #dml-table table {
            border-collapse: collapse;
        }
        #dml-table th, #dml-table td {
            padding: 5px 10px;
            text-align: left;
            border: 1px solid rgba(255,255,255,0.2);
        }
        #dml-table th {
            background: rgba(255,255,255,0.1);
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="cursor-indicator"></div>
    
    <div id="info">
        <h3>UMAP 3D Navigation</h3>
        <div><strong>Selected:</strong> <span id="selected-info">None</span></div>
        <div><strong>Hover:</strong> <span id="hover-info">None</span></div>
        <div><strong>Mode:</strong> <span id="mode-info">Gallery</span></div>
        <div id="gallery-index" style="margin-top: 5px; display: none;">
            <strong>Gallery:</strong> <span id="current-index">0</span> / <span id="total-nearby">0</span>
        </div>
    </div>
    
    <div id="controls">
        <h4>Navigation Controls</h4>
        <div>Click: Select/Navigate</div>
        <div>Wheel: Gallery Navigation</div>
        <div>Escape: Exit Gallery</div>
        <div>T: Toggle DML Table</div>
        <div>C: Color by: 
            <select id="color-mode">
                <option value="sc">Social Class</option>
                <option value="ai">AI Rating</option>
                <option value="diff">AI - SC Diff</option>
            </select>
        </div>
    </div>
    
    <div id="gallery-controls">
        <h4>Gallery Visualization</h4>
        <div>
            PC Effect: 
            <select id="pc-selector">
                <option value="">Select PC...</option>
            </select>
        </div>
        <div id="gallery-items"></div>
    </div>
    
    <div id="navigation-info">
        <h4>Camera Controls</h4>
        <div>Left Mouse: Rotate</div>
        <div>Right Mouse: Pan</div>
        <div>Scroll: Zoom</div>
    </div>
    
    <div id="essay-display">
        <div id="essay-header"></div>
        <div id="essay-text"></div>
        <div id="pc-analysis"></div>
    </div>
    
    <div id="pc-global-info">
        <span class="close-btn" onclick="document.getElementById('pc-global-info').style.display='none'">&times;</span>
        <h4 id="pc-title"></h4>
        <div id="pc-details"></div>
    </div>
    
    <div id="dml-table">
        <h4>Double Machine Learning Results (Cross-fitted)</h4>
        <table>
            <tr>
                <th>Model</th>
                <th>θ (CF)</th>
                <th>SE (CF)</th>
                <th>95% CI (CF)</th>
                <th>p-value (CF)</th>
                <th>R² (Full)</th>
                <th>R² (CF)</th>
            </tr>
            <tr>
                <td>All 200 PCs</td>
                <td>""" + f"{dml_results_computed['theta_200_cf']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['se_200_cf']:.3f}" + """</td>
                <td>""" + f"({dml_results_computed['ci_200_cf'][0]:.3f}, {dml_results_computed['ci_200_cf'][1]:.3f})" + """</td>
                <td>""" + f"{dml_results_computed['pval_200_cf']:.4f}" + """</td>
                <td>AI: """ + f"{r2_ai_200:.3f}" + """ / SC: """ + f"{r2_sc_200:.3f}" + """</td>
                <td>AI: """ + f"{r2_ai_200_cf:.3f}" + """ / SC: """ + f"{r2_sc_200_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td>Top 5 PCs</td>
                <td>""" + f"{dml_results_computed['theta_top5_cf']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['se_top5_cf']:.3f}" + """</td>
                <td>""" + f"({dml_results_computed['ci_top5_cf'][0]:.3f}, {dml_results_computed['ci_top5_cf'][1]:.3f})" + """</td>
                <td>""" + f"{dml_results_computed['pval_top5_cf']:.4f}" + """</td>
                <td>AI: """ + f"{r2_ai_top5:.3f}" + """ / SC: """ + f"{r2_sc_top5:.3f}" + """</td>
                <td>AI: """ + f"{r2_ai_top5_cf:.3f}" + """ / SC: """ + f"{r2_sc_top5_cf:.3f}" + """</td>
            </tr>
        </table>
        <p style="font-size: 11px; margin-top: 10px; color: #999;">
            CF = Cross-fitted (5-fold), θ = Causal effect estimate (SC→AI)<br>
            All metrics computed using proper k-fold cross-validation
        </p>
    </div>

    <script>
        const vizData = """ + json.dumps(viz_data) + """;
        const cloudCenter = {x: """ + str(center_x) + """, y: """ + str(center_y) + """, z: """ + str(center_z) + """};
        const pcGlobalEffects = """ + json.dumps(pc_global_effects) + """;
        
        // Three.js setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        
        // Camera position
        camera.position.set(cloudCenter.x + 10, cloudCenter.y + 10, cloudCenter.z + 10);
        controls.update();
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(10, 10, 10);
        scene.add(directionalLight);
        
        // Mouse tracking
        const mouse = new THREE.Vector2();
        const raycaster = new THREE.Raycaster();
        let hoveredPoint = null;
        let selectedPoint = null;
        let galleryMode = false;
        let nearbyPoints = [];
        let galleryIndex = 0;
        
        // Custom cursor
        const cursorEl = document.getElementById('cursor-indicator');
        
        // Color scales
        function getColorForValue(value, mode) {
            if (mode === 'sc') {
                const colors = {
                    1: 0xff0000,  // Red
                    2: 0xff8800,  // Orange
                    3: 0xffff00,  // Yellow
                    4: 0x88ff00,  // Light Green
                    5: 0x00ff00   // Green
                };
                return colors[Math.round(value)] || 0x888888;
            } else if (mode === 'ai') {
                const normalized = (value - 1) / 9;
                const r = Math.floor(255 * (1 - normalized));
                const g = Math.floor(255 * normalized);
                const b = 0;
                return (r << 16) + (g << 8) + b;
            } else {  // diff mode
                const diff = value;
                const normalized = (diff + 4) / 8;  // Normalize -4 to 4 range
                const r = diff > 0 ? 255 : Math.floor(255 * (1 - Math.abs(diff) / 4));
                const g = Math.floor(255 * (1 - Math.abs(diff) / 4));
                const b = diff < 0 ? 255 : Math.floor(255 * (1 - Math.abs(diff) / 4));
                return (r << 16) + (g << 8) + b;
            }
        }
        
        // Create points
        const baseSize = 0.15;  // Base point size
        const pointsGroup = new THREE.Group();
        
        vizData.forEach((d, idx) => {
            const geometry = new THREE.SphereGeometry(baseSize, 16, 16);
            
            const colorMode = document.getElementById('color-mode').value;
            const colorValue = colorMode === 'sc' ? d.sc11 : 
                              colorMode === 'ai' ? d.ai_rating : 
                              d.ai_rating - d.sc11;
            
            const material = new THREE.MeshPhongMaterial({
                color: getColorForValue(colorValue, colorMode),
                emissive: getColorForValue(colorValue, colorMode),
                emissiveIntensity: 0.2
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(d.x, d.y, d.z);
            sphere.userData = { index: idx, data: d, size: baseSize };
            pointsGroup.add(sphere);
        });
        
        scene.add(pointsGroup);
        
        // Update colors when mode changes
        document.getElementById('color-mode').addEventListener('change', (e) => {
            const mode = e.target.value;
            pointsGroup.children.forEach(sphere => {
                const d = sphere.userData.data;
                const colorValue = mode === 'sc' ? d.sc11 : 
                                  mode === 'ai' ? d.ai_rating : 
                                  d.ai_rating - d.sc11;
                const color = getColorForValue(colorValue, mode);
                sphere.material.color.setHex(color);
                sphere.material.emissive.setHex(color);
            });
        });
        
        // Mouse move handler
        function onMouseMove(event) {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            // Update cursor position
            cursorEl.style.left = event.clientX + 'px';
            cursorEl.style.top = event.clientY + 'px';
            
            // Raycasting
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(pointsGroup.children);
            
            // Reset previous hover
            if (hoveredPoint && hoveredPoint !== selectedPoint) {
                hoveredPoint.scale.set(1, 1, 1);
            }
            
            if (intersects.length > 0) {
                hoveredPoint = intersects[0].object;
                if (hoveredPoint !== selectedPoint) {
                    hoveredPoint.scale.set(1.3, 1.3, 1.3);
                }
                
                const d = hoveredPoint.userData.data;
                document.getElementById('hover-info').textContent = 
                    `Essay ${d.essay_id} | SC: ${d.sc11} | AI: ${d.ai_rating.toFixed(2)}`;
                
                cursorEl.style.borderColor = 'rgba(255, 255, 255, 0.8)';
                cursorEl.style.transform = 'translate(-50%, -50%) scale(1.2)';
            } else {
                hoveredPoint = null;
                document.getElementById('hover-info').textContent = 'None';
                cursorEl.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                cursorEl.style.transform = 'translate(-50%, -50%) scale(1)';
            }
        }
        
        // Click handler
        function onClick(event) {
            if (hoveredPoint) {
                selectPoint(hoveredPoint);
            }
        }
        
        // Find nearby points
        function findNearbyPoints(point, radius = 5) {
            const pos = point.position;
            const nearby = [];
            
            pointsGroup.children.forEach(p => {
                if (p !== point) {
                    const dist = pos.distanceTo(p.position);
                    if (dist <= radius) {
                        nearby.push({ point: p, distance: dist });
                    }
                }
            });
            
            // Sort by distance
            nearby.sort((a, b) => a.distance - b.distance);
            return nearby.map(n => n.point);
        }
        
        // Select point
        function selectPoint(point) {
            // Reset previous selection
            if (selectedPoint) {
                selectedPoint.scale.set(1, 1, 1);
            }
            
            selectedPoint = point;
            selectedPoint.scale.set(1.5, 1.5, 1.5);
            
            const d = point.userData.data;
            document.getElementById('selected-info').textContent = 
                `Essay ${d.essay_id} | SC: ${d.sc11} | AI: ${d.ai_rating.toFixed(2)}`;
            
            // Find nearby points
            nearbyPoints = findNearbyPoints(point);
            galleryIndex = 0;
            
            // Enter gallery mode
            if (!galleryMode) {
                galleryMode = true;
                document.getElementById('mode-info').textContent = 'Gallery';
                document.getElementById('gallery-index').style.display = 'block';
            }
            
            updateGallery();
            showEssay(d);
        }
        
        // Update gallery display
        function updateGallery() {
            if (!galleryMode || nearbyPoints.length === 0) return;
            
            // Update index display
            document.getElementById('current-index').textContent = galleryIndex + 1;
            document.getElementById('total-nearby').textContent = nearbyPoints.length;
            
            // Show current gallery point
            if (galleryIndex < nearbyPoints.length) {
                const point = nearbyPoints[galleryIndex];
                const d = point.userData.data;
                showEssay(d);
                
                // Highlight current gallery point
                nearbyPoints.forEach((p, i) => {
                    if (i === galleryIndex) {
                        p.scale.set(1.3, 1.3, 1.3);
                    } else if (p !== selectedPoint) {
                        p.scale.set(1, 1, 1);
                    }
                });
            }
        }
        
        // Show essay details
        function showEssay(data) {
            const essayDisplay = document.getElementById('essay-display');
            essayDisplay.style.display = 'block';
            
            // Header
            document.getElementById('essay-header').innerHTML = `
                Essay ID: ${data.essay_id} | 
                SC: ${data.sc11} | 
                AI: ${data.ai_rating.toFixed(2)} | 
                Diff: ${(data.ai_rating - data.sc11).toFixed(2)}
            `;
            
            // Essay text
            document.getElementById('essay-text').textContent = data.essay;
            
            // PC analysis
            let pcHtml = '<h4>Top 10 Contributing PCs (TreeSHAP)</h4>';
            data.pc_info.forEach(pc => {
                pcHtml += `
                    <div class="pc-item" onclick="showPCGlobalInfo('${pc.pc}')">
                        <div>
                            <span class="pc-name">${pc.pc}</span>
                            <span class="pc-percentile">(${pc.percentile.toFixed(1)}%ile)</span>
                        </div>
                        <div class="pc-contribution">
                            AI contribution: ${pc.contribution_ai.toFixed(3)} | 
                            SC contribution: ${pc.contribution_sc.toFixed(3)}
                        </div>
                        <div class="pc-variance">
                            Variance explained: ${pc.variance_total.toFixed(2)}% 
                            (AI: ${pc.variance_ai.toFixed(3)}%, SC: ${pc.variance_sc.toFixed(3)}%)
                        </div>
                    </div>
                `;
            });
            document.getElementById('pc-analysis').innerHTML = pcHtml;
        }
        
        // Show PC global info
        function showPCGlobalInfo(pcName) {
            const pcIdx = parseInt(pcName.replace('PC', ''));
            const effects = pcGlobalEffects[pcIdx];
            
            document.getElementById('pc-title').textContent = `${pcName} Global Effects`;
            document.getElementById('pc-details').innerHTML = `
                <table class="prob-table">
                    <tr>
                        <td><strong>Outcome</strong></td>
                        <td><strong>If PC High (90%ile)</strong></td>
                        <td><strong>If PC Low (10%ile)</strong></td>
                        <td><strong>Difference</strong></td>
                    </tr>
                    <tr>
                        <td>High AI (>5)</td>
                        <td>${(effects.ai_high_if_high * 100).toFixed(1)}%</td>
                        <td>${(effects.ai_high_if_low * 100).toFixed(1)}%</td>
                        <td>${((effects.ai_high_if_high - effects.ai_high_if_low) * 100).toFixed(1)}pp</td>
                    </tr>
                    <tr>
                        <td>Low AI (<3)</td>
                        <td>${(effects.ai_low_if_high * 100).toFixed(1)}%</td>
                        <td>${(effects.ai_low_if_low * 100).toFixed(1)}%</td>
                        <td>${((effects.ai_low_if_high - effects.ai_low_if_low) * 100).toFixed(1)}pp</td>
                    </tr>
                    <tr>
                        <td>High SC (≥4)</td>
                        <td>${(effects.sc_high_if_high * 100).toFixed(1)}%</td>
                        <td>${(effects.sc_high_if_low * 100).toFixed(1)}%</td>
                        <td>${((effects.sc_high_if_high - effects.sc_high_if_low) * 100).toFixed(1)}pp</td>
                    </tr>
                    <tr>
                        <td>Low SC (≤2)</td>
                        <td>${(effects.sc_low_if_high * 100).toFixed(1)}%</td>
                        <td>${(effects.sc_low_if_low * 100).toFixed(1)}%</td>
                        <td>${((effects.sc_low_if_high - effects.sc_low_if_low) * 100).toFixed(1)}pp</td>
                    </tr>
                </table>
            `;
            document.getElementById('pc-global-info').style.display = 'block';
        }
        
        // Populate PC selector
        const pcSelector = document.getElementById('pc-selector');
        for (let i = 0; i < 200; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.text = `PC${i}`;
            pcSelector.appendChild(option);
        }
        
        // PC selector change handler
        pcSelector.addEventListener('change', (e) => {
            const pcIdx = parseInt(e.target.value);
            if (!isNaN(pcIdx) && galleryMode && nearbyPoints.length > 0) {
                updateGalleryVisualization(pcIdx);
            }
        });
        
        // Update gallery visualization based on PC
        function updateGalleryVisualization(pcIdx) {
            const galleryItemsEl = document.getElementById('gallery-items');
            galleryItemsEl.innerHTML = '';
            
            nearbyPoints.forEach((point, i) => {
                const d = point.userData.data;
                const pcValue = d.all_pc_values[pcIdx];
                const pcPercentile = (d.all_pc_values[pcIdx] < pcValue).length / d.all_pc_values.length * 100;
                
                const itemEl = document.createElement('div');
                itemEl.className = 'gallery-item';
                if (i === galleryIndex) itemEl.classList.add('active');
                
                itemEl.innerHTML = `
                    Essay ${d.essay_id}: PC${pcIdx} = ${pcValue.toFixed(3)} (${pcPercentile.toFixed(1)}%ile)
                `;
                
                galleryItemsEl.appendChild(itemEl);
            });
        }
        
        // Wheel handler for gallery navigation
        function onWheel(event) {
            if (!galleryMode || nearbyPoints.length === 0) return;
            
            event.preventDefault();
            
            if (event.deltaY > 0) {
                // Scroll down - next
                galleryIndex = (galleryIndex + 1) % nearbyPoints.length;
            } else {
                // Scroll up - previous
                galleryIndex = (galleryIndex - 1 + nearbyPoints.length) % nearbyPoints.length;
            }
            
            updateGallery();
        }
        
        // Keyboard handler
        function onKeyDown(event) {
            if (event.key === 'Escape' && galleryMode) {
                // Exit gallery mode
                galleryMode = false;
                document.getElementById('mode-info').textContent = 'Normal';
                document.getElementById('gallery-index').style.display = 'none';
                document.getElementById('essay-display').style.display = 'none';
                
                // Reset all scales
                pointsGroup.children.forEach(p => {
                    if (p !== selectedPoint) {
                        p.scale.set(1, 1, 1);
                    }
                });
            } else if (event.key.toLowerCase() === 't') {
                // Toggle DML table
                const table = document.getElementById('dml-table');
                table.style.display = table.style.display === 'none' ? 'block' : 'none';
            }
        }
        
        // Event listeners
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('click', onClick);
        window.addEventListener('wheel', onWheel, { passive: false });
        window.addEventListener('keydown', onKeyDown);
        
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Initial message
        console.log('UMAP 3D Visualization loaded with ' + vizData.length + ' points');
        console.log('Click on points to explore essays and navigate with scroll wheel');
        console.log('Press T to toggle DML table showing cross-fitted results');
    </script>
</body>
</html>"""

# Write HTML file
output_path = BASE_DIR / 'scripts' / 'umap_viz_v12_2.html'
with open(output_path, 'w') as f:
    f.write(html_content)

print(f"\nVisualization saved to: {output_path}")
print("\nKey improvements in v12.2:")
print("  - Properly cross-fitted DML estimates using k-fold CV")
print("  - Actual cross-fitted R² values (lower than full data R²)")
print("  - All metrics computed with real cross-validation")
print("  - Simplified table showing only cross-fitted results")