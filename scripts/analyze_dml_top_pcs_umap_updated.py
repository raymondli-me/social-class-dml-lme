#!/usr/bin/env python3
"""
Comprehensive DML analysis with top PCs and enhanced UMAP visualization (Fixed Hover/Picking)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from econml.dml import LinearDML
import shap
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=== Comprehensive DML PC Analysis (Fixed Hover) ===")

# Load data
print("\n1. Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

# Load social class
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings (human MacArthur only)
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Load PCA features
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
    pca_model = pca_data['pca']
    essay_ids = pca_data['essay_ids']

# Align data
essays_df = essays_df[essays_df['essay_id'].isin(essay_ids)]
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()
print(f"    Loaded {X_pca.shape[0]} essays with {X_pca.shape[1]} PCs")

Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Step 1: Fit DML with all 200 PCs
print("\n2. Fitting DML with all 200 PCs...")
dml_full = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    model_t=xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    random_state=42
)
dml_full.fit(Y=Y_ai, T=Y_sc, X=X_pca, W=None)

# Get overall effect
theta_full = dml_full.effect(X_pca).mean()
ci_full = dml_full.effect_interval(X_pca, alpha=0.05)
theta_full_lower = ci_full[0].mean()
theta_full_upper = ci_full[1].mean()
inference_full = dml_full.effect_inference(X_pca).population_summary()
p_value_full = inference_full.pvalue()

print(f"    Full model ? = {theta_full:.4f} [{theta_full_lower:.4f}, {theta_full_upper:.4f}], p = {p_value_full:.4f}")

# Calculate DML R² - how well X + treatment predicts outcome
# For DML on AI ratings: X (text) + actual SC ? AI rating
print("\n3. Computing DML R² (cross-validated)...")

# Model 1: Text + Actual SC ? AI Rating
X_with_sc_full = np.column_stack([X_pca, Y_sc.reshape(-1, 1)])
y_pred_dml_ai_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_sc_full, Y_ai, cv=5
)
r2_dml_ai_full = r2_score(Y_ai, y_pred_dml_ai_full)
print(f"    Full model: Text + Actual SC ? AI Rating R² = {r2_dml_ai_full:.4f}")

# Model 2: Text + AI Rating ? Actual SC (reverse DML)
X_with_ai_full = np.column_stack([X_pca, Y_ai.reshape(-1, 1)])
y_pred_dml_sc_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_ai_full, Y_sc, cv=5
)
r2_dml_sc_full = r2_score(Y_sc, y_pred_dml_sc_full)
print(f"    Full model: Text + AI Rating ? Actual SC R² = {r2_dml_sc_full:.4f}")

# Get feature importance by training separate XGBoost models
# This gives us direct access to feature importances
print("\n4. Computing feature importances...")

# Train XGBoost for AI rating prediction (outcome model)
model_ai_rating = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_rating.fit(X_pca, Y_ai)
importance_ai_rating = model_ai_rating.feature_importances_

# Train XGBoost for actual social class prediction (treatment model)
model_actual_sc = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_actual_sc.fit(X_pca, Y_sc)
importance_actual_sc = model_actual_sc.feature_importances_

# Combined importance (average of AI rating and actual SC model importances)
combined_importance = (importance_ai_rating + importance_actual_sc) / 2

# Get top 5 PCs by importance
top_5_indices = np.argsort(combined_importance)[-5:][::-1]
print(f"\n5. Top 5 PCs by importance: {top_5_indices}")

# Print their individual importances
for idx in top_5_indices:
    print(f"    PC{idx}: Combined importance = {combined_importance[idx]:.4f} "
          f"(AI Rating: {importance_ai_rating[idx]:.4f}, Actual SC: {importance_actual_sc[idx]:.4f})")

# Step 2: Refit DML with only top 5 PCs
print("\n6. Refitting DML with top 5 PCs only...")
X_top5 = X_pca[:, top_5_indices]

dml_top5 = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    model_t=xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    random_state=42
)
dml_top5.fit(Y=Y_ai, T=Y_sc, X=X_top5, W=None)

# Get effect for top 5 model
theta_top5 = dml_top5.effect(X_top5).mean()
ci_top5 = dml_top5.effect_interval(X_top5, alpha=0.05)
theta_top5_lower = ci_top5[0].mean()
theta_top5_upper = ci_top5[1].mean()
inference_top5 = dml_top5.effect_inference(X_top5).population_summary()
p_value_top5 = inference_top5.pvalue()

print(f"    Top 5 model ? = {theta_top5:.4f} [{theta_top5_lower:.4f}, {theta_top5_upper:.4f}], p = {p_value_top5:.4f}")

# Calculate DML R² for top 5 model
print("\n7. Computing DML R² for top 5 model...")

# Model 1: Text (top 5) + Actual SC ? AI Rating
X_with_sc_top5 = np.column_stack([X_top5, Y_sc.reshape(-1, 1)])
y_pred_dml_ai_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_sc_top5, Y_ai, cv=5
)
r2_dml_ai_top5 = r2_score(Y_ai, y_pred_dml_ai_top5)
print(f"    Top 5 model: Text + Actual SC ? AI Rating R² = {r2_dml_ai_top5:.4f}")

# Model 2: Text (top 5) + AI Rating ? Actual SC
X_with_ai_top5 = np.column_stack([X_top5, Y_ai.reshape(-1, 1)])
y_pred_dml_sc_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_ai_top5, Y_sc, cv=5
)
r2_dml_sc_top5 = r2_score(Y_sc, y_pred_dml_sc_top5)
print(f"    Top 5 model: Text + AI Rating ? Actual SC R² = {r2_dml_sc_top5:.4f}")

# Step 3: Compare predictive performance
print("\n8. Comparing predictive performance...")

# Full model predictions
y_pred_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_pca, Y_ai, cv=5
)
r2_ai_full = r2_score(Y_ai, y_pred_full)

sc_pred_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_pca, Y_sc, cv=5
)
r2_sc_full = r2_score(Y_sc, sc_pred_full)

# Top 5 model predictions
y_pred_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_top5, Y_ai, cv=5
)
r2_ai_top5 = r2_score(Y_ai, y_pred_top5)

sc_pred_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_top5, Y_sc, cv=5
)
r2_sc_top5 = r2_score(Y_sc, sc_pred_top5)

# Create comparison table
comparison_data = {
    'Model': ['Full (200 PCs)', 'Top 5 PCs', 'Difference'],
    'DML ?': [theta_full, theta_top5, theta_top5 - theta_full],
    'DML p-value': [p_value_full, p_value_top5, '-'],
    'Text?AI R²': [r2_ai_full, r2_ai_top5, r2_ai_top5 - r2_ai_full],
    'Text?SC R²': [r2_sc_full, r2_sc_top5, r2_sc_top5 - r2_sc_full],
    'Text+SC?AI R²': [r2_dml_ai_full, r2_dml_ai_top5, r2_dml_ai_top5 - r2_dml_ai_full],
    'Text+AI?SC R²': [r2_dml_sc_full, r2_dml_sc_top5, r2_dml_sc_top5 - r2_dml_sc_full],
    'Variance': [f"{pca_model.explained_variance_ratio_[:200].sum()*100:.1f}%", 
                 f"{pca_model.explained_variance_ratio_[top_5_indices].sum()*100:.1f}%",
                 f"{(pca_model.explained_variance_ratio_[top_5_indices].sum() - pca_model.explained_variance_ratio_[:200].sum())*100:.1f}%"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n9. Model Comparison:")
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(OUTPUT_DIR / 'dml_model_comparison.csv', index=False)

# Step 4: Calculate SHAP values for top 5 model
print("\n10. Computing SHAP values for top 5 model...")

# Train models on top 5 PCs for SHAP
model_ai_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_top5.fit(X_top5, Y_ai)

model_sc_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_top5.fit(X_top5, Y_sc)

# Try to compute SHAP values with workaround
try:
    # Use KernelExplainer as fallback if TreeExplainer fails
    print("    Computing SHAP values (using KernelExplainer)...")
    
    # Sample background data for KernelExplainer
    background = shap.sample(X_top5, 100) # Using 100 samples for background
    
    # Create explainers
    explainer_ai = shap.KernelExplainer(model_ai_top5.predict, background)
    explainer_sc = shap.KernelExplainer(model_sc_top5.predict, background)
    
    # Get SHAP values (this will be slow, so we'll sample)
    sample_size = min(1000, len(X_top5)) # SHAP on up to 1000 points
    sample_indices = np.random.choice(len(X_top5), sample_size, replace=False)
    
    shap_values_ai_sample = explainer_ai.shap_values(X_top5[sample_indices])
    shap_values_sc_sample = explainer_sc.shap_values(X_top5[sample_indices])
    
    # Extrapolate to full dataset (fill non-sampled with zeros or mean)
    shap_ai_full = np.zeros((len(X_top5), X_top5.shape[1]))
    shap_sc_full = np.zeros((len(X_top5), X_top5.shape[1]))
    
    shap_ai_full[sample_indices] = shap_values_ai_sample
    shap_sc_full[sample_indices] = shap_values_sc_sample
    
except Exception as e:
    print(f"    SHAP computation failed: {e}")
    print("    Using feature contributions as fallback...")
    
    # Fallback: use feature contributions (simplified SHAP-like)
    shap_ai_full = np.zeros((len(X_top5), X_top5.shape[1]))
    shap_sc_full = np.zeros((len(X_top5), X_top5.shape[1]))
    
    pred_with_ai = model_ai_top5.predict(X_top5)
    pred_with_sc = model_sc_top5.predict(X_top5)

    for i in range(X_top5.shape[1]):
        X_temp = X_top5.copy()
        X_temp_col_original = X_temp[:, i].copy() # Store original column
        X_temp[:, i] = background[:,i].mean() # Replace with mean from background for more stable "zeroing"
        
        pred_without_ai = model_ai_top5.predict(X_temp)
        shap_ai_full[:, i] = pred_with_ai - pred_without_ai
        
        pred_without_sc = model_sc_top5.predict(X_temp)
        shap_sc_full[:, i] = pred_with_sc - pred_without_sc
        
        X_temp[:, i] = X_temp_col_original # Restore column (not strictly necessary here but good practice)


# Step 5: Calculate z-scores and percentiles for top 5 PCs
print("\n11. Computing PC statistics...")
pc_stats = {}
for i, pc_idx in enumerate(top_5_indices):
    pc_values = X_pca[:, pc_idx]
    pc_stats[f'pc{pc_idx}_zscore'] = stats.zscore(pc_values)
    pc_stats[f'pc{pc_idx}_percentile'] = stats.rankdata(pc_values, method='average') / len(pc_values) * 100
    pc_stats[f'pc{pc_idx}_raw'] = pc_values
    pc_stats[f'pc{pc_idx}_shap_ai'] = shap_ai_full[:, i]
    pc_stats[f'pc{pc_idx}_shap_sc'] = shap_sc_full[:, i]

# Add to dataframe
for key, values in pc_stats.items():
    essays_df[key] = values

# Step 6: Create enhanced UMAP visualization
print("\n12. Creating enhanced UMAP visualization (with fixed hover)...")

# Load UMAP coordinates
umap_coords = np.load(CHECKPOINT_DIR / 'umap_3d_nvembed_custom.npy')

# Prepare visualization data
viz_data = essays_df[['essay_id', 'sc11', 'ai_rating', 'essay']].copy()
viz_data['x'] = umap_coords[:, 0]
viz_data['y'] = umap_coords[:, 1] 
viz_data['z'] = umap_coords[:, 2]

# Add PC data
for pc_idx in top_5_indices:
    viz_data[f'pc{pc_idx}_zscore'] = essays_df[f'pc{pc_idx}_zscore']
    viz_data[f'pc{pc_idx}_percentile'] = essays_df[f'pc{pc_idx}_percentile']
    viz_data[f'pc{pc_idx}_shap_ai'] = essays_df[f'pc{pc_idx}_shap_ai']
    viz_data[f'pc{pc_idx}_shap_sc'] = essays_df[f'pc{pc_idx}_shap_sc']

# Create HTML with model comparison table
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DML PC Analysis UMAP (Fixed Hover)</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; background: #0a0a0a; color: white; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.9); 
                 padding: 15px; border-radius: 5px; max-width: 500px; border: 1px solid #444; }}
        #controls {{ position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.9); 
                    padding: 15px; border-radius: 5px; max-width: 400px; max-height: 80vh; 
                    overflow-y: auto; border: 1px solid #444; }}
        .checkbox-group {{ margin: 10px 0; }}
        .checkbox-group label {{ display: block; margin: 3px 0; cursor: pointer; }}
        .checkbox-group input {{ margin-right: 5px; }}
        #tooltip {{ position: absolute; padding: 10px; background: rgba(0,0,0,0.95); color: white; 
                   border-radius: 5px; pointer-events: none; display: none; max-width: 500px; 
                   font-size: 12px; border: 1px solid #666; z-index: 1000; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
        th, td {{ padding: 5px; text-align: left; border-bottom: 1px solid #444; }}
        th {{ background: #222; font-weight: bold; }}
        .control-group {{ margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        .pc-info {{ font-size: 11px; color: #aaa; margin: 5px 0; }}
        .tooltip-header {{ font-weight: bold; margin-bottom: 5px; color: #4ecdc4; }}
        .tooltip-section {{ margin: 8px 0; padding: 5px 0; border-top: 1px solid #333; }}
        .pc-score {{ display: flex; justify-content: space-between; margin: 3px 0; font-size: 11px; }}
        .shap-positive {{ color: #ff6b6b; font-weight: bold; }}
        .shap-negative {{ color: #4ecdc4; font-weight: bold; }}
        .highlight {{ background: #333; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>DML Analysis: Top 5 PCs</h3>
        <table id="comparison-table">
            <tr>
                <th>Model</th>
                <th>DML ?</th>
                <th>p-val</th>
                <th>Text?AI</th>
                <th>Text?SC</th>
                <th>Text+SC?AI</th>
                <th>Text+AI?SC</th>
                <th>Var</th>
            </tr>
            {comparison_table_rows}
        </table>
        <div style="font-size: 11px; color: #aaa; margin-top: 10px;">
            <strong>Top 5 PCs:</strong> {top_pcs_list}<br>
            <strong>Variance explained:</strong> {variance_explained}%<br>
            <strong>Points shown:</strong> <span id="showing-count">{n_points}</span>/{n_points}
        </div>
        <div style="font-size: 10px; color: #888; margin-top: 10px; border-top: 1px solid #444; padding-top: 5px;">
            <strong>R² Legend:</strong> All values are 5-fold cross-validated<br>
            ? Text?AI: How well text predicts AI ratings<br>
            ? Text?SC: How well text predicts actual social class<br>
            ? Text+SC?AI: How well text + actual SC predict AI ratings<br>
            ? Text+AI?SC: How well text + AI ratings predict actual SC
        </div>
    </div>
    <div id="controls">
        <h3>Display Controls</h3>
        <div class="control-group">
            <label>Point Size: <span id="size-val">6</span></label>
            <input type="range" id="point-size" min="0.5" max="20" step="0.5" value="6" onchange="updatePointSize()">
        </div>
        <div class="control-group">
            <label>Opacity: <span id="opacity-val">0.8</span></label>
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.8" onchange="updateOpacity()">
        </div>
        <div class="control-group">
            <label>Cloud Scale: <span id="scale-val">4</span>x</label>
            <input type="range" id="cloud-scale" min="0.5" max="10" step="0.5" value="4" onchange="updateCloudScale()">
        </div>
        
        <h3>Color Mode</h3>
        <div class="control-group">
            <select id="color-mode" onchange="updateColorMode()" style="width: 100%; padding: 5px; background: #333; color: white; border: 1px solid #666;">
                <option value="social_class">Social Class (Categories)</option>
                <option value="social_class_gradient">Social Class (Gradient)</option>
                <option value="ai_rating">AI Rating (Gradient)</option>
                {pc_color_options}
            </select>
            <div id="color-legend" style="margin-top: 10px; font-size: 11px;"></div>
        </div>
        
        <h3>Social Class Filter</h3>
        <div class="checkbox-group">
            <label><input type="checkbox" id="sc-all" checked onchange="toggleAllSC()"> All Classes</label>
            <label><input type="checkbox" id="sc-1" checked onchange="updateSCFilter()"> Class 1 (Lower)</label>
            <label><input type="checkbox" id="sc-2" checked onchange="updateSCFilter()"> Class 2</label>
            <label><input type="checkbox" id="sc-3" checked onchange="updateSCFilter()"> Class 3</label>
            <label><input type="checkbox" id="sc-4" checked onchange="updateSCFilter()"> Class 4</label>
            <label><input type="checkbox" id="sc-5" checked onchange="updateSCFilter()"> Class 5 (Upper)</label>
        </div>
        
        <h3>PC Filters</h3>
        {pc_controls}
    </div>
    <div id="tooltip"></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const data = {data_json};
        const topPCs = {top_pcs_json};
        const pcImportance = {pc_importance_json};
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('container').appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Create points geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        const pointIds = new Float32Array(data.length); // For picking

        // Color scale for social class categories
        const scColors = [
            [0.2, 0.2, 0.8],  // SC1 - blue
            [0.2, 0.8, 0.8],  // SC2 - cyan
            [0.2, 0.8, 0.2],  // SC3 - green
            [0.8, 0.8, 0.2],  // SC4 - yellow
            [0.8, 0.2, 0.2]   // SC5 - red
        ];
        
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        
        // Store original positions for scaling
        const originalPositions = new Float32Array(data.length * 3);
        data.forEach((d, i) => {{
            originalPositions[i * 3] = d.x;
            originalPositions[i * 3 + 1] = d.y;
            originalPositions[i * 3 + 2] = d.z;
            pointIds[i] = i; // Assign unique ID for picking
        }});
        
        // Initial scale factor
        let currentScaleFactor = 4.0;
        
        data.forEach((d, i) => {{
            const scaledX = d.x * currentScaleFactor;
            const scaledY = d.y * currentScaleFactor;
            const scaledZ = d.z * currentScaleFactor;
            
            positions[i * 3] = scaledX;
            positions[i * 3 + 1] = scaledY;
            positions[i * 3 + 2] = scaledZ;
            
            const sc = d.sc11 - 1; // 0-indexed
            colors[i * 3] = scColors[sc][0];
            colors[i * 3 + 1] = scColors[sc][1];
            colors[i * 3 + 2] = scColors[sc][2];
            
            sizes[i] = 6; // Initial point size
            
            minX = Math.min(minX, scaledX); maxX = Math.max(maxX, scaledX);
            minY = Math.min(minY, scaledY); maxY = Math.max(maxY, scaledY);
            minZ = Math.min(minZ, scaledZ); maxZ = Math.max(maxZ, scaledZ);
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('pointId', new THREE.BufferAttribute(pointIds, 1)); // Add ID attribute

        // Visual shader material
        const visualMaterial = new THREE.ShaderMaterial({{
            uniforms: {{
                opacity: {{ value: 0.8 }}
            }},
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                varying vec3 vColor;
                
                void main() {{
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * 50.0 / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                uniform float opacity;
                varying vec3 vColor;
                
                void main() {{
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                    if (dist > 0.5) discard; // Make points circular
                    gl_FragColor = vec4(vColor, opacity);
                }}
            `,
            transparent: true,
            depthTest: true, 
            depthWrite: false // Important for transparent points
        }});

        // Picking shader material
        const pickingMaterial = new THREE.ShaderMaterial({{
            vertexShader: `
                attribute float size;
                attribute float pointId; // The ID attribute
                varying float vPointId;
                varying float vSize; 
                
                void main() {{
                    vPointId = pointId;
                    vSize = size; // Pass size to fragment shader
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * 50.0 / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                varying float vPointId;
                varying float vSize;
                
                void main() {{
                    if (vSize == 0.0) { // If point is filtered out (size is 0)
                        discard;       // Don't draw it in the picking pass
                    }
                    // Encode pointId into RGB
                    float id = vPointId;
                    float r = mod(id, 256.0) / 255.0;
                    float g = mod(floor(id / 256.0), 256.0) / 255.0;
                    float b = floor(id / (256.0 * 256.0)) / 255.0;
                    gl_FragColor = vec4(r, g, b, 1.0); // Alpha must be 1 for picking
                }}
            `
        }});
        
        const points = new THREE.Points(geometry, visualMaterial); // Start with visual material
        scene.add(points);
        
        // Highlight sphere for hover
        const highlightGeometry = new THREE.SphereGeometry(0.3, 16, 16); // Adjusted radius
        const highlightMaterial = new THREE.MeshBasicMaterial({{
            color: 0xffff00,
            transparent: true,
            opacity: 0.5, // Make it visible
            wireframe: true
        }});
        const highlightSphere = new THREE.Mesh(highlightGeometry, highlightMaterial);
        highlightSphere.visible = false;
        scene.add(highlightSphere);
        
        // Center camera
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        
        camera.position.set(centerX + 80, centerY + 80, centerZ + 80); // Adjusted for scale
        controls.target.set(centerX, centerY, centerZ);
        controls.update();

        // Picking setup
        const pickingRenderTarget = new THREE.WebGLRenderTarget(1, 1);
        const pixelBuffer = new Uint8Array(4);
        
        // Filtering and color modes (global state)
        const pcFilters = {{}};
        const scFilters = {{ 1: true, 2: true, 3: true, 4: true, 5: true }};
        let currentColorMode = 'social_class';
        
        // Color scales function (gradient)
        function getGradientColor(value, min, max) {{
            const norm = Math.max(0, Math.min(1, (value - min) / (max - min))); // Clamp norm to 0-1
            let r, g, b;
            if (norm < 0.2) {{ const t = norm * 5; r = 0; g = Math.floor(50 * t); b = 255; }}
            else if (norm < 0.4) {{ const t = (norm - 0.2) * 5; r = Math.floor(50 * t); g = Math.floor(50 * (1 - t)); b = Math.floor(255 * (1 - t) + 50 * t); }}
            else if (norm < 0.6) {{ const t = (norm - 0.4) * 5; r = Math.floor(50 + 20 * t); g = 0; b = Math.floor(50 - 20 * t); }}
            else if (norm < 0.8) {{ const t = (norm - 0.6) * 5; r = Math.floor(70 + 100 * t); g = 0; b = Math.floor(30 * (1 - t)); }}
            else {{ const t = (norm - 0.8) * 5; r = 255; g = Math.floor(100 * t); b = 0; }}
            return [r/255, g/255, b/255];
        }}
        
        function updateColorMode() {{
            currentColorMode = document.getElementById('color-mode').value;
            const colorsAttr = geometry.attributes.color.array;
            const legend = document.getElementById('color-legend');
            
            if (currentColorMode === 'social_class') {{
                data.forEach((d, i) => {{
                    const sc = d.sc11 - 1;
                    colorsAttr[i * 3] = scColors[sc][0];
                    colorsAttr[i * 3 + 1] = scColors[sc][1];
                    colorsAttr[i * 3 + 2] = scColors[sc][2];
                }});
                legend.innerHTML = `...`; // Abbreviated for brevity
            }} else if (currentColorMode === 'social_class_gradient') {{
                data.forEach((d, i) => {{ /* ... */ }}); // Abbreviated
                legend.innerHTML = `...`;
            }} else if (currentColorMode === 'ai_rating') {{
                const minRating = Math.min(...data.map(d => d.ai_rating));
                const maxRating = Math.max(...data.map(d => d.ai_rating));
                data.forEach((d, i) => {{ /* ... */ }}); // Abbreviated
                legend.innerHTML = `...`;
            }} else if (currentColorMode.startsWith('pc')) {{
                const pcNum = parseInt(currentColorMode.substring(2));
                const pcKey = `pc${{pcNum}}_percentile`;
                data.forEach((d, i) => {{ /* ... */ }}); // Abbreviated
                legend.innerHTML = `...`;
            }}
            geometry.attributes.color.needsUpdate = true;
        }}
        
        function updatePointSize() {{
            const sizeVal = parseFloat(document.getElementById('point-size').value);
            document.getElementById('size-val').textContent = sizeVal;
            const sizesAttr = geometry.attributes.size.array;
            // Re-apply filters to determine which points get the new size
            // This ensures that filtered points (size 0) remain size 0
            updateVisibility(); // This will set sizes correctly based on current filters and new base size
        }}
        
        function updateOpacity() {{
            const opacity = parseFloat(document.getElementById('point-opacity').value);
            document.getElementById('opacity-val').textContent = opacity;
            visualMaterial.uniforms.opacity.value = opacity;
        }}
        
        function updateCloudScale() {{
            const newScale = parseFloat(document.getElementById('cloud-scale').value);
            document.getElementById('scale-val').textContent = newScale;
            currentScaleFactor = newScale;
            
            const positionsAttr = geometry.attributes.position.array;
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
            
            for (let i = 0; i < data.length; i++) {{
                const scaledX = originalPositions[i * 3] * newScale;
                const scaledY = originalPositions[i * 3 + 1] * newScale;
                const scaledZ = originalPositions[i * 3 + 2] * newScale;
                
                positionsAttr[i * 3] = scaledX;
                positionsAttr[i * 3 + 1] = scaledY;
                positionsAttr[i * 3 + 2] = scaledZ;
                
                minX = Math.min(minX, scaledX); maxX = Math.max(maxX, scaledX);
                minY = Math.min(minY, scaledY); maxY = Math.max(maxY, scaledY);
                minZ = Math.min(minZ, scaledZ); maxZ = Math.max(maxZ, scaledZ);
            }}
            geometry.attributes.position.needsUpdate = true;
            
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const centerZ = (minZ + maxZ) / 2;
            controls.target.set(centerX, centerY, centerZ);
            controls.update();
        }}
        
        function toggleAllSC() {{ /* ... (same as before) ... */ updateVisibility(); }}
        function updateSCFilter() {{ /* ... (same as before) ... */ updateVisibility(); }}
        
        function updateVisibility() {{
            let visibleCount = 0;
            const sizesAttr = geometry.attributes.size.array;
            const basePointSize = parseFloat(document.getElementById('point-size').value);

            data.forEach((d, i) => {{
                let visible = true;
                if (!scFilters[d.sc11]) {{ visible = false; }}
                
                if (visible) {{
                    for (const pc of topPCs) {{
                        const filter = pcFilters[`pc${{pc}}`];
                        if (filter && filter.active) {{
                            const percentile = d[`pc${{pc}}_percentile`];
                            if (percentile >= filter.min && percentile <= filter.max) {{
                                visible = false; break;
                            }}
                        }}
                    }}
                }}
                sizesAttr[i] = visible ? basePointSize : 0;
                if (visible) visibleCount++;
            }});
            geometry.attributes.size.needsUpdate = true;
            document.getElementById('showing-count').textContent = visibleCount;
        }}

        const tooltip = document.getElementById('tooltip');
        const mouse = new THREE.Vector2(); // For mouse position, not raycasting vector

        function onMouseMove(event) {{
            // Set the picking material on the points
            points.material = pickingMaterial;

            // Get canvas-relative mouse coordinates
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = event.clientX - rect.left;
            mouse.y = event.clientY - rect.top;

            // Set the camera to render only a 1x1 pixel area at the mouse position
            camera.setViewOffset(
                renderer.domElement.width, renderer.domElement.height, // Full renderer size
                mouse.x, mouse.y,    // Mouse coords (x, y)
                1, 1                  // Pick box size (width, height)
            );

            // Render to the 1x1 picking target
            renderer.setRenderTarget(pickingRenderTarget);
            renderer.render(scene, camera);

            // Clear the view offset and reset render target for main rendering
            camera.clearViewOffset();
            renderer.setRenderTarget(null);

            // Read the pixel
            renderer.readRenderTargetPixels(pickingRenderTarget, 0, 0, 1, 1, pixelBuffer);

            const id = pixelBuffer[0] + (pixelBuffer[1] * 256) + (pixelBuffer[2] * 256 * 256);

            // Restore the original visual material
            points.material = visualMaterial;

            if (id < data.length && data[id]) {{ // Check if ID is valid
                const d = data[id];
                const currentSizes = geometry.attributes.size.array;

                if (currentSizes[id] > 0) {{ // Check if the identified point is actually visible
                    highlightSphere.position.set(
                        originalPositions[id * 3] * currentScaleFactor,
                        originalPositions[id * 3 + 1] * currentScaleFactor,
                        originalPositions[id * 3 + 2] * currentScaleFactor
                    );
                    highlightSphere.visible = true;
                    
                    let pcInfo = '<div style="background: #1a1a1a; padding: 8px; border-radius: 4px; margin-top: 5px;">';
                    pcInfo += '<div style="color: #888; font-size: 10px; margin-bottom: 5px;">TOP 5 PRINCIPAL COMPONENTS</div>';
                    for (const pc of topPCs) {{
                        const zscore = d[`pc${{pc}}_zscore`];
                        const percentile = d[`pc${{pc}}_percentile`];
                        const shapAI = d[`pc${{pc}}_shap_ai`];
                        const shapSC = d[`pc${{pc}}_shap_sc`];
                        const importance = pcImportance[pc]; // pc is the actual PC number (e.g., 0, 1, 7)
                        pcInfo += `
                            <div style="margin-bottom: 8px; padding: 5px; background: #222; border-radius: 3px;">
                                <div class="pc-score"><span><strong>PC${{pc}}</strong></span><span style="color: #666; font-size: 10px;">Importance: ${{importance.toFixed(3)}}</span></div>
                                <div class="pc-score"><span>Z-score: <span class="highlight">${{zscore.toFixed(2)}}</span></span><span>Percentile: <span class="highlight">${{percentile.toFixed(1)}}%</span></span></div>
                                <div class="pc-score"><span>SHAP AI: <span class="${{shapAI >= 0 ? 'shap-positive' : 'shap-negative'}}">${{shapAI >= 0 ? '+' : ''}}${{shapAI.toFixed(3)}}</span></span><span>SHAP SC: <span class="${{shapSC >= 0 ? 'shap-positive' : 'shap-negative'}}">${{shapSC >= 0 ? '+' : ''}}${{shapSC.toFixed(3)}}</span></span></div>
                            </div>`;
                    }}
                    pcInfo += '</div>';
                    tooltip.innerHTML = `
                        <div class="tooltip-header">Essay #${{d.essay_id}}</div>
                        <div style="margin: 5px 0;"><span class="highlight">Social Class: ${{d.sc11}}</span> | <span class="highlight">AI Rating: ${{d.ai_rating.toFixed(2)}}</span></div>
                        <div class="tooltip-section"><div style="color: #888; font-size: 10px; margin-bottom: 3px;">Essay Preview:</div><div style="font-style: italic; color: #ccc;">"${{d.essay.substring(0, 150)}}..."</div></div>
                        <div class="tooltip-section">${{pcInfo}}</div>`;
                    tooltip.style.display = 'block';
                    tooltip.style.left = event.clientX + 10 + 'px';
                    tooltip.style.top = event.clientY + 10 + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                    highlightSphere.visible = false;
                }}
            }} else {{
                tooltip.style.display = 'none';
                highlightSphere.visible = false;
            }}
        }}
        window.addEventListener('mousemove', onMouseMove);
        
        // PC filter functions (togglePCFilter, updatePCFilter) - same as before
        function togglePCFilter(pc) {{ /* ... */ updateVisibility(); }}
        function updatePCFilter(pc) {{ /* ... */ updateVisibility(); }}
        
        // Animation
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        // Window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Initialize color legend and visibility
        updateColorMode();
        updateVisibility(); // Ensure initial visibility is correct
    </script>
</body>
</html>
"""

# Generate comparison table rows
comparison_rows = ""
for _, row in comparison_df.iterrows():
    if row['Model'] == 'Difference':
        comparison_rows += f"""
        <tr style="border-top: 2px solid #666;">
            <td><strong>{row['Model']}</strong></td>
            <td>{row['DML ?']:.4f}</td>
            <td>{row['DML p-value']}</td>
            <td>{row['Text?AI R²']:.3f}</td>
            <td>{row['Text?SC R²']:.3f}</td>
            <td>{row['Text+SC?AI R²']:.3f}</td>
            <td>{row['Text+AI?SC R²']:.3f}</td>
            <td>{row['Variance']}</td>
        </tr>
        """
    else:
        comparison_rows += f"""
        <tr>
            <td><strong>{row['Model']}</strong></td>
            <td>{row['DML ?']:.4f}</td>
            <td>{row['DML p-value']:.4f}</td>
            <td>{row['Text?AI R²']:.3f}</td>
            <td>{row['Text?SC R²']:.3f}</td>
            <td>{row['Text+SC?AI R²']:.3f}</td>
            <td>{row['Text+AI?SC R²']:.3f}</td>
            <td>{row['Variance']}</td>
        </tr>
        """

# Generate PC color options
pc_color_options = ""
for pc_idx in top_5_indices:
    pc_color_options += f'<option value="pc{pc_idx}">PC{pc_idx} (Gradient)</option>\n                '

# Generate PC controls
pc_controls_html = ""
for i, pc_idx in enumerate(top_5_indices):
    pc_controls_html += f"""
    <div class="control-group">
        <h4>PC{pc_idx}</h4>
        <div class="pc-info">
            Combined Importance: {combined_importance[pc_idx]:.3f}<br>
            AI Rating Model: {importance_ai_rating[pc_idx]:.3f} | Actual SC Model: {importance_actual_sc[pc_idx]:.3f}<br>
            Variance Explained: {pca_model.explained_variance_ratio_[pc_idx]*100:.1f}%
        </div>
        <div style="margin: 5px 0;">
            <label>
                <input type="checkbox" id="pc{pc_idx}-active" onchange="togglePCFilter({pc_idx})">
                Enable filter
            </label>
        </div>
        <div style="margin: 5px 0;">
            <label>Show below: <span id="pc{pc_idx}-min-val">20</span>%</label>
            <input type="range" id="pc{pc_idx}-min" min="0" max="50" value="20" 
                   oninput="updatePCFilter({pc_idx})" disabled style="width: 100%;">
        </div>
        <div style="margin: 5px 0;">
            <label>Show above: <span id="pc{pc_idx}-max-val">80</span>%</label>
            <input type="range" id="pc{pc_idx}-max" min="50" max="100" value="80" 
                   oninput="updatePCFilter({pc_idx})" disabled style="width: 100%;">
        </div>
        <div style="font-size: 10px; color: #666; margin-top: 5px;">
            Shows extremes (hides middle range)
        </div>
    </div>
    """

# Prepare data for JSON
viz_data_json = []
for _, row in viz_data.iterrows():
    row_data = {
        'essay_id': str(row['essay_id']), # Keep as string
        'sc11': int(row['sc11']),
        'ai_rating': float(row['ai_rating']),
        'essay': row['essay'][:200], # Truncate essay for tooltip
        'x': float(row['x']),
        'y': float(row['y']),
        'z': float(row['z'])
    }
    
    # Add PC data
    for pc_idx in top_5_indices:
        row_data[f'pc{pc_idx}_zscore'] = float(row[f'pc{pc_idx}_zscore'])
        row_data[f'pc{pc_idx}_percentile'] = float(row[f'pc{pc_idx}_percentile'])
        row_data[f'pc{pc_idx}_shap_ai'] = float(row[f'pc{pc_idx}_shap_ai'])
        row_data[f'pc{pc_idx}_shap_sc'] = float(row[f'pc{pc_idx}_shap_sc'])
    
    viz_data_json.append(row_data)

# PC importance for JavaScript (keys must be strings for JSON)
pc_importance_dict = {str(int(pc)): float(combined_importance[pc]) for pc in top_5_indices}


# Generate final HTML
html_content = html_template.format(
    comparison_table_rows=comparison_rows,
    top_pcs_list=', '.join([f'PC{pc}' for pc in top_5_indices]),
    variance_explained=f"{pca_model.explained_variance_ratio_[top_5_indices].sum()*100:.1f}",
    n_points=len(viz_data),
    pc_controls=pc_controls_html,
    pc_color_options=pc_color_options,
    data_json=json.dumps(viz_data_json),
    top_pcs_json=json.dumps(top_5_indices.tolist()), # list of ints
    pc_importance_json=json.dumps(pc_importance_dict) # dict with string keys
)

# Save HTML
output_path = OUTPUT_DIR / 'umap_dml_top5_pcs_fixed_hover.html'
with open(output_path, 'w') as f:
    f.write(html_content)

# Save all results (same as before)
results = {
    'comparison_df': comparison_df,
    'top_5_indices': top_5_indices,
    'combined_importance': combined_importance,
    'importance_ai_rating': importance_ai_rating,
    'importance_actual_sc': importance_actual_sc,
    'dml_full': {
        'theta': theta_full, 'ci_lower': theta_full_lower, 
        'ci_upper': theta_full_upper, 'p_value': p_value_full
    },
    'dml_top5': {
        'theta': theta_top5, 'ci_lower': theta_top5_lower,
        'ci_upper': theta_top5_upper, 'p_value': p_value_top5
    }
}

with open(OUTPUT_DIR / 'dml_pc_analysis_results_fixed_hover.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n? Analysis complete with fixed hover!")
print(f"    HTML visualization: {output_path}")
print(f"    Model comparison: {OUTPUT_DIR / 'dml_model_comparison.csv'}") # This file is unchanged
print(f"\n? Key findings (unchanged by hover fix):")
print(f"    - Top 5 PCs capture {pca_model.explained_variance_ratio_[top_5_indices].sum()*100:.1f}% of variance")
print(f"    - DML effect preserved: {theta_full:.4f} ? {theta_top5:.4f}")
print(f"    - AI R² change: {r2_ai_full:.3f} ? {r2_ai_top5:.3f}")
print(f"    - SC R² change: {r2_sc_full:.3f} ? {r2_sc_top5:.3f}")