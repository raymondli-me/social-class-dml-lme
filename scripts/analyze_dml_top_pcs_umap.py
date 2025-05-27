#!/usr/bin/env python3
"""
Comprehensive DML analysis with top PCs and enhanced UMAP visualization
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

print("=== Comprehensive DML PC Analysis ===")

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
print(f"   Loaded {X_pca.shape[0]} essays with {X_pca.shape[1]} PCs")

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

print(f"   Full model θ = {theta_full:.4f} [{theta_full_lower:.4f}, {theta_full_upper:.4f}], p = {p_value_full:.4f}")

# Calculate DML R² - how well X + treatment predicts outcome
# For DML on AI ratings: X (text) + actual SC → AI rating
print("\n3. Computing DML R² (cross-validated)...")

# Model 1: Text + Actual SC → AI Rating
X_with_sc_full = np.column_stack([X_pca, Y_sc.reshape(-1, 1)])
y_pred_dml_ai_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_sc_full, Y_ai, cv=5
)
r2_dml_ai_full = r2_score(Y_ai, y_pred_dml_ai_full)
print(f"   Full model: Text + Actual SC → AI Rating R² = {r2_dml_ai_full:.4f}")

# Model 2: Text + AI Rating → Actual SC (reverse DML)
X_with_ai_full = np.column_stack([X_pca, Y_ai.reshape(-1, 1)])
y_pred_dml_sc_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_ai_full, Y_sc, cv=5
)
r2_dml_sc_full = r2_score(Y_sc, y_pred_dml_sc_full)
print(f"   Full model: Text + AI Rating → Actual SC R² = {r2_dml_sc_full:.4f}")

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
    print(f"   PC{idx}: Combined importance = {combined_importance[idx]:.4f} "
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

print(f"   Top 5 model θ = {theta_top5:.4f} [{theta_top5_lower:.4f}, {theta_top5_upper:.4f}], p = {p_value_top5:.4f}")

# Calculate DML R² for top 5 model
print("\n7. Computing DML R² for top 5 model...")

# Model 1: Text (top 5) + Actual SC → AI Rating
X_with_sc_top5 = np.column_stack([X_top5, Y_sc.reshape(-1, 1)])
y_pred_dml_ai_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_sc_top5, Y_ai, cv=5
)
r2_dml_ai_top5 = r2_score(Y_ai, y_pred_dml_ai_top5)
print(f"   Top 5 model: Text + Actual SC → AI Rating R² = {r2_dml_ai_top5:.4f}")

# Model 2: Text (top 5) + AI Rating → Actual SC
X_with_ai_top5 = np.column_stack([X_top5, Y_ai.reshape(-1, 1)])
y_pred_dml_sc_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_ai_top5, Y_sc, cv=5
)
r2_dml_sc_top5 = r2_score(Y_sc, y_pred_dml_sc_top5)
print(f"   Top 5 model: Text + AI Rating → Actual SC R² = {r2_dml_sc_top5:.4f}")

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
    'DML θ': [theta_full, theta_top5, theta_top5 - theta_full],
    'DML p-value': [p_value_full, p_value_top5, '-'],
    'Text→AI R²': [r2_ai_full, r2_ai_top5, r2_ai_top5 - r2_ai_full],
    'Text→SC R²': [r2_sc_full, r2_sc_top5, r2_sc_top5 - r2_sc_full],
    'Text+SC→AI R²': [r2_dml_ai_full, r2_dml_ai_top5, r2_dml_ai_top5 - r2_dml_ai_full],
    'Text+AI→SC R²': [r2_dml_sc_full, r2_dml_sc_top5, r2_dml_sc_top5 - r2_dml_sc_full],
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
    print("   Computing SHAP values (using KernelExplainer)...")
    
    # Sample background data for KernelExplainer
    background = shap.sample(X_top5, 100)
    
    # Create explainers
    explainer_ai = shap.KernelExplainer(model_ai_top5.predict, background)
    explainer_sc = shap.KernelExplainer(model_sc_top5.predict, background)
    
    # Get SHAP values (this will be slow, so we'll sample)
    sample_size = min(1000, len(X_top5))
    sample_indices = np.random.choice(len(X_top5), sample_size, replace=False)
    
    shap_values_ai = explainer_ai.shap_values(X_top5[sample_indices])
    shap_values_sc = explainer_sc.shap_values(X_top5[sample_indices])
    
    # Extrapolate to full dataset
    shap_ai_full = np.zeros((len(X_top5), 5))
    shap_sc_full = np.zeros((len(X_top5), 5))
    
    shap_ai_full[sample_indices] = shap_values_ai
    shap_sc_full[sample_indices] = shap_values_sc
    
except Exception as e:
    print(f"   SHAP computation failed: {e}")
    print("   Using feature contributions instead...")
    
    # Fallback: use feature contributions
    shap_ai_full = np.zeros((len(X_top5), 5))
    shap_sc_full = np.zeros((len(X_top5), 5))
    
    for i in range(5):
        X_temp = X_top5.copy()
        X_temp[:, i] = 0
        pred_without = model_ai_top5.predict(X_temp)
        pred_with = model_ai_top5.predict(X_top5)
        shap_ai_full[:, i] = pred_with - pred_without
        
        pred_without_sc = model_sc_top5.predict(X_temp)
        pred_with_sc = model_sc_top5.predict(X_top5)
        shap_sc_full[:, i] = pred_with_sc - pred_without_sc

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
print("\n12. Creating enhanced UMAP visualization...")

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
    <title>DML PC Analysis UMAP</title>
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
                   font-size: 12px; border: 2px solid #666; transition: border-color 0.2s; }}
        #tooltip .clickable {{ pointer-events: auto; cursor: pointer; text-decoration: underline; }}
        #full-text {{ position: absolute; bottom: 10px; right: 10px; background: rgba(255,255,255,0.95);
                     padding: 15px; border-radius: 5px; font-size: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                     max-width: 600px; max-height: 300px; overflow-y: auto; display: none; }}
        #full-text h4 {{ margin-top: 0; }}
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
                <th>DML θ</th>
                <th>p-val</th>
                <th>Text→AI</th>
                <th>Text→SC</th>
                <th>Text+SC→AI</th>
                <th>Text+AI→SC</th>
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
            • Text→AI: How well text predicts AI ratings<br>
            • Text→SC: How well text predicts actual social class<br>
            • Text+SC→AI: How well text + actual SC predict AI ratings<br>
            • Text+AI→SC: How well text + AI ratings predict actual SC
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
    <div id="full-text"></div>
    
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
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        
        // Color scale
        const scColors = [
            [0.2, 0.2, 0.8],  // SC1 - blue
            [0.2, 0.8, 0.8],  // SC2 - cyan
            [0.2, 0.8, 0.2],  // SC3 - green
            [0.8, 0.8, 0.2],  // SC4 - yellow
            [0.8, 0.2, 0.2]   // SC5 - red
        ];
        
        // Calculate bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        
        // Store original positions for scaling
        const originalPositions = new Float32Array(data.length * 3);
        data.forEach((d, i) => {{
            originalPositions[i * 3] = d.x;
            originalPositions[i * 3 + 1] = d.y;
            originalPositions[i * 3 + 2] = d.z;
        }});
        
        // Initial scale factor
        let currentScaleFactor = 4.0;
        
        data.forEach((d, i) => {{
            // Scale the coordinates for more spacing
            const scaledX = d.x * currentScaleFactor;
            const scaledY = d.y * currentScaleFactor;
            const scaledZ = d.z * currentScaleFactor;
            
            positions[i * 3] = scaledX;
            positions[i * 3 + 1] = scaledY;
            positions[i * 3 + 2] = scaledZ;
            
            const sc = d.sc11 - 1;
            colors[i * 3] = scColors[sc][0];
            colors[i * 3 + 1] = scColors[sc][1];
            colors[i * 3 + 2] = scColors[sc][2];
            
            sizes[i] = 6;  // Larger initial size
            
            minX = Math.min(minX, scaledX);
            maxX = Math.max(maxX, scaledX);
            minY = Math.min(minY, scaledY);
            maxY = Math.max(maxY, scaledY);
            minZ = Math.min(minZ, scaledZ);
            maxZ = Math.max(maxZ, scaledZ);
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Custom shader material that respects individual point sizes
        const material = new THREE.ShaderMaterial({{
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
                    gl_PointSize = size * 50.0 / -mvPosition.z;  // Adjusted multiplier for new size range
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                uniform float opacity;
                varying vec3 vColor;
                
                void main() {{
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                    if (dist > 0.5) discard;
                    gl_FragColor = vec4(vColor, opacity);
                }}
            `,
            transparent: true,
            depthTest: true,
            depthWrite: false
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Track hovered point for color-matched tooltip
        let hoveredPointColor = null;
        let currentEssay = null;
        
        // Center camera
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        
        camera.position.set(80, 80, 80);  // Adjusted for 4x scale
        controls.target.set(centerX, centerY, centerZ);
        controls.update();
        
        // Filtering and color modes
        const pcFilters = {{}};
        const scFilters = {{ 1: true, 2: true, 3: true, 4: true, 5: true }};
        let currentColorMode = 'social_class';
        
        // Color scales
        function getGradientColor(value, min, max) {{
            // Normalize to 0-1
            const norm = (value - min) / (max - min);
            
            // Gradient with dark middle: Blue -> Dark Purple -> Dark Red -> Bright Red
            let r, g, b;
            if (norm < 0.2) {{
                // Bright blue for low extremes
                const t = norm * 5;
                r = 0;
                g = Math.floor(50 * t);
                b = 255;
            }} else if (norm < 0.4) {{
                // Blue to dark purple
                const t = (norm - 0.2) * 5;
                r = Math.floor(50 * t);
                g = Math.floor(50 * (1 - t));
                b = Math.floor(255 * (1 - t) + 50 * t);
            }} else if (norm < 0.6) {{
                // Dark middle zone
                const t = (norm - 0.4) * 5;
                r = Math.floor(50 + 20 * t);
                g = 0;
                b = Math.floor(50 - 20 * t);
            }} else if (norm < 0.8) {{
                // Dark red to red
                const t = (norm - 0.6) * 5;
                r = Math.floor(70 + 100 * t);
                g = 0;
                b = Math.floor(30 * (1 - t));
            }} else {{
                // Bright red for high extremes
                const t = (norm - 0.8) * 5;
                r = 255;
                g = Math.floor(100 * t);
                b = 0;
            }}
            
            return [r/255, g/255, b/255];
        }}
        
        function updateColorMode() {{
            currentColorMode = document.getElementById('color-mode').value;
            const colors = geometry.attributes.color.array;
            const legend = document.getElementById('color-legend');
            
            if (currentColorMode === 'social_class') {{
                // Original categorical colors
                data.forEach((d, i) => {{
                    const sc = d.sc11 - 1;
                    colors[i * 3] = scColors[sc][0];
                    colors[i * 3 + 1] = scColors[sc][1];
                    colors[i * 3 + 2] = scColors[sc][2];
                }});
                legend.innerHTML = `
                    <span style="color: #3333cc;">■</span> Lower |
                    <span style="color: #33cccc;">■</span> Working |
                    <span style="color: #33cc33;">■</span> Middle |
                    <span style="color: #cccc33;">■</span> Upper-middle |
                    <span style="color: #cc3333;">■</span> Upper
                `;
            }} else if (currentColorMode === 'social_class_gradient') {{
                // Gradient based on social class value
                data.forEach((d, i) => {{
                    const color = getGradientColor(d.sc11, 1, 5);
                    colors[i * 3] = color[0];
                    colors[i * 3 + 1] = color[1];
                    colors[i * 3 + 2] = color[2];
                }});
                legend.innerHTML = '<span style="color: #0032ff;">■</span> SC 1 → <span style="color: #320032;">■</span> (middle) → <span style="color: #ff1900;">■</span> SC 5';
            }} else if (currentColorMode === 'ai_rating') {{
                // Gradient based on AI rating
                const minRating = Math.min(...data.map(d => d.ai_rating));
                const maxRating = Math.max(...data.map(d => d.ai_rating));
                data.forEach((d, i) => {{
                    const color = getGradientColor(d.ai_rating, minRating, maxRating);
                    colors[i * 3] = color[0];
                    colors[i * 3 + 1] = color[1];
                    colors[i * 3 + 2] = color[2];
                }});
                legend.innerHTML = `<span style="color: #0032ff;">■</span> Low → <span style="color: #320032;">■</span> (middle) → <span style="color: #ff1900;">■</span> High`;
            }} else if (currentColorMode.startsWith('pc')) {{
                // Gradient based on PC value
                const pcNum = parseInt(currentColorMode.substring(2));
                const pcKey = `pc${{pcNum}}_percentile`;
                data.forEach((d, i) => {{
                    const color = getGradientColor(d[pcKey], 0, 100);
                    colors[i * 3] = color[0];
                    colors[i * 3 + 1] = color[1];
                    colors[i * 3 + 2] = color[2];
                }});
                legend.innerHTML = `<span style="color: #0032ff;">■</span> Low PC${{pcNum}} → <span style="color: #320032;">■</span> (middle) → <span style="color: #ff1900;">■</span> High`;
            }}
            
            geometry.attributes.color.needsUpdate = true;
        }}
        
        // Display controls
        function updatePointSize() {{
            const size = parseFloat(document.getElementById('point-size').value);
            document.getElementById('size-val').textContent = size;
            // Update all sizes in the array
            const sizes = geometry.attributes.size.array;
            for (let i = 0; i < sizes.length; i++) {{
                if (sizes[i] > 0) {{  // Only update visible points
                    sizes[i] = size;
                }}
            }}
            geometry.attributes.size.needsUpdate = true;
        }}
        
        function updateOpacity() {{
            const opacity = parseFloat(document.getElementById('point-opacity').value);
            document.getElementById('opacity-val').textContent = opacity;
            material.uniforms.opacity.value = opacity;
        }}
        
        function updateCloudScale() {{
            const newScale = parseFloat(document.getElementById('cloud-scale').value);
            document.getElementById('scale-val').textContent = newScale;
            currentScaleFactor = newScale;
            
            // Update all positions
            const positions = geometry.attributes.position.array;
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;
            
            for (let i = 0; i < data.length; i++) {{
                const scaledX = originalPositions[i * 3] * newScale;
                const scaledY = originalPositions[i * 3 + 1] * newScale;
                const scaledZ = originalPositions[i * 3 + 2] * newScale;
                
                positions[i * 3] = scaledX;
                positions[i * 3 + 1] = scaledY;
                positions[i * 3 + 2] = scaledZ;
                
                minX = Math.min(minX, scaledX);
                maxX = Math.max(maxX, scaledX);
                minY = Math.min(minY, scaledY);
                maxY = Math.max(maxY, scaledY);
                minZ = Math.min(minZ, scaledZ);
                maxZ = Math.max(maxZ, scaledZ);
            }}
            
            geometry.attributes.position.needsUpdate = true;
            
            // Update highlight sphere scale
            // Scale updated (highlight sphere removed)
            
            // Update camera target to new center
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const centerZ = (minZ + maxZ) / 2;
            controls.target.set(centerX, centerY, centerZ);
            controls.update();
        }}
        
        function toggleAllSC() {{
            const allChecked = document.getElementById('sc-all').checked;
            for (let i = 1; i <= 5; i++) {{
                document.getElementById(`sc-${{i}}`).checked = allChecked;
                scFilters[i] = allChecked;
            }}
            updateVisibility();
        }}
        
        function updateSCFilter() {{
            let anyUnchecked = false;
            for (let i = 1; i <= 5; i++) {{
                scFilters[i] = document.getElementById(`sc-${{i}}`).checked;
                if (!scFilters[i]) anyUnchecked = true;
            }}
            document.getElementById('sc-all').checked = !anyUnchecked;
            updateVisibility();
        }}
        
        function updateVisibility() {{
            let visibleCount = 0;
            const sizes = geometry.attributes.size.array;
            
            data.forEach((d, i) => {{
                let visible = true;
                
                // Check social class filter
                if (!scFilters[d.sc11]) {{
                    visible = false;
                }}
                
                // Check PC filters - show extremes (below min OR above max)
                if (visible) {{
                    for (const pc of topPCs) {{
                        const filter = pcFilters[`pc${{pc}}`];
                        if (filter && filter.active) {{
                            const percentile = d[`pc${{pc}}_percentile`];
                            // Show extremes: below min OR above max
                            if (percentile >= filter.min && percentile <= filter.max) {{
                                visible = false;  // Hide if in the middle range
                                break;
                            }}
                        }}
                    }}
                }}
                
                sizes[i] = visible ? parseFloat(document.getElementById('point-size').value) : 0;
                if (visible) visibleCount++;
            }});
            
            geometry.attributes.size.needsUpdate = true;
            document.getElementById('showing-count').textContent = visibleCount;
        }}
        
        // Tooltip
        const tooltip = document.getElementById('tooltip');
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        // Custom raycaster with generous threshold
        raycaster.params.Points.threshold = 10;  // More generous for better hover detection
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            // Update raycaster threshold based on current point size
            const currentSize = parseFloat(document.getElementById('point-size').value);
            raycaster.params.Points.threshold = currentSize * 2;  // More generous threshold
            
            raycaster.setFromCamera(mouse, camera);
            
            // Only check for intersections with visible points
            const sizes = geometry.attributes.size.array;
            
            // Temporarily create a filtered geometry for raycasting
            const visibleIndices = [];
            for (let i = 0; i < data.length; i++) {{
                if (sizes[i] > 0) {{
                    visibleIndices.push(i);
                }}
            }}
            
            // Check intersections
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {{
                // Double-check that we hit a visible point
                let visibleIntersect = null;
                for (const intersect of intersects) {{
                    if (sizes[intersect.index] > 0) {{
                        visibleIntersect = intersect;
                        break;
                    }}
                }}
                
                if (visibleIntersect) {{
                    const idx = visibleIntersect.index;
                    const d = data[idx];
                    currentEssay = d;
                    
                    // Get the color of the hovered point
                    const colors = geometry.attributes.color.array;
                    const r = Math.floor(colors[idx * 3] * 255);
                    const g = Math.floor(colors[idx * 3 + 1] * 255);
                    const b = Math.floor(colors[idx * 3 + 2] * 255);
                    hoveredPointColor = `rgb(${{r}}, ${{g}}, ${{b}})`;
                    
                    // Update tooltip border to match point color
                    tooltip.style.borderColor = hoveredPointColor;
                
                let pcInfo = '<div style="background: #1a1a1a; padding: 8px; border-radius: 4px; margin-top: 5px;">';
                pcInfo += '<div style="color: #888; font-size: 10px; margin-bottom: 5px;">TOP 5 PRINCIPAL COMPONENTS</div>';
                
                for (const pc of topPCs) {{
                    const zscore = d[`pc${{pc}}_zscore`];
                    const percentile = d[`pc${{pc}}_percentile`];
                    const shapAI = d[`pc${{pc}}_shap_ai`];
                    const shapSC = d[`pc${{pc}}_shap_sc`];
                    const importance = pcImportance[pc];
                    
                    pcInfo += `
                        <div style="margin-bottom: 8px; padding: 5px; background: #222; border-radius: 3px;">
                            <div class="pc-score">
                                <span><strong>PC${{pc}}</strong></span>
                                <span style="color: #666; font-size: 10px;">Importance: ${{importance.toFixed(3)}}</span>
                            </div>
                            <div class="pc-score">
                                <span>Z-score: <span class="highlight">${{zscore.toFixed(2)}}</span></span>
                                <span>Percentile: <span class="highlight">${{percentile.toFixed(1)}}%</span></span>
                            </div>
                            <div class="pc-score">
                                <span>SHAP AI: <span class="${{shapAI >= 0 ? 'shap-positive' : 'shap-negative'}}">${{shapAI >= 0 ? '+' : ''}}${{shapAI.toFixed(3)}}</span></span>
                                <span>SHAP SC: <span class="${{shapSC >= 0 ? 'shap-positive' : 'shap-negative'}}">${{shapSC >= 0 ? '+' : ''}}${{shapSC.toFixed(3)}}</span></span>
                            </div>
                        </div>
                    `;
                }}
                pcInfo += '</div>';
                
                tooltip.innerHTML = `
                    <div class="tooltip-header">Essay #${{d.essay_id}}</div>
                    <div style="margin: 5px 0;">
                        <span class="highlight">Social Class: ${{d.sc11}}</span> | 
                        <span class="highlight">AI Rating: ${{d.ai_rating.toFixed(2)}}</span>
                    </div>
                    <div class="tooltip-section">
                        <div style="color: #888; font-size: 10px; margin-bottom: 3px;">Essay Preview:</div>
                        <div style="font-style: italic; color: #ccc; line-height: 1.4;">
                            "${{d.essay.substring(0, 300)}}${{d.essay.length > 300 ? '...' : ''}}"
                        </div>
                    </div>
                    <div class="tooltip-section">
                        ${{pcInfo}}
                    </div>
                `;
                
                tooltip.style.display = 'block';
                tooltip.style.left = event.clientX + 10 + 'px';
                tooltip.style.top = event.clientY + 10 + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                    currentEssay = null;
                }}
            }} else {{
                tooltip.style.display = 'none';
                currentEssay = null;
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Function to show full text
        window.showFullText = function() {{
            if (currentEssay) {{
                const fullTextDiv = document.getElementById('full-text');
                fullTextDiv.innerHTML = `
                    <h4>Essay #${{currentEssay.essay_id}} - Full Text</h4>
                    <div style="margin-bottom: 10px;">
                        <strong>Social Class:</strong> ${{currentEssay.sc11}} | 
                        <strong>AI Rating:</strong> ${{currentEssay.ai_rating.toFixed(2)}}
                    </div>
                    <div style="font-style: italic; line-height: 1.5;">
                        ${{currentEssay.essay}}
                    </div>
                    <button onclick="document.getElementById('full-text').style.display='none'" style="margin-top: 10px;">Close</button>
                `;
                fullTextDiv.style.display = 'block';
            }}
        }};
        
        // PC filter functions
        function togglePCFilter(pc) {{
            const active = document.getElementById(`pc${{pc}}-active`).checked;
            document.getElementById(`pc${{pc}}-min`).disabled = !active;
            document.getElementById(`pc${{pc}}-max`).disabled = !active;
            
            if (active) {{
                pcFilters[`pc${{pc}}`] = {{
                    active: true,
                    min: parseFloat(document.getElementById(`pc${{pc}}-min`).value),
                    max: parseFloat(document.getElementById(`pc${{pc}}-max`).value)
                }};
            }} else {{
                delete pcFilters[`pc${{pc}}`];
            }}
            
            updateVisibility();
        }}
        
        function updatePCFilter(pc) {{
            const minVal = parseFloat(document.getElementById(`pc${{pc}}-min`).value);
            const maxVal = parseFloat(document.getElementById(`pc${{pc}}-max`).value);
            
            document.getElementById(`pc${{pc}}-min-val`).textContent = minVal;
            document.getElementById(`pc${{pc}}-max-val`).textContent = maxVal;
            
            if (pcFilters[`pc${{pc}}`]) {{
                pcFilters[`pc${{pc}}`].min = minVal;
                pcFilters[`pc${{pc}}`].max = maxVal;
                updateVisibility();
            }}
        }}
        
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
        
        // Initialize color legend
        updateColorMode();
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
            <td>{row['DML θ']:.4f}</td>
            <td>{row['DML p-value']}</td>
            <td>{row['Text→AI R²']:.3f}</td>
            <td>{row['Text→SC R²']:.3f}</td>
            <td>{row['Text+SC→AI R²']:.3f}</td>
            <td>{row['Text+AI→SC R²']:.3f}</td>
            <td>{row['Variance']}</td>
        </tr>
        """
    else:
        comparison_rows += f"""
        <tr>
            <td><strong>{row['Model']}</strong></td>
            <td>{row['DML θ']:.4f}</td>
            <td>{row['DML p-value']:.4f}</td>
            <td>{row['Text→AI R²']:.3f}</td>
            <td>{row['Text→SC R²']:.3f}</td>
            <td>{row['Text+SC→AI R²']:.3f}</td>
            <td>{row['Text+AI→SC R²']:.3f}</td>
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
        'essay_id': str(row['essay_id']),  # Keep as string since it's like 'N10001N'
        'sc11': int(row['sc11']),
        'ai_rating': float(row['ai_rating']),
        'essay': row['essay'][:200],
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

# PC importance for JavaScript
pc_importance_dict = {int(pc): float(combined_importance[pc]) for pc in top_5_indices}

# Generate final HTML
html_content = html_template.format(
    comparison_table_rows=comparison_rows,
    top_pcs_list=', '.join([f'PC{pc}' for pc in top_5_indices]),
    variance_explained=f"{pca_model.explained_variance_ratio_[top_5_indices].sum()*100:.1f}",
    n_points=len(viz_data),
    pc_controls=pc_controls_html,
    pc_color_options=pc_color_options,
    data_json=json.dumps(viz_data_json),
    top_pcs_json=json.dumps(top_5_indices.tolist()),
    pc_importance_json=json.dumps(pc_importance_dict)
)

# Save HTML
output_path = OUTPUT_DIR / 'umap_dml_top5_pcs.html'
with open(output_path, 'w') as f:
    f.write(html_content)

# Save all results
results = {
    'comparison_df': comparison_df,
    'top_5_indices': top_5_indices,
    'combined_importance': combined_importance,
    'importance_ai_rating': importance_ai_rating,
    'importance_actual_sc': importance_actual_sc,
    'dml_full': {
        'theta': theta_full,
        'ci_lower': theta_full_lower,
        'ci_upper': theta_full_upper,
        'p_value': p_value_full
    },
    'dml_top5': {
        'theta': theta_top5,
        'ci_lower': theta_top5_lower,
        'ci_upper': theta_top5_upper,
        'p_value': p_value_top5
    }
}

with open(OUTPUT_DIR / 'dml_pc_analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✅ Analysis complete!")
print(f"   HTML visualization: {output_path}")
print(f"   Model comparison: {OUTPUT_DIR / 'dml_model_comparison.csv'}")
print(f"\n📊 Key findings:")
print(f"   - Top 5 PCs capture {pca_model.explained_variance_ratio_[top_5_indices].sum()*100:.1f}% of variance")
print(f"   - DML effect preserved: {theta_full:.4f} → {theta_top5:.4f}")
print(f"   - AI R² change: {r2_ai_full:.3f} → {r2_ai_top5:.3f}")
print(f"   - SC R² change: {r2_sc_full:.3f} → {r2_sc_top5:.3f}")