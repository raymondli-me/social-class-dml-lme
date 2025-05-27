#!/usr/bin/env python3
"""
Comprehensive DML analysis with top PCs and enhanced UMAP visualization
Modified to highlight the actual dots instead of using a separate sphere
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

print("=== Comprehensive DML PC Analysis (Dot Highlight Version) ===")

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

# Step 2: Identify top PCs by combined importance
print("\n4. Analyzing feature importance...")

# Get AI rating prediction importance
xgb_ai = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
xgb_ai.fit(X_pca, Y_ai)
importance_ai = pd.Series(xgb_ai.feature_importances_, index=[f'PC{i}' for i in range(X_pca.shape[1])])

# Get actual SC prediction importance  
xgb_sc = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
xgb_sc.fit(X_pca, Y_sc)
importance_sc = pd.Series(xgb_sc.feature_importances_, index=[f'PC{i}' for i in range(X_pca.shape[1])])

# Combined importance score (weighted)
combined_importance = 0.7 * importance_ai + 0.3 * importance_sc
top_5_indices = combined_importance.nlargest(5).index

print("\nTop 5 PCs by combined importance (AI * 0.7 + SC * 0.3):")
for pc in top_5_indices:
    idx = int(pc.replace('PC', ''))
    print(f"   {pc}: AI={importance_ai[pc]:.4f}, SC={importance_sc[pc]:.4f}, Combined={combined_importance[pc]:.4f}")

# Step 3: Fit DML with only top 5 PCs
print("\n5. Fitting DML with only top 5 PCs...")
top_5_pc_indices = [int(pc.replace('PC', '')) for pc in top_5_indices]
X_top5 = X_pca[:, top_5_pc_indices]

dml_top5 = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    model_t=xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    random_state=42
)
dml_top5.fit(Y=Y_ai, T=Y_sc, X=X_top5, W=None)

# Get top 5 effect
theta_top5 = dml_top5.effect(X_top5).mean()
ci_top5 = dml_top5.effect_interval(X_top5, alpha=0.05)
theta_top5_lower = ci_top5[0].mean()
theta_top5_upper = ci_top5[1].mean()
inference_top5 = dml_top5.effect_inference(X_top5).population_summary()
p_value_top5 = inference_top5.pvalue()

print(f"   Top 5 model θ = {theta_top5:.4f} [{theta_top5_lower:.4f}, {theta_top5_upper:.4f}], p = {p_value_top5:.4f}")

# Calculate DML R² for top 5
X_with_sc_top5 = np.column_stack([X_top5, Y_sc.reshape(-1, 1)])
y_pred_dml_ai_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_sc_top5, Y_ai, cv=5
)
r2_dml_ai_top5 = r2_score(Y_ai, y_pred_dml_ai_top5)

X_with_ai_top5 = np.column_stack([X_top5, Y_ai.reshape(-1, 1)])
y_pred_dml_sc_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_with_ai_top5, Y_sc, cv=5
)
r2_dml_sc_top5 = r2_score(Y_sc, y_pred_dml_sc_top5)

# Also calculate standard prediction R²
y_pred_ai_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_pca, Y_ai, cv=5
)
r2_ai_full = r2_score(Y_ai, y_pred_ai_full)

y_pred_sc_full = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_pca, Y_sc, cv=5
)
r2_sc_full = r2_score(Y_sc, y_pred_sc_full)

y_pred_ai_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_top5, Y_ai, cv=5
)
r2_ai_top5 = r2_score(Y_ai, y_pred_ai_top5)

y_pred_sc_top5 = cross_val_predict(
    xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    X_top5, Y_sc, cv=5
)
r2_sc_top5 = r2_score(Y_sc, y_pred_sc_top5)

# Try to calculate SHAP values (optional)
try:
    print("\n6. Attempting SHAP value calculation for top 5 PCs...")
    # Note: This might fail with XGBoost in DML context
    # Using proxy approach: feature contributions
    
    # For AI rating model
    contributions_ai = []
    for i in range(X_top5.shape[0]):
        pred = xgb_ai.predict(X_pca[i:i+1, :])[0]
        contribs = []
        for j, pc_idx in enumerate(top_5_pc_indices):
            # Approximate contribution
            X_zeroed = X_pca[i:i+1, :].copy()
            X_zeroed[0, pc_idx] = 0
            pred_zeroed = xgb_ai.predict(X_zeroed)[0]
            contrib = pred - pred_zeroed
            contribs.append(contrib)
        contributions_ai.append(contribs)
    contributions_ai = np.array(contributions_ai)
    
    # For SC model
    contributions_sc = []
    for i in range(X_top5.shape[0]):
        pred = xgb_sc.predict(X_pca[i:i+1, :])[0]
        contribs = []
        for j, pc_idx in enumerate(top_5_pc_indices):
            X_zeroed = X_pca[i:i+1, :].copy()
            X_zeroed[0, pc_idx] = 0
            pred_zeroed = xgb_sc.predict(X_zeroed)[0]
            contrib = pred - pred_zeroed
            contribs.append(contrib)
        contributions_sc.append(contribs)
    contributions_sc = np.array(contributions_sc)
    
    shap_available = True
    print("   SHAP approximation computed successfully")
except Exception as e:
    print(f"   SHAP calculation failed: {str(e)}")
    contributions_ai = np.zeros((X_top5.shape[0], 5))
    contributions_sc = np.zeros((X_top5.shape[0], 5))
    shap_available = False

# Create comparison dataframe
comparison_data = [
    {
        'Model': 'Full (200 PCs)',
        'DML θ': theta_full,
        'p-value': p_value_full,
        'Text→AI R²': r2_ai_full,
        'Text→SC R²': r2_sc_full,
        'Text+SC→AI R²': r2_dml_ai_full,
        'Text+AI→SC R²': r2_dml_sc_full,
        'Variance': pca_model.explained_variance_ratio_[:200].sum()
    },
    {
        'Model': 'Top 5 PCs',
        'DML θ': theta_top5,
        'p-value': p_value_top5,
        'Text→AI R²': r2_ai_top5,
        'Text→SC R²': r2_sc_top5,
        'Text+SC→AI R²': r2_dml_ai_top5,
        'Text+AI→SC R²': r2_dml_sc_top5,
        'Variance': pca_model.explained_variance_ratio_[[int(pc.replace('PC', '')) for pc in top_5_indices]].sum()
    }
]
comparison_df = pd.DataFrame(comparison_data)

print("\n7. Model Comparison:")
print(comparison_df.to_string())

# Save results
results = {
    'comparison_df': comparison_df,
    'top_5_indices': top_5_indices,
    'combined_importance': combined_importance,
    'importance_ai_rating': importance_ai,
    'importance_actual_sc': importance_sc,
    'dml_full': dml_full,
    'dml_top5': dml_top5,
}

results_file = OUTPUT_DIR / 'dml_pc_analysis_results.pkl'
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to {results_file}")

# Calculate PC values for all essays
pc_data = pd.DataFrame(X_pca[:, top_5_pc_indices], 
                      columns=[f'PC{pc}' for pc in top_5_pc_indices])

# Add PC z-scores and percentiles
for i, pc_idx in enumerate(top_5_pc_indices):
    pc_col = f'PC{pc_idx}'
    values = pc_data[pc_col]
    
    # Z-scores
    z_scores = stats.zscore(values)
    pc_data[f'pc{pc_idx}_zscore'] = z_scores
    
    # Percentiles
    percentiles = stats.rankdata(values, method='average') / len(values) * 100
    pc_data[f'pc{pc_idx}_percentile'] = percentiles
    
    # Add contributions (SHAP-like values)
    pc_data[f'pc{pc_idx}_shap_ai'] = contributions_ai[:, i]
    pc_data[f'pc{pc_idx}_shap_sc'] = contributions_sc[:, i]

# Add basic info
pc_data['essay_id'] = essays_df['essay_id']
pc_data['sc11'] = essays_df['sc11']
pc_data['ai_rating'] = essays_df['ai_rating']
pc_data['essay'] = essays_df['essay']

# Generate UMAP
print("\n8. Computing UMAP projection...")
import umap

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='cosine',
    random_state=42
)
umap_3d = umap_model.fit_transform(X_pca)
print(f"   UMAP projection computed: {umap_3d.shape}")

# Create visualization data
viz_data = []
for i in range(len(pc_data)):
    point = {
        'essay_id': pc_data.iloc[i]['essay_id'],
        'sc11': int(pc_data.iloc[i]['sc11']),
        'ai_rating': float(pc_data.iloc[i]['ai_rating']),
        'essay': pc_data.iloc[i]['essay'][:200],  # First 200 chars
        'x': float(umap_3d[i, 0]),
        'y': float(umap_3d[i, 1]),
        'z': float(umap_3d[i, 2])
    }
    
    # Add PC data
    for pc_idx in top_5_pc_indices:
        point[f'pc{pc_idx}_zscore'] = float(pc_data.iloc[i][f'pc{pc_idx}_zscore'])
        point[f'pc{pc_idx}_percentile'] = float(pc_data.iloc[i][f'pc{pc_idx}_percentile'])
        point[f'pc{pc_idx}_shap_ai'] = float(pc_data.iloc[i][f'pc{pc_idx}_shap_ai'])
        point[f'pc{pc_idx}_shap_sc'] = float(pc_data.iloc[i][f'pc{pc_idx}_shap_sc'])
    
    viz_data.append(point)

# HTML template with dot highlighting instead of sphere
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DML PC Analysis UMAP (Dot Highlight)</title>
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
                   font-size: 12px; border: 1px solid #666; }}
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
            <div>Total points: {total_points}</div>
            <div>Visible points: <span id="visible-count">{total_points}</span></div>
            <div>Top 5 PCs: {top_pcs_list}</div>
        </div>
    </div>
    
    <div id="controls">
        <h4>Visualization Controls</h4>
        
        <div class="control-group">
            <strong>View Settings</strong><br>
            <label>Point Size: <input type="range" id="point-size" min="0.5" max="20" value="6" step="0.5"></label>
            <label>Opacity: <input type="range" id="point-opacity" min="0.1" max="1" value="0.8" step="0.1"></label>
            <label>Cloud Scale: <input type="range" id="cloud-scale" min="0.5" max="10" value="4" step="0.1"></label>
        </div>
        
        <div class="control-group">
            <strong>Color by:</strong><br>
            <select id="color-mode">
                <option value="sc">Social Class</option>
                <option value="ai">AI Rating</option>
                <option value="pc1">PC {pc1}</option>
                <option value="pc2">PC {pc2}</option>
                <option value="pc3">PC {pc3}</option>
                <option value="pc4">PC {pc4}</option>
                <option value="pc5">PC {pc5}</option>
            </select>
        </div>
        
        <div class="control-group">
            <strong>Filter by Social Class</strong>
            <button onclick="toggleAllSC()">Toggle All</button>
            <div class="checkbox-group">
                <label><input type="checkbox" class="sc-filter" value="1" checked> Lower (1)</label>
                <label><input type="checkbox" class="sc-filter" value="2" checked> Working (2)</label>
                <label><input type="checkbox" class="sc-filter" value="3" checked> Middle (3)</label>
                <label><input type="checkbox" class="sc-filter" value="4" checked> Upper-Middle (4)</label>
                <label><input type="checkbox" class="sc-filter" value="5" checked> Upper (5)</label>
            </div>
        </div>
        
        <div class="control-group">
            <strong>PC Percentile Filter</strong><br>
            <label>Show extremes (top/bottom %): <input type="range" id="percentile-threshold" min="5" max="50" value="100" step="5"></label>
            <div id="percentile-value">100%</div>
            <select id="pc-filter">
                <option value="none">No filter</option>
                <option value="pc{pc1}">PC {pc1} extremes</option>
                <option value="pc{pc2}">PC {pc2} extremes</option>
                <option value="pc{pc3}">PC {pc3} extremes</option>
                <option value="pc{pc4}">PC {pc4} extremes</option>
                <option value="pc{pc5}">PC {pc5} extremes</option>
            </select>
        </div>
    </div>
    
    <div id="tooltip"></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const data = {data_json};
        const topPCs = {top_pcs_json};
        
        // Three.js setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('container').appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        const originalSizes = new Float32Array(data.length);
        
        // Color schemes
        const scColors = {{
            1: [1, 0, 0],      // Red
            2: [1, 0.5, 0],    // Orange
            3: [1, 1, 0],      // Yellow
            4: [0, 1, 0],      // Green
            5: [0, 0, 1]       // Blue
        }};
        
        // Scale factor for cloud
        let scaleFactor = 4;
        
        // Compute bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        
        data.forEach((d, i) => {{
            const x = d.x * scaleFactor;
            const y = d.y * scaleFactor;
            const z = d.z * scaleFactor;
            
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;
            
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
            
            const color = scColors[d.sc11];
            colors[i * 3] = color[0];
            colors[i * 3 + 1] = color[1];
            colors[i * 3 + 2] = color[2];
            
            sizes[i] = 6;
            originalSizes[i] = 6;
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Shader material that respects size changes
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
                    gl_PointSize = size * 50.0 / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                uniform float opacity;
                varying vec3 vColor;
                
                void main() {{
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                    if (dist > 0.5) discard;
                    
                    // Smooth edge
                    float alpha = 1.0 - smoothstep(0.45, 0.5, dist);
                    gl_FragColor = vec4(vColor, opacity * alpha);
                }}
            `,
            transparent: true,
            depthTest: true,
            depthWrite: false
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Keep track of hovered point
        let hoveredIndex = -1;
        
        // Center camera
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        
        camera.position.set(80, 80, 80);  // Adjusted for 4x scale
        controls.target.set(centerX, centerY, centerZ);
        controls.update();
        
        // Raycaster for interaction
        const raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 3;  // Adjusted for larger point sizes
        const mouse = new THREE.Vector2();
        
        // Update functions
        function updateColors() {{
            const mode = document.getElementById('color-mode').value;
            const colors = geometry.attributes.color.array;
            
            data.forEach((d, i) => {{
                let color;
                if (mode === 'sc') {{
                    color = scColors[d.sc11];
                }} else if (mode === 'ai') {{
                    const normalized = (d.ai_rating - 1) / 9;
                    color = [normalized, 0, 1 - normalized];
                }} else if (mode.startsWith('pc')) {{
                    const pcNum = mode.replace('pc', '');
                    const percentile = d[`pc${{pcNum}}_percentile`] / 100;
                    color = [percentile, 0.5, 1 - percentile];
                }}
                
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            }});
            
            geometry.attributes.color.needsUpdate = true;
        }}
        
        function updateVisibility() {{
            const scFilters = Array.from(document.querySelectorAll('.sc-filter:checked')).map(cb => parseInt(cb.value));
            const percentileThreshold = parseInt(document.getElementById('percentile-threshold').value);
            const pcFilter = document.getElementById('pc-filter').value;
            
            let visibleCount = 0;
            const sizes = geometry.attributes.size.array;
            const currentSize = parseFloat(document.getElementById('point-size').value);
            
            data.forEach((d, i) => {{
                let visible = scFilters.includes(d.sc11);
                
                if (visible && pcFilter !== 'none') {{
                    const percentile = d[`${{pcFilter}}_percentile`];
                    visible = percentile <= percentileThreshold || percentile >= (100 - percentileThreshold);
                }}
                
                // Highlight hovered point by making it larger
                if (i === hoveredIndex) {{
                    sizes[i] = visible ? currentSize * 1.5 : 0;  // 50% larger when hovered
                }} else {{
                    sizes[i] = visible ? currentSize : 0;
                }}
                
                originalSizes[i] = visible ? currentSize : 0;
                
                if (visible) visibleCount++;
            }});
            
            geometry.attributes.size.needsUpdate = true;
            document.getElementById('visible-count').textContent = visibleCount;
        }}
        
        function updateSize() {{
            const size = parseFloat(document.getElementById('point-size').value);
            const sizes = geometry.attributes.size.array;
            
            for (let i = 0; i < sizes.length; i++) {{
                if (originalSizes[i] > 0) {{
                    sizes[i] = (i === hoveredIndex) ? size * 1.5 : size;
                    originalSizes[i] = size;
                }}
            }}
            
            geometry.attributes.size.needsUpdate = true;
        }}
        
        function updateOpacity() {{
            const opacity = parseFloat(document.getElementById('point-opacity').value);
            material.uniforms.opacity.value = opacity;
        }}
        
        function updateCloudScale() {{
            const newScale = parseFloat(document.getElementById('cloud-scale').value);
            const scaleRatio = newScale / scaleFactor;
            scaleFactor = newScale;
            
            const positions = geometry.attributes.position.array;
            
            // Update bounds
            minX = Infinity; maxX = -Infinity;
            minY = Infinity; maxY = -Infinity;
            minZ = Infinity; maxZ = -Infinity;
            
            for (let i = 0; i < positions.length; i += 3) {{
                positions[i] *= scaleRatio;
                positions[i + 1] *= scaleRatio;
                positions[i + 2] *= scaleRatio;
                
                minX = Math.min(minX, positions[i]);
                maxX = Math.max(maxX, positions[i]);
                minY = Math.min(minY, positions[i + 1]);
                maxY = Math.max(maxY, positions[i + 1]);
                minZ = Math.min(minZ, positions[i + 2]);
                maxZ = Math.max(maxZ, positions[i + 2]);
            }}
            
            geometry.attributes.position.needsUpdate = true;
            
            // Update camera target to new center
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const centerZ = (minZ + maxZ) / 2;
            controls.target.set(centerX, centerY, centerZ);
            controls.update();
        }}
        
        function toggleAllSC() {{
            const checkboxes = document.querySelectorAll('.sc-filter');
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            checkboxes.forEach(cb => cb.checked = !allChecked);
            updateVisibility();
        }}
        
        // Event listeners
        document.getElementById('color-mode').addEventListener('change', updateColors);
        document.getElementById('point-size').addEventListener('input', updateSize);
        document.getElementById('point-opacity').addEventListener('input', updateOpacity);
        document.getElementById('cloud-scale').addEventListener('input', updateCloudScale);
        document.getElementById('percentile-threshold').addEventListener('input', function() {{
            document.getElementById('percentile-value').textContent = this.value + '%';
            updateVisibility();
        }});
        document.getElementById('pc-filter').addEventListener('change', updateVisibility);
        
        document.querySelectorAll('.sc-filter').forEach(cb => {{
            cb.addEventListener('change', updateVisibility);
        }});
        
        // Mouse interaction
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            // Update raycaster threshold based on current point size
            const currentSize = parseFloat(document.getElementById('point-size').value);
            raycaster.params.Points.threshold = currentSize / 2;
            
            raycaster.setFromCamera(mouse, camera);
            
            // Only check for intersections with visible points
            const sizes = geometry.attributes.size.array;
            
            // Check intersections
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {{
                // Find first visible intersection
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
                    
                    // Update hovered index
                    if (hoveredIndex !== idx) {{
                        // Restore previous hovered point
                        if (hoveredIndex >= 0 && originalSizes[hoveredIndex] > 0) {{
                            sizes[hoveredIndex] = originalSizes[hoveredIndex];
                        }}
                        
                        // Highlight new point
                        hoveredIndex = idx;
                        sizes[idx] = originalSizes[idx] * 1.5;  // Make 50% larger
                        geometry.attributes.size.needsUpdate = true;
                    }}
                
                let pcInfo = '<div style="background: #1a1a1a; padding: 8px; border-radius: 4px; margin-top: 5px;">';
                pcInfo += '<div style="color: #888; font-size: 10px; margin-bottom: 5px;">TOP 5 PRINCIPAL COMPONENTS</div>';
                
                for (const pc of topPCs) {{
                    const zscore = d[`pc${{pc}}_zscore`];
                    const percentile = d[`pc${{pc}}_percentile`];
                    const shapAI = d[`pc${{pc}}_shap_ai`];
                    const shapSC = d[`pc${{pc}}_shap_sc`];
                    
                    pcInfo += `
                        <div style="margin-bottom: 8px; padding: 5px; background: #222; border-radius: 3px;">
                            <div class="pc-score"><b>PC ${{pc}}</b></div>
                            <div class="pc-score">
                                <span>Z-score: ${{zscore > 0 ? '+' : ''}}${{zscore.toFixed(2)}}</span>
                                <span>Percentile: ${{percentile.toFixed(0)}}</span>
                            </div>
                            <div class="pc-score">
                                <span class="${{shapAI > 0 ? 'shap-positive' : 'shap-negative'}}">
                                    AI: ${{shapAI > 0 ? '+' : ''}}${{shapAI.toFixed(3)}}
                                </span>
                                <span class="${{shapSC > 0 ? 'shap-positive' : 'shap-negative'}}">
                                    SC: ${{shapSC > 0 ? '+' : ''}}${{shapSC.toFixed(3)}}
                                </span>
                            </div>
                        </div>
                    `;
                }}
                pcInfo += '</div>';
                
                const tooltip = document.getElementById('tooltip');
                tooltip.innerHTML = `
                    <div class="tooltip-header">Essay #${{d.essay_id}}</div>
                    <div>Social Class: <span class="highlight">${{d.sc11}}</span> | AI Rating: <span class="highlight">${{d.ai_rating.toFixed(2)}}</span></div>
                    ${{pcInfo}}
                    <div style="margin-top: 10px; font-style: italic; color: #888; font-size: 11px;">
                        "${{d.essay.substring(0, 150)}}..."
                    </div>
                `;
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 10) + 'px';
                tooltip.style.top = (event.clientY + 10) + 'px';
                }} else {{
                    hideTooltip();
                }}
            }} else {{
                hideTooltip();
            }}
        }}
        
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
            
            // Restore hovered point size
            if (hoveredIndex >= 0 && originalSizes[hoveredIndex] > 0) {{
                const sizes = geometry.attributes.size.array;
                sizes[hoveredIndex] = originalSizes[hoveredIndex];
                geometry.attributes.size.needsUpdate = true;
                hoveredIndex = -1;
            }}
        }}
        
        document.addEventListener('mousemove', onMouseMove);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""

# Format the comparison table rows
comparison_table_rows = ""
for _, row in comparison_df.iterrows():
    comparison_table_rows += f"""
        <tr>
            <td>{row['Model']}</td>
            <td>{row['DML θ']:.4f}</td>
            <td>{row['p-value']:.4f}</td>
            <td>{row['Text→AI R²']:.3f}</td>
            <td>{row['Text→SC R²']:.3f}</td>
            <td>{row['Text+SC→AI R²']:.3f}</td>
            <td>{row['Text+AI→SC R²']:.3f}</td>
            <td>{row['Variance']:.1%}</td>
        </tr>
    """

# Get PC indices for template
pc_indices = [int(pc.replace('PC', '')) for pc in top_5_indices]

# Create the HTML content
html_content = html_template.format(
    data_json=json.dumps(viz_data),
    top_pcs_json=json.dumps(pc_indices),
    comparison_table_rows=comparison_table_rows,
    total_points=len(viz_data),
    top_pcs_list=', '.join([f'PC{pc}' for pc in pc_indices]),
    pc1=pc_indices[0],
    pc2=pc_indices[1],
    pc3=pc_indices[2],
    pc4=pc_indices[3],
    pc5=pc_indices[4]
)

# Save the visualization
output_file = OUTPUT_DIR / 'umap_dml_top5_pcs_dot_highlight.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n9. Enhanced UMAP visualization saved to:")
print(f"   {output_file}")
print("\nVisualization features:")
print("  - Dots themselves highlight on hover (50% larger)")
print("  - No separate highlight sphere")
print("  - Shows top 5 PC z-scores, percentiles, and contributions")
print("  - Interactive filtering by social class and PC percentiles")
print("  - Multiple coloring modes")
print("  - Adjustable point size, opacity, and cloud scale")

# Save additional data for future use
additional_results = {
    'data_with_pcs': pc_data,
    'umap_3d': umap_3d,
    'contributions_ai': contributions_ai,
    'contributions_sc': contributions_sc,
    'feature_importance_ai': importance_ai.to_dict(),
    'feature_importance_sc': importance_sc.to_dict(),
    'top_pcs': pc_indices,
    'shap_available': shap_available
}

additional_file = OUTPUT_DIR / 'dml_pc_analysis_results_with_umap.pkl'
with open(additional_file, 'wb') as f:
    pickle.dump(additional_results, f)

print(f"\nAdditional results saved to {additional_file}")
print("\n=== Analysis Complete ===")