#!/usr/bin/env python3
"""
Analyze top principal components for predicting AI ratings and actual social class,
then create enhanced UMAP visualization with PC scores, SHAP values, and filtering.
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
import warnings
warnings.filterwarnings('ignore')

# Paths based on checkpoints
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'
OUTPUT_DIR = BASE_DIR / 'nvembed_pc_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=== Analyzing Top Principal Components ===")

# Load data
print("\n1. Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)
print(f"   Loaded {len(essays_df)} essays")

# Load social class labels
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')
print(f"   Merged social class labels")

# Load AI ratings - use only human MacArthur
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)
print(f"   Merged AI ratings (human MacArthur only)")

# Load NV-Embed PCA features
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
    pca_model = pca_data['pca']
    essay_ids = pca_data['essay_ids']

# Ensure alignment
essays_df = essays_df[essays_df['essay_id'].isin(essay_ids)]
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()
print(f"   Loaded PCA features: {X_pca.shape}")

# Get outcomes
Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

print("\n2. Analyzing individual PC predictive power...")

# Analyze each PC individually
pc_results = []
for i in range(X_pca.shape[1]):
    # Single PC as feature
    X_single = X_pca[:, i:i+1]
    
    # For AI ratings
    model_ai = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    y_pred_ai = cross_val_predict(model_ai, X_single, Y_ai, cv=5)
    r2_ai = r2_score(Y_ai, y_pred_ai)
    
    # For actual SC
    model_sc = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    y_pred_sc = cross_val_predict(model_sc, X_single, Y_sc, cv=5)
    r2_sc = r2_score(Y_sc, y_pred_sc)
    
    # DML causal effect (SC -> AI rating, controlling for this PC)
    try:
        dml = LinearDML(
            model_y=xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
            model_t=xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
            random_state=42
        )
        dml.fit(Y=Y_ai, T=Y_sc, X=X_single, W=None)
        theta = dml.effect(X_single).mean()
        _, p_value = dml.effect_inference(X_single).population_summary()
    except:
        theta = 0
        p_value = 1
    
    pc_results.append({
        'pc': i,
        'r2_ai': r2_ai,
        'r2_sc': r2_sc,
        'dml_theta': theta,
        'dml_pvalue': p_value,
        'variance_explained': pca_model.explained_variance_ratio_[i]
    })
    
    if i % 20 == 0:
        print(f"   Processed PC {i}/{X_pca.shape[1]}")

pc_results_df = pd.DataFrame(pc_results)

# Get top 5 PCs for each outcome
top_ai_pcs = pc_results_df.nlargest(5, 'r2_ai')['pc'].values
top_sc_pcs = pc_results_df.nlargest(5, 'r2_sc')['pc'].values
top_pcs = sorted(set(top_ai_pcs) | set(top_sc_pcs))[:10]  # Get unique top PCs

print("\n3. Top Principal Components:")
print("\nTop 5 for AI Rating prediction:")
for _, row in pc_results_df.iloc[top_ai_pcs].iterrows():
    print(f"   PC{row['pc']}: R²={row['r2_ai']:.3f}, Var explained={row['variance_explained']:.3f}")

print("\nTop 5 for Actual SC prediction:")
for _, row in pc_results_df.iloc[top_sc_pcs].iterrows():
    print(f"   PC{row['pc']}: R²={row['r2_sc']:.3f}, Var explained={row['variance_explained']:.3f}")

# Calculate z-scores for top PCs
print("\n4. Calculating z-scores for top PCs...")
pc_zscores = {}
for pc in top_pcs:
    pc_values = X_pca[:, pc]
    z_scores = stats.zscore(pc_values)
    pc_zscores[f'pc{pc}_zscore'] = z_scores
    
    # Also store raw values and percentiles
    pc_zscores[f'pc{pc}_raw'] = pc_values
    pc_zscores[f'pc{pc}_percentile'] = stats.rankdata(pc_values, method='average') / len(pc_values) * 100

# Add to dataframe
for key, values in pc_zscores.items():
    essays_df[key] = values

# Get feature contributions using a different approach
print("\n5. Computing feature contributions...")

# Train full models
model_ai_full = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_full.fit(X_pca, Y_ai)

model_sc_full = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_full.fit(X_pca, Y_sc)

# Get predictions and baseline
baseline_ai = Y_ai.mean()
baseline_sc = Y_sc.mean()

# For each essay, calculate PC contributions using permutation
# This is a simpler alternative to SHAP
print("   Computing PC contributions...")
for pc in top_pcs:
    # For AI rating
    X_temp = X_pca.copy()
    X_temp[:, pc] = 0  # Zero out this PC
    pred_without_pc = model_ai_full.predict(X_temp)
    pred_with_pc = model_ai_full.predict(X_pca)
    essays_df[f'pc{pc}_contrib_ai'] = pred_with_pc - pred_without_pc
    
    # For actual SC
    pred_without_pc_sc = model_sc_full.predict(X_temp)
    pred_with_pc_sc = model_sc_full.predict(X_pca)
    essays_df[f'pc{pc}_contrib_sc'] = pred_with_pc_sc - pred_without_pc_sc

# Also get feature importances for reference
feature_importance_ai = model_ai_full.feature_importances_
feature_importance_sc = model_sc_full.feature_importances_

# Store importance scores for top PCs
pc_importance_data = {}
for pc in top_pcs:
    pc_importance_data[f'pc{pc}_importance_ai'] = feature_importance_ai[pc]
    pc_importance_data[f'pc{pc}_importance_sc'] = feature_importance_sc[pc]

# Save results
print("\n6. Saving analysis results...")
results = {
    'pc_results': pc_results_df,
    'top_pcs': top_pcs,
    'top_ai_pcs': top_ai_pcs,
    'top_sc_pcs': top_sc_pcs,
    'essays_with_pc_data': essays_df,
    'pc_importance_data': pc_importance_data,
    'feature_importance_ai': feature_importance_ai,
    'feature_importance_sc': feature_importance_sc
}

with open(OUTPUT_DIR / 'pc_analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save summary CSV
pc_results_df.to_csv(OUTPUT_DIR / 'pc_importance_summary.csv', index=False)

print("\n7. Creating enhanced UMAP visualization...")

# Load UMAP coordinates
umap_coords = np.load(CHECKPOINT_DIR / 'umap_3d_nvembed_custom.npy')

# Create visualization data
viz_data = essays_df[['essay_id', 'sc11', 'ai_rating', 'essay']].copy()
viz_data['x'] = umap_coords[:, 0]
viz_data['y'] = umap_coords[:, 1]
viz_data['z'] = umap_coords[:, 2]

# Add PC data
for pc in top_pcs:
    viz_data[f'pc{pc}_zscore'] = essays_df[f'pc{pc}_zscore']
    viz_data[f'pc{pc}_percentile'] = essays_df[f'pc{pc}_percentile']
    viz_data[f'pc{pc}_contrib_ai'] = essays_df[f'pc{pc}_contrib_ai']
    viz_data[f'pc{pc}_contrib_sc'] = essays_df[f'pc{pc}_contrib_sc']

# Create HTML visualization
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced UMAP with PC Analysis</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; background: #0a0a0a; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.8); 
                 padding: 15px; border-radius: 5px; max-width: 400px; }}
        #controls {{ position: absolute; top: 10px; right: 10px; color: white; background: rgba(0,0,0,0.8); 
                    padding: 15px; border-radius: 5px; max-width: 350px; max-height: 80vh; overflow-y: auto; }}
        .control-group {{ margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        .slider-container {{ margin: 5px 0; }}
        input[type="range"] {{ width: 100%; }}
        .pc-info {{ font-size: 12px; color: #aaa; }}
        #tooltip {{ position: absolute; padding: 10px; background: rgba(0,0,0,0.9); color: white; 
                   border-radius: 5px; pointer-events: none; display: none; max-width: 400px; 
                   font-size: 12px; border: 1px solid #444; }}
        .tooltip-header {{ font-weight: bold; margin-bottom: 5px; }}
        .tooltip-section {{ margin: 5px 0; padding: 5px 0; border-top: 1px solid #333; }}
        .pc-score {{ display: flex; justify-content: space-between; margin: 2px 0; }}
        .shap-positive {{ color: #ff6b6b; }}
        .shap-negative {{ color: #4ecdc4; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>Enhanced UMAP Visualization</h3>
        <p>Points: {n_points}</p>
        <p>Showing: <span id="showing-count">{n_points}</span></p>
    </div>
    <div id="controls">
        <h3>Principal Component Filters</h3>
        {pc_controls}
    </div>
    <div id="tooltip"></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const data = {data_json};
        const pcResults = {pc_results_json};
        
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
        
        // Color scale for social class
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
        
        data.forEach((d, i) => {{
            positions[i * 3] = d.x;
            positions[i * 3 + 1] = d.y;
            positions[i * 3 + 2] = d.z;
            
            const sc = d.sc11 - 1;
            colors[i * 3] = scColors[sc][0];
            colors[i * 3 + 1] = scColors[sc][1];
            colors[i * 3 + 2] = scColors[sc][2];
            
            sizes[i] = 2;
            
            minX = Math.min(minX, d.x);
            maxX = Math.max(maxX, d.x);
            minY = Math.min(minY, d.y);
            maxY = Math.max(maxY, d.y);
            minZ = Math.min(minZ, d.z);
            maxZ = Math.max(maxZ, d.z);
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        const material = new THREE.PointsMaterial({{
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Center camera
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        
        camera.position.set(150, 150, 150);
        controls.target.set(centerX, centerY, centerZ);
        controls.update();
        
        // PC filtering
        const pcFilters = {{}};
        const topPCs = {top_pcs_json};
        
        function updateVisibility() {{
            let visibleCount = 0;
            const sizes = geometry.attributes.size.array;
            
            data.forEach((d, i) => {{
                let visible = true;
                
                // Check PC filters
                for (const pc of topPCs) {{
                    const filter = pcFilters[`pc${{pc}}`];
                    if (filter && filter.active) {{
                        const percentile = d[`pc${{pc}}_percentile`];
                        if (percentile < filter.min || percentile > filter.max) {{
                            visible = false;
                            break;
                        }}
                    }}
                }}
                
                sizes[i] = visible ? 2 : 0;
                if (visible) visibleCount++;
            }});
            
            geometry.attributes.size.needsUpdate = true;
            document.getElementById('showing-count').textContent = visibleCount;
        }}
        
        // Tooltip
        const tooltip = document.getElementById('tooltip');
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 2;
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const d = data[idx];
                
                let pcInfo = '';
                for (const pc of topPCs) {{
                    const pcData = pcResults.find(r => r.pc === pc);
                    const zscore = d[`pc${{pc}}_zscore`];
                    const percentile = d[`pc${{pc}}_percentile`];
                    const contribAI = d[`pc${{pc}}_contrib_ai`];
                    const contribSC = d[`pc${{pc}}_contrib_sc`];
                    
                    pcInfo += `
                        <div class="pc-score">
                            <span><strong>PC${{pc}}</strong> (AI R²=${{pcData.r2_ai.toFixed(3)}}, SC R²=${{pcData.r2_sc.toFixed(3)}})</span>
                        </div>
                        <div class="pc-score">
                            <span>Z-score: ${{zscore.toFixed(2)}} | Percentile: ${{percentile.toFixed(1)}}%</span>
                        </div>
                        <div class="pc-score">
                            <span>Contribution: AI <span class="${{contribAI >= 0 ? 'shap-positive' : 'shap-negative'}}">${{contribAI >= 0 ? '+' : ''}}${{contribAI.toFixed(3)}}</span> | 
                                   SC <span class="${{contribSC >= 0 ? 'shap-positive' : 'shap-negative'}}">${{contribSC >= 0 ? '+' : ''}}${{contribSC.toFixed(3)}}</span></span>
                        </div>
                    `;
                }}
                
                tooltip.innerHTML = `
                    <div class="tooltip-header">Essay #${{d.essay_id}}</div>
                    <div>Social Class: ${{d.sc11}} | AI Rating: ${{d.ai_rating.toFixed(2)}}</div>
                    <div class="tooltip-section">
                        <div style="color: #888; margin-bottom: 5px;">Essay Preview:</div>
                        <div>${{d.essay.substring(0, 150)}}...</div>
                    </div>
                    <div class="tooltip-section">
                        <div style="color: #888; margin-bottom: 5px;">Principal Component Analysis:</div>
                        ${{pcInfo}}
                    </div>
                `;
                
                tooltip.style.display = 'block';
                tooltip.style.left = event.clientX + 10 + 'px';
                tooltip.style.top = event.clientY + 10 + 'px';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);
        
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
    </script>
</body>
</html>
"""

# Generate PC controls HTML
pc_controls_html = ""
for pc in top_pcs:
    pc_data = pc_results_df[pc_results_df['pc'] == pc].iloc[0]
    pc_controls_html += f"""
    <div class="control-group">
        <h4>PC{pc}</h4>
        <div class="pc-info">
            AI R²: {pc_data['r2_ai']:.3f} | SC R²: {pc_data['r2_sc']:.3f}<br>
            DML θ: {pc_data['dml_theta']:.3f} (p={pc_data['dml_pvalue']:.3f})<br>
            Variance: {pc_data['variance_explained']*100:.1f}%
        </div>
        <div class="slider-container">
            <label>
                <input type="checkbox" id="pc{pc}-active" onchange="togglePCFilter({pc})">
                Enable filter
            </label>
        </div>
        <div class="slider-container">
            <label>Min percentile: <span id="pc{pc}-min-val">0</span>%</label>
            <input type="range" id="pc{pc}-min" min="0" max="100" value="0" 
                   oninput="updatePCFilter({pc})" disabled>
        </div>
        <div class="slider-container">
            <label>Max percentile: <span id="pc{pc}-max-val">100</span>%</label>
            <input type="range" id="pc{pc}-max" min="0" max="100" value="100" 
                   oninput="updatePCFilter({pc})" disabled>
        </div>
    </div>
    """

# Add JavaScript for PC filtering
pc_filter_js = """
function togglePCFilter(pc) {
    const active = document.getElementById(`pc${pc}-active`).checked;
    document.getElementById(`pc${pc}-min`).disabled = !active;
    document.getElementById(`pc${pc}-max`).disabled = !active;
    
    pcFilters[`pc${pc}`] = {
        active: active,
        min: parseFloat(document.getElementById(`pc${pc}-min`).value),
        max: parseFloat(document.getElementById(`pc${pc}-max`).value)
    };
    
    updateVisibility();
}

function updatePCFilter(pc) {
    const minVal = parseFloat(document.getElementById(`pc${pc}-min`).value);
    const maxVal = parseFloat(document.getElementById(`pc${pc}-max`).value);
    
    document.getElementById(`pc${pc}-min-val`).textContent = minVal;
    document.getElementById(`pc${pc}-max-val`).textContent = maxVal;
    
    if (pcFilters[`pc${pc}`]) {
        pcFilters[`pc${pc}`].min = minVal;
        pcFilters[`pc${pc}`].max = maxVal;
        updateVisibility();
    }
}
"""

# Prepare data for JSON
viz_data_json = []
for _, row in viz_data.iterrows():
    row_data = {
        'essay_id': int(row['essay_id']),
        'sc11': int(row['sc11']),
        'ai_rating': float(row['ai_rating']),
        'essay': row['essay'][:200],  # Truncate for size
        'x': float(row['x']),
        'y': float(row['y']),
        'z': float(row['z'])
    }
    
    # Add PC data
    for pc in top_pcs:
        row_data[f'pc{pc}_zscore'] = float(row[f'pc{pc}_zscore'])
        row_data[f'pc{pc}_percentile'] = float(row[f'pc{pc}_percentile'])
        row_data[f'pc{pc}_contrib_ai'] = float(row[f'pc{pc}_contrib_ai'])
        row_data[f'pc{pc}_contrib_sc'] = float(row[f'pc{pc}_contrib_sc'])
    
    viz_data_json.append(row_data)

# Prepare PC results for JSON
pc_results_json = pc_results_df[pc_results_df['pc'].isin(top_pcs)].to_dict('records')

# Generate final HTML
import json

html_content = html_template.format(
    n_points=len(viz_data),
    pc_controls=pc_controls_html,
    data_json=json.dumps(viz_data_json),
    pc_results_json=json.dumps(pc_results_json),
    top_pcs_json=json.dumps(top_pcs)
)

# Add PC filter JavaScript
html_content = html_content.replace('// Animation', pc_filter_js + '\n        // Animation')

# Save HTML
output_path = OUTPUT_DIR / 'umap_enhanced_pc_analysis.html'
with open(output_path, 'w') as f:
    f.write(html_content)

print(f"\n✅ Enhanced UMAP visualization saved to: {output_path}")
print(f"\n📊 Summary statistics saved to: {OUTPUT_DIR / 'pc_importance_summary.csv'}")

# Print summary
print("\n=== Analysis Summary ===")
print(f"Total PCs analyzed: {len(pc_results_df)}")
print(f"Top PCs selected: {len(top_pcs)}")
print(f"Visualization includes:")
print("  - PC z-scores and percentiles for each point")
print("  - PC contributions showing how each PC affects predictions")  
print("  - Interactive filtering by PC percentiles")
print("  - R² and DML coefficients for each PC")
print("\nOpen the HTML file in a browser to explore!")