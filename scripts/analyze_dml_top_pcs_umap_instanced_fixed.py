#!/usr/bin/env python3
"""
Fixed InstancedMesh DML analysis - generates its own data
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

print("=== Fixed InstancedMesh DML PC Analysis ===")

# Load data
print("\n1. Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

# Load social class
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings
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

# Load UMAP coordinates
with open(OUTPUT_DIR / 'dml_pc_analysis_results_with_umap.pkl', 'rb') as f:
    umap_results = pickle.load(f)
    X_umap_3d = umap_results['umap_3d']

# Ensure same order
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

print("\n2. Running DML analysis...")

# Calculate feature importances and get top 5
print("   Computing feature importances...")
Y = essays_df['ai_rating'].values
T = essays_df['sc11'].values
valid_idx = ~(np.isnan(T) | np.isnan(Y))

# Quick importance calculation
importances = []
for i in range(200):
    corr_y = abs(np.corrcoef(X_pca[valid_idx, i], Y[valid_idx])[0, 1])
    corr_t = abs(np.corrcoef(X_pca[valid_idx, i], T[valid_idx])[0, 1])
    importances.append(corr_y + corr_t)

top_pcs_idx = np.argsort(importances)[::-1][:5]
top_pcs = [i+1 for i in top_pcs_idx]  # 1-based

print(f"   Top 5 PCs by importance: {top_pcs}")

# Get features for top PCs
W = X_pca[:, top_pcs_idx]

# Drop missing values
T = T[valid_idx]
Y = Y[valid_idx]
W = W[valid_idx]
X_umap_valid = X_umap_3d[valid_idx]
essays_valid = essays_df[valid_idx].copy()

# Make T binary
T_binary = (T > 2).astype(float)

# XGBoost models
xgb_Y = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_T = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

# Fit DML
print("   Fitting DML model...")
est = LinearDML(model_y=xgb_Y, model_t=xgb_T, discrete_treatment=True, cv=5, random_state=42)
est.fit(Y, T_binary, X=None, W=W)

# Get predictions
Y_pred_cv = cross_val_predict(xgb_Y, W, Y, cv=5)
T_pred_cv = cross_val_predict(xgb_T, W, T_binary, cv=5, method='predict_proba')[:, 1]

# Calculate treatment effects - use residuals as proxy
Y_residuals = Y - Y_pred_cv
treatment_effects = Y_residuals  # Simple proxy for visualization

# Prepare visualization data
viz_data = []
for i in range(len(essays_valid)):
    pc_data = {}
    for j, pc in enumerate(top_pcs):
        pc_val = W[i, j]
        percentile = stats.percentileofscore(W[:, j], pc_val)
        pc_data[f'pc{pc}_value'] = float(pc_val)
        pc_data[f'pc{pc}_percentile'] = float(percentile)
    
    viz_data.append({
        'x': float(X_umap_valid[i, 0]),
        'y': float(X_umap_valid[i, 1]),
        'z': float(X_umap_valid[i, 2]),
        'essay_id': essays_valid.iloc[i]['essay_id'],
        'essay': essays_valid.iloc[i]['essay'],
        'sc11': int(essays_valid.iloc[i]['sc11']),
        'ai_rating': float(essays_valid.iloc[i]['ai_rating']),
        'y_pred': float(Y_pred_cv[i]),
        't_pred': float(T_pred_cv[i]),
        'treatment_effect': float(treatment_effects[i]),
        **pc_data
    })

# DML summary
ate_inference = est.ate_inference()
dml_summary = {
    'ate': float(ate_inference.mean_point),
    'ate_stderr': float(ate_inference.stderr_mean),
    'r2_y': float(r2_score(Y, Y_pred_cv)),
    'r2_t': float(np.mean((T_pred_cv > 0.5) == T_binary))
}

print(f"\n3. Creating InstancedMesh visualization...")

# HTML template with InstancedMesh
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DML Analysis - InstancedMesh Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            pointer-events: none;
            display: none;
            max-width: 600px;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 3px solid #666;
            transition: border-color 0.1s;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            max-height: 70vh;
            overflow-y: auto;
        }}
        .control-group {{ margin: 10px 0; }}
        .control-group label {{ display: inline-block; width: 120px; }}
        .pc-filter {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 3px; }}
        #legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        input[type="range"] {{ width: 150px; }}
        .value-display {{ display: inline-block; width: 40px; text-align: right; }}
        button {{ margin: 2px; padding: 5px 10px; cursor: pointer; }}
    </style>
</head>
<body>
    <div id="tooltip"></div>
    <div id="info">
        <h3>DML Analysis - InstancedMesh</h3>
        <div>ATE: {ate:.3f} ± {ate_se:.3f}</div>
        <div>Y Model R²: {y_r2:.3f}</div>
        <div>T Model Accuracy: {t_auc:.3f}</div>
        <div id="point-count">Points: {n_points}</div>
    </div>
    <div id="controls">
        <h4>Display Controls</h4>
        <div class="control-group">
            <label>Point Size:</label>
            <input type="range" id="point-size" min="0.5" max="5" step="0.1" value="1">
            <span class="value-display" id="size-val">1</span>
        </div>
        <div class="control-group">
            <label>Opacity:</label>
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.8">
            <span class="value-display" id="opacity-val">0.8</span>
        </div>
        <div class="control-group">
            <label>Cloud Scale:</label>
            <input type="range" id="cloud-scale" min="50" max="300" step="10" value="100">
            <span class="value-display" id="scale-val">100</span>
        </div>
        <div class="control-group">
            <label>Color By:</label>
            <select id="color-mode">
                <option value="social_class">Social Class (Categories)</option>
                <option value="social_class_gradient">Social Class (Gradient)</option>
                <option value="ai_rating">AI Rating</option>
                <option value="pc1">PC1</option>
                <option value="pc2">PC2</option>
                <option value="pc3">PC3</option>
                <option value="pc4">PC4</option>
                <option value="pc5">PC5</option>
            </select>
        </div>
        <h4>PC Filters</h4>
        <div id="pc-filters"></div>
        <button onclick="resetFilters()">Reset All Filters</button>
    </div>
    <div id="legend"></div>
    
    <script>
        // Data
        const data = {data_json};
        console.log('Loaded data points:', data.length);
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 5000);
        camera.position.set(150, 150, 150);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // Create base geometry - small sphere
        const baseGeometry = new THREE.SphereGeometry(0.5, 8, 6);  // Small, low-poly sphere
        
        // Create material
        const material = new THREE.MeshBasicMaterial({{
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        }});
        
        // Create InstancedMesh
        const instanceCount = data.length;
        const instancedMesh = new THREE.InstancedMesh(baseGeometry, material, instanceCount);
        scene.add(instancedMesh);
        
        // Matrix for transformations
        const matrix = new THREE.Matrix4();
        const zeroMatrix = new THREE.Matrix4().makeScale(0, 0, 0);
        const color = new THREE.Color();
        
        // Visibility tracking
        const visibilityArray = new Array(instanceCount).fill(true);
        
        // Current settings
        let currentScaleFactor = 100;
        let currentPointSize = 1;
        let currentColorMode = 'social_class';
        
        // Color schemes
        const scColors = [
            [0.2, 0.2, 0.8],  // Lower - Blue
            [0.2, 0.8, 0.8],  // Working - Cyan
            [0.2, 0.8, 0.2],  // Middle - Green
            [0.8, 0.8, 0.2],  // Upper-middle - Yellow
            [0.8, 0.2, 0.2]   // Upper - Red
        ];
        
        function getGradientColor(value, min, max) {{
            const t = (value - min) / (max - min);
            
            if (t <= 0.5) {{
                const s = t * 2;
                return [
                    0 + s * 0.196,
                    0.196 + s * (-0.196),
                    1 - s * 0.804
                ];
            }} else {{
                const s = (t - 0.5) * 2;
                return [
                    0.196 + s * 0.804,
                    0 + s * 0.098,
                    0.196 - s * 0.196
                ];
            }}
        }}
        
        // Initialize instances
        function initializeInstances() {{
            data.forEach((d, i) => {{
                // Set position and scale
                matrix.makeScale(currentPointSize, currentPointSize, currentPointSize);
                matrix.setPosition(d.x * currentScaleFactor, d.y * currentScaleFactor, d.z * currentScaleFactor);
                instancedMesh.setMatrixAt(i, matrix);
                
                // Set initial color
                updateInstanceColor(i);
            }});
            
            instancedMesh.instanceMatrix.needsUpdate = true;
            instancedMesh.instanceColor.needsUpdate = true;
        }}
        
        // Update single instance color
        function updateInstanceColor(index) {{
            const d = data[index];
            let rgb;
            
            if (currentColorMode === 'social_class') {{
                const sc = d.sc11 - 1;
                rgb = scColors[sc];
            }} else if (currentColorMode === 'social_class_gradient') {{
                rgb = getGradientColor(d.sc11, 1, 5);
            }} else if (currentColorMode === 'ai_rating') {{
                const minRating = Math.min(...data.map(d => d.ai_rating));
                const maxRating = Math.max(...data.map(d => d.ai_rating));
                rgb = getGradientColor(d.ai_rating, minRating, maxRating);
            }} else if (currentColorMode.startsWith('pc')) {{
                const pcNum = parseInt(currentColorMode.substring(2));
                const pcKey = `pc${{pcNum}}_percentile`;
                rgb = getGradientColor(d[pcKey], 0, 100);
            }}
            
            color.setRGB(rgb[0], rgb[1], rgb[2]);
            instancedMesh.setColorAt(index, color);
        }}
        
        // Update all colors
        function updateColors() {{
            const legend = document.getElementById('legend');
            
            data.forEach((d, i) => {{
                updateInstanceColor(i);
            }});
            
            instancedMesh.instanceColor.needsUpdate = true;
            
            // Update legend
            if (currentColorMode === 'social_class') {{
                legend.innerHTML = `
                    <span style="color: #3333cc;">■</span> Lower |
                    <span style="color: #33cccc;">■</span> Working |
                    <span style="color: #33cc33;">■</span> Middle |
                    <span style="color: #cccc33;">■</span> Upper-middle |
                    <span style="color: #cc3333;">■</span> Upper
                `;
            }} else if (currentColorMode === 'social_class_gradient') {{
                legend.innerHTML = '<span style="color: #0032ff;">■</span> SC 1 → <span style="color: #ff1900;">■</span> SC 5';
            }} else if (currentColorMode === 'ai_rating') {{
                legend.innerHTML = `<span style="color: #0032ff;">■</span> Low → <span style="color: #ff1900;">■</span> High`;
            }} else if (currentColorMode.startsWith('pc')) {{
                const pcNum = currentColorMode.substring(2);
                legend.innerHTML = `<span style="color: #0032ff;">■</span> Low PC${{pcNum}} → <span style="color: #ff1900;">■</span> High`;
            }}
        }}
        
        // Initialize
        initializeInstances();
        updateColors();
        
        // Raycaster for precise hovering
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');
        let hoveredInstanceId = -1;
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            
            // Intersect with InstancedMesh - precise and fast!
            const intersects = raycaster.intersectObject(instancedMesh);
            
            if (intersects.length > 0) {{
                const intersect = intersects[0];
                const instanceId = intersect.instanceId;
                
                // Check if this instance is visible
                if (visibilityArray[instanceId] && instanceId !== hoveredInstanceId) {{
                    hoveredInstanceId = instanceId;
                    const d = data[instanceId];
                    
                    // Get the color of the hovered point
                    instancedMesh.getColorAt(instanceId, color);
                    const r = Math.floor(color.r * 255);
                    const g = Math.floor(color.g * 255);
                    const b = Math.floor(color.b * 255);
                    const pointColor = `rgb(${{r}}, ${{g}}, ${{b}})`;
                    
                    // Update tooltip border to match point color
                    tooltip.style.borderColor = pointColor;
                    
                    // Show 300 characters of essay
                    const essayPreview = d.essay.substring(0, 300) + (d.essay.length > 300 ? '...' : '');
                    
                    tooltip.innerHTML = `
                        <div style="font-weight: bold; font-size: 14px; margin-bottom: 10px;">
                            Essay #${{d.essay_id}}
                        </div>
                        <div style="margin-bottom: 10px;">
                            <span style="color: #aaa;">Social Class:</span> <strong>${{d.sc11}}</strong> | 
                            <span style="color: #aaa;">AI Rating:</span> <strong>${{d.ai_rating.toFixed(2)}}</strong>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <span style="color: #aaa;">Y Pred:</span> ${{d.y_pred.toFixed(2)}} | 
                            <span style="color: #aaa;">T Pred:</span> ${{d.t_pred.toFixed(3)}} | 
                            <span style="color: #aaa;">Treatment Effect:</span> ${{d.treatment_effect.toFixed(3)}}
                        </div>
                        <div style="margin-bottom: 10px;">
                            <div style="color: #888; font-size: 11px; margin-bottom: 5px;">PRINCIPAL COMPONENTS:</div>
                            PC1: <strong>${{d.pc1_value.toFixed(2)}}</strong> (P${{d.pc1_percentile.toFixed(0)}}) | 
                            PC2: <strong>${{d.pc2_value.toFixed(2)}}</strong> (P${{d.pc2_percentile.toFixed(0)}}) | 
                            PC3: <strong>${{d.pc3_value.toFixed(2)}}</strong> (P${{d.pc3_percentile.toFixed(0)}})<br>
                            PC4: <strong>${{d.pc4_value.toFixed(2)}}</strong> (P${{d.pc4_percentile.toFixed(0)}}) | 
                            PC5: <strong>${{d.pc5_value.toFixed(2)}}</strong> (P${{d.pc5_percentile.toFixed(0)}})
                        </div>
                        <div style="font-style: italic; color: #ddd; line-height: 1.4;">
                            "${{essayPreview}}"
                        </div>
                    `;
                    
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 15) + 'px';
                    tooltip.style.top = (event.clientY + 15) + 'px';
                    
                    // Adjust position if tooltip goes off screen
                    const rect = tooltip.getBoundingClientRect();
                    if (rect.right > window.innerWidth) {{
                        tooltip.style.left = (event.clientX - rect.width - 15) + 'px';
                    }}
                    if (rect.bottom > window.innerHeight) {{
                        tooltip.style.top = (event.clientY - rect.height - 15) + 'px';
                    }}
                }}
            }} else {{
                hoveredInstanceId = -1;
                tooltip.style.display = 'none';
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);
        
        // PC filter setup
        const pcFilters = {{}};
        const pcFiltersDiv = document.getElementById('pc-filters');
        
        for (let pc = 1; pc <= 5; pc++) {{
            const filterDiv = document.createElement('div');
            filterDiv.className = 'pc-filter';
            filterDiv.innerHTML = `
                <label>PC${{pc}}: </label>
                <input type="checkbox" id="pc${{pc}}-active">
                <label>Min P:</label>
                <input type="number" id="pc${{pc}}-min" min="0" max="100" value="0" disabled style="width: 50px;">
                <label>Max P:</label>
                <input type="number" id="pc${{pc}}-max" min="0" max="100" value="100" disabled style="width: 50px;">
            `;
            pcFiltersDiv.appendChild(filterDiv);
        }}
        
        // Filter functions
        function shouldPointBeVisible(d) {{
            for (const [pcKey, filter] of Object.entries(pcFilters)) {{
                if (filter.active) {{
                    const pcNum = parseInt(pcKey.substring(2));
                    const percentile = d[`pc${{pcNum}}_percentile`];
                    if (percentile < filter.min || percentile > filter.max) {{
                        return false;
                    }}
                }}
            }}
            return true;
        }}
        
        function updateVisibility() {{
            let visibleCount = 0;
            
            data.forEach((d, i) => {{
                const isVisible = shouldPointBeVisible(d);
                visibilityArray[i] = isVisible;
                
                if (isVisible) {{
                    matrix.makeScale(currentPointSize, currentPointSize, currentPointSize);
                    matrix.setPosition(d.x * currentScaleFactor, d.y * currentScaleFactor, d.z * currentScaleFactor);
                    instancedMesh.setMatrixAt(i, matrix);
                    visibleCount++;
                }} else {{
                    instancedMesh.setMatrixAt(i, zeroMatrix);
                }}
            }});
            
            instancedMesh.instanceMatrix.needsUpdate = true;
            document.getElementById('point-count').textContent = `Points: ${{visibleCount}} / ${{data.length}}`;
        }}
        
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
        
        function resetFilters() {{
            for (let pc = 1; pc <= 5; pc++) {{
                document.getElementById(`pc${{pc}}-active`).checked = false;
                document.getElementById(`pc${{pc}}-min`).value = 0;
                document.getElementById(`pc${{pc}}-max`).value = 100;
                document.getElementById(`pc${{pc}}-min`).disabled = true;
                document.getElementById(`pc${{pc}}-max`).disabled = true;
            }}
            pcFilters = {{}};
            updateVisibility();
        }}
        
        // Display controls
        function updatePointSize() {{
            currentPointSize = parseFloat(document.getElementById('point-size').value);
            document.getElementById('size-val').textContent = currentPointSize.toFixed(1);
            updateVisibility();  // Reapply all transforms with new size
        }}
        
        function updateOpacity() {{
            const opacity = parseFloat(document.getElementById('point-opacity').value);
            document.getElementById('opacity-val').textContent = opacity.toFixed(1);
            material.opacity = opacity;
        }}
        
        function updateCloudScale() {{
            currentScaleFactor = parseFloat(document.getElementById('cloud-scale').value);
            document.getElementById('scale-val').textContent = currentScaleFactor;
            updateVisibility();  // Reapply all transforms with new scale
        }}
        
        // Event listeners
        document.getElementById('point-size').addEventListener('input', updatePointSize);
        document.getElementById('point-opacity').addEventListener('input', updateOpacity);
        document.getElementById('cloud-scale').addEventListener('input', updateCloudScale);
        document.getElementById('color-mode').addEventListener('change', (e) => {{
            currentColorMode = e.target.value;
            updateColors();
        }});
        
        // PC filter listeners
        for (let pc = 1; pc <= 5; pc++) {{
            document.getElementById(`pc${{pc}}-active`).addEventListener('change', () => togglePCFilter(pc));
            document.getElementById(`pc${{pc}}-min`).addEventListener('input', () => {{
                if (pcFilters[`pc${{pc}}`]) {{
                    pcFilters[`pc${{pc}}`].min = parseFloat(document.getElementById(`pc${{pc}}-min`).value);
                    updateVisibility();
                }}
            }});
            document.getElementById(`pc${{pc}}-max`).addEventListener('input', () => {{
                if (pcFilters[`pc${{pc}}`]) {{
                    pcFilters[`pc${{pc}}`].max = parseFloat(document.getElementById(`pc${{pc}}-max`).value);
                    updateVisibility();
                }}
            }});
        }}
        
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

# Format HTML with data
html_content = html_template.format(
    data_json=json.dumps(viz_data),
    ate=dml_summary['ate'],
    ate_se=dml_summary['ate_stderr'],
    y_r2=dml_summary['r2_y'],
    t_auc=dml_summary['r2_t'],
    n_points=len(viz_data)
)

# Save HTML
output_file = OUTPUT_DIR / 'umap_dml_instanced_mesh.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ InstancedMesh visualization saved to: {output_file}")
print("\nKey features:")
print("- Precise hover detection using THREE.InstancedMesh")
print("- Each point is a proper 3D sphere (not abstract Points)")
print("- Color-matched tooltip borders")
print("- 300 characters of essay preview")
print("- Smooth filtering with zero-scale hiding")
print("- No raycasting threshold issues!")