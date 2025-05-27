#!/usr/bin/env python3
"""
Fixed version of DML analysis with reliable hover using screen-space detection
Based on the original working script but with improved hover mechanism
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

print("=== Fixed Hover DML PC Analysis ===")

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

# Ensure same order
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

# Load UMAP
with open(CHECKPOINT_DIR / 'nvembed_umap_3d_coords.pkl', 'rb') as f:
    umap_data = pickle.load(f)
    X_umap_3d = umap_data['coords_3d']

print("\n2. Running DML analysis...")

# Top 5 PCs for W
top_pcs = [1, 2, 3, 4, 5]
W = X_pca[:, [pc-1 for pc in top_pcs]]

# Treatment and outcome
T = essays_df['sc11'].values
Y = essays_df['ai_rating'].values

# Drop missing values
valid_idx = ~(np.isnan(T) | np.isnan(Y))
T = T[valid_idx]
Y = Y[valid_idx]
W = W[valid_idx]
X_umap_valid = X_umap_3d[valid_idx]
essays_valid = essays_df[valid_idx].copy()

# Make T binary (lower class vs middle/upper)
T_binary = (T > 2).astype(float)

# XGBoost models
xgb_Y = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_T = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

# Fit DML
est = LinearDML(model_y=xgb_Y, model_t=xgb_T, discrete_treatment=True, cv=5, random_state=42)
est.fit(Y, T_binary, X=None, W=W)

# Get predicted outcomes
Y_pred_cv = cross_val_predict(xgb_Y, W, Y, cv=5)
T_pred_cv = cross_val_predict(xgb_T, W, T_binary, cv=5, method='predict_proba')[:, 1]

# Calculate treatment effects
treatment_effects = est.effect(X=None, T0=0, T1=1)

# Prepare data for visualization
viz_data = []
for i in range(len(essays_valid)):
    # PC percentiles
    pc_data = {}
    for j, pc in enumerate(top_pcs):
        pc_val = W[i, j]
        percentile = stats.percentileofscore(W[:, j], pc_val)
        pc_data[f'pc{pc}_value'] = pc_val
        pc_data[f'pc{pc}_percentile'] = percentile
    
    viz_data.append({
        'x': float(X_umap_valid[i, 0]),
        'y': float(X_umap_valid[i, 1]),
        'z': float(X_umap_valid[i, 2]),
        'essay_id': int(essays_valid.iloc[i]['essay_id']),
        'essay': essays_valid.iloc[i]['essay'][:500] + '...' if len(essays_valid.iloc[i]['essay']) > 500 else essays_valid.iloc[i]['essay'],
        'sc11': int(essays_valid.iloc[i]['sc11']),
        'ai_rating': float(essays_valid.iloc[i]['ai_rating']),
        'y_pred': float(Y_pred_cv[i]),
        't_pred': float(T_pred_cv[i]),
        'treatment_effect': float(treatment_effects[i]),
        **pc_data
    })

# Save results
results = {
    'data': viz_data,
    'dml_summary': {
        'ate': float(est.ate()),
        'ate_se': float(est.ate_stderr()),
        'y_r2': float(r2_score(Y, Y_pred_cv)),
        't_auc': float(np.mean((T_pred_cv > 0.5) == T_binary))
    }
}

with open(OUTPUT_DIR / 'dml_top5_pcs_viz_data_fixed.json', 'w') as f:
    json.dump(results, f)

print(f"\n3. Creating fixed hover visualization...")

# HTML template with screen-space hover detection
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DML Analysis - Top 5 PCs UMAP (Fixed Hover)</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 10px;
            border-radius: 5px;
            display: none;
            pointer-events: none;
            max-width: 400px;
            font-size: 12px;
            z-index: 1000;
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
            max-height: 80vh;
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
        <h3>DML Analysis - Top 5 PCs</h3>
        <div>ATE: {ate:.3f} ± {ate_se:.3f}</div>
        <div>Y Model R²: {y_r2:.3f}</div>
        <div>T Model Accuracy: {t_auc:.3f}</div>
        <div>Points: {n_points}</div>
    </div>
    <div id="controls">
        <h4>Display Controls</h4>
        <div class="control-group">
            <label>Point Size:</label>
            <input type="range" id="point-size" min="0.5" max="10" step="0.5" value="3">
            <span class="value-display" id="size-val">3</span>
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
            <label>Auto Rotate:</label>
            <input type="checkbox" id="auto-rotate">
            <label>Speed:</label>
            <input type="range" id="rotate-speed" min="0.1" max="2" step="0.1" value="0.5" style="width: 80px;">
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
        <div id="pc-filters">
            <!-- PC filters will be added here -->
        </div>
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
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            5000
        );
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
        
        // Create point cloud
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        
        // Store original positions for scaling
        const originalPositions = new Float32Array(data.length * 3);
        let currentScaleFactor = 100;
        
        // Initialize positions
        data.forEach((d, i) => {{
            originalPositions[i * 3] = d.x;
            originalPositions[i * 3 + 1] = d.y;
            originalPositions[i * 3 + 2] = d.z;
            
            positions[i * 3] = d.x * currentScaleFactor;
            positions[i * 3 + 1] = d.y * currentScaleFactor;
            positions[i * 3 + 2] = d.z * currentScaleFactor;
            
            sizes[i] = 3;
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Custom shader material
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
                    gl_PointSize = size * (300.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                uniform float opacity;
                varying vec3 vColor;
                void main() {{
                    float r = length(gl_PointCoord - vec2(0.5));
                    if (r > 0.5) discard;
                    gl_FragColor = vec4(vColor, opacity);
                }}
            `,
            transparent: true,
            vertexColors: true
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
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
                    0 + s * 0.196,     // 0 to 50 (0.196)
                    0.196 + s * (-0.196),  // 50 to 0
                    1 - s * 0.804      // 255 to 50 (0.196)
                ];
            }} else {{
                const s = (t - 0.5) * 2;
                return [
                    0.196 + s * 0.804,  // 50 to 255 (1.0)
                    0 + s * 0.098,      // 0 to 25 (0.098)
                    0.196 - s * 0.196   // 50 to 0
                ];
            }}
        }}
        
        // Initialize colors
        let currentColorMode = 'social_class';
        
        function updateColors() {{
            const legend = document.getElementById('legend');
            
            if (currentColorMode === 'social_class') {{
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
                data.forEach((d, i) => {{
                    const color = getGradientColor(d.sc11, 1, 5);
                    colors[i * 3] = color[0];
                    colors[i * 3 + 1] = color[1];
                    colors[i * 3 + 2] = color[2];
                }});
                legend.innerHTML = '<span style="color: #0032ff;">■</span> SC 1 → <span style="color: #ff1900;">■</span> SC 5';
            }} else if (currentColorMode === 'ai_rating') {{
                const minRating = Math.min(...data.map(d => d.ai_rating));
                const maxRating = Math.max(...data.map(d => d.ai_rating));
                data.forEach((d, i) => {{
                    const color = getGradientColor(d.ai_rating, minRating, maxRating);
                    colors[i * 3] = color[0];
                    colors[i * 3 + 1] = color[1];
                    colors[i * 3 + 2] = color[2];
                }});
                legend.innerHTML = `<span style="color: #0032ff;">■</span> Low → <span style="color: #ff1900;">■</span> High`;
            }} else if (currentColorMode.startsWith('pc')) {{
                const pcNum = parseInt(currentColorMode.substring(2));
                const pcKey = `pc${{pcNum}}_percentile`;
                data.forEach((d, i) => {{
                    const color = getGradientColor(d[pcKey], 0, 100);
                    colors[i * 3] = color[0];
                    colors[i * 3 + 1] = color[1];
                    colors[i * 3 + 2] = color[2];
                }});
                legend.innerHTML = `<span style="color: #0032ff;">■</span> Low PC${{pcNum}} → <span style="color: #ff1900;">■</span> High`;
            }}
            
            geometry.attributes.color.needsUpdate = true;
        }}
        
        // Initialize colors
        updateColors();
        
        // Screen-space hover detection
        let hoveredIndex = -1;
        const projectedPositions = [];
        
        function updateProjectedPositions() {{
            projectedPositions.length = 0;
            const tempVec = new THREE.Vector3();
            
            for (let i = 0; i < data.length; i++) {{
                if (sizes[i] > 0) {{  // Only project visible points
                    tempVec.set(
                        positions[i * 3],
                        positions[i * 3 + 1],
                        positions[i * 3 + 2]
                    );
                    tempVec.project(camera);
                    
                    projectedPositions.push({{
                        x: (tempVec.x + 1) / 2 * window.innerWidth,
                        y: (-tempVec.y + 1) / 2 * window.innerHeight,
                        z: tempVec.z,
                        index: i
                    }});
                }}
            }}
        }}
        
        // Mouse interaction
        const tooltip = document.getElementById('tooltip');
        
        function onMouseMove(event) {{
            const mouseX = event.clientX;
            const mouseY = event.clientY;
            
            // Update projected positions
            updateProjectedPositions();
            
            // Find nearest point in screen space
            let minDist = Infinity;
            let nearestIndex = -1;
            const maxDist = parseFloat(document.getElementById('point-size').value) * 10;
            
            for (const proj of projectedPositions) {{
                if (proj.z > 1 || proj.z < -1) continue;  // Behind camera or too far
                
                const dx = mouseX - proj.x;
                const dy = mouseY - proj.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < minDist && dist < maxDist) {{
                    minDist = dist;
                    nearestIndex = proj.index;
                }}
            }}
            
            if (nearestIndex >= 0 && nearestIndex !== hoveredIndex) {{
                hoveredIndex = nearestIndex;
                const d = data[nearestIndex];
                
                tooltip.innerHTML = `
                    <strong>Essay ID:</strong> ${{d.essay_id}}<br>
                    <strong>Social Class:</strong> ${{d.sc11}}<br>
                    <strong>AI Rating:</strong> ${{d.ai_rating.toFixed(2)}}<br>
                    <strong>Y Pred:</strong> ${{d.y_pred.toFixed(2)}}<br>
                    <strong>T Pred:</strong> ${{d.t_pred.toFixed(3)}}<br>
                    <strong>Treatment Effect:</strong> ${{d.treatment_effect.toFixed(3)}}<br>
                    <strong>PC1:</strong> ${{d.pc1_value.toFixed(2)}} (P${{d.pc1_percentile.toFixed(0)}})<br>
                    <strong>PC2:</strong> ${{d.pc2_value.toFixed(2)}} (P${{d.pc2_percentile.toFixed(0)}})<br>
                    <strong>PC3:</strong> ${{d.pc3_value.toFixed(2)}} (P${{d.pc3_percentile.toFixed(0)}})<br>
                    <strong>PC4:</strong> ${{d.pc4_value.toFixed(2)}} (P${{d.pc4_percentile.toFixed(0)}})<br>
                    <strong>PC5:</strong> ${{d.pc5_value.toFixed(2)}} (P${{d.pc5_percentile.toFixed(0)}})<br>
                    <hr>
                    <small>${{d.essay}}</small>
                `;
                
                tooltip.style.display = 'block';
                tooltip.style.left = (mouseX + 10) + 'px';
                tooltip.style.top = (mouseY + 10) + 'px';
                
                // Adjust tooltip position if it goes off screen
                const rect = tooltip.getBoundingClientRect();
                if (rect.right > window.innerWidth) {{
                    tooltip.style.left = (mouseX - rect.width - 10) + 'px';
                }}
                if (rect.bottom > window.innerHeight) {{
                    tooltip.style.top = (mouseY - rect.height - 10) + 'px';
                }}
            }} else if (nearestIndex < 0) {{
                hoveredIndex = -1;
                tooltip.style.display = 'none';
            }}
        }}
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Display controls
        function updatePointSize() {{
            const size = parseFloat(document.getElementById('point-size').value);
            document.getElementById('size-val').textContent = size;
            for (let i = 0; i < sizes.length; i++) {{
                if (sizes[i] > 0) {{
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
            
            for (let i = 0; i < data.length; i++) {{
                positions[i * 3] = originalPositions[i * 3] * newScale;
                positions[i * 3 + 1] = originalPositions[i * 3 + 1] * newScale;
                positions[i * 3 + 2] = originalPositions[i * 3 + 2] * newScale;
            }}
            
            geometry.attributes.position.needsUpdate = true;
        }}
        
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
            
            applyFilters();
        }}
        
        function applyFilters() {{
            let visibleCount = 0;
            
            data.forEach((d, i) => {{
                let visible = true;
                
                // Check each active PC filter
                for (const [pcKey, filter] of Object.entries(pcFilters)) {{
                    if (filter.active) {{
                        const pcNum = parseInt(pcKey.substring(2));
                        const percentile = d[`pc${{pcNum}}_percentile`];
                        if (percentile < filter.min || percentile > filter.max) {{
                            visible = false;
                            break;
                        }}
                    }}
                }}
                
                sizes[i] = visible ? parseFloat(document.getElementById('point-size').value) : 0;
                if (visible) visibleCount++;
            }});
            
            geometry.attributes.size.needsUpdate = true;
            document.querySelector('#info div:nth-child(5)').textContent = `Points: ${{visibleCount}} / ${{data.length}}`;
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
            applyFilters();
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
                    applyFilters();
                }}
            }});
            document.getElementById(`pc${{pc}}-max`).addEventListener('input', () => {{
                if (pcFilters[`pc${{pc}}`]) {{
                    pcFilters[`pc${{pc}}`].max = parseFloat(document.getElementById(`pc${{pc}}-max`).value);
                    applyFilters();
                }}
            }});
        }}
        
        // Animation loop
        let autoRotate = false;
        let rotateSpeed = 0.5;
        
        document.getElementById('auto-rotate').addEventListener('change', (e) => {{
            autoRotate = e.target.checked;
        }});
        
        document.getElementById('rotate-speed').addEventListener('input', (e) => {{
            rotateSpeed = parseFloat(e.target.value);
        }});
        
        function animate() {{
            requestAnimationFrame(animate);
            
            if (autoRotate) {{
                points.rotation.y += 0.001 * rotateSpeed;
            }}
            
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
    ate=results['dml_summary']['ate'],
    ate_se=results['dml_summary']['ate_se'],
    y_r2=results['dml_summary']['y_r2'],
    t_auc=results['dml_summary']['t_auc'],
    n_points=len(viz_data)
)

# Save HTML
output_file = OUTPUT_DIR / 'umap_dml_top5_pcs_fixed_hover.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Fixed hover visualization saved to: {output_file}")
print("\nThis version uses screen-space detection for reliable hover functionality.")
print("The hover should now work precisely at all zoom levels and with all filters.")