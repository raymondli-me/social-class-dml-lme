#!/usr/bin/env python3
"""
Simplified version with better hover detection and more text preview
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

print("=== Simple Hover DML PC Analysis ===")

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

# Load PCA features and UMAP
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
    pca_model = pca_data['pca']
    essay_ids = pca_data['essay_ids']

with open(CHECKPOINT_DIR / 'nvembed_umap_3d_coords.pkl', 'rb') as f:
    umap_data = pickle.load(f)
    X_umap_3d = umap_data['coords_3d']

# Ensure same order
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

print("\n2. Running DML analysis...")

# Top 5 PCs
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

# Make T binary
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
        'essay': essays_valid.iloc[i]['essay'],
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

print(f"\n3. Creating simple hover visualization...")

# HTML template with improved hover
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DML Analysis - Simple Hover</title>
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
            <input type="range" id="point-size" min="1" max="15" step="0.5" value="5">
            <span class="value-display" id="size-val">5</span>
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
            
            sizes[i] = 5;
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
        
        // Simple distance-based hover detection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 5;
        
        const tooltip = document.getElementById('tooltip');
        let hoveredIndex = -1;
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            // Update raycaster threshold based on current point size
            const currentSize = parseFloat(document.getElementById('point-size').value);
            raycaster.params.Points.threshold = currentSize;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {{
                // Find closest visible point
                let closestIntersect = null;
                let minDistance = Infinity;
                
                for (const intersect of intersects) {{
                    if (sizes[intersect.index] > 0 && intersect.distance < minDistance) {{
                        minDistance = intersect.distance;
                        closestIntersect = intersect;
                    }}
                }}
                
                if (closestIntersect) {{
                    const idx = closestIntersect.index;
                    if (idx !== hoveredIndex) {{
                        hoveredIndex = idx;
                        const d = data[idx];
                        
                        // Get the color of the hovered point
                        const r = Math.floor(colors[idx * 3] * 255);
                        const g = Math.floor(colors[idx * 3 + 1] * 255);
                        const b = Math.floor(colors[idx * 3 + 2] * 255);
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
                    hoveredIndex = -1;
                    tooltip.style.display = 'none';
                }}
            }} else {{
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
        
        // Event listeners
        document.getElementById('point-size').addEventListener('input', updatePointSize);
        document.getElementById('point-opacity').addEventListener('input', updateOpacity);
        document.getElementById('cloud-scale').addEventListener('input', updateCloudScale);
        document.getElementById('color-mode').addEventListener('change', (e) => {{
            currentColorMode = e.target.value;
            updateColors();
        }});
        
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
    ate=results['dml_summary']['ate'],
    ate_se=results['dml_summary']['ate_se'],
    y_r2=results['dml_summary']['y_r2'],
    t_auc=results['dml_summary']['t_auc'],
    n_points=len(viz_data)
)

# Save HTML
output_file = OUTPUT_DIR / 'umap_dml_simple_hover.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Simple hover visualization saved to: {output_file}")
print("\nFeatures:")
print("- Color-matched tooltip border")
print("- 300 characters of essay preview")
print("- Simplified hover detection")
print("- No yellow mesh highlight")