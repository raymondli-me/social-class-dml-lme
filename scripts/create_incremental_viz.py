#!/usr/bin/env python3
"""
Start from WORKING version, add features incrementally
Based on: create_custom_visualization_fixed.py (KNOWN WORKING)
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime

# Define paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"
VIZ_DIR = BASE_DIR / "custom_visualizations"
VIZ_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all necessary data - SAME AS WORKING VERSION"""
    print("Loading data...")
    
    # Load essays
    df = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
    
    # Load social class labels
    sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    df = df.merge(sc_labels, on='TID', how='left')
    df = df.rename(columns={'TID': 'id', 'original': 'response', 'sc11': 'self_perceived_social_class'})
    
    # Load AI ratings
    ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['id', 'ai_average']
    df = df.merge(ai_avg, on='id', how='left')
    
    return df

def load_analysis_results():
    """Load UMAP coordinates - SAME AS WORKING VERSION"""
    print("Loading analysis results...")
    
    # Load UMAP
    umap_3d = np.load(CHECKPOINT_DIR / "umap_3d_openai.npy")
    
    # Load DML results for SHAP
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        dml_results = pickle.load(f)
    
    # Load PCA features
    with open(CHECKPOINT_DIR / "pca_200_features.pkl", 'rb') as f:
        pca_data = pickle.load(f)
        X_pca = pca_data['features']
    
    return umap_3d, dml_results, X_pca

def compute_shap_simple(dml_results, X_pca, df):
    """Compute SHAP values SIMPLY - not blocking main thread"""
    import shap
    import xgboost as xgb
    
    print("Computing simple SHAP values...")
    
    # Train simple model on SAMPLE of data for speed
    sample_size = min(1000, len(df))
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=1)
    model.fit(X_pca[sample_idx], df['ai_average'].iloc[sample_idx])
    
    # Get SHAP for full dataset but with simple model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pca)
    
    # Get top features for each point (quick version)
    top_features_list = []
    for i in range(len(df)):
        abs_shap = np.abs(shap_values[i])
        top_idx = np.argsort(abs_shap)[-3:][::-1]  # Only top 3
        top_features = [(f"PC{idx+1}", float(shap_values[i][idx])) for idx in top_idx]
        top_features_list.append(top_features)
    
    return top_features_list

def create_html_step1(df, umap_3d, top_features, color_by='ai_rating'):
    """STEP 1: Start with WORKING version, add ONLY color legend"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data - SAME AS WORKING VERSION
    data_points = []
    scale_factor = 20
    umap_scaled = umap_3d * scale_factor
    
    for i in range(len(df)):
        point = {
            'id': str(df.iloc[i]['id']),
            'x': float(umap_scaled[i, 0]),
            'y': float(umap_scaled[i, 1]),
            'z': float(umap_scaled[i, 2]),
            'ai_rating': float(df.iloc[i]['ai_average']),
            'actual_sc': int(df.iloc[i]['self_perceived_social_class']),
            'essay_preview': df.iloc[i]['response'][:100] + '...',
            'essay_full': df.iloc[i]['response'],
            'top_features': top_features[i]
        }
        data_points.append(point)
    
    # HTML - EXACT COPY of working version + color legend ONLY
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Class UMAP - Step 1: Add Color Legend</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #000;
            color: #fff;
            overflow: hidden;
        }}
        
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        
        #tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            max-width: 300px;
            pointer-events: none;
            display: none;
            backdrop-filter: blur(10px);
            z-index: 1000;
            transition: background-color 0.3s ease;
        }}
        
        #controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            z-index: 100;
        }}
        
        #essay-viewer {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-height: 300px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 20px;
            display: none;
            overflow-y: auto;
            backdrop-filter: blur(10px);
            z-index: 100;
        }}
        
        #debug {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 8px;
            font-size: 11px;
            font-family: monospace;
        }}
        
        #stats {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
        }}
        
        /* NEW: Color legend styles */
        #color-legend {{
            position: absolute;
            bottom: 120px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 3px 0;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 2px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="tooltip"></div>
    
    <div id="debug">
        <div id="debug-info">Loading...</div>
    </div>
    
    <div id="controls">
        <h3>Controls</h3>
        <div style="margin-bottom: 15px;">
            <strong>Filter by Social Class:</strong>
            <label><input type="checkbox" id="sc1" checked> Class 1</label>
            <label><input type="checkbox" id="sc2" checked> Class 2</label>
            <label><input type="checkbox" id="sc3" checked> Class 3</label>
            <label><input type="checkbox" id="sc4" checked> Class 4</label>
            <label><input type="checkbox" id="sc5" checked> Class 5</label>
        </div>
        
        <div style="margin-bottom: 15px;">
            <strong>Point Size:</strong>
            <input type="range" id="size" min="0.1" max="2" step="0.1" value="0.5">
            <span id="size-value">0.5</span>
        </div>
    </div>
    
    <div id="essay-viewer">
        <h3>
            <span>Full Essay</span>
            <button onclick="document.getElementById('essay-viewer').style.display='none'" style="float:right;background:none;border:none;color:#fff;font-size:20px;cursor:pointer;">×</button>
        </h3>
        <div id="essay-content"></div>
        <div id="essay-info" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);"></div>
    </div>
    
    <div id="stats">
        <strong>Dataset:</strong> 9,513 essays<br>
        <strong>Colored by:</strong> {'AI Rating (1-10)' if color_by == 'ai_rating' else 'Social Class (1-5)'}
    </div>
    
    <!-- NEW: Color legend -->
    <div id="color-legend">
        <strong>Color Legend:</strong>
        <div id="legend-items"></div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const data = {json.dumps(data_points)};
        const colorBy = '{color_by}';
        
        // Three.js variables
        let scene, camera, renderer, controls;
        let pointCloud;
        let raycaster, mouse;
        
        // NEW: Color functions
        function getColor(value, isRating) {{
            const min = isRating ? 1 : 1;
            const max = isRating ? 10 : 5;
            const normalized = (value - min) / (max - min);
            
            const hue = (1 - normalized) * 240 / 360; // Blue to red
            const color = new THREE.Color();
            color.setHSL(hue, 0.8, 0.5);
            return color;
        }}
        
        function getColorRGB(value, isRating) {{
            const color = getColor(value, isRating);
            return `rgb(${{Math.floor(color.r * 255)}}, ${{Math.floor(color.g * 255)}}, ${{Math.floor(color.b * 255)}})`;
        }}
        
        // NEW: Create color legend
        function createColorLegend() {{
            const legendItems = document.getElementById('legend-items');
            legendItems.innerHTML = '';
            
            if (colorBy === 'ai_rating') {{
                // Show subset for AI ratings (1, 3, 5, 7, 9, 10)
                [1, 3, 5, 7, 9, 10].forEach(i => {{
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    item.innerHTML = `
                        <div class="legend-color" style="background: ${{getColorRGB(i, true)}}"></div>
                        <span>${{i}}</span>
                    `;
                    legendItems.appendChild(item);
                }});
            }} else {{
                // Show all 5 social classes
                const labels = ['Lowest', 'Low', 'Middle', 'High', 'Highest'];
                for (let i = 1; i <= 5; i++) {{
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    item.innerHTML = `
                        <div class="legend-color" style="background: ${{getColorRGB(i, false)}}"></div>
                        <span>Class ${{i}} (${{labels[i-1]}})</span>
                    `;
                    legendItems.appendChild(item);
                }}
            }}
        }}
        
        function init() {{
            // Debug info
            document.getElementById('debug-info').innerHTML = `
                Points: ${{data.length}}<br>
                First: (${{data[0].x.toFixed(1)}}, ${{data[0].y.toFixed(1)}}, ${{data[0].z.toFixed(1)}})
            `;
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            scene.fog = new THREE.Fog(0x000000, 200, 600);
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                50,
                window.innerWidth / window.innerHeight,
                0.1,
                2000
            );
            camera.position.set(100, 100, 100);
            camera.lookAt(0, 0, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.rotateSpeed = 0.5;
            controls.target.set(0, 0, 0);
            
            // Create point cloud
            createPointCloud();
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const pointLight = new THREE.PointLight(0xffffff, 0.4);
            pointLight.position.set(50, 50, 50);
            scene.add(pointLight);
            
            // Grid helper
            const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222);
            scene.add(gridHelper);
            
            // Axes helper
            const axesHelper = new THREE.AxesHelper(100);
            scene.add(axesHelper);
            
            // Raycaster
            raycaster = new THREE.Raycaster();
            raycaster.params.Points.threshold = 1;
            mouse = new THREE.Vector2();
            
            // Events
            window.addEventListener('resize', onWindowResize);
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('click', onMouseClick);
            
            // Controls
            document.getElementById('size').addEventListener('input', (e) => {{
                const size = parseFloat(e.target.value);
                document.getElementById('size-value').textContent = size;
                if (pointCloud) {{
                    pointCloud.material.size = size;
                }}
            }});
            
            ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].forEach(id => {{
                document.getElementById(id).addEventListener('change', updateFilter);
            }});
            
            // NEW: Create color legend after init
            createColorLegend();
        }}
        
        function createPointCloud() {{
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(data.length * 3);
            const colors = new Float32Array(data.length * 3);
            
            // Fill arrays
            data.forEach((point, i) => {{
                positions[i * 3] = point.x;
                positions[i * 3 + 1] = point.y;
                positions[i * 3 + 2] = point.z;
                
                const value = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                const color = getColor(value, colorBy === 'ai_rating');
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }});
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            geometry.computeBoundingSphere();
            
            // Material
            const material = new THREE.PointsMaterial({{
                size: 0.5,
                vertexColors: true,
                sizeAttenuation: true,
                transparent: true,
                opacity: 0.8,
                blending: THREE.AdditiveBlending
            }});
            
            // Create points
            pointCloud = new THREE.Points(geometry, material);
            pointCloud.userData = data;
            scene.add(pointCloud);
            
            // Update debug
            const bounds = geometry.boundingSphere;
            document.getElementById('debug-info').innerHTML += `<br>Radius: ${{bounds.radius.toFixed(1)}}`;
        }}
        
        function updateFilter() {{
            const checkboxes = {{
                1: document.getElementById('sc1').checked,
                2: document.getElementById('sc2').checked,
                3: document.getElementById('sc3').checked,
                4: document.getElementById('sc4').checked,
                5: document.getElementById('sc5').checked
            }};
            
            const positions = [];
            const colors = [];
            const filteredUserData = [];
            
            data.forEach((point, i) => {{
                if (checkboxes[point.actual_sc]) {{
                    positions.push(point.x, point.y, point.z);
                    
                    const value = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                    const color = getColor(value, colorBy === 'ai_rating');
                    colors.push(color.r, color.g, color.b);
                    
                    filteredUserData.push(point);
                }}
            }});
            
            // Update geometry
            const geometry = pointCloud.geometry;
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
            geometry.computeBoundingSphere();
            
            // Update user data
            pointCloud.userData = filteredUserData;
            
            // Update debug
            document.getElementById('debug-info').innerHTML = `
                Points: ${{filteredUserData.length}} / ${{data.length}}<br>
                Filtered: ${{Object.values(checkboxes).filter(v => !v).length > 0 ? 'Yes' : 'No'}}
            `;
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(pointCloud);
            
            const tooltip = document.getElementById('tooltip');
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = pointCloud.userData[idx];
                
                if (point) {{
                    let featuresHtml = point.top_features
                        .map(f => `<div>${{f[0]}}: ${{f[1] > 0 ? '+' : ''}}${{f[1].toFixed(3)}}</div>`)
                        .join('');
                    
                    tooltip.innerHTML = `
                        <strong>Essay Preview:</strong><br>
                        ${{point.essay_preview}}<br><br>
                        <strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}}<br>
                        <strong>Actual SC:</strong> ${{point.actual_sc}}<br><br>
                        <strong>Top SHAP Features:</strong><br>
                        ${{featuresHtml}}
                    `;
                    
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 20) + 'px';
                    tooltip.style.top = (event.clientY - 100) + 'px';
                }}
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        function onMouseClick() {{
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(pointCloud);
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = pointCloud.userData[idx];
                
                if (point) {{
                    document.getElementById('essay-content').textContent = point.essay_full;
                    document.getElementById('essay-info').innerHTML = `
                        <strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}} | 
                        <strong>Actual SC:</strong> ${{point.actual_sc}} | 
                        <strong>ID:</strong> ${{point.id}}
                    `;
                    document.getElementById('essay-viewer').style.display = 'block';
                }}
            }}
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Initialize
        init();
        animate();
    </script>
</body>
</html>"""
    
    # Save HTML
    filename = f"umap_step1_{color_by}_{timestamp}.html"
    filepath = VIZ_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved: {filepath}")
    return filepath

def main():
    """Create Step 1: Working version + color legend only"""
    print("="*60)
    print("STEP 1: Start from WORKING version, add color legend ONLY")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} essays")
    
    # Load analysis results
    umap_3d, dml_results, X_pca = load_analysis_results()
    
    # Compute simple SHAP (fast version)
    top_features = compute_shap_simple(dml_results, X_pca, df)
    
    # Create step 1 visualization
    print("\nCreating Step 1 visualization...")
    
    # AI rating version only for now
    create_html_step1(df, umap_3d, top_features, color_by='ai_rating')
    
    print("\nStep 1 complete! TEST THIS BEFORE PROCEEDING")
    print("If working, we can add next feature (color-coded backgrounds)")

if __name__ == "__main__":
    main()