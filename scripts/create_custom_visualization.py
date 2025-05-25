#!/usr/bin/env python3
"""
Create custom 3D visualization using Three.js instead of Plotly
Much more flexible and performant for 9,513 points
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
    """Load all necessary data"""
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
    
    # Add dummy demographics
    df['age'] = 25
    df['female'] = np.random.binomial(1, 0.5, size=len(df))
    df['education_level_numeric'] = df['self_perceived_social_class']
    
    return df

def load_analysis_results():
    """Load UMAP coordinates and SHAP values"""
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

def compute_shap_values(dml_results, X_pca, df):
    """Compute SHAP values for each point"""
    import shap
    
    print("Computing SHAP values...")
    xgb_model = dml_results['AI_ratings']['XGBoost']['model']
    X_res = dml_results['AI_ratings']['XGBoost']['X_res']
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_res)
    
    # Get top features for each point
    top_features_list = []
    for i in range(len(df)):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(df)}...")
        abs_shap = np.abs(shap_values[i])
        top_idx = np.argsort(abs_shap)[-5:][::-1]
        top_features = [(f"PC{idx+1}", float(shap_values[i][idx])) for idx in top_idx]
        top_features_list.append(top_features)
    
    return top_features_list

def create_html_visualization(df, umap_3d, top_features, color_by='ai_rating'):
    """Create custom Three.js visualization"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for JSON
    data_points = []
    for i in range(len(df)):
        point = {
            'id': str(df.iloc[i]['id']),
            'x': float(umap_3d[i, 0]),
            'y': float(umap_3d[i, 1]),
            'z': float(umap_3d[i, 2]),
            'ai_rating': float(df.iloc[i]['ai_average']),
            'actual_sc': int(df.iloc[i]['self_perceived_social_class']),
            'essay_preview': df.iloc[i]['response'][:100] + '...',
            'essay_full': df.iloc[i]['response'],
            'top_features': top_features[i]
        }
        data_points.append(point)
    
    # Scale positions for better visualization (multiply by 10 to make visible)
    positions = np.array([[p['x'], p['y'], p['z']] for p in data_points])
    pos_center = positions.mean(axis=0)
    
    # Center and scale
    for i, point in enumerate(data_points):
        point['x'] = float((point['x'] - pos_center[0]) * 10)
        point['y'] = float((point['y'] - pos_center[1]) * 10)
        point['z'] = float((point['z'] - pos_center[2]) * 10)
    
    # Create HTML with embedded Three.js
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Class UMAP - Custom Visualization</title>
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
        }}
        
        #tooltip h4 {{
            margin: 0 0 8px 0;
            color: #fff;
        }}
        
        #tooltip .feature {{
            color: #aaa;
            margin: 2px 0;
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
        
        #controls h3 {{
            margin: 0 0 15px 0;
            font-size: 14px;
        }}
        
        #controls label {{
            display: block;
            margin: 8px 0;
            cursor: pointer;
            font-size: 13px;
        }}
        
        #controls input[type="checkbox"] {{
            margin-right: 8px;
        }}
        
        #controls input[type="range"] {{
            width: 100%;
            margin: 5px 0;
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
        
        #essay-viewer h3 {{
            margin: 0 0 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        #essay-viewer .close {{
            background: none;
            border: none;
            color: #fff;
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            margin: 0;
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
        
        .color-legend {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .color-bar {{
            width: 100%;
            height: 20px;
            background: linear-gradient(to right, #440154, #3b528b, #21908c, #5dc863, #fde725);
            border-radius: 3px;
            margin: 8px 0;
        }}
        
        .color-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #999;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="tooltip"></div>
    
    <div id="controls">
        <h3>Controls</h3>
        <div style="margin-bottom: 15px;">
            <strong>Filter by Social Class:</strong>
            <label><input type="checkbox" id="sc1" checked> Class 1 (Lowest)</label>
            <label><input type="checkbox" id="sc2" checked> Class 2</label>
            <label><input type="checkbox" id="sc3" checked> Class 3</label>
            <label><input type="checkbox" id="sc4" checked> Class 4</label>
            <label><input type="checkbox" id="sc5" checked> Class 5 (Highest)</label>
        </div>
        
        <div style="margin-bottom: 15px;">
            <strong>Point Opacity:</strong>
            <input type="range" id="opacity" min="0.1" max="1" step="0.1" value="0.7">
            <span id="opacity-value">0.7</span>
        </div>
        
        <div style="margin-bottom: 15px;">
            <strong>Point Size:</strong>
            <input type="range" id="size" min="0.5" max="5" step="0.5" value="2">
            <span id="size-value">2</span>
        </div>
        
        <div class="color-legend">
            <strong>{'AI Rating' if color_by == 'ai_rating' else 'Social Class'}:</strong>
            <div class="color-bar"></div>
            <div class="color-labels">
                <span>{'1' if color_by == 'ai_rating' else '1 (Low)'}</span>
                <span>{'10' if color_by == 'ai_rating' else '5 (High)'}</span>
            </div>
        </div>
    </div>
    
    <div id="essay-viewer">
        <h3>
            <span>Full Essay</span>
            <button class="close" onclick="document.getElementById('essay-viewer').style.display='none'">×</button>
        </h3>
        <div id="essay-content"></div>
        <div id="essay-info" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);"></div>
    </div>
    
    <div id="stats">
        <strong>Dataset:</strong> 9,513 essays<br>
        <strong>Model:</strong> XGBoost R² = 0.923<br>
        <strong>Colored by:</strong> {'AI Rating' if color_by == 'ai_rating' else 'Actual Social Class'}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const data = {json.dumps(data_points)};
        
        // Color configuration
        const colorBy = '{color_by}';
        const colorRange = colorBy === 'ai_rating' ? [1, 10] : [1, 5];
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let particles;
        let raycaster, mouse;
        let hoveredPoint = null;
        let selectedPoint = null;
        
        // Initialize
        init();
        animate();
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                75,
                window.innerWidth / window.innerHeight,
                0.1,
                1000
            );
            camera.position.set(50, 50, 50);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Raycaster for mouse interaction
            raycaster = new THREE.Raycaster();
            raycaster.params.Points.threshold = 3;
            mouse = new THREE.Vector2();
            
            // Create points
            createPoints();
            
            // Add axes helper
            const axesHelper = new THREE.AxesHelper(50);
            scene.add(axesHelper);
            
            // Add ambient light
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            // Events
            window.addEventListener('resize', onWindowResize);
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('click', onMouseClick);
            
            // Filter controls
            ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].forEach(id => {{
                document.getElementById(id).addEventListener('change', updateFilter);
            }});
            
            // Opacity control
            document.getElementById('opacity').addEventListener('input', (e) => {{
                const opacity = parseFloat(e.target.value);
                document.getElementById('opacity-value').textContent = opacity;
                particles.material.opacity = opacity;
            }});
            
            // Size control
            document.getElementById('size').addEventListener('input', (e) => {{
                const size = parseFloat(e.target.value);
                document.getElementById('size-value').textContent = size;
                particles.material.size = size;
            }});
        }}
        
        function createPoints() {{
            console.log('Creating points, data length:', data.length);
            console.log('First point:', data[0]);
            
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            const sizes = [];
            
            // Prepare data
            data.forEach((point, idx) => {{
                positions.push(point.x, point.y, point.z);
                
                // Color based on value
                const value = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                const normalized = (value - colorRange[0]) / (colorRange[1] - colorRange[0]);
                const color = new THREE.Color();
                color.setHSL(0.7 - normalized * 0.7, 0.8, 0.5);
                colors.push(color.r, color.g, color.b);
                
                sizes.push(2);
            }});
            
            console.log('Total positions:', positions.length / 3);
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
            
            // Material
            const material = new THREE.PointsMaterial({{
                size: 2,
                vertexColors: true,
                transparent: true,
                opacity: 0.7,
                sizeAttenuation: true
            }});
            
            // Create points
            particles = new THREE.Points(geometry, material);
            particles.userData = data;
            scene.add(particles);
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
            const filteredData = [];
            
            data.forEach(point => {{
                if (checkboxes[point.actual_sc]) {{
                    positions.push(point.x, point.y, point.z);
                    
                    const value = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                    const normalized = (value - colorRange[0]) / (colorRange[1] - colorRange[0]);
                    const color = new THREE.Color();
                    color.setHSL(0.7 - normalized * 0.7, 0.8, 0.5);
                    colors.push(color.r, color.g, color.b);
                    
                    filteredData.push(point);
                }}
            }});
            
            particles.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            particles.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            particles.userData = filteredData;
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(particles);
            
            const tooltip = document.getElementById('tooltip');
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = particles.userData[idx];
                
                if (point) {{
                    hoveredPoint = point;
                    
                    // Update tooltip content
                    let featuresHtml = point.top_features
                        .map(f => `<div class="feature">${{f[0]}}: ${{f[1].toFixed(3)}}</div>`)
                        .join('');
                    
                    tooltip.innerHTML = `
                        <h4>Essay Preview</h4>
                        <div>${{point.essay_preview}}</div>
                        <div style="margin-top: 10px;">
                            <strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}}<br>
                            <strong>Actual SC:</strong> ${{point.actual_sc}}
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>Top SHAP Features:</strong>
                            ${{featuresHtml}}
                        </div>
                    `;
                    
                    // Position tooltip
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 20) + 'px';
                    tooltip.style.top = (event.clientY - 100) + 'px';
                }}
            }} else {{
                hoveredPoint = null;
                tooltip.style.display = 'none';
            }}
        }}
        
        function onMouseClick(event) {{
            if (hoveredPoint) {{
                selectedPoint = hoveredPoint;
                
                const viewer = document.getElementById('essay-viewer');
                document.getElementById('essay-content').textContent = selectedPoint.essay_full;
                document.getElementById('essay-info').innerHTML = `
                    <strong>AI Rating:</strong> ${{selectedPoint.ai_rating.toFixed(2)}} | 
                    <strong>Actual SC:</strong> ${{selectedPoint.actual_sc}} | 
                    <strong>Essay ID:</strong> ${{selectedPoint.id}}
                `;
                viewer.style.display = 'block';
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
    </script>
</body>
</html>"""
    
    # Save HTML
    filename = f"umap_custom_{color_by}_{timestamp}.html"
    filepath = VIZ_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved: {filepath}")
    return filepath

def main():
    """Create custom visualizations"""
    print("="*60)
    print("Creating Custom Three.js Visualizations")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} essays")
    
    # Load analysis results
    umap_3d, dml_results, X_pca = load_analysis_results()
    
    # Compute SHAP values
    top_features = compute_shap_values(dml_results, X_pca, df)
    
    # Create visualizations
    print("\nCreating custom visualizations...")
    
    # Version 1: Colored by AI rating
    create_html_visualization(df, umap_3d, top_features, color_by='ai_rating')
    
    # Version 2: Colored by actual social class
    create_html_visualization(df, umap_3d, top_features, color_by='actual_sc')
    
    print("\nCustom visualizations complete!")
    print(f"Files saved in: {VIZ_DIR}")

if __name__ == "__main__":
    main()