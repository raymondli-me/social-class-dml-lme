#!/usr/bin/env python3
"""
Minimal working visualization - starting from known good base
Adding ONE feature at a time with testing
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
    
    return df

def create_minimal_html(df, umap_3d, color_by='ai_rating'):
    """Create minimal working visualization - TESTED APPROACH"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare minimal data - ONLY what we need
    scale_factor = 20
    umap_scaled = umap_3d * scale_factor
    
    data_points = []
    for i in range(len(df)):
        point = {
            'x': float(umap_scaled[i, 0]),
            'y': float(umap_scaled[i, 1]),
            'z': float(umap_scaled[i, 2]),
            'ai_rating': float(df.iloc[i]['ai_average']),
            'actual_sc': int(df.iloc[i]['self_perceived_social_class']),
            'essay_preview': df.iloc[i]['response'][:80] + '...',
            'essay_full': df.iloc[i]['response']
        }
        data_points.append(point)
    
    print(f"Created {len(data_points)} data points")
    
    # Create MINIMAL HTML - based on WORKING version
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Social Class UMAP - Minimal Working Version</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
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
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            max-width: 300px;
            pointer-events: none;
            display: none;
            z-index: 1000;
        }}
        
        #controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            font-size: 12px;
            max-width: 200px;
        }}
        
        #controls label {{
            display: block;
            margin: 5px 0;
            cursor: pointer;
        }}
        
        #essay-viewer {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-height: 300px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 15px;
            display: none;
            overflow-y: auto;
            font-size: 14px;
        }}
        
        #debug {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
            font-family: monospace;
        }}
        
        #stats {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        button {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        
        button:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="tooltip"></div>
    
    <div id="debug">
        <div>Loading...</div>
    </div>
    
    <div id="controls">
        <h4>Filters</h4>
        <label><input type="checkbox" id="sc1" checked> Class 1</label>
        <label><input type="checkbox" id="sc2" checked> Class 2</label>
        <label><input type="checkbox" id="sc3" checked> Class 3</label>
        <label><input type="checkbox" id="sc4" checked> Class 4</label>
        <label><input type="checkbox" id="sc5" checked> Class 5</label>
        
        <h4>Controls</h4>
        <label>Size: <input type="range" id="size" min="0.1" max="2" step="0.1" value="0.5"></label>
    </div>
    
    <div id="essay-viewer">
        <h4>
            Essay 
            <button onclick="document.getElementById('essay-viewer').style.display='none'" style="float:right;">×</button>
        </h4>
        <div id="essay-content"></div>
        <div id="essay-info" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;"></div>
    </div>
    
    <div id="stats">
        <strong>Dataset:</strong> 9,513 essays<br>
        <strong>Colored by:</strong> {'AI Rating (1-10)' if color_by == 'ai_rating' else 'Social Class (1-5)'}
    </div>

    <!-- Use EXACT SAME CDN versions -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        console.log('Starting visualization...');
        
        // Data
        const data = {json.dumps(data_points)};
        const colorBy = '{color_by}';
        
        console.log('Data loaded:', data.length, 'points');
        
        // Three.js variables
        let scene, camera, renderer, controls;
        let pointCloud;
        let raycaster, mouse;
        
        // Update debug immediately
        document.querySelector('#debug div').textContent = `Points: ${{data.length}}`;
        
        // Color function - SIMPLE
        function getColor(value, isRating) {{
            const min = isRating ? 1 : 1;
            const max = isRating ? 10 : 5;
            const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
            
            // Simple blue to red gradient
            const hue = (1 - normalized) * 240 / 360;
            const color = new THREE.Color();
            color.setHSL(hue, 0.8, 0.5);
            return color;
        }}
        
        function init() {{
            console.log('Initializing Three.js...');
            
            try {{
                // Scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);
                
                // Camera  
                camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
                camera.position.set(100, 100, 100);
                camera.lookAt(0, 0, 0);
                
                // Renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.getElementById('container').appendChild(renderer.domElement);
                
                console.log('Basic Three.js setup complete');
                
                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                
                console.log('Controls setup complete');
                
                // Create points
                createPoints();
                
                // Grid
                const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222);
                scene.add(gridHelper);
                
                // Lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                // Raycaster for mouse interaction
                raycaster = new THREE.Raycaster();
                raycaster.params.Points.threshold = 1;
                mouse = new THREE.Vector2();
                
                console.log('Setup complete, starting animation');
                
                // Events
                window.addEventListener('resize', onWindowResize);
                window.addEventListener('mousemove', onMouseMove);
                window.addEventListener('click', onMouseClick);
                
                // Controls
                document.getElementById('size').addEventListener('input', (e) => {{
                    if (pointCloud) {{
                        pointCloud.material.size = parseFloat(e.target.value);
                    }}
                }});
                
                ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].forEach(id => {{
                    document.getElementById(id).addEventListener('change', updateFilter);
                }});
                
                // Update debug
                document.querySelector('#debug div').textContent = `Points: ${{data.length}} | Ready`;
                
            }} catch (error) {{
                console.error('Initialization error:', error);
                document.querySelector('#debug div').textContent = `Error: ${{error.message}}`;
            }}
        }}
        
        function createPoints() {{
            console.log('Creating point cloud...');
            
            try {{
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
                
                const material = new THREE.PointsMaterial({{
                    size: 0.5,
                    vertexColors: true,
                    sizeAttenuation: true,
                    transparent: true,
                    opacity: 0.8
                }});
                
                pointCloud = new THREE.Points(geometry, material);
                pointCloud.userData = data;
                scene.add(pointCloud);
                
                console.log('Point cloud created successfully');
                
            }} catch (error) {{
                console.error('Point creation error:', error);
                document.querySelector('#debug div').textContent = `Point Error: ${{error.message}}`;
            }}
        }}
        
        function updateFilter() {{
            if (!pointCloud) return;
            
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
            
            const geometry = pointCloud.geometry;
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
            
            pointCloud.userData = filteredUserData;
            
            document.querySelector('#debug div').textContent = `Points: ${{filteredUserData.length}} / ${{data.length}}`;
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            if (!pointCloud) return;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(pointCloud);
            
            const tooltip = document.getElementById('tooltip');
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = pointCloud.userData[idx];
                
                if (point) {{
                    tooltip.innerHTML = `
                        <strong>Preview:</strong><br>
                        ${{point.essay_preview}}<br><br>
                        <strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}}<br>
                        <strong>Social Class:</strong> ${{point.actual_sc}}
                    `;
                    
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 20) + 'px';
                    tooltip.style.top = (event.clientY - 80) + 'px';
                }}
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        function onMouseClick() {{
            if (!pointCloud) return;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(pointCloud);
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = pointCloud.userData[idx];
                
                if (point) {{
                    document.getElementById('essay-content').textContent = point.essay_full;
                    document.getElementById('essay-info').innerHTML = `
                        <strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}} | 
                        <strong>Social Class:</strong> ${{point.actual_sc}}
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
        
        // Initialize everything
        init();
        animate();
        
        console.log('Visualization started');
        
    </script>
</body>
</html>"""
    
    # Save HTML
    filename = f"umap_minimal_{color_by}_{timestamp}.html"
    filepath = VIZ_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved: {filepath}")
    return filepath

def main():
    """Create minimal working visualization"""
    print("="*60)
    print("Creating MINIMAL Working Visualization")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} essays")
    
    # Load UMAP (skip SHAP for now - add later)
    umap_3d = np.load(CHECKPOINT_DIR / "umap_3d_openai.npy")
    print(f"UMAP shape: {umap_3d.shape}")
    
    # Create minimal visualizations
    print("\nCreating minimal visualizations...")
    
    # Start with AI rating version only
    create_minimal_html(df, umap_3d, color_by='ai_rating')
    
    print("\nMinimal visualization complete!")
    print("NEXT: Test this in browser, then add ONE feature at a time")

if __name__ == "__main__":
    main()