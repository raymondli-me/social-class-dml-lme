#!/usr/bin/env python3
"""
Minimal UMAP visualization v2 - sidebar, auto-rotate, new color scheme
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'

print("=== Creating Minimal UMAP Visualization v2 ===")

# Load essays and social class
print("Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Load UMAP coordinates
with open(OUTPUT_DIR / 'dml_pc_analysis_results_with_umap.pkl', 'rb') as f:
    umap_results = pickle.load(f)
    X_umap_3d = umap_results['umap_3d']

# Get essay IDs from PCA data to ensure order
with open(BASE_DIR / 'nvembed_checkpoints' / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    essay_ids = pca_data['essay_ids']

# Align data
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

# Calculate quartiles for AI ratings
ai_q1 = essays_df['ai_rating'].quantile(0.25)
ai_q3 = essays_df['ai_rating'].quantile(0.75)

print(f"AI Rating quartiles: Q1={ai_q1:.2f}, Q3={ai_q3:.2f}")
print(f"Social class distribution: {essays_df['sc11'].value_counts().sort_index().to_dict()}")

# Prepare data with new categories
viz_data = []
for i in range(len(essays_df)):
    if not pd.isna(essays_df.iloc[i]['sc11']) and not pd.isna(essays_df.iloc[i]['ai_rating']):
        ai_rating = essays_df.iloc[i]['ai_rating']
        sc = essays_df.iloc[i]['sc11']
        
        # Categorize based on AI rating and social class
        high_ai = ai_rating >= ai_q3  # Top quartile
        high_sc = sc >= 4  # Upper-middle and Upper classes
        
        if high_ai and high_sc:
            category = 'both_high'
        elif high_ai and not high_sc:
            category = 'ai_high'
        elif not high_ai and high_sc:
            category = 'sc_high'
        else:
            category = 'both_low'
        
        viz_data.append({
            'x': float(X_umap_3d[i, 0]),
            'y': float(X_umap_3d[i, 1]),
            'z': float(X_umap_3d[i, 2]),
            'essay_id': essays_df.iloc[i]['essay_id'],
            'essay': essays_df.iloc[i]['essay'],
            'sc11': int(essays_df.iloc[i]['sc11']),
            'ai_rating': float(essays_df.iloc[i]['ai_rating']),
            'category': category
        })

print(f"Prepared {len(viz_data)} points")

# Count categories
category_counts = {}
for d in viz_data:
    cat = d['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1
print(f"Category distribution: {category_counts}")

# Create HTML with sidebar
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>UMAP Visualization - Enhanced</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #f0f0f0;
            display: flex;
        }
        #main-view {
            flex: 1;
            position: relative;
        }
        #sidebar {
            width: 400px;
            background: rgba(255,255,255,0.95);
            box-shadow: -2px 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 5px;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        #controls {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 5px;
            font-size: 13px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        #essay-header {
            padding: 15px;
            background: #333;
            color: white;
            font-size: 16px;
            font-weight: bold;
            margin: 0;
        }
        #essay-content {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            font-size: 14px;
            line-height: 1.6;
        }
        #essay-text {
            white-space: pre-wrap;
            color: #333;
        }
        .legend-item {
            margin: 5px 0;
        }
        .color-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 8px;
            vertical-align: middle;
        }
        #no-selection {
            color: #888;
            font-style: italic;
            text-align: center;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div id="main-view">
        <div id="info">
            <h3>UMAP Visualization</h3>
            <div>Essays: """ + str(len(viz_data)) + """</div>
            <div style="margin-top: 15px;">
                <div class="legend-item">
                    <span class="color-box" style="background: #ff4444;"></span>
                    High AI + High SC (""" + str(category_counts.get('both_high', 0)) + """)
                </div>
                <div class="legend-item">
                    <span class="color-box" style="background: #4444ff;"></span>
                    High AI + Low SC (""" + str(category_counts.get('ai_high', 0)) + """)
                </div>
                <div class="legend-item">
                    <span class="color-box" style="background: #44ff44;"></span>
                    Low AI + High SC (""" + str(category_counts.get('sc_high', 0)) + """)
                </div>
                <div class="legend-item">
                    <span class="color-box" style="background: #888888;"></span>
                    Low AI + Low SC (""" + str(category_counts.get('both_low', 0)) + """)
                </div>
            </div>
        </div>
        
        <div id="controls">
            <label>
                <input type="checkbox" id="auto-rotate" checked> Auto-rotate
            </label>
            <br>
            <label>
                Speed: <input type="range" id="rotate-speed" min="0.1" max="2" step="0.1" value="0.5" style="width: 100px;">
            </label>
        </div>
    </div>
    
    <div id="sidebar">
        <div id="essay-header">Essay Viewer</div>
        <div id="essay-content">
            <div id="no-selection">Hover over a point to view essay</div>
            <div id="essay-details" style="display: none;">
                <div style="margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
                    <strong>Essay ID:</strong> <span id="essay-id"></span><br>
                    <strong>Social Class:</strong> <span id="essay-sc"></span><br>
                    <strong>AI Rating:</strong> <span id="essay-ai"></span>
                </div>
                <div id="essay-text"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Data
        const data = """ + json.dumps(viz_data) + """;
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const mainView = document.getElementById('main-view');
        const camera = new THREE.PerspectiveCamera(
            75, 
            mainView.clientWidth / mainView.clientHeight, 
            0.1, 
            1000
        );
        camera.position.set(150, 150, 150);
        camera.lookAt(0, 0, 0);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(mainView.clientWidth, mainView.clientHeight);
        mainView.appendChild(renderer.domElement);
        
        // Controls with zoom enabled
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.enableZoom = true;
        controls.zoomSpeed = 1.2;
        controls.minDistance = 50;
        controls.maxDistance = 500;
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
        dirLight.position.set(1, 1, 1);
        scene.add(dirLight);
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        
        // Category colors
        const categoryColors = {
            'both_high': [1.0, 0.27, 0.27],    // Red
            'ai_high': [0.27, 0.27, 1.0],      // Blue
            'sc_high': [0.27, 1.0, 0.27],      // Green
            'both_low': [0.53, 0.53, 0.53]     // Gray
        };
        
        // Fill positions and colors
        data.forEach((d, i) => {
            positions[i * 3] = d.x * 100;
            positions[i * 3 + 1] = d.y * 100;
            positions[i * 3 + 2] = d.z * 100;
            
            const color = categoryColors[d.category];
            colors[i * 3] = color[0];
            colors[i * 3 + 1] = color[1];
            colors[i * 3 + 2] = color[2];
        });
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // Simple point material
        const material = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            sizeAttenuation: true
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Raycaster for hover
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 5;
        
        let hoveredIndex = -1;
        
        function onMouseMove(event) {
            // Calculate mouse position relative to the main view
            const rect = mainView.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {
                const newIndex = intersects[0].index;
                if (newIndex !== hoveredIndex) {
                    hoveredIndex = newIndex;
                    const d = data[hoveredIndex];
                    
                    // Update sidebar
                    document.getElementById('no-selection').style.display = 'none';
                    document.getElementById('essay-details').style.display = 'block';
                    
                    document.getElementById('essay-id').textContent = d.essay_id;
                    document.getElementById('essay-sc').textContent = d.sc11;
                    document.getElementById('essay-ai').textContent = d.ai_rating.toFixed(2);
                    document.getElementById('essay-text').textContent = d.essay;
                }
            } else {
                if (hoveredIndex !== -1) {
                    hoveredIndex = -1;
                    document.getElementById('no-selection').style.display = 'block';
                    document.getElementById('essay-details').style.display = 'none';
                }
            }
        }
        
        mainView.addEventListener('mousemove', onMouseMove);
        
        // Controls
        document.getElementById('auto-rotate').addEventListener('change', (e) => {
            controls.autoRotate = e.target.checked;
        });
        
        document.getElementById('rotate-speed').addEventListener('input', (e) => {
            controls.autoRotateSpeed = parseFloat(e.target.value);
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = mainView.clientWidth / mainView.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(mainView.clientWidth, mainView.clientHeight);
        });
    </script>
</body>
</html>"""

# Save
output_file = OUTPUT_DIR / 'minimal_umap_viz_v2.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Enhanced visualization saved to: {output_file}")
print("\nFeatures:")
print("- Fixed sidebar for essay display")
print("- Auto-rotating point cloud")
print("- New 4-color scheme based on AI/SC quartiles")
print(f"- AI rating quartiles: Q1={ai_q1:.2f}, Q3={ai_q3:.2f}")
print("- Categories: High AI+SC (red), High AI only (blue), High SC only (green), Low both (gray)")