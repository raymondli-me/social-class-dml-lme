#!/usr/bin/env python3
"""
Minimal UMAP visualization - just points and hover with full text
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

print("=== Creating Minimal UMAP Visualization ===")

# Load essays and social class
print("Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

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

# Prepare simple data
viz_data = []
for i in range(len(essays_df)):
    if not pd.isna(essays_df.iloc[i]['sc11']):  # Skip missing social class
        viz_data.append({
            'x': float(X_umap_3d[i, 0]),
            'y': float(X_umap_3d[i, 1]),
            'z': float(X_umap_3d[i, 2]),
            'essay_id': essays_df.iloc[i]['essay_id'],
            'essay': essays_df.iloc[i]['essay'],
            'sc11': int(essays_df.iloc[i]['sc11'])
        })

print(f"Prepared {len(viz_data)} points")

# Create minimal HTML
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Minimal UMAP Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #f0f0f0;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        #essay-display {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            max-height: 40vh;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            overflow-y: auto;
            display: none;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        #essay-display h4 {
            margin-top: 0;
            color: #333;
        }
        #essay-text {
            line-height: 1.5;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>UMAP Visualization</h3>
        <div>Essays: """ + str(len(viz_data)) + """</div>
        <div style="margin-top: 10px; font-size: 12px;">
            <span style="color: #3333cc;">■</span> Lower |
            <span style="color: #33cccc;">■</span> Working |
            <span style="color: #33cc33;">■</span> Middle |
            <span style="color: #cccc33;">■</span> Upper-middle |
            <span style="color: #cc3333;">■</span> Upper
        </div>
    </div>
    
    <div id="essay-display">
        <h4 id="essay-header">Essay</h4>
        <div id="essay-text"></div>
    </div>
    
    <script>
        // Data
        const data = """ + json.dumps(viz_data) + """;
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            1000
        );
        camera.position.set(150, 150, 150);
        camera.lookAt(0, 0, 0);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
        dirLight.position.set(1, 1, 1);
        scene.add(dirLight);
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        
        // Social class colors
        const scColors = {
            1: [0.2, 0.2, 0.8],  // Lower - Blue
            2: [0.2, 0.8, 0.8],  // Working - Cyan
            3: [0.2, 0.8, 0.2],  // Middle - Green
            4: [0.8, 0.8, 0.2],  // Upper-middle - Yellow
            5: [0.8, 0.2, 0.2]   // Upper - Red
        };
        
        // Fill positions and colors
        data.forEach((d, i) => {
            positions[i * 3] = d.x * 100;
            positions[i * 3 + 1] = d.y * 100;
            positions[i * 3 + 2] = d.z * 100;
            
            const color = scColors[d.sc11];
            colors[i * 3] = color[0];
            colors[i * 3 + 1] = color[1];
            colors[i * 3 + 2] = color[2];
        });
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // Simple point material
        const material = new THREE.PointsMaterial({
            size: 3,
            vertexColors: true,
            sizeAttenuation: true
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Raycaster for hover
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 3;
        
        let hoveredIndex = -1;
        
        function onMouseMove(event) {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {
                const newIndex = intersects[0].index;
                if (newIndex !== hoveredIndex) {
                    hoveredIndex = newIndex;
                    const d = data[hoveredIndex];
                    
                    // Show essay
                    document.getElementById('essay-header').textContent = 
                        `Essay ${d.essay_id} - Social Class ${d.sc11}`;
                    document.getElementById('essay-text').textContent = d.essay;
                    document.getElementById('essay-display').style.display = 'block';
                }
            } else {
                if (hoveredIndex !== -1) {
                    hoveredIndex = -1;
                    document.getElementById('essay-display').style.display = 'none';
                }
            }
        }
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>"""

# Save
output_file = OUTPUT_DIR / 'minimal_umap_viz.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Minimal visualization saved to: {output_file}")
print("\nFeatures:")
print("- Simple THREE.Points rendering")
print("- Full essay text shown on hover")
print("- Social class coloring")
print("- Clean, minimal interface")