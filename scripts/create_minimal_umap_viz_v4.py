#!/usr/bin/env python3
"""
Minimal UMAP visualization v4 - new color scheme, percentile thresholds, cursor indicator
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

print("=== Creating Minimal UMAP Visualization v4 ===")

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

# Calculate statistics
ai_ratings_clean = essays_df['ai_rating'].dropna()
ai_percentiles = {
    10: ai_ratings_clean.quantile(0.10),
    25: ai_ratings_clean.quantile(0.25),
    75: ai_ratings_clean.quantile(0.75),
    90: ai_ratings_clean.quantile(0.90)
}

print(f"AI Rating range: {ai_ratings_clean.min():.2f} - {ai_ratings_clean.max():.2f}")
print(f"AI Rating percentiles: P10={ai_percentiles[10]:.2f}, P25={ai_percentiles[25]:.2f}, P75={ai_percentiles[75]:.2f}, P90={ai_percentiles[90]:.2f}")
print(f"Social class distribution: {essays_df['sc11'].value_counts().sort_index().to_dict()}")

# Calculate center of point cloud
center_x = X_umap_3d[:, 0].mean()
center_y = X_umap_3d[:, 1].mean()
center_z = X_umap_3d[:, 2].mean()

# Prepare data
viz_data = []
for i in range(len(essays_df)):
    if not pd.isna(essays_df.iloc[i]['sc11']) and not pd.isna(essays_df.iloc[i]['ai_rating']):
        viz_data.append({
            'x': float(X_umap_3d[i, 0]),
            'y': float(X_umap_3d[i, 1]),
            'z': float(X_umap_3d[i, 2]),
            'essay_id': essays_df.iloc[i]['essay_id'],
            'essay': essays_df.iloc[i]['essay'],
            'sc11': int(essays_df.iloc[i]['sc11']),
            'ai_rating': float(essays_df.iloc[i]['ai_rating'])
        })

print(f"Prepared {len(viz_data)} points")
print(f"Cloud center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

# Create HTML
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>UMAP Visualization - Enhanced Controls</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #f0f0f0;
            cursor: none;
        }
        #cursor-indicator {
            position: absolute;
            width: 30px;
            height: 30px;
            border: 2px solid rgba(0, 0, 0, 0.3);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            max-width: 400px;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        #essay-display {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            max-height: 35vh;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            overflow-y: auto;
            display: none;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
            transition: background-color 0.3s;
        }
        #essay-header {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        #essay-text {
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .control-group {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .control-group:last-child {
            border-bottom: none;
        }
        .control-group label {
            display: block;
            margin-bottom: 3px;
            font-weight: bold;
        }
        .threshold-input {
            width: 60px;
            margin: 0 5px;
        }
        .legend-item {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .color-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 8px;
            border: 1px solid #ccc;
        }
        button {
            padding: 5px 10px;
            margin-top: 5px;
            cursor: pointer;
        }
        #counts {
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
        .tab-buttons {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .tab-button {
            padding: 5px 10px;
            background: #e0e0e0;
            border: none;
            cursor: pointer;
            border-radius: 3px;
        }
        .tab-button.active {
            background: #4CAF50;
            color: white;
        }
        .threshold-panel {
            display: none;
        }
        .threshold-panel.active {
            display: block;
        }
    </style>
</head>
<body>
    <div id="cursor-indicator"></div>
    
    <div id="info">
        <h3>UMAP Visualization</h3>
        <div>Total Essays: """ + str(len(viz_data)) + """</div>
        
        <div class="tab-buttons">
            <button class="tab-button active" onclick="switchTab('values')">By Values</button>
            <button class="tab-button" onclick="switchTab('percentiles')">By Percentiles</button>
        </div>
        
        <div id="values-panel" class="threshold-panel active">
            <div class="control-group">
                <label>AI Rating Thresholds:</label>
                <div>
                    Low: &lt; <input type="number" id="ai-low-val" class="threshold-input" value=\"""" + f"{ai_percentiles[25]:.1f}" + """\" min="1" max="10" step="0.1">
                    High: &gt; <input type="number" id="ai-high-val" class="threshold-input" value=\"""" + f"{ai_percentiles[75]:.1f}" + """\" min="1" max="10" step="0.1">
                </div>
            </div>
            
            <div class="control-group">
                <label>Social Class Thresholds:</label>
                <div>
                    Low: ≤ <input type="number" id="sc-low-val" class="threshold-input" value="2" min="1" max="5" step="1">
                    High: ≥ <input type="number" id="sc-high-val" class="threshold-input" value="4" min="1" max="5" step="1">
                </div>
            </div>
        </div>
        
        <div id="percentiles-panel" class="threshold-panel">
            <div class="control-group">
                <label>AI Rating Percentiles:</label>
                <div>
                    Low: P<input type="number" id="ai-low-pct" class="threshold-input" value="25" min="0" max="100" step="5">
                    High: P<input type="number" id="ai-high-pct" class="threshold-input" value="75" min="0" max="100" step="5">
                </div>
            </div>
            
            <div class="control-group">
                <label>Social Class is categorical (1-5)</label>
                <div style="color: #666; font-size: 12px;">
                    Using same thresholds as values tab
                </div>
            </div>
        </div>
        
        <button onclick="updateCategories()">Apply Thresholds</button>
        
        <div style="margin-top: 15px;">
            <div class="legend-item">
                <span class="color-box" style="background: rgba(68,255,68,0.8);"></span>
                <span>High AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: rgba(255,165,0,0.8);"></span>
                <span>High AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: rgba(128,0,128,0.8);"></span>
                <span>Low AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: rgba(255,68,68,0.8);"></span>
                <span>Low AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: rgba(136,136,136,0.5);"></span>
                <span>Middle (neither extreme)</span>
            </div>
        </div>
        
        <div id="counts"></div>
    </div>
    
    <div id="controls">
        <div class="control-group">
            <label>
                <input type="checkbox" id="auto-rotate" checked> Auto-rotate
            </label>
            <div>
                Speed: <input type="range" id="rotate-speed" min="0.1" max="2" step="0.1" value="0.5" style="width: 100px;">
            </div>
        </div>
        <div class="control-group">
            <label>Point Opacity:</label>
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.6" style="width: 100px;">
            <span id="opacity-val">0.6</span>
        </div>
    </div>
    
    <div id="essay-display">
        <div id="essay-header">
            <strong>Essay ID:</strong> <span id="essay-id"></span> | 
            <strong>SC:</strong> <span id="essay-sc"></span> | 
            <strong>AI:</strong> <span id="essay-ai"></span>
        </div>
        <div id="essay-text"></div>
    </div>
    
    <script>
        // Data and center
        const data = """ + json.dumps(viz_data) + """;
        const cloudCenter = {
            x: """ + str(center_x * 100) + """,
            y: """ + str(center_y * 100) + """,
            z: """ + str(center_z * 100) + """
        };
        const aiPercentiles = """ + json.dumps(ai_percentiles) + """;
        
        // Calculate AI rating percentile function
        const aiRatings = data.map(d => d.ai_rating).sort((a, b) => a - b);
        function getAIPercentile(percentile) {
            const index = Math.floor((percentile / 100) * (aiRatings.length - 1));
            return aiRatings[index];
        }
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            2000
        );
        camera.position.set(250, 250, 250);  // Start more zoomed out
        camera.lookAt(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls centered on cloud
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.enableZoom = true;
        controls.zoomSpeed = 1.2;
        controls.minDistance = 50;
        controls.maxDistance = 1000;  // Allow zooming out more
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
        dirLight.position.set(1, 1, 1);
        scene.add(dirLight);
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        
        // Fill positions
        data.forEach((d, i) => {
            positions[i * 3] = d.x * 100;
            positions[i * 3 + 1] = d.y * 100;
            positions[i * 3 + 2] = d.z * 100;
        });
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // Translucent point material
        const material = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.6
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Category system
        const categories = new Array(data.length);
        
        // New color scheme
        const categoryColors = {
            'both_high': [0.27, 1.0, 0.27],      // Green - high in both
            'ai_high': [1.0, 0.65, 0.0],         // Orange - high AI, low SC
            'sc_high': [0.5, 0.0, 0.5],          // Purple - low AI, high SC
            'both_low': [1.0, 0.27, 0.27],       // Red - low in both
            'middle': [0.53, 0.53, 0.53]         // Gray - middle
        };
        
        // Tab switching
        let currentTab = 'values';
        window.switchTab = function(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.threshold-panel').forEach(panel => panel.classList.remove('active'));
            
            document.querySelector(`.tab-button:nth-child(${tab === 'values' ? 1 : 2})`).classList.add('active');
            document.getElementById(`${tab}-panel`).classList.add('active');
        };
        
        function updateCategories() {
            let aiLow, aiHigh, scLow, scHigh;
            
            if (currentTab === 'values') {
                aiLow = parseFloat(document.getElementById('ai-low-val').value);
                aiHigh = parseFloat(document.getElementById('ai-high-val').value);
                scLow = parseInt(document.getElementById('sc-low-val').value);
                scHigh = parseInt(document.getElementById('sc-high-val').value);
            } else {
                const aiLowPct = parseFloat(document.getElementById('ai-low-pct').value);
                const aiHighPct = parseFloat(document.getElementById('ai-high-pct').value);
                aiLow = getAIPercentile(aiLowPct);
                aiHigh = getAIPercentile(aiHighPct);
                scLow = parseInt(document.getElementById('sc-low-val').value);
                scHigh = parseInt(document.getElementById('sc-high-val').value);
            }
            
            const counts = {
                'both_high': 0,
                'ai_high': 0,
                'sc_high': 0,
                'both_low': 0,
                'middle': 0
            };
            
            data.forEach((d, i) => {
                const highAI = d.ai_rating > aiHigh;
                const lowAI = d.ai_rating < aiLow;
                const highSC = d.sc11 >= scHigh;
                const lowSC = d.sc11 <= scLow;
                
                let category;
                if (highAI && highSC) {
                    category = 'both_high';
                } else if (highAI && lowSC) {
                    category = 'ai_high';
                } else if (lowAI && highSC) {
                    category = 'sc_high';
                } else if (lowAI && lowSC) {
                    category = 'both_low';
                } else {
                    category = 'middle';
                }
                
                categories[i] = category;
                counts[category]++;
                
                const color = categoryColors[category];
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            });
            
            geometry.attributes.color.needsUpdate = true;
            
            // Update counts display
            document.getElementById('counts').innerHTML = `
                <strong>Counts:</strong><br>
                High AI + High SC: ${counts.both_high}<br>
                High AI + Low SC: ${counts.ai_high}<br>
                Low AI + High SC: ${counts.sc_high}<br>
                Low AI + Low SC: ${counts.both_low}<br>
                Middle: ${counts.middle}
            `;
        }
        
        // Initialize categories
        updateCategories();
        
        // Custom cursor
        const cursorIndicator = document.getElementById('cursor-indicator');
        
        function updateCursor(event) {
            cursorIndicator.style.left = event.clientX + 'px';
            cursorIndicator.style.top = event.clientY + 'px';
        }
        
        window.addEventListener('mousemove', updateCursor);
        
        // Raycaster for hover
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 8;  // Larger threshold for imprecision
        
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
                    const category = categories[hoveredIndex];
                    
                    // Update essay display
                    document.getElementById('essay-id').textContent = d.essay_id;
                    document.getElementById('essay-sc').textContent = d.sc11;
                    document.getElementById('essay-ai').textContent = d.ai_rating.toFixed(2);
                    document.getElementById('essay-text').textContent = d.essay;
                    
                    // Set background color based on point color
                    const color = categoryColors[category];
                    const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, 0.15)`;
                    document.getElementById('essay-display').style.backgroundColor = bgColor;
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
        
        // Controls
        document.getElementById('auto-rotate').addEventListener('change', (e) => {
            controls.autoRotate = e.target.checked;
        });
        
        document.getElementById('rotate-speed').addEventListener('input', (e) => {
            controls.autoRotateSpeed = parseFloat(e.target.value);
        });
        
        document.getElementById('point-opacity').addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value);
            material.opacity = opacity;
            document.getElementById('opacity-val').textContent = opacity.toFixed(1);
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
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>"""

# Save
output_file = OUTPUT_DIR / 'minimal_umap_viz_v4.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Final visualization saved to: {output_file}")
print("\nFeatures:")
print("- New color scheme: Green=high both, Orange=high AI, Purple=high SC, Red=low both")
print("- Threshold setting by values OR percentiles (tabbed interface)")
print("- More zoomed out starting view (camera at 250,250,250)")
print("- Larger zoom out limit (max distance 1000)")
print("- Custom cursor indicator (30px translucent circle)")
print("- Essay display with matching background color")