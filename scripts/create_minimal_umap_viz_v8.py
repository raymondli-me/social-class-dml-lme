#!/usr/bin/env python3
"""
Minimal UMAP visualization v8 - proximity-based gallery, proper rotation center
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

print("=== Creating Minimal UMAP Visualization v8 ===")

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
    <title>UMAP Visualization - Proximity Gallery</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #000;
            cursor: none;
            color: #fff;
        }
        #cursor-indicator {
            position: absolute;
            width: 30px;
            height: 30px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 400px;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        #gallery-controls {
            position: absolute;
            top: 50%;
            right: 10px;
            transform: translateY(-50%);
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        #essay-display {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            max-height: 35vh;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            overflow-y: auto;
            display: none;
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
        }
        #essay-header {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        #essay-text {
            line-height: 1.6;
            white-space: pre-wrap;
            color: #ddd;
        }
        .control-group {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
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
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 2px 5px;
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
            border: 1px solid rgba(255,255,255,0.3);
        }
        button {
            padding: 5px 10px;
            margin-top: 5px;
            cursor: pointer;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
        }
        button:hover {
            background: rgba(255,255,255,0.2);
        }
        #counts {
            margin-top: 10px;
            font-size: 12px;
            color: #ccc;
        }
        .tab-buttons {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .tab-button {
            padding: 5px 10px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            cursor: pointer;
            border-radius: 3px;
            color: white;
        }
        .tab-button.active {
            background: rgba(255,255,255,0.3);
        }
        .threshold-panel {
            display: none;
        }
        .threshold-panel.active {
            display: block;
        }
        .gallery-button {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 8px;
            text-align: left;
            font-size: 12px;
            border: 2px solid;
            background: rgba(0,0,0,0.5);
        }
        .gallery-button:hover {
            background: rgba(255,255,255,0.1);
        }
        .gallery-button.active {
            background: rgba(255,255,255,0.2);
        }
        .nav-buttons {
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }
        .nav-button {
            flex: 1;
            padding: 5px;
            font-size: 11px;
        }
        #gallery-info {
            text-align: center;
            margin: 10px 0;
            font-size: 11px;
            color: #ccc;
        }
        h3 {
            margin-top: 0;
            color: #fff;
        }
        input[type="range"] {
            background: rgba(255,255,255,0.1);
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
                <div style="color: #888; font-size: 12px;">
                    Using same thresholds as values tab
                </div>
            </div>
        </div>
        
        <button onclick="updateCategories()">Apply Thresholds</button>
        
        <div style="margin-top: 15px;">
            <div class="legend-item">
                <span class="color-box" style="background: #00ff00;"></span>
                <span>High AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #ff00ff;"></span>
                <span>High AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #00ffff;"></span>
                <span>Low AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #ffff00;"></span>
                <span>Low AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #666666;"></span>
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
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.8" style="width: 100px;">
            <span id="opacity-val">0.8</span>
        </div>
        <div class="control-group">
            <label>Essay BG Opacity:</label>
            <input type="range" id="essay-opacity" min="0" max="0.5" step="0.05" value="0.2" style="width: 100px;">
            <span id="essay-opacity-val">0.2</span>
        </div>
        <div class="control-group">
            <label>Transition Speed:</label>
            <input type="range" id="transition-speed" min="0.5" max="3" step="0.1" value="1.5" style="width: 100px;">
            <span id="transition-val">1.5</span>s
        </div>
    </div>
    
    <div id="gallery-controls">
        <h4 style="margin-top: 0;">Gallery Mode</h4>
        <button class="gallery-button" style="border-color: #00ff00;" onclick="startGallery('both_high')">
            High AI + High SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #ff00ff;" onclick="startGallery('ai_high')">
            High AI + Low SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #00ffff;" onclick="startGallery('sc_high')">
            Low AI + High SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #ffff00;" onclick="startGallery('both_low')">
            Low AI + Low SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        
        <div id="gallery-info"></div>
        
        <div class="nav-buttons" style="display: none;">
            <button class="nav-button" onclick="navigateGallery(-1)">← Previous</button>
            <button class="nav-button" onclick="stopGallery()">Stop</button>
            <button class="nav-button" onclick="navigateGallery(1)">Next →</button>
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
        
        // Scene setup with black background
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            2000
        );
        camera.position.set(250, 250, 250);
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
        controls.maxDistance = 1000;
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
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
        
        // Point material
        const material = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.8
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Category system
        const categories = new Array(data.length);
        const categoryIndices = {
            'both_high': [],
            'ai_high': [],
            'sc_high': [],
            'both_low': [],
            'middle': []
        };
        
        // Store sorted indices for proximity navigation
        const sortedCategoryIndices = {
            'both_high': [],
            'ai_high': [],
            'sc_high': [],
            'both_low': [],
            'middle': []
        };
        
        // Colors
        const categoryColors = {
            'both_high': [0.0, 1.0, 0.0],      // Bright Green
            'ai_high': [1.0, 0.0, 1.0],        // Magenta
            'sc_high': [0.0, 1.0, 1.0],        // Cyan
            'both_low': [1.0, 1.0, 0.0],       // Yellow
            'middle': [0.4, 0.4, 0.4]          // Dark Gray
        };
        
        // Gallery state
        let galleryMode = false;
        let currentGalleryCategory = null;
        let currentGalleryIndex = 0;
        let savedAutoRotate = true;
        let returningToOverview = false;
        
        // Animation state
        let isAnimating = false;
        let animationStartTime = 0;
        let animationDuration = 1500; // milliseconds
        let animationStart = {
            cameraPos: new THREE.Vector3(),
            targetPos: new THREE.Vector3()
        };
        let animationEnd = {
            cameraPos: new THREE.Vector3(),
            targetPos: new THREE.Vector3()
        };
        
        // Easing function
        function easeInOutCubic(t) {
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }
        
        // Calculate distance between two points in 3D space
        function calculateDistance(idx1, idx2) {
            const dx = positions[idx1 * 3] - positions[idx2 * 3];
            const dy = positions[idx1 * 3 + 1] - positions[idx2 * 3 + 1];
            const dz = positions[idx1 * 3 + 2] - positions[idx2 * 3 + 2];
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        
        // Sort indices by proximity using nearest neighbor algorithm
        function sortByProximity(indices) {
            if (indices.length <= 1) return indices;
            
            const sorted = [];
            const remaining = [...indices];
            
            // Start with the first point
            sorted.push(remaining.shift());
            
            // Repeatedly find the nearest unvisited point
            while (remaining.length > 0) {
                const currentIdx = sorted[sorted.length - 1];
                let nearestIdx = 0;
                let nearestDist = Infinity;
                
                for (let i = 0; i < remaining.length; i++) {
                    const dist = calculateDistance(currentIdx, remaining[i]);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearestIdx = i;
                    }
                }
                
                sorted.push(remaining.splice(nearestIdx, 1)[0]);
            }
            
            return sorted;
        }
        
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
            
            // Reset category indices
            Object.keys(categoryIndices).forEach(key => categoryIndices[key] = []);
            
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
                categoryIndices[category].push(i);
                counts[category]++;
                
                const color = categoryColors[category];
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            });
            
            // Sort category indices by proximity
            Object.keys(categoryIndices).forEach(category => {
                sortedCategoryIndices[category] = sortByProximity(categoryIndices[category]);
            });
            
            geometry.attributes.color.needsUpdate = true;
            
            // Update counts display with percentages of non-middle
            const total = data.length;
            const nonMiddle = total - counts.middle;
            document.getElementById('counts').innerHTML = `
                <strong>Counts:</strong><br>
                High AI + High SC: ${counts.both_high} (${(counts.both_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.both_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                High AI + Low SC: ${counts.ai_high} (${(counts.ai_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.ai_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Low AI + High SC: ${counts.sc_high} (${(counts.sc_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.sc_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Low AI + Low SC: ${counts.both_low} (${(counts.both_low/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.both_low/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Middle: ${counts.middle} (${(counts.middle/total*100).toFixed(1)}% of all)
            `;
            
            // Update gallery buttons
            document.querySelectorAll('.gallery-button').forEach(btn => {
                const category = btn.getAttribute('onclick').match(/startGallery\('(.+?)'\)/)[1];
                const count = counts[category];
                btn.querySelector('span').textContent = `${count} essays`;
            });
        }
        
        // Gallery functions
        window.startGallery = function(category) {
            if (sortedCategoryIndices[category].length === 0) {
                alert('No essays in this category with current thresholds');
                return;
            }
            
            galleryMode = true;
            currentGalleryCategory = category;
            currentGalleryIndex = 0;
            returningToOverview = false;
            
            // Save and disable auto-rotate
            savedAutoRotate = controls.autoRotate;
            controls.autoRotate = false;
            document.getElementById('auto-rotate').checked = false;
            
            // Update UI
            document.querySelectorAll('.gallery-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelector('.nav-buttons').style.display = 'flex';
            
            // Navigate to first essay (using sorted indices)
            navigateToEssay(sortedCategoryIndices[category][0]);
        };
        
        window.stopGallery = function() {
            galleryMode = false;
            currentGalleryCategory = null;
            isAnimating = false;
            returningToOverview = true;
            
            // Restore auto-rotate
            controls.autoRotate = savedAutoRotate;
            document.getElementById('auto-rotate').checked = savedAutoRotate;
            
            // Animate back to overview
            animateToPosition(
                new THREE.Vector3(250, 250, 250),
                new THREE.Vector3(cloudCenter.x, cloudCenter.y, cloudCenter.z)
            );
            
            // Update UI
            document.querySelectorAll('.gallery-button').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.nav-buttons').style.display = 'none';
            document.getElementById('gallery-info').textContent = '';
            document.getElementById('essay-display').style.display = 'none';
        };
        
        window.navigateGallery = function(direction) {
            if (!galleryMode || !currentGalleryCategory || isAnimating) return;
            
            const indices = sortedCategoryIndices[currentGalleryCategory];
            currentGalleryIndex += direction;
            
            // Wrap around
            if (currentGalleryIndex < 0) currentGalleryIndex = indices.length - 1;
            if (currentGalleryIndex >= indices.length) currentGalleryIndex = 0;
            
            navigateToEssay(indices[currentGalleryIndex]);
        };
        
        function animateToPosition(targetCameraPos, targetLookAt) {
            isAnimating = true;
            animationStartTime = Date.now();
            animationDuration = parseFloat(document.getElementById('transition-speed').value) * 1000;
            
            // Save start positions
            animationStart.cameraPos.copy(camera.position);
            animationStart.targetPos.copy(controls.target);
            
            // Set end positions
            animationEnd.cameraPos.copy(targetCameraPos);
            animationEnd.targetPos.copy(targetLookAt);
        }
        
        function navigateToEssay(index) {
            const d = data[index];
            const category = categories[index];
            
            // Update gallery info
            const indices = sortedCategoryIndices[currentGalleryCategory];
            document.getElementById('gallery-info').textContent = 
                `Essay ${currentGalleryIndex + 1} of ${indices.length}`;
            
            // Target position
            const targetPosition = new THREE.Vector3(
                positions[index * 3],
                positions[index * 3 + 1],
                positions[index * 3 + 2]
            );
            
            // Calculate camera position with nice angle
            const distance = 100;
            const angle = Date.now() * 0.0001; // Slowly changing angle
            const cameraPos = new THREE.Vector3(
                targetPosition.x + distance * Math.cos(angle),
                targetPosition.y + distance * 0.5,
                targetPosition.z + distance * Math.sin(angle)
            );
            
            // Animate to new position
            animateToPosition(cameraPos, targetPosition);
            
            // Show essay
            document.getElementById('essay-id').textContent = d.essay_id;
            document.getElementById('essay-sc').textContent = d.sc11;
            document.getElementById('essay-ai').textContent = d.ai_rating.toFixed(2);
            document.getElementById('essay-text').textContent = d.essay;
            
            // Set colors with adjustable opacity
            const color = categoryColors[category];
            const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
            const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
            document.getElementById('essay-display').style.backgroundColor = bgColor;
            document.getElementById('essay-display').style.borderColor = 
                `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
            document.getElementById('essay-display').style.display = 'block';
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
        raycaster.params.Points.threshold = 8;
        
        let hoveredIndex = -1;
        
        function onMouseMove(event) {
            if (galleryMode) return; // Disable hover in gallery mode
            
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
                    
                    // Set colors with adjustable opacity
                    const color = categoryColors[category];
                    const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
                    const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
                    document.getElementById('essay-display').style.backgroundColor = bgColor;
                    document.getElementById('essay-display').style.borderColor = 
                        `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
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
        
        document.getElementById('essay-opacity').addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value);
            document.getElementById('essay-opacity-val').textContent = opacity.toFixed(2);
            
            // Update current essay display if visible
            if (document.getElementById('essay-display').style.display === 'block' && hoveredIndex >= 0) {
                const category = categories[hoveredIndex];
                const color = categoryColors[category];
                const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${opacity})`;
                document.getElementById('essay-display').style.backgroundColor = bgColor;
            }
        });
        
        document.getElementById('transition-speed').addEventListener('input', (e) => {
            document.getElementById('transition-val').textContent = e.target.value;
        });
        
        // Keyboard navigation for gallery
        window.addEventListener('keydown', (e) => {
            if (!galleryMode) return;
            
            if (e.key === 'ArrowLeft') navigateGallery(-1);
            else if (e.key === 'ArrowRight') navigateGallery(1);
            else if (e.key === 'Escape') stopGallery();
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            // Handle smooth camera transitions
            if (isAnimating) {
                const elapsed = Date.now() - animationStartTime;
                const progress = Math.min(elapsed / animationDuration, 1);
                const eased = easeInOutCubic(progress);
                
                // Interpolate camera position
                camera.position.lerpVectors(
                    animationStart.cameraPos,
                    animationEnd.cameraPos,
                    eased
                );
                
                // Interpolate target position
                controls.target.lerpVectors(
                    animationStart.targetPos,
                    animationEnd.targetPos,
                    eased
                );
                
                if (progress >= 1) {
                    isAnimating = false;
                    
                    // If we're returning to overview, ensure controls.target is set to cloud center
                    if (returningToOverview) {
                        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
                        returningToOverview = false;
                    }
                }
            }
            
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
output_file = OUTPUT_DIR / 'minimal_umap_viz_v8.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Proximity-based gallery visualization saved to: {output_file}")
print("\nNew features in v8:")
print("- Gallery navigation organized by spatial proximity")
print("- Rotation center properly restored to cloud center when exiting gallery")
print("- Nearest neighbor algorithm minimizes travel distance between essays")
print("- All previous features maintained")