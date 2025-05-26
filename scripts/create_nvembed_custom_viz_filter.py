#!/usr/bin/env python3
"""
Create custom 3D visualization for NV-Embed using Three.js with social class filtering
Simplified version with easier filtering implementation
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Define paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = BASE_DIR / "nvembed_visualizations"
VIZ_DIR.mkdir(exist_ok=True)

print("="*80)
print("NV-EMBED CUSTOM 3D VISUALIZATION WITH FILTERING")
print("="*80)

# Load NV-Embed embeddings
print("\nLoading NV-Embed embeddings...")
embeddings_full = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"✓ Embeddings shape: {embeddings_full.shape}")

# Load data files
print("\nLoading data files...")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
ai_ratings_df = pd.read_csv(BASE_DIR / "asc_analysis_2prompts" / "run_20250524_162055" / "all_results_9513x2_20250524_174149.csv")

# Filter for human MacArthur ratings only
print("\nProcessing human MacArthur ratings...")
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
ai_ratings_avg = human_mac_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_ratings_avg.columns = ['TID', 'ai_rating']

# Merge all data
df = essays.merge(sc_labels, on='TID', how='inner')
df = df.merge(ai_ratings_avg, on='TID', how='inner')

# Align with embeddings
df = df[df['TID'].isin(essay_ids)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

print(f"\n✓ Final dataset: {len(df)} essays")

# Get aligned embeddings
X_full = embeddings_full[df['essay_idx'].values]

# Apply PCA
print("\nApplying PCA (4096 → 200 dimensions)...")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_full)
variance_explained = pca.explained_variance_ratio_.sum()
print(f"✓ Variance explained: {variance_explained:.1%}")

# Standardize
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca)

# Compute or load UMAP
print("\nComputing 3D UMAP embedding...")
umap_file = NVEMBED_DIR / "umap_3d_nvembed_custom.npy"

if umap_file.exists():
    print("Loading existing UMAP coordinates...")
    umap_3d = np.load(umap_file)
else:
    print("Computing new UMAP coordinates...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    umap_3d = reducer.fit_transform(X_pca_scaled)
    np.save(umap_file, umap_3d)

print(f"✓ UMAP shape: {umap_3d.shape}")

def create_custom_viz_with_filter(df, umap_3d, color_by='sc'):
    """Create custom Three.js visualization with filtering capability"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for JSON
    data_points = []
    
    # Scale UMAP coordinates
    scale_factor = 20
    umap_scaled = umap_3d * scale_factor
    
    # Prepare color mappings
    sc_colors = {
        1: '#d62728',  # Red (Lower class)
        2: '#ff7f0e',  # Orange
        3: '#2ca02c',  # Green (Middle class)
        4: '#1f77b4',  # Blue
        5: '#9467bd'   # Purple (Upper class)
    }
    
    sc_labels_map = {
        1: 'Lower class',
        2: 'Working class', 
        3: 'Middle class',
        4: 'Upper-middle class',
        5: 'Upper class'
    }
    
    for i in range(len(df)):
        point = {
            'id': str(df.iloc[i]['TID']),
            'x': float(umap_scaled[i, 0]),
            'y': float(umap_scaled[i, 1]),
            'z': float(umap_scaled[i, 2]),
            'ai_rating': float(df.iloc[i]['ai_rating']),
            'actual_sc': int(df.iloc[i]['sc11']),
            'essay_preview': df.iloc[i]['original'][:150] + '...',
            'sc_label': sc_labels_map[df.iloc[i]['sc11']],
            'color': sc_colors[df.iloc[i]['sc11']] if color_by == 'sc' else None
        }
        data_points.append(point)
    
    # Create HTML with Three.js
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NV-Embed UMAP - Social Class with Filtering</title>
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
            background: rgba(0, 0, 0, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
            font-size: 13px;
            max-width: 400px;
            pointer-events: none;
            display: none;
            backdrop-filter: blur(10px);
            z-index: 1000;
            line-height: 1.4;
        }}
        
        #controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            width: 250px;
            backdrop-filter: blur(10px);
        }}
        
        #title {{
            position: absolute;
            top: 20px;
            left: 20px;
            font-size: 24px;
            font-weight: 600;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        
        #legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            backdrop-filter: blur(10px);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 13px;
            opacity: 1;
            transition: opacity 0.3s;
        }}
        
        .legend-item.inactive {{
            opacity: 0.3;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        #stats {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            font-size: 12px;
            backdrop-filter: blur(10px);
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-label {{
            font-size: 12px;
            margin-bottom: 5px;
            opacity: 0.8;
        }}
        
        input[type="range"] {{
            width: 100%;
            margin: 5px 0;
        }}
        
        .highlight {{
            color: #4A90E2;
            font-weight: bold;
        }}
        
        .filter-checkbox {{
            margin-right: 5px;
        }}
        
        .filter-label {{
            display: block;
            margin: 5px 0;
            cursor: pointer;
            user-select: none;
        }}
        
        .filter-label:hover {{
            opacity: 0.8;
        }}
        
        button {{
            padding: 5px 10px;
            margin-right: 5px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        
        button:hover {{
            background: rgba(255,255,255,0.2);
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>
    <div id="title">NV-Embed UMAP: Social Class Distribution</div>
    
    <div id="tooltip"></div>
    
    <div id="controls">
        <div class="control-group">
            <div class="control-label">Point Size</div>
            <input type="range" id="sizeSlider" min="0.5" max="8" step="0.1" value="3">
        </div>
        <div class="control-group">
            <div class="control-label">Opacity</div>
            <input type="range" id="opacitySlider" min="0.1" max="1" step="0.05" value="0.85">
        </div>
        <div class="control-group">
            <label style="font-size: 12px;">
                <input type="checkbox" id="autoRotate"> Auto Rotate
            </label>
        </div>
        <div class="control-group" style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 15px;">
            <div class="control-label" style="margin-bottom: 10px;">Filter Social Classes</div>
            <div id="classFilters">
                <label class="filter-label">
                    <input type="checkbox" class="filter-checkbox" data-class="1" checked> 
                    <span style="color: #d62728;">■</span> Lower class (<span id="count-1">0</span>)
                </label>
                <label class="filter-label">
                    <input type="checkbox" class="filter-checkbox" data-class="2" checked> 
                    <span style="color: #ff7f0e;">■</span> Working class (<span id="count-2">0</span>)
                </label>
                <label class="filter-label">
                    <input type="checkbox" class="filter-checkbox" data-class="3" checked> 
                    <span style="color: #2ca02c;">■</span> Middle class (<span id="count-3">0</span>)
                </label>
                <label class="filter-label">
                    <input type="checkbox" class="filter-checkbox" data-class="4" checked> 
                    <span style="color: #1f77b4;">■</span> Upper-middle (<span id="count-4">0</span>)
                </label>
                <label class="filter-label">
                    <input type="checkbox" class="filter-checkbox" data-class="5" checked> 
                    <span style="color: #9467bd;">■</span> Upper class (<span id="count-5">0</span>)
                </label>
            </div>
            <div style="margin-top: 10px;">
                <button id="selectAll">All</button>
                <button id="selectNone">None</button>
                <button id="onlyExtremes">1 & 5 Only</button>
            </div>
        </div>
    </div>
    
    <div id="legend">
        <div class="legend-item" data-class="1">
            <div class="legend-color" style="background: #d62728"></div>Lower class
        </div>
        <div class="legend-item" data-class="2">
            <div class="legend-color" style="background: #ff7f0e"></div>Working class
        </div>
        <div class="legend-item" data-class="3">
            <div class="legend-color" style="background: #2ca02c"></div>Middle class
        </div>
        <div class="legend-item" data-class="4">
            <div class="legend-color" style="background: #1f77b4"></div>Upper-middle class
        </div>
        <div class="legend-item" data-class="5">
            <div class="legend-color" style="background: #9467bd"></div>Upper class
        </div>
    </div>
    
    <div id="stats">
        <div>Total Essays: <span class="highlight">{len(df):,}</span></div>
        <div>Visible: <span class="highlight" id="visibleCount">{len(df):,}</span></div>
        <div>Embeddings: <span class="highlight">NV-Embed-v2</span></div>
        <div>PCA Variance: <span class="highlight">{variance_explained:.1%}</span></div>
    </div>
    
    <script>
        // Data
        const dataPoints = {json.dumps(data_points)};
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let particleGroups = {{}};
        let raycaster, mouse;
        let hoveredPoint = null;
        let activeClasses = new Set([1, 2, 3, 4, 5]);
        
        const container = document.getElementById('container');
        const tooltip = document.getElementById('tooltip');
        
        // Count points per class
        const classCounts = {{}};
        dataPoints.forEach(point => {{
            classCounts[point.actual_sc] = (classCounts[point.actual_sc] || 0) + 1;
        }});
        
        // Update counts in UI
        Object.entries(classCounts).forEach(([sc, count]) => {{
            const elem = document.getElementById(`count-${{sc}}`);
            if (elem) elem.textContent = count;
        }});
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            scene.fog = new THREE.Fog(0x000000, 300, 800);
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                60, 
                window.innerWidth / window.innerHeight, 
                0.1, 
                5000
            );
            camera.position.set(150, 150, 150);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            
            // Calculate data center
            let centerX = 0, centerY = 0, centerZ = 0;
            dataPoints.forEach(point => {{
                centerX += point.x;
                centerY += point.y;
                centerZ += point.z;
            }});
            centerX /= dataPoints.length;
            centerY /= dataPoints.length;
            centerZ /= dataPoints.length;
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.minDistance = 20;
            controls.maxDistance = 500;
            controls.target.set(centerX, centerY, centerZ);
            controls.update();
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(50, 50, 50);
            scene.add(directionalLight);
            
            // Create particle groups for each social class
            createParticleGroups();
            
            // Raycaster
            raycaster = new THREE.Raycaster();
            raycaster.params.Points.threshold = 2;
            mouse = new THREE.Vector2();
            
            // Event listeners
            window.addEventListener('resize', onWindowResize);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            
            // Control listeners
            document.getElementById('sizeSlider').addEventListener('input', updateParticleSize);
            document.getElementById('opacitySlider').addEventListener('input', updateOpacity);
            document.getElementById('autoRotate').addEventListener('change', toggleAutoRotate);
            
            // Filter controls
            document.querySelectorAll('.filter-checkbox').forEach(checkbox => {{
                checkbox.addEventListener('change', updateVisibility);
            }});
            
            document.getElementById('selectAll').addEventListener('click', () => {{
                document.querySelectorAll('.filter-checkbox').forEach(cb => cb.checked = true);
                updateVisibility();
            }});
            
            document.getElementById('selectNone').addEventListener('click', () => {{
                document.querySelectorAll('.filter-checkbox').forEach(cb => cb.checked = false);
                updateVisibility();
            }});
            
            document.getElementById('onlyExtremes').addEventListener('click', () => {{
                document.querySelectorAll('.filter-checkbox').forEach(cb => {{
                    cb.checked = (cb.dataset.class === '1' || cb.dataset.class === '5');
                }});
                updateVisibility();
            }});
        }}
        
        function createParticleGroups() {{
            // Group data points by social class
            const pointsByClass = {{}};
            dataPoints.forEach((point, idx) => {{
                if (!pointsByClass[point.actual_sc]) {{
                    pointsByClass[point.actual_sc] = {{
                        positions: [],
                        indices: [],
                        color: point.color
                    }};
                }}
                pointsByClass[point.actual_sc].positions.push(point.x, point.y, point.z);
                pointsByClass[point.actual_sc].indices.push(idx);
            }});
            
            // Create particle system for each class
            Object.entries(pointsByClass).forEach(([sc, data]) => {{
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
                
                const material = new THREE.PointsMaterial({{
                    color: data.color,
                    size: 3,
                    transparent: true,
                    opacity: 0.85,
                    sizeAttenuation: true
                }});
                
                const particles = new THREE.Points(geometry, material);
                particles.userData = {{ socialClass: parseInt(sc), indices: data.indices }};
                
                particleGroups[sc] = particles;
                scene.add(particles);
            }});
        }}
        
        function updateVisibility() {{
            activeClasses.clear();
            let visibleCount = 0;
            
            document.querySelectorAll('.filter-checkbox').forEach(checkbox => {{
                const sc = parseInt(checkbox.dataset.class);
                const isVisible = checkbox.checked;
                
                if (isVisible) {{
                    activeClasses.add(sc);
                    visibleCount += classCounts[sc] || 0;
                }}
                
                // Update particle visibility
                if (particleGroups[sc]) {{
                    particleGroups[sc].visible = isVisible;
                }}
                
                // Update legend
                const legendItem = document.querySelector(`.legend-item[data-class="${{sc}}"]`);
                if (legendItem) {{
                    legendItem.classList.toggle('inactive', !isVisible);
                }}
            }});
            
            // Update visible count
            document.getElementById('visibleCount').textContent = visibleCount.toLocaleString();
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            
            // Check intersections with all visible groups
            let closestIntersection = null;
            let closestDistance = Infinity;
            
            Object.values(particleGroups).forEach(group => {{
                if (!group.visible) return;
                
                const intersects = raycaster.intersectObject(group);
                if (intersects.length > 0 && intersects[0].distance < closestDistance) {{
                    closestDistance = intersects[0].distance;
                    closestIntersection = {{
                        intersection: intersects[0],
                        group: group
                    }};
                }}
            }});
            
            if (closestIntersection) {{
                const idx = closestIntersection.group.userData.indices[closestIntersection.intersection.index];
                const point = dataPoints[idx];
                
                tooltip.innerHTML = `
                    <div style="margin-bottom: 8px;"><strong>Essay ID:</strong> ${{point.id}}</div>
                    <div style="margin-bottom: 8px;"><strong>Social Class:</strong> ${{point.sc_label}}</div>
                    <div style="margin-bottom: 8px;"><strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}}</div>
                    <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px; margin-top: 8px;">
                        <strong>Preview:</strong><br>
                        <div style="opacity: 0.8; font-size: 11px; margin-top: 4px;">
                            ${{point.essay_preview}}
                        </div>
                    </div>
                `;
                
                tooltip.style.display = 'block';
                tooltip.style.left = event.clientX + 10 + 'px';
                tooltip.style.top = event.clientY + 10 + 'px';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        function updateParticleSize() {{
            const size = parseFloat(this.value);
            Object.values(particleGroups).forEach(group => {{
                group.material.size = size;
            }});
        }}
        
        function updateOpacity() {{
            const opacity = parseFloat(this.value);
            Object.values(particleGroups).forEach(group => {{
                group.material.opacity = opacity;
            }});
        }}
        
        function toggleAutoRotate() {{
            controls.autoRotate = this.checked;
            controls.autoRotateSpeed = 0.5;
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
        
        init();
        animate();
    </script>
</body>
</html>"""
    
    return html_content

# Create visualization with filtering
print("\nCreating visualization with filtering...")

html_sc = create_custom_viz_with_filter(df, umap_3d, color_by='sc')
output_sc = VIZ_DIR / f"nvembed_custom_sc_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
with open(output_sc, 'w') as f:
    f.write(html_sc)
print(f"✓ Saved: {output_sc}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nOpen this file in your browser:")
print(f"  {output_sc}")
print("\nFeatures:")
print("  - Interactive social class filtering")
print("  - Click checkboxes to show/hide classes")
print("  - Quick buttons: All, None, '1 & 5 Only'")
print("  - Live point count updates")
print("  - Smooth transitions")
print("  - Hover for essay details")
print("="*80)