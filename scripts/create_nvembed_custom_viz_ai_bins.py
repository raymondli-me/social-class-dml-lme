#!/usr/bin/env python3
"""
Create custom 3D visualization for NV-Embed using Three.js with AI rating filtering
Uses percentile bins with checkboxes for filtering
"""

import numpy as np
import pandas as pd
import json
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
print("NV-EMBED CUSTOM 3D VISUALIZATION - AI RATING BINS")
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

# Calculate AI rating distribution
print("\nCalculating AI rating bins...")
print(f"AI Rating range: {df['ai_rating'].min():.1f} - {df['ai_rating'].max():.1f}")
print(f"AI Rating distribution:")
print(df['ai_rating'].value_counts().sort_index())

# Use quantile-based binning to handle duplicates
try:
    # Try 10 bins first
    df['ai_bin'], bin_edges = pd.qcut(df['ai_rating'], q=10, labels=False, retbins=True, duplicates='drop')
    n_bins = len(bin_edges) - 1
    print(f"\nCreated {n_bins} bins after dropping duplicates")
except ValueError:
    # If still fails, use unique values
    print("\nUsing unique rating values for binning...")
    unique_ratings = sorted(df['ai_rating'].unique())
    n_bins = min(10, len(unique_ratings))
    df['ai_bin'] = pd.qcut(df['ai_rating'], q=n_bins, labels=False, duplicates='drop')
    _, bin_edges = pd.qcut(df['ai_rating'], q=n_bins, labels=False, retbins=True, duplicates='drop')

# Create bin labels based on actual bins
bins = []
bin_labels = []
bin_counts_list = []

for i in range(n_bins):
    bin_mask = df['ai_bin'] == i
    bin_count = bin_mask.sum()
    bin_min = df.loc[bin_mask, 'ai_rating'].min()
    bin_max = df.loc[bin_mask, 'ai_rating'].max()
    
    bins.append((bin_min, bin_max))
    
    # Calculate percentile position
    pct_start = (i / n_bins) * 100
    pct_end = ((i + 1) / n_bins) * 100
    
    if bin_min == bin_max:
        bin_labels.append(f"Rating {bin_min:.1f} ({bin_count} essays)")
    else:
        bin_labels.append(f"{pct_start:.0f}-{pct_end:.0f}% ({bin_min:.1f}-{bin_max:.1f})")
    
    bin_counts_list.append(bin_count)

# Prepare data for JSON
data_points = []

# Scale UMAP coordinates
scale_factor = 20
umap_scaled = umap_3d * scale_factor

# Prepare color mappings
sc_labels_map = {
    1: 'Lower class',
    2: 'Working class', 
    3: 'Middle class',
    4: 'Upper-middle class',
    5: 'Upper class'
}

# Create color scale for bins (using Viridis colors)
# Generate colors based on actual number of bins
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.get_cmap('viridis')
bin_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
              for r,g,b,_ in [cmap(i/max(n_bins-1, 1)) for i in range(n_bins)]]

for i in range(len(df)):
    point = {
        'id': str(df.iloc[i]['TID']),
        'x': float(umap_scaled[i, 0]),
        'y': float(umap_scaled[i, 1]),
        'z': float(umap_scaled[i, 2]),
        'ai_rating': float(df.iloc[i]['ai_rating']),
        'ai_bin': int(df.iloc[i]['ai_bin']),
        'actual_sc': int(df.iloc[i]['sc11']),
        'essay_preview': df.iloc[i]['original'][:150] + '...',
        'sc_label': sc_labels_map[df.iloc[i]['sc11']],
        'color': bin_colors[int(df.iloc[i]['ai_bin'])]
    }
    data_points.append(point)

# Count points per bin
bin_counts = df['ai_bin'].value_counts().sort_index().to_dict()

# Create HTML
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NV-Embed UMAP - AI Rating Percentile Bins</title>
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
            width: 300px;
            backdrop-filter: blur(10px);
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        #title {{
            position: absolute;
            top: 20px;
            left: 20px;
            font-size: 24px;
            font-weight: 600;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
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
        
        .filter-label {{
            display: block;
            margin: 5px 0;
            cursor: pointer;
            user-select: none;
            font-size: 12px;
        }}
        
        .filter-label.inactive {{
            opacity: 0.5;
        }}
        
        button {{
            padding: 5px 10px;
            margin-right: 5px;
            margin-bottom: 5px;
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
        
        .highlight {{
            color: #4A90E2;
            font-weight: bold;
        }}
        
        /* Scrollbar styling */
        #controls::-webkit-scrollbar {{
            width: 8px;
        }}
        
        #controls::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }}
        
        #controls::-webkit-scrollbar-thumb {{
            background: rgba(255,255,255,0.3);
            border-radius: 4px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>
    <div id="title">NV-Embed UMAP: AI Rating Percentiles</div>
    
    <div id="tooltip"></div>
    
    <div id="controls">
        <div class="control-group">
            <div class="control-label">Point Size</div>
            <input type="range" id="sizeSlider" min="1" max="8" step="0.5" value="3">
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
            <div class="control-label" style="margin-bottom: 10px;">Filter by AI Rating Percentile</div>
            <div id="binFilters">
                {''.join([f'''
                <label class="filter-label" data-bin="{i}">
                    <input type="checkbox" class="filter-checkbox" data-bin="{i}" checked> 
                    <span style="color: {bin_colors[i]};">■</span> {bin_labels[i]}
                </label>''' for i in range(n_bins)])}
            </div>
            <div style="margin-top: 10px;">
                <button id="selectAll">All</button>
                <button id="selectNone">None</button>
                <button id="selectBottom">Bottom 30%</button>
                <button id="selectTop">Top 30%</button>
                <button id="selectExtremes">Extremes</button>
            </div>
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
        const binCounts = {json.dumps(bin_counts)};
        const binColors = {json.dumps(bin_colors)};
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let particlesByBin = {{}};
        let raycaster, mouse;
        let activeBins = new Set([...Array({n_bins}).keys()]);
        
        const container = document.getElementById('container');
        const tooltip = document.getElementById('tooltip');
        
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
            
            // Create particles for each bin
            createParticlesByBin();
            
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
            
            document.getElementById('selectBottom').addEventListener('click', () => {{
                const nBins = {n_bins};
                const bottomThreshold = Math.floor(nBins * 0.3);
                document.querySelectorAll('.filter-checkbox').forEach(cb => {{
                    const bin = parseInt(cb.dataset.bin);
                    cb.checked = (bin < bottomThreshold);
                }});
                updateVisibility();
            }});
            
            document.getElementById('selectTop').addEventListener('click', () => {{
                const nBins = {n_bins};
                const topThreshold = Math.floor(nBins * 0.7);
                document.querySelectorAll('.filter-checkbox').forEach(cb => {{
                    const bin = parseInt(cb.dataset.bin);
                    cb.checked = (bin >= topThreshold);
                }});
                updateVisibility();
            }});
            
            document.getElementById('selectExtremes').addEventListener('click', () => {{
                const nBins = {n_bins};
                document.querySelectorAll('.filter-checkbox').forEach(cb => {{
                    const bin = parseInt(cb.dataset.bin);
                    cb.checked = (bin === 0 || bin === nBins - 1);
                }});
                updateVisibility();
            }});
        }}
        
        function createParticlesByBin() {{
            // Group points by bin
            const pointsByBin = {{}};
            
            dataPoints.forEach((point, idx) => {{
                const bin = point.ai_bin;
                if (!pointsByBin[bin]) {{
                    pointsByBin[bin] = {{
                        positions: [],
                        color: point.color,
                        indices: []
                    }};
                }}
                pointsByBin[bin].positions.push(point.x, point.y, point.z);
                pointsByBin[bin].indices.push(idx);
            }});
            
            // Create separate particle system for each bin
            Object.entries(pointsByBin).forEach(([bin, data]) => {{
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(data.positions);
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                
                const material = new THREE.PointsMaterial({{
                    color: data.color,
                    size: 3,
                    transparent: true,
                    opacity: 0.85,
                    sizeAttenuation: true
                }});
                
                const particles = new THREE.Points(geometry, material);
                particles.userData = {{ 
                    bin: parseInt(bin), 
                    indices: data.indices,
                    visible: true
                }};
                
                particlesByBin[bin] = particles;
                scene.add(particles);
            }});
        }}
        
        function updateVisibility() {{
            activeBins.clear();
            let visibleCount = 0;
            
            document.querySelectorAll('.filter-checkbox').forEach(checkbox => {{
                const bin = parseInt(checkbox.dataset.bin);
                const isChecked = checkbox.checked;
                const label = checkbox.parentElement;
                
                if (isChecked) {{
                    activeBins.add(bin);
                    visibleCount += binCounts[bin] || 0;
                    label.classList.remove('inactive');
                }} else {{
                    label.classList.add('inactive');
                }}
                
                // Update particle visibility
                if (particlesByBin[bin]) {{
                    particlesByBin[bin].visible = isChecked;
                    particlesByBin[bin].userData.visible = isChecked;
                }}
            }});
            
            // Update visible count
            document.getElementById('visibleCount').textContent = visibleCount.toLocaleString();
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            
            let found = false;
            
            // Check each visible particle group
            Object.values(particlesByBin).forEach(particles => {{
                if (!particles.userData.visible || found) return;
                
                const intersects = raycaster.intersectObject(particles);
                
                if (intersects.length > 0) {{
                    const localIdx = intersects[0].index;
                    const globalIdx = particles.userData.indices[localIdx];
                    const point = dataPoints[globalIdx];
                    
                    tooltip.innerHTML = `
                        <div style="margin-bottom: 8px;"><strong>Essay ID:</strong> ${{point.id}}</div>
                        <div style="margin-bottom: 8px;"><strong>AI Rating:</strong> ${{point.ai_rating.toFixed(2)}}</div>
                        <div style="margin-bottom: 8px;"><strong>Actual Social Class:</strong> ${{point.sc_label}}</div>
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
                    found = true;
                }}
            }});
            
            if (!found) {{
                tooltip.style.display = 'none';
            }}
        }}
        
        function updateParticleSize() {{
            const size = parseFloat(this.value);
            Object.values(particlesByBin).forEach(particles => {{
                particles.material.size = size;
            }});
        }}
        
        function updateOpacity() {{
            const opacity = parseFloat(this.value);
            Object.values(particlesByBin).forEach(particles => {{
                particles.material.opacity = opacity;
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

# Save visualization
output_file = VIZ_DIR / f"nvembed_custom_ai_bins_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✓ Saved: {output_file}")
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nFeatures:")
print("  - 10 percentile bins for AI ratings")
print("  - Checkbox filtering like social class version")
print("  - Color gradient from purple (low) to yellow (high)")
print("  - Quick filter buttons:")
print("    - All: Show all bins")
print("    - None: Hide all bins")
print("    - Bottom 30%: Show lowest 3 bins")
print("    - Top 30%: Show highest 3 bins")
print("    - Extremes: Show only bottom and top 10%")
print("  - Shows count per bin")
print("  - Same architecture as working social class version")
print("="*80)