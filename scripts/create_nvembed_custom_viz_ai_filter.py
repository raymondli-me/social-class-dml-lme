#!/usr/bin/env python3
"""
Create custom 3D visualization for NV-Embed using Three.js with AI rating filtering
Handles continuous values from 1-10 with range slider
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
print("NV-EMBED CUSTOM 3D VISUALIZATION - AI RATING WITH FILTER")
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

# Prepare data for JSON
data_points = []

# Scale UMAP coordinates
scale_factor = 20
umap_scaled = umap_3d * scale_factor

# Prepare color mappings for social class labels
sc_labels_map = {
    1: 'Lower class',
    2: 'Working class', 
    3: 'Middle class',
    4: 'Upper-middle class',
    5: 'Upper class'
}

# Calculate AI rating statistics
ai_min = df['ai_rating'].min()
ai_max = df['ai_rating'].max()
ai_mean = df['ai_rating'].mean()
ai_std = df['ai_rating'].std()

print(f"\nAI Rating Statistics:")
print(f"  Min: {ai_min:.2f}")
print(f"  Max: {ai_max:.2f}")
print(f"  Mean: {ai_mean:.2f}")
print(f"  Std: {ai_std:.2f}")

for i in range(len(df)):
    point = {
        'id': str(df.iloc[i]['TID']),
        'x': float(umap_scaled[i, 0]),
        'y': float(umap_scaled[i, 1]),
        'z': float(umap_scaled[i, 2]),
        'ai_rating': float(df.iloc[i]['ai_rating']),
        'actual_sc': int(df.iloc[i]['sc11']),
        'essay_preview': df.iloc[i]['original'][:150] + '...',
        'sc_label': sc_labels_map[df.iloc[i]['sc11']]
    }
    data_points.append(point)

# Create HTML
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NV-Embed UMAP - AI Ratings with Range Filter</title>
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
        
        .range-container {{
            position: relative;
            margin: 20px 0;
        }}
        
        .range-track {{
            width: 100%;
            height: 40px;
            background: linear-gradient(to right, #440154, #21908c, #fde725);
            border-radius: 5px;
            position: relative;
        }}
        
        .range-slider {{
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }}
        
        .range-values {{
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 11px;
        }}
        
        .range-fill {{
            position: absolute;
            height: 100%;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            pointer-events: none;
        }}
        
        .range-handle {{
            position: absolute;
            width: 20px;
            height: 40px;
            background: white;
            border: 2px solid #333;
            border-radius: 5px;
            top: 0;
            transform: translateX(-50%);
            cursor: grab;
        }}
        
        .range-handle:active {{
            cursor: grabbing;
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
        
        #colorbar {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            width: 300px;
            height: 30px;
            background: linear-gradient(to right, #440154, #21908c, #fde725);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 5px;
        }}
        
        #colorbar-labels {{
            position: absolute;
            bottom: 55px;
            left: 20px;
            width: 300px;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>
    <div id="title">NV-Embed UMAP: AI Ratings (Human MacArthur)</div>
    
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
            <div class="control-label" style="margin-bottom: 10px;">Filter by AI Rating Range</div>
            <div class="range-container">
                <div class="range-track" id="rangeTrack">
                    <div class="range-fill" id="rangeFill"></div>
                    <div class="range-handle" id="minHandle"></div>
                    <div class="range-handle" id="maxHandle"></div>
                </div>
            </div>
            <div class="range-values">
                <span>Min: <span id="minValue">1.0</span></span>
                <span>Max: <span id="maxValue">10.0</span></span>
            </div>
            <div style="margin-top: 10px;">
                <button id="resetRange">Reset Range</button>
                <button id="selectMiddle">Middle 50%</button>
                <button id="selectExtremes">Extremes Only</button>
            </div>
        </div>
        
        <div class="control-group" style="margin-top: 15px;">
            <div class="control-label">Visible Points</div>
            <div><span id="visibleCount" class="highlight">{len(df)}</span> / {len(df):,}</div>
        </div>
    </div>
    
    <div id="colorbar-labels">
        <span>1</span>
        <span>5.5</span>
        <span>10</span>
    </div>
    <div id="colorbar"></div>
    
    <div id="stats">
        <div>Total Essays: <span class="highlight">{len(df):,}</span></div>
        <div>AI Rating Range: <span class="highlight">{ai_min:.1f} - {ai_max:.1f}</span></div>
        <div>Mean AI Rating: <span class="highlight">{ai_mean:.2f}</span></div>
        <div>Embeddings: <span class="highlight">NV-Embed-v2</span></div>
    </div>
    
    <script>
        // Data
        const dataPoints = {json.dumps(data_points)};
        const aiMin = {ai_min};
        const aiMax = {ai_max};
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let particles;
        let raycaster, mouse;
        
        // Range filter state
        let rangeMin = aiMin;
        let rangeMax = aiMax;
        
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
            
            // Create particles
            createParticles();
            
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
            
            // Range filter setup
            setupRangeFilter();
            
            // Button listeners
            document.getElementById('resetRange').addEventListener('click', () => {{
                rangeMin = aiMin;
                rangeMax = aiMax;
                updateRangeUI();
                updateVisibility();
            }});
            
            document.getElementById('selectMiddle').addEventListener('click', () => {{
                const range = aiMax - aiMin;
                const center = (aiMin + aiMax) / 2;
                rangeMin = center - range * 0.25;
                rangeMax = center + range * 0.25;
                updateRangeUI();
                updateVisibility();
            }});
            
            document.getElementById('selectExtremes').addEventListener('click', () => {{
                const range = aiMax - aiMin;
                rangeMin = aiMin;
                rangeMax = aiMin + range * 0.2;
                updateVisibility();
                setTimeout(() => {{
                    rangeMin = aiMax - range * 0.2;
                    rangeMax = aiMax;
                    updateRangeUI();
                    updateVisibility();
                }}, 500);
            }});
        }}
        
        function createParticles() {{
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            const sizes = [];
            const ratings = [];
            
            // Color scale - Viridis
            const colorScale = d3.scaleSequential(d3.interpolateViridis)
                .domain([1, 10]);
            
            dataPoints.forEach(point => {{
                positions.push(point.x, point.y, point.z);
                
                const color = new THREE.Color(colorScale(point.ai_rating));
                colors.push(color.r, color.g, color.b);
                
                sizes.push(3);
                ratings.push(point.ai_rating);
            }});
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
            geometry.setAttribute('rating', new THREE.Float32BufferAttribute(ratings, 1));
            
            // Custom shader to handle visibility based on rating
            const material = new THREE.ShaderMaterial({{
                uniforms: {{
                    rangeMin: {{ value: rangeMin }},
                    rangeMax: {{ value: rangeMax }}
                }},
                vertexShader: `
                    attribute float size;
                    attribute float rating;
                    varying vec3 vColor;
                    varying float vRating;
                    
                    void main() {{
                        vColor = color;
                        vRating = rating;
                        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                        
                        // Hide points outside range
                        float visible = (rating >= rangeMin && rating <= rangeMax) ? 1.0 : 0.0;
                        gl_PointSize = size * (300.0 / -mvPosition.z) * visible;
                        
                        gl_Position = projectionMatrix * mvPosition;
                    }}
                `,
                fragmentShader: `
                    uniform float rangeMin;
                    uniform float rangeMax;
                    varying vec3 vColor;
                    varying float vRating;
                    
                    void main() {{
                        // Discard if outside range
                        if (vRating < rangeMin || vRating > rangeMax) discard;
                        
                        vec2 coord = gl_PointCoord - vec2(0.5);
                        if (length(coord) > 0.5) discard;
                        
                        gl_FragColor = vec4(vColor, 0.85);
                    }}
                `,
                transparent: true,
                vertexColors: true
            }});
            
            particles = new THREE.Points(geometry, material);
            scene.add(particles);
        }}
        
        function setupRangeFilter() {{
            const track = document.getElementById('rangeTrack');
            const minHandle = document.getElementById('minHandle');
            const maxHandle = document.getElementById('maxHandle');
            const fill = document.getElementById('rangeFill');
            
            let activeHandle = null;
            
            function updateHandles() {{
                const range = aiMax - aiMin;
                const minPos = ((rangeMin - aiMin) / range) * 100;
                const maxPos = ((rangeMax - aiMin) / range) * 100;
                
                minHandle.style.left = minPos + '%';
                maxHandle.style.left = maxPos + '%';
                fill.style.left = minPos + '%';
                fill.style.width = (maxPos - minPos) + '%';
                
                document.getElementById('minValue').textContent = rangeMin.toFixed(1);
                document.getElementById('maxValue').textContent = rangeMax.toFixed(1);
            }}
            
            function handleMouseDown(e, handle) {{
                activeHandle = handle;
                e.preventDefault();
            }}
            
            function handleMouseMove(e) {{
                if (!activeHandle) return;
                
                const rect = track.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const pct = Math.max(0, Math.min(1, x / rect.width));
                const value = aiMin + pct * (aiMax - aiMin);
                
                if (activeHandle === minHandle) {{
                    rangeMin = Math.min(value, rangeMax - 0.1);
                }} else {{
                    rangeMax = Math.max(value, rangeMin + 0.1);
                }}
                
                updateHandles();
                updateVisibility();
            }}
            
            function handleMouseUp() {{
                activeHandle = null;
            }}
            
            minHandle.addEventListener('mousedown', (e) => handleMouseDown(e, minHandle));
            maxHandle.addEventListener('mousedown', (e) => handleMouseDown(e, maxHandle));
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            
            updateHandles();
        }}
        
        function updateRangeUI() {{
            const range = aiMax - aiMin;
            const minPos = ((rangeMin - aiMin) / range) * 100;
            const maxPos = ((rangeMax - aiMin) / range) * 100;
            
            document.getElementById('minHandle').style.left = minPos + '%';
            document.getElementById('maxHandle').style.left = maxPos + '%';
            document.getElementById('rangeFill').style.left = minPos + '%';
            document.getElementById('rangeFill').style.width = (maxPos - minPos) + '%';
            
            document.getElementById('minValue').textContent = rangeMin.toFixed(1);
            document.getElementById('maxValue').textContent = rangeMax.toFixed(1);
        }}
        
        function updateVisibility() {{
            // Update shader uniforms
            if (particles) {{
                particles.material.uniforms.rangeMin.value = rangeMin;
                particles.material.uniforms.rangeMax.value = rangeMax;
                particles.material.needsUpdate = true;
            }}
            
            // Count visible points
            let visibleCount = 0;
            dataPoints.forEach(point => {{
                if (point.ai_rating >= rangeMin && point.ai_rating <= rangeMax) {{
                    visibleCount++;
                }}
            }});
            
            document.getElementById('visibleCount').textContent = visibleCount.toLocaleString();
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(particles);
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = dataPoints[idx];
                
                // Only show tooltip if point is in range
                if (point.ai_rating >= rangeMin && point.ai_rating <= rangeMax) {{
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
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        
        function updateParticleSize() {{
            const size = parseFloat(this.value);
            const sizes = particles.geometry.attributes.size.array;
            for (let i = 0; i < sizes.length; i++) {{
                sizes[i] = size;
            }}
            particles.geometry.attributes.size.needsUpdate = true;
        }}
        
        function updateOpacity() {{
            // Update opacity in fragment shader would require modifying the shader
            // For simplicity, we'll skip this for now
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
        
        // Load D3 for color scale
        const script = document.createElement('script');
        script.src = 'https://d3js.org/d3.v7.min.js';
        script.onload = () => {{
            init();
            animate();
        }};
        document.head.appendChild(script);
    </script>
</body>
</html>"""

# Save visualization
output_file = VIZ_DIR / f"nvembed_custom_ai_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✓ Saved: {output_file}")
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print("\nFeatures:")
print("  - Continuous AI rating filtering (1-10)")
print("  - Interactive range slider with visual feedback")
print("  - Quick filter buttons:")
print("    - Reset Range: Show all")
print("    - Middle 50%: Show middle range")
print("    - Extremes Only: Show high ratings")
print("  - Custom WebGL shaders for efficient filtering")
print("  - Live point count updates")
print("  - Color gradient (Viridis) for AI ratings")
print("="*80)