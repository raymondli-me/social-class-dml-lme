#!/usr/bin/env python3
"""
Create custom 3D visualization for NV-Embed using Three.js (no Plotly)
Based on the working custom visualization code
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
print("NV-EMBED CUSTOM 3D VISUALIZATION (Three.js)")
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

def create_custom_viz(df, umap_3d, color_by='ai_rating'):
    """Create custom Three.js visualization"""
    
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
    <title>NV-Embed UMAP - {'Social Class' if color_by == 'sc' else 'AI Ratings'}</title>
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
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>
    <div id="title">NV-Embed UMAP: {'Colored by Social Class' if color_by == 'sc' else 'Colored by AI Rating (Human MacArthur)'}</div>
    
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
        {'''<div class="control-group" style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 15px;">
            <div class="control-label" style="margin-bottom: 10px;">Filter Social Classes</div>
            <div id="classFilters">
                <label style="display: block; margin: 5px 0; cursor: pointer;">
                    <input type="checkbox" class="class-filter" data-class="1" checked> 
                    <span style="color: #d62728;">■</span> Lower class
                </label>
                <label style="display: block; margin: 5px 0; cursor: pointer;">
                    <input type="checkbox" class="class-filter" data-class="2" checked> 
                    <span style="color: #ff7f0e;">■</span> Working class
                </label>
                <label style="display: block; margin: 5px 0; cursor: pointer;">
                    <input type="checkbox" class="class-filter" data-class="3" checked> 
                    <span style="color: #2ca02c;">■</span> Middle class
                </label>
                <label style="display: block; margin: 5px 0; cursor: pointer;">
                    <input type="checkbox" class="class-filter" data-class="4" checked> 
                    <span style="color: #1f77b4;">■</span> Upper-middle class
                </label>
                <label style="display: block; margin: 5px 0; cursor: pointer;">
                    <input type="checkbox" class="class-filter" data-class="5" checked> 
                    <span style="color: #9467bd;">■</span> Upper class
                </label>
            </div>
            <div style="margin-top: 10px;">
                <button id="selectAll" style="padding: 5px 10px; margin-right: 5px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; border-radius: 4px; cursor: pointer;">All</button>
                <button id="selectNone" style="padding: 5px 10px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; border-radius: 4px; cursor: pointer;">None</button>
            </div>
        </div>''' if color_by == 'sc' else ''}
    </div>
    
    {'<div id="legend">' + ''.join([f'<div class="legend-item"><div class="legend-color" style="background: {color}"></div>{label}</div>' for sc, (label, color) in enumerate([(v, sc_colors[k]) for k, v in sc_labels_map.items()], 1)]) + '</div>' if color_by == 'sc' else ''}
    
    <div id="stats">
        <div>Total Essays: <span class="highlight">{len(df):,}</span></div>
        <div>Embeddings: <span class="highlight">NV-Embed-v2</span></div>
        <div>PCA Dims: <span class="highlight">200</span></div>
        <div>Variance: <span class="highlight">{variance_explained:.1%}</span></div>
    </div>
    
    <script>
        // Data
        const dataPoints = {json.dumps(data_points)};
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let particles;
        let raycaster, mouse;
        let hoveredPoint = null;
        
        const container = document.getElementById('container');
        const tooltip = document.getElementById('tooltip');
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            scene.fog = new THREE.Fog(0x000000, 300, 800);
            
            // Camera - start farther out
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
            
            // Calculate data center for proper rotation
            let centerX = 0, centerY = 0, centerZ = 0;
            dataPoints.forEach(point => {{
                centerX += point.x;
                centerY += point.y;
                centerZ += point.z;
            }});
            centerX /= dataPoints.length;
            centerY /= dataPoints.length;
            centerZ /= dataPoints.length;
            
            // Controls - rotate around data center
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
            
            // Raycaster for interaction
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();
            
            // Event listeners
            window.addEventListener('resize', onWindowResize);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            
            // Control listeners
            document.getElementById('sizeSlider').addEventListener('input', updateParticleSize);
            document.getElementById('opacitySlider').addEventListener('input', updateOpacity);
            document.getElementById('autoRotate').addEventListener('change', toggleAutoRotate);
            
            // Add filter controls for social class visualization
            {'if (document.getElementById("classFilters")) {' if color_by == 'sc' else ''}
            {'    document.querySelectorAll(".class-filter").forEach(checkbox => {' if color_by == 'sc' else ''}
            {'        checkbox.addEventListener("change", updateVisibility);' if color_by == 'sc' else ''}
            {'    });' if color_by == 'sc' else ''}
            {'    document.getElementById("selectAll").addEventListener("click", () => {' if color_by == 'sc' else ''}
            {'        document.querySelectorAll(".class-filter").forEach(cb => cb.checked = true);' if color_by == 'sc' else ''}
            {'        updateVisibility();' if color_by == 'sc' else ''}
            {'    });' if color_by == 'sc' else ''}
            {'    document.getElementById("selectNone").addEventListener("click", () => {' if color_by == 'sc' else ''}
            {'        document.querySelectorAll(".class-filter").forEach(cb => cb.checked = false);' if color_by == 'sc' else ''}
            {'        updateVisibility();' if color_by == 'sc' else ''}
            {'    });' if color_by == 'sc' else ''}
            {'}' if color_by == 'sc' else ''}
        }}
        
        function createParticles() {{
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            const sizes = [];
            const visibility = [];
            
            // Color scale for AI ratings
            const colorScale = d3.scaleSequential(d3.interpolateViridis)
                .domain([1, 10]);
            
            dataPoints.forEach(point => {{
                positions.push(point.x, point.y, point.z);
                
                {'if (point.color) {' if color_by == 'sc' else ''}
                {'    const color = new THREE.Color(point.color);' if color_by == 'sc' else '    const color = new THREE.Color(colorScale(point.ai_rating));'}
                {'    colors.push(color.r, color.g, color.b);' if color_by == 'sc' else '    colors.push(color.r, color.g, color.b);'}
                {'}' if color_by == 'sc' else ''}
                
                sizes.push(3);
                visibility.push(1); // All visible by default
            }});
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
            geometry.setAttribute('visibility', new THREE.Float32BufferAttribute(visibility, 1));
            
            const material = new THREE.PointsMaterial({{
                size: 3,
                vertexColors: true,
                alphaTest: 0.5,
                transparent: true,
                opacity: 0.85,
                sizeAttenuation: true,
                vertexShader: `
                    attribute float visibility;
                    attribute float size;
                    varying float vVisibility;
                    
                    void main() {{
                        vVisibility = visibility;
                        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                        gl_PointSize = size * (300.0 / -mvPosition.z) * visibility;
                        gl_Position = projectionMatrix * mvPosition;
                    }}
                `,
                fragmentShader: `
                    varying float vVisibility;
                    varying vec3 vColor;
                    
                    void main() {{
                        if (vVisibility < 0.5) discard;
                        
                        vec2 coord = gl_PointCoord - vec2(0.5);
                        if (length(coord) > 0.5) discard;
                        
                        gl_FragColor = vec4(vColor, 1.0);
                    }}
                `
            }});
            
            // Enable custom shaders for social class filtering
            {'material.onBeforeCompile = function(shader) {' if color_by == 'sc' else ''}
            {'    shader.vertexShader = shader.vertexShader.replace(' if color_by == 'sc' else ''}
            {'        "varying vec3 vColor;",`' if color_by == 'sc' else ''}
            {'            attribute float visibility;' if color_by == 'sc' else ''}
            {'            varying float vVisibility;' if color_by == 'sc' else ''}
            {'            varying vec3 vColor;' if color_by == 'sc' else ''}
            {'        `' if color_by == 'sc' else ''}
            {'    );' if color_by == 'sc' else ''}
            {'    shader.vertexShader = shader.vertexShader.replace(' if color_by == 'sc' else ''}
            {'        "vColor = color;",`' if color_by == 'sc' else ''}
            {'            vColor = color;' if color_by == 'sc' else ''}
            {'            vVisibility = visibility;' if color_by == 'sc' else ''}
            {'            gl_PointSize = gl_PointSize * visibility;' if color_by == 'sc' else ''}
            {'        `' if color_by == 'sc' else ''}
            {'    );' if color_by == 'sc' else ''}
            {'    shader.fragmentShader = shader.fragmentShader.replace(' if color_by == 'sc' else ''}
            {'        "varying vec3 vColor;",`' if color_by == 'sc' else ''}
            {'            varying float vVisibility;' if color_by == 'sc' else ''}
            {'            varying vec3 vColor;' if color_by == 'sc' else ''}
            {'        `' if color_by == 'sc' else ''}
            {'    );' if color_by == 'sc' else ''}
            {'    shader.fragmentShader = shader.fragmentShader.replace(' if color_by == 'sc' else ''}
            {'        "void main() {",`' if color_by == 'sc' else ''}
            {'            void main() {' if color_by == 'sc' else ''}
            {'                if (vVisibility < 0.5) discard;' if color_by == 'sc' else ''}
            {'        `' if color_by == 'sc' else ''}
            {'    );' if color_by == 'sc' else ''}
            {'};' if color_by == 'sc' else ''}
            
            particles = new THREE.Points(geometry, material);
            scene.add(particles);
        }}
        
        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(particles);
            
            if (intersects.length > 0) {{
                const idx = intersects[0].index;
                const point = dataPoints[idx];
                
                hoveredPoint = idx;
                
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
                
                // Highlight point
                const sizes = particles.geometry.attributes.size.array;
                sizes[idx] = parseFloat(document.getElementById('sizeSlider').value) * 1.5;
                particles.geometry.attributes.size.needsUpdate = true;
            }} else {{
                tooltip.style.display = 'none';
                
                // Reset highlighted point
                if (hoveredPoint !== null) {{
                    const sizes = particles.geometry.attributes.size.array;
                    sizes[hoveredPoint] = parseFloat(document.getElementById('sizeSlider').value);
                    particles.geometry.attributes.size.needsUpdate = true;
                    hoveredPoint = null;
                }}
            }}
        }}
        
        function updateParticleSize() {{
            const size = parseFloat(this.value);
            particles.material.size = size;
            
            const sizes = particles.geometry.attributes.size.array;
            for (let i = 0; i < sizes.length; i++) {{
                sizes[i] = size;
            }}
            particles.geometry.attributes.size.needsUpdate = true;
        }}
        
        function updateOpacity() {{
            particles.material.opacity = parseFloat(this.value);
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
        
        // D3 for color scale
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
    
    return html_content

# Create visualizations
print("\nCreating visualizations...")

# 1. Colored by social class
html_sc = create_custom_viz(df, umap_3d, color_by='sc')
output_sc = VIZ_DIR / f"nvembed_custom_actual_sc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
with open(output_sc, 'w') as f:
    f.write(html_sc)
print(f"✓ Saved: {output_sc}")

# 2. Colored by AI rating
html_ai = create_custom_viz(df, umap_3d, color_by='ai')
output_ai = VIZ_DIR / f"nvembed_custom_ai_rating_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
with open(output_ai, 'w') as f:
    f.write(html_ai)
print(f"✓ Saved: {output_ai}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nOpen these files in your browser:")
print(f"  1. {output_sc}")
print(f"  2. {output_ai}")
print("\nFeatures:")
print("  - Interactive 3D rotation with mouse")
print("  - Hover for essay details")
print("  - Adjustable point size and opacity")
print("  - Auto-rotate option")
print("  - WebGL accelerated rendering")
print("="*80)