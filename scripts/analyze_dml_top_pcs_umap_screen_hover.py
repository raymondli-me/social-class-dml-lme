#!/usr/bin/env python3
"""
DML analysis with UMAP visualization using screen-space hover detection for precision
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'

print("=== DML PC Analysis with Screen-Space Hover ===")

# Load existing results
results_file = OUTPUT_DIR / 'dml_pc_analysis_results.pkl'
with open(results_file, 'rb') as f:
    results = pickle.load(f)

comparison_df = results['comparison_df']
top_5_indices = results['top_5_indices']
pc_indices = [int(str(pc).replace('PC', '')) for pc in top_5_indices]

# Load data
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Load PCA and UMAP
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
    essay_ids = pca_data['essay_ids']

essays_df = essays_df[essays_df['essay_id'].isin(essay_ids)]
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

# Load UMAP coordinates
umap_file = OUTPUT_DIR / 'umap_3d_coordinates.npy'
if umap_file.exists():
    umap_3d = np.load(umap_file)
else:
    print("Computing UMAP...")
    import umap
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric='cosine', random_state=42)
    umap_3d = umap_model.fit_transform(X_pca)
    np.save(umap_file, umap_3d)

# Create visualization data
from scipy import stats
viz_data = []
for i in range(len(essays_df)):
    point = {
        'essay_id': essays_df.iloc[i]['essay_id'],
        'sc11': int(essays_df.iloc[i]['sc11']),
        'ai_rating': float(essays_df.iloc[i]['ai_rating']),
        'essay': essays_df.iloc[i]['essay'][:200],
        'x': float(umap_3d[i, 0]),
        'y': float(umap_3d[i, 1]),
        'z': float(umap_3d[i, 2])
    }
    
    for pc_idx in pc_indices:
        values = X_pca[:, pc_idx]
        z_score = (X_pca[i, pc_idx] - values.mean()) / values.std()
        percentile = (values < X_pca[i, pc_idx]).sum() / len(values) * 100
        
        point[f'pc{pc_idx}_zscore'] = float(z_score)
        point[f'pc{pc_idx}_percentile'] = float(percentile)
    
    viz_data.append(point)

# HTML template with screen-space hover
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>DML PC Analysis - Precise Hover</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; background: #0a0a0a; color: white; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #canvas2d {{ position: absolute; top: 0; left: 0; pointer-events: all; z-index: 10; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.9); 
                 padding: 15px; border-radius: 5px; max-width: 500px; border: 1px solid #444; z-index: 100; }}
        #controls {{ position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.9); 
                    padding: 15px; border-radius: 5px; max-width: 400px; max-height: 80vh; 
                    overflow-y: auto; border: 1px solid #444; z-index: 100; }}
        .checkbox-group {{ margin: 10px 0; }}
        .checkbox-group label {{ display: block; margin: 3px 0; cursor: pointer; }}
        .checkbox-group input {{ margin-right: 5px; }}
        #tooltip {{ position: absolute; padding: 10px; background: rgba(0,0,0,0.95); color: white; 
                   border-radius: 5px; pointer-events: none; display: none; max-width: 500px; 
                   font-size: 12px; border: 1px solid #666; z-index: 1000; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
        th, td {{ padding: 5px; text-align: left; border-bottom: 1px solid #444; }}
        th {{ background: #222; font-weight: bold; }}
        .control-group {{ margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        .pc-score {{ display: flex; justify-content: space-between; margin: 3px 0; font-size: 11px; }}
        .highlight {{ background: #333; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="container">
        <canvas id="canvas2d"></canvas>
    </div>
    <div id="info">
        <h3>DML Analysis: Top 5 PCs (Screen-Space Hover)</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>DML θ</th>
                <th>p-val</th>
                <th>Text→AI</th>
                <th>Text→SC</th>
                <th>Text+SC→AI</th>
                <th>Text+AI→SC</th>
                <th>Var</th>
            </tr>
            {comparison_table_rows}
        </table>
        <div style="font-size: 11px; color: #aaa; margin-top: 10px;">
            <div>Total points: {total_points}</div>
            <div>Visible points: <span id="visible-count">{total_points}</span></div>
        </div>
    </div>
    
    <div id="controls">
        <h4>Visualization Controls</h4>
        
        <div class="control-group">
            <strong>View Settings</strong><br>
            <label>Point Size: <input type="range" id="point-size" min="2" max="20" value="6" step="1"></label>
            <label>Cloud Scale: <input type="range" id="cloud-scale" min="0.5" max="10" value="4" step="0.1"></label>
        </div>
        
        <div class="control-group">
            <strong>Color by:</strong><br>
            <select id="color-mode">
                <option value="sc">Social Class</option>
                <option value="ai">AI Rating</option>
                <option value="pc1">PC {pc1}</option>
                <option value="pc2">PC {pc2}</option>
                <option value="pc3">PC {pc3}</option>
                <option value="pc4">PC {pc4}</option>
                <option value="pc5">PC {pc5}</option>
            </select>
        </div>
        
        <div class="control-group">
            <strong>Filter by Social Class</strong>
            <button onclick="toggleAllSC()">Toggle All</button>
            <div class="checkbox-group">
                <label><input type="checkbox" class="sc-filter" value="1" checked> Lower (1)</label>
                <label><input type="checkbox" class="sc-filter" value="2" checked> Working (2)</label>
                <label><input type="checkbox" class="sc-filter" value="3" checked> Middle (3)</label>
                <label><input type="checkbox" class="sc-filter" value="4" checked> Upper-Middle (4)</label>
                <label><input type="checkbox" class="sc-filter" value="5" checked> Upper (5)</label>
            </div>
        </div>
    </div>
    
    <div id="tooltip"></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const data = {data_json};
        const topPCs = {top_pcs_json};
        
        // Three.js setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('container').appendChild(renderer.domElement);
        
        // 2D canvas for hover detection
        const canvas2d = document.getElementById('canvas2d');
        const ctx2d = canvas2d.getContext('2d');
        canvas2d.width = window.innerWidth;
        canvas2d.height = window.innerHeight;
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        
        // Color schemes
        const scColors = {{
            1: [1, 0, 0],      // Red
            2: [1, 0.5, 0],    // Orange
            3: [1, 1, 0],      // Yellow
            4: [0, 1, 0],      // Green
            5: [0, 0, 1]       // Blue
        }};
        
        // Scale factor
        let scaleFactor = 4;
        
        // Initialize positions
        data.forEach((d, i) => {{
            positions[i * 3] = d.x * scaleFactor;
            positions[i * 3 + 1] = d.y * scaleFactor;
            positions[i * 3 + 2] = d.z * scaleFactor;
            
            const color = scColors[d.sc11];
            colors[i * 3] = color[0];
            colors[i * 3 + 1] = color[1];
            colors[i * 3 + 2] = color[2];
            
            sizes[i] = 6;
        }});
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Simple shader
        const material = new THREE.ShaderMaterial({{
            uniforms: {{
                opacity: {{ value: 0.8 }}
            }},
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                varying vec3 vColor;
                
                void main() {{
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * 300.0 / length(mvPosition.xyz);
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                uniform float opacity;
                varying vec3 vColor;
                
                void main() {{
                    vec2 coord = gl_PointCoord - vec2(0.5);
                    if (length(coord) > 0.5) discard;
                    gl_FragColor = vec4(vColor, opacity);
                }}
            `,
            transparent: true
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Calculate bounds and center camera
        const bounds = {{
            minX: Math.min(...data.map(d => d.x * scaleFactor)),
            maxX: Math.max(...data.map(d => d.x * scaleFactor)),
            minY: Math.min(...data.map(d => d.y * scaleFactor)),
            maxY: Math.max(...data.map(d => d.y * scaleFactor)),
            minZ: Math.min(...data.map(d => d.z * scaleFactor)),
            maxZ: Math.max(...data.map(d => d.z * scaleFactor))
        }};
        
        const center = {{
            x: (bounds.minX + bounds.maxX) / 2,
            y: (bounds.minY + bounds.maxY) / 2,
            z: (bounds.minZ + bounds.maxZ) / 2
        }};
        
        camera.position.set(center.x + 50, center.y + 50, center.z + 50);
        controls.target.set(center.x, center.y, center.z);
        controls.update();
        
        // Screen positions for each point
        const screenPositions = new Array(data.length);
        let hoveredIndex = -1;
        
        // Update screen positions
        function updateScreenPositions() {{
            const vector = new THREE.Vector3();
            const positions = geometry.attributes.position.array;
            const sizes = geometry.attributes.size.array;
            
            for (let i = 0; i < data.length; i++) {{
                if (sizes[i] > 0) {{
                    vector.set(
                        positions[i * 3],
                        positions[i * 3 + 1],
                        positions[i * 3 + 2]
                    );
                    
                    vector.project(camera);
                    
                    const x = (vector.x + 1) / 2 * window.innerWidth;
                    const y = -(vector.y - 1) / 2 * window.innerHeight;
                    const z = vector.z;
                    
                    // Calculate screen size based on distance
                    const distance = vector.z > 0 ? vector.z : 0.1;
                    const screenSize = sizes[i] * 300 / (distance * 100);
                    
                    screenPositions[i] = {{
                        x: x,
                        y: y,
                        z: z,
                        size: screenSize,
                        visible: z > 0 && z < 1 // In front of camera and not clipped
                    }};
                }} else {{
                    screenPositions[i] = {{ visible: false }};
                }}
            }}
        }}
        
        // Precise 2D hover detection
        function onMouseMove(event) {{
            const mouseX = event.clientX;
            const mouseY = event.clientY;
            
            let closestIndex = -1;
            let closestDistance = Infinity;
            
            for (let i = 0; i < data.length; i++) {{
                const sp = screenPositions[i];
                if (sp && sp.visible) {{
                    const dx = mouseX - sp.x;
                    const dy = mouseY - sp.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < sp.size && distance < closestDistance) {{
                        closestDistance = distance;
                        closestIndex = i;
                    }}
                }}
            }}
            
            if (closestIndex !== hoveredIndex) {{
                // Update sizes
                const sizes = geometry.attributes.size.array;
                const baseSize = parseFloat(document.getElementById('point-size').value);
                
                if (hoveredIndex >= 0) {{
                    sizes[hoveredIndex] = baseSize;
                }}
                
                hoveredIndex = closestIndex;
                
                if (hoveredIndex >= 0) {{
                    sizes[hoveredIndex] = baseSize * 1.5;
                    geometry.attributes.size.needsUpdate = true;
                    
                    // Show tooltip
                    const d = data[hoveredIndex];
                    
                    let pcInfo = '<div style="background: #1a1a1a; padding: 8px; border-radius: 4px; margin-top: 5px;">';
                    pcInfo += '<div style="color: #888; font-size: 10px; margin-bottom: 5px;">TOP 5 PRINCIPAL COMPONENTS</div>';
                    
                    for (const pc of topPCs) {{
                        const zscore = d[`pc${{pc}}_zscore`];
                        const percentile = d[`pc${{pc}}_percentile`];
                        
                        pcInfo += `
                            <div style="margin-bottom: 8px; padding: 5px; background: #222; border-radius: 3px;">
                                <div class="pc-score"><b>PC ${{pc}}</b></div>
                                <div class="pc-score">
                                    <span>Z-score: ${{zscore > 0 ? '+' : ''}}${{zscore.toFixed(2)}}</span>
                                    <span>Percentile: ${{percentile.toFixed(0)}}</span>
                                </div>
                            </div>
                        `;
                    }}
                    pcInfo += '</div>';
                    
                    const tooltip = document.getElementById('tooltip');
                    tooltip.innerHTML = `
                        <div style="font-weight: bold; color: #4ecdc4;">Essay #${{d.essay_id}}</div>
                        <div>Social Class: <span class="highlight">${{d.sc11}}</span> | AI Rating: <span class="highlight">${{d.ai_rating.toFixed(2)}}</span></div>
                        ${{pcInfo}}
                        <div style="margin-top: 10px; font-style: italic; color: #888; font-size: 11px;">
                            "${{d.essay.substring(0, 150)}}..."
                        </div>
                    `;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 10) + 'px';
                    tooltip.style.top = (event.clientY + 10) + 'px';
                }} else {{
                    geometry.attributes.size.needsUpdate = true;
                    document.getElementById('tooltip').style.display = 'none';
                }}
            }}
        }}
        
        // Update functions
        function updateVisibility() {{
            const scFilters = Array.from(document.querySelectorAll('.sc-filter:checked')).map(cb => parseInt(cb.value));
            const sizes = geometry.attributes.size.array;
            const baseSize = parseFloat(document.getElementById('point-size').value);
            
            let visibleCount = 0;
            
            for (let i = 0; i < data.length; i++) {{
                if (scFilters.includes(data[i].sc11)) {{
                    sizes[i] = (i === hoveredIndex) ? baseSize * 1.5 : baseSize;
                    visibleCount++;
                }} else {{
                    sizes[i] = 0;
                }}
            }}
            
            geometry.attributes.size.needsUpdate = true;
            document.getElementById('visible-count').textContent = visibleCount;
        }}
        
        function updateColors() {{
            const mode = document.getElementById('color-mode').value;
            const colors = geometry.attributes.color.array;
            
            data.forEach((d, i) => {{
                let color;
                if (mode === 'sc') {{
                    color = scColors[d.sc11];
                }} else if (mode === 'ai') {{
                    const normalized = (d.ai_rating - 1) / 9;
                    color = [normalized, 0, 1 - normalized];
                }} else if (mode.startsWith('pc')) {{
                    const pcNum = mode.replace('pc', '');
                    const percentile = d[`pc${{pcNum}}_percentile`] / 100;
                    color = [percentile, 0.5, 1 - percentile];
                }}
                
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
            }});
            
            geometry.attributes.color.needsUpdate = true;
        }}
        
        function toggleAllSC() {{
            const checkboxes = document.querySelectorAll('.sc-filter');
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            checkboxes.forEach(cb => cb.checked = !allChecked);
            updateVisibility();
        }}
        
        // Event listeners
        document.getElementById('color-mode').addEventListener('change', updateColors);
        document.getElementById('point-size').addEventListener('input', updateVisibility);
        document.getElementById('cloud-scale').addEventListener('input', () => {{
            const newScale = parseFloat(document.getElementById('cloud-scale').value);
            const ratio = newScale / scaleFactor;
            scaleFactor = newScale;
            
            const positions = geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i++) {{
                positions[i] *= ratio;
            }}
            geometry.attributes.position.needsUpdate = true;
            
            controls.target.multiplyScalar(ratio);
            controls.update();
        }});
        
        document.querySelectorAll('.sc-filter').forEach(cb => {{
            cb.addEventListener('change', updateVisibility);
        }});
        
        canvas2d.addEventListener('mousemove', onMouseMove);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
            updateScreenPositions(); // Update screen positions every frame
        }}
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            canvas2d.width = window.innerWidth;
            canvas2d.height = window.innerHeight;
        }});
    </script>
</body>
</html>
"""

# Format comparison table
comparison_table_rows = ""
for _, row in comparison_df.iterrows():
    if row['Model'] != 'Difference':  # Skip the difference row
        comparison_table_rows += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['DML θ']:.4f}</td>
                <td>{row['DML p-value']:.4f}</td>
                <td>{row['Text→AI R²']:.3f}</td>
                <td>{row['Text→SC R²']:.3f}</td>
                <td>{row['Text+SC→AI R²']:.3f}</td>
                <td>{row['Text+AI→SC R²']:.3f}</td>
                <td>{row['Variance']}</td>
            </tr>
        """

# Create HTML
html_content = html_template.format(
    data_json=json.dumps(viz_data),
    top_pcs_json=json.dumps(pc_indices),
    comparison_table_rows=comparison_table_rows,
    total_points=len(viz_data),
    top_pcs_list=', '.join([f'PC{pc}' for pc in pc_indices]),
    pc1=pc_indices[0],
    pc2=pc_indices[1],
    pc3=pc_indices[2],
    pc4=pc_indices[3],
    pc5=pc_indices[4]
)

# Save
output_file = OUTPUT_DIR / 'umap_dml_top5_pcs_screen_hover.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\nScreen-space hover visualization saved to:")
print(f"   {output_file}")
print("\nKey improvements:")
print("  ✓ Precise 2D hover detection - no raycasting!")
print("  ✓ Projects 3D points to screen coordinates every frame")
print("  ✓ Finds closest point in 2D space")
print("  ✓ Accounts for point size on screen")
print("  ✓ Hidden points (size=0) truly invisible")
print("  ✓ Hover works at pixel precision!")