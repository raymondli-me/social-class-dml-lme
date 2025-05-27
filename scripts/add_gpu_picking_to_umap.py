#!/usr/bin/env python3
"""
Add GPU picking to the existing analyze_dml_top_pcs_umap.py
"""

import re

# Read the original file
with open('analyze_dml_top_pcs_umap.py', 'r') as f:
    content = f.read()

# Find where to add the pointIds attribute
geometry_section = """        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);"""

new_geometry_section = """        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colors = new Float32Array(data.length * 3);
        const sizes = new Float32Array(data.length);
        const pointIds = new Float32Array(data.length); // For GPU picking"""

content = content.replace(geometry_section, new_geometry_section)

# Add pointIds initialization in the data loop
data_loop_pattern = r"(data\.forEach\(\(d, i\) => \{\{[\s\S]*?)(originalPositions\[i \* 3 \+ 2\] = d\.z;)"
data_loop_replacement = r"\1\2\n            pointIds[i] = i; // Assign unique ID for picking"
content = re.sub(data_loop_pattern, data_loop_replacement, content)

# Add pointId attribute to geometry after size attribute
geometry_attrs = """        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));"""

new_geometry_attrs = """        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('pointId', new THREE.BufferAttribute(pointIds, 1)); // Add ID attribute"""

content = content.replace(geometry_attrs, new_geometry_attrs)

# Add picking material after the visual material
material_insert_end = content.find("const points = new THREE.Points(geometry, material);")
material_insert = """
        // Picking shader material for GPU picking
        const pickingMaterial = new THREE.ShaderMaterial({{
            vertexShader: `
                attribute float size;
                attribute float pointId;
                varying float vPointId;
                varying float vSize;
                
                void main() {{
                    vPointId = pointId;
                    vSize = size;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * 50.0 / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }}
            `,
            fragmentShader: `
                varying float vPointId;
                varying float vSize;
                
                void main() {{
                    if (vSize == 0.0) discard; // Don't render hidden points
                    
                    // Encode pointId into RGB
                    float id = vPointId;
                    float r = mod(id, 256.0) / 255.0;
                    float g = mod(floor(id / 256.0), 256.0) / 255.0;
                    float b = floor(id / (256.0 * 256.0)) / 255.0;
                    gl_FragColor = vec4(r, g, b, 1.0);
                }}
            `
        }});

        """
content = content[:material_insert_end] + material_insert + content[material_insert_end:]

# Add picking setup before the raycaster
raycaster_start = content.find("// Tooltip")
picking_setup = """        // GPU Picking setup
        const pickingRenderTarget = new THREE.WebGLRenderTarget(1, 1);
        const pixelBuffer = new Uint8Array(4);
        
        """
content = content[:raycaster_start] + picking_setup + content[raycaster_start:]

# Replace the entire onMouseMove function with GPU picking version
old_onmousemove_start = content.find("function onMouseMove(event) {{")
old_onmousemove_end = content.find("document.addEventListener('mousemove', onMouseMove);")

new_onmousemove = """function onMouseMove(event) {{
            // GPU Picking implementation
            points.material = pickingMaterial;
            
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = event.clientX - rect.left;
            mouse.y = event.clientY - rect.top;
            
            // Set camera to render only 1x1 pixel at mouse position
            camera.setViewOffset(
                renderer.domElement.width, renderer.domElement.height,
                mouse.x, mouse.y,
                1, 1
            );
            
            // Render to picking target
            renderer.setRenderTarget(pickingRenderTarget);
            renderer.render(scene, camera);
            
            // Reset for normal rendering
            camera.clearViewOffset();
            renderer.setRenderTarget(null);
            
            // Read the pixel
            renderer.readRenderTargetPixels(pickingRenderTarget, 0, 0, 1, 1, pixelBuffer);
            
            const id = pixelBuffer[0] + (pixelBuffer[1] * 256) + (pixelBuffer[2] * 256 * 256);
            
            // Restore visual material
            points.material = material;
            
            if (id < data.length && data[id]) {{
                const d = data[id];
                const currentSizes = geometry.attributes.size.array;
                
                if (currentSizes[id] > 0) {{ // Check if point is visible
                    // Position highlight sphere at the point
                    const positions = geometry.attributes.position.array;
                    highlightSphere.position.set(
                        positions[id * 3],
                        positions[id * 3 + 1],
                        positions[id * 3 + 2]
                    );
                    highlightSphere.visible = true;
                    highlightMaterial.opacity = 0.5;
                    
                    let pcInfo = '<div style="background: #1a1a1a; padding: 8px; border-radius: 4px; margin-top: 5px;">';
                    pcInfo += '<div style="color: #888; font-size: 10px; margin-bottom: 5px;">TOP 5 PRINCIPAL COMPONENTS</div>';
                    
                    for (const pc of topPCs) {{
                        const zscore = d[`pc${{pc}}_zscore`];
                        const percentile = d[`pc${{pc}}_percentile`];
                        const shapAI = d[`pc${{pc}}_shap_ai`];
                        const shapSC = d[`pc${{pc}}_shap_sc`];
                        const importance = pcImportance[pc];
                        
                        pcInfo += `
                            <div style="margin-bottom: 8px; padding: 5px; background: #222; border-radius: 3px;">
                                <div class="pc-score"><span><strong>PC${{pc}}</strong></span><span style="color: #666; font-size: 10px;">Importance: ${{importance.toFixed(3)}}</span></div>
                                <div class="pc-score"><span>Z-score: <span class="highlight">${{zscore.toFixed(2)}}</span></span><span>Percentile: <span class="highlight">${{percentile.toFixed(1)}}%</span></span></div>
                                <div class="pc-score"><span>SHAP AI: <span class="${{shapAI >= 0 ? 'shap-positive' : 'shap-negative'}}">${{shapAI >= 0 ? '+' : ''}}${{shapAI.toFixed(3)}}</span></span><span>SHAP SC: <span class="${{shapSC >= 0 ? 'shap-positive' : 'shap-negative'}}">${{shapSC >= 0 ? '+' : ''}}${{shapSC.toFixed(3)}}</span></span></div>
                            </div>
                        `;
                    }}
                    pcInfo += '</div>';
                    
                    tooltip.innerHTML = `
                        <div class="tooltip-header">Essay #${{d.essay_id}}</div>
                        <div style="margin: 5px 0;"><span class="highlight">Social Class: ${{d.sc11}}</span> | <span class="highlight">AI Rating: ${{d.ai_rating.toFixed(2)}}</span></div>
                        <div class="tooltip-section"><div style="color: #888; font-size: 10px; margin-bottom: 3px;">Essay Preview:</div><div style="font-style: italic; color: #ccc;">"${{d.essay.substring(0, 150)}}..."</div></div>
                        <div class="tooltip-section">${{pcInfo}}</div>
                    `;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 10) + 'px';
                    tooltip.style.top = (event.clientY + 10) + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                    highlightSphere.visible = false;
                }}
            }} else {{
                tooltip.style.display = 'none';
                highlightSphere.visible = false;
            }}
        }}
        
        """

content = content[:old_onmousemove_start] + new_onmousemove + content[old_onmousemove_end:]

# Save the modified file
with open('analyze_dml_top_pcs_umap_gpu_picking.py', 'w') as f:
    f.write(content)

print("Created: analyze_dml_top_pcs_umap_gpu_picking.py")
print("This version uses GPU picking for precise hover detection while keeping all other functionality intact.")