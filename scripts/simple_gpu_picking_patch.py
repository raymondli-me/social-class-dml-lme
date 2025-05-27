#!/usr/bin/env python3
"""
Simple patch to add GPU picking to analyze_dml_top_pcs_umap.py
"""

# Read the original file
with open('analyze_dml_top_pcs_umap.py', 'r') as f:
    content = f.read()

# 1. Add pointIds array after sizes
old_line = "        const sizes = new Float32Array(data.length);"
new_line = """        const sizes = new Float32Array(data.length);
        const pointIds = new Float32Array(data.length); // For GPU picking"""
content = content.replace(old_line, new_line)

# 2. Initialize pointIds in the data loop
old_line2 = "            originalPositions[i * 3 + 2] = d.z;"
new_line2 = """            originalPositions[i * 3 + 2] = d.z;
            pointIds[i] = i; // Assign unique ID for picking"""
content = content.replace(old_line2, new_line2)

# 3. Add pointId attribute to geometry
old_line3 = '        geometry.setAttribute(\'size\', new THREE.BufferAttribute(sizes, 1));'
new_line3 = """        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('pointId', new THREE.BufferAttribute(pointIds, 1)); // Add ID attribute"""
content = content.replace(old_line3, new_line3)

# 4. Add the picking material and GPU picking setup right after the highlight sphere creation
old_highlight = """        const highlightSphere = new THREE.Mesh(highlightGeometry, highlightMaterial);
        highlightSphere.visible = false;
        scene.add(highlightSphere);"""

new_highlight = """        const highlightSphere = new THREE.Mesh(highlightGeometry, highlightMaterial);
        highlightSphere.visible = false;
        scene.add(highlightSphere);
        
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
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                    if (dist > 0.5) discard;
                    
                    // Encode pointId into RGB
                    float id = vPointId;
                    float r = mod(id, 256.0) / 255.0;
                    float g = mod(floor(id / 256.0), 256.0) / 255.0;
                    float b = floor(id / (256.0 * 256.0)) / 255.0;
                    gl_FragColor = vec4(r, g, b, 1.0);
                }}
            `
        }});
        
        // GPU Picking setup
        const pickingRenderTarget = new THREE.WebGLRenderTarget(1, 1);
        const pixelBuffer = new Uint8Array(4);"""

content = content.replace(old_highlight, new_highlight)

# 5. Replace the onMouseMove function completely
# Find the start and end of the function
start_idx = content.find("function onMouseMove(event) {{")
end_idx = content.find("document.addEventListener('mousemove', onMouseMove);")

# Extract everything between
old_function = content[start_idx:end_idx]

new_function = """function onMouseMove(event) {{
            // GPU Picking implementation
            points.material = pickingMaterial;
            
            const rect = renderer.domElement.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            
            // Set camera to render only 1x1 pixel at mouse position
            camera.setViewOffset(
                renderer.domElement.width, renderer.domElement.height,
                mouseX, mouseY,
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
                    // Position highlight sphere
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
                            <div class="pc-score"><span><strong>PC${{pc}}</strong></span><span style="color: #666; font-size: 10px;">Importance: ${{(importance * 100).toFixed(1)}}%</span></div>
                            <div class="pc-score"><span>Z-score: <span class="highlight">${{zscore > 0 ? '+' : ''}}${{zscore.toFixed(2)}}</span></span><span>Percentile: <span class="highlight">${{percentile.toFixed(0)}}</span></span></div>
                            <div class="pc-score"><span>AI: <span class="${{shapAI > 0 ? 'shap-positive' : 'shap-negative'}}">${{shapAI > 0 ? '+' : ''}}${{shapAI.toFixed(3)}}</span></span><span>SC: <span class="${{shapSC > 0 ? 'shap-positive' : 'shap-negative'}}">${{shapSC > 0 ? '+' : ''}}${{shapSC.toFixed(3)}}</span></span></div>
                        </div>
                    `;
                }}
                pcInfo += '</div>';
                
                tooltip.innerHTML = `
                    <div class="tooltip-header">Essay #${{d.essay_id}}</div>
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
                    tooltip.style.display = 'none';
                    highlightSphere.visible = false;
                }}
            }} else {{
                tooltip.style.display = 'none';
                highlightSphere.visible = false;
            }}
        }}
        
        """

content = content[:start_idx] + new_function + content[end_idx:]

# Save the modified file
with open('analyze_dml_top_pcs_umap_gpu_final.py', 'w') as f:
    f.write(content)

print("Created: analyze_dml_top_pcs_umap_gpu_final.py")
print("This version adds GPU picking while preserving all original functionality.")