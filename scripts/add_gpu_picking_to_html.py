#!/usr/bin/env python3
"""
Add GPU picking to the generated HTML file
"""

import re

# Read the HTML file
html_file = '/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/umap_dml_top5_pcs.html'
with open(html_file, 'r') as f:
    content = f.read()

# 1. Add pointIds array
content = content.replace(
    "const sizes = new Float32Array(data.length);",
    """const sizes = new Float32Array(data.length);
        const pointIds = new Float32Array(data.length); // For GPU picking"""
)

# 2. Initialize pointIds
content = content.replace(
    "originalPositions[i * 3 + 2] = d.z;",
    """originalPositions[i * 3 + 2] = d.z;
            pointIds[i] = i; // Assign unique ID for picking"""
)

# 3. Add pointId attribute
content = content.replace(
    "geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));",
    """geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('pointId', new THREE.BufferAttribute(pointIds, 1));"""
)

# 4. Add picking material after highlight sphere
highlight_pattern = r"(const highlightSphere = new THREE\.Mesh\(highlightGeometry, highlightMaterial\);[\s\n]*highlightSphere\.visible = false;[\s\n]*scene\.add\(highlightSphere\);)"
picking_addition = r"""\1
        
        // Picking shader material for GPU picking
        const pickingMaterial = new THREE.ShaderMaterial({
            vertexShader: `
                attribute float size;
                attribute float pointId;
                varying float vPointId;
                varying float vSize;
                
                void main() {
                    vPointId = pointId;
                    vSize = size;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * 50.0 / -mvPosition.z;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying float vPointId;
                varying float vSize;
                
                void main() {
                    if (vSize == 0.0) discard;
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                    if (dist > 0.5) discard;
                    
                    float id = vPointId;
                    float r = mod(id, 256.0) / 255.0;
                    float g = mod(floor(id / 256.0), 256.0) / 255.0;
                    float b = floor(id / (256.0 * 256.0)) / 255.0;
                    gl_FragColor = vec4(r, g, b, 1.0);
                }
            `
        });
        
        // GPU Picking setup
        const pickingRenderTarget = new THREE.WebGLRenderTarget(1, 1);
        const pixelBuffer = new Uint8Array(4);"""

content = re.sub(highlight_pattern, picking_addition, content, flags=re.DOTALL)

# 5. Replace the entire onMouseMove function
# Find the function
func_start = content.find("function onMouseMove(event) {")
func_end = content.find("}", content.find("highlightSphere.visible = false;", func_start)) + 1

new_function = """function onMouseMove(event) {
            // GPU Picking
            points.material = pickingMaterial;
            
            const rect = renderer.domElement.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            
            camera.setViewOffset(
                renderer.domElement.width, renderer.domElement.height,
                mouseX, mouseY, 1, 1
            );
            
            renderer.setRenderTarget(pickingRenderTarget);
            renderer.render(scene, camera);
            camera.clearViewOffset();
            renderer.setRenderTarget(null);
            
            renderer.readRenderTargetPixels(pickingRenderTarget, 0, 0, 1, 1, pixelBuffer);
            const id = pixelBuffer[0] + (pixelBuffer[1] * 256) + (pixelBuffer[2] * 256 * 256);
            
            points.material = material;
            
            if (id < data.length && data[id]) {
                const d = data[id];
                const currentSizes = geometry.attributes.size.array;
                
                if (currentSizes[id] > 0) {
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
                
                for (const pc of topPCs) {
                    const zscore = d[`pc${pc}_zscore`];
                    const percentile = d[`pc${pc}_percentile`];
                    const shapAI = d[`pc${pc}_shap_ai`];
                    const shapSC = d[`pc${pc}_shap_sc`];
                    const importance = pcImportance[pc];
                    
                    pcInfo += `
                        <div style="margin-bottom: 8px; padding: 5px; background: #222; border-radius: 3px;">
                            <div class="pc-score"><span><strong>PC${pc}</strong></span><span style="color: #666; font-size: 10px;">Importance: ${(importance * 100).toFixed(1)}%</span></div>
                            <div class="pc-score"><span>Z-score: <span class="highlight">${zscore > 0 ? '+' : ''}${zscore.toFixed(2)}</span></span><span>Percentile: <span class="highlight">${percentile.toFixed(0)}</span></span></div>
                            <div class="pc-score"><span>AI: <span class="${shapAI > 0 ? 'shap-positive' : 'shap-negative'}">${shapAI > 0 ? '+' : ''}${shapAI.toFixed(3)}</span></span><span>SC: <span class="${shapSC > 0 ? 'shap-positive' : 'shap-negative'}">${shapSC > 0 ? '+' : ''}${shapSC.toFixed(3)}</span></span></div>
                        </div>
                    `;
                }
                pcInfo += '</div>';
                
                tooltip.innerHTML = `
                    <div class="tooltip-header">Essay #${d.essay_id}</div>
                    <div>Social Class: <span class="highlight">${d.sc11}</span> | AI Rating: <span class="highlight">${d.ai_rating.toFixed(2)}</span></div>
                    ${pcInfo}
                    <div style="margin-top: 10px; font-style: italic; color: #888; font-size: 11px;">
                        "${d.essay.substring(0, 150)}..."
                    </div>
                `;
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 10) + 'px';
                tooltip.style.top = (event.clientY + 10) + 'px';
                } else {
                    tooltip.style.display = 'none';
                    highlightSphere.visible = false;
                }
            } else {
                tooltip.style.display = 'none';
                highlightSphere.visible = false;
            }
        }"""

content = content[:func_start] + new_function + content[func_end:]

# Save the modified HTML
output_file = '/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/umap_dml_top5_pcs_gpu_picking.html'
with open(output_file, 'w') as f:
    f.write(content)

print(f"Created GPU picking version: {output_file}")
print("This HTML file now uses GPU picking for precise hover detection!")