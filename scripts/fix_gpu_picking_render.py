#!/usr/bin/env python3
"""
Fix the GPU picking to not interfere with normal rendering
"""

# Read the original HTML
with open('../nvembed_dml_pc_analysis/umap_dml_top5_pcs.html', 'r') as f:
    original = f.read()

# Just add a simple 2D screen-space hover instead of complex GPU picking
# This avoids all the shader/material switching issues

new_hover = """
        // Screen-space hover detection (simpler and more reliable)
        const screenPositions = new Array(data.length);
        
        function updateScreenPositions() {
            const vector = new THREE.Vector3();
            const positions = geometry.attributes.position.array;
            const sizes = geometry.attributes.size.array;
            
            for (let i = 0; i < data.length; i++) {
                if (sizes[i] > 0) {
                    vector.set(
                        positions[i * 3],
                        positions[i * 3 + 1],
                        positions[i * 3 + 2]
                    );
                    
                    vector.project(camera);
                    
                    const x = (vector.x + 1) / 2 * window.innerWidth;
                    const y = -(vector.y - 1) / 2 * window.innerHeight;
                    
                    screenPositions[i] = {
                        x: x,
                        y: y,
                        z: vector.z,
                        size: sizes[i] * 2, // Adjust for screen size
                        visible: vector.z > 0 && vector.z < 1
                    };
                } else {
                    screenPositions[i] = { visible: false };
                }
            }
        }"""

# Replace the raycaster setup with screen-space setup
original = original.replace(
    "// Tooltip\n        const tooltip = document.getElementById('tooltip');\n        const raycaster = new THREE.Raycaster();\n        const mouse = new THREE.Vector2();\n        \n        // Custom raycaster that respects point sizes\n        raycaster.params.Points.threshold = 3;  // Adjusted for larger point sizes",
    "// Tooltip\n        const tooltip = document.getElementById('tooltip');" + new_hover
)

# Replace onMouseMove with screen-space version
new_mouse_handler = """
        function onMouseMove(event) {
            updateScreenPositions();
            
            const mouseX = event.clientX;
            const mouseY = event.clientY;
            
            let closestIndex = -1;
            let closestDistance = Infinity;
            
            // Find closest visible point in screen space
            for (let i = 0; i < data.length; i++) {
                const sp = screenPositions[i];
                if (sp && sp.visible) {
                    const dx = mouseX - sp.x;
                    const dy = mouseY - sp.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < sp.size && distance < closestDistance) {
                        closestDistance = distance;
                        closestIndex = i;
                    }
                }
            }
            
            if (closestIndex >= 0) {
                const d = data[closestIndex];
                const positions = geometry.attributes.position.array;
                
                // Position highlight sphere
                highlightSphere.position.set(
                    positions[closestIndex * 3],
                    positions[closestIndex * 3 + 1],
                    positions[closestIndex * 3 + 2]
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
        }"""

# Find and replace the old onMouseMove
start = original.find("function onMouseMove(event) {")
end = original.find("document.addEventListener('mousemove', onMouseMove);")
old_func = original[start:end]

original = original.replace(old_func, new_mouse_handler + "\n        ")

# Save the fixed version
with open('../nvembed_dml_pc_analysis/umap_dml_top5_pcs_screen_hover.html', 'w') as f:
    f.write(original)

print("Created screen-space hover version: umap_dml_top5_pcs_screen_hover.html")
print("This version uses 2D screen-space detection which is simpler and more reliable")