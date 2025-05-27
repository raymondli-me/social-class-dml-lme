#!/usr/bin/env python3
"""
Fix the hover-visual alignment issue in the existing UMAP visualization.

This script:
1. Reads the existing UMAP HTML file
2. Extracts the data
3. Applies fixes for hover-visual alignment
4. Saves a new fixed version
"""

import re
import json
from pathlib import Path

# Paths
base_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
input_file = base_dir / "nvembed_dml_pc_analysis" / "umap_dml_top5_pcs.html"
output_file = base_dir / "nvembed_dml_pc_analysis" / "umap_dml_top5_pcs_hover_fixed.html"

print("Reading original visualization...")
with open(input_file, 'r') as f:
    html_content = f.read()

# Extract the data array
data_match = re.search(r'const data = (\[.*?\]);', html_content, re.DOTALL)
if not data_match:
    print("Error: Could not find data array in HTML")
    exit(1)

# Key fixes to apply:
# 1. Fix the raycasting implementation
# 2. Store world coordinates in data objects
# 3. Implement custom raycasting function
# 4. Fix the shader size calculation consistency

# Find and replace the raycaster setup and mouse move function
fixes = [
    # Fix 1: Initialize currentScaleFactor before using it
    (
        r'// Initial scale factor\s*\n\s*let currentScaleFactor = 4\.0;',
        '''// Initial scale factor
        let currentScaleFactor = 4.0;
        
        // Store world positions in data for consistent hover detection'''
    ),
    
    # Fix 2: Update data objects with world coordinates in the initial setup
    (
        r'data\.forEach\(\(d, i\) => \{\s*\n\s*// Scale the coordinates for more spacing',
        '''data.forEach((d, i) => {
            // Store scaled world positions
            d.worldX = d.x * currentScaleFactor;
            d.worldY = d.y * currentScaleFactor;
            d.worldZ = d.z * currentScaleFactor;
            
            // Scale the coordinates for more spacing'''
    ),
    
    # Fix 3: Use world coordinates in positioning
    (
        r'const scaledX = d\.x \* currentScaleFactor;\s*\n\s*const scaledY = d\.y \* currentScaleFactor;\s*\n\s*const scaledZ = d\.z \* currentScaleFactor;',
        '''const scaledX = d.worldX;
            const scaledY = d.worldY;
            const scaledZ = d.worldZ;'''
    ),
    
    # Fix 4: Update cloud scale function to maintain world coordinates
    (
        r'function updateCloudScale\(\) \{\s*\n\s*const newScale = parseFloat\(document\.getElementById\(\'cloud-scale\'\)\.value\);',
        '''function updateCloudScale() {
            const newScale = parseFloat(document.getElementById('cloud-scale').value);
            document.getElementById('scale-val').textContent = newScale;
            currentScaleFactor = newScale;
            
            // Update world coordinates in data'''
    ),
    
    # Fix 5: Add world coordinate updates in the scaling loop
    (
        r'for \(let i = 0; i < data\.length; i\+\+\) \{\s*\n\s*const scaledX = originalPositions\[i \* 3\] \* newScale;',
        '''for (let i = 0; i < data.length; i++) {
                // Update world coordinates in data
                data[i].worldX = originalPositions[i * 3] * newScale;
                data[i].worldY = originalPositions[i * 3 + 1] * newScale;
                data[i].worldZ = originalPositions[i * 3 + 2] * newScale;
                
                const scaledX = data[i].worldX;'''
    ),
    
    # Fix 6: Replace the raycasting implementation
    (
        r'function onMouseMove\(event\) \{[\s\S]*?// Position highlight sphere',
        '''// Custom raycasting function that accounts for point size and scale
        function performRaycast() {
            raycaster.setFromCamera(mouse, camera);
            
            // Get visible points only
            const visibleData = [];
            const sizes = geometry.attributes.size.array;
            
            for (let i = 0; i < data.length; i++) {
                if (sizes[i] > 0) {
                    visibleData.push({
                        index: i,
                        position: new THREE.Vector3(
                            data[i].worldX,
                            data[i].worldY,
                            data[i].worldZ
                        ),
                        size: sizes[i]
                    });
                }
            }
            
            // Calculate dynamic threshold based on camera distance
            const baseSizeOnScreen = 50.0;  // Match shader constant
            
            // Find closest visible point
            let closestPoint = null;
            let closestDistance = Infinity;
            
            for (const point of visibleData) {
                const worldPos = point.position;
                const distance = raycaster.ray.distanceToPoint(worldPos);
                
                // Calculate screen-space size for this point
                const pointDistance = worldPos.distanceTo(camera.position);
                const screenSize = (point.size * baseSizeOnScreen) / pointDistance;
                const worldThreshold = (screenSize * pointDistance) / baseSizeOnScreen;
                
                if (distance < worldThreshold && distance < closestDistance) {
                    closestDistance = distance;
                    closestPoint = point;
                }
            }
            
            return closestPoint;
        }
        
        function onMouseMove(event) {
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            const hoveredPoint = performRaycast();
            
            if (hoveredPoint) {
                const idx = hoveredPoint.index;
                const d = data[idx];
                
                // Position highlight sphere'''
    ),
    
    # Fix 7: Update highlight sphere positioning to use world coordinates
    (
        r'const pos = visibleIntersect\.point;\s*\n\s*highlightSphere\.position\.copy\(pos\);',
        '''highlightSphere.position.set(d.worldX, d.worldY, d.worldZ);'''
    ),
    
    # Fix 8: Fix the shader to not render hidden points
    (
        r'void main\(\) \{\s*\n\s*float dist = length\(gl_PointCoord - vec2\(0\.5, 0\.5\)\);',
        '''void main() {
                    if (vSize == 0.0) discard;  // Don't render hidden points
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));'''
    ),
    
    # Fix 9: Add vSize varying to vertex shader
    (
        r'varying vec3 vColor;\s*\n\s*void main\(\) \{',
        '''varying vec3 vColor;
                varying float vSize;
                
                void main() {'''
    ),
    
    # Fix 10: Pass size to fragment shader
    (
        r'vColor = color;\s*\n\s*vec4 mvPosition',
        '''vColor = color;
                    vSize = size;
                    vec4 mvPosition'''
    )
]

# Apply all fixes
fixed_html = html_content
for pattern, replacement in fixes:
    fixed_html = re.sub(pattern, replacement, fixed_html, flags=re.MULTILINE | re.DOTALL)

# Add debug info display
debug_section = '''
    <div id="debug-info"></div>
    
    <style>
        #debug-info {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 10px;
            font-family: monospace;
            display: none;
        }
    </style>
'''

# Insert debug section before closing body tag
fixed_html = fixed_html.replace('</body>', debug_section + '</body>')

# Add debug mode checkbox if not present
if 'debug-mode' not in fixed_html:
    debug_control = '''
        <div class="control-group">
            <label>
                <input type="checkbox" id="debug-mode"> Show Debug Info
            </label>
        </div>
    '''
    fixed_html = re.sub(
        r'(</div>\s*</div>\s*<div id="tooltip")',
        debug_control + r'\1',
        fixed_html
    )

# Add debug info update in hover handler
debug_update = '''
                // Debug info
                if (document.getElementById('debug-mode') && document.getElementById('debug-mode').checked) {
                    const debugInfo = document.getElementById('debug-info');
                    debugInfo.innerHTML = `
                        Point Index: ${idx}<br>
                        World Pos: (${d.worldX.toFixed(2)}, ${d.worldY.toFixed(2)}, ${d.worldZ.toFixed(2)})<br>
                        Original Pos: (${d.x.toFixed(2)}, ${d.y.toFixed(2)}, ${d.z.toFixed(2)})<br>
                        Scale Factor: ${currentScaleFactor}<br>
                        Point Size: ${sizes[idx]}<br>
                        Camera Dist: ${camera.position.length().toFixed(2)}
                    `;
                    debugInfo.style.display = 'block';
                }
'''

# Insert debug update after tooltip display
fixed_html = re.sub(
    r'(tooltip\.style\.top = \(event\.clientY \+ 10\) \+ \'px\';)',
    r'\1\n' + debug_update,
    fixed_html
)

# Add event listener for debug mode toggle
debug_listener = '''
        // Debug mode toggle
        const debugCheckbox = document.getElementById('debug-mode');
        if (debugCheckbox) {
            debugCheckbox.addEventListener('change', function() {
                const debugInfo = document.getElementById('debug-info');
                if (!this.checked) {
                    debugInfo.style.display = 'none';
                }
            });
        }
'''

# Insert before the animation loop
fixed_html = re.sub(
    r'(// Animation loop)',
    debug_listener + r'\n        \1',
    fixed_html
)

# Hide debug info when not hovering
hide_debug = '''
            } else {
                document.getElementById('tooltip').style.display = 'none';
                highlightSphere.visible = false;
                if (document.getElementById('debug-info')) {
                    document.getElementById('debug-info').style.display = 'none';
                }
            }
'''

# Replace the else block in hover handler
fixed_html = re.sub(
    r'} else \{\s*document\.getElementById\(\'tooltip\'\)\.style\.display = \'none\';\s*highlightSphere\.visible = false;\s*}',
    hide_debug,
    fixed_html
)

# Save the fixed version
print(f"Saving fixed visualization to: {output_file}")
with open(output_file, 'w') as f:
    f.write(fixed_html)

print("\nKey fixes applied:")
print("1. ✓ World coordinates stored in data objects")
print("2. ✓ Custom raycasting implementation")
print("3. ✓ Synchronized scaling between visual and hover")
print("4. ✓ Hidden points excluded from rendering")
print("5. ✓ Debug mode added for troubleshooting")
print("\nThe hover-visual alignment should now work correctly!")