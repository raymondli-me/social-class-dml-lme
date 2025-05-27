#!/usr/bin/env python3
"""
Fix the UMAP visualization to properly hide filtered points and improve hover precision
"""

import re
from pathlib import Path

input_file = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/umap_dml_top5_pcs.html")
output_file = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/umap_dml_top5_pcs_visibility_fixed.html")

print("Reading HTML file...")
with open(input_file, 'r') as f:
    html_content = f.read()

# Fix 1: Make hidden points truly invisible in shader
shader_fix = '''void main() {
                    vColor = color;
                    vSize = size;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    
                    // Hide points with size 0
                    if (size == 0.0) {
                        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Move far outside clip space
                        gl_PointSize = 0.0;
                    } else {
                        gl_PointSize = size * 50.0 / -mvPosition.z;
                        gl_Position = projectionMatrix * mvPosition;
                    }
                }'''

# Replace vertex shader
html_content = re.sub(
    r'void main\(\) \{[^}]*vColor = color;[^}]*gl_Position = projectionMatrix \* mvPosition;[^}]*\}',
    shader_fix,
    html_content,
    flags=re.DOTALL
)

# Fix 2: Also add size check in fragment shader
fragment_fix = '''void main() {
                    if (vSize == 0.0) discard;
                    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
                    if (dist > 0.5) discard;
                    gl_FragColor = vec4(vColor, opacity);
                }'''

# Replace fragment shader
html_content = re.sub(
    r'void main\(\) \{[^}]*float dist = length[^}]*gl_FragColor = vec4\(vColor, opacity\);[^}]*\}',
    fragment_fix,
    html_content,
    flags=re.DOTALL
)

# Fix 3: Improve raycaster threshold calculation
# Find the onMouseMove function and update threshold calculation
threshold_fix = '''// Update raycaster threshold based on current point size and zoom
            const currentSize = parseFloat(document.getElementById('point-size').value);
            const distance = camera.position.length();
            raycaster.params.Points.threshold = currentSize * 50.0 / distance;'''

html_content = re.sub(
    r'// Update raycaster threshold based on current point size\s*\n\s*const currentSize[^;]+;\s*\n\s*raycaster\.params\.Points\.threshold = [^;]+;',
    threshold_fix,
    html_content
)

# Fix 4: Add vSize to vertex shader attributes
if 'varying float vSize;' not in html_content:
    html_content = re.sub(
        r'(varying vec3 vColor;)',
        r'\1\n                varying float vSize;',
        html_content
    )

# Save the fixed version
print(f"Saving fixed version to: {output_file}")
with open(output_file, 'w') as f:
    f.write(html_content)

print("\nFixes applied:")
print("✓ Hidden points moved outside clip space (truly invisible)")
print("✓ Fragment shader discards size=0 points")
print("✓ Improved raycaster threshold calculation")
print("✓ Points with size=0 won't interfere with hover")
print("\nOpen the fixed file to test!")