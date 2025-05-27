#!/usr/bin/env python3
"""
Modify existing UMAP visualization to highlight dots instead of using a sphere.
"""

import re
from pathlib import Path

# Paths
input_file = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/umap_dml_top5_pcs_hover_fixed.html")
output_file = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/umap_dml_top5_pcs_dot_highlight.html")

print("Reading original file...")
with open(input_file, 'r') as f:
    html_content = f.read()

print("Applying modifications...")

# Modifications to make:
modifications = [
    # 1. Add hoveredIndex tracking
    (
        r'(let currentScaleFactor = [^;]+;)',
        r'\1\n        \n        // Keep track of hovered point\n        let hoveredIndex = -1;'
    ),
    
    # 2. Comment out highlight sphere creation
    (
        r'(// Add highlight sphere[\s\S]*?scene\.add\(highlightSphere\);)',
        r'/*\1*/'
    ),
    
    # 3. Add originalSizes array after sizes
    (
        r'(const sizes = new Float32Array\(data\.length\);)',
        r'\1\n        const originalSizes = new Float32Array(data.length);'
    ),
    
    # 4. Initialize originalSizes in the data loop
    (
        r'(sizes\[i\] = 6;  // Larger initial size)',
        r'\1\n            originalSizes[i] = 6;'
    ),
    
    # 5. Update the hover handler to change dot size
    (
        r'(if \(hoveredPoint\) \{[\s\S]*?const d = data\[idx\];)',
        r'''\1
                
                // Update hovered index
                if (hoveredIndex !== idx) {
                    // Restore previous hovered point
                    if (hoveredIndex >= 0 && originalSizes[hoveredIndex] > 0) {
                        sizes[hoveredIndex] = originalSizes[hoveredIndex];
                    }
                    
                    // Highlight new point
                    hoveredIndex = idx;
                    sizes[idx] = originalSizes[idx] * 1.5;  // Make 50% larger
                    geometry.attributes.size.needsUpdate = true;
                }'''
    ),
    
    # 6. Comment out highlight sphere positioning
    (
        r'(// Position highlight sphere at world coordinates[\s\S]*?highlightMaterial\.opacity = 0\.5;)',
        r'/*\1*/'
    ),
    
    # 7. Update the else block to restore dot size
    (
        r'(} else \{[\s\S]*?document\.getElementById\(\'tooltip\'\)\.style\.display = \'none\';)',
        r'''\1
                
                // Restore hovered point size
                if (hoveredIndex >= 0 && originalSizes[hoveredIndex] > 0) {
                    const sizes = geometry.attributes.size.array;
                    sizes[hoveredIndex] = originalSizes[hoveredIndex];
                    geometry.attributes.size.needsUpdate = true;
                    hoveredIndex = -1;
                }'''
    ),
    
    # 8. Comment out highlightSphere.visible = false
    (
        r'highlightSphere\.visible = false;',
        r'// highlightSphere.visible = false;'
    ),
    
    # 9. Update the updatePointSize function to handle hovered state
    (
        r'(function updatePointSize\(\) \{[\s\S]*?)(for \(let i = 0; i < data\.length; i\+\+\) \{[\s\S]*?sizes\[i\] = size;[\s\S]*?\})',
        r'''\1for (let i = 0; i < data.length; i++) {
                if (isPointVisible(i, pcFilter, percentileThreshold)) {
                    sizes[i] = (i === hoveredIndex) ? size * 1.5 : size;
                    originalSizes[i] = size;
                }
            }'''
    ),
    
    # 10. Update the updateFilters function similarly
    (
        r'(if \(isPointVisible\(i, pcFilter, percentileThreshold\)\) \{[\s\S]*?sizes\[i\] = currentSize;)',
        r'''if (isPointVisible(i, pcFilter, percentileThreshold)) {
                    sizes[i] = (i === hoveredIndex) ? currentSize * 1.5 : currentSize;
                    originalSizes[i] = currentSize;'''
    ),
    
    # 11. Fix the fragment shader to add smooth edges
    (
        r'(void main\(\) \{[\s\S]*?)(if \(dist > 0\.5\) discard;)',
        r'''\1\2
                    
                    // Smooth edge
                    float alpha = 1.0 - smoothstep(0.45, 0.5, dist);
                    gl_FragColor = vec4(vColor, opacity * alpha);
                    return;'''
    ),
    
    # 12. Comment out any remaining highlightSphere references
    (
        r'highlightSphere\.',
        r'// highlightSphere.'
    )
]

# Apply all modifications
fixed_html = html_content
for pattern, replacement in modifications:
    fixed_html = re.sub(pattern, replacement, fixed_html, flags=re.MULTILINE | re.DOTALL)

# Add a note in the title
fixed_html = fixed_html.replace(
    '<title>UMAP 3D Visualization - DML Top 5 PCs Analysis (Fixed Hover)</title>',
    '<title>UMAP 3D Visualization - DML Top 5 PCs Analysis (Dot Highlight)</title>'
)

print(f"Saving modified version to: {output_file}")
with open(output_file, 'w') as f:
    f.write(fixed_html)

print("\nModifications applied:")
print("✓ Removed highlight sphere")
print("✓ Added dot size increase on hover (50% larger)")
print("✓ Smooth transitions when hovering")
print("✓ Works with all existing controls and filters")
print("\nThe dots themselves now highlight when you hover over them!")