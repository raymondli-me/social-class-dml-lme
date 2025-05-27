#!/usr/bin/env python3
"""
Add debounced controls to the UMAP visualization
"""

import re
from pathlib import Path

# Read the original file
input_file = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/scripts/analyze_dml_top_pcs_umap.py")
with open(input_file, 'r') as f:
    content = f.read()

# Find where the JavaScript starts
js_start = content.find('// Setup Three.js')

# Add debounce function after Three.js setup
debounce_code = '''
        // Debounce function for smooth performance
        function debounce(func, wait) {
            let timeout;
            return function(...args) {
                clearTimeout(timeout);
                timeout = setTimeout(() => func.apply(this, args), wait);
            };
        }
        
'''

# Insert debounce function
insert_pos = content.find('// Setup Three.js')
content = content[:insert_pos] + debounce_code + content[insert_pos:]

# Replace onchange with oninput for immediate feedback, but debounced
replacements = [
    # Point size
    (
        r'<input type="range" id="point-size"([^>]+)onchange="updatePointSize\(\)">',
        r'<input type="range" id="point-size"\1oninput="debouncedUpdatePointSize()">'
    ),
    # Opacity
    (
        r'<input type="range" id="point-opacity"([^>]+)onchange="updateOpacity\(\)">',
        r'<input type="range" id="point-opacity"\1oninput="debouncedUpdateOpacity()">'
    ),
    # Cloud scale
    (
        r'<input type="range" id="cloud-scale"([^>]+)onchange="updateCloudScale\(\)">',
        r'<input type="range" id="cloud-scale"\1oninput="debouncedUpdateCloudScale()">'
    ),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Add debounced versions of the functions after their definitions
debounced_functions = '''
        // Create debounced versions of update functions
        const debouncedUpdatePointSize = debounce(updatePointSize, 50);
        const debouncedUpdateOpacity = debounce(updateOpacity, 50);
        const debouncedUpdateCloudScale = debounce(updateCloudScale, 100);
        const debouncedUpdateVisibility = debounce(updateVisibility, 150);
        
'''

# Find where to insert (after the update functions are defined)
insert_pos = content.find('// Initialize color legend')
content = content[:insert_pos] + debounced_functions + content[insert_pos:]

# Also update the PC filter inputs to use debounced version
content = re.sub(
    r'oninput="updatePCFilter\((\d+)\)"',
    r'oninput="debouncedUpdatePCFilter(\1)"',
    content
)

# Add debounced PC filter update
pc_debounce = '''
        // Debounced PC filter update
        const debouncedUpdatePCFilter = debounce(updatePCFilter, 100);
        
'''

# Insert before animation loop
insert_pos = content.find('// Animation')
content = content[:insert_pos] + pc_debounce + content[insert_pos:]

# Also add immediate visual feedback for sliders
slider_feedback = '''
        // Immediate visual feedback for sliders (value display only)
        document.getElementById('point-size').addEventListener('input', function() {
            document.getElementById('size-val').textContent = this.value;
        });
        
        document.getElementById('point-opacity').addEventListener('input', function() {
            document.getElementById('opacity-val').textContent = this.value;
        });
        
        document.getElementById('cloud-scale').addEventListener('input', function() {
            document.getElementById('scale-val').textContent = this.value;
        });
        
'''

# Insert before initialization
insert_pos = content.find('// Initialize color legend')
content = content[:insert_pos] + slider_feedback + content[insert_pos:]

# Save as new file
output_file = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/scripts/analyze_dml_top_pcs_umap_debounced.py")
with open(output_file, 'w') as f:
    f.write(content)

print(f"Created debounced version: {output_file}")
print("\nChanges made:")
print("✓ Added debounce function")
print("✓ Changed onchange to oninput for immediate feedback")
print("✓ Debounced all heavy update operations")
print("✓ Added immediate visual feedback for slider values")
print("\nDebounce delays:")
print("- Point size: 50ms")
print("- Opacity: 50ms")
print("- Cloud scale: 100ms")
print("- Visibility filters: 150ms")