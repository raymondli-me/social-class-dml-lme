#!/usr/bin/env python3
"""
Debug why the GPU picking version shows black screen
"""

# Read the GPU picking HTML
with open('../nvembed_dml_pc_analysis/umap_dml_top5_pcs_gpu_picking.html', 'r') as f:
    content = f.read()

# Add console logging to debug
debug_additions = """
        console.log('Data loaded:', data.length, 'points');
        console.log('First point:', data[0]);
        console.log('Geometry created');
        console.log('Positions:', positions.slice(0, 9));
        console.log('Colors:', colors.slice(0, 9));
        console.log('Sizes:', sizes.slice(0, 3));
        """

# Insert after data loading
insert_pos = content.find("geometry.setAttribute('position'")
content = content[:insert_pos] + debug_additions + "\n        " + content[insert_pos:]

# Add material debug
material_debug = """
        console.log('Visual material created:', material);
        console.log('Picking material created:', pickingMaterial);
        """

insert_pos2 = content.find("const pickingRenderTarget")
content = content[:insert_pos2] + material_debug + "\n        " + content[insert_pos2:]

# Add render debug
render_debug = """
            console.log('Rendering frame');
            if (!points.material) {
                console.error('Points have no material!');
            }
            """

insert_pos3 = content.find("controls.update();")
content = content[:insert_pos3] + render_debug + "\n            " + content[insert_pos3:]

# Fix potential issue - make sure points use visual material by default
fix = content.replace(
    "const points = new THREE.Points(geometry, material);",
    """const points = new THREE.Points(geometry, material);
        console.log('Points created with material:', points.material);"""
)
content = fix

# Save debug version
with open('../nvembed_dml_pc_analysis/umap_dml_top5_pcs_gpu_debug.html', 'w') as f:
    f.write(content)

print("Created debug version: umap_dml_top5_pcs_gpu_debug.html")
print("Open this in browser and check console for errors")

# Also check if the original non-GPU version still works
print("\nAlso verify the original still works:")
print("umap_dml_top5_pcs.html")