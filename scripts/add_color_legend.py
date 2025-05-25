#!/usr/bin/env python3
"""
Add ONLY color legend to working version
Modify HTML directly instead of regenerating
"""

import re
from datetime import datetime

def add_color_legend():
    """Add color legend to working HTML"""
    
    # Read the working file
    working_file = "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_fixed_ai_rating_20250525_085916.html"
    
    with open(working_file, 'r') as f:
        html_content = f.read()
    
    # 1. Add color legend CSS
    legend_css = """
        
        #color-legend {
            position: absolute;
            bottom: 120px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            backdrop-filter: blur(10px);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin: 3px 0;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 2px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }"""
    
    # Insert CSS before </style>
    html_content = html_content.replace('</style>', legend_css + '\n    </style>')
    
    # 2. Add color legend HTML
    legend_html = """
    
    <div id="color-legend">
        <strong>Color Legend:</strong>
        <div id="legend-items"></div>
    </div>"""
    
    # Insert HTML before the script tags
    html_content = html_content.replace('<script src="https://cdnjs.cloudflare.com', legend_html + '\n    \n    <script src="https://cdnjs.cloudflare.com')
    
    # 3. Add JavaScript functions
    js_addition = """
        
        // Color legend functions
        function getColorRGB(value, isRating) {
            const min = isRating ? 1 : 1;
            const max = isRating ? 10 : 5;
            const normalized = (value - min) / (max - min);
            
            const hue = (1 - normalized) * 240 / 360; // Blue to red
            const color = new THREE.Color();
            color.setHSL(hue, 0.8, 0.5);
            
            return `rgb(${Math.floor(color.r * 255)}, ${Math.floor(color.g * 255)}, ${Math.floor(color.b * 255)})`;
        }
        
        function createColorLegend() {
            const legendItems = document.getElementById('legend-items');
            legendItems.innerHTML = '';
            
            // For AI ratings, show subset (1, 3, 5, 7, 9, 10)
            [1, 3, 5, 7, 9, 10].forEach(i => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background: ${getColorRGB(i, true)}"></div>
                    <span>${i}</span>
                `;
                legendItems.appendChild(item);
            });
        }"""
    
    # Insert before the init() function
    html_content = html_content.replace('        function init() {', js_addition + '\n        \n        function init() {')
    
    # 4. Add call to createColorLegend() after the existing setup
    # Find the line with "// Update debug" and add the legend creation after it
    debug_update = '            // Update debug'
    if debug_update in html_content:
        html_content = html_content.replace(
            debug_update + '\n            const bounds = geometry.boundingSphere;\n            document.getElementById(\'debug-info\').innerHTML += `<br>Radius: ${bounds.radius.toFixed(1)}`;',
            debug_update + '\n            const bounds = geometry.boundingSphere;\n            document.getElementById(\'debug-info\').innerHTML += `<br>Radius: ${bounds.radius.toFixed(1)}`;\n            \n            // Create color legend\n            createColorLegend();'
        )
    
    # 5. Update title to indicate this version has color legend
    html_content = html_content.replace(
        '<title>Social Class UMAP - Fixed Custom Visualization</title>',
        '<title>Social Class UMAP - With Color Legend</title>'
    )
    
    # Save new version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_with_legend_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Added color legend to: {output_file}")
    print("Changes made:")
    print("- Added color legend CSS styles")
    print("- Added color legend HTML div")
    print("- Added getColorRGB() and createColorLegend() functions")
    print("- Called createColorLegend() after point cloud creation")
    print("- Shows AI rating scale: 1, 3, 5, 7, 9, 10 with colors")
    
    return output_file

if __name__ == "__main__":
    add_color_legend()