#!/usr/bin/env python3
"""
Fix AI rating display to show full 1-10 range clearly
Also show actual range of AI ratings in stats
"""

import re
from datetime import datetime

def fix_ai_rating_display():
    """Fix color scaling and display for AI ratings"""
    
    # Read the color legend version
    working_file = "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_with_legend_20250525_090240.html"
    
    with open(working_file, 'r') as f:
        html_content = f.read()
    
    # 1. Update stats to show actual AI rating range
    html_content = html_content.replace(
        '<strong>Colored by:</strong> AI Rating',
        '<strong>Colored by:</strong> AI Rating (1-10 scale, avg of 2 prompts)'
    )
    
    # 2. Add more legend points to show full scale
    # Replace the legend creation to show more values
    old_legend = """            // For AI ratings, show subset (1, 3, 5, 7, 9, 10)
            [1, 3, 5, 7, 9, 10].forEach(i => {"""
    
    new_legend = """            // For AI ratings, show full scale 1-10
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].forEach(i => {"""
    
    html_content = html_content.replace(old_legend, new_legend)
    
    # 3. Add a note about the color scale
    # Update the legend title
    html_content = html_content.replace(
        '<strong>Color Legend:</strong>',
        '<strong>AI Rating Scale (1-10):</strong>'
    )
    
    # 4. Add JavaScript to show min/max AI ratings in debug
    debug_addition = """
            
            // Calculate actual AI rating range
            let minAI = 10, maxAI = 1;
            data.forEach(point => {
                minAI = Math.min(minAI, point.ai_rating);
                maxAI = Math.max(maxAI, point.ai_rating);
            });
            document.getElementById('debug-info').innerHTML += `<br>AI Range: ${minAI.toFixed(1)} - ${maxAI.toFixed(1)}`;"""
    
    # Insert after radius display
    html_content = html_content.replace(
        'document.getElementById(\'debug-info\').innerHTML += `<br>Radius: ${bounds.radius.toFixed(1)}`;',
        'document.getElementById(\'debug-info\').innerHTML += `<br>Radius: ${bounds.radius.toFixed(1)}`;' + debug_addition
    )
    
    # 5. Make legend more compact for 10 values
    legend_css_update = """        .legend-item {
            display: flex;
            align-items: center;
            margin: 3px 0;
        }"""
    
    new_legend_css = """        .legend-item {
            display: flex;
            align-items: center;
            margin: 2px 0;
            font-size: 11px;
        }"""
    
    html_content = html_content.replace(legend_css_update, new_legend_css)
    
    # 6. Update title
    html_content = html_content.replace(
        '<title>Social Class UMAP - With Color Legend</title>',
        '<title>Social Class UMAP - Fixed AI Rating Display</title>'
    )
    
    # Save new version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_fixed_ai_display_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Fixed AI rating display in: {output_file}")
    print("Changes made:")
    print("- Shows full 1-10 scale in legend (all values)")
    print("- Updated stats to clarify it's 1-10 scale")
    print("- Added actual AI range display in debug info")
    print("- Made legend more compact")
    print("- Note: AI ratings are averages of 2 prompts, so can be 1.5, 2.5, etc.")
    
    return output_file

if __name__ == "__main__":
    fix_ai_rating_display()