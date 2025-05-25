#!/usr/bin/env python3
"""
Add filters for AI ratings 1-10 (in addition to social class 1-5)
Allow filtering by the actual values being displayed
"""

import re
from datetime import datetime

def add_ai_rating_filters():
    """Add appropriate filters based on what's being displayed"""
    
    # Read the current version
    working_file = "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_fixed_ai_display_20250525_090811.html"
    
    with open(working_file, 'r') as f:
        html_content = f.read()
    
    # 1. Replace the filter controls section
    old_controls = """        <div style="margin-bottom: 15px;">
            <strong>Filter by Social Class:</strong>
            <label><input type="checkbox" id="sc1" checked> Class 1</label>
            <label><input type="checkbox" id="sc2" checked> Class 2</label>
            <label><input type="checkbox" id="sc3" checked> Class 3</label>
            <label><input type="checkbox" id="sc4" checked> Class 4</label>
            <label><input type="checkbox" id="sc5" checked> Class 5</label>
        </div>"""
    
    new_controls = """        <div style="margin-bottom: 15px;">
            <strong>Filter by Value:</strong>
            <div id="filter-container" style="max-height: 200px; overflow-y: auto;">
                <!-- Filters will be added dynamically based on color mode -->
            </div>
        </div>"""
    
    html_content = html_content.replace(old_controls, new_controls)
    
    # 2. Add dynamic filter creation based on colorBy mode
    filter_js = """
        
        // Create filters based on color mode
        function createFilters() {
            const filterContainer = document.getElementById('filter-container');
            filterContainer.innerHTML = '';
            
            if (colorBy === 'ai_rating') {
                // Create filters for AI ratings 1-10
                filterContainer.innerHTML = '<em style="font-size:11px">AI Rating (1-10):</em><br>';
                for (let i = 1; i <= 10; i++) {
                    const label = document.createElement('label');
                    label.innerHTML = `<input type="checkbox" id="ai${i}" checked> ${i}`;
                    label.style.display = 'inline-block';
                    label.style.marginRight = '10px';
                    label.style.marginBottom = '5px';
                    filterContainer.appendChild(label);
                    
                    // Add line break after 5
                    if (i === 5) {
                        filterContainer.appendChild(document.createElement('br'));
                    }
                }
            } else {
                // Create filters for social class 1-5
                filterContainer.innerHTML = '<em style="font-size:11px">Social Class (1-5):</em><br>';
                for (let i = 1; i <= 5; i++) {
                    const label = document.createElement('label');
                    label.innerHTML = `<input type="checkbox" id="sc${i}" checked> Class ${i}`;
                    label.style.display = 'block';
                    filterContainer.appendChild(label);
                }
            }
            
            // Add event listeners
            filterContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.addEventListener('change', updateFilter);
            });
        }"""
    
    # Insert the filter creation function
    html_content = html_content.replace(
        '        // Color legend functions',
        filter_js + '\n        \n        // Color legend functions'
    )
    
    # 3. Update the updateFilter function to handle both AI ratings and social class
    old_update_filter = """        function updateFilter() {
            const checkboxes = {
                1: document.getElementById('sc1').checked,
                2: document.getElementById('sc2').checked,
                3: document.getElementById('sc3').checked,
                4: document.getElementById('sc4').checked,
                5: document.getElementById('sc5').checked
            };"""
    
    new_update_filter = """        function updateFilter() {
            const checkboxes = {};
            
            if (colorBy === 'ai_rating') {
                // Check AI rating filters (1-10)
                for (let i = 1; i <= 10; i++) {
                    const checkbox = document.getElementById(`ai${i}`);
                    checkboxes[i] = checkbox ? checkbox.checked : true;
                }
            } else {
                // Check social class filters (1-5)
                for (let i = 1; i <= 5; i++) {
                    const checkbox = document.getElementById(`sc${i}`);
                    checkboxes[i] = checkbox ? checkbox.checked : true;
                }
            }"""
    
    html_content = html_content.replace(old_update_filter, new_update_filter)
    
    # 4. Update the filter logic to filter by the appropriate value
    old_filter_logic = """            data.forEach((point, i) => {
                if (checkboxes[point.actual_sc]) {"""
    
    new_filter_logic = """            data.forEach((point, i) => {
                const value = colorBy === 'ai_rating' ? Math.round(point.ai_rating) : point.actual_sc;
                if (checkboxes[value]) {"""
    
    html_content = html_content.replace(old_filter_logic, new_filter_logic)
    
    # 5. Call createFilters after createColorLegend
    html_content = html_content.replace(
        '            // Create color legend\n            createColorLegend();',
        '            // Create color legend\n            createColorLegend();\n            \n            // Create filters\n            createFilters();'
    )
    
    # 6. Remove the old filter initialization
    old_filter_init = """            ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].forEach(id => {
                document.getElementById(id).addEventListener('change', updateFilter);
            });"""
    
    html_content = html_content.replace(old_filter_init, '            // Filter listeners added dynamically in createFilters()')
    
    # 7. Update title
    html_content = html_content.replace(
        '<title>Social Class UMAP - Fixed AI Rating Display</title>',
        '<title>Social Class UMAP - With AI Rating Filters</title>'
    )
    
    # Save new version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_ai_filters_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Added AI rating filters to: {output_file}")
    print("Changes made:")
    print("- Dynamic filters: 1-10 for AI ratings, 1-5 for social class")
    print("- Filters change based on what's being displayed")
    print("- AI rating filters round to nearest integer (e.g., 3.5 → 4)")
    print("- Compact layout with checkboxes in rows")
    
    return output_file

if __name__ == "__main__":
    add_ai_rating_filters()