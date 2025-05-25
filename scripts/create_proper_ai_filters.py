#!/usr/bin/env python3
"""Create a proper AI rating filter visualization from the working version"""

import shutil
from datetime import datetime
from pathlib import Path
import re

def create_proper_ai_filters():
    # Start from the known working version with legend
    viz_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations")
    source_file = viz_dir / "umap_with_legend_20250525_090240.html"
    
    # Create a new file with AI rating filters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = viz_dir / f"umap_proper_ai_filters_{timestamp}.html"
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # 1. Update the getColor function to use full 1-10 scale for AI ratings
    color_func_pattern = r'function getColor\(value, type\) \{[^}]+\}'
    
    new_color_func = '''function getColor(value, type) {
            if (type === 'ai_rating') {
                // AI ratings: 1-10 scale (averaged values range from 1.0 to 9.5)
                // Map to full color spectrum
                const normalizedValue = (value - 1) / 9;  // Normalize to 0-1
                
                if (normalizedValue <= 0.25) {
                    // Purple to blue (1-3.25)
                    const t = normalizedValue * 4;
                    return new THREE.Color(
                        1 - t * 0.5,    // R: 1 -> 0.5
                        t * 0.5,        // G: 0 -> 0.5
                        1               // B: 1 -> 1
                    );
                } else if (normalizedValue <= 0.5) {
                    // Blue to green (3.25-5.5)
                    const t = (normalizedValue - 0.25) * 4;
                    return new THREE.Color(
                        0.5 - t * 0.5,  // R: 0.5 -> 0
                        0.5 + t * 0.5,  // G: 0.5 -> 1
                        1 - t           // B: 1 -> 0
                    );
                } else if (normalizedValue <= 0.75) {
                    // Green to yellow (5.5-7.75)
                    const t = (normalizedValue - 0.5) * 4;
                    return new THREE.Color(
                        t,              // R: 0 -> 1
                        1,              // G: 1 -> 1
                        0               // B: 0 -> 0
                    );
                } else {
                    // Yellow to red (7.75-10)
                    const t = (normalizedValue - 0.75) * 4;
                    return new THREE.Color(
                        1,              // R: 1 -> 1
                        1 - t,          // G: 1 -> 0
                        0               // B: 0 -> 0
                    );
                }
            } else {
                // Social class colors
                const colors = [
                    new THREE.Color(0.2, 0.2, 0.8),  // Class 1: Blue
                    new THREE.Color(0.2, 0.8, 0.8),  // Class 2: Cyan
                    new THREE.Color(0.2, 0.8, 0.2),  // Class 3: Green
                    new THREE.Color(0.8, 0.8, 0.2),  // Class 4: Yellow
                    new THREE.Color(0.8, 0.2, 0.2)   // Class 5: Red
                ];
                return colors[Math.floor(value) - 1] || colors[0];
            }
        }'''
    
    content = re.sub(color_func_pattern, new_color_func, content, flags=re.DOTALL)
    
    # 2. Replace the static social class filters with dynamic ones
    # Find the filter section in controls
    filter_section_pattern = r'<div style="margin-bottom: 15px;">\s*<strong>Filter by Social Class:</strong>.*?</div>'
    
    new_filter_section = '''<div style="margin-bottom: 15px;">
            <div id="filter-controls">
                <!-- Dynamic filters will be inserted here -->
            </div>
        </div>'''
    
    content = re.sub(filter_section_pattern, new_filter_section, content, flags=re.DOTALL)
    
    # 3. Add the dynamic filter creation code after data loading
    # Find where the debug info is updated and add our filter code after it
    debug_update = "document.getElementById('debug-info').innerHTML"
    debug_index = content.find(debug_update)
    if debug_index > 0:
        # Find the end of this statement
        end_index = content.find(';', debug_index) + 1
        
        filter_creation_code = '''
        
        // Create dynamic filters based on colorBy
        const filterDiv = document.getElementById('filter-controls');
        const checkboxes = {};
        
        if (colorBy === 'ai_rating') {
            // Create filters for AI rating values (1.0 to 9.5 in 0.5 increments)
            filterDiv.innerHTML = '<strong>Filter by AI Rating:</strong><br>';
            const values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5];
            
            values.forEach((val, index) => {
                const label = document.createElement('label');
                label.style.marginRight = '10px';
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `ai${val}`;
                checkbox.checked = true;
                checkbox.addEventListener('change', updateFilter);
                
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(` ${val}`));
                filterDiv.appendChild(label);
                
                checkboxes[val] = checkbox;
                
                // Line break after every 6 values
                if ((index + 1) % 6 === 0) {
                    filterDiv.appendChild(document.createElement('br'));
                }
            });
        } else {
            // Create filters for social class (1-5)
            filterDiv.innerHTML = '<strong>Filter by Social Class:</strong><br>';
            for (let i = 1; i <= 5; i++) {
                const label = document.createElement('label');
                label.style.marginRight = '15px';
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `sc${i}`;
                checkbox.checked = true;
                checkbox.addEventListener('change', updateFilter);
                
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(` Class ${i}`));
                filterDiv.appendChild(label);
                
                checkboxes[i] = checkbox;
            }
        }'''
        
        content = content[:end_index] + filter_creation_code + content[end_index:]
    
    # 4. Update the updateFilter function to handle both types
    # Find the updateFilter function
    update_filter_pattern = r'function updateFilter\(\) \{[^}]+\}'
    
    # First, let's see what the current pattern looks like
    match = re.search(update_filter_pattern, content, re.DOTALL)
    if match:
        # Replace with a more comprehensive version
        new_update_filter = '''function updateFilter() {
            const positions = [];
            const colors = [];
            const filteredUserData = [];
            
            data.forEach((point, i) => {
                let shouldInclude = false;
                
                if (colorBy === 'ai_rating') {
                    // Check if this AI rating value is enabled
                    const checkbox = checkboxes[point.ai_rating];
                    shouldInclude = checkbox && checkbox.checked;
                } else {
                    // Check if this social class is enabled
                    const checkbox = checkboxes[point.actual_sc];
                    shouldInclude = checkbox && checkbox.checked;
                }
                
                if (shouldInclude) {
                    positions.push(point.x, point.y, point.z);
                    const value = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                    const color = getColor(value, colorBy);
                    colors.push(color.r, color.g, color.b);
                    filteredUserData.push(point);
                }
            });
            
            // Update geometry
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            // Update raycaster data
            points.userData = filteredUserData;
            
            // Update stats
            document.getElementById('debug-info').innerHTML = `
                Points: ${filteredUserData.length}<br>
                Filtered: ${data.length - filteredUserData.length} hidden
            `;
        }'''
        
        content = re.sub(update_filter_pattern, new_update_filter, content, flags=re.DOTALL)
    
    # 5. Make sure the checkbox references are removed from the initial setup
    # Remove old checkbox setup code
    checkbox_pattern = r"const checkboxes = \{[^}]+\};"
    content = re.sub(checkbox_pattern, "", content)
    
    # Save the file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Created proper AI filters visualization at: {output_file}")
    print("\nFeatures:")
    print("- Dynamic filters: 1-9.5 for AI ratings, 1-5 for social class")
    print("- Full color spectrum for AI ratings (purple to red)")
    print("- Filters update based on what's being displayed")

if __name__ == "__main__":
    create_proper_ai_filters()