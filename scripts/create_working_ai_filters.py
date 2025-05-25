#!/usr/bin/env python3
"""Create a working AI rating filter visualization by modifying the working base"""

import shutil
from datetime import datetime
from pathlib import Path

def create_working_ai_filters():
    # Start from the known working version
    viz_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations")
    source_file = viz_dir / "umap_with_legend_20250525_090240.html"
    
    # Read the working file
    with open(source_file, 'r') as f:
        lines = f.readlines()
    
    # Find and modify specific sections
    modified_lines = []
    in_checkbox_section = False
    in_getcolor_function = False
    in_updatefilter_function = False
    
    for i, line in enumerate(lines):
        # Replace the checkbox section
        if '<strong>Filter by Social Class:</strong>' in line:
            modified_lines.append('            <strong>Filter by Rating:</strong>\n')
            modified_lines.append('            <div id="dynamic-filters"></div>\n')
            # Skip the next 5 lines (the old checkboxes)
            for j in range(5):
                if i + j + 1 < len(lines):
                    lines[i + j + 1] = ''
            continue
            
        # Replace the getColor function
        elif 'function getColor(value, type) {' in line:
            modified_lines.append(line)
            # Replace the entire function
            modified_lines.append('''            if (type === 'ai_rating') {
                // Map 1-10 to rainbow spectrum
                const norm = (value - 1) / 9;
                const hue = (1 - norm) * 240 / 360; // 240 (blue) to 0 (red)
                const color = new THREE.Color();
                color.setHSL(hue, 0.8, 0.5);
                return color;
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
        }
''')
            # Skip the original function body
            j = i + 1
            brace_count = 1
            while j < len(lines) and brace_count > 0:
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                lines[j] = ''
                j += 1
            continue
            
        # Add dynamic filter creation after scene setup
        elif "document.getElementById('debug-info').innerHTML" in line:
            modified_lines.append(line)
            modified_lines.append('''
        // Create dynamic filters
        const filterContainer = document.getElementById('dynamic-filters');
        window.activeFilters = {};
        
        if (colorBy === 'ai_rating') {
            // Create AI rating filters (1, 1.5, 2, ..., 9.5)
            const values = [];
            for (let v = 1; v <= 9.5; v += 0.5) {
                values.push(v);
            }
            
            values.forEach((val, idx) => {
                const label = document.createElement('label');
                label.style.marginRight = '8px';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = true;
                cb.value = val;
                cb.onchange = updateFilter;
                window.activeFilters[val] = cb;
                
                label.appendChild(cb);
                label.appendChild(document.createTextNode(' ' + val));
                filterContainer.appendChild(label);
                
                if ((idx + 1) % 6 === 0) {
                    filterContainer.appendChild(document.createElement('br'));
                }
            });
        } else {
            // Create social class filters (1-5)
            for (let i = 1; i <= 5; i++) {
                const label = document.createElement('label');
                label.style.marginRight = '10px';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = true;
                cb.value = i;
                cb.onchange = updateFilter;
                window.activeFilters[i] = cb;
                
                label.appendChild(cb);
                label.appendChild(document.createTextNode(' Class ' + i));
                filterContainer.appendChild(label);
            }
        }
''')
            continue
            
        # Update the updateFilter function
        elif 'function updateFilter() {' in line:
            modified_lines.append(line)
            modified_lines.append('''            const positions = [];
            const colors = [];
            const filteredUserData = [];
            
            data.forEach((point) => {
                let include = false;
                
                if (colorBy === 'ai_rating') {
                    // Check if this AI rating is selected
                    const filter = window.activeFilters[point.ai_rating];
                    include = filter && filter.checked;
                } else {
                    // Check if this social class is selected
                    const filter = window.activeFilters[point.actual_sc];
                    include = filter && filter.checked;
                }
                
                if (include) {
                    positions.push(point.x, point.y, point.z);
                    const value = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                    const color = getColor(value, colorBy);
                    colors.push(color.r, color.g, color.b);
                    filteredUserData.push(point);
                }
            });
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            points.userData = filteredUserData;
            
            document.getElementById('debug-info').innerHTML = 
                `Points: ${filteredUserData.length}<br>Hidden: ${data.length - filteredUserData.length}`;
        }
''')
            # Skip original function body
            j = i + 1
            brace_count = 1
            while j < len(lines) and brace_count > 0:
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                lines[j] = ''
                j += 1
            continue
            
        # Remove old checkbox references
        elif 'const checkboxes = {' in line:
            # Skip this section
            j = i
            while j < len(lines) and '};' not in lines[j]:
                lines[j] = ''
                j += 1
            if j < len(lines):
                lines[j] = ''
            continue
            
        else:
            modified_lines.append(line)
    
    # Save the modified file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = viz_dir / f"umap_working_ai_filters_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.writelines(modified_lines)
    
    print(f"Created working AI filters at: {output_file}")

if __name__ == "__main__":
    create_working_ai_filters()