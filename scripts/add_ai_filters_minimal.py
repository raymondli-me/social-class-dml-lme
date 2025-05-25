#!/usr/bin/env python3
"""Add AI rating filters with minimal changes to working file"""

from pathlib import Path
from datetime import datetime
import re

def add_ai_filters_minimal():
    viz_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations")
    
    # Find the clean copy we just made
    import glob
    clean_files = sorted(glob.glob(str(viz_dir / "umap_ai_filters_clean_*.html")))
    if not clean_files:
        print("No clean file found!")
        return
        
    source_file = clean_files[-1]
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # 1. First, let's fix the color function for AI ratings
    # Find the getColor function and enhance it
    color_func = """function getColor(value, type) {
            if (type === 'ai_rating') {
                // AI ratings 1-10: purple to red spectrum
                const norm = (value - 1) / 9;
                if (norm <= 0.25) {
                    // Purple to blue
                    const t = norm * 4;
                    return new THREE.Color(1 - t * 0.5, t * 0.5, 1);
                } else if (norm <= 0.5) {
                    // Blue to green
                    const t = (norm - 0.25) * 4;
                    return new THREE.Color(0.5 - t * 0.5, 0.5 + t * 0.5, 1 - t);
                } else if (norm <= 0.75) {
                    // Green to yellow
                    const t = (norm - 0.5) * 4;
                    return new THREE.Color(t, 1, 0);
                } else {
                    // Yellow to red
                    const t = (norm - 0.75) * 4;
                    return new THREE.Color(1, 1 - t, 0);
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
        }"""
    
    # Replace the getColor function
    content = re.sub(
        r'function getColor\(value, type\) \{[^}]+\}',
        color_func,
        content,
        flags=re.DOTALL
    )
    
    # 2. Replace static filters with dynamic container
    content = re.sub(
        r'<strong>Filter by Social Class:</strong>(.*?)</div>',
        '<strong id="filter-title">Filter by Rating:</strong>\\n            <div id="filter-checkboxes"></div>\\n        </div>',
        content,
        flags=re.DOTALL
    )
    
    # 3. Add dynamic filter creation code after the debug info update
    # Find where we update debug info and add our code there
    debug_line = "document.getElementById('debug-info').innerHTML = `"
    idx = content.find(debug_line)
    if idx > 0:
        # Find the end of this statement
        end_idx = content.find('`;', idx) + 2
        
        insert_code = """
        
        // Create dynamic filters
        const filterContainer = document.getElementById('filter-checkboxes');
        const filterTitle = document.getElementById('filter-title');
        
        if (colorBy === 'ai_rating') {
            filterTitle.textContent = 'Filter by AI Rating:';
            // Create filters for AI ratings
            const values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5];
            values.forEach((val, idx) => {
                const label = document.createElement('label');
                label.style.marginRight = '8px';
                label.innerHTML = `<input type="checkbox" class="rating-filter" data-value="${val}" checked> ${val}`;
                filterContainer.appendChild(label);
                if ((idx + 1) % 6 === 0) filterContainer.appendChild(document.createElement('br'));
            });
        } else {
            filterTitle.textContent = 'Filter by Social Class:';
            // Create filters for social class
            for (let i = 1; i <= 5; i++) {
                const label = document.createElement('label');
                label.style.marginRight = '10px';
                label.innerHTML = `<input type="checkbox" class="rating-filter" data-value="${i}" checked> Class ${i}`;
                filterContainer.appendChild(label);
            }
        }
        
        // Add event listeners
        document.querySelectorAll('.rating-filter').forEach(cb => {
            cb.addEventListener('change', updateFilter);
        });
"""
        
        content = content[:end_idx] + insert_code + content[end_idx:]
    
    # 4. Update the updateFilter function to use dynamic filters
    new_update_filter = """function updateFilter() {
            const positions = [];
            const colors = [];
            const filteredUserData = [];
            
            // Get active filters
            const activeValues = new Set();
            document.querySelectorAll('.rating-filter:checked').forEach(cb => {
                activeValues.add(parseFloat(cb.dataset.value));
            });
            
            data.forEach((point, i) => {
                const checkValue = colorBy === 'ai_rating' ? point.ai_rating : point.actual_sc;
                
                if (activeValues.has(checkValue)) {
                    positions.push(point.x, point.y, point.z);
                    const color = getColor(checkValue, colorBy);
                    colors.push(color.r, color.g, color.b);
                    filteredUserData.push(point);
                }
            });
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            points.userData = filteredUserData;
            
            document.getElementById('debug-info').innerHTML = `
                Points: ${filteredUserData.length}<br>
                Hidden: ${data.length - filteredUserData.length}
            `;
        }"""
    
    content = re.sub(
        r'function updateFilter\(\) \{[^}]+\}',
        new_update_filter,
        content,
        flags=re.DOTALL
    )
    
    # 5. Remove old checkbox object
    content = re.sub(
        r'const checkboxes = \{[^}]+\};',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Save the result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = viz_dir / f"umap_ai_filters_final_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Created final AI filters visualization at: {output_file}")
    print("Features:")
    print("- Dynamic filters for AI ratings (1-9.5) or social class (1-5)")
    print("- Full color spectrum for AI ratings")
    print("- Minimal changes to preserve working state")

if __name__ == "__main__":
    add_ai_filters_minimal()