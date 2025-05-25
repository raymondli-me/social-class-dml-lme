#!/usr/bin/env python3
"""Fix AI rating color scale to properly show full 1-10 range with averaged values"""

import re
from datetime import datetime
from pathlib import Path

def fix_ai_rating_colors():
    # Read the most recent AI filters HTML
    viz_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations")
    html_file = viz_dir / "umap_ai_filters_20250525_091153.html"
    
    with open(html_file, 'r') as f:
        content = f.read()
    
    # Find the getColor function and replace it with a fixed version
    # This regex finds the entire getColor function
    color_func_pattern = r'function getColor\(value, type\) \{[^}]+\}'
    
    new_color_func = '''function getColor(value, type) {
            if (type === 'ai_rating') {
                // AI ratings range from 1-10 (but averaged values go from 1.0 to 9.5)
                // Use full spectrum: purple (1) -> blue -> green -> yellow -> red (10)
                const normalizedValue = (value - 1) / 9; // Normalize to 0-1
                
                if (normalizedValue <= 0.25) {
                    // Purple to blue (1-3.25)
                    const t = normalizedValue * 4;
                    return new THREE.Color(
                        1 - t * 0.5,     // R: 1 -> 0.5
                        t * 0.5,         // G: 0 -> 0.5
                        1                // B: 1 -> 1
                    );
                } else if (normalizedValue <= 0.5) {
                    // Blue to green (3.25-5.5)
                    const t = (normalizedValue - 0.25) * 4;
                    return new THREE.Color(
                        0.5 - t * 0.5,   // R: 0.5 -> 0
                        0.5 + t * 0.5,   // G: 0.5 -> 1
                        1 - t            // B: 1 -> 0
                    );
                } else if (normalizedValue <= 0.75) {
                    // Green to yellow (5.5-7.75)
                    const t = (normalizedValue - 0.5) * 4;
                    return new THREE.Color(
                        t,               // R: 0 -> 1
                        1,               // G: 1 -> 1
                        0                // B: 0 -> 0
                    );
                } else {
                    // Yellow to red (7.75-10)
                    const t = (normalizedValue - 0.75) * 4;
                    return new THREE.Color(
                        1,               // R: 1 -> 1
                        1 - t,           // G: 1 -> 0
                        0                // B: 0 -> 0
                    );
                }
            } else {
                // Social class: 1-5
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
    
    # Also fix the legend to show the actual AI rating range
    # Find and replace the legend creation for AI ratings
    legend_pattern = r'// Add labels.*?legendLabels\.push\(.*?\);'
    
    new_legend_code = '''// Add labels
        if (colorBy === 'ai_rating') {
            // Show actual range: 1.0 to 9.5 (no essay has perfect 10.0 average)
            legendLabels.push(
                { text: '10', y: 10 },
                { text: '9', y: 20 },
                { text: '8', y: 30 },
                { text: '7', y: 40 },
                { text: '6', y: 50 },
                { text: '5', y: 60 },
                { text: '4', y: 70 },
                { text: '3', y: 80 },
                { text: '2', y: 90 },
                { text: '1', y: 100 }
            );
        } else {
            legendLabels.push(
                { text: '5', y: 10 },
                { text: '4', y: 30 },
                { text: '3', y: 50 },
                { text: '2', y: 70 },
                { text: '1', y: 90 }
            );
        }'''
    
    content = re.sub(legend_pattern, new_legend_code, content, flags=re.DOTALL)
    
    # Fix the filter checkboxes to show half values for AI ratings
    filter_creation_pattern = r'if \(colorBy === \'ai_rating\'\) \{[^}]+for \(let i = 1; i <= 10; i\+\+\) \{[^}]+\}[^}]+\}'
    
    new_filter_code = '''if (colorBy === 'ai_rating') {
            // AI ratings are averaged, so we get values like 1.0, 1.5, 2.0, etc.
            filterDiv.innerHTML = '<strong>Filter by AI Rating:</strong><br>';
            const values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5];
            
            values.forEach(val => {
                const label = document.createElement('label');
                label.innerHTML = `<input type="checkbox" id="ai${val}" checked> ${val}`;
                label.style.marginRight = '10px';
                if (val % 2 === 0) {
                    label.style.marginRight = '20px'; // Extra space after whole numbers
                }
                filterDiv.appendChild(label);
                
                // Add line break after every 4 values
                if (val === 2.5 || val === 4.5 || val === 6.5 || val === 8.5) {
                    filterDiv.appendChild(document.createElement('br'));
                }
            });
        }'''
    
    content = re.sub(filter_creation_pattern, new_filter_code, content, flags=re.DOTALL)
    
    # Also update the filter update function to handle half values
    update_filter_pattern = r'if \(colorBy === \'ai_rating\'\) \{[^}]+checkboxes\[Math\.round\(point\.ai_rating\)\][^}]+\}'
    
    new_update_filter = '''if (colorBy === 'ai_rating') {
                    // Check the exact value (1.0, 1.5, 2.0, etc.)
                    const checkbox = document.getElementById(`ai${point.ai_rating}`);
                    if (checkbox && checkbox.checked) {
                        positions.push(point.x, point.y, point.z);
                        colors.push(color.r, color.g, color.b);
                        filteredUserData.push(point);
                    }
                }'''
    
    content = re.sub(update_filter_pattern, new_update_filter, content, flags=re.DOTALL)
    
    # Save the fixed version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = viz_dir / f"umap_ai_colors_fixed_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Fixed AI rating colors in: {output_file}")
    print("\nChanges made:")
    print("- Full color spectrum from purple (1) through red (10)")
    print("- Filters show actual values: 1, 1.5, 2, 2.5, ..., 9, 9.5")
    print("- Legend shows full 1-10 scale")
    print("- Proper mapping of averaged ratings to colors")

if __name__ == "__main__":
    fix_ai_rating_colors()