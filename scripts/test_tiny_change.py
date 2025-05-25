#!/usr/bin/env python3
"""
Test by making the TINIEST possible change to working version
Just change the title - nothing else
"""

import os
import re
from pathlib import Path
from datetime import datetime

def modify_working_html():
    """Take working HTML and make TINY change"""
    
    # Read the working file
    working_file = "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_fixed_ai_rating_20250525_085916.html"
    
    with open(working_file, 'r') as f:
        html_content = f.read()
    
    # Make TINY change - just update title
    html_content = html_content.replace(
        '<title>Social Class UMAP - Fixed Custom Visualization</title>',
        '<title>Social Class UMAP - Testing Tiny Change</title>'
    )
    
    # Add TINY visual indicator
    html_content = html_content.replace(
        '<strong>Dataset:</strong> 9,513 essays<br>',
        '<strong>Dataset:</strong> 9,513 essays [TINY TEST]<br>'
    )
    
    # Save with new timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations/umap_tiny_test_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Created tiny test version: {output_file}")
    print("ONLY changes: title and '[TINY TEST]' in stats")
    return output_file

if __name__ == "__main__":
    modify_working_html()