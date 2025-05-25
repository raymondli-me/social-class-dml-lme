#!/usr/bin/env python3
"""Create a debug version to check data loading"""

from pathlib import Path
from datetime import datetime

def create_debug_viz():
    viz_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/custom_visualizations")
    
    # Minimal HTML with console logging
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Debug Visualization</title>
    <style>
        body { font-family: monospace; padding: 20px; }
        pre { background: #f0f0f0; padding: 10px; overflow: auto; }
    </style>
</head>
<body>
    <h1>Debug Visualization</h1>
    <div id="status">Loading data...</div>
    <pre id="output"></pre>
    
    <script>
        const log = (msg) => {
            document.getElementById('output').innerHTML += msg + '\\n';
            console.log(msg);
        };
        
        try {
            // Check if data exists
            log('Starting debug...');
            
            // Simple data array for testing
            const testData = [
                {id: "TEST1", x: 0, y: 0, z: 0, ai_rating: 3.5, actual_sc: 3},
                {id: "TEST2", x: 10, y: 10, z: 10, ai_rating: 7.5, actual_sc: 4}
            ];
            
            log(`Test data loaded: ${testData.length} points`);
            
            // Check AI ratings
            const aiRatings = testData.map(d => d.ai_rating);
            log(`AI ratings: ${aiRatings.join(', ')}`);
            
            // Check unique values
            const uniqueRatings = [...new Set(aiRatings)];
            log(`Unique AI ratings: ${uniqueRatings.join(', ')}`);
            
            document.getElementById('status').innerHTML = 'Debug complete - check output below';
            
        } catch (error) {
            log(`ERROR: ${error.message}`);
            log(`Stack: ${error.stack}`);
        }
    </script>
</body>
</html>'''
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = viz_dir / f"debug_viz_{timestamp}.html"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Created debug visualization at: {output_file}")

if __name__ == "__main__":
    create_debug_viz()