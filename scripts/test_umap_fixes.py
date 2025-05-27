#!/usr/bin/env python3
"""
Automated tests for UMAP visualization fixes.

This script tests:
1. Data structure integrity
2. Coordinate calculations
3. HTML modifications
4. JavaScript functionality
"""

import re
import json
import numpy as np
from pathlib import Path
import sys

class UMAPVisualizationTester:
    def __init__(self, html_file):
        self.html_file = Path(html_file)
        self.html_content = None
        self.data = None
        self.test_results = []
        
    def load_html(self):
        """Load and parse the HTML file."""
        try:
            with open(self.html_file, 'r') as f:
                self.html_content = f.read()
            return True, "HTML loaded successfully"
        except Exception as e:
            return False, f"Failed to load HTML: {str(e)}"
    
    def extract_data(self):
        """Extract the data array from the HTML."""
        try:
            # Find the data array
            data_match = re.search(r'const data = (\[.*?\]);', self.html_content, re.DOTALL)
            if not data_match:
                return False, "Could not find data array in HTML"
            
            data_str = data_match.group(1)
            self.data = json.loads(data_str)
            return True, f"Extracted {len(self.data)} data points"
        except Exception as e:
            return False, f"Failed to extract data: {str(e)}"
    
    def test_world_coordinates(self):
        """Test if world coordinates are properly stored in data."""
        try:
            missing_coords = []
            for i, point in enumerate(self.data[:10]):  # Check first 10 points
                if 'worldX' not in point or 'worldY' not in point or 'worldZ' not in point:
                    missing_coords.append(i)
            
            if missing_coords:
                return False, f"Missing world coordinates in points: {missing_coords}"
            
            # Check if world coordinates are scaled versions of original
            scale_factor = 4.0  # Default scale factor
            tolerance = 0.001
            
            for point in self.data[:5]:
                if 'worldX' in point and 'x' in point:
                    expected_world_x = point['x'] * scale_factor
                    if abs(point['worldX'] - expected_world_x) > tolerance:
                        return False, f"World coordinate mismatch: worldX={point['worldX']}, expected={expected_world_x}"
            
            return True, "World coordinates properly initialized"
        except Exception as e:
            return False, f"World coordinate test failed: {str(e)}"
    
    def test_custom_raycasting(self):
        """Test if custom raycasting function exists."""
        try:
            # Check for performRaycast function
            if 'function performRaycast()' not in self.html_content:
                return False, "Custom performRaycast function not found"
            
            # Check for visible point filtering
            if 'visibleData = []' not in self.html_content:
                return False, "Visible data filtering not implemented"
            
            # Check for dynamic threshold calculation
            if 'baseSizeOnScreen' not in self.html_content:
                return False, "Dynamic threshold calculation not found"
            
            return True, "Custom raycasting implementation found"
        except Exception as e:
            return False, f"Raycasting test failed: {str(e)}"
    
    def test_scale_synchronization(self):
        """Test if scale updates synchronize world coordinates."""
        try:
            # Check updateCloudScale function
            if 'function updateCloudScale()' not in self.html_content:
                return False, "updateCloudScale function not found"
            
            # Check if it updates data world coordinates
            scale_update_pattern = r'data\[i\]\.worldX\s*=\s*originalPositions\[i \* 3\]\s*\*\s*newScale'
            if not re.search(scale_update_pattern, self.html_content):
                return False, "Scale update doesn't synchronize world coordinates"
            
            return True, "Scale synchronization implemented correctly"
        except Exception as e:
            return False, f"Scale synchronization test failed: {str(e)}"
    
    def test_hidden_point_handling(self):
        """Test if hidden points are properly excluded."""
        try:
            # Check fragment shader for size check
            shader_pattern = r'if\s*\(\s*vSize\s*==\s*0\.0\s*\)\s*discard;'
            if not re.search(shader_pattern, self.html_content):
                return False, "Fragment shader doesn't discard hidden points"
            
            # Check if raycasting filters by size
            size_filter_pattern = r'if\s*\(\s*sizes\[i\]\s*>\s*0\s*\)'
            if not re.search(size_filter_pattern, self.html_content):
                return False, "Raycasting doesn't filter hidden points"
            
            return True, "Hidden points properly handled"
        except Exception as e:
            return False, f"Hidden point test failed: {str(e)}"
    
    def test_debug_mode(self):
        """Test if debug mode is implemented."""
        try:
            # Check for debug checkbox
            if 'id="debug-mode"' not in self.html_content:
                return False, "Debug mode checkbox not found"
            
            # Check for debug info div
            if 'id="debug-info"' not in self.html_content:
                return False, "Debug info display not found"
            
            # Check for debug update code
            if 'Debug info' not in self.html_content or 'World Pos:' not in self.html_content:
                return False, "Debug info update code not found"
            
            return True, "Debug mode fully implemented"
        except Exception as e:
            return False, f"Debug mode test failed: {str(e)}"
    
    def test_highlight_sphere_positioning(self):
        """Test if highlight sphere uses world coordinates."""
        try:
            # Check for world coordinate positioning
            highlight_pattern = r'highlightSphere\.position\.set\(d\.worldX,\s*d\.worldY,\s*d\.worldZ\)'
            if not re.search(highlight_pattern, self.html_content):
                # Also check for alternative patterns
                alt_pattern = r'highlightSphere\.position\.set\(\s*d\.worldX'
                if not re.search(alt_pattern, self.html_content):
                    return False, "Highlight sphere doesn't use world coordinates"
            
            return True, "Highlight sphere positioning correct"
        except Exception as e:
            return False, f"Highlight sphere test failed: {str(e)}"
    
    def test_pc_data_integrity(self):
        """Test if PC data is properly included."""
        try:
            required_fields = ['pc1_zscore', 'pc1_percentile', 'pc1_shap_ai', 'pc1_shap_sc']
            
            missing_fields = []
            for field in required_fields:
                found = False
                for point in self.data[:5]:  # Check first 5 points
                    if field in point:
                        found = True
                        break
                if not found:
                    missing_fields.append(field)
            
            if missing_fields:
                return False, f"Missing PC fields: {missing_fields}"
            
            # Check PC value ranges
            for i in range(1, 6):  # PC1 through PC5
                pc_field = f'pc{i}_percentile'
                percentiles = [p.get(pc_field, -1) for p in self.data if pc_field in p]
                if percentiles:
                    min_p, max_p = min(percentiles), max(percentiles)
                    if min_p < 0 or max_p > 100:
                        return False, f"Invalid percentile range for PC{i}: {min_p}-{max_p}"
            
            return True, "PC data integrity verified"
        except Exception as e:
            return False, f"PC data test failed: {str(e)}"
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print(f"\n{'='*60}")
        print(f"UMAP Visualization Test Suite")
        print(f"Testing: {self.html_file}")
        print(f"{'='*60}\n")
        
        # Define all tests
        tests = [
            ("HTML Loading", self.load_html),
            ("Data Extraction", self.extract_data),
            ("World Coordinates", self.test_world_coordinates),
            ("Custom Raycasting", self.test_custom_raycasting),
            ("Scale Synchronization", self.test_scale_synchronization),
            ("Hidden Point Handling", self.test_hidden_point_handling),
            ("Debug Mode", self.test_debug_mode),
            ("Highlight Sphere", self.test_highlight_sphere_positioning),
            ("PC Data Integrity", self.test_pc_data_integrity)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                success, message = test_func()
                status = "PASS" if success else "FAIL"
                color = "\033[92m" if success else "\033[91m"
                reset = "\033[0m"
                
                print(f"{color}[{status}]{reset} {test_name}: {message}")
                
                if success:
                    passed += 1
                else:
                    failed += 1
                    
                self.test_results.append({
                    'name': test_name,
                    'status': status,
                    'message': message
                })
            except Exception as e:
                print(f"\033[91m[ERROR]\033[0m {test_name}: {str(e)}")
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Test Summary: {passed} passed, {failed} failed")
        print(f"{'='*60}\n")
        
        return passed, failed
    
    def generate_detailed_report(self):
        """Generate a detailed test report."""
        report = []
        report.append("UMAP Visualization Test Report")
        report.append("=" * 60)
        report.append(f"File: {self.html_file}")
        report.append(f"Data points: {len(self.data) if self.data else 'N/A'}")
        report.append("")
        
        if self.data and len(self.data) > 0:
            # Sample data analysis
            report.append("Sample Data Point Analysis:")
            point = self.data[0]
            report.append(f"  TID: {point.get('TID', 'N/A')}")
            report.append(f"  Original coords: ({point.get('x', 'N/A')}, {point.get('y', 'N/A')}, {point.get('z', 'N/A')})")
            report.append(f"  World coords: ({point.get('worldX', 'N/A')}, {point.get('worldY', 'N/A')}, {point.get('worldZ', 'N/A')})")
            report.append(f"  Social class: {point.get('sc11', 'N/A')}")
            report.append(f"  AI rating: {point.get('rating', 'N/A')}")
            report.append("")
            
            # PC data summary
            report.append("Principal Component Summary:")
            for i in range(1, 6):
                z_field = f'pc{i}_zscore'
                p_field = f'pc{i}_percentile'
                if z_field in point:
                    report.append(f"  PC{i}: z-score={point[z_field]:.3f}, percentile={point[p_field]:.1f}")
        
        report.append("")
        report.append("Test Results:")
        for result in self.test_results:
            report.append(f"  {result['status']}: {result['name']} - {result['message']}")
        
        return "\n".join(report)


def main():
    # Test both the original and fixed versions
    base_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
    
    files_to_test = [
        base_dir / "nvembed_dml_pc_analysis" / "umap_dml_top5_pcs.html",
        base_dir / "nvembed_dml_pc_analysis" / "umap_dml_top5_pcs_hover_fixed.html"
    ]
    
    for html_file in files_to_test:
        if html_file.exists():
            tester = UMAPVisualizationTester(html_file)
            passed, failed = tester.run_all_tests()
            
            # Save detailed report
            report_file = html_file.with_suffix('.test_report.txt')
            with open(report_file, 'w') as f:
                f.write(tester.generate_detailed_report())
            print(f"Detailed report saved to: {report_file}")
        else:
            print(f"File not found: {html_file}")
    
    # Compare the two versions
    if len(files_to_test) == 2 and all(f.exists() for f in files_to_test):
        print("\n" + "="*60)
        print("Comparison between original and fixed versions:")
        print("="*60)
        
        # Load both files
        with open(files_to_test[0], 'r') as f:
            original = f.read()
        with open(files_to_test[1], 'r') as f:
            fixed = f.read()
        
        # Key improvements to check
        improvements = {
            'World coordinates': 'worldX' in fixed and 'worldX' not in original,
            'Custom raycasting': 'performRaycast' in fixed and 'performRaycast' not in original,
            'Debug mode': 'debug-mode' in fixed and 'debug-mode' not in original,
            'Hidden point handling': 'vSize == 0.0' in fixed and 'vSize == 0.0' not in original
        }
        
        for feature, present in improvements.items():
            status = "Added" if present else "Not added"
            print(f"  {feature}: {status}")


if __name__ == "__main__":
    main()