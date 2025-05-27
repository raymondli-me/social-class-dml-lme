#!/usr/bin/env python3
"""
Create fixed UMAP visualization with proper hover-visual alignment.

This script addresses the critical hover-visual misalignment issue by:
1. Storing world coordinates in the data objects
2. Implementing custom raycasting that accounts for screen-space point sizes
3. Synchronizing scaling between visual rendering and hover detection
4. Properly filtering hidden points from raycasting
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# Set up paths
base_dir = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
data_dir = base_dir / "nvembed_dml_pc_analysis"
output_dir = data_dir

# Load the analysis results
print("Loading analysis results...")
results_file = data_dir / "dml_pc_analysis_results.pkl"
if not results_file.exists():
    print(f"Error: Results file not found at {results_file}")
    print("Please run analyze_dml_top_pcs_umap.py first")
    exit(1)

results = pd.read_pickle(results_file)

# Extract data
df = results['data_with_pcs']
umap_3d = results['umap_3d']
top_pcs = results['top_pcs'][:5]  # Top 5 PCs
feature_importance_ai = results['feature_importance_ai']
feature_importance_sc = results['feature_importance_sc']

print(f"Loaded data for {len(df)} essays")
print(f"Top 5 PCs: {top_pcs}")

# Prepare data for visualization
viz_data = []
for idx, row in df.iterrows():
    essay_excerpt = row['essay'][:150] if pd.notna(row['essay']) else "No essay text"
    
    point_data = {
        'x': float(umap_3d[idx, 0]),
        'y': float(umap_3d[idx, 1]),
        'z': float(umap_3d[idx, 2]),
        'TID': int(row['TID']),
        'essay_excerpt': essay_excerpt.replace('"', '\\"').replace('\n', ' '),
        'sc11': int(row['sc11']),
        'rating': float(row['rating'])
    }
    
    # Add PC data
    for pc in top_pcs:
        pc_col = f'PC{pc}'
        if pc_col in df.columns:
            pc_value = row[pc_col]
            
            # Calculate z-score
            pc_mean = df[pc_col].mean()
            pc_std = df[pc_col].std()
            z_score = (pc_value - pc_mean) / pc_std if pc_std > 0 else 0
            
            # Calculate percentile
            percentile = (df[pc_col] < pc_value).sum() / len(df) * 100
            
            # Get SHAP-like contributions (simplified)
            # These would ideally come from SHAP values, but we'll use feature importance as proxy
            shap_ai = feature_importance_ai.get(pc_col, 0) * z_score * 0.1  # Scaled contribution
            shap_sc = feature_importance_sc.get(pc_col, 0) * z_score * 0.1
            
            point_data[f'pc{pc}_value'] = float(pc_value)
            point_data[f'pc{pc}_zscore'] = float(z_score)
            point_data[f'pc{pc}_percentile'] = float(percentile)
            point_data[f'pc{pc}_shap_ai'] = float(shap_ai)
            point_data[f'pc{pc}_shap_sc'] = float(shap_sc)
    
    viz_data.append(point_data)

# Read the HTML template
template_path = data_dir / "umap_dml_top5_pcs_fixed.html"
with open(template_path, 'r') as f:
    html_template = f.read()

# Replace the data placeholder
html_content = html_template.replace('@@DATA@@', json.dumps(viz_data))

# Save the visualization
output_path = output_dir / "umap_dml_top5_pcs_fixed_final.html"
with open(output_path, 'w') as f:
    f.write(html_content)

print(f"\nFixed visualization saved to: {output_path}")
print("\nKey improvements in this version:")
print("1. World coordinates stored in data objects for consistent hover detection")
print("2. Custom raycasting that accounts for screen-space point sizes")
print("3. Synchronized scaling between visual and hover systems")
print("4. Hidden points properly excluded from raycasting")
print("5. Debug mode to visualize hover detection accuracy")
print("\nThe hover-visual alignment should now work correctly!")