#!/usr/bin/env python3
import pickle
import numpy as np

# Load the pickle file
with open('nvembed_dml_pc_analysis/dml_pc_analysis_results_with_umap.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in pickle file:", list(data.keys()))
print("\nData shapes/types:")
for k, v in data.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {type(v)}")

# Check if we have TreeSHAP data
if 'shap_values' in data:
    print("\nSHAP values shape:", data['shap_values'].shape)
    print("SHAP values type:", type(data['shap_values']))
    
# Check for PC data
if 'data_with_pcs' in data:
    df = data['data_with_pcs']
    print("\nDataFrame columns:", list(df.columns)[:10], "...")
    print("DataFrame shape:", df.shape)
    
    # Check for PC columns
    pc_cols = [col for col in df.columns if col.startswith('PC')]
    print(f"\nFound {len(pc_cols)} PC columns")
    if pc_cols:
        print("First few PC columns:", pc_cols[:5])