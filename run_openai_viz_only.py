#!/usr/bin/env python3
"""Run only the visualization part of OpenAI analysis"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.openai_embedding_analysis import *

def main():
    print("Loading data and checkpoints...")
    
    # Load data
    df = load_data()
    
    # Load existing embeddings
    embeddings = compute_embeddings(df)
    
    # Load existing PCA
    X_pca, pca, scaler = compute_pca(embeddings, n_components=200)
    
    # Load existing DML results
    with open(CHECKPOINT_DIR / "dml_results_openai.pkl", 'rb') as f:
        dml_results = pickle.load(f)
    
    print("\nDML Results loaded:")
    for outcome in ['AI_ratings', 'actual_SC']:
        print(f"\n{outcome}:")
        for model in ['Linear', 'Lasso', 'RF', 'XGBoost']:
            r2 = dml_results[outcome][model]['r2']
            print(f"  {model:8s}: R² = {r2:.3f}")
    
    # Compute UMAP if needed
    umap_3d = compute_umap_3d(X_pca)
    
    # Create visualizations
    print("\nCreating interactive visualizations...")
    create_interactive_umap(df, umap_3d, X_pca, dml_results)
    
    # Import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create SHAP visualizations
    create_shap_visualizations(dml_results, X_pca)
    
    print("\nVisualization complete!")
    print(f"Check outputs in: {VIZ_DIR}")

if __name__ == "__main__":
    main()