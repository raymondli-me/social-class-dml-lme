#!/usr/bin/env python3
"""
OpenAI Text-Embedding-3-Large Analysis Pipeline
Uses OpenAI's latest embeddings instead of sentence-transformers
Implements: Embeddings → 200 PCs → UMAP 3D → TreeSHAP → Interactive HTML
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from umap import UMAP
import shap

# OpenAI imports
from openai import OpenAI

# Set API key from environment or provided key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 
    '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')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "openai_checkpoints"
OUTPUT_DIR = BASE_DIR / "openai_outputs"
VIZ_DIR = BASE_DIR / "openai_visualizations"

# Create directories
for dir_path in [CHECKPOINT_DIR, OUTPUT_DIR, VIZ_DIR]:
    dir_path.mkdir(exist_ok=True)

def load_data():
    """Load the 9,513 essays dataset"""
    print("Loading data...")
    
    # Load essays
    df = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
    print(f"Loaded {len(df)} essays")
    
    # Load social class labels
    sc_path = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
    if sc_path.exists():
        sc_labels = pd.read_csv(sc_path)
        df = df.merge(sc_labels, on='TID', how='left')
        print(f"Merged social class labels")
    else:
        # If not found, create dummy labels
        print("Warning: Social class labels not found, using dummy values")
        df['sc11'] = np.random.randint(1, 6, size=len(df))
    
    # Rename columns for consistency
    df = df.rename(columns={'TID': 'id', 'original': 'response', 'sc11': 'self_perceived_social_class'})
    
    # Load AI ratings
    ai_ratings = pd.read_csv(BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv")
    
    # Calculate average AI rating per essay
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['id', 'ai_average']
    
    # Merge
    df = df.merge(ai_avg, on='id', how='left')
    
    # Add dummy demographic variables for DML (since they're not in the dataset)
    df['age'] = 25  # All participants were 25
    df['female'] = np.random.binomial(1, 0.5, size=len(df))  # Random gender
    df['education_level_numeric'] = df['self_perceived_social_class']  # Use SC as proxy
    
    print(f"Final dataset: {len(df)} essays with AI ratings")
    print(f"Social class distribution: {df['self_perceived_social_class'].value_counts().sort_index()}")
    return df

def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_openai_embeddings(texts, model="text-embedding-3-large", batch_size=100):
    """Get embeddings from OpenAI API with batching"""
    embeddings = []
    
    print(f"Getting embeddings for {len(texts)} texts using {model}...")
    
    # Process in batches
    batch_list = list(chunks(texts, batch_size))
    for i, batch in enumerate(batch_list):
        print(f"Processing batch {i+1}/{len(batch_list)}...", end='\r')
        try:
            response = client.embeddings.create(
                input=batch,
                model=model,
                encoding_format="float"
            )
            embeddings.extend([e.embedding for e in response.data])
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in batch: {e}")
            # Retry with smaller batch
            for text in batch:
                try:
                    response = client.embeddings.create(
                        input=[text],
                        model=model,
                        encoding_format="float"
                    )
                    embeddings.extend([e.embedding for e in response.data])
                    time.sleep(0.5)
                except Exception as e2:
                    print(f"Failed on individual text: {e2}")
                    embeddings.append(np.zeros(3072))  # Default for text-embedding-3-large
    
    print(f"\nCompleted! Generated embeddings for {len(embeddings)} texts.")
    return np.array(embeddings)

def compute_embeddings(df):
    """Compute or load OpenAI embeddings"""
    embeddings_path = CHECKPOINT_DIR / "openai_embeddings.npy"
    
    if embeddings_path.exists():
        print("Loading existing OpenAI embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        print("Computing OpenAI embeddings...")
        texts = df['response'].fillna('').tolist()
        embeddings = get_openai_embeddings(texts)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings: {embeddings.shape}")
    
    return embeddings

def compute_pca(embeddings, n_components=200):
    """Compute PCA with 200 components"""
    pca_path = CHECKPOINT_DIR / "pca_200_features.pkl"
    
    if pca_path.exists():
        print("Loading existing PCA...")
        with open(pca_path, 'rb') as f:
            pca_data = pickle.load(f)
        return pca_data['features'], pca_data['pca'], pca_data['scaler']
    
    print(f"Computing PCA with {n_components} components...")
    
    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(embeddings_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Save
    with open(pca_path, 'wb') as f:
        pickle.dump({
            'features': pca_features,
            'pca': pca,
            'scaler': scaler
        }, f)
    
    return pca_features, pca, scaler

def run_dml_analysis(df, X_pca):
    """Run DML with multiple ML methods"""
    results_path = CHECKPOINT_DIR / "dml_results_openai.pkl"
    
    if results_path.exists():
        print("Loading existing DML results...")
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    
    print("Running DML analysis...")
    
    # Prepare data
    Y_ai = df['ai_average'].values
    Y_sc = df['self_perceived_social_class'].values
    W = df[['age', 'female', 'education_level_numeric']].values
    
    # Models to test
    models = {
        'Linear': LinearRegression(),
        'Lasso': LassoCV(cv=5, max_iter=5000),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for outcome_name, Y in [('AI_ratings', Y_ai), ('actual_SC', Y_sc)]:
        print(f"\n{outcome_name}:")
        results[outcome_name] = {}
        
        for model_name, model in models.items():
            print(f"  {model_name}...")
            
            # Cross-fitting
            Y_res = np.zeros_like(Y, dtype=float)
            X_res = np.zeros_like(X_pca)
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, test_idx in kf.split(X_pca):
                # Residualize Y
                model_Y = LinearRegression()
                model_Y.fit(W[train_idx], Y[train_idx])
                Y_res[test_idx] = Y[test_idx] - model_Y.predict(W[test_idx])
                
                # Residualize each X
                for j in range(X_pca.shape[1]):
                    model_X = LinearRegression()
                    model_X.fit(W[train_idx], X_pca[train_idx, j])
                    X_res[test_idx, j] = X_pca[test_idx, j] - model_X.predict(W[test_idx])
            
            # Final model
            model.fit(X_res, Y_res)
            Y_pred = model.predict(X_res)
            
            # R-squared
            ss_res = np.sum((Y_res - Y_pred) ** 2)
            ss_tot = np.sum((Y_res - np.mean(Y_res)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            results[outcome_name][model_name] = {
                'r2': r2,
                'model': model,
                'X_res': X_res,
                'Y_res': Y_res
            }
            
            print(f" R² = {r2:.3f}")
    
    # Save results
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def compute_umap_3d(X_pca):
    """Compute 3D UMAP embedding"""
    umap_path = CHECKPOINT_DIR / "umap_3d_openai.npy"
    
    if umap_path.exists():
        print("Loading existing UMAP...")
        return np.load(umap_path)
    
    print("Computing 3D UMAP...")
    umap_model = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_3d = umap_model.fit_transform(X_pca)
    
    np.save(umap_path, umap_3d)
    return umap_3d

def create_interactive_umap(df, umap_3d, X_pca, dml_results):
    """Create interactive 3D UMAP with TreeSHAP explanations"""
    print("Creating interactive UMAP visualizations...")
    
    # Get best model (XGBoost) for SHAP
    xgb_model = dml_results['AI_ratings']['XGBoost']['model']
    X_res = dml_results['AI_ratings']['XGBoost']['X_res']
    
    # Compute SHAP values
    print("Computing TreeSHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_res)
    
    # Get top features for each point
    print("Computing top SHAP features for each point...")
    top_features_per_point = []
    for i in range(len(df)):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(df)}...")
        # Get absolute SHAP values
        abs_shap = np.abs(shap_values[i])
        # Get top 5 features
        top_idx = np.argsort(abs_shap)[-5:][::-1]
        top_features = [f"PC{idx+1}: {shap_values[i][idx]:.3f}" for idx in top_idx]
        top_features_per_point.append("<br>".join(top_features))
    print(f"  Completed {len(df)} points")
    
    # Prepare data for plotting
    plot_df = pd.DataFrame({
        'x': umap_3d[:, 0],
        'y': umap_3d[:, 1],
        'z': umap_3d[:, 2],
        'ai_rating': df['ai_average'],
        'actual_sc': df['self_perceived_social_class'],
        'education': df['education_level_numeric'],
        'essay_preview': df['response'].str[:100] + '...',  # Shorter preview
        'essay_full': df['response'],
        'top_features': top_features_per_point
    })
    
    # Create two versions with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for color_var, filename_base, title in [
        ('actual_sc', 'umap_actual_sc_openai', 'UMAP colored by Actual Social Class'),
        ('ai_rating', 'umap_ai_rating_openai', 'UMAP colored by AI Rating')
    ]:
        filename = f"{filename_base}_{timestamp}.html"
        fig = go.Figure()
        
        # Add scatter plot
        scatter = go.Scatter3d(
            x=plot_df['x'],
            y=plot_df['y'],
            z=plot_df['z'],
            mode='markers',
            marker=dict(
                size=4,
                color=plot_df[color_var],
                colorscale='Viridis',
                opacity=0.7,  # Make dots semi-transparent
                showscale=True,
                colorbar=dict(title=color_var.replace('_', ' ').title())
            ),
            customdata=np.column_stack((
                plot_df['essay_preview'],
                plot_df['ai_rating'],
                plot_df['actual_sc'],
                plot_df['education'],
                plot_df['top_features'],
                plot_df['essay_full']
            )),
            hovertemplate=(
                '<b>Essay Preview:</b><br>%{customdata[0]}<br><br>' +
                '<b>AI Rating:</b> %{customdata[1]:.2f}<br>' +
                '<b>Actual SC:</b> %{customdata[2]}<br>' +
                '<b>Education:</b> %{customdata[3]}<br><br>' +
                '<b>Top SHAP Features:</b><br>%{customdata[4]}<br>' +
                '<extra></extra>'
            ),
            hoverlabel=dict(
                bgcolor="rgba(0, 0, 0, 0.6)",
                bordercolor="rgba(255, 255, 255, 0.3)",
                font=dict(size=12, color="white"),
                align="left"
            )
        )
        
        fig.add_trace(scatter)
        
        # Update layout with black background
        fig.update_layout(
            title=dict(text=title, font=dict(color='white')),
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis=dict(
                    backgroundcolor="black",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    title=dict(font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                yaxis=dict(
                    backgroundcolor="black",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    title=dict(font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                zaxis=dict(
                    backgroundcolor="black",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    title=dict(font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                bgcolor="black"
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        # Add custom JS for click to expand
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Reset View',
                    'method': 'relayout',
                    'args': ['scene.camera', dict(eye=dict(x=1.5, y=1.5, z=1.5))]
                }]
            }]
        )
        
        # Add custom JavaScript for click to view full essay and hover positioning
        custom_js = """
        <style>
        /* Custom hover tooltip styling */
        .hoverlayer .hovertext {
            opacity: 1 !important;
        }
        .hoverlayer .hovertext rect {
            opacity: 0.6 !important;
        }
        .hoverlayer .hovertext text {
            font-size: 12px !important;
        }
        </style>
        <script>
        // Position hover tooltip at top right of cursor
        document.addEventListener('DOMContentLoaded', function() {
            var plot = document.querySelector('.plotly');
            if (plot) {
                plot.on('plotly_hover', function(data) {
                    setTimeout(function() {
                        var hovertext = document.querySelector('.hoverlayer .hovertext');
                        if (hovertext) {
                            var transform = hovertext.getAttribute('transform');
                            if (transform) {
                                var match = transform.match(/translate\(([^,]+),([^)]+)\)/);
                                if (match) {
                                    var x = parseFloat(match[1]) + 20; // 20px to the right
                                    var y = parseFloat(match[2]) - 100; // 100px up
                                    hovertext.setAttribute('transform', 'translate(' + x + ',' + y + ')');
                                }
                            }
                        }
                    }, 10);
                });
            }
        });
        </script>
        <script>
        // Create essay viewer div
        var essayViewer = document.createElement('div');
        essayViewer.id = 'essay-viewer';
        essayViewer.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-height: 300px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #666;
            border-radius: 8px;
            padding: 15px;
            display: none;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        `;
        essayViewer.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="color: white; margin: 0; font-size: 16px;">Full Essay</h3>
                <button onclick="document.getElementById('essay-viewer').style.display='none'" 
                        style="background: none; border: none; color: white; font-size: 20px; cursor: pointer;">×</button>
            </div>
            <div id="essay-content" style="color: #ccc; font-size: 14px; line-height: 1.5;"></div>
            <div id="essay-info" style="color: #999; font-size: 12px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;"></div>
        `;
        document.body.appendChild(essayViewer);
        
        // Add click handler to plot
        document.querySelector('.plotly').on('plotly_click', function(data) {
            var point = data.points[0];
            var essayFull = point.customdata[5];
            var aiRating = point.customdata[1];
            var actualSC = point.customdata[2];
            var education = point.customdata[3];
            
            document.getElementById('essay-content').textContent = essayFull;
            document.getElementById('essay-info').innerHTML = 
                '<strong>AI Rating:</strong> ' + parseFloat(aiRating).toFixed(2) + ' | ' +
                '<strong>Actual SC:</strong> ' + actualSC + ' | ' +
                '<strong>Education:</strong> ' + education;
            document.getElementById('essay-viewer').style.display = 'block';
        });
        
        // Add help text and filters
        var controlPanel = document.createElement('div');
        controlPanel.style.cssText = 'position: absolute; top: 10px; right: 10px; background: rgba(0, 0, 0, 0.7); padding: 15px; border-radius: 8px; color: white; font-size: 12px;';
        controlPanel.innerHTML = `
            <div style="margin-bottom: 10px;">
                <strong>Controls:</strong><br>
                • Hover: Preview essay<br>
                • Click: View full essay
            </div>
            <div style="border-top: 1px solid #444; padding-top: 10px;">
                <strong>Filter by Social Class:</strong><br>
                <label style="display: block; margin: 5px 0;"><input type="checkbox" id="sc1" checked> Class 1 (Lowest)</label>
                <label style="display: block; margin: 5px 0;"><input type="checkbox" id="sc2" checked> Class 2</label>
                <label style="display: block; margin: 5px 0;"><input type="checkbox" id="sc3" checked> Class 3</label>
                <label style="display: block; margin: 5px 0;"><input type="checkbox" id="sc4" checked> Class 4</label>
                <label style="display: block; margin: 5px 0;"><input type="checkbox" id="sc5" checked> Class 5 (Highest)</label>
            </div>
        `;
        document.querySelector('.plotly').style.position = 'relative';
        document.querySelector('.plotly').appendChild(controlPanel);
        
        // Add filter functionality
        function updateFilters() {
            var plot = document.querySelector('.plotly');
            var checkboxes = {
                1: document.getElementById('sc1').checked,
                2: document.getElementById('sc2').checked,
                3: document.getElementById('sc3').checked,
                4: document.getElementById('sc4').checked,
                5: document.getElementById('sc5').checked
            };
            
            // Get the original data
            var originalData = plot.data[0];
            var x = [], y = [], z = [];
            var customdata = [];
            var colors = [];
            
            for (var i = 0; i < originalData.x.length; i++) {
                var sc = parseInt(originalData.customdata[i][2]);
                if (checkboxes[sc]) {
                    x.push(originalData.x[i]);
                    y.push(originalData.y[i]);
                    z.push(originalData.z[i]);
                    customdata.push(originalData.customdata[i]);
                    colors.push(originalData.marker.color[i]);
                }
            }
            
            Plotly.restyle(plot, {
                'x': [x],
                'y': [y], 
                'z': [z],
                'customdata': [customdata],
                'marker.color': [colors]
            }, [0]);
        }
        
        // Attach event listeners
        ['sc1', 'sc2', 'sc3', 'sc4', 'sc5'].forEach(function(id) {
            document.getElementById(id).addEventListener('change', updateFilters);
        });
        </script>
        """
        
        # Save with custom JS
        output_path = VIZ_DIR / filename
        with open(output_path, 'w') as f:
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'displaylogo': False}
            )
            # Insert custom JS before closing body tag
            html_content = html_content.replace('</body>', custom_js + '</body>')
            f.write(html_content)
        
        print(f"Saved: {output_path}")

def create_shap_visualizations(dml_results, X_pca):
    """Create comprehensive SHAP visualizations"""
    print("Creating SHAP visualizations...")
    
    shap_dir = VIZ_DIR / "shap_analysis"
    shap_dir.mkdir(exist_ok=True)
    
    # Focus on XGBoost model (best performer)
    xgb_model = dml_results['AI_ratings']['XGBoost']['model']
    X_res = dml_results['AI_ratings']['XGBoost']['X_res']
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_res)
    
    # 1. Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_res, feature_names=[f'PC{i+1}' for i in range(200)], 
                      show=False, max_display=20)
    plt.title('SHAP Summary Plot - Top 20 PCs')
    plt.tight_layout()
    plt.savefig(shap_dir / 'shap_summary_openai.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_res, feature_names=[f'PC{i+1}' for i in range(200)], 
                      plot_type='bar', show=False, max_display=20)
    plt.title('SHAP Feature Importance - Top 20 PCs')
    plt.tight_layout()
    plt.savefig(shap_dir / 'shap_importance_openai.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall plot for a few examples
    for i in [0, 100, 500]:
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[i], 
                                            base_values=explainer.expected_value,
                                            feature_names=[f'PC{j+1}' for j in range(200)]),
                           max_display=15, show=False)
        plt.title(f'SHAP Waterfall Plot - Essay {i}')
        plt.tight_layout()
        plt.savefig(shap_dir / f'shap_waterfall_essay_{i}_openai.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"SHAP visualizations saved to {shap_dir}")

def main():
    """Main analysis pipeline"""
    print("="*60)
    print("OpenAI Text-Embedding-3-Large Analysis Pipeline")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Get embeddings
    embeddings = compute_embeddings(df)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # PCA to 200 components
    X_pca, pca, scaler = compute_pca(embeddings, n_components=200)
    print(f"PCA features shape: {X_pca.shape}")
    
    # Run DML analysis
    dml_results = run_dml_analysis(df, X_pca)
    
    # Print results summary
    print("\n" + "="*60)
    print("DML Results Summary (OpenAI Embeddings)")
    print("="*60)
    for outcome in ['AI_ratings', 'actual_SC']:
        print(f"\n{outcome}:")
        for model in ['Linear', 'Lasso', 'RF', 'XGBoost']:
            r2 = dml_results[outcome][model]['r2']
            print(f"  {model:8s}: R² = {r2:.3f}")
    
    # Compute UMAP
    umap_3d = compute_umap_3d(X_pca)
    
    # Create interactive visualizations
    create_interactive_umap(df, umap_3d, X_pca, dml_results)
    
    # Import matplotlib for SHAP
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create SHAP visualizations
    create_shap_visualizations(dml_results, X_pca)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Outputs saved to: {VIZ_DIR}")
    print("\nKey files created:")
    print("- umap_actual_sc_openai.html")
    print("- umap_ai_rating_openai.html")
    print("- shap_analysis/*.png")

if __name__ == "__main__":
    main()