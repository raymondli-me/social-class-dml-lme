#!/usr/bin/env python3
"""
Create interactive 3D UMAP visualizations with LIME explanations
Two versions: colored by actual SC (discrete) and AI ratings (continuous)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class InteractiveUMAPVisualizer:
    def __init__(self, checkpoint_dir="dml_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.load_data()
        
    def load_data(self):
        """Load all necessary data"""
        print("Loading data...")
        
        # Load essays and ratings
        essays = pd.read_csv('data/asc_9513_essays.csv')
        ai_ratings = pd.read_csv('asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv')
        sc_data = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
        
        # Pivot AI ratings
        ai_pivot = ai_ratings.pivot_table(
            index='essay_id', 
            columns='prompt_name', 
            values='rating',
            aggfunc='first'
        ).reset_index()
        
        # Merge all data
        self.data = essays.merge(ai_pivot, left_on='TID', right_on='essay_id', how='inner')
        self.data = self.data.merge(sc_data, on='TID', how='inner')
        self.data['ai_average'] = self.data[['ladder_standard_improved', 'human_macarthur_ladder_improved']].mean(axis=1)
        
        # Load PCA features
        print("Loading PCA features...")
        with open(self.checkpoint_dir / "pca_features.pkl", 'rb') as f:
            pca_data = pickle.load(f)
            self.X_pca = pca_data['X']
            self.pca = pca_data['pca']
            
        # Truncate essays for display
        self.data['essay_preview'] = self.data['original'].str[:150] + '...'
        self.data['essay_full'] = self.data['original']
        
        print(f"Loaded {len(self.data)} essays")
        
    def compute_umap(self, n_neighbors=15, min_dist=0.1, n_components=3):
        """Compute 3D UMAP embedding"""
        print("Computing UMAP embedding...")
        
        umap_file = self.checkpoint_dir / f"umap_3d_n{n_neighbors}_d{min_dist}.npy"
        
        if umap_file.exists():
            print("Loading existing UMAP...")
            self.umap_embedding = np.load(umap_file)
        else:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=42,
                verbose=True
            )
            self.umap_embedding = reducer.fit_transform(self.X_pca)
            np.save(umap_file, self.umap_embedding)
            
        print(f"UMAP shape: {self.umap_embedding.shape}")
        
    def compute_lime_explanations(self, sample_size=None):
        """Compute LIME explanations for all points"""
        if sample_size is None:
            sample_size = len(self.data)
        print(f"Computing LIME explanations for {sample_size} samples...")
        
        lime_file = self.checkpoint_dir / f"lime_explanations_full_{len(self.data)}.pkl"
        
        if lime_file.exists():
            print("Loading existing LIME explanations...")
            with open(lime_file, 'rb') as f:
                self.lime_explanations = pickle.load(f)
        else:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_pca,
                feature_names=[f'PC{i+1}' for i in range(self.X_pca.shape[1])],
                class_names=['social_class'],
                mode='regression'
            )
            
            # Use all indices
            sample_indices = range(len(self.data))
            
            # Train XGBoost model for LIME to explain (best performer from DML)
            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(self.X_pca, self.data['ai_average'])
            
            # Get explanations
            self.lime_explanations = {}
            
            # Process in batches to show progress
            batch_size = 100
            for i in tqdm(range(0, len(self.data), batch_size)):
                batch_indices = list(range(i, min(i + batch_size, len(self.data))))
                for idx in batch_indices:
                    exp = explainer.explain_instance(
                        self.X_pca[idx], 
                        model.predict, 
                        num_features=3  # Top 3 features
                    )
                    
                    # Extract top 3 PC contributions
                    top_features = []
                    for feat, weight in exp.as_list()[:3]:
                        pc_num = int(feat.split('PC')[1].split(' ')[0])
                        top_features.append((pc_num, weight))
                    
                    self.lime_explanations[idx] = top_features
            
            # Save explanations
            with open(lime_file, 'wb') as f:
                pickle.dump(self.lime_explanations, f)
                
        print(f"Computed explanations for {len(self.lime_explanations)} points")
        
    def create_interactive_plot(self, color_by='sc11', title_suffix='Actual Social Class'):
        """Create interactive 3D UMAP plot"""
        print(f"Creating interactive plot colored by {color_by}...")
        
        # Prepare data for plotting
        plot_data = self.data.copy()
        plot_data['x'] = self.umap_embedding[:, 0]
        plot_data['y'] = self.umap_embedding[:, 1]
        plot_data['z'] = self.umap_embedding[:, 2]
        
        # Add LIME explanations
        plot_data['lime_text'] = ''
        for idx in self.lime_explanations:
            if idx < len(plot_data):
                lime_parts = []
                for pc_num, weight in self.lime_explanations[idx]:
                    lime_parts.append(f"PC{pc_num}: {weight:+.2f}")
                plot_data.loc[idx, 'lime_text'] = " | ".join(lime_parts)
        
        # Create hover text
        plot_data['hover_text'] = plot_data.apply(
            lambda row: (
                f"ID: {row['TID']}<br>"
                f"Actual SC: {row['sc11']}<br>"
                f"AI Score: {row['ai_average']:.1f}<br>"
                f"Essay: {row['essay_preview']}<br>"
                f"LIME: {row['lime_text'] if row['lime_text'] else 'N/A'}"
            ), axis=1
        )
        
        # Create figure
        fig = go.Figure()
        
        if color_by == 'sc11':
            # Discrete coloring for actual SC
            colors = px.colors.qualitative.Set1[:5]  # 5 distinct colors
            for i, sc_value in enumerate(sorted(plot_data['sc11'].unique())):
                mask = plot_data['sc11'] == sc_value
                fig.add_trace(go.Scatter3d(
                    x=plot_data.loc[mask, 'x'],
                    y=plot_data.loc[mask, 'y'],
                    z=plot_data.loc[mask, 'z'],
                    mode='markers',
                    name=f'SC {int(sc_value)}',
                    marker=dict(
                        size=5,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=0.5, color='white')
                    ),
                    text=plot_data.loc[mask, 'hover_text'],
                    hoverinfo='text',
                    customdata=np.column_stack((
                        plot_data.loc[mask, 'TID'],
                        plot_data.loc[mask, 'essay_full'],
                        plot_data.loc[mask, 'sc11'],
                        plot_data.loc[mask, 'ai_average'],
                        plot_data.loc[mask, 'lime_text']
                    ))
                ))
        else:
            # Continuous coloring for AI ratings
            fig.add_trace(go.Scatter3d(
                x=plot_data['x'],
                y=plot_data['y'],
                z=plot_data['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=plot_data[color_by],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="AI Rating"),
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=plot_data['hover_text'],
                hoverinfo='text',
                customdata=np.column_stack((
                    plot_data['TID'],
                    plot_data['essay_full'],
                    plot_data['sc11'],
                    plot_data['ai_average'],
                    plot_data['lime_text']
                ))
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D UMAP Visualization - {title_suffix}',
            scene=dict(
                xaxis=dict(title='UMAP 1', showgrid=False, zeroline=False),
                yaxis=dict(title='UMAP 2', showgrid=False, zeroline=False),
                zaxis=dict(title='UMAP 3', showgrid=False, zeroline=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            width=1200,
            height=800,
            showlegend=True,
            hovermode='closest',
            margin=dict(r=200, b=200)  # Space for text display
        )
        
        # Add custom JavaScript for click handling
        fig.add_annotation(
            text="Click on a point to see full essay",
            xref="paper", yref="paper",
            x=0.99, y=0.01,
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="right"
        )
        
        # Add JavaScript for interactivity
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        # Insert custom JavaScript for click handling
        custom_js = """
        <div id="essay-display" style="position: fixed; bottom: 20px; right: 20px; 
             width: 400px; max-height: 300px; overflow-y: auto; 
             background: white; border: 2px solid #ccc; padding: 15px; 
             border-radius: 10px; display: none; font-size: 14px;">
            <button onclick="this.parentElement.style.display='none'" 
                    style="float: right; border: none; background: none; font-size: 20px;">×</button>
            <div id="essay-content"></div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plot = document.getElementsByClassName('plotly-graph-div')[0];
            
            plot.on('plotly_click', function(data) {
                var point = data.points[0];
                var customdata = point.customdata;
                
                var content = '<h4>Essay ID: ' + customdata[0] + '</h4>' +
                             '<p><b>Actual SC:</b> ' + customdata[2] + 
                             ' | <b>AI Score:</b> ' + parseFloat(customdata[3]).toFixed(1) + '</p>' +
                             '<p><b>LIME Features:</b> ' + (customdata[4] || 'N/A') + '</p>' +
                             '<hr>' +
                             '<p style="max-height: 150px; overflow-y: auto;">' + 
                             customdata[1] + '</p>';
                
                document.getElementById('essay-content').innerHTML = content;
                document.getElementById('essay-display').style.display = 'block';
            });
        });
        </script>
        """
        
        # Insert custom JS before closing body tag
        html_str = html_str.replace('</body>', custom_js + '</body>')
        
        return html_str
        
    def create_both_visualizations(self):
        """Create both visualization versions"""
        # Compute UMAP
        self.compute_umap(n_neighbors=15, min_dist=0.1)
        
        # Compute LIME explanations for all points
        self.compute_lime_explanations()
        
        # Create visualizations
        print("\nCreating visualization 1: Colored by actual social class...")
        html1 = self.create_interactive_plot(
            color_by='sc11', 
            title_suffix='Colored by Actual Social Class (1-5)'
        )
        
        print("\nCreating visualization 2: Colored by AI ratings...")
        html2 = self.create_interactive_plot(
            color_by='ai_average', 
            title_suffix='Colored by AI Ratings (1-10)'
        )
        
        # Save HTML files
        with open('umap_actual_social_class.html', 'w') as f:
            f.write(html1)
        print("Saved: umap_actual_social_class.html")
        
        with open('umap_ai_ratings.html', 'w') as f:
            f.write(html2)
        print("Saved: umap_ai_ratings.html")
        
        print("\nVisualization complete! Open the HTML files in your browser.")
        print("Features:")
        print("- Rotate: Click and drag")
        print("- Zoom: Scroll or pinch")
        print("- Hover: See preview and scores")
        print("- Click: View full essay in bottom right")


def main():
    print("Creating Interactive 3D UMAP Visualizations")
    print("=" * 50)
    
    visualizer = InteractiveUMAPVisualizer()
    visualizer.create_both_visualizations()
    

if __name__ == "__main__":
    main()