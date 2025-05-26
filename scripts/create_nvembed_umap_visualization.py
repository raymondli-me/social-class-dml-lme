#!/usr/bin/env python3
"""
Create interactive 3D UMAP visualizations using NV-Embed embeddings
Colored by actual social class and AI ratings (human MacArthur only)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import umap
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NV-EMBED 3D UMAP VISUALIZATION")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = BASE_DIR / "nvembed_visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Load NV-Embed embeddings
print("\nLoading NV-Embed embeddings...")
embeddings_full = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"✓ Embeddings shape: {embeddings_full.shape}")

# Load data files
print("\nLoading data files...")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
sc_labels = pd.read_csv("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
ai_ratings_df = pd.read_csv(BASE_DIR / "asc_analysis_2prompts" / "run_20250524_162055" / "all_results_9513x2_20250524_174149.csv")

# Filter for human MacArthur ratings only
print("\nFiltering for human MacArthur ratings...")
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
ai_ratings_avg = human_mac_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_ratings_avg.columns = ['TID', 'ai_rating']

# Merge all data
df = essays.merge(sc_labels, on='TID', how='inner')
df = df.merge(ai_ratings_avg, on='TID', how='inner')

# Align with embeddings
df = df[df['TID'].isin(essay_ids)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

print(f"\n✓ Final dataset: {len(df)} essays")

# Get aligned embeddings
X_full = embeddings_full[df['essay_idx'].values]

# Apply PCA
print("\nApplying PCA (4096 → 200 dimensions)...")
pca = PCA(n_components=200, random_state=42)
X_pca = pca.fit_transform(X_full)
variance_explained = pca.explained_variance_ratio_.sum()
print(f"✓ Variance explained: {variance_explained:.1%}")

# Standardize PCA features
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca)

# Compute UMAP
print("\nComputing 3D UMAP embedding...")
umap_file = NVEMBED_DIR / "umap_3d_nvembed.npy"

if umap_file.exists():
    print("Loading existing UMAP coordinates...")
    umap_3d = np.load(umap_file)
else:
    print("Computing new UMAP coordinates...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    umap_3d = reducer.fit_transform(X_pca_scaled)
    np.save(umap_file, umap_3d)

print(f"✓ UMAP shape: {umap_3d.shape}")

# Add UMAP coordinates to dataframe
df['umap_x'] = umap_3d[:, 0]
df['umap_y'] = umap_3d[:, 1]
df['umap_z'] = umap_3d[:, 2]

# Prepare essay previews
df['essay_preview'] = df['original'].str[:200] + '...'
df['essay_length'] = df['original'].str.len()

# Define color schemes
sc_colors = {
    1: '#d62728',  # Red (Lower class)
    2: '#ff7f0e',  # Orange
    3: '#2ca02c',  # Green (Middle class)
    4: '#1f77b4',  # Blue
    5: '#9467bd'   # Purple (Upper class)
}

sc_labels = {
    1: 'Lower class',
    2: 'Working class', 
    3: 'Middle class',
    4: 'Upper-middle class',
    5: 'Upper class'
}

# Function to create visualization
def create_umap_viz(color_by='sc11', title_suffix=''):
    """Create interactive 3D UMAP visualization"""
    
    if color_by == 'sc11':
        # Discrete coloring by social class
        df['color_label'] = df['sc11'].map(sc_labels)
        
        fig = go.Figure()
        
        # Add trace for each social class
        for sc in sorted(df['sc11'].unique()):
            mask = df['sc11'] == sc
            subset = df[mask]
            
            fig.add_trace(go.Scatter3d(
                x=subset['umap_x'],
                y=subset['umap_y'],
                z=subset['umap_z'],
                mode='markers',
                name=sc_labels[sc],
                marker=dict(
                    size=4,
                    color=sc_colors[sc],
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=subset.apply(lambda r: f"<b>Essay ID:</b> {r['TID']}<br>"
                                          f"<b>Social Class:</b> {sc_labels[r['sc11']]}<br>"
                                          f"<b>AI Rating:</b> {r['ai_rating']:.1f}<br>"
                                          f"<b>Length:</b> {r['essay_length']} chars<br>"
                                          f"<b>Preview:</b><br>{r['essay_preview']}", axis=1),
                hovertemplate='%{text}<extra></extra>'
            ))
        
        title = f"NV-Embed UMAP: Colored by Actual Social Class{title_suffix}"
        
    else:  # color by AI rating
        # Continuous coloring by AI rating
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df['umap_x'],
                y=df['umap_y'],
                z=df['umap_z'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df['ai_rating'],
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=0.5, color='white'),
                    colorbar=dict(
                        title="AI Rating<br>(Human MacArthur)",
                        thickness=20,
                        len=0.7
                    )
                ),
                text=df.apply(lambda r: f"<b>Essay ID:</b> {r['TID']}<br>"
                                       f"<b>Social Class:</b> {sc_labels[r['sc11']]}<br>"
                                       f"<b>AI Rating:</b> {r['ai_rating']:.1f}<br>"
                                       f"<b>Length:</b> {r['essay_length']} chars<br>"
                                       f"<b>Preview:</b><br>{r['essay_preview']}", axis=1),
                hovertemplate='%{text}<extra></extra>'
            )
        ])
        
        title = f"NV-Embed UMAP: Colored by AI Rating (Human MacArthur){title_suffix}"
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor='white',
            xaxis=dict(gridcolor='lightgray', showbackground=True, backgroundcolor='white'),
            yaxis=dict(gridcolor='lightgray', showbackground=True, backgroundcolor='white'),
            zaxis=dict(gridcolor='lightgray', showbackground=True, backgroundcolor='white'),
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True if color_by == 'sc11' else False,
        legend=dict(
            x=1.02,
            y=0.5,
            yanchor='middle',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

# Create both visualizations
print("\nCreating visualizations...")

# 1. Colored by actual social class
fig_sc = create_umap_viz(color_by='sc11')
output_sc = VIZ_DIR / f"nvembed_umap_actual_sc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
fig_sc.write_html(str(output_sc))
print(f"✓ Saved: {output_sc}")

# 2. Colored by AI rating
fig_ai = create_umap_viz(color_by='ai_rating')
output_ai = VIZ_DIR / f"nvembed_umap_ai_rating_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
fig_ai.write_html(str(output_ai))
print(f"✓ Saved: {output_ai}")

# Compute statistics
print("\n" + "="*80)
print("VISUALIZATION STATISTICS")
print("="*80)

# Class distribution
print("\nSocial Class Distribution:")
for sc in sorted(df['sc11'].unique()):
    count = (df['sc11'] == sc).sum()
    pct = count / len(df) * 100
    print(f"  {sc_labels[sc]}: {count:,} ({pct:.1f}%)")

# AI rating statistics by class
print("\nAI Ratings by Social Class:")
for sc in sorted(df['sc11'].unique()):
    ratings = df[df['sc11'] == sc]['ai_rating']
    print(f"  {sc_labels[sc]}: mean={ratings.mean():.2f}, std={ratings.std():.2f}")

# Correlation
corr = df[['sc11', 'ai_rating']].corr().iloc[0, 1]
print(f"\nCorrelation (SC vs AI): {corr:.3f}")

# Save summary
summary = {
    'n_essays': len(df),
    'embeddings': 'NV-Embed-v2',
    'pca_dims': 200,
    'pca_variance': variance_explained,
    'umap_params': {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'n_components': 3,
        'metric': 'cosine'
    },
    'ai_rating_type': 'human_macarthur_only',
    'correlation_sc_ai': corr,
    'files': {
        'actual_sc': str(output_sc),
        'ai_rating': str(output_ai)
    }
}

summary_file = VIZ_DIR / f"nvembed_umap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
with open(summary_file, 'wb') as f:
    pickle.dump(summary, f)

print(f"\n✓ Summary saved: {summary_file}")
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nOpen these files in your browser:")
print(f"  1. {output_sc}")
print(f"  2. {output_ai}")
print("\nFeatures:")
print("  - Interactive 3D rotation (click and drag)")
print("  - Zoom (scroll wheel)")
print("  - Hover for essay details")
print("  - Toggle classes in legend (for SC visualization)")
print("="*80)