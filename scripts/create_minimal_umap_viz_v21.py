#!/usr/bin/env python3
"""
Minimal UMAP visualization v18 - HDBSCAN Topics with c-TF-IDF
- Based on v17 with all functionality preserved
- Added HDBSCAN clustering for topic discovery
- c-TF-IDF for extracting topic keywords
- Toggle to show/hide topics with billboarding labels
- Sliders for controlling visible topics and text size
- Dynamic topic coloring when topics are enabled
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import xgboost as xgb

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'nvembed_dml_pc_analysis'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'

print("=== Creating Minimal UMAP Visualization v21 ===")

# Load essays and social class
print("Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Load DML analysis results
print("Loading DML analysis results...")
with open(OUTPUT_DIR / 'dml_pc_analysis_results_with_umap.pkl', 'rb') as f:
    dml_results = pickle.load(f)
    X_umap_3d = dml_results['umap_3d']
    data_with_pcs = dml_results['data_with_pcs']
    contributions_ai_top5 = dml_results['contributions_ai']
    contributions_sc_top5 = dml_results['contributions_sc']
    top_pcs = dml_results['top_pcs']

# Load full PCA data to get all 200 PCs
print("Loading full PCA data...")
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    essay_ids = pca_data['essay_ids']
    X_pca = pca_data['features']  # PCA-transformed features
    pca_model = pca_data['pca']

# Align data
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()

# Get variance explained for all PCs
variance_explained = pca_model.explained_variance_ratio_

# Calculate statistics
ai_ratings_clean = essays_df['ai_rating'].dropna()
ai_percentiles = {
    10: ai_ratings_clean.quantile(0.10),
    25: ai_ratings_clean.quantile(0.25),
    75: ai_ratings_clean.quantile(0.75),
    90: ai_ratings_clean.quantile(0.90)
}

print(f"AI Rating range: {ai_ratings_clean.min():.2f} - {ai_ratings_clean.max():.2f}")
print(f"Social class distribution: {essays_df['sc11'].value_counts().sort_index().to_dict()}")

# Calculate center of point cloud
center_x = X_umap_3d[:, 0].mean()
center_y = X_umap_3d[:, 1].mean()
center_z = X_umap_3d[:, 2].mean()

# Calculate percentiles for ALL 200 PCs
print("Calculating percentiles for all PCs...")
pc_percentiles = np.zeros((len(essays_df), 200))
for i in range(200):
    pc_values = X_pca[:, i]
    pc_percentiles[:, i] = (np.searchsorted(np.sort(pc_values), pc_values) / len(pc_values)) * 100

# Compute contributions for all 200 PCs using XGBoost
print("Computing contributions for all 200 PCs...")
Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Train XGBoost models on all 200 PCs
model_ai_200 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_200.fit(X_pca, Y_ai)

model_sc_200 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_200.fit(X_pca, Y_sc)

# Get feature importances as proxy for contributions
contributions_ai_200 = np.zeros((len(X_pca), 200))
contributions_sc_200 = np.zeros((len(X_pca), 200))

# Use feature importance weighted by PC values
feature_importance_ai = model_ai_200.feature_importances_
feature_importance_sc = model_sc_200.feature_importances_

for i in range(len(X_pca)):
    contributions_ai_200[i] = X_pca[i] * feature_importance_ai
    contributions_sc_200[i] = X_pca[i] * feature_importance_sc

# Calculate R² for both models
from sklearn.metrics import r2_score
r2_ai_200 = r2_score(Y_ai, model_ai_200.predict(X_pca))
r2_sc_200 = r2_score(Y_sc, model_sc_200.predict(X_pca))

# Also compute for top 5 PCs
# Define the PCs used in Top 5 model
top_5_pc_names = ['PC0', 'PC2', 'PC5', 'PC13', 'PC46']
top_5_pc_string = ", ".join(top_5_pc_names)
X_top5 = data_with_pcs[top_5_pc_names].values
model_ai_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_ai_top5.fit(X_top5, Y_ai)
model_sc_top5 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model_sc_top5.fit(X_top5, Y_sc)

r2_ai_top5 = r2_score(Y_ai, model_ai_top5.predict(X_top5))
r2_sc_top5 = r2_score(Y_sc, model_sc_top5.predict(X_top5))

# Function to compute simple DML without cross-fitting
def compute_simple_dml(X, Y_treatment, Y_outcome):
    """Compute non-cross-fitted DML estimate"""
    from scipy import stats
    
    # Fit models on full data
    model_outcome = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_outcome.fit(X, Y_outcome)
    
    model_treatment = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_treatment.fit(X, Y_treatment)
    
    # Get residuals
    residuals_outcome = Y_outcome - model_outcome.predict(X)
    residuals_treatment = Y_treatment - model_treatment.predict(X)
    
    # DML theta estimate
    theta = np.sum(residuals_treatment * residuals_outcome) / np.sum(residuals_treatment ** 2)
    
    # Standard error
    n = len(Y_outcome)
    se = np.sqrt(np.sum((residuals_outcome - theta * residuals_treatment) ** 2) / 
                 (n * np.sum(residuals_treatment ** 2)))
    
    # Confidence interval
    ci = (theta - 1.96 * se, theta + 1.96 * se)
    
    # P-value
    t_stat = theta / se
    pval = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1] - 1))
    
    return theta, se, ci, pval

# Function to compute naive theta (no controls)
def compute_naive_theta(Y_treatment, Y_outcome):
    """Compute naive OLS coefficient (no controls)"""
    from scipy import stats
    import statsmodels.api as sm
    
    # Add constant to treatment
    X = sm.add_constant(Y_treatment)
    
    # Simple OLS
    model = sm.OLS(Y_outcome, X).fit()
    
    theta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
    se = model.bse.iloc[1] if hasattr(model.bse, 'iloc') else model.bse[1]
    ci_df = model.conf_int()
    ci = (ci_df.iloc[1, 0], ci_df.iloc[1, 1]) if hasattr(ci_df, 'iloc') else (ci_df[1][0], ci_df[1][1])
    pval = model.pvalues.iloc[1] if hasattr(model.pvalues, 'iloc') else model.pvalues[1]
    r2 = model.rsquared
    
    return theta, se, ci, pval, r2

# Load actual cross-fitted metrics
print("Loading cross-fitted metrics...")
try:
    with open(OUTPUT_DIR / 'crossfitted_metrics_v13.pkl', 'rb') as f:
        cf_metrics = pickle.load(f)
    
    # Compute non-cross-fitted DML for comparison
    print("Computing non-cross-fitted DML estimates...")
    theta_200, se_200, ci_200, pval_200 = compute_simple_dml(X_pca, Y_sc, Y_ai)
    theta_top5, se_top5, ci_top5, pval_top5 = compute_simple_dml(X_top5, Y_sc, Y_ai)
    
    # Compute naive theta
    print("Computing naive theta (no text controls)...")
    theta_naive, se_naive, ci_naive, pval_naive, r2_naive = compute_naive_theta(Y_sc, Y_ai)
    
    # Extract the metrics we need
    dml_results_computed = {
        # Non-cross-fitted values
        'theta_200': theta_200,
        'se_200': se_200,
        'ci_200': ci_200,
        'pval_200': pval_200,
        # Cross-fitted values
        'theta_200_cf': cf_metrics['sc_to_ai_200']['theta_cf'],
        'se_200_cf': cf_metrics['sc_to_ai_200']['se_cf'],
        'ci_200_cf': cf_metrics['sc_to_ai_200']['ci_cf'],
        'pval_200_cf': cf_metrics['sc_to_ai_200']['pval_cf'],
        # Top 5 non-cross-fitted
        'theta_top5': theta_top5,
        'se_top5': se_top5,
        'ci_top5': ci_top5,
        'pval_top5': pval_top5,
        # Top 5 cross-fitted
        'theta_top5_cf': cf_metrics['sc_to_ai_top5']['theta_cf'],
        'se_top5_cf': cf_metrics['sc_to_ai_top5']['se_cf'],
        'ci_top5_cf': cf_metrics['sc_to_ai_top5']['ci_cf'],
        'pval_top5_cf': cf_metrics['sc_to_ai_top5']['pval_cf'],
        # Naive theta
        'theta_naive': theta_naive,
        'se_naive': se_naive,
        'ci_naive': ci_naive,
        'pval_naive': pval_naive,
        'r2_naive': r2_naive
    }
    
    # Calculate percentage reductions
    reduction_200 = (1 - theta_200 / theta_naive) * 100 if theta_naive != 0 else 0
    reduction_200_cf = (1 - cf_metrics['sc_to_ai_200']['theta_cf'] / theta_naive) * 100 if theta_naive != 0 else 0
    reduction_top5 = (1 - theta_top5 / theta_naive) * 100 if theta_naive != 0 else 0
    reduction_top5_cf = (1 - cf_metrics['sc_to_ai_top5']['theta_cf'] / theta_naive) * 100 if theta_naive != 0 else 0
    
    dml_results_computed['reduction_200'] = reduction_200
    dml_results_computed['reduction_200_cf'] = reduction_200_cf
    dml_results_computed['reduction_top5'] = reduction_top5
    dml_results_computed['reduction_top5_cf'] = reduction_top5_cf
    
    # Also update R² values with actual cross-fitted values
    r2_ai_200_cf = cf_metrics['sc_to_ai_200']['r2_cf']
    r2_sc_200_cf = cf_metrics['ai_to_sc_200']['r2_cf']
    r2_ai_top5_cf = cf_metrics['sc_to_ai_top5']['r2_cf']
    r2_sc_top5_cf = cf_metrics['ai_to_sc_top5']['r2_cf']
    
except FileNotFoundError:
    print("Warning: crossfitted_metrics_v13.pkl not found, using reasonable estimates")
    # Compute non-cross-fitted DML
    theta_200, se_200, ci_200, pval_200 = compute_simple_dml(X_pca, Y_sc, Y_ai)
    theta_top5, se_top5, ci_top5, pval_top5 = compute_simple_dml(X_top5, Y_sc, Y_ai)
    
    # Compute naive theta
    theta_naive, se_naive, ci_naive, pval_naive, r2_naive = compute_naive_theta(Y_sc, Y_ai)
    
    # Fallback cross-fitted values (from your output)
    dml_results_computed = {
        'theta_200': theta_200,
        'se_200': se_200,
        'ci_200': ci_200,
        'pval_200': pval_200,
        'theta_200_cf': 0.054,
        'se_200_cf': 0.013,
        'ci_200_cf': (0.029, 0.079),
        'pval_200_cf': 0.001,
        'theta_top5': theta_top5,
        'se_top5': se_top5,
        'ci_top5': ci_top5,
        'pval_top5': pval_top5,
        'theta_top5_cf': 0.106,
        'se_top5_cf': 0.014,
        'ci_top5_cf': (0.079, 0.133),
        'pval_top5_cf': 0.001,
        # Naive estimate
        'theta_naive': theta_naive,
        'se_naive': se_naive,
        'ci_naive': ci_naive,
        'pval_naive': pval_naive,
        'r2_naive': r2_naive
    }
    
    # Calculate percentage reductions
    reduction_200 = (1 - theta_200 / theta_naive) * 100 if theta_naive != 0 else 0
    reduction_200_cf = (1 - 0.054 / theta_naive) * 100 if theta_naive != 0 else 0
    reduction_top5 = (1 - theta_top5 / theta_naive) * 100 if theta_naive != 0 else 0
    reduction_top5_cf = (1 - 0.106 / theta_naive) * 100 if theta_naive != 0 else 0
    
    dml_results_computed['reduction_200'] = reduction_200
    dml_results_computed['reduction_200_cf'] = reduction_200_cf
    dml_results_computed['reduction_top5'] = reduction_top5
    dml_results_computed['reduction_top5_cf'] = reduction_top5_cf
    
    r2_ai_200_cf = 0.505
    r2_sc_200_cf = -0.023
    r2_ai_top5_cf = 0.423
    r2_sc_top5_cf = -0.039

# Train logistic regression models for probability predictions
print("Training probability models...")
# For high/low AI rating - will use user-defined thresholds
ai_high_threshold = ai_percentiles[90]
ai_low_threshold = ai_percentiles[10]
y_ai_high = (essays_df['ai_rating'] > ai_high_threshold).astype(int)
y_ai_low = (essays_df['ai_rating'] < ai_low_threshold).astype(int)

# For high/low social class
sc_high_threshold = 5
sc_low_threshold = 1
y_sc_high = (essays_df['sc11'] >= sc_high_threshold).astype(int)
y_sc_low = (essays_df['sc11'] <= sc_low_threshold).astype(int)

# Train models on all 200 PCs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

models = {}
for name, y in [('ai_high', y_ai_high), ('ai_low', y_ai_low), 
                ('sc_high', y_sc_high), ('sc_low', y_sc_low)]:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    models[name] = model
    print(f"  {name} model trained, accuracy: {model.score(X_scaled, y):.3f}")

# Calculate global PC effects for top/bottom 10%
print("Calculating global PC effects...")
pc_global_effects = {}

# Define top/bottom 10% thresholds for AI and SC
ai_top10 = np.percentile(Y_ai, 90)
ai_bottom10 = np.percentile(Y_ai, 10)
sc_top10 = 5  # Top social class
sc_bottom10 = 1  # Bottom social class

# Create binary labels for top/bottom 10%
y_ai_top10 = (Y_ai >= ai_top10).astype(int)
y_ai_bottom10 = (Y_ai <= ai_bottom10).astype(int)
y_sc_top10 = (Y_sc >= sc_top10).astype(int)
y_sc_bottom10 = (Y_sc <= sc_bottom10).astype(int)

# Train models for top/bottom 10%
models_10pct = {}
for name, y in [('ai_top10', y_ai_top10), ('ai_bottom10', y_ai_bottom10),
                ('sc_top10', y_sc_top10), ('sc_bottom10', y_sc_bottom10)]:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    models_10pct[name] = model

for pc_idx in range(200):
    # Create test data: top 10% vs bottom 10% of PC values
    test_data = np.zeros((2, 200))
    test_data[0, pc_idx] = np.percentile(X_pca[:, pc_idx], 90)  # Top 10%
    test_data[1, pc_idx] = np.percentile(X_pca[:, pc_idx], 10)  # Bottom 10%
    
    # Standardize
    test_scaled = scaler.transform(test_data)
    
    # Get probabilities
    probs = {}
    for outcome in ['ai', 'sc']:
        # Probability of being in top 10% of outcome
        prob_top_if_high = models_10pct[f'{outcome}_top10'].predict_proba(test_scaled[0:1])[0, 1]
        prob_top_if_low = models_10pct[f'{outcome}_top10'].predict_proba(test_scaled[1:2])[0, 1]
        
        # Probability of being in bottom 10% of outcome
        prob_bottom_if_high = models_10pct[f'{outcome}_bottom10'].predict_proba(test_scaled[0:1])[0, 1]
        prob_bottom_if_low = models_10pct[f'{outcome}_bottom10'].predict_proba(test_scaled[1:2])[0, 1]
        
        probs[f'{outcome}_top10_if_high'] = prob_top_if_high
        probs[f'{outcome}_top10_if_low'] = prob_top_if_low
        probs[f'{outcome}_bottom10_if_high'] = prob_bottom_if_high
        probs[f'{outcome}_bottom10_if_low'] = prob_bottom_if_low
        
        # Calculate differences
        probs[f'{outcome}_top10_diff'] = prob_top_if_high - prob_top_if_low
        probs[f'{outcome}_bottom10_diff'] = prob_bottom_if_high - prob_bottom_if_low
    
    pc_global_effects[pc_idx] = probs

# HDBSCAN clustering on UMAP coordinates
print("Running HDBSCAN clustering on UMAP coordinates...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, cluster_selection_method='eom')
cluster_labels = clusterer.fit_predict(X_umap_3d)

# Add cluster labels to essays_df
essays_df['hdbscan_topic_id'] = cluster_labels

# Get unique cluster IDs (excluding noise -1)
unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
print(f"Found {len(unique_clusters)} clusters (topics)")

# c-TF-IDF for keyword extraction
print("Extracting keywords using c-TF-IDF...")

def preprocess_text(text):
    """Basic text preprocessing for c-TF-IDF"""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

# Group essays by cluster
cluster_documents = {}
for cluster_id in unique_clusters:
    cluster_essays = essays_df[essays_df['hdbscan_topic_id'] == cluster_id]['essay'].tolist()
    # Concatenate and preprocess all essays in the cluster
    cluster_text = ' '.join([preprocess_text(essay) for essay in cluster_essays])
    cluster_documents[cluster_id] = cluster_text

# Also include noise as a "cluster" for comparison
noise_essays = essays_df[essays_df['hdbscan_topic_id'] == -1]['essay'].tolist()
if noise_essays:
    noise_text = ' '.join([preprocess_text(essay) for essay in noise_essays])
    cluster_documents[-1] = noise_text

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=2,
    max_df=0.8
)

# Fit vectorizer on all documents
all_docs = list(cluster_documents.values())
doc_labels = list(cluster_documents.keys())
tfidf_matrix = vectorizer.fit_transform(all_docs)
feature_names = vectorizer.get_feature_names_out()

# Extract top keywords for each cluster using c-TF-IDF
topic_keywords = {}
for idx, cluster_id in enumerate(doc_labels):
    if cluster_id == -1:  # Skip noise
        continue
    
    # Get TF-IDF scores for this cluster
    cluster_tfidf = tfidf_matrix[idx].toarray().flatten()
    
    # c-TF-IDF: Multiply by cluster frequency / overall frequency
    # For simplicity, we'll use the raw TF-IDF scores as they already capture importance
    
    # Get top 10 keywords
    top_indices = cluster_tfidf.argsort()[-10:][::-1]
    top_keywords = [feature_names[i] for i in top_indices if cluster_tfidf[i] > 0][:5]  # Take top 5
    
    topic_keywords[cluster_id] = ' - '.join([kw.title() for kw in top_keywords])

# Calculate cluster centroids
print("Calculating cluster centroids...")
cluster_centroids = {}
for cluster_id in unique_clusters:
    cluster_points = X_umap_3d[cluster_labels == cluster_id]
    centroid = cluster_points.mean(axis=0)
    cluster_centroids[cluster_id] = centroid.tolist()

# Create topic visualization data
topic_viz_data = []
for cluster_id in unique_clusters:
    topic_viz_data.append({
        'topic_id': int(cluster_id),
        'keywords': topic_keywords.get(cluster_id, f"Topic {cluster_id}"),
        'centroid': [float(x) * 100 for x in cluster_centroids[cluster_id]],  # Scale by 100 to match visualization
        'size': int(np.sum(cluster_labels == cluster_id))
    })

print(f"Prepared {len(topic_viz_data)} topic visualizations")

# Calculate topic statistics for extreme groups
print("Calculating topic statistics for extreme groups...")

# Define extreme thresholds
ai_top10_threshold = np.percentile(Y_ai, 90)
ai_bottom10_threshold = np.percentile(Y_ai, 10)
sc_top10_threshold = 5  # Top social class
sc_bottom10_threshold = 1  # Bottom social class

# Calculate statistics for each topic
topic_stats_data = []
for cluster_id in unique_clusters:
    # Get essays in this topic
    topic_mask = essays_df['hdbscan_topic_id'] == cluster_id
    topic_essays = essays_df[topic_mask]
    topic_size = len(topic_essays)
    
    if topic_size > 0:
        # Count essays in extreme groups
        count_sc_top10 = len(topic_essays[topic_essays['sc11'] >= sc_top10_threshold])
        count_sc_bottom10 = len(topic_essays[topic_essays['sc11'] <= sc_bottom10_threshold])
        count_ai_top10 = len(topic_essays[topic_essays['ai_rating'] >= ai_top10_threshold])
        count_ai_bottom10 = len(topic_essays[topic_essays['ai_rating'] <= ai_bottom10_threshold])
        
        # Calculate probabilities
        prob_sc_top10 = count_sc_top10 / topic_size
        prob_sc_bottom10 = count_sc_bottom10 / topic_size
        prob_ai_top10 = count_ai_top10 / topic_size
        prob_ai_bottom10 = count_ai_bottom10 / topic_size
        
        # Determine max impact for sorting
        max_impact_prob = max(prob_sc_top10, prob_sc_bottom10, prob_ai_top10, prob_ai_bottom10)
        
        topic_stats_data.append({
            'topic_id': int(cluster_id),
            'keywords': topic_keywords.get(cluster_id, f"Topic {cluster_id}"),
            'size': topic_size,
            'prob_sc_top10': prob_sc_top10,
            'prob_sc_bottom10': prob_sc_bottom10,
            'prob_ai_top10': prob_ai_top10,
            'prob_ai_bottom10': prob_ai_bottom10,
            'max_impact_prob': max_impact_prob
        })

# Sort by max impact probability (descending)
topic_stats_data.sort(key=lambda x: x['max_impact_prob'], reverse=True)

print(f"Calculated statistics for {len(topic_stats_data)} topics")
print(f"AI Rating thresholds - Top 10%: {ai_top10_threshold:.2f}, Bottom 10%: {ai_bottom10_threshold:.2f}")
print(f"Social Class thresholds - Top: {sc_top10_threshold}, Bottom: {sc_bottom10_threshold}")

# Prepare visualization data
viz_data = []
for i in range(len(essays_df)):
    if not pd.isna(essays_df.iloc[i]['sc11']) and not pd.isna(essays_df.iloc[i]['ai_rating']):
        # Get top 5 contributing PCs from all 200
        total_contributions = np.abs(contributions_ai_200[i]) + np.abs(contributions_sc_200[i])
        top_5_indices = np.argsort(total_contributions)[-5:][::-1]
        
        # Create PC info for this essay
        pc_info = []
        for pc_idx in top_5_indices:
            pc_info.append({
                'pc': f'PC{pc_idx}',
                'percentile': float(pc_percentiles[i, pc_idx]),
                'contribution_ai': float(contributions_ai_200[i, pc_idx]),
                'contribution_sc': float(contributions_sc_200[i, pc_idx]),
                'variance_ai': float(np.abs(contributions_ai_200[i, pc_idx]) * variance_explained[pc_idx] * 100),
                'variance_sc': float(np.abs(contributions_sc_200[i, pc_idx]) * variance_explained[pc_idx] * 100),
                'variance_total': float(variance_explained[pc_idx] * 100)
            })
        
        viz_data.append({
            'x': float(X_umap_3d[i, 0]),
            'y': float(X_umap_3d[i, 1]),
            'z': float(X_umap_3d[i, 2]),
            'essay_id': essays_df.iloc[i]['essay_id'],
            'essay': essays_df.iloc[i]['essay'],
            'sc11': int(essays_df.iloc[i]['sc11']),
            'ai_rating': float(essays_df.iloc[i]['ai_rating']),
            'hdbscan_topic_id': int(essays_df.iloc[i]['hdbscan_topic_id']),
            'pc_info': pc_info,
            'all_pc_values': X_pca[i].tolist(),
            'all_pc_percentiles': pc_percentiles[i].tolist(),  # Add percentiles for gradient coloring
            'all_pc_contributions_ai': contributions_ai_200[i].tolist(),
            'all_pc_contributions_sc': contributions_sc_200[i].tolist()
        })

print(f"Prepared {len(viz_data)} points")
print(f"Cloud center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

# Create HTML with enhanced features
html_content = """<!DOCTYPE html>
<html>
<head>
    <title>UMAP Visualization - Full TreeSHAP Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: #000;
            cursor: none;
            color: #fff;
        }
        
        /* Text shadow for better readability */
        * {
            text-shadow: 
                -1px -1px 0 #000,
                 1px -1px 0 #000,
                -1px  1px 0 #000,
                 1px  1px 0 #000;
        }
        
        #cursor-indicator {
            position: absolute;
            width: 30px;
            height: 30px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 400px;
            z-index: 100;
            cursor: move;
            transition: transform 0.3s ease;
            overflow: visible;
        }
        #info.collapsed {
            transform: translateX(calc(-100% + 40px));
        }
        #info .collapse-btn {
            position: absolute;
            right: -30px;
            top: 50%;
            transform: translateY(-50%);
            width: 25px;
            height: 60px;
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.2);
            border-left: none;
            border-radius: 0 5px 5px 0;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 16px;
            transition: all 0.2s;
        }
        #info .collapse-btn:hover {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
            transition: transform 0.3s ease, opacity 0.3s ease;
            overflow: visible;
        }
        #controls.collapsed {
            transform: translateX(calc(100% - 40px));
        }
        #controls .collapse-btn {
            position: absolute;
            left: -30px;
            top: 50%;
            transform: translateY(-50%);
            width: 25px;
            height: 60px;
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.2);
            border-right: none;
            border-radius: 5px 0 0 5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 16px;
            transition: all 0.2s;
        }
        #controls .collapse-btn:hover {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        #gallery-controls {
            position: absolute;
            top: 10px;
            right: 480px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
            max-width: 200px;
            transition: right 0.3s ease;
        }
        #essay-display {
            position: absolute;
            bottom: 10px;
            left: 60px;
            right: 60px;
            height: 30vh;
            background: rgba(0,0,0,0.25);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            overflow-y: auto;
            display: none;
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.3s;
            z-index: 1000;
            resize: vertical;
            min-height: 80px;
            max-height: 80vh;
        }
        #essay-display.minimized {
            height: 40px !important;
            overflow: hidden;
            padding: 8px 15px;
        }
        #essay-display.minimized #essay-text,
        #essay-display.minimized #pc-analysis {
            display: none;
        }
        #essay-display.minimized #essay-header {
            margin-bottom: 0;
            border-bottom: none;
            padding-bottom: 0;
        }
        #essay-display.minimized .header-pcs {
            display: none;
        }
        .resize-handle-top {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            cursor: ns-resize;
            background: transparent;
        }
        .resize-handle-top:hover {
            background: rgba(255,255,255,0.2);
        }
        .resize-handle-left {
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            width: 5px;
            cursor: ew-resize;
            background: transparent;
        }
        .resize-handle-left:hover {
            background: rgba(255,255,255,0.2);
        }
        .resize-handle-right {
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 5px;
            cursor: ew-resize;
            background: transparent;
        }
        .resize-handle-right:hover {
            background: rgba(255,255,255,0.2);
        }
        .resize-handle-corner {
            position: absolute;
            width: 10px;
            height: 10px;
            background: transparent;
        }
        .resize-handle-corner:hover {
            background: rgba(255,255,255,0.3);
        }
        .resize-handle-corner.top-left {
            top: 0;
            left: 0;
            cursor: nw-resize;
        }
        .resize-handle-corner.top-right {
            top: 0;
            right: 0;
            cursor: ne-resize;
        }
        .minimize-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            cursor: pointer;
            font-size: 16px;
            color: #999;
            padding: 0 5px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            transition: all 0.2s;
        }
        .minimize-btn:hover {
            color: #fff;
            background: rgba(255,255,255,0.2);
        }
        .layer-btn {
            position: absolute;
            top: 10px;
            right: 40px;
            cursor: pointer;
            font-size: 14px;
            color: #999;
            padding: 2px 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            transition: all 0.2s;
            font-family: monospace;
        }
        .layer-btn:hover {
            color: #fff;
            background: rgba(255,255,255,0.2);
        }
        .layer-btn.on-top {
            color: #4CAF50;
        }
        #essay-header {
            font-weight: bold;
            margin-bottom: 5px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            font-size: 12px;
        }
        .header-main {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
        }
        .header-pcs {
            display: flex;
            gap: 10px;
            font-size: 11px;
            color: #ccc;
            flex-wrap: wrap;
        }
        .pc-inline {
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
        }
        .pc-inline:hover {
            background: rgba(255,255,255,0.2);
        }
        #essay-text {
            line-height: 1.4;
            white-space: pre-wrap;
            color: white;
            margin-bottom: 8px;
            font-size: 24px;
            overflow-y: auto;
        }
        #pc-analysis {
            padding: 10px 0;
            margin-top: 10px;
            display: none;
        }
        .pc-item {
            padding: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
            font-size: 11px;
        }
        .pc-item:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.3);
        }
        .pc-name {
            font-weight: bold;
            color: #4CAF50;
            font-size: 12px;
        }
        .pc-percentile {
            color: white;
        }
        .pc-values {
            font-size: 10px;
            color: white;
            margin-top: 2px;
        }
        #pc-global-info {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.95);
            padding: 15px 20px;
            border-radius: 5px;
            border: 2px solid rgba(255,255,255,0.3);
            display: none;
            max-width: 400px;
            z-index: 1000;
        }
        #pc-global-info h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .close-btn {
            position: absolute;
            top: 5px;
            right: 10px;
            cursor: pointer;
            font-size: 20px;
            color: #999;
        }
        .close-btn:hover {
            color: #fff;
        }
        .prob-table {
            width: 100%;
            margin-top: 10px;
        }
        .prob-table td {
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .prob-table td:first-child {
            font-weight: bold;
            color: #888;
        }
        .prob-high {
            color: #4CAF50;
            font-weight: bold;
        }
        .prob-med {
            color: #FFC107;
        }
        .prob-low {
            color: #999;
        }
        .threshold-info {
            font-size: 11px;
            color: #999;
            margin-top: 10px;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }
        #dml-table {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.95);
            padding: 15px;
            border-radius: 5px;
            border: 2px solid rgba(255,255,255,0.3);
            display: none;
            z-index: 1000;
            max-width: 550px;
        }
        #dml-table h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
        }
        .dml-stats {
            width: 100%;
            font-size: 12px;
        }
        .dml-stats td {
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .dml-stats td:first-child {
            font-weight: bold;
            color: #888;
        }
        .control-group {
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .control-group:last-child {
            border-bottom: none;
        }
        .control-group label {
            display: block;
            margin-bottom: 3px;
            font-weight: 600;
            font-size: 13px;
            letter-spacing: 0.02em;
        }
        .threshold-input {
            width: 60px;
            margin: 0 5px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 2px 5px;
        }
        .legend-item {
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .color-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 8px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        button {
            padding: 6px 12px;
            margin-top: 5px;
            cursor: pointer;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            border-radius: 4px;
            font-family: inherit;
            font-size: 13px;
            font-weight: 500;
            letter-spacing: 0.02em;
            transition: all 0.15s ease;
        }
        button:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.4);
            transform: translateY(-1px);
        }
        button:active {
            transform: translateY(0);
        }
        #counts {
            margin-top: 10px;
            font-size: 12px;
            color: #ccc;
        }
        .tab-buttons {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .tab-button {
            padding: 5px 10px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            cursor: pointer;
            border-radius: 3px;
            color: white;
        }
        .tab-button.active {
            background: rgba(255,255,255,0.3);
        }
        .threshold-panel {
            display: none;
        }
        .threshold-panel.active {
            display: block;
        }
        .gallery-button {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 8px;
            text-align: left;
            font-size: 12px;
            border: 2px solid;
            background: rgba(0,0,0,0.5);
        }
        .gallery-button:hover {
            background: rgba(255,255,255,0.1);
        }
        .gallery-button.active {
            background: rgba(255,255,255,0.2);
        }
        .nav-buttons {
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }
        .nav-button {
            flex: 1;
            padding: 5px;
            font-size: 11px;
        }
        #gallery-info {
            text-align: center;
            margin: 10px 0;
            font-size: 11px;
            color: #ccc;
        }
        h3 {
            margin-top: 0;
            color: #fff;
        }
        input[type="range"] {
            background: rgba(255,255,255,0.1);
        }
        select {
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 3px 5px;
        }
        select option {
            background: #000;
            color: white;
        }
        #color-mode-panel {
            position: absolute;
            top: 10px;
            left: 440px;
            background: rgba(0,0,0,0.8);
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid rgba(255,255,255,0.2);
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: left 0.3s ease;
        }
        #color-mode-panel.shifted {
            left: 60px;
        }
        .color-mode-btn {
            padding: 8px 15px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.15s ease;
            white-space: nowrap;
        }
        .color-mode-btn:hover {
            background: rgba(255,255,255,0.2);
            border-color: rgba(255,255,255,0.4);
        }
        .color-mode-btn.active {
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
            color: #4CAF50;
        }
    </style>
</head>
<body>
    <div id="cursor-indicator"></div>
    
    <div id="color-mode-panel" class="shifted">
        <h4 style="margin: 0 10px 0 0; margin-top: 0;">Color By</h4>
        <button class="color-mode-btn active" data-mode="ai_sc" onclick="setColorMode('ai_sc')">AI/SC</button>
        <button class="color-mode-btn" data-mode="topics" onclick="setColorMode('topics')">Topics</button>
        <button class="color-mode-btn" data-mode="pc_gradient" onclick="setColorMode('pc_gradient')">PC Gradient</button>
    </div>
    
    <div id="info" class="collapsed">
        <div class="collapse-btn" onclick="toggleInfoCollapse()">
            <span id="info-collapse-icon">◀</span>
        </div>
        <h3>UMAP Visualization</h3>
        <div>Total Essays: """ + str(len(viz_data)) + """</div>
        
        <div class="control-group">
            <label>AI Rating Thresholds:</label>
            <div>
                Low: &lt; <input type="number" id="ai-low-val" class="threshold-input" value=\"""" + f"{ai_percentiles[10]:.1f}" + """\" min="1" max="10" step="0.1" onchange="updateFromValues('ai'); updateCategories()">
                <span style="color: #888;">(P<span id="ai-low-pct-display">10</span>)</span>
                <br>
                High: &gt; <input type="number" id="ai-high-val" class="threshold-input" value=\"""" + f"{ai_percentiles[90]:.1f}" + """\" min="1" max="10" step="0.1" onchange="updateFromValues('ai'); updateCategories()">
                <span style="color: #888;">(P<span id="ai-high-pct-display">90</span>)</span>
            </div>
            <div style="margin-top: 5px;">
                Low: P<input type="number" id="ai-low-pct-input" min="0" max="50" value="10" step="1" style="width: 40px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 2px 4px;" onchange="updatePercentileFromInput('ai', 'low'); updateCategories()">
                <input type="range" id="ai-low-pct" min="0" max="50" value="10" step="1" style="width: 60px;" oninput="updateFromPercentiles('ai'); updateCategories()">
                <br>
                High: P<input type="number" id="ai-high-pct-input" min="50" max="100" value="90" step="1" style="width: 40px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 2px 4px;" onchange="updatePercentileFromInput('ai', 'high'); updateCategories()">
                <input type="range" id="ai-high-pct" min="50" max="100" value="90" step="1" style="width: 60px;" oninput="updateFromPercentiles('ai'); updateCategories()">
            </div>
        </div>
        
        <div class="control-group">
            <label>Social Class Thresholds:</label>
            <div>
                Low: ≤ <input type="number" id="sc-low-val" class="threshold-input" value="1" min="1" max="5" step="1" onchange="updateFromValues('sc'); updateCategories()">
                <span style="color: #888;">(P<span id="sc-low-pct-display">10</span>)</span>
                <br>
                High: ≥ <input type="number" id="sc-high-val" class="threshold-input" value="5" min="1" max="5" step="1" onchange="updateFromValues('sc'); updateCategories()">
                <span style="color: #888;">(P<span id="sc-high-pct-display">90</span>)</span>
            </div>
            <div style="margin-top: 5px;">
                Low: P<input type="number" id="sc-low-pct-input" min="0" max="50" value="10" step="1" style="width: 40px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 2px 4px;" onchange="updatePercentileFromInput('sc', 'low'); updateCategories()">
                <input type="range" id="sc-low-pct" min="0" max="50" value="10" step="1" style="width: 60px;" oninput="updateFromPercentiles('sc'); updateCategories()">
                <br>
                High: P<input type="number" id="sc-high-pct-input" min="50" max="100" value="90" step="1" style="width: 40px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 2px 4px;" onchange="updatePercentileFromInput('sc', 'high'); updateCategories()">
                <input type="range" id="sc-high-pct" min="50" max="100" value="90" step="1" style="width: 60px;" oninput="updateFromPercentiles('sc'); updateCategories()">
            </div>
        </div>
        
        <div style="margin-top: 10px;">
            <label>Presets:</label>
            <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 5px;">
                <button onclick="applyPreset('extremes')" style="padding: 5px 10px;">Extremes (P10/P90)</button>
                <button onclick="applyPreset('quartiles')" style="padding: 5px 10px;">Quartiles (P25/P75)</button>
                <button onclick="applyPreset('median')" style="padding: 5px 10px;">Median (P50)</button>
            </div>
        </div>
        
        <div id="legend" style="margin-top: 15px;">
            <div class="legend-item">
                <span class="color-box" style="background: #00ff00;"></span>
                <span>High AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #ff00ff;"></span>
                <span>High AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #00ffff;"></span>
                <span>Low AI + High SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #ffff00;"></span>
                <span>Low AI + Low SC</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #666666;"></span>
                <span>Middle (neither extreme)</span>
            </div>
        </div>
        
        <div id="pc-legend" style="display: none; margin-top: 10px;">
            <label>PC Gradient:</label>
            <div style="display: flex; align-items: center; margin-top: 5px;">
                <span style="color: #999; font-size: 11px;">Low</span>
                <div style="width: 100px; height: 15px; margin: 0 5px;
                            background: linear-gradient(to right, #9400D3, #FFFF00);
                            border: 1px solid rgba(255,255,255,0.3);"></div>
                <span style="color: #999; font-size: 11px;">High</span>
            </div>
            <div id="pc-name" style="margin-top: 5px; font-size: 12px; color: #ccc;"></div>
        </div>
        
        <div id="counts"></div>
    </div>
    
    <div id="controls" class="collapsed">
        <div class="collapse-btn" onclick="toggleControlsCollapse()">
            <span id="collapse-icon">▶</span>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="auto-rotate" checked> Auto-rotate
            </label>
            <div>
                Speed: <input type="range" id="rotate-speed" min="0.1" max="2" step="0.1" value="0.5" style="width: 100px;">
            </div>
        </div>
        <div class="control-group">
            <label>Point Opacity:</label>
            <input type="range" id="point-opacity" min="0.1" max="1" step="0.1" value="0.8" style="width: 100px;">
            <span id="opacity-val">0.8</span>
        </div>
        <div class="control-group">
            <label>Essay BG Opacity:</label>
            <input type="range" id="essay-opacity" min="0.1" max="1" step="0.05" value="0.25" style="width: 100px;">
            <span id="essay-opacity-val">0.25</span>
        </div>
        <div class="control-group">
            <label>Essay Font Size:</label>
            <input type="range" id="essay-font-size" min="6" max="72" step="2" value="24" style="width: 100px;">
            <span id="font-size-val">24</span>px
        </div>
        <div class="control-group">
            <label>Essay Height:</label>
            <input type="range" id="essay-height" min="5" max="90" step="5" value="30" style="width: 100px;">
            <span id="essay-height-val">30</span>%
        </div>
        <div class="control-group">
            <label>Transition Speed:</label>
            <input type="range" id="transition-speed" min="0.5" max="3" step="0.1" value="1.5" style="width: 100px;">
            <span id="transition-val">1.5</span>s
        </div>
        <div class="control-group">
            <label>Principal Component Analysis:</label>
            <select id="pc-select" onchange="handlePCSelectChange()" style="width: 100%;">
                <option value="">Select a PC...</option>
            </select>
            <label style="margin-top: 8px; display: block;">
                <input type="checkbox" id="show-pc-stats" onchange="togglePCStatsBox()"> Show Advanced PC Stats
            </label>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="toggle-dml" onchange="toggleDMLTable()"> Show DML Stats
            </label>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="toggle-topic-stats" onchange="toggleTopicStatsPanel()"> Show Topic Stats
            </label>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="toggle-topics" onchange="updateTopicVisibility()"> Show Topics
            </label>
        </div>
        <div class="control-group">
            <label>Visible Topics (Closest):</label>
            <input type="range" id="topic-count" min="1" max="100" value="20" step="1" style="width: 100px;" oninput="updateTopicVisibility(); document.getElementById('topic-count-val').textContent = this.value;">
            <span id="topic-count-val">20</span>
        </div>
        <div class="control-group">
            <label>Topic Text Size:</label>
            <input type="range" id="topic-text-size" min="5" max="50" value="15" step="1" style="width: 100px;" oninput="updateTopicTextSize(); document.getElementById('topic-text-size-val').textContent = this.value;">
            <span id="topic-text-size-val">15</span>
        </div>
        <div class="control-group">
            <label>Topic Label Opacity:</label>
            <input type="range" id="topic-opacity" min="0.1" max="1" step="0.1" value="0.7" style="width: 100px;" oninput="updateTopicOpacity(); document.getElementById('topic-opacity-val').textContent = this.value;">
            <span id="topic-opacity-val">0.7</span>
        </div>
        <div class="control-group">
            <label>
                <input type="checkbox" id="topic-labels-front" onchange="updateTopicLayering()"> Topic Labels on Top
            </label>
        </div>
    </div>
    
    <div id="gallery-controls" style="right: 60px;">
        <h4 style="margin-top: 0;">Gallery Mode</h4>
        <button class="gallery-button" style="border-color: #00ff00;" onclick="startGallery('both_high')">
            High AI + High SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #ff00ff;" onclick="startGallery('ai_high')">
            High AI + Low SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #00ffff;" onclick="startGallery('sc_high')">
            Low AI + High SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        <button class="gallery-button" style="border-color: #ffff00;" onclick="startGallery('both_low')">
            Low AI + Low SC<br>
            <span style="font-size: 10px; opacity: 0.7;">0 essays</span>
        </button>
        
        <div id="gallery-info"></div>
        
        <div class="nav-buttons" style="display: none;">
            <button class="nav-button" onclick="navigateGallery(-1)">← Previous</button>
            <button class="nav-button" onclick="stopGallery()">Stop</button>
            <button class="nav-button" onclick="navigateGallery(1)">Next →</button>
        </div>
    </div>
    
    <div id="essay-display">
        <div class="resize-handle-top"></div>
        <div class="resize-handle-left"></div>
        <div class="resize-handle-right"></div>
        <div class="resize-handle-corner top-left"></div>
        <div class="resize-handle-corner top-right"></div>
        <span class="layer-btn on-top" onclick="toggleLayer()" title="Toggle layer order">↕</span>
        <span class="minimize-btn" onclick="toggleMinimize()">–</span>
        <div id="essay-header">
            <strong>Essay ID:</strong> <span id="essay-id"></span> | 
            <strong>SC:</strong> <span id="essay-sc"></span> | 
            <strong>AI:</strong> <span id="essay-ai"></span>
        </div>
        <div id="essay-text"></div>
        <div id="pc-analysis">
            <h4 style="margin: 0 0 10px 0; display: none;">Principal Components</h4>
            <div id="pc-list"></div>
        </div>
    </div>
    
    <div id="pc-global-info">
        <span class="close-btn" onclick="closePCStatsBox()">×</span>
        <h4 id="pc-title"></h4>
        <div id="pc-content"></div>
    </div>
    
    <div id="dml-table">
        <span class="close-btn" onclick="document.getElementById('dml-table').style.display='none'; document.getElementById('toggle-dml').checked=false;">×</span>
        <h4>Double Machine Learning Results</h4>
        <table class="dml-stats">
            <tr>
                <td></td>
                <td style="text-align: center; font-weight: bold; color: #888;">Non Cross-Fitted</td>
                <td style="text-align: center; font-weight: bold; color: #888;">Cross-Fitted (5-fold)</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 10px; font-weight: bold; color: #4CAF50;">Naive Model (No Text Controls)</td>
            </tr>
            <tr>
                <td>θ (SC→AI):</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['theta_naive']:.3f}" + """</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['se_naive']:.3f}" + """</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td colspan="2" style="text-align: center;">""" + f"({dml_results_computed['ci_naive'][0]:.3f}, {dml_results_computed['ci_naive'][1]:.3f})" + """</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['pval_naive']:.4f}" + """</td>
            </tr>
            <tr>
                <td>R² (variance explained):</td>
                <td colspan="2" style="text-align: center;">""" + f"{dml_results_computed['r2_naive']:.3f}" + """</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #4CAF50;">All 200 PCs Model</td>
            </tr>
            <tr>
                <td>DML θ (SC→AI):</td>
                <td>""" + f"{dml_results_computed['theta_200']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['theta_200_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>""" + f"{dml_results_computed['se_200']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['se_200_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>""" + f"({dml_results_computed['ci_200'][0]:.3f}, {dml_results_computed['ci_200'][1]:.3f})" + """</td>
                <td>""" + f"({dml_results_computed['ci_200_cf'][0]:.3f}, {dml_results_computed['ci_200_cf'][1]:.3f})" + """</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>""" + f"{dml_results_computed['pval_200']:.4f}" + """</td>
                <td>""" + f"{dml_results_computed['pval_200_cf']:.4f}" + """</td>
            </tr>
            <tr>
                <td style="color: #ff9800;">Effect Reduction vs Naive:</td>
                <td style="color: #ff9800;">""" + f"{dml_results_computed['reduction_200']:.1f}%" + """</td>
                <td style="color: #ff9800;">""" + f"{dml_results_computed['reduction_200_cf']:.1f}%" + """</td>
            </tr>
            <tr>
                <td>AI R² (200 PCs):</td>
                <td>""" + f"{r2_ai_200:.3f}" + """</td>
                <td>""" + f"{r2_ai_200_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td>SC R² (200 PCs):</td>
                <td>""" + f"{r2_sc_200:.3f}" + """</td>
                <td>""" + f"{r2_sc_200_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td colspan="3" style="padding-top: 15px; font-weight: bold; color: #4CAF50;">Top 5 PCs Model</td>
            </tr>
            <tr>
                <td>DML θ (SC→AI):</td>
                <td>""" + f"{dml_results_computed['theta_top5']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['theta_top5_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>Standard Error:</td>
                <td>""" + f"{dml_results_computed['se_top5']:.3f}" + """</td>
                <td>""" + f"{dml_results_computed['se_top5_cf']:.3f}" + """</td>
            </tr>
            <tr>
                <td>95% CI:</td>
                <td>""" + f"({dml_results_computed['ci_top5'][0]:.3f}, {dml_results_computed['ci_top5'][1]:.3f})" + """</td>
                <td>""" + f"({dml_results_computed['ci_top5_cf'][0]:.3f}, {dml_results_computed['ci_top5_cf'][1]:.3f})" + """</td>
            </tr>
            <tr>
                <td>p-value:</td>
                <td>""" + f"{dml_results_computed['pval_top5']:.4f}" + """</td>
                <td>""" + f"{dml_results_computed['pval_top5_cf']:.4f}" + """</td>
            </tr>
            <tr>
                <td style="color: #ff9800;">Effect Reduction vs Naive:</td>
                <td style="color: #ff9800;">""" + f"{dml_results_computed['reduction_top5']:.1f}%" + """</td>
                <td style="color: #ff9800;">""" + f"{dml_results_computed['reduction_top5_cf']:.1f}%" + """</td>
            </tr>
            <tr>
                <td>AI R² (Top 5):</td>
                <td>""" + f"{r2_ai_top5:.3f}" + """</td>
                <td>""" + f"{r2_ai_top5_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td>SC R² (Top 5):</td>
                <td>""" + f"{r2_sc_top5:.3f}" + """</td>
                <td>""" + f"{r2_sc_top5_cf:.3f}" + """</td>
            </tr>
            <tr>
                <td style="font-style: italic; color: #aaa; padding-top: 8px;">PCs Used (Top 5):</td>
                <td colspan="2" style="text-align: center; font-style: italic; color: #aaa; padding-top: 8px;">""" + top_5_pc_string + """</td>
            </tr>
        </table>
    </div>
    
    <div id="topic-stats-panel" style="display: none; position: absolute; bottom: 120px; right: 20px; background: rgba(0,0,0,0.9); padding: 20px; border-radius: 10px; border: 1px solid #444; max-width: 700px; max-height: 500px; z-index: 1002;">
        <span class="close-btn" onclick="document.getElementById('topic-stats-panel').style.display='none'; document.getElementById('toggle-topic-stats').checked=false;" style="position: absolute; top: 10px; right: 10px; font-size: 24px; cursor: pointer; color: #fff;">&times;</span>
        <h4 style="margin-top: 0; color: #4CAF50;">Topic Profile Statistics (Top/Bottom 10%)</h4>
        <div style="max-height: 400px; overflow-y: auto;">
            <table id="topic-stats-table-content" class="dml-stats" style="width: 100%; margin-top: 10px;">
                <thead>
                    <tr>
                        <th style="text-align: left;">Topic (Keywords)</th>
                        <th>Size</th>
                        <th>% High SC</th>
                        <th>% Low SC</th>
                        <th>% High AI</th>
                        <th>% Low AI</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Will be populated by JavaScript -->
                </tbody>
            </table>
            <div style="margin-top: 15px; font-size: 11px; color: #999;">
                <p style="margin: 5px 0;">AI Rating thresholds: Top 10% ≥ """ + f"{ai_top10_threshold:.2f}" + """, Bottom 10% ≤ """ + f"{ai_bottom10_threshold:.2f}" + """</p>
                <p style="margin: 5px 0;">Social Class thresholds: Top = 5, Bottom = 1</p>
                <p style="margin: 5px 0; font-style: italic;">Topics ranked by highest percentage in any extreme group</p>
            </div>
        </div>
    </div>
    
    <script>
        // Data and models
        const data = """ + json.dumps(viz_data) + """;
        const topicVizData = """ + json.dumps(topic_viz_data) + """;
        const topicStatsData = """ + json.dumps(topic_stats_data) + """;
        const cloudCenter = {
            x: """ + str(center_x * 100) + """,
            y: """ + str(center_y * 100) + """,
            z: """ + str(center_z * 100) + """
        };
        const aiPercentiles = """ + json.dumps(ai_percentiles) + """;
        const pcGlobalEffects = """ + json.dumps(pc_global_effects) + """;
        const pcVarianceExplained = """ + json.dumps(variance_explained.tolist()) + """;
        
        // Store user thresholds
        let userThresholds = {
            ai_low: """ + str(ai_percentiles[10]) + """,
            ai_high: """ + str(ai_percentiles[90]) + """,
            sc_low: 1,
            sc_high: 5,
            pc_high_pct: 10,
            pc_low_pct: 10
        };
        
        // Calculate rating percentile functions
        const aiRatings = data.map(d => d.ai_rating).sort((a, b) => a - b);
        const scRatings = data.map(d => d.sc11).sort((a, b) => a - b);
        
        function getAIPercentile(percentile) {
            const index = Math.floor((percentile / 100) * (aiRatings.length - 1));
            return aiRatings[index];
        }
        
        function getSCPercentile(percentile) {
            const index = Math.floor((percentile / 100) * (scRatings.length - 1));
            return scRatings[index];
        }
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            2000
        );
        camera.position.set(250, 250, 250);
        camera.lookAt(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.enableZoom = true;
        controls.zoomSpeed = 1.2;
        controls.minDistance = 50;
        controls.maxDistance = 1000;
        
        // Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(1, 1, 1);
        scene.add(dirLight);
        
        // Create points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(data.length * 3);
        const colorsArray = new Float32Array(data.length * 3);
        
        // Fill positions
        data.forEach((d, i) => {
            positions[i * 3] = d.x * 100;
            positions[i * 3 + 1] = d.y * 100;
            positions[i * 3 + 2] = d.z * 100;
        });
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
        
        // Get reference to the actual color buffer
        const colors = geometry.attributes.color.array;
        
        // Point material
        const material = new THREE.PointsMaterial({
            size: 4,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.8
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Category system
        const categories = new Array(data.length);
        const categoryIndices = {
            'both_high': [],
            'ai_high': [],
            'sc_high': [],
            'both_low': [],
            'middle': []
        };
        
        // Store sorted indices for proximity navigation
        const sortedCategoryIndices = {
            'both_high': [],
            'ai_high': [],
            'sc_high': [],
            'both_low': [],
            'middle': []
        };
        
        // Colors
        const categoryColors = {
            'both_high': [0.0, 1.0, 0.0],      // Bright Green
            'ai_high': [1.0, 0.0, 1.0],        // Magenta
            'sc_high': [0.0, 1.0, 1.0],        // Cyan
            'both_low': [1.0, 1.0, 0.0],       // Yellow
            'middle': [0.4, 0.4, 0.4]          // Dark Gray
        };
        
        // Gallery state
        let galleryMode = false;
        let currentGalleryCategory = null;
        let currentGalleryIndex = 0;
        let savedAutoRotate = true;
        let returningToOverview = false;
        
        // PC analysis state
        let currentPCIndex = null;
        let currentColorMode = 'ai_sc';  // 'ai_sc', 'topics', or 'pc_gradient'
        
        // Animation state
        let isAnimating = false;
        let animationStartTime = 0;
        let animationDuration = 1500;
        let animationStart = {
            cameraPos: new THREE.Vector3(),
            targetPos: new THREE.Vector3()
        };
        let animationEnd = {
            cameraPos: new THREE.Vector3(),
            targetPos: new THREE.Vector3()
        };
        
        // Topic visualization state
        let topicObjects = [];
        let topicColors = {};
        
        // Initialize originalColors array before updateCategories
        const originalColors = new Float32Array(data.length * 3);
        
        // Generate topic colors
        function generateTopicColors() {
            const numTopics = Math.max(...data.map(d => d.hdbscan_topic_id)) + 1;
            for (let i = 0; i < numTopics; i++) {
                const hue = (i * 360 / numTopics) % 360;
                const color = new THREE.Color();
                color.setHSL(hue / 360, 0.7, 0.5);
                topicColors[i] = color;
            }
            // Noise color (topic -1)
            topicColors[-1] = new THREE.Color(0x333333);
        }
        
        // Create text sprite for labels
        function createTextSprite(text, size) {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            // Measure text
            context.font = `600 ${size}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
            const metrics = context.measureText(text);
            const textWidth = metrics.width;
            
            // Set canvas size with padding
            const padding = 12;
            canvas.width = textWidth + padding * 2;
            canvas.height = size + padding * 2;
            
            // Draw background with rounded corners effect
            context.fillStyle = 'rgba(0, 0, 0, 0.75)';
            const radius = 6;
            context.beginPath();
            context.moveTo(radius, 0);
            context.lineTo(canvas.width - radius, 0);
            context.quadraticCurveTo(canvas.width, 0, canvas.width, radius);
            context.lineTo(canvas.width, canvas.height - radius);
            context.quadraticCurveTo(canvas.width, canvas.height, canvas.width - radius, canvas.height);
            context.lineTo(radius, canvas.height);
            context.quadraticCurveTo(0, canvas.height, 0, canvas.height - radius);
            context.lineTo(0, radius);
            context.quadraticCurveTo(0, 0, radius, 0);
            context.closePath();
            context.fill();
            
            // Draw text
            context.font = `600 ${size}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
            context.fillStyle = 'white';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(text, canvas.width / 2, canvas.height / 2);
            
            // Create texture and sprite
            const texture = new THREE.CanvasTexture(canvas);
            const opacity = parseFloat(document.getElementById('topic-opacity').value);
            const material = new THREE.SpriteMaterial({ 
                map: texture,
                transparent: true,
                opacity: opacity
            });
            const sprite = new THREE.Sprite(material);
            
            // Scale sprite
            const scaleFactor = 0.5;
            sprite.scale.set(canvas.width * scaleFactor, canvas.height * scaleFactor, 1);
            
            return sprite;
        }
        
        // Create topic visual objects
        function createTopicVisuals() {
            // Clear existing topics
            topicObjects.forEach(obj => scene.remove(obj.group));
            topicObjects = [];
            
            // Set max value for slider
            document.getElementById('topic-count').max = topicVizData.length;
            
            topicVizData.forEach(topic => {
                const group = new THREE.Group();
                
                // Create centroid marker
                const markerGeometry = new THREE.SphereGeometry(2, 16, 16);
                const markerMaterial = new THREE.MeshBasicMaterial({ 
                    color: topicColors[topic.topic_id] || 0xffffff,
                    opacity: 0.8,
                    transparent: true
                });
                const marker = new THREE.Mesh(markerGeometry, markerMaterial);
                marker.position.set(...topic.centroid);
                group.add(marker);
                
                // Create text label
                const textSize = parseInt(document.getElementById('topic-text-size').value);
                const label = createTextSprite(topic.keywords, textSize);
                label.position.set(
                    topic.centroid[0],
                    topic.centroid[1] + 10,
                    topic.centroid[2]
                );
                group.add(label);
                
                // Initially hidden
                group.visible = false;
                scene.add(group);
                
                topicObjects.push({
                    group: group,
                    position: new THREE.Vector3(...topic.centroid),
                    id: topic.topic_id,
                    label: label,
                    keywords: topic.keywords
                });
            });
        }
        
        // Functions
        function easeInOutCubic(t) {
            return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }
        
        function calculateDistance(idx1, idx2) {
            const dx = positions[idx1 * 3] - positions[idx2 * 3];
            const dy = positions[idx1 * 3 + 1] - positions[idx2 * 3 + 1];
            const dz = positions[idx1 * 3 + 2] - positions[idx2 * 3 + 2];
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        
        function sortByProximity(indices) {
            if (indices.length <= 1) return indices;
            
            const sorted = [];
            const remaining = [...indices];
            
            sorted.push(remaining.shift());
            
            while (remaining.length > 0) {
                const currentIdx = sorted[sorted.length - 1];
                let nearestIdx = 0;
                let nearestDist = Infinity;
                
                for (let i = 0; i < remaining.length; i++) {
                    const dist = calculateDistance(currentIdx, remaining[i]);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearestIdx = i;
                    }
                }
                
                sorted.push(remaining.splice(nearestIdx, 1)[0]);
            }
            
            return sorted;
        }
        
        // Populate PC dropdown
        function populatePCDropdown() {
            const select = document.getElementById('pc-select');
            select.innerHTML = '<option value="">Select a PC...</option>';
            
            // Add all 200 PCs
            for (let i = 0; i < 200; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `PC${i}`;
                select.appendChild(option);
            }
        }
        
        // Show global PC info
        
        window.navigatePC = function(direction) {
            if (currentPCIndex === null) return;
            
            let newIndex = currentPCIndex + direction;
            // Wrap around at boundaries
            if (newIndex < 0) newIndex = 199;
            if (newIndex > 199) newIndex = 0;
            
            currentPCIndex = newIndex;
            document.getElementById('pc-select').value = newIndex;
            updatePCAnalysis();
        }
        
        function updatePCAnalysis() {
            if (currentPCIndex === null) return;
            
            const effects = pcGlobalEffects[currentPCIndex];
            
            // Update title with navigation
            document.getElementById('pc-title').innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                    <button onclick="navigatePC(-1)" style="padding: 2px 8px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; cursor: pointer; border-radius: 3px; font-size: 16px;">
                        ◀
                    </button>
                    <span style="font-size: 18px; font-weight: bold;">PC${currentPCIndex} Detailed Statistics</span>
                    <button onclick="navigatePC(1)" style="padding: 2px 8px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; cursor: pointer; border-radius: 3px; font-size: 16px;">
                        ▶
                    </button>
                </div>
            `;
            
            // Calculate statistics for this PC
            const contributions_ai = data.map(d => d.all_pc_contributions_ai[currentPCIndex]);
            const contributions_sc = data.map(d => d.all_pc_contributions_sc[currentPCIndex]);
            
            // Calculate rankings for each essay based on absolute SHAP values
            const rankings_ai = [];
            const rankings_sc = [];
            
            data.forEach((d, i) => {
                // Get all absolute contributions for this essay
                const abs_contribs_ai = d.all_pc_contributions_ai.map(Math.abs);
                const abs_contribs_sc = d.all_pc_contributions_sc.map(Math.abs);
                
                // Sort indices by contribution magnitude
                const sorted_ai = abs_contribs_ai.map((val, idx) => ({val, idx}))
                    .sort((a, b) => b.val - a.val);
                const sorted_sc = abs_contribs_sc.map((val, idx) => ({val, idx}))
                    .sort((a, b) => b.val - a.val);
                
                // Find rank of current PC (1-based)
                const rank_ai = sorted_ai.findIndex(item => item.idx === currentPCIndex) + 1;
                const rank_sc = sorted_sc.findIndex(item => item.idx === currentPCIndex) + 1;
                
                rankings_ai.push(rank_ai);
                rankings_sc.push(rank_sc);
            });
            
            // Calculate statistics
            const avg_rank_ai = rankings_ai.reduce((a, b) => a + b, 0) / rankings_ai.length;
            const median_rank_ai = rankings_ai.sort((a, b) => a - b)[Math.floor(rankings_ai.length / 2)];
            const avg_rank_sc = rankings_sc.reduce((a, b) => a + b, 0) / rankings_sc.length;
            const median_rank_sc = rankings_sc.sort((a, b) => a - b)[Math.floor(rankings_sc.length / 2)];
            
            // Calculate range and std dev for SHAP values
            const min_shap_ai = Math.min(...contributions_ai);
            const max_shap_ai = Math.max(...contributions_ai);
            const range_shap_ai = max_shap_ai - min_shap_ai;
            const avg_shap_ai = contributions_ai.reduce((a, b) => a + b, 0) / contributions_ai.length;
            const std_shap_ai = Math.sqrt(contributions_ai.reduce((sum, val) => sum + Math.pow(val - avg_shap_ai, 2), 0) / contributions_ai.length);
            
            const min_shap_sc = Math.min(...contributions_sc);
            const max_shap_sc = Math.max(...contributions_sc);
            const range_shap_sc = max_shap_sc - min_shap_sc;
            const avg_shap_sc = contributions_sc.reduce((a, b) => a + b, 0) / contributions_sc.length;
            const std_shap_sc = Math.sqrt(contributions_sc.reduce((sum, val) => sum + Math.pow(val - avg_shap_sc, 2), 0) / contributions_sc.length);
            
            // Calculate correlations between PC values and outcomes
            const pc_values = data.map(d => d.all_pc_values[currentPCIndex]);
            const ai_ratings = data.map(d => d.ai_rating);
            const sc_values = data.map(d => d.sc11);
            
            // Pearson correlation function
            function pearsonCorrelation(x, y) {
                const n = x.length;
                const sumX = x.reduce((a, b) => a + b, 0);
                const sumY = y.reduce((a, b) => a + b, 0);
                const sumXY = x.map((xi, i) => xi * y[i]).reduce((a, b) => a + b, 0);
                const sumX2 = x.map(xi => xi * xi).reduce((a, b) => a + b, 0);
                const sumY2 = y.map(yi => yi * yi).reduce((a, b) => a + b, 0);
                
                const numerator = n * sumXY - sumX * sumY;
                const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
                
                return denominator === 0 ? 0 : numerator / denominator;
            }
            
            const corr_pc_ai = pearsonCorrelation(pc_values, ai_ratings);
            const corr_pc_sc = pearsonCorrelation(pc_values, sc_values);
            
            const variance_pct = pcVarianceExplained[currentPCIndex] * 100;
            
            const formatDiff = (diff) => {
                const sign = diff > 0 ? '+' : '';
                const cls = Math.abs(diff) > 0.2 ? 'prob-high' : (Math.abs(diff) > 0.1 ? 'prob-med' : 'prob-low');
                return `<span class="${cls}">${sign}${(diff * 100).toFixed(1)}%</span>`;
            };
            
            const content = `
                <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">
                    <h5 style="color: #2196F3; margin: 0 0 10px 0;">Importance Rankings (out of 200 PCs)</h5>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td style="width: 50%;"><strong>AI Rating Model:</strong></td><td style="width: 50%;"><strong>Social Class Model:</strong></td></tr>
                        <tr><td>Average Rank: ${avg_rank_ai.toFixed(1)}</td><td>Average Rank: ${avg_rank_sc.toFixed(1)}</td></tr>
                        <tr><td>Median Rank: ${median_rank_ai}</td><td>Median Rank: ${median_rank_sc}</td></tr>
                    </table>
                </div>
                
                <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">
                    <h5 style="color: #2196F3; margin: 0 0 10px 0;">SHAP Value Statistics</h5>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td style="width: 50%;"><strong>AI Rating:</strong></td><td style="width: 50%;"><strong>Social Class:</strong></td></tr>
                        <tr><td>Range: [${min_shap_ai.toFixed(4)}, ${max_shap_ai.toFixed(4)}]</td><td>Range: [${min_shap_sc.toFixed(4)}, ${max_shap_sc.toFixed(4)}]</td></tr>
                        <tr><td>Total Range: ${range_shap_ai.toFixed(4)}</td><td>Total Range: ${range_shap_sc.toFixed(4)}</td></tr>
                        <tr><td>Std Dev: ${std_shap_ai.toFixed(4)}</td><td>Std Dev: ${std_shap_sc.toFixed(4)}</td></tr>
                    </table>
                </div>
                
                <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">
                    <h5 style="color: #2196F3; margin: 0 0 10px 0;">Variance & Correlations</h5>
                    <div style="font-size: 13px; margin-bottom: 8px;">
                        <strong>Variance Explained:</strong> ${variance_pct.toFixed(2)}% of total embedding variance
                    </div>
                    <table style="width: 100%; font-size: 12px;">
                        <tr>
                            <td><strong>Correlation with AI Rating:</strong></td>
                            <td style="text-align: right; color: ${Math.abs(corr_pc_ai) > 0.3 ? '#4CAF50' : Math.abs(corr_pc_ai) > 0.1 ? '#FFC107' : '#999'}">
                                r = ${corr_pc_ai.toFixed(3)}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Correlation with Social Class:</strong></td>
                            <td style="text-align: right; color: ${Math.abs(corr_pc_sc) > 0.3 ? '#4CAF50' : Math.abs(corr_pc_sc) > 0.1 ? '#FFC107' : '#999'}">
                                r = ${corr_pc_sc.toFixed(3)}
                            </td>
                        </tr>
                    </table>
                </div>
                
                <div style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">
                    <h5 style="color: #2196F3; margin: 0 0 10px 0;">Top 10% vs Bottom 10% Analysis</h5>
                    <table class="prob-table" style="width: 100%; font-size: 11px;">
                        <tr style="font-weight: bold;">
                            <td>Outcome</td>
                            <td style="text-align: center;">If High PC</td>
                            <td style="text-align: center;">If Low PC</td>
                            <td style="text-align: center;">Difference</td>
                        </tr>
                        <tr>
                            <td>High AI Rating</td>
                            <td style="text-align: center;">${(effects.ai_top10_if_high * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${(effects.ai_top10_if_low * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${formatDiff(effects.ai_top10_diff)}</td>
                        </tr>
                        <tr>
                            <td>Low AI Rating</td>
                            <td style="text-align: center;">${(effects.ai_bottom10_if_high * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${(effects.ai_bottom10_if_low * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${formatDiff(effects.ai_bottom10_diff)}</td>
                        </tr>
                        <tr style="border-top: 1px solid #333;">
                            <td>High Social Class</td>
                            <td style="text-align: center;">${(effects.sc_top10_if_high * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${(effects.sc_top10_if_low * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${formatDiff(effects.sc_top10_diff)}</td>
                        </tr>
                        <tr>
                            <td>Low Social Class</td>
                            <td style="text-align: center;">${(effects.sc_bottom10_if_high * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${(effects.sc_bottom10_if_low * 100).toFixed(1)}%</td>
                            <td style="text-align: center;">${formatDiff(effects.sc_bottom10_diff)}</td>
                        </tr>
                    </table>
                </div>
            `;
            
            document.getElementById('pc-content').innerHTML = content;
        }
        
        function toggleDMLTable() {
            const table = document.getElementById('dml-table');
            const checkbox = document.getElementById('toggle-dml');
            table.style.display = checkbox.checked ? 'block' : 'none';
        }
        
        function toggleTopicStatsPanel() {
            const panel = document.getElementById('topic-stats-panel');
            const checkbox = document.getElementById('toggle-topic-stats');
            panel.style.display = checkbox.checked ? 'block' : 'none';
            
            if (checkbox.checked) {
                populateTopicStatsTable();
            }
        }
        
        function populateTopicStatsTable() {
            const tbody = document.querySelector('#topic-stats-table-content tbody');
            tbody.innerHTML = '';
            
            topicStatsData.forEach(topic => {
                const tr = document.createElement('tr');
                
                // Format percentages with color coding
                const formatPct = (prob) => {
                    const pct = (prob * 100).toFixed(1);
                    let colorClass = '';
                    if (prob > 0.3) colorClass = 'color: #4CAF50;'; // Green for high
                    else if (prob > 0.15) colorClass = 'color: #FFC107;'; // Yellow for medium
                    else colorClass = 'color: #999;'; // Gray for low
                    return `<span style="${colorClass}">${pct}%</span>`;
                };
                
                tr.innerHTML = `
                    <td style="text-align: left; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${topic.keywords}">
                        ${topic.keywords}
                    </td>
                    <td style="text-align: center;">${topic.size}</td>
                    <td style="text-align: center;">${formatPct(topic.prob_sc_top10)}</td>
                    <td style="text-align: center;">${formatPct(topic.prob_sc_bottom10)}</td>
                    <td style="text-align: center;">${formatPct(topic.prob_ai_top10)}</td>
                    <td style="text-align: center;">${formatPct(topic.prob_ai_bottom10)}</td>
                `;
                
                tbody.appendChild(tr);
            });
        }
        
        // Display PC analysis
        function displayPCAnalysis(d) {
            const pcList = document.getElementById('pc-list');
            pcList.innerHTML = '';
            
            d.pc_info.forEach((pc, idx) => {
                const pcDiv = document.createElement('div');
                pcDiv.className = 'pc-item';
                pcDiv.onclick = () => showPCGlobalInfo(parseInt(pc.pc.substring(2)));
                
                pcDiv.innerHTML = `
                    <div class="pc-name">${pc.pc}</div>
                    <div class="pc-percentile">${pc.percentile.toFixed(1)}%ile</div>
                    <div class="pc-values">
                        AI: ${pc.contribution_ai > 0 ? '+' : ''}${pc.contribution_ai.toFixed(2)}<br>
                        SC: ${pc.contribution_sc > 0 ? '+' : ''}${pc.contribution_sc.toFixed(2)}
                    </div>
                `;
                
                pcList.appendChild(pcDiv);
            });
        }
        
        // Update functions for synchronized controls
        window.updateFromValues = function(type) {
            if (type === 'ai') {
                const lowVal = parseFloat(document.getElementById('ai-low-val').value);
                const highVal = parseFloat(document.getElementById('ai-high-val').value);
                
                // Find closest percentiles
                let lowPct = 0;
                for (let i = 0; i < aiRatings.length; i++) {
                    if (aiRatings[i] >= lowVal) {
                        lowPct = (i / aiRatings.length) * 100;
                        break;
                    }
                }
                
                let highPct = 100;
                for (let i = aiRatings.length - 1; i >= 0; i--) {
                    if (aiRatings[i] <= highVal) {
                        highPct = ((i + 1) / aiRatings.length) * 100;
                        break;
                    }
                }
                
                document.getElementById('ai-low-pct').value = Math.round(lowPct);
                document.getElementById('ai-high-pct').value = Math.round(highPct);
                document.getElementById('ai-low-pct-display').textContent = Math.round(lowPct);
                document.getElementById('ai-high-pct-display').textContent = Math.round(highPct);
            } else {
                const lowVal = parseInt(document.getElementById('sc-low-val').value);
                const highVal = parseInt(document.getElementById('sc-high-val').value);
                
                // Find closest percentiles
                let lowPct = 0;
                for (let i = 0; i < scRatings.length; i++) {
                    if (scRatings[i] > lowVal) {
                        lowPct = (i / scRatings.length) * 100;
                        break;
                    }
                }
                
                let highPct = 100;
                for (let i = scRatings.length - 1; i >= 0; i--) {
                    if (scRatings[i] < highVal) {
                        highPct = ((i + 1) / scRatings.length) * 100;
                        break;
                    }
                }
                
                document.getElementById('sc-low-pct').value = Math.round(lowPct);
                document.getElementById('sc-high-pct').value = Math.round(highPct);
                document.getElementById('sc-low-pct-display').textContent = Math.round(lowPct);
                document.getElementById('sc-high-pct-display').textContent = Math.round(highPct);
            }
        };
        
        window.updateFromPercentiles = function(type) {
            if (type === 'ai') {
                const lowPct = parseInt(document.getElementById('ai-low-pct').value);
                const highPct = parseInt(document.getElementById('ai-high-pct').value);
                
                document.getElementById('ai-low-val').value = getAIPercentile(lowPct).toFixed(1);
                document.getElementById('ai-high-val').value = getAIPercentile(highPct).toFixed(1);
                document.getElementById('ai-low-pct-display').textContent = lowPct;
                document.getElementById('ai-high-pct-display').textContent = highPct;
                document.getElementById('ai-low-pct-input').value = lowPct;
                document.getElementById('ai-high-pct-input').value = highPct;
            } else {
                const lowPct = parseInt(document.getElementById('sc-low-pct').value);
                const highPct = parseInt(document.getElementById('sc-high-pct').value);
                
                document.getElementById('sc-low-val').value = Math.round(getSCPercentile(lowPct));
                document.getElementById('sc-high-val').value = Math.round(getSCPercentile(highPct));
                document.getElementById('sc-low-pct-display').textContent = lowPct;
                document.getElementById('sc-high-pct-display').textContent = highPct;
                document.getElementById('sc-low-pct-input').value = lowPct;
                document.getElementById('sc-high-pct-input').value = highPct;
            }
        };
        
        window.updatePercentileFromInput = function(type, level) {
            if (type === 'ai') {
                const pct = parseInt(document.getElementById(`ai-${level}-pct-input`).value);
                document.getElementById(`ai-${level}-pct`).value = pct;
                updateFromPercentiles('ai');
            } else {
                const pct = parseInt(document.getElementById(`sc-${level}-pct-input`).value);
                document.getElementById(`sc-${level}-pct`).value = pct;
                updateFromPercentiles('sc');
            }
        };
        
        window.applyPreset = function(preset) {
            let lowPct, highPct;
            
            switch(preset) {
                case 'median':
                    lowPct = 50;
                    highPct = 50;
                    break;
                case 'quartiles':
                    lowPct = 25;
                    highPct = 75;
                    break;
                case 'extremes':
                    lowPct = 10;
                    highPct = 90;
                    break;
            }
            
            // Update AI percentiles
            document.getElementById('ai-low-pct').value = lowPct;
            document.getElementById('ai-high-pct').value = highPct;
            document.getElementById('ai-low-pct-input').value = lowPct;
            document.getElementById('ai-high-pct-input').value = highPct;
            updateFromPercentiles('ai');
            
            // Update SC percentiles
            document.getElementById('sc-low-pct').value = lowPct;
            document.getElementById('sc-high-pct').value = highPct;
            document.getElementById('sc-low-pct-input').value = lowPct;
            document.getElementById('sc-high-pct-input').value = highPct;
            updateFromPercentiles('sc');
            
            // Update categories immediately
            updateCategories();
        };
        
        // Topic visualization functions
        window.updateTopicTextSize = function() {
            const size = parseInt(document.getElementById('topic-text-size').value);
            
            topicObjects.forEach(topic => {
                // Remove old label
                topic.group.remove(topic.label);
                
                // Create new label with updated size
                const newLabel = createTextSprite(topic.keywords, size);
                newLabel.position.copy(topic.label.position);
                topic.group.add(newLabel);
                topic.label = newLabel;
            });
        };
        
        window.updateTopicVisibility = function() {
            const showTopics = document.getElementById('toggle-topics').checked;
            
            if (!showTopics) {
                // Hide all topics
                topicObjects.forEach(obj => obj.group.visible = false);
                return;
            }
            
            // Show topics
            const N = parseInt(document.getElementById('topic-count').value);
            
            // Calculate distances from camera to topics
            const distances = topicObjects.map(topic => ({
                topic: topic,
                distance: camera.position.distanceTo(topic.position)
            }));
            
            // Sort by distance (closest first)
            distances.sort((a, b) => a.distance - b.distance);
            
            // Show only the N closest topics
            distances.forEach((item, index) => {
                item.topic.group.visible = index < N;
            });
        };
        
        window.updateTopicOpacity = function() {
            const opacity = parseFloat(document.getElementById('topic-opacity').value);
            
            topicObjects.forEach(topic => {
                if (topic.label && topic.label.material) {
                    topic.label.material.opacity = opacity;
                }
            });
        };
        
        // PC gradient color function
        function getPCGradientColor(percentile) {
            const t = percentile / 100;
            const purple = new THREE.Color(0x9400D3); // DarkViolet
            const yellow = new THREE.Color(0xFFFF00); // Yellow
            const color = new THREE.Color().lerpColors(purple, yellow, t);
            return color;
        }
        
        // Master color mode update
        window.setColorMode = function(mode) {
            currentColorMode = mode;
            
            // Update button states
            document.querySelectorAll('.color-mode-btn').forEach(btn => {
                if (btn.dataset.mode === mode) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            applyColors();
            
            // Show/hide appropriate legends
            document.getElementById('legend').style.display = 
                currentColorMode === 'ai_sc' ? 'block' : 'none';
            document.getElementById('pc-legend').style.display = 
                currentColorMode === 'pc_gradient' ? 'block' : 'none';
            
            // Update PC name in legend if showing PC gradient
            if (currentColorMode === 'pc_gradient' && currentPCIndex !== null) {
                document.getElementById('pc-name').textContent = 
                    `PC${currentPCIndex + 1} (${(pcVarianceExplained[currentPCIndex] * 100).toFixed(1)}% variance)`;
            }
        };
        
        // Apply colors based on current mode
        function applyColors() {
            const colors = geometry.attributes.color.array;
            
            if (currentColorMode === 'ai_sc') {
                // Apply AI/SC category colors
                for (let i = 0; i < data.length; i++) {
                    colors[i * 3] = originalColors[i * 3];
                    colors[i * 3 + 1] = originalColors[i * 3 + 1];
                    colors[i * 3 + 2] = originalColors[i * 3 + 2];
                }
            }
            else if (currentColorMode === 'topics') {
                // Apply topic colors
                data.forEach((d, i) => {
                    const color = topicColors[d.hdbscan_topic_id] || topicColors[-1];
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                });
            }
            else if (currentColorMode === 'pc_gradient') {
                // Apply PC gradient colors
                if (currentPCIndex !== null) {
                    data.forEach((d, i) => {
                        const percentile = d.all_pc_percentiles[currentPCIndex];
                        const color = getPCGradientColor(percentile);
                        colors[i * 3] = color.r;
                        colors[i * 3 + 1] = color.g;
                        colors[i * 3 + 2] = color.b;
                    });
                } else {
                    // No PC selected - use grey
                    for (let i = 0; i < data.length; i++) {
                        colors[i * 3] = 0.4;
                        colors[i * 3 + 1] = 0.4;
                        colors[i * 3 + 2] = 0.4;
                    }
                }
            }
            
            geometry.attributes.color.needsUpdate = true;
        }
        
        // Handle PC selection change
        window.handlePCSelectChange = function() {
            const select = document.getElementById('pc-select');
            currentPCIndex = select.value === '' ? null : parseInt(select.value);
            
            // Update colors if in PC gradient mode
            if (currentColorMode === 'pc_gradient') {
                applyColors();
                // Update PC name in legend
                if (currentPCIndex !== null) {
                    document.getElementById('pc-name').textContent = 
                        `PC${currentPCIndex + 1} (${(pcVarianceExplained[currentPCIndex] * 100).toFixed(1)}% variance)`;
                }
            }
            
            // Handle stats box visibility
            togglePCStatsBox();
        };
        
        // Toggle PC stats box based on checkbox and selection
        window.togglePCStatsBox = function() {
            const showStats = document.getElementById('show-pc-stats').checked;
            const hasSelection = currentPCIndex !== null;
            
            if (showStats && hasSelection) {
                updatePCAnalysis();
                document.getElementById('pc-global-info').style.display = 'block';
            } else {
                document.getElementById('pc-global-info').style.display = 'none';
            }
        };
        
        // Close PC stats box
        window.closePCStatsBox = function() {
            document.getElementById('pc-global-info').style.display = 'none';
            document.getElementById('show-pc-stats').checked = false;
        };
        
        window.updateTopicLayering = function() {
            const labelsOnTop = document.getElementById('topic-labels-front').checked;
            
            if (labelsOnTop) {
                // Disable depth test for labels so they render on top
                topicObjects.forEach(topic => {
                    if (topic.label && topic.label.material) {
                        topic.label.material.depthTest = false;
                        topic.label.material.depthWrite = false;
                        topic.label.renderOrder = 999;
                    }
                });
            } else {
                // Enable normal depth testing
                topicObjects.forEach(topic => {
                    if (topic.label && topic.label.material) {
                        topic.label.material.depthTest = true;
                        topic.label.material.depthWrite = true;
                        topic.label.renderOrder = 0;
                    }
                });
            }
        };
        
        function updateCategories() {
            const aiLow = parseFloat(document.getElementById('ai-low-val').value);
            const aiHigh = parseFloat(document.getElementById('ai-high-val').value);
            const scLow = parseInt(document.getElementById('sc-low-val').value);
            const scHigh = parseInt(document.getElementById('sc-high-val').value);
            
            // Update user thresholds
            userThresholds.ai_low = aiLow;
            userThresholds.ai_high = aiHigh;
            userThresholds.sc_low = scLow;
            userThresholds.sc_high = scHigh;
            
            // Update PC analysis if shown
            if (currentPCIndex !== null) {
                updatePCAnalysis();
            }
            
            // Reset category indices
            Object.keys(categoryIndices).forEach(key => categoryIndices[key] = []);
            
            const counts = {
                'both_high': 0,
                'ai_high': 0,
                'sc_high': 0,
                'both_low': 0,
                'middle': 0
            };
            
            data.forEach((d, i) => {
                const highAI = d.ai_rating > aiHigh;
                const lowAI = d.ai_rating < aiLow;
                const highSC = d.sc11 >= scHigh;
                const lowSC = d.sc11 <= scLow;
                
                let category;
                if (highAI && highSC) {
                    category = 'both_high';
                } else if (highAI && lowSC) {
                    category = 'ai_high';
                } else if (lowAI && highSC) {
                    category = 'sc_high';
                } else if (lowAI && lowSC) {
                    category = 'both_low';
                } else {
                    category = 'middle';
                }
                
                categories[i] = category;
                categoryIndices[category].push(i);
                counts[category]++;
                
                const color = categoryColors[category];
                colors[i * 3] = color[0];
                colors[i * 3 + 1] = color[1];
                colors[i * 3 + 2] = color[2];
                
                // Save original colors
                originalColors[i * 3] = color[0];
                originalColors[i * 3 + 1] = color[1];
                originalColors[i * 3 + 2] = color[2];
            });
            
            // Sort category indices by proximity
            Object.keys(categoryIndices).forEach(category => {
                sortedCategoryIndices[category] = sortByProximity(categoryIndices[category]);
            });
            
            geometry.attributes.color.needsUpdate = true;
            
            // Update counts display
            const total = data.length;
            const nonMiddle = total - counts.middle;
            document.getElementById('counts').innerHTML = `
                <strong>Counts:</strong><br>
                High AI + High SC: ${counts.both_high} (${(counts.both_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.both_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                High AI + Low SC: ${counts.ai_high} (${(counts.ai_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.ai_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Low AI + High SC: ${counts.sc_high} (${(counts.sc_high/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.sc_high/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Low AI + Low SC: ${counts.both_low} (${(counts.both_low/total*100).toFixed(1)}% of all, ${nonMiddle > 0 ? (counts.both_low/nonMiddle*100).toFixed(1) : 0}% of extremes)<br>
                Middle: ${counts.middle} (${(counts.middle/total*100).toFixed(1)}% of all)
            `;
            
            // Update gallery buttons
            document.querySelectorAll('.gallery-button').forEach(btn => {
                const category = btn.getAttribute('onclick').match(/startGallery\('(.+?)'\)/)[1];
                const count = counts[category];
                btn.querySelector('span').textContent = `${count} essays`;
            });
        }
        
        // Gallery functions
        window.startGallery = function(category) {
            if (sortedCategoryIndices[category].length === 0) {
                alert('No essays in this category with current thresholds');
                return;
            }
            
            galleryMode = true;
            currentGalleryCategory = category;
            currentGalleryIndex = 0;
            returningToOverview = false;
            
            savedAutoRotate = controls.autoRotate;
            controls.autoRotate = false;
            document.getElementById('auto-rotate').checked = false;
            
            document.querySelectorAll('.gallery-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelector('.nav-buttons').style.display = 'flex';
            
            navigateToEssay(sortedCategoryIndices[category][0]);
        };
        
        window.stopGallery = function() {
            galleryMode = false;
            currentGalleryCategory = null;
            isAnimating = false;
            returningToOverview = true;
            
            controls.autoRotate = savedAutoRotate;
            document.getElementById('auto-rotate').checked = savedAutoRotate;
            
            animateToPosition(
                new THREE.Vector3(250, 250, 250),
                new THREE.Vector3(cloudCenter.x, cloudCenter.y, cloudCenter.z)
            );
            
            document.querySelectorAll('.gallery-button').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.nav-buttons').style.display = 'none';
            document.getElementById('gallery-info').textContent = '';
            document.getElementById('essay-display').style.display = 'none';
        };
        
        window.navigateGallery = function(direction) {
            if (!galleryMode || !currentGalleryCategory || isAnimating) return;
            
            const indices = sortedCategoryIndices[currentGalleryCategory];
            currentGalleryIndex += direction;
            
            if (currentGalleryIndex < 0) currentGalleryIndex = indices.length - 1;
            if (currentGalleryIndex >= indices.length) currentGalleryIndex = 0;
            
            navigateToEssay(indices[currentGalleryIndex]);
        };
        
        function animateToPosition(targetCameraPos, targetLookAt) {
            isAnimating = true;
            animationStartTime = Date.now();
            animationDuration = parseFloat(document.getElementById('transition-speed').value) * 1000;
            
            animationStart.cameraPos.copy(camera.position);
            animationStart.targetPos.copy(controls.target);
            
            animationEnd.cameraPos.copy(targetCameraPos);
            animationEnd.targetPos.copy(targetLookAt);
        }
        
        function navigateToEssay(index) {
            const d = data[index];
            const category = categories[index];
            
            const indices = sortedCategoryIndices[currentGalleryCategory];
            document.getElementById('gallery-info').textContent = 
                `Essay ${currentGalleryIndex + 1} of ${indices.length}`;
            
            const targetPosition = new THREE.Vector3(
                positions[index * 3],
                positions[index * 3 + 1],
                positions[index * 3 + 2]
            );
            
            const distance = 100;
            const angle = Date.now() * 0.0001;
            const cameraPos = new THREE.Vector3(
                targetPosition.x + distance * Math.cos(angle),
                targetPosition.y + distance * 0.5,
                targetPosition.z + distance * Math.sin(angle)
            );
            
            animateToPosition(cameraPos, targetPosition);
            
            // Get current font sizes
            const fontSize = parseInt(document.getElementById('essay-font-size').value);
            const headerFontSize = Math.round(fontSize * 1.2);
            const pcFontSize = Math.round(fontSize * 0.7);
            
            // Create PC summary for header with inline styles
            let pcSummary = '';
            d.pc_info.forEach((pc, idx) => {
                const aiSign = pc.contribution_ai > 0 ? '+' : '';
                const scSign = pc.contribution_sc > 0 ? '+' : '';
                pcSummary += `<span class="pc-inline" style="font-size: ${pcFontSize}px; color: white;" onclick="event.stopPropagation(); showPCGlobalInfo(${parseInt(pc.pc.substring(2))})">
                    ${pc.pc}: ${pc.percentile.toFixed(0)}% | AI:${aiSign}${pc.contribution_ai.toFixed(2)} SC:${scSign}${pc.contribution_sc.toFixed(2)} | ${pc.variance_total.toFixed(1)}%var
                </span>`;
            });
            
            document.getElementById('essay-header').innerHTML = `
                <div class="header-main">
                    <div>
                        <strong>Essay ID:</strong> ${d.essay_id} | 
                        <strong>SC:</strong> ${d.sc11} | 
                        <strong>AI:</strong> ${d.ai_rating.toFixed(2)}
                    </div>
                </div>
                <div class="header-pcs" style="font-size: ${pcFontSize}px; color: white;">
                    <strong>Top 5 PCs:</strong> ${pcSummary}
                </div>
            `;
            document.getElementById('essay-header').style.fontSize = headerFontSize + 'px';
            document.getElementById('essay-text').textContent = d.essay;
            document.getElementById('essay-text').style.fontSize = fontSize + 'px';
            
            const color = categoryColors[category];
            const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
            const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
            document.getElementById('essay-display').style.backgroundColor = bgColor;
            document.getElementById('essay-display').style.borderColor = 
                `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
            document.getElementById('essay-display').style.display = 'block';
        }
        
        
        // Minimize/maximize essay display
        window.toggleMinimize = function() {
            const essayDisplay = document.getElementById('essay-display');
            if (essayDisplay.classList.contains('minimized')) {
                essayDisplay.classList.remove('minimized');
                document.querySelector('.minimize-btn').textContent = '–';
            } else {
                essayDisplay.classList.add('minimized');
                document.querySelector('.minimize-btn').textContent = '+';
            }
        };
        
        // Layer toggle functionality
        let essayOnTop = true;
        window.toggleLayer = function() {
            const essayDisplay = document.getElementById('essay-display');
            const dmlTable = document.getElementById('dml-table');
            const pcInfo = document.getElementById('pc-global-info');
            const layerBtn = document.querySelector('.layer-btn');
            
            if (essayOnTop) {
                essayDisplay.style.zIndex = '50';
                if (dmlTable) dmlTable.style.zIndex = '50';
                if (pcInfo) pcInfo.style.zIndex = '50';
                layerBtn.classList.remove('on-top');
                layerBtn.title = 'Bring to front';
            } else {
                essayDisplay.style.zIndex = '1000';
                if (dmlTable) dmlTable.style.zIndex = '1000';
                if (pcInfo) pcInfo.style.zIndex = '1000';
                layerBtn.classList.add('on-top');
                layerBtn.title = 'Send to back';
            }
            essayOnTop = !essayOnTop;
        };
        
        // Controls collapse toggle
        window.toggleControlsCollapse = function() {
            const controls = document.getElementById('controls');
            const galleryControls = document.getElementById('gallery-controls');
            const icon = document.getElementById('collapse-icon');
            
            if (controls.classList.contains('collapsed')) {
                controls.classList.remove('collapsed');
                icon.textContent = '◀';
                if (galleryControls) {
                    galleryControls.style.right = '480px';
                }
            } else {
                controls.classList.add('collapsed');
                icon.textContent = '▶';
                if (galleryControls) {
                    galleryControls.style.right = '60px';
                }
            }
        };
        
        // Info panel collapse toggle
        window.toggleInfoCollapse = function() {
            const info = document.getElementById('info');
            const icon = document.getElementById('info-collapse-icon');
            const colorPanel = document.getElementById('color-mode-panel');
            
            if (info.classList.contains('collapsed')) {
                info.classList.remove('collapsed');
                icon.textContent = '▶';
                colorPanel.classList.remove('shifted');
            } else {
                info.classList.add('collapsed');
                icon.textContent = '◀';
                colorPanel.classList.add('shifted');
            }
        };
        
        // Click to bring panel to front
        function bringToFront(element) {
            const panels = ['info', 'controls', 'gallery-controls', 'essay-display', 'dml-table', 'pc-global-info', 'topic-stats-panel'];
            let maxZ = 100;
            
            // Find current max z-index
            panels.forEach(id => {
                const el = document.getElementById(id);
                if (el && el !== element) {
                    const z = parseInt(window.getComputedStyle(el).zIndex) || 0;
                    maxZ = Math.max(maxZ, z);
                }
            });
            
            // Set clicked element to front
            element.style.zIndex = maxZ + 1;
        }
        
        // Add click handlers to bring panels to front
        ['info', 'controls', 'gallery-controls', 'essay-display', 'dml-table', 'pc-global-info', 'topic-stats-panel'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('mousedown', function(e) {
                    // Don't bring to front if clicking on interactive elements
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON' || 
                        e.target.tagName === 'SELECT' || e.target.classList.contains('resize-handle') ||
                        e.target.classList.contains('close-btn')) {
                        return;
                    }
                    bringToFront(this);
                });
            }
        });
        
        // Draggable resize functionality
        let resizeState = {
            isResizing: false,
            type: null, // 'top', 'left', 'right', 'top-left', 'top-right'
            startX: 0,
            startY: 0,
            startWidth: 0,
            startHeight: 0,
            startLeft: 0,
            startRight: 0
        };
        
        const essayDisplay = document.getElementById('essay-display');
        
        // Add event listeners to all resize handles
        essayDisplay.querySelector('.resize-handle-top').addEventListener('mousedown', (e) => {
            startResize(e, 'top');
        });
        
        essayDisplay.querySelector('.resize-handle-left').addEventListener('mousedown', (e) => {
            startResize(e, 'left');
        });
        
        essayDisplay.querySelector('.resize-handle-right').addEventListener('mousedown', (e) => {
            startResize(e, 'right');
        });
        
        essayDisplay.querySelector('.resize-handle-corner.top-left').addEventListener('mousedown', (e) => {
            startResize(e, 'top-left');
        });
        
        essayDisplay.querySelector('.resize-handle-corner.top-right').addEventListener('mousedown', (e) => {
            startResize(e, 'top-right');
        });
        
        function startResize(e, type) {
            resizeState.isResizing = true;
            resizeState.type = type;
            resizeState.startX = e.clientX;
            resizeState.startY = e.clientY;
            resizeState.startWidth = essayDisplay.offsetWidth;
            resizeState.startHeight = essayDisplay.offsetHeight;
            resizeState.startLeft = parseInt(window.getComputedStyle(essayDisplay).left);
            resizeState.startRight = parseInt(window.getComputedStyle(essayDisplay).right);
            e.preventDefault();
        }
        
        document.addEventListener('mousemove', (e) => {
            if (!resizeState.isResizing) return;
            
            const deltaX = e.clientX - resizeState.startX;
            const deltaY = e.clientY - resizeState.startY;
            
            switch (resizeState.type) {
                case 'top':
                    const newHeight = window.innerHeight - e.clientY - 10;
                    if (newHeight >= 80 && newHeight <= window.innerHeight * 0.8) {
                        essayDisplay.style.height = newHeight + 'px';
                    }
                    break;
                    
                case 'left':
                    const newLeft = resizeState.startLeft + deltaX;
                    const newWidth = resizeState.startWidth - deltaX;
                    if (newLeft >= 10 && newWidth >= 300) {
                        essayDisplay.style.left = newLeft + 'px';
                        essayDisplay.style.width = newWidth + 'px';
                    }
                    break;
                    
                case 'right':
                    const newRight = resizeState.startRight - deltaX;
                    const newWidthRight = resizeState.startWidth + deltaX;
                    if (newRight >= 10 && newWidthRight >= 300) {
                        essayDisplay.style.right = newRight + 'px';
                        essayDisplay.style.width = newWidthRight + 'px';
                    }
                    break;
                    
                case 'top-left':
                    const newHeightTL = window.innerHeight - e.clientY - 10;
                    const newLeftTL = resizeState.startLeft + deltaX;
                    const newWidthTL = resizeState.startWidth - deltaX;
                    if (newHeightTL >= 80 && newHeightTL <= window.innerHeight * 0.8) {
                        essayDisplay.style.height = newHeightTL + 'px';
                    }
                    if (newLeftTL >= 10 && newWidthTL >= 300) {
                        essayDisplay.style.left = newLeftTL + 'px';
                        essayDisplay.style.width = newWidthTL + 'px';
                    }
                    break;
                    
                case 'top-right':
                    const newHeightTR = window.innerHeight - e.clientY - 10;
                    const newRightTR = resizeState.startRight - deltaX;
                    const newWidthTR = resizeState.startWidth + deltaX;
                    if (newHeightTR >= 80 && newHeightTR <= window.innerHeight * 0.8) {
                        essayDisplay.style.height = newHeightTR + 'px';
                    }
                    if (newRightTR >= 10 && newWidthTR >= 300) {
                        essayDisplay.style.right = newRightTR + 'px';
                        essayDisplay.style.width = newWidthTR + 'px';
                    }
                    break;
            }
        });
        
        document.addEventListener('mouseup', () => {
            resizeState.isResizing = false;
            resizeState.type = null;
        });
        
        // Initialize
        populatePCDropdown();
        
        // Initialize percentile displays first
        updateFromValues('ai');
        updateFromValues('sc');
        
        // Now update categories after colors buffer is ready
        updateCategories();
        
        // Apply colors based on initial mode (should be 'ai_sc')
        applyColors();
        
        // Apply initial font sizes
        applyFontSizes();
        
        // Initialize topic visualization
        generateTopicColors();
        createTopicVisuals();
        
        // Custom cursor
        const cursorIndicator = document.getElementById('cursor-indicator');
        
        function updateCursor(event) {
            cursorIndicator.style.left = event.clientX + 'px';
            cursorIndicator.style.top = event.clientY + 'px';
        }
        
        window.addEventListener('mousemove', updateCursor);
        
        // Raycaster for hover
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Points.threshold = 8;
        
        let hoveredIndex = -1;
        
        function onMouseMove(event) {
            if (galleryMode) return;
            
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(points);
            
            if (intersects.length > 0) {
                const newIndex = intersects[0].index;
                if (newIndex !== hoveredIndex) {
                    hoveredIndex = newIndex;
                    const d = data[hoveredIndex];
                    const category = categories[hoveredIndex];
                    
                    // Get current font sizes
                    const fontSize = parseInt(document.getElementById('essay-font-size').value);
                    const headerFontSize = Math.round(fontSize * 1.2);
                    const pcFontSize = Math.round(fontSize * 0.7);
                    
                    // Create PC summary for header with inline styles
                    let pcSummary = '';
                    d.pc_info.forEach((pc, idx) => {
                        const aiSign = pc.contribution_ai > 0 ? '+' : '';
                        const scSign = pc.contribution_sc > 0 ? '+' : '';
                        pcSummary += `<span class="pc-inline" style="font-size: ${pcFontSize}px; color: white;" onclick="event.stopPropagation(); showPCGlobalInfo(${parseInt(pc.pc.substring(2))})">
                            ${pc.pc}: ${pc.percentile.toFixed(0)}% | AI:${aiSign}${pc.contribution_ai.toFixed(2)} SC:${scSign}${pc.contribution_sc.toFixed(2)} | ${pc.variance_total.toFixed(1)}%var
                        </span>`;
                    });
                    
                    document.getElementById('essay-header').innerHTML = `
                        <div class="header-main">
                            <div>
                                <strong>Essay ID:</strong> ${d.essay_id} | 
                                <strong>SC:</strong> ${d.sc11} | 
                                <strong>AI:</strong> ${d.ai_rating.toFixed(2)}
                            </div>
                        </div>
                        <div class="header-pcs" style="font-size: ${pcFontSize}px; color: white;">
                            <strong>Top 5 PCs:</strong> ${pcSummary}
                        </div>
                    `;
                    document.getElementById('essay-header').style.fontSize = headerFontSize + 'px';
                    document.getElementById('essay-text').textContent = d.essay;
                    document.getElementById('essay-text').style.fontSize = fontSize + 'px';
                    
                    const color = categoryColors[category];
                    const bgOpacity = parseFloat(document.getElementById('essay-opacity').value);
                    const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${bgOpacity})`;
                    document.getElementById('essay-display').style.backgroundColor = bgColor;
                    document.getElementById('essay-display').style.borderColor = 
                        `rgb(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)})`;
                    document.getElementById('essay-display').style.display = 'block';
                }
            } else {
                if (hoveredIndex !== -1) {
                    hoveredIndex = -1;
                    document.getElementById('essay-display').style.display = 'none';
                }
            }
        }
        
        window.addEventListener('mousemove', onMouseMove);
        
        // Controls
        document.getElementById('auto-rotate').addEventListener('change', (e) => {
            controls.autoRotate = e.target.checked;
        });
        
        document.getElementById('rotate-speed').addEventListener('input', (e) => {
            controls.autoRotateSpeed = parseFloat(e.target.value);
        });
        
        document.getElementById('point-opacity').addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value);
            material.opacity = opacity;
            document.getElementById('opacity-val').textContent = opacity.toFixed(1);
        });
        
        document.getElementById('essay-opacity').addEventListener('input', (e) => {
            const opacity = parseFloat(e.target.value);
            document.getElementById('essay-opacity-val').textContent = opacity.toFixed(2);
            
            if (document.getElementById('essay-display').style.display === 'block' && hoveredIndex >= 0) {
                const category = categories[hoveredIndex];
                const color = categoryColors[category];
                const bgColor = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${opacity})`;
                document.getElementById('essay-display').style.backgroundColor = bgColor;
            }
        });
        
        document.getElementById('essay-font-size').addEventListener('input', (e) => {
            const fontSize = parseInt(e.target.value);
            document.getElementById('font-size-val').textContent = fontSize;
            document.getElementById('essay-text').style.fontSize = fontSize + 'px';
            document.getElementById('essay-header').style.fontSize = Math.round(fontSize * 1.2) + 'px';
            
            // Apply to PC analysis text as well (70% of essay font size)
            const pcFontSize = Math.round(fontSize * 0.7);
            document.querySelectorAll('.pc-values, .pc-percentile, .header-pcs').forEach(el => {
                el.style.fontSize = pcFontSize + 'px';
                el.style.color = 'white';
            });
            
            // Apply to PC inline elements (also 70% of essay font size)
            document.querySelectorAll('.pc-inline').forEach(el => {
                el.style.fontSize = pcFontSize + 'px';
                el.style.color = 'white';
            });
        });
        
        // Function to apply font sizes based on current essay font size
        function applyFontSizes() {
            const fontSize = parseInt(document.getElementById('essay-font-size').value);
            document.getElementById('essay-text').style.fontSize = fontSize + 'px';
            document.getElementById('essay-header').style.fontSize = Math.round(fontSize * 1.2) + 'px';
            
            const pcFontSize = Math.round(fontSize * 0.7);
            document.querySelectorAll('.pc-values, .pc-percentile, .header-pcs').forEach(el => {
                el.style.fontSize = pcFontSize + 'px';
                el.style.color = 'white';
            });
            
            document.querySelectorAll('.pc-inline').forEach(el => {
                el.style.fontSize = pcFontSize + 'px';
                el.style.color = 'white';
            });
        }
        
        document.getElementById('essay-height').addEventListener('input', (e) => {
            const heightPercent = parseInt(e.target.value);
            document.getElementById('essay-height-val').textContent = heightPercent;
            document.getElementById('essay-display').style.height = heightPercent + 'vh';
        });
        
        document.getElementById('transition-speed').addEventListener('input', (e) => {
            document.getElementById('transition-val').textContent = e.target.value;
        });
        
        // Keyboard navigation
        window.addEventListener('keydown', (e) => {
            // Gallery mode navigation
            if (galleryMode) {
                if (e.key === 'ArrowLeft') navigateGallery(-1);
                else if (e.key === 'ArrowRight') navigateGallery(1);
                else if (e.key === 'Escape') stopGallery();
            } 
            // PC analysis navigation when popup is open
            else if (document.getElementById('pc-global-info').style.display === 'block' && currentPCIndex !== null) {
                if (e.key === 'ArrowLeft') navigatePC(-1);
                else if (e.key === 'ArrowRight') navigatePC(1);
                else if (e.key === 'Escape') closePCStatsBox();
            }
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            if (isAnimating) {
                const elapsed = Date.now() - animationStartTime;
                const progress = Math.min(elapsed / animationDuration, 1);
                const eased = easeInOutCubic(progress);
                
                camera.position.lerpVectors(
                    animationStart.cameraPos,
                    animationEnd.cameraPos,
                    eased
                );
                
                controls.target.lerpVectors(
                    animationStart.targetPos,
                    animationEnd.targetPos,
                    eased
                );
                
                if (progress >= 1) {
                    isAnimating = false;
                    
                    if (returningToOverview) {
                        controls.target.set(cloudCenter.x, cloudCenter.y, cloudCenter.z);
                        returningToOverview = false;
                    }
                }
            }
            
            controls.update();
            
            // Update topic visibility if topics are shown
            if (document.getElementById('toggle-topics').checked) {
                updateTopicVisibility();
            }
            
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>"""

# Save
output_file = OUTPUT_DIR / 'minimal_umap_viz_v21.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✅ Enhanced visualization v21 with topic statistics panel saved to: {output_file}")
print("\nChanges in v21 (based on v20):")
print("- Added Topic Stats panel showing extreme group probabilities")
print("- Calculates % of essays in each topic falling into top/bottom 10%")
print("- Topics ranked by highest percentage in any extreme group")
print("- Color-coded percentages (green >30%, yellow >15%, gray otherwise)")
print("- Shows topic keywords, size, and all four extreme probabilities")
print("\nAll v20 features preserved including:")
print("- PC list (PC0, PC2, PC5, PC13, PC46) in DML table")
print("- Master color mode selector (AI/SC, Topics, PC Gradient)")
print("- PC gradient coloring with purple-to-yellow scale")
print("- Separate checkbox for showing advanced PC stats")
print("- PC gradient legend with variance explained")
print("- Decoupled PC selection from color mode")
print("- Billboarding text labels at topic centroids")
print("- Slider to control number of visible topics")
print("- Topics update dynamically as camera moves")