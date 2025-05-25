#!/usr/bin/env python3
"""
Simplified DML analysis - uses simpler features instead of full embeddings
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Simple text features
from sklearn.feature_extraction.text import TfidfVectorizer

print("Starting Simplified DML Social Class Analysis...")
print("="*50)

# Load data
print("Loading data...")
essays = pd.read_csv('/home/raymondli/social-class-dml-lme/data/asc_9513_essays.csv')
sc11_data = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
ai_standard = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_ladder_standard_improved_20250524_165833.csv')
ai_human = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_human_macarthur_ladder_improved_20250524_174149.csv')

# Merge data
data = essays.merge(sc11_data, on='TID')
data = data.merge(ai_standard[['essay_id', 'rating']], left_on='TID', right_on='essay_id')
data = data.rename(columns={'rating': 'ai_standard'})
data = data.merge(ai_human[['essay_id', 'rating']], left_on='TID', right_on='essay_id')
data = data.rename(columns={'rating': 'ai_human'})
data['ai_average'] = (data['ai_standard'] + data['ai_human']) / 2

print(f"Loaded {len(data)} essays")

# Extract simple features using TF-IDF
print("\nExtracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)
tfidf_features = vectorizer.fit_transform(data['original']).toarray()
print(f"TF-IDF features shape: {tfidf_features.shape}")

# Apply PCA
print("\nApplying PCA...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(tfidf_features)
pca = PCA(n_components=200, random_state=42)
X = pca.fit_transform(features_scaled)
print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.1%}")

# DML Analysis Function
def run_dml(X, y, method='linear', n_folds=5):
    """Simple DML implementation"""
    n = len(y)
    
    # For simplicity, use first PC as treatment, rest as confounders
    D = X[:, 0]
    W = X[:, 1:]
    
    # Cross-fitting
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    res_y = np.zeros(n)
    res_d = np.zeros(n)
    
    for train_idx, test_idx in kf.split(X):
        # Split data
        W_train, W_test = W[train_idx], W[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # First stage regressions
        if method == 'linear':
            model_y = LinearRegression()
            model_d = LinearRegression()
        elif method == 'lasso':
            model_y = Lasso(alpha=0.1)
            model_d = Lasso(alpha=0.1)
        elif method == 'rf':
            model_y = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model_d = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        # Fit models and get residuals
        model_y.fit(W_train, y_train)
        res_y[test_idx] = y_test - model_y.predict(W_test)
        
        model_d.fit(W_train, D_train)
        res_d[test_idx] = D_test - model_d.predict(W_test)
    
    # Second stage
    theta = np.sum(res_y * res_d) / np.sum(res_d ** 2)
    
    # Standard error
    sigma2 = np.mean((res_y - theta * res_d) ** 2)
    se = np.sqrt(sigma2 / np.sum(res_d ** 2))
    
    # Also calculate simple R² for the target
    if method == 'linear':
        full_model = LinearRegression()
    elif method == 'lasso':
        full_model = Lasso(alpha=0.1)
    elif method == 'rf':
        full_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    full_model.fit(X, y)
    y_pred = full_model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return {
        'theta': theta,
        'se': se,
        't_stat': theta / se,
        'p_value': 2 * (1 - stats.t.cdf(abs(theta / se), df=n-2)),
        'r2': r2
    }

# Run analysis for all targets and methods
print("\nRunning DML analysis...")
results = []
targets = ['sc11', 'ai_standard', 'ai_human', 'ai_average']
methods = ['linear', 'lasso', 'rf']

for target in targets:
    print(f"\nTarget: {target}")
    y = data[target].values
    
    for method in methods:
        result = run_dml(X, y, method=method)
        result['target'] = target
        result['method'] = method
        results.append(result)
        print(f"  {method}: R²={result['r2']:.3f}, θ={result['theta']:.3f} (p={result['p_value']:.3f})")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. R² comparison
ax1 = axes[0]
pivot_r2 = results_df.pivot(index='method', columns='target', values='r2')
sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
ax1.set_title('R² Scores by Method and Target')

# 2. Coefficient estimates
ax2 = axes[1]
for target in targets:
    subset = results_df[results_df['target'] == target]
    ax2.errorbar(subset['theta'], range(len(subset)), xerr=1.96*subset['se'], 
                fmt='o', label=target, capsize=5)
ax2.set_yticks(range(len(methods)))
ax2.set_yticklabels(methods)
ax2.set_xlabel('DML Coefficient (θ)')
ax2.set_title('DML Coefficients with 95% CI')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/raymondli/social-class-dml-lme/dml_simple_results.png', dpi=300)
print("\nVisualization saved!")

# Save results
results_df.to_csv('/home/raymondli/social-class-dml-lme/dml_simple_results.csv', index=False)

# Print summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"\nBest R² by target:")
for target in targets:
    best = results_df[results_df['target'] == target].nlargest(1, 'r2').iloc[0]
    print(f"{target}: {best['method']} (R²={best['r2']:.3f})")

print("\nAnalysis complete!")