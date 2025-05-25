#!/usr/bin/env python3
"""
Quick DML analysis using subset of data for demonstration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("Quick DML Analysis (subset of data)")
print("="*50)

# Load data
print("Loading data...")
essays = pd.read_csv('/home/raymondli/social-class-dml-lme/data/asc_9513_essays.csv')
sc11_data = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
ai_standard = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_ladder_standard_improved_20250524_165833.csv')

# Use first 1000 essays for quick demo
essays = essays.head(1000)
sc11_data = sc11_data[sc11_data['TID'].isin(essays['TID'])]
ai_standard = ai_standard[ai_standard['essay_id'].isin(essays['TID'])]

# Merge
data = essays.merge(sc11_data, on='TID')
data = data.merge(ai_standard[['essay_id', 'rating']], left_on='TID', right_on='essay_id')
print(f"Using {len(data)} essays for quick analysis")

# Simple features: essay length and basic statistics
print("\nExtracting simple features...")
data['essay_length'] = data['original'].str.len()
data['word_count'] = data['original'].str.split().str.len()
data['avg_word_length'] = data['original'].str.replace('[^a-zA-Z ]', '').str.split().apply(
    lambda x: np.mean([len(word) for word in x]) if x else 0
)
data['sentence_count'] = data['original'].str.count('[.!?]') + 1

# Create feature matrix
features = ['essay_length', 'word_count', 'avg_word_length', 'sentence_count']
X = data[features].values
y_sc11 = data['sc11'].values
y_ai = data['rating'].values

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Feature matrix shape: {X.shape}")

# Simple DML implementation
def simple_dml(X, y, n_folds=3):
    """Very simple DML with linear models only"""
    n = len(y)
    D = X[:, 0]  # First feature as "treatment"
    W = X[:, 1:]  # Rest as confounders
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    res_y = np.zeros(n)
    res_d = np.zeros(n)
    
    for train_idx, test_idx in kf.split(X):
        # Fit linear models
        model_y = LinearRegression()
        model_d = LinearRegression()
        
        model_y.fit(W[train_idx], y[train_idx])
        res_y[test_idx] = y[test_idx] - model_y.predict(W[test_idx])
        
        model_d.fit(W[train_idx], D[train_idx])
        res_d[test_idx] = D[test_idx] - model_d.predict(W[test_idx])
    
    # Second stage
    theta = np.sum(res_y * res_d) / np.sum(res_d ** 2)
    se = np.sqrt(np.mean((res_y - theta * res_d) ** 2) / np.sum(res_d ** 2))
    
    # Simple R²
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return {
        'theta': theta,
        'se': se,
        'p_value': 2 * (1 - stats.t.cdf(abs(theta / se), df=n-2)),
        'r2': r2
    }

# Run analysis
print("\nRunning DML...")
result_sc11 = simple_dml(X, y_sc11)
result_ai = simple_dml(X, y_ai)

print(f"\nResults for actual social class (sc11):")
print(f"  R² = {result_sc11['r2']:.3f}")
print(f"  θ = {result_sc11['theta']:.3f} (SE = {result_sc11['se']:.3f})")
print(f"  p-value = {result_sc11['p_value']:.3f}")

print(f"\nResults for AI rating:")
print(f"  R² = {result_ai['r2']:.3f}")
print(f"  θ = {result_ai['theta']:.3f} (SE = {result_ai['se']:.3f})")
print(f"  p-value = {result_ai['p_value']:.3f}")

# Simple visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Feature correlations with targets
ax1 = axes[0]
corr_sc11 = [np.corrcoef(X[:, i], y_sc11)[0, 1] for i in range(X.shape[1])]
corr_ai = [np.corrcoef(X[:, i], y_ai)[0, 1] for i in range(X.shape[1])]
x_pos = np.arange(len(features))
width = 0.35
ax1.bar(x_pos - width/2, corr_sc11, width, label='sc11')
ax1.bar(x_pos + width/2, corr_ai, width, label='AI rating')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(features, rotation=45)
ax1.set_ylabel('Correlation')
ax1.set_title('Feature Correlations with Targets')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot
ax2 = axes[1]
ax2.scatter(y_sc11, y_ai, alpha=0.5)
ax2.set_xlabel('Actual Social Class (sc11)')
ax2.set_ylabel('AI Rating')
ax2.set_title(f'AI vs Actual (r = {np.corrcoef(y_sc11, y_ai)[0,1]:.3f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/raymondli/social-class-dml-lme/dml_quick_results.png', dpi=300)

print("\nAnalysis complete! Results saved to dml_quick_results.png")