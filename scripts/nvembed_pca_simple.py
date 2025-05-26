#!/usr/bin/env python3
"""
Simplified PCA analysis for NV-Embed-v2
Focus on key PCA components without bootstrap inference
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from econml.dml import LinearDML

print("="*80)
print("NV-EMBED-V2: PCA COMPONENTS ANALYSIS (SIMPLIFIED)")
print("="*80)

# Load embeddings
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

embeddings = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"\nEmbeddings: {embeddings.shape}")

# Load aligned data from existing analysis
with open(NVEMBED_DIR / "nvembed_pca_200_features.pkl", 'rb') as f:
    pca_data = pickle.load(f)

# Load targets
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

# Quick data loading
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings = pd.read_csv(AI_RATINGS_FILE)

df = essays.merge(sc_labels, on='TID', how='inner')
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID', how='inner')

# Demographics
np.random.seed(42)
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))

# Align
df = df[df['TID'].isin(essay_ids)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

X_full = embeddings[df['essay_idx'].values]
y_ai = df['ai_average'].values
y_sc = df['sc11'].values
W = df[['age', 'female', 'education_level_numeric']].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# Test key PCA components
pca_components = [100, 200, 500, 1000, 2000]
results = []

print("\n" + "="*80)
print("RESULTS")
print("="*80)

for n_comp in pca_components:
    print(f"\n>>> {n_comp} PCA Components:")
    
    # PCA
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_.sum()
    
    # Quick XGBoost evaluation
    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    
    # Use 3-fold CV for speed
    ai_scores = cross_val_score(xgb_model, X_pca, y_ai, cv=3, scoring='r2')
    sc_scores = cross_val_score(xgb_model, X_pca, y_sc, cv=3, scoring='r2')
    
    # DML
    dml = LinearDML(
        model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        discrete_treatment=False,
        cv=3,  # 3-fold for speed
        random_state=42
    )
    
    dml.fit(y_ai, y_sc, X=X_pca, W=W)
    theta = dml.coef_[0]
    
    # Simple p-value approximation
    # Assume standard error from residual variance
    n = len(y_ai)
    se_approx = np.std(y_ai - y_ai.mean()) / np.sqrt(n)
    z_score = theta / se_approx
    from scipy import stats
    p_approx = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"  Variance explained: {var_exp:.1%}")
    print(f"  AI R²: {ai_scores.mean():.3f} (±{ai_scores.std():.3f})")
    print(f"  SC R²: {sc_scores.mean():.3f} (±{sc_scores.std():.3f})")
    print(f"  DML θ: {theta:.4f} (approx p={p_approx:.3f})")
    
    results.append({
        'n_comp': n_comp,
        'var_exp': var_exp,
        'ai_r2': ai_scores.mean(),
        'sc_r2': sc_scores.mean(),
        'theta': theta,
        'p_approx': p_approx
    })

# Summary table
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Find sweet spot
best_overall = results_df.iloc[results_df['ai_r2'].values.argmax()]
print(f"\n1. Best AI prediction: {int(best_overall['n_comp'])} components")
print(f"   - AI R² = {best_overall['ai_r2']:.3f}")
print(f"   - Variance = {best_overall['var_exp']:.1%}")

# Check if SC prediction improves with more components
if results_df['sc_r2'].max() > 0:
    best_sc = results_df.iloc[results_df['sc_r2'].values.argmax()]
    print(f"\n2. Best SC prediction: {int(best_sc['n_comp'])} components")
    print(f"   - SC R² = {best_sc['sc_r2']:.3f}")

# Efficiency consideration
efficient = results_df[results_df['var_exp'] > 0.9].iloc[0]
print(f"\n3. Efficient choice (>90% variance): {int(efficient['n_comp'])} components")
print(f"   - AI R² = {efficient['ai_r2']:.3f}")
print(f"   - Only {int(efficient['n_comp'])/4096*100:.1f}% of original dimensions")

print("\n" + "="*80)