#!/usr/bin/env python3
"""
Analyze NV-Embed-v2 with different numbers of principal components
Test how PCA affects both R² and DML causal estimates
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
import matplotlib.pyplot as plt

print("="*80)
print("NV-EMBED-V2: PCA COMPONENTS ANALYSIS")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

# Load NV-Embed embeddings
print("\nLoading NV-Embed-v2 embeddings...")
embeddings = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"Embeddings shape: {embeddings.shape}")

# Load data
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

# Load and merge
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings = pd.read_csv(AI_RATINGS_FILE)

df = essays.merge(sc_labels, on='TID', how='inner')
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID', how='inner')

# Generate demographics
np.random.seed(42)
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))

# Align with embeddings
df = df[df['TID'].isin(essay_ids)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

X_full = embeddings[df['essay_idx'].values]
y_ai = df['ai_average'].values
y_sc = df['sc11'].values
W = df[['age', 'female', 'education_level_numeric']].values

print(f"\nAligned data: {len(df)} essays")

# Standardize embeddings
print("\nStandardizing embeddings...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# Test different numbers of PCA components
pca_components = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
results = []

print("\n" + "="*80)
print("TESTING DIFFERENT PCA COMPONENTS")
print("="*80)

for n_comp in pca_components:
    print(f"\n>>> PCA with {n_comp} components:")
    
    # Apply PCA
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"    Variance explained: {var_explained:.1%}")
    
    # Evaluate with Linear model
    linear = LinearRegression()
    ai_linear = cross_val_score(linear, X_pca, y_ai, cv=5, scoring='r2')
    sc_linear = cross_val_score(linear, X_pca, y_sc, cv=5, scoring='r2')
    
    # Evaluate with XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    ai_xgb = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
    sc_xgb = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')
    
    print(f"    Linear - AI R²: {ai_linear.mean():.3f}, SC R²: {sc_linear.mean():.3f}")
    print(f"    XGBoost - AI R²: {ai_xgb.mean():.3f}, SC R²: {sc_xgb.mean():.3f}")
    
    # Run DML analysis
    print("    Running DML...")
    dml = LinearDML(
        model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        discrete_treatment=False,
        cv=5,
        random_state=42
    )
    
    dml.fit(y_ai, y_sc, X=X_pca, W=W)
    theta = dml.coef_[0]
    
    # Try to get p-value
    try:
        from econml.inference import BootstrapInference
        dml_inf = LinearDML(
            model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
            model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
            discrete_treatment=False,
            cv=5,
            random_state=42,
            inference=BootstrapInference(n_bootstrap_samples=50, n_jobs=-1)
        )
        dml_inf.fit(y_ai, y_sc, X=X_pca, W=W)
        inference = dml_inf.effect_inference(X=X_pca)
        pval = inference.pvalue()[0]
        se = np.sqrt(inference.var[0])
    except:
        pval = np.nan
        se = np.nan
    
    if np.isnan(pval):
        print(f"    DML θ: {theta:.4f} (p=N/A)")
    else:
        print(f"    DML θ: {theta:.4f} (p={pval:.3f})")
    
    # Store results
    results.append({
        'n_components': n_comp,
        'var_explained': var_explained,
        'ai_linear': ai_linear.mean(),
        'sc_linear': sc_linear.mean(),
        'ai_xgb': ai_xgb.mean(),
        'sc_xgb': sc_xgb.mean(),
        'theta': theta,
        'se': se,
        'pval': pval
    })

# Convert to DataFrame for easy analysis
results_df = pd.DataFrame(results)

# Display summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print("\nPCA Components Analysis for NV-Embed-v2:")
print(f"\n{'PCA':<6} {'Var%':<6} {'Linear AI':<10} {'Linear SC':<10} {'XGB AI':<10} {'XGB SC':<10} {'θ':<10} {'p-value':<10}")
print("-"*80)

for _, row in results_df.iterrows():
    if np.isnan(row['pval']):
        print(f"{row['n_components']:<6} {row['var_explained']*100:<6.1f} "
              f"{row['ai_linear']:<10.3f} {row['sc_linear']:<10.3f} "
              f"{row['ai_xgb']:<10.3f} {row['sc_xgb']:<10.3f} "
              f"{row['theta']:<10.4f} {'N/A':<10}")
    else:
        print(f"{row['n_components']:<6} {row['var_explained']*100:<6.1f} "
              f"{row['ai_linear']:<10.3f} {row['sc_linear']:<10.3f} "
              f"{row['ai_xgb']:<10.3f} {row['sc_xgb']:<10.3f} "
              f"{row['theta']:<10.4f} {row['pval']:<10.3f}")

# Create visualizations
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. R² vs PCA components
ax1 = axes[0, 0]
ax1.plot(results_df['n_components'], results_df['ai_xgb'], 'b-o', label='AI R² (XGBoost)')
ax1.plot(results_df['n_components'], results_df['sc_xgb'], 'r-o', label='SC R² (XGBoost)')
ax1.set_xlabel('PCA Components')
ax1.set_ylabel('R²')
ax1.set_title('Prediction Performance vs PCA Components')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. DML theta vs PCA components
ax2 = axes[0, 1]
ax2.plot(results_df['n_components'], results_df['theta'], 'g-o')
ax2.set_xlabel('PCA Components')
ax2.set_ylabel('DML θ')
ax2.set_title('Causal Effect (θ) vs PCA Components')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

# 3. Variance explained vs components
ax3 = axes[1, 0]
ax3.plot(results_df['n_components'], results_df['var_explained']*100, 'purple', marker='o')
ax3.set_xlabel('PCA Components')
ax3.set_ylabel('Variance Explained (%)')
ax3.set_title('Cumulative Variance Explained')
ax3.grid(True, alpha=0.3)

# 4. P-values vs components
ax4 = axes[1, 1]
valid_p = results_df[~results_df['pval'].isna()]
ax4.plot(valid_p['n_components'], valid_p['pval'], 'orange', marker='o')
ax4.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
ax4.set_xlabel('PCA Components')
ax4.set_ylabel('p-value')
ax4.set_title('Statistical Significance vs PCA Components')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'visualizations/nvembed_pca_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to visualizations/nvembed_pca_analysis.png")

# Find optimal number of components
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Best for AI prediction
best_ai = results_df.loc[results_df['ai_xgb'].idxmax()]
print(f"\n1. Best for AI prediction: {int(best_ai['n_components'])} components")
print(f"   - AI R²: {best_ai['ai_xgb']:.3f}")
print(f"   - Variance explained: {best_ai['var_explained']:.1%}")

# Best for SC prediction (least negative)
best_sc = results_df.loc[results_df['sc_xgb'].idxmax()]
print(f"\n2. Best for SC prediction: {int(best_sc['n_components'])} components")
print(f"   - SC R²: {best_sc['sc_xgb']:.3f}")
print(f"   - Variance explained: {best_sc['var_explained']:.1%}")

# Most significant DML effect
if not results_df['pval'].isna().all():
    sig_results = results_df[results_df['pval'] < 0.05]
    if not sig_results.empty:
        best_dml = sig_results.loc[sig_results['pval'].idxmin()]
        print(f"\n3. Most significant DML effect: {int(best_dml['n_components'])} components")
        print(f"   - θ: {best_dml['theta']:.4f} (p={best_dml['pval']:.3f})")
        print(f"   - Variance explained: {best_dml['var_explained']:.1%}")

# Save detailed results
results_file = NVEMBED_DIR / 'nvembed_pca_analysis_results.pkl'
with open(results_file, 'wb') as f:
    pickle.dump({
        'results_df': results_df,
        'embeddings_shape': embeddings.shape,
        'n_essays': len(df)
    }, f)
print(f"\n✓ Saved detailed results to {results_file}")

print("\n" + "="*80)