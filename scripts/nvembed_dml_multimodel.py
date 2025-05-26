#!/usr/bin/env python3
"""
DML analysis for NV-Embed (200 PCA components) with multiple first-stage models
Get theta, SE, t-stat, p-value, and CI for each model
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from econml.dml import LinearDML
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NV-EMBED DML ANALYSIS WITH MULTIPLE MODELS")
print("="*80)

# Load NV-Embed PCA features (200 components)
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

print("\nLoading NV-Embed PCA features (200 components)...")
with open(NVEMBED_DIR / "nvembed_pca_200_features.pkl", 'rb') as f:
    pca_data = pickle.load(f)

X_pca = pca_data['features']
essay_ids = pca_data['essay_ids']
print(f"PCA features shape: {X_pca.shape}")
print(f"Variance explained: {pca_data['explained_variance_ratio'].sum():.1%}")

# Load targets and demographics
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

# Load and merge data
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
print(f"\nAligned dataset: {len(df)} essays")

# Get variables for DML
Y = df['ai_average'].values  # Outcome: AI ratings
D = df['sc11'].values  # Treatment: actual social class
X = X_pca  # High-dimensional controls: embeddings
W = df[['age', 'female', 'education_level_numeric']].values  # Low-dim controls

print(f"\nDML setup:")
print(f"Y (AI ratings): mean={Y.mean():.2f}, std={Y.std():.2f}")
print(f"D (Social class): mean={D.mean():.2f}, std={D.std():.2f}")
print(f"X (Embeddings): {X.shape}")
print(f"W (Demographics): {W.shape}")

# Define models to test
models = {
    'Linear': {
        'model_y': LinearRegression(),
        'model_t': LinearRegression()
    },
    'Ridge': {
        'model_y': Ridge(alpha=1.0, random_state=42),
        'model_t': Ridge(alpha=1.0, random_state=42)
    },
    'Lasso': {
        'model_y': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        'model_t': Lasso(alpha=0.1, random_state=42, max_iter=2000)
    },
    'RandomForest': {
        'model_y': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'model_t': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    },
    'XGBoost': {
        'model_y': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        'model_t': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    }
}

# Run DML for each model
results = {}

print("\n" + "="*80)
print("RUNNING DML WITH DIFFERENT FIRST-STAGE MODELS")
print("="*80)

for model_name, model_config in models.items():
    print(f"\n>>> {model_name} First Stage:")
    
    try:
        # Create DML estimator
        dml = LinearDML(
            model_y=model_config['model_y'],
            model_t=model_config['model_t'],
            discrete_treatment=False,
            cv=5,  # 5-fold cross-fitting
            random_state=42
        )
        
        # Fit the model
        print("    Fitting DML...")
        dml.fit(Y, D, X=X, W=W)
        
        # Get coefficient
        theta = dml.coef_[0]
        
        # Get inference results
        # Note: LinearDML uses asymptotic inference by default
        inference = dml.effect_inference(X=X)
        
        # Extract statistics
        theta_se = np.sqrt(inference.var[0])
        
        # Confidence interval
        ci = inference.conf_int(alpha=0.05)
        ci_lower = ci[0][0]
        ci_upper = ci[1][0]
        
        # T-statistic
        t_stat = theta / theta_se
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        # Store results
        results[model_name] = {
            'theta': theta,
            'se': theta_se,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05
        }
        
        print(f"    θ = {theta:.6f}")
        print(f"    SE = {theta_se:.6f}")
        print(f"    t-stat = {t_stat:.3f}")
        print(f"    p-value = {p_value:.4f}")
        print(f"    95% CI = [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"    Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
        
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        results[model_name] = {
            'theta': np.nan,
            'se': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'significant': False
        }

# Create summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

# Convert to DataFrame for nice display
results_df = pd.DataFrame(results).T
results_df = results_df.round(6)

print("\nDML Results (Y=AI ratings, D=Social class, X=NV-Embed PCA-200):")
print(f"\n{'Model':<15} {'θ':<10} {'SE':<10} {'t-stat':<10} {'p-value':<10} {'95% CI':<25} {'Sig.':<5}")
print("-"*90)

for model_name, row in results_df.iterrows():
    ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
    sig_str = "Yes" if row['significant'] else "No"
    print(f"{model_name:<15} {row['theta']:<10.6f} {row['se']:<10.6f} "
          f"{row['t_stat']:<10.3f} {row['p_value']:<10.4f} {ci_str:<25} {sig_str:<5}")

# Additional analysis
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

# Find significant effects
sig_models = results_df[results_df['significant']]
if not sig_models.empty:
    print(f"\nSignificant effects found with: {', '.join(sig_models.index)}")
    
    # Average effect across significant models
    avg_theta = sig_models['theta'].mean()
    print(f"\nAverage θ across significant models: {avg_theta:.6f}")
    print(f"This means: A 1-unit increase in actual social class is associated with")
    print(f"a {avg_theta:.4f} unit increase in AI-perceived social class rating,")
    print(f"after controlling for essay content.")
else:
    print("\nNo significant effects found with any model.")

# Check consistency
theta_range = results_df['theta'].max() - results_df['theta'].min()
print(f"\nConsistency check:")
print(f"Range of θ estimates: {theta_range:.6f}")
if theta_range < 0.01:
    print("Estimates are very consistent across models ✓")
elif theta_range < 0.05:
    print("Estimates show moderate variation across models")
else:
    print("Estimates vary substantially across models ⚠")

# Save results
output_file = NVEMBED_DIR / 'nvembed_dml_multimodel_results.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({
        'results_df': results_df,
        'raw_results': results,
        'data_info': {
            'n_essays': len(df),
            'pca_components': 200,
            'variance_explained': pca_data['explained_variance_ratio'].sum()
        }
    }, f)

print(f"\n✓ Results saved to {output_file}")

print("\n" + "="*80)
print("NOTES ON INFERENCE METHOD")
print("="*80)
print("- Standard errors computed using asymptotic theory (econml default)")
print("- Assumes √n(θ̂ - θ) → N(0, σ²) as n → ∞")
print("- Cross-fitting (5-fold) used to avoid overfitting bias")
print("- P-values based on normal approximation")
print("="*80)