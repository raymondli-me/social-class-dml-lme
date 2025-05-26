#!/usr/bin/env python3
"""
DML analysis for NV-Embed (200 PCA components) with multiple first-stage models
NO DEMOGRAPHICS - Only using text embeddings as controls
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
print("NV-EMBED DML ANALYSIS - NO DEMOGRAPHICS")
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

# Load targets
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

# NO DEMOGRAPHICS GENERATED - IMPORTANT CHANGE

# Align with embeddings
df = df[df['TID'].isin(essay_ids)]
print(f"\nAligned dataset: {len(df)} essays")

# Get variables for DML
Y = df['ai_average'].values  # Outcome: AI ratings
D = df['sc11'].values  # Treatment: actual social class
X = X_pca  # Controls: ONLY embeddings (no demographics)

print(f"\nDML setup (NO DEMOGRAPHICS):")
print(f"Y (AI ratings): mean={Y.mean():.2f}, std={Y.std():.2f}")
print(f"D (Social class): mean={D.mean():.2f}, std={D.std():.2f}")
print(f"X (Embeddings ONLY): {X.shape}")
print(f"W (Demographics): NOT USED")

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

# First evaluate prediction performance
print("\n" + "="*80)
print("EVALUATING PREDICTION PERFORMANCE (R²)")
print("="*80)

from sklearn.model_selection import cross_val_score

r2_results = {}
for model_name, model_config in models.items():
    print(f"\n{model_name}:")
    
    # Use the Y model for predictions
    model = model_config['model_y']
    
    # AI ratings prediction
    ai_scores = cross_val_score(model, X, Y, cv=5, scoring='r2', n_jobs=-1)
    ai_r2 = ai_scores.mean()
    ai_r2_std = ai_scores.std()
    
    # Social class prediction
    sc_scores = cross_val_score(model, X, D, cv=5, scoring='r2', n_jobs=-1)
    sc_r2 = sc_scores.mean()
    sc_r2_std = sc_scores.std()
    
    r2_results[model_name] = {
        'ai_r2': ai_r2,
        'ai_r2_std': ai_r2_std,
        'sc_r2': sc_r2,
        'sc_r2_std': sc_r2_std
    }
    
    print(f"  AI ratings R²: {ai_r2:.4f} (±{ai_r2_std:.4f})")
    print(f"  Social class R²: {sc_r2:.4f} (±{sc_r2_std:.4f})")

# Run DML for each model
results = {}

print("\n" + "="*80)
print("RUNNING DML WITH DIFFERENT FIRST-STAGE MODELS")
print("CONTROLS: TEXT EMBEDDINGS ONLY (NO DEMOGRAPHICS)")
print("="*80)

for model_name, model_config in models.items():
    print(f"\n>>> {model_name} First Stage:")
    
    try:
        # Create DML estimator - NO W parameter
        dml = LinearDML(
            model_y=model_config['model_y'],
            model_t=model_config['model_t'],
            discrete_treatment=False,
            cv=5,  # 5-fold cross-fitting
            random_state=42
        )
        
        # Fit the model - NO W parameter
        print("    Fitting DML (embeddings only)...")
        dml.fit(Y, D, X=X, W=None)  # W=None explicitly
        
        # Get coefficient
        theta = dml.coef_[0]
        
        # Get inference results
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

# Create comprehensive summary table
print("\n" + "="*120)
print("COMPREHENSIVE RESULTS TABLE (NO DEMOGRAPHICS)")
print("="*120)

# Combine R² and DML results
results_df = pd.DataFrame(results).T
r2_df = pd.DataFrame(r2_results).T

print("\nNV-Embed-v2 (200 PCA) Results:")
print(f"\n{'Model':<12} | {'AI R²':<8} {'SC R²':<8} | {'θ':<10} {'SE':<10} {'t-stat':<8} {'p-value':<10} {'95% CI':<22} {'Sig.':<5}")
print("-"*120)

for model_name in models.keys():
    # Get R² values
    ai_r2 = r2_df.loc[model_name, 'ai_r2']
    sc_r2 = r2_df.loc[model_name, 'sc_r2']
    
    # Get DML results
    theta = results_df.loc[model_name, 'theta']
    se = results_df.loc[model_name, 'se']
    t_stat = results_df.loc[model_name, 't_stat']
    p_val = results_df.loc[model_name, 'p_value']
    ci_lower = results_df.loc[model_name, 'ci_lower']
    ci_upper = results_df.loc[model_name, 'ci_upper']
    sig = results_df.loc[model_name, 'significant']
    
    ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    sig_str = "Yes" if sig else "No"
    
    print(f"{model_name:<12} | {ai_r2:<8.4f} {sc_r2:<8.4f} | {theta:<10.6f} {se:<10.6f} {t_stat:<8.3f} {p_val:<10.4f} {ci_str:<22} {sig_str:<5}")

# Additional analysis
print("\n" + "="*80)
print("INTERPRETATION (WITHOUT DEMOGRAPHICS)")
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
    print(f"after controlling for essay content (text embeddings only).")
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
output_file = NVEMBED_DIR / 'nvembed_dml_multimodel_no_demographics.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({
        'dml_results': results_df,
        'r2_results': r2_df,
        'raw_dml_results': results,
        'raw_r2_results': r2_results,
        'data_info': {
            'n_essays': len(df),
            'pca_components': 200,
            'variance_explained': pca_data['explained_variance_ratio'].sum(),
            'demographics_used': False,
            'note': 'No demographics - only text embeddings as controls'
        }
    }, f)

print(f"\n✓ Results saved to {output_file}")

print("\n" + "="*80)
print("IMPORTANT NOTE")
print("="*80)
print("This analysis uses ONLY text embeddings as controls (X).")
print("No demographic variables (age, gender, education) are included.")
print("This provides a cleaner estimate of the SC→AI effect controlling")
print("only for essay content, without confounding from fake demographics.")
print("="*80)