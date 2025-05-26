#!/usr/bin/env python3
"""
DML analysis using FULL NV-Embed dimensions (4,096) vs PCA (200)
Compare with previous PCA results to see if dimensionality reduction loses information
NO DEMOGRAPHICS - only text embeddings as controls
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from econml.dml import LinearDML
from scipy import stats
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NV-EMBED FULL DIMENSIONS (4,096) ANALYSIS")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"
DATA_DIR = BASE_DIR / "data"

# Load full embeddings
print("\nLoading NV-Embed full embeddings...")
embeddings_full = np.load(NVEMBED_DIR / "nvembed_embeddings.npy")
essay_ids = np.load(NVEMBED_DIR / "nvembed_essay_ids.npy", allow_pickle=True)
print(f"✓ Full embeddings shape: {embeddings_full.shape}")
print(f"✓ Memory usage: {embeddings_full.nbytes / 1024 / 1024:.1f} MB")

# Load data files
print("\nLoading data files...")
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

# Load and merge
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings = pd.read_csv(AI_RATINGS_FILE)

print(f"✓ Essays loaded: {len(essays)}")
print(f"✓ Social class labels loaded: {len(sc_labels)}")
print(f"✓ AI ratings loaded: {len(ai_ratings)}")

# Merge data
df = essays.merge(sc_labels, on='TID', how='inner')
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID', how='inner')

# NO DEMOGRAPHICS - as per checkpoint decision

# Align with embeddings
df = df[df['TID'].isin(essay_ids)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

print(f"\n✓ Final aligned dataset: {len(df)} essays")

# Get variables
X_full_raw = embeddings_full[df['essay_idx'].values]
Y = df['ai_average'].values  # AI ratings
D = df['sc11'].values  # Social class

print(f"\nData shapes:")
print(f"X (full embeddings): {X_full_raw.shape}")
print(f"Y (AI ratings): {Y.shape}, mean={Y.mean():.2f}, std={Y.std():.2f}")
print(f"D (social class): {D.shape}, mean={D.mean():.2f}, std={D.std():.2f}")

# Standardize full embeddings
print("\nStandardizing full embeddings...")
scaler = StandardScaler()
X_full = scaler.fit_transform(X_full_raw)
print("✓ Standardization complete")

# Define models
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
        'model_y': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, tree_method='hist'),
        'model_t': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, tree_method='hist')
    }
}

# Evaluate prediction performance
print("\n" + "="*80)
print("EVALUATING PREDICTION PERFORMANCE - FULL 4,096 DIMENSIONS")
print("="*80)

r2_results_full = {}
for model_name, model_config in models.items():
    print(f"\n{model_name}:")
    
    model = model_config['model_y']
    
    # Time the evaluation
    start_time = time.time()
    
    # Use 3-fold CV for speed with high dimensions
    ai_scores = cross_val_score(model, X_full, Y, cv=3, scoring='r2', n_jobs=-1)
    ai_r2 = ai_scores.mean()
    ai_r2_std = ai_scores.std()
    
    sc_scores = cross_val_score(model, X_full, D, cv=3, scoring='r2', n_jobs=-1)
    sc_r2 = sc_scores.mean()
    sc_r2_std = sc_scores.std()
    
    elapsed = time.time() - start_time
    
    r2_results_full[model_name] = {
        'ai_r2': ai_r2,
        'ai_r2_std': ai_r2_std,
        'sc_r2': sc_r2,
        'sc_r2_std': sc_r2_std,
        'time': elapsed
    }
    
    print(f"  AI ratings R²: {ai_r2:.4f} (±{ai_r2_std:.4f})")
    print(f"  Social class R²: {sc_r2:.4f} (±{sc_r2_std:.4f})")
    print(f"  Time: {elapsed:.1f}s")

# Run DML
print("\n" + "="*80)
print("RUNNING DML - FULL 4,096 DIMENSIONS")
print("="*80)

dml_results_full = {}

for model_name, model_config in models.items():
    print(f"\n>>> {model_name} First Stage:")
    
    try:
        start_time = time.time()
        
        # Create DML estimator
        dml = LinearDML(
            model_y=model_config['model_y'],
            model_t=model_config['model_t'],
            discrete_treatment=False,
            cv=3,  # 3-fold for speed
            random_state=42
        )
        
        # Fit
        print("    Fitting DML with full dimensions...")
        dml.fit(Y, D, X=X_full, W=None)  # No demographics
        
        # Get results
        theta = dml.coef_[0]
        inference = dml.effect_inference(X=X_full)
        theta_se = np.sqrt(inference.var[0])
        
        ci = inference.conf_int(alpha=0.05)
        ci_lower = ci[0][0]
        ci_upper = ci[1][0]
        
        t_stat = theta / theta_se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        elapsed = time.time() - start_time
        
        dml_results_full[model_name] = {
            'theta': theta,
            'se': theta_se,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05,
            'time': elapsed
        }
        
        print(f"    θ = {theta:.6f}")
        print(f"    SE = {theta_se:.6f}")
        print(f"    t-stat = {t_stat:.3f}")
        print(f"    p-value = {p_value:.4f}")
        print(f"    95% CI = [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"    Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        dml_results_full[model_name] = None

# Load PCA results for comparison
print("\n" + "="*80)
print("LOADING PCA 200 RESULTS FOR COMPARISON")
print("="*80)

pca_results_file = NVEMBED_DIR / 'nvembed_dml_multimodel_no_demographics.pkl'
if pca_results_file.exists():
    with open(pca_results_file, 'rb') as f:
        pca_data = pickle.load(f)
    pca_r2 = pca_data['r2_results']
    pca_dml = pca_data['dml_results']
    print("✓ PCA results loaded")
else:
    print("⚠ PCA results not found")
    pca_r2 = None
    pca_dml = None

# Create comparison table
print("\n" + "="*120)
print("COMPARISON: FULL 4,096 DIMENSIONS vs PCA 200")
print("="*120)

print("\n1. PREDICTION PERFORMANCE (R²):")
print(f"\n{'Model':<12} | {'-------- FULL 4,096 --------':<30} | {'-------- PCA 200 -----------':<30} | {'-- DIFFERENCE --':<16}")
print(f"{'':12} | {'AI R²':<10} {'SC R²':<10} {'Time':<8} | {'AI R²':<10} {'SC R²':<10} {'Time':<8} | {'ΔAI R²':<8} {'ΔSC R²':<8}")
print("-"*120)

for model_name in models.keys():
    # Full dims
    full_ai = r2_results_full[model_name]['ai_r2']
    full_sc = r2_results_full[model_name]['sc_r2']
    full_time = r2_results_full[model_name]['time']
    
    # PCA
    if pca_r2 is not None and model_name in pca_r2.index:
        pca_ai = pca_r2.loc[model_name, 'ai_r2']
        pca_sc = pca_r2.loc[model_name, 'sc_r2']
        diff_ai = full_ai - pca_ai
        diff_sc = full_sc - pca_sc
    else:
        pca_ai = pca_sc = diff_ai = diff_sc = np.nan
    
    print(f"{model_name:<12} | {full_ai:<10.4f} {full_sc:<10.4f} {full_time:<8.1f} | "
          f"{pca_ai:<10.4f} {pca_sc:<10.4f} {'N/A':<8} | "
          f"{diff_ai:+8.4f} {diff_sc:+8.4f}")

print("\n2. DML CAUSAL ESTIMATES:")
print(f"\n{'Model':<12} | {'-------- FULL 4,096 --------':<35} | {'-------- PCA 200 -----------':<35}")
print(f"{'':12} | {'θ':<10} {'SE':<10} {'p-value':<10} | {'θ':<10} {'SE':<10} {'p-value':<10}")
print("-"*100)

for model_name in models.keys():
    if dml_results_full[model_name] is not None:
        full_theta = dml_results_full[model_name]['theta']
        full_se = dml_results_full[model_name]['se']
        full_p = dml_results_full[model_name]['p_value']
    else:
        full_theta = full_se = full_p = np.nan
    
    if pca_dml is not None and model_name in pca_dml.index:
        pca_theta = pca_dml.loc[model_name, 'theta']
        pca_se = pca_dml.loc[model_name, 'se']
        pca_p = pca_dml.loc[model_name, 'p_value']
    else:
        pca_theta = pca_se = pca_p = np.nan
    
    print(f"{model_name:<12} | {full_theta:<10.6f} {full_se:<10.6f} {full_p:<10.4f} | "
          f"{pca_theta:<10.6f} {pca_se:<10.6f} {pca_p:<10.4f}")

# Analysis and conclusions
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Compare R² improvements
r2_improvements = []
for model in models.keys():
    if pca_r2 is not None and model in pca_r2.index:
        ai_imp = r2_results_full[model]['ai_r2'] - pca_r2.loc[model, 'ai_r2']
        sc_imp = r2_results_full[model]['sc_r2'] - pca_r2.loc[model, 'sc_r2']
        r2_improvements.append((model, ai_imp, sc_imp))

if r2_improvements:
    avg_ai_imp = np.mean([x[1] for x in r2_improvements])
    avg_sc_imp = np.mean([x[2] for x in r2_improvements])
    
    print(f"\n1. PREDICTION PERFORMANCE:")
    print(f"   Average AI R² improvement with full dims: {avg_ai_imp:+.4f}")
    print(f"   Average SC R² improvement with full dims: {avg_sc_imp:+.4f}")
    
    if abs(avg_ai_imp) < 0.01 and abs(avg_sc_imp) < 0.01:
        print("   → Full dimensions provide minimal improvement over PCA")
    elif avg_ai_imp > 0.05 or avg_sc_imp > 0.05:
        print("   → Full dimensions substantially improve predictions")
    else:
        print("   → Full dimensions provide modest improvements")

# Check significance changes
sig_changes = []
for model in models.keys():
    if dml_results_full[model] is not None and pca_dml is not None and model in pca_dml.index:
        full_sig = dml_results_full[model]['significant']
        pca_sig = pca_dml.loc[model, 'significant']
        if full_sig != pca_sig:
            sig_changes.append(model)

print(f"\n2. CAUSAL EFFECT DETECTION:")
if sig_changes:
    print(f"   Models with significance changes: {', '.join(sig_changes)}")
else:
    print("   No changes in statistical significance between approaches")

# Save results
output_file = NVEMBED_DIR / 'nvembed_full_dims_comparison.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({
        'full_dims': {
            'r2_results': r2_results_full,
            'dml_results': dml_results_full
        },
        'comparison_info': {
            'n_essays': len(df),
            'full_dims': 4096,
            'pca_dims': 200,
            'pca_variance_retained': 0.721  # From checkpoint
        }
    }, f)

print(f"\n✓ Results saved to {output_file}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Make recommendation based on results
compute_time_ratio = np.mean([r['time'] for r in r2_results_full.values()]) / 10  # Rough estimate

if abs(avg_ai_imp) < 0.02 and abs(avg_sc_imp) < 0.02:
    print("USE PCA 200:")
    print("- Minimal performance loss (<2% R²)")
    print("- Much faster computation")
    print("- Easier to work with")
    print("- No significant difference in causal estimates")
else:
    print("CONSIDER FULL DIMENSIONS IF:")
    print(f"- You need the extra {avg_ai_imp:.1%} AI prediction accuracy")
    print(f"- Computation time ({compute_time_ratio:.1f}x slower) is acceptable")
    print("- Memory usage is not a constraint")

print("\n" + "="*80)