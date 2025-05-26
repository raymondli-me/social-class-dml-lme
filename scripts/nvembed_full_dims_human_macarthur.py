#!/usr/bin/env python3
"""
DML analysis using FULL NV-Embed dimensions (4,096) vs PCA (200)
USING ONLY HUMAN MACARTHUR RATINGS (not ladder standard)
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
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NV-EMBED ANALYSIS: FULL DIMENSIONS (4,096) vs PCA (200)")
print("USING ONLY HUMAN MACARTHUR RATINGS")
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
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts" / "run_20250524_162055" / "all_results_9513x2_20250524_174149.csv"

# Load essays and labels
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings_df = pd.read_csv(AI_RATINGS_FILE)

print(f"✓ Essays loaded: {len(essays)}")
print(f"✓ Social class labels loaded: {len(sc_labels)}")
print(f"✓ AI ratings file loaded: {ai_ratings_df.shape}")

# Filter for ONLY human MacArthur ratings
print("\nFiltering for human MacArthur ratings only...")
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
print(f"✓ Human MacArthur ratings: {len(human_mac_ratings)}")

# Average ratings by essay
ai_ratings_avg = human_mac_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_ratings_avg.columns = ['TID', 'ai_average']

# Check rating distribution
print(f"\nHuman MacArthur ratings statistics:")
print(f"  Mean: {ai_ratings_avg['ai_average'].mean():.2f}")
print(f"  Std: {ai_ratings_avg['ai_average'].std():.2f}")
print(f"  Min: {ai_ratings_avg['ai_average'].min():.2f}")
print(f"  Max: {ai_ratings_avg['ai_average'].max():.2f}")

# Merge data
df = essays.merge(sc_labels, on='TID', how='inner')
df = df.merge(ai_ratings_avg, on='TID', how='inner')

# Align with embeddings
df = df[df['TID'].isin(essay_ids)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

print(f"\n✓ Final aligned dataset: {len(df)} essays")

# Get variables
X_full_raw = embeddings_full[df['essay_idx'].values]
Y = df['ai_average'].values  # Human MacArthur AI ratings
D = df['sc11'].values  # Social class

print(f"\nData shapes:")
print(f"X (full embeddings): {X_full_raw.shape}")
print(f"Y (human MacArthur ratings): {Y.shape}, mean={Y.mean():.2f}, std={Y.std():.2f}")
print(f"D (social class): {D.shape}, mean={D.mean():.2f}, std={D.std():.2f}")

# Standardize full embeddings
print("\nStandardizing full embeddings...")
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X_full_raw)
print("✓ Full embeddings standardized")

# Create PCA embeddings
print("\nCreating PCA embeddings (200 components)...")
pca = PCA(n_components=200, random_state=42)
X_pca_raw = pca.fit_transform(X_full_raw)
variance_explained = pca.explained_variance_ratio_.sum()
print(f"✓ PCA variance explained: {variance_explained:.1%}")

# Standardize PCA embeddings
scaler_pca = StandardScaler()
X_pca = scaler_pca.fit_transform(X_pca_raw)
print("✓ PCA embeddings standardized")

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

# Store results
results = {
    'full_dims': {'r2': {}, 'dml': {}},
    'pca_200': {'r2': {}, 'dml': {}}
}

# Function to run analysis
def run_analysis(X, X_name, results_key):
    print("\n" + "="*80)
    print(f"ANALYZING {X_name}")
    print("="*80)
    
    # Prediction performance
    print("\nEvaluating prediction performance...")
    
    for model_name, model_config in models.items():
        print(f"\n{model_name}:")
        
        model = model_config['model_y']
        
        # Time the evaluation
        start_time = time.time()
        
        # Use 5-fold CV
        ai_scores = cross_val_score(model, X, Y, cv=5, scoring='r2', n_jobs=-1)
        ai_r2 = ai_scores.mean()
        ai_r2_std = ai_scores.std()
        
        sc_scores = cross_val_score(model, X, D, cv=5, scoring='r2', n_jobs=-1)
        sc_r2 = sc_scores.mean()
        sc_r2_std = sc_scores.std()
        
        elapsed = time.time() - start_time
        
        results[results_key]['r2'][model_name] = {
            'ai_r2': ai_r2,
            'ai_r2_std': ai_r2_std,
            'sc_r2': sc_r2,
            'sc_r2_std': sc_r2_std,
            'time': elapsed
        }
        
        print(f"  AI ratings R²: {ai_r2:.4f} (±{ai_r2_std:.4f})")
        print(f"  Social class R²: {sc_r2:.4f} (±{sc_r2_std:.4f})")
        print(f"  Time: {elapsed:.1f}s")
    
    # DML analysis
    print("\nRunning DML analysis...")
    
    for model_name, model_config in models.items():
        print(f"\n>>> {model_name} First Stage:")
        
        try:
            start_time = time.time()
            
            # Create DML estimator
            dml = LinearDML(
                model_y=model_config['model_y'],
                model_t=model_config['model_t'],
                discrete_treatment=False,
                cv=5,
                random_state=42
            )
            
            # Fit
            print(f"    Fitting DML with {X_name}...")
            dml.fit(Y, D, X=X, W=None)  # No demographics
            
            # Get results
            theta = dml.coef_[0]
            inference = dml.effect_inference(X=X)
            theta_se = np.sqrt(inference.var[0])
            
            ci = inference.conf_int(alpha=0.05)
            ci_lower = ci[0][0]
            ci_upper = ci[1][0]
            
            t_stat = theta / theta_se
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            elapsed = time.time() - start_time
            
            results[results_key]['dml'][model_name] = {
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
            results[results_key]['dml'][model_name] = None

# Run both analyses
run_analysis(X_full, "FULL 4,096 DIMENSIONS", 'full_dims')
run_analysis(X_pca, "PCA 200 DIMENSIONS", 'pca_200')

# Create comparison table
print("\n" + "="*120)
print("FINAL COMPARISON: FULL 4,096 DIMENSIONS vs PCA 200")
print("="*120)

print("\n1. PREDICTION PERFORMANCE (R²):")
print(f"\n{'Model':<12} | {'-------- FULL 4,096 --------':<30} | {'-------- PCA 200 -----------':<30} | {'-- DIFFERENCE --':<16}")
print(f"{'':12} | {'AI R²':<10} {'SC R²':<10} {'Time':<8} | {'AI R²':<10} {'SC R²':<10} {'Time':<8} | {'ΔAI R²':<8} {'ΔSC R²':<8}")
print("-"*120)

for model_name in models.keys():
    # Full dims
    full_ai = results['full_dims']['r2'][model_name]['ai_r2']
    full_sc = results['full_dims']['r2'][model_name]['sc_r2']
    full_time = results['full_dims']['r2'][model_name]['time']
    
    # PCA
    pca_ai = results['pca_200']['r2'][model_name]['ai_r2']
    pca_sc = results['pca_200']['r2'][model_name]['sc_r2']
    pca_time = results['pca_200']['r2'][model_name]['time']
    
    # Differences
    diff_ai = full_ai - pca_ai
    diff_sc = full_sc - pca_sc
    
    print(f"{model_name:<12} | {full_ai:<10.4f} {full_sc:<10.4f} {full_time:<8.1f} | "
          f"{pca_ai:<10.4f} {pca_sc:<10.4f} {pca_time:<8.1f} | "
          f"{diff_ai:+8.4f} {diff_sc:+8.4f}")

print("\n2. DML CAUSAL ESTIMATES:")
print(f"\n{'Model':<12} | {'-------- FULL 4,096 --------':<35} | {'-------- PCA 200 -----------':<35}")
print(f"{'':12} | {'θ':<10} {'SE':<10} {'p-value':<10} | {'θ':<10} {'SE':<10} {'p-value':<10}")
print("-"*100)

for model_name in models.keys():
    if results['full_dims']['dml'][model_name] is not None:
        full_theta = results['full_dims']['dml'][model_name]['theta']
        full_se = results['full_dims']['dml'][model_name]['se']
        full_p = results['full_dims']['dml'][model_name]['p_value']
    else:
        full_theta = full_se = full_p = np.nan
    
    if results['pca_200']['dml'][model_name] is not None:
        pca_theta = results['pca_200']['dml'][model_name]['theta']
        pca_se = results['pca_200']['dml'][model_name]['se']
        pca_p = results['pca_200']['dml'][model_name]['p_value']
    else:
        pca_theta = pca_se = pca_p = np.nan
    
    print(f"{model_name:<12} | {full_theta:<10.6f} {full_se:<10.6f} {full_p:<10.4f} | "
          f"{pca_theta:<10.6f} {pca_se:<10.6f} {pca_p:<10.4f}")

# Analysis summary
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Calculate average improvements
avg_ai_imp = np.mean([results['full_dims']['r2'][m]['ai_r2'] - results['pca_200']['r2'][m]['ai_r2'] 
                      for m in models.keys()])
avg_sc_imp = np.mean([results['full_dims']['r2'][m]['sc_r2'] - results['pca_200']['r2'][m]['sc_r2'] 
                      for m in models.keys()])

print(f"\n1. PREDICTION PERFORMANCE:")
print(f"   Average AI R² improvement with full dims: {avg_ai_imp:+.4f}")
print(f"   Average SC R² improvement with full dims: {avg_sc_imp:+.4f}")
print(f"   PCA variance retained: {variance_explained:.1%}")

if abs(avg_ai_imp) < 0.01 and abs(avg_sc_imp) < 0.01:
    print("   → Full dimensions provide minimal improvement over PCA")
elif avg_ai_imp > 0.05 or avg_sc_imp > 0.05:
    print("   → Full dimensions substantially improve predictions")
else:
    print("   → Full dimensions provide modest improvements")

# Check significance changes
sig_changes = []
for model in models.keys():
    if results['full_dims']['dml'][model] is not None and results['pca_200']['dml'][model] is not None:
        full_sig = results['full_dims']['dml'][model]['significant']
        pca_sig = results['pca_200']['dml'][model]['significant']
        if full_sig != pca_sig:
            sig_changes.append(model)

print(f"\n2. CAUSAL EFFECT DETECTION:")
if sig_changes:
    print(f"   Models with significance changes: {', '.join(sig_changes)}")
else:
    print("   No changes in statistical significance between approaches")

# Time comparison
avg_time_full = np.mean([results['full_dims']['r2'][m]['time'] for m in models.keys()])
avg_time_pca = np.mean([results['pca_200']['r2'][m]['time'] for m in models.keys()])
time_ratio = avg_time_full / avg_time_pca

print(f"\n3. COMPUTATIONAL EFFICIENCY:")
print(f"   Full dimensions: {avg_time_full:.1f}s average")
print(f"   PCA dimensions: {avg_time_pca:.1f}s average")
print(f"   Full dims are {time_ratio:.1f}x slower")

# Save results
output_file = NVEMBED_DIR / 'nvembed_human_macarthur_comparison.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({
        'results': results,
        'metadata': {
            'n_essays': len(df),
            'full_dims': 4096,
            'pca_dims': 200,
            'pca_variance_retained': variance_explained,
            'rating_type': 'human_macarthur_only'
        }
    }, f)

print(f"\n✓ Results saved to {output_file}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if abs(avg_ai_imp) < 0.02 and abs(avg_sc_imp) < 0.02:
    print("USE PCA 200:")
    print("- Minimal performance loss (<2% R²)")
    print(f"- {time_ratio:.1f}x faster computation")
    print("- Easier to work with and store")
    print("- No significant difference in causal estimates")
else:
    print("CONSIDER FULL DIMENSIONS IF:")
    print(f"- You need the extra {avg_ai_imp:.1%} AI prediction accuracy")
    print(f"- Computation time ({time_ratio:.1f}x slower) is acceptable")
    print("- Memory usage is not a constraint")

print("\n" + "="*80)