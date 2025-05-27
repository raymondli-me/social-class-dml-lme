#!/usr/bin/env python3
"""
Fast version: Analyze only top 20 PCs instead of all 200
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from econml.dml import LinearDML
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme')
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'nvembed_checkpoints'
OUTPUT_DIR = BASE_DIR / 'nvembed_pc_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=== Fast PC Analysis (Top 20 only) ===")

# Load data
print("\n1. Loading data...")
essays_df = pd.read_csv(DATA_DIR / 'asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'}, inplace=True)

# Load social class
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'}, inplace=True)
essays_df = essays_df.merge(sc_df[['essay_id', 'sc11']], on='essay_id', how='left')

# Load AI ratings
ai_ratings_df = pd.read_csv(BASE_DIR / 'asc_analysis_2prompts' / 'run_20250524_162055' / 'all_results_9513x2_20250524_174149.csv')
human_mac_ratings = ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved'].copy()
essays_df = essays_df.merge(human_mac_ratings[['essay_id', 'rating']], on='essay_id', how='left')
essays_df.rename(columns={'rating': 'ai_rating'}, inplace=True)

# Load PCA features
with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    X_pca = pca_data['features']
    pca_model = pca_data['pca']
    essay_ids = pca_data['essay_ids']

# Align data
essays_df = essays_df[essays_df['essay_id'].isin(essay_ids)]
essays_df = essays_df.set_index('essay_id').loc[essay_ids].reset_index()
print(f"   Loaded {X_pca.shape[0]} essays with {X_pca.shape[1]} PCs")

Y_ai = essays_df['ai_rating'].values
Y_sc = essays_df['sc11'].values

# Analyze only top 20 PCs
print("\n2. Analyzing top 20 PCs by variance...")
n_pcs_to_analyze = 20
pc_results = []

for i in range(n_pcs_to_analyze):
    print(f"   Analyzing PC{i}...", end='', flush=True)
    X_single = X_pca[:, i:i+1]
    
    # Quick models
    model_ai = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=-1)
    y_pred_ai = cross_val_predict(model_ai, X_single, Y_ai, cv=3)
    r2_ai = r2_score(Y_ai, y_pred_ai)
    
    model_sc = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=-1)
    y_pred_sc = cross_val_predict(model_sc, X_single, Y_sc, cv=3)
    r2_sc = r2_score(Y_sc, y_pred_sc)
    
    pc_results.append({
        'pc': i,
        'r2_ai': r2_ai,
        'r2_sc': r2_sc,
        'variance_explained': pca_model.explained_variance_ratio_[i]
    })
    print(f" AI R²={r2_ai:.3f}, SC R²={r2_sc:.3f}")

pc_results_df = pd.DataFrame(pc_results)

# Get top 5 for each
top_ai_pcs = pc_results_df.nlargest(5, 'r2_ai')['pc'].values
top_sc_pcs = pc_results_df.nlargest(5, 'r2_sc')['pc'].values
top_pcs = sorted(set(top_ai_pcs) | set(top_sc_pcs))[:10]

print("\n3. Top PCs identified:")
print("   AI prediction:", top_ai_pcs)
print("   SC prediction:", top_sc_pcs)
print("   Combined top PCs:", top_pcs)

# Calculate DML for top PCs only
print("\n4. Computing DML effects for top PCs...")
for idx, pc in enumerate(top_pcs):
    X_single = X_pca[:, pc:pc+1]
    try:
        dml = LinearDML(
            model_y=xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
            model_t=xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
            random_state=42
        )
        dml.fit(Y=Y_ai, T=Y_sc, X=X_single, W=None)
        theta = dml.effect(X_single).mean()
        _, p_value = dml.effect_inference(X_single).population_summary()
        
        # Update the results
        pc_idx = pc_results_df[pc_results_df['pc'] == pc].index[0]
        pc_results_df.loc[pc_idx, 'dml_theta'] = theta
        pc_results_df.loc[pc_idx, 'dml_pvalue'] = p_value
        print(f"   PC{pc}: θ={theta:.4f}, p={p_value:.3f}")
    except:
        pass

# Calculate z-scores and percentiles
print("\n5. Computing z-scores and percentiles...")
pc_data = {}
for pc in top_pcs:
    pc_values = X_pca[:, pc]
    pc_data[f'pc{pc}_zscore'] = stats.zscore(pc_values)
    pc_data[f'pc{pc}_raw'] = pc_values
    pc_data[f'pc{pc}_percentile'] = stats.rankdata(pc_values, method='average') / len(pc_values) * 100

for key, values in pc_data.items():
    essays_df[key] = values

# Get SHAP values
print("\n6. Computing SHAP values...")
# Use only top PCs for SHAP to speed up
X_top_pcs = X_pca[:, top_pcs]

model_ai_top = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
model_ai_top.fit(X_top_pcs, Y_ai)

model_sc_top = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
model_sc_top.fit(X_top_pcs, Y_sc)

explainer_ai = shap.TreeExplainer(model_ai_top)
shap_values_ai = explainer_ai.shap_values(X_top_pcs)

explainer_sc = shap.TreeExplainer(model_sc_top)
shap_values_sc = explainer_sc.shap_values(X_top_pcs)

# Map SHAP values back to original PC indices
for i, pc in enumerate(top_pcs):
    essays_df[f'pc{pc}_shap_ai'] = shap_values_ai[:, i]
    essays_df[f'pc{pc}_shap_sc'] = shap_values_sc[:, i]

# Save results
print("\n7. Saving results...")
results = {
    'pc_results': pc_results_df,
    'top_pcs': top_pcs,
    'top_ai_pcs': top_ai_pcs,
    'top_sc_pcs': top_sc_pcs,
    'essays_with_pc_data': essays_df
}

with open(OUTPUT_DIR / 'pc_analysis_results_fast.pkl', 'wb') as f:
    pickle.dump(results, f)

pc_results_df.to_csv(OUTPUT_DIR / 'pc_importance_summary_fast.csv', index=False)

print("\n✅ Analysis complete!")
print(f"   Results saved to: {OUTPUT_DIR}")

# Print summary
print("\n=== Summary of Top PCs ===")
print("\nFor AI Rating Prediction:")
for pc in top_ai_pcs:
    row = pc_results_df[pc_results_df['pc'] == pc].iloc[0]
    print(f"   PC{pc}: R²={row['r2_ai']:.3f}, Var={row['variance_explained']*100:.1f}%")

print("\nFor Actual SC Prediction:")  
for pc in top_sc_pcs:
    row = pc_results_df[pc_results_df['pc'] == pc].iloc[0]
    print(f"   PC{pc}: R²={row['r2_sc']:.3f}, Var={row['variance_explained']*100:.1f}%")