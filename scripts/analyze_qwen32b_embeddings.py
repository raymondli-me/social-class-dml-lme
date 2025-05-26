#!/usr/bin/env python3
"""
DML Analysis for Qwen-32B AWQ Embeddings
Compare with OpenAI, NV-Embed, and MPNet results
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
import time

print("="*80)
print("QWEN-32B EMBEDDING DML ANALYSIS")
print("="*80)

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
QWEN_DIR = BASE_DIR / "qwen32b_embeddings"
OUTPUT_DIR = BASE_DIR / "qwen32b_checkpoints"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load embeddings
print("\n=== Loading Qwen-32B Embeddings ===")
embeddings = np.load(QWEN_DIR / "qwen32b_awq_embeddings.npy")
essay_ids_embed = np.load(QWEN_DIR / "qwen32b_awq_essay_ids.npy", allow_pickle=True)
print(f"✓ Embeddings shape: {embeddings.shape}")
print(f"✓ Dimensions: {embeddings.shape[1]}")

# Load data (same as other analyses)
print("\n=== Loading Data ===")
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"

# Load essays
essays = pd.read_csv(ESSAYS_FILE)
sc_labels = pd.read_csv(SC_LABELS_FILE)
ai_ratings = pd.read_csv(AI_RATINGS_FILE)

# Merge data
df = essays.merge(sc_labels, on='TID', how='inner')
ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
ai_avg.columns = ['TID', 'ai_average']
df = df.merge(ai_avg, on='TID', how='inner')

# Align with embeddings
df = df[df['TID'].isin(essay_ids_embed)]
df['essay_idx'] = df['TID'].apply(lambda x: np.where(essay_ids_embed == x)[0][0])
df = df.sort_values('essay_idx').reset_index(drop=True)

print(f"✓ Final dataset: {len(df)} essays with all data")

# Generate demographics (same seed as other analyses)
np.random.seed(42)
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))

# Get aligned embeddings
X_full = embeddings[df['essay_idx'].values]
y_ai = df['ai_average'].values
y_sc = df['sc11'].values

print(f"\n=== Data Shapes ===")
print(f"Embeddings: {X_full.shape}")
print(f"AI ratings: {y_ai.shape}")
print(f"Social class: {y_sc.shape}")

# Standardize embeddings
print("\n=== Standardizing Embeddings ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# PCA reduction
print("\n=== PCA Reduction ===")
n_components = 200
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum()
print(f"✓ Reduced to {n_components} components")
print(f"✓ Explained variance: {explained_var:.1%}")

# Save PCA features
pca_data = {
    'features': X_pca,
    'pca': pca,
    'essay_ids': df['TID'].values,
    'explained_variance_ratio': pca.explained_variance_ratio_
}
with open(OUTPUT_DIR / 'qwen32b_pca_200_features.pkl', 'wb') as f:
    pickle.dump(pca_data, f)

# Evaluate prediction performance
print("\n=== Evaluating Prediction Performance ===")

# Linear model
print("\nLINEAR Model:")
linear_model = LinearRegression()
scores_ai_linear = cross_val_score(linear_model, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_linear = cross_val_score(linear_model, X_pca, y_sc, cv=5, scoring='r2')
print(f"  ai_ratings: R² = {scores_ai_linear.mean():.3f}")
print(f"  actual_sc: R² = {scores_sc_linear.mean():.3f}")

# XGBoost model
print("\nXGBOOST Model:")
xgb_model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
scores_ai_xgb = cross_val_score(xgb_model, X_pca, y_ai, cv=5, scoring='r2')
scores_sc_xgb = cross_val_score(xgb_model, X_pca, y_sc, cv=5, scoring='r2')
print(f"  ai_ratings: R² = {scores_ai_xgb.mean():.3f}")
print(f"  actual_sc: R² = {scores_sc_xgb.mean():.3f}")

gap = (scores_ai_xgb.mean() - scores_sc_xgb.mean()) * 100
print(f"\nGap (AI - SC): {gap:.1f}%")

# DML Analysis
print("\n=== Running DML Analysis ===")
print("Running DML with XGBoost first stage...")

# Prepare data for DML
W = df[['age', 'female', 'education_level_numeric']].values  # Demographics
X = X_pca  # Embeddings
D = y_sc   # Treatment: actual social class
Y = y_ai   # Outcome: AI ratings

# Run DML
dml = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    discrete_treatment=False,
    cv=5,
    random_state=42
)

start_time = time.time()
dml.fit(Y, D, X=X, W=W)
dml_time = time.time() - start_time

# Get results
theta = dml.coef_[0]
inference = dml.effect_inference(X=X)  # Need to pass X for inference
theta_se = np.sqrt(inference.var[0])
conf_int = inference.conf_int()
ci_lower, ci_upper = conf_int[0][0], conf_int[1][0]
from scipy import stats
p_value = 2 * (1 - stats.norm.cdf(abs(theta / theta_se)))

# Compute partial R²
from sklearn.metrics import r2_score
dml.fit(Y, D, X=X, W=W)
Y_pred = dml.predict(Y, D, X=X, W=W)
residual_var = np.var(Y - Y_pred)
total_var = np.var(Y)
partial_r2 = (1 - residual_var / total_var) * 100

print(f"\nDML Results (SC → AI Ratings):")
print(f"  θ = {theta:.4f}")
print(f"  SE = {theta_se:.4f}")
print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  p-value = {p_value:.4f}")
print(f"  Partial R² ≈ {partial_r2:.2f}%")

# Save results
results = {
    'model': 'Qwen/Qwen2.5-32B-Instruct-AWQ',
    'n_essays': len(df),
    'embedding_dim_original': embeddings.shape[1],
    'embedding_dim_pca': n_components,
    'pca_explained_variance': explained_var,
    'prediction': {
        'linear_ai_ratings_r2': scores_ai_linear.mean(),
        'linear_actual_sc_r2': scores_sc_linear.mean(),
        'xgboost_ai_ratings_r2': scores_ai_xgb.mean(),
        'xgboost_actual_sc_r2': scores_sc_xgb.mean(),
        'gap': gap
    },
    'dml': {
        'theta': theta,
        'se': theta_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'partial_r2': partial_r2
    },
    'timestamp': pd.Timestamp.now().isoformat()
}

with open(OUTPUT_DIR / 'qwen32b_analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✓ Saved all results to {OUTPUT_DIR}")

# Comparison with other models
print("\n" + "="*80)
print("COMPARISON WITH OTHER MODELS")
print("="*80)

print("\nModel Comparison Table:")
print(f"{'Model':<15} {'Dims':<10} {'PCA Var':<10} {'AI R²':<10} {'SC R²':<10} {'Gap':<10} {'θ':<10} {'p-value':<10}")
print("-"*95)

# Qwen-32B (current)
print(f"{'Qwen-32B':<15} {'5120→200':<10} {f'{explained_var:.1%}':<10} "
      f"{f'{scores_ai_xgb.mean():.3f}':<10} {f'{scores_sc_xgb.mean():.3f}':<10} "
      f"{f'{gap:.1f}%':<10} {f'{theta:.4f}':<10} {f'{p_value:.4f}':<10}")

# Other models (from checkpoints)
print(f"{'OpenAI':<15} {'3072→200':<10} {'95.0%':<10} {'0.923':<10} {'0.537':<10} {'38.6%':<10} {'0.0527':<10} {'<0.001':<10}")
print(f"{'NV-Embed':<15} {'4096→200':<10} {'72.1%':<10} {'0.597':<10} {'0.073':<10} {'52.4%':<10} {'0.0016':<10} {'0.0303':<10}")
print(f"{'MPNet':<15} {'768→200':<10} {'93.6%':<10} {'0.451':<10} {'0.050':<10} {'40.1%':<10} {'0.0018':<10} {'0.1071':<10}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Analyze results
if scores_ai_xgb.mean() > 0.923:
    print("🚨 Qwen-32B OUTPERFORMS OpenAI for AI rating prediction!")
elif scores_ai_xgb.mean() > 0.597:
    print("✅ Qwen-32B performs better than NV-Embed but not OpenAI")
else:
    print("❌ Qwen-32B underperforms dedicated embedding models")

if p_value < 0.05:
    print(f"✅ Qwen-32B detects significant causal effect (p={p_value:.4f})")
else:
    print(f"❌ Qwen-32B fails to detect significant causal effect (p={p_value:.4f})")

print(f"\nPCA variance retention: {explained_var:.1%}")
if explained_var < 0.8:
    print("⚠️  Low variance retention - consider using more PCA components")

print("\nSummary: Qwen-32B provides a unique perspective as a generative model")
print("used for embeddings, showing how LLMs 'understand' text differently")
print("than purpose-built embedding models.")