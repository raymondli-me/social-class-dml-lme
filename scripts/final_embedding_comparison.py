#!/usr/bin/env python3
"""
Final comparison of all embedding models with corrected methodology
Focus on both AI ratings (AI R²) and Actual Social Class (SC R²)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

print("="*100)
print("FINAL EMBEDDING MODEL COMPARISON - CORRECTED RESULTS")
print("="*100)

# Base paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")

# Load common data
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

y_ai = df['ai_average'].values
y_sc = df['sc11'].values

print(f"Dataset: {len(df)} essays")
print(f"AI ratings range: {y_ai.min():.1f} - {y_ai.max():.1f}")
print(f"Social class range: {y_sc.min()} - {y_sc.max()}")

# Function to evaluate embeddings
def evaluate_embeddings(X, y_ai, y_sc, model_name):
    """Evaluate embeddings using consistent methodology"""
    
    # Linear model
    linear = LinearRegression()
    ai_linear = cross_val_score(linear, X, y_ai, cv=5, scoring='r2')
    sc_linear = cross_val_score(linear, X, y_sc, cv=5, scoring='r2')
    
    # XGBoost with consistent hyperparameters
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    ai_xgb = cross_val_score(xgb_model, X, y_ai, cv=5, scoring='r2')
    sc_xgb = cross_val_score(xgb_model, X, y_sc, cv=5, scoring='r2')
    
    return {
        'ai_linear': ai_linear.mean(),
        'sc_linear': sc_linear.mean(),
        'ai_xgb': ai_xgb.mean(),
        'sc_xgb': sc_xgb.mean(),
        'ai_linear_std': ai_linear.std(),
        'sc_linear_std': sc_linear.std(),
        'ai_xgb_std': ai_xgb.std(),
        'sc_xgb_std': sc_xgb.std()
    }

results = {}

# 1. OpenAI embeddings
print("\n1. Evaluating OpenAI embeddings...")
openai_file = BASE_DIR / "openai_checkpoints/pca_200_features.pkl"
if openai_file.exists():
    with open(openai_file, 'rb') as f:
        openai_data = pickle.load(f)
    X_openai = openai_data['features']
    results['OpenAI'] = evaluate_embeddings(X_openai, y_ai, y_sc, 'OpenAI')
    print(f"   Shape: {X_openai.shape}")

# 2. NV-Embed embeddings
print("\n2. Evaluating NV-Embed embeddings...")
nvembed_file = BASE_DIR / "nvembed_checkpoints/nvembed_pca_200_features.pkl"
if nvembed_file.exists():
    with open(nvembed_file, 'rb') as f:
        nvembed_data = pickle.load(f)
    X_nvembed = nvembed_data['features']
    results['NV-Embed'] = evaluate_embeddings(X_nvembed, y_ai, y_sc, 'NV-Embed')
    print(f"   Shape: {X_nvembed.shape}")

# 3. MPNet embeddings
print("\n3. Evaluating MPNet embeddings...")
mpnet_file = BASE_DIR / "mpnet_checkpoints/mpnet_pca_200_features.pkl"
if mpnet_file.exists():
    with open(mpnet_file, 'rb') as f:
        mpnet_data = pickle.load(f)
    X_mpnet = mpnet_data['features']
    results['MPNet'] = evaluate_embeddings(X_mpnet, y_ai, y_sc, 'MPNet')
    print(f"   Shape: {X_mpnet.shape}")

# 4. Qwen-32B embeddings  
print("\n4. Evaluating Qwen-32B embeddings...")
qwen_file = BASE_DIR / "qwen32b_checkpoints/qwen32b_pca_200_features.pkl"
if qwen_file.exists():
    with open(qwen_file, 'rb') as f:
        qwen_data = pickle.load(f)
    X_qwen = qwen_data['features']
    results['Qwen-32B'] = evaluate_embeddings(X_qwen, y_ai, y_sc, 'Qwen-32B')
    print(f"   Shape: {X_qwen.shape}")
else:
    # Try loading raw embeddings and doing PCA
    qwen_emb = BASE_DIR / "qwen32b_embeddings/qwen32b_awq_embeddings.npy"
    if qwen_emb.exists():
        print("   Loading raw embeddings and applying PCA...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        embeddings = np.load(qwen_emb)[:len(df)]  # Ensure alignment
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=200, random_state=42)
        X_qwen = pca.fit_transform(X_scaled)
        results['Qwen-32B'] = evaluate_embeddings(X_qwen, y_ai, y_sc, 'Qwen-32B')
        print(f"   Shape: {X_qwen.shape}, Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# Display results
print("\n" + "="*100)
print("RESULTS COMPARISON - AI RATINGS AND ACTUAL SOCIAL CLASS")
print("="*100)

# Table header
print(f"\n{'Model':<12} | {'Linear AI R²':<12} | {'Linear SC R²':<12} | {'XGBoost AI R²':<14} | {'XGBoost SC R²':<14} | {'Gap (XGB)':<10}")
print("-"*90)

# Sort by XGBoost AI R²
sorted_models = sorted(results.items(), key=lambda x: x[1]['ai_xgb'], reverse=True)

for model, scores in sorted_models:
    gap = (scores['ai_xgb'] - scores['sc_xgb']) * 100
    print(f"{model:<12} | "
          f"{scores['ai_linear']:>5.3f} (±{scores['ai_linear_std']:.3f}) | "
          f"{scores['sc_linear']:>5.3f} (±{scores['sc_linear_std']:.3f}) | "
          f"{scores['ai_xgb']:>6.3f} (±{scores['ai_xgb_std']:.3f}) | "
          f"{scores['sc_xgb']:>6.3f} (±{scores['sc_xgb_std']:.3f}) | "
          f"{gap:>8.1f}%")

# Rankings
print("\n" + "="*100)
print("RANKINGS")
print("="*100)

print("\n1. AI RATINGS PREDICTION (XGBoost R²):")
for i, (model, scores) in enumerate(sorted_models):
    print(f"   {i+1}. {model}: {scores['ai_xgb']:.3f}")

print("\n2. ACTUAL SOCIAL CLASS PREDICTION (XGBoost R²):")
sc_sorted = sorted(results.items(), key=lambda x: x[1]['sc_xgb'], reverse=True)
for i, (model, scores) in enumerate(sc_sorted):
    print(f"   {i+1}. {model}: {scores['sc_xgb']:.3f}")

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

print("\n1. For AI ratings prediction:")
best_ai = sorted_models[0]
print(f"   - {best_ai[0]} performs best ({best_ai[1]['ai_xgb']:.3f})")
print(f"   - Linear models achieve {best_ai[1]['ai_linear']:.3f} R²")
print(f"   - XGBoost improves performance by {(best_ai[1]['ai_xgb'] - best_ai[1]['ai_linear']):.3f}")

print("\n2. For actual social class prediction:")
best_sc = sc_sorted[0]
print(f"   - {best_sc[0]} performs best ({best_sc[1]['sc_xgb']:.3f})")
print(f"   - All models struggle with social class (best R² = {best_sc[1]['sc_xgb']:.3f})")
if best_sc[1]['sc_xgb'] < 0.1:
    print("   - Performance is near random (R² < 0.1)")

print("\n3. The AI-SC gap:")
gaps = [(m, (s['ai_xgb'] - s['sc_xgb']) * 100) for m, s in results.items()]
min_gap = min(gaps, key=lambda x: x[1])
max_gap = max(gaps, key=lambda x: x[1])
print(f"   - Smallest gap: {min_gap[0]} ({min_gap[1]:.1f}%)")
print(f"   - Largest gap: {max_gap[0]} ({max_gap[1]:.1f}%)")
print("   - This gap represents AI bias in perceiving social class")

print("\n" + "="*100)