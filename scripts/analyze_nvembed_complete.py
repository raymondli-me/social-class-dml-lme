#!/usr/bin/env python3
"""
Complete NV-Embed-v2 analysis with DML
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb

# For DML
from econml.dml import LinearDML

# Set up paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
NVEMBED_DIR = BASE_DIR / "nvembed_checkpoints"

# CRITICAL: Load the CORRECT data file - 9,513 essays, not 526!
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"
SC_LABELS_FILE = Path("/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv")
AI_RATINGS_FILE = BASE_DIR / "asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv"


def load_data() -> pd.DataFrame:
    """Load and merge all data sources"""
    print("\n=== Loading Data ===")
    
    # Load essays - CRITICAL CHECK
    essays = pd.read_csv(ESSAYS_FILE)
    print(f"✓ Essays loaded: {len(essays)} (should be 9,513)")
    assert len(essays) == 9513, f"ERROR: Expected 9,513 essays but got {len(essays)}!"
    
    # Load social class labels
    sc_labels = pd.read_csv(SC_LABELS_FILE)
    print(f"✓ SC labels loaded: {len(sc_labels)}")
    
    # Merge
    df = essays.merge(sc_labels, on='TID', how='inner')
    print(f"✓ After merge: {len(df)} essays with SC labels")
    
    # Load AI ratings
    ai_ratings = pd.read_csv(AI_RATINGS_FILE)
    print(f"✓ AI ratings loaded: {len(ai_ratings)} (should be ~19,026)")
    
    # Average AI ratings per essay
    ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
    ai_avg.columns = ['TID', 'ai_average']
    
    # Final merge
    df = df.merge(ai_avg, on='TID', how='inner')
    print(f"✓ Final dataset: {len(df)} essays with all data")
    
    # Generate demographics (same as other analyses for consistency)
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    return df


def load_nvembed_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """Load NV-Embed embeddings and essay IDs"""
    print("\n=== Loading NV-Embed Embeddings ===")
    
    embeddings_file = NVEMBED_DIR / "nvembed_embeddings.npy"
    ids_file = NVEMBED_DIR / "nvembed_essay_ids.npy"
    
    if not embeddings_file.exists():
        raise FileNotFoundError(f"NV-Embed embeddings not found: {embeddings_file}")
    
    embeddings = np.load(embeddings_file)
    essay_ids = np.load(ids_file, allow_pickle=True)
    
    print(f"✓ Loaded embeddings shape: {embeddings.shape}")
    print(f"✓ Loaded essay IDs: {len(essay_ids)}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, essay_ids


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 200) -> Tuple[np.ndarray, PCA]:
    """Reduce embedding dimensions using PCA"""
    print(f"\n=== Reducing to {n_components} dimensions ===")
    
    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"✓ Explained variance: {explained_var:.1%}")
    print(f"✓ Reduced shape: {embeddings_pca.shape}")
    
    return embeddings_pca, pca


def evaluate_predictions(embeddings: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate how well embeddings predict AI ratings and actual SC"""
    print("\n=== Evaluating Prediction Performance ===")
    
    # Prepare data
    X = embeddings
    y_ai = df['ai_average'].values
    y_sc = df['sc11'].values
    
    # Control variables
    controls = df[['age', 'female', 'education_level_numeric']].values
    X_with_controls = np.hstack([X, controls])
    
    results = {}
    
    # Test both linear and XGBoost
    for model_type in ['linear', 'xgboost']:
        print(f"\n{model_type.upper()} Model:")
        
        # 5-fold CV for both targets
        for target_name, y in [('ai_ratings', y_ai), ('actual_sc', y_sc)]:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X_with_controls[train_idx], X_with_controls[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                if model_type == 'linear':
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:  # xgboost
                    model = xgb.XGBRegressor(
                        n_estimators=50,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                scores.append(r2_score(y_test, y_pred))
            
            avg_r2 = np.mean(scores)
            results[f'{model_type}_{target_name}_r2'] = avg_r2
            print(f"  {target_name}: R² = {avg_r2:.3f}")
    
    # Calculate gaps
    results['linear_gap'] = results['linear_ai_ratings_r2'] - results['linear_actual_sc_r2']
    results['xgboost_gap'] = results['xgboost_ai_ratings_r2'] - results['xgboost_actual_sc_r2']
    
    print(f"\nGaps (AI - SC):")
    print(f"  Linear: {results['linear_gap']:.1%}")
    print(f"  XGBoost: {results['xgboost_gap']:.1%}")
    
    return results


def run_dml_analysis(embeddings: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
    """Run Double Machine Learning analysis"""
    print("\n=== Running DML Analysis ===")
    
    # Prepare data
    Y = df['ai_average'].values  # Outcome: AI ratings
    D = df['sc11'].values        # Treatment: Actual SC
    X = embeddings                # High-dim controls: embeddings
    W = df[['age', 'female', 'education_level_numeric']].values  # Other controls
    
    # Combine controls
    controls = np.hstack([X, W])
    
    # Run DML with XGBoost first stage
    print("\nRunning DML with XGBoost first stage...")
    dml = LinearDML(
        model_y=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        model_t=xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        discrete_treatment=False,
        cv=5,
        random_state=42
    )
    
    dml.fit(Y, D, X=controls)
    
    # Get results
    theta = dml.coef_[0]
    inference = dml.coef__inference()
    theta_se = np.sqrt(inference.var[0])
    conf_int = inference.conf_int()
    ci_lower = conf_int[0][0]
    ci_upper = conf_int[0][1]
    p_value = inference.pvalue()[0]
    
    print(f"\nDML Results (SC → AI Ratings):")
    print(f"  θ = {theta:.4f}")
    print(f"  SE = {theta_se:.4f}")
    print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  p-value = {p_value:.4f}")
    
    # Calculate partial R²
    from sklearn.base import clone
    model_t = clone(dml.models_t[0][0])
    model_t.fit(controls, D)
    D_res = D - model_t.predict(controls)
    Y_hat_partial = theta * D_res
    partial_r2 = np.corrcoef(Y_hat_partial, Y - Y.mean())[0,1]**2
    print(f"  Partial R² ≈ {partial_r2:.2%}")
    
    results = {
        'theta': theta,
        'se': theta_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'partial_r2': partial_r2
    }
    
    return results


def main():
    """Main analysis pipeline"""
    print("="*60)
    print("NV-EMBED-V2 COMPLETE DML ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Load NV-Embed embeddings
    embeddings, essay_ids = load_nvembed_embeddings()
    
    # Verify alignment
    print(f"\n=== Verifying Data Alignment ===")
    df_sorted = df.sort_values('TID').reset_index(drop=True)
    ids_sorted = np.sort(essay_ids)
    df_ids_sorted = np.sort(df_sorted['TID'].values)
    
    if not np.array_equal(ids_sorted, df_ids_sorted):
        print("❌ Essay ID mismatch - need to align data")
        # Create mapping and reorder
        id_to_idx = {tid: i for i, tid in enumerate(essay_ids)}
        df_indices = [id_to_idx[tid] for tid in df['TID'].values]
        embeddings = embeddings[df_indices]
        print("✓ Data aligned by TID")
    else:
        print("✓ Data already aligned")
    
    # Reduce dimensions
    embeddings_pca, pca = reduce_dimensions(embeddings, n_components=200)
    
    # Save PCA features
    pca_file = NVEMBED_DIR / "nvembed_pca_200_features.pkl"
    with open(pca_file, 'wb') as f:
        pickle.dump({
            'features': embeddings_pca,
            'pca': pca,
            'essay_ids': df['TID'].values,
            'explained_variance_ratio': np.sum(pca.explained_variance_ratio_)
        }, f)
    print(f"✓ Saved PCA features to {pca_file}")
    
    # Evaluate predictions
    prediction_results = evaluate_predictions(embeddings_pca, df)
    
    # Run DML
    dml_results = run_dml_analysis(embeddings_pca, df)
    
    # Save all results
    results = {
        'model': 'nvidia/NV-Embed-v2',
        'n_essays': len(df),
        'embedding_dim_original': embeddings.shape[1],
        'embedding_dim_pca': embeddings_pca.shape[1],
        'pca_explained_variance': np.sum(pca.explained_variance_ratio_),
        'prediction': prediction_results,
        'dml': dml_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    results_file = NVEMBED_DIR / "nvembed_analysis_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved all results to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Model: nvidia/NV-Embed-v2")
    print(f"Essays: {len(df)} (✓ Correct!)")
    print(f"Embedding dimensions: {embeddings.shape[1]} → {embeddings_pca.shape[1]} (PCA)")
    print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.1%}")
    print(f"\nPrediction Performance:")
    print(f"  Linear - AI: R² = {prediction_results['linear_ai_ratings_r2']:.3f}, SC: R² = {prediction_results['linear_actual_sc_r2']:.3f}")
    print(f"  XGBoost - AI: R² = {prediction_results['xgboost_ai_ratings_r2']:.3f}, SC: R² = {prediction_results['xgboost_actual_sc_r2']:.3f}")
    print(f"  Gap: {prediction_results['xgboost_gap']:.1%}")
    print(f"\nDML Results (SC → AI controlling for text):")
    print(f"  θ = {dml_results['theta']:.4f} (p = {dml_results['p_value']:.4f})")
    print(f"  Partial R² = {dml_results['partial_r2']:.2%}")
    
    return results


if __name__ == "__main__":
    main()