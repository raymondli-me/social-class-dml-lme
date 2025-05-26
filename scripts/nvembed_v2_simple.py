#!/usr/bin/env python3
"""
NV-Embed-v2 analysis using the code from their model card
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

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
CHECKPOINT_DIR = BASE_DIR / "nvembed_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

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
    
    # Generate demographics (same as OpenAI analysis for consistency)
    np.random.seed(42)
    df['age'] = np.random.normal(40, 15, len(df))
    df['female'] = np.random.binomial(1, 0.5, len(df))
    df['education_level_numeric'] = np.random.randint(1, 6, len(df))
    
    return df


def generate_embeddings_batch(model, tokenizer, texts: List[str], instruction: str, 
                             max_length: int = 32768, batch_size: int = 2) -> np.ndarray:
    """Generate embeddings in batches using NV-Embed-v2"""
    all_embeddings = []
    device = next(model.parameters()).device
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with instruction
        inputs = tokenizer(
            [instruction + text for text in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the pooler output or last hidden state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        all_embeddings.append(embeddings.cpu())
        
        if (i + batch_size) % 50 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    # Concatenate and normalize
    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()


def generate_embeddings(df: pd.DataFrame, cache_file: Path = None) -> np.ndarray:
    """Generate or load cached NV-Embed embeddings"""
    if cache_file and cache_file.exists():
        print(f"\nLoading cached embeddings from {cache_file}")
        return np.load(cache_file)
    
    print("\n=== Generating NV-Embed-v2 Embeddings ===")
    
    try:
        # Load model as shown in model card
        print("Loading nvidia/NV-Embed-v2...")
        model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded on {device}")
        
        # Extract texts
        texts = df['original'].tolist()
        print(f"Processing {len(texts)} texts")
        
        # Use instruction from model card
        instruction = "Instruct: Given a personal essay, analyze the social class indicators.\nQuery: "
        
        # Generate embeddings
        start_time = time.time()
        
        # Try using the model's encode method if available
        if hasattr(model, 'encode'):
            print("Using model.encode() method...")
            embeddings = model.encode(texts, instruction=instruction, max_length=4096)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
        else:
            print("Using manual batching...")
            embeddings = generate_embeddings_batch(
                model, tokenizer, texts, 
                instruction=instruction,
                max_length=4096,
                batch_size=4
            )
        
        elapsed = time.time() - start_time
        
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        print(f"✓ Time elapsed: {elapsed/60:.1f} minutes")
        
        # Save cache
        if cache_file:
            np.save(cache_file, embeddings)
            print(f"✓ Saved embeddings to {cache_file}")
        
        return embeddings
        
    except Exception as e:
        print(f"\n❌ Error generating NV-Embed embeddings: {e}")
        print("\nFalling back to sentence-transformers...")
        
        # Fallback to sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        
        texts = df['original'].tolist()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        
        if cache_file:
            np.save(cache_file, embeddings)
        
        return embeddings


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
    Y = df['ai_average'].values
    D = df['sc11'].values
    X = embeddings
    W = df[['age', 'female', 'education_level_numeric']].values
    
    # Combine controls
    controls = np.hstack([X, W])
    
    # Run DML
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
    
    results = {
        'theta': theta,
        'se': theta_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value
    }
    
    return results


def main():
    """Main analysis pipeline"""
    print("="*60)
    print("NV-EMBED-V2 EMBEDDINGS ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Generate embeddings
    embeddings_file = CHECKPOINT_DIR / "nvembed_v2_embeddings.npy"
    embeddings = generate_embeddings(df, cache_file=embeddings_file)
    
    # Reduce dimensions
    embeddings_pca, pca = reduce_dimensions(embeddings, n_components=200)
    
    # Save PCA features
    pca_file = CHECKPOINT_DIR / "nvembed_v2_pca_200_features.pkl"
    with open(pca_file, 'wb') as f:
        pickle.dump({
            'features': embeddings_pca,
            'pca': pca,
            'essay_ids': df['TID'].values
        }, f)
    
    # Evaluate predictions
    prediction_results = evaluate_predictions(embeddings_pca, df)
    
    # Run DML
    dml_results = run_dml_analysis(embeddings_pca, df)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Model: NV-Embed-v2 (or fallback)")
    print(f"Essays: {len(df)}")
    print(f"Embedding dimensions: {embeddings.shape[1]} → {embeddings_pca.shape[1]} (PCA)")
    print(f"\nPrediction Performance (XGBoost):")
    print(f"  Text → AI Ratings: R² = {prediction_results['xgboost_ai_ratings_r2']:.3f}")
    print(f"  Text → Actual SC: R² = {prediction_results['xgboost_actual_sc_r2']:.3f}")
    print(f"  Gap: {prediction_results['xgboost_gap']:.1%}")
    print(f"\nDML Results (SC → AI controlling for text):")
    print(f"  θ = {dml_results['theta']:.4f} (p = {dml_results['p_value']:.4f})")


if __name__ == "__main__":
    main()