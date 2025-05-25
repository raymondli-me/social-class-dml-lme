#!/usr/bin/env python3
"""
Optimized Double Machine Learning (DML) analysis for social class prediction
Processes embeddings in batches to avoid timeouts
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For embeddings
from sentence_transformers import SentenceTransformer
import torch

# For progress tracking
from tqdm import tqdm

class OptimizedDMLAnalysis:
    """Optimized DML analysis with batch processing and checkpointing"""
    
    def __init__(self, n_folds=5, n_components=100, batch_size=500, use_gpu=True):
        self.n_folds = n_folds
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = 42
        self.results = {}
        self.checkpoint_dir = Path("dml_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Check GPU availability
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def load_data(self):
        """Load essays and social class data"""
        print("Loading data...")
        
        # Load essays
        essays = pd.read_csv('data/asc_9513_essays.csv')
        
        # Load AI ratings
        ai_ratings = pd.read_csv('asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv')
        
        # Pivot AI ratings to get one row per essay
        ai_pivot = ai_ratings.pivot_table(
            index='essay_id', 
            columns='prompt_name', 
            values='rating',
            aggfunc='first'
        ).reset_index()
        
        # Load actual social class
        sc_data = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
        
        # Merge all data
        self.data = essays.merge(ai_pivot, left_on='TID', right_on='essay_id', how='inner')
        self.data = self.data.merge(sc_data, on='TID', how='inner')
        
        # Calculate average AI rating
        self.data['ai_average'] = self.data[['ladder_standard_improved', 'human_macarthur_ladder_improved']].mean(axis=1)
        
        # Keep only necessary columns
        self.data = self.data[['TID', 'original', 'sc11', 'ladder_standard_improved', 
                               'human_macarthur_ladder_improved', 'ai_average']]
        
        print(f"Loaded {len(self.data)} essays with all labels")
        
    def generate_embeddings_batch(self):
        """Generate embeddings in batches with checkpointing"""
        embeddings_file = self.checkpoint_dir / "embeddings_complete.npy"
        
        # Check if embeddings already exist
        if embeddings_file.exists():
            print("Loading existing embeddings...")
            self.embeddings = np.load(embeddings_file)
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            return
        
        print("\nGenerating embeddings in batches...")
        
        # Use efficient model with GPU if available
        model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        all_embeddings = []
        texts = self.data['original'].tolist()
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), total=n_batches):
            batch_file = self.checkpoint_dir / f"embeddings_batch_{i}.npy"
            
            # Check if batch already processed
            if batch_file.exists():
                batch_embeddings = np.load(batch_file)
            else:
                # Generate embeddings for batch
                batch = texts[i:i+self.batch_size]
                batch_embeddings = model.encode(
                    batch, 
                    batch_size=32 if self.device == 'cuda' else 16,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                # Save batch
                np.save(batch_file, batch_embeddings)
            
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        self.embeddings = np.vstack(all_embeddings)
        
        # Save complete embeddings
        np.save(embeddings_file, self.embeddings)
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        # Clean up batch files
        for batch_file in self.checkpoint_dir.glob("embeddings_batch_*.npy"):
            batch_file.unlink()
    
    def apply_pca(self):
        """Apply PCA with checkpointing"""
        pca_file = self.checkpoint_dir / "pca_features.pkl"
        
        if pca_file.exists():
            print("Loading existing PCA features...")
            with open(pca_file, 'rb') as f:
                pca_data = pickle.load(f)
                self.X = pca_data['X']
                self.pca = pca_data['pca']
                self.results['pca_explained_variance'] = pca_data['explained_var']
            print(f"Loaded PCA features with shape: {self.X.shape}")
            return
        
        print(f"\nApplying PCA to reduce to {self.n_components} dimensions...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.X = self.pca.fit_transform(embeddings_scaled)
        
        # Calculate explained variance
        explained_var = np.cumsum(self.pca.explained_variance_ratio_)[-1]
        print(f"PCA explains {explained_var:.1%} of variance with {self.n_components} components")
        
        # Save PCA results
        self.results['pca_explained_variance'] = explained_var
        with open(pca_file, 'wb') as f:
            pickle.dump({
                'X': self.X,
                'pca': self.pca,
                'explained_var': explained_var
            }, f)
    
    def run_simple_dml(self, Y_name, method='lasso'):
        """Simplified DML focusing on key results"""
        print(f"\nRunning simplified DML for {Y_name} with {method}...")
        
        Y = self.data[Y_name].values
        X = self.X
        
        # Use first PC as treatment, rest as controls
        D = X[:, 0]  # Treatment
        W = X[:, 1:]  # Controls
        
        # Storage for cross-fitted residuals
        Y_res = np.zeros_like(Y, dtype=float)
        D_res = np.zeros_like(D, dtype=float)
        
        # Simple cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X):
            # Fit models on train, predict on test
            if method == 'lasso':
                model_Y = Lasso(alpha=0.1, max_iter=1000)
                model_D = Lasso(alpha=0.1, max_iter=1000)
            elif method == 'rf':
                model_Y = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                               random_state=self.random_state, n_jobs=-1)
                model_D = RandomForestRegressor(n_estimators=100, max_depth=10,
                                               random_state=self.random_state, n_jobs=-1)
            elif method == 'xgboost':
                model_Y = xgb.XGBRegressor(n_estimators=100, max_depth=5,
                                          random_state=self.random_state, n_jobs=-1)
                model_D = xgb.XGBRegressor(n_estimators=100, max_depth=5,
                                          random_state=self.random_state, n_jobs=-1)
            else:  # linear
                model_Y = LinearRegression()
                model_D = LinearRegression()
            
            # Residualize Y
            model_Y.fit(W[train_idx], Y[train_idx])
            Y_res[test_idx] = Y[test_idx] - model_Y.predict(W[test_idx])
            
            # Residualize D
            model_D.fit(W[train_idx], D[train_idx])
            D_res[test_idx] = D[test_idx] - model_D.predict(W[test_idx])
        
        # Second stage: regress residuals
        theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)
        
        # Calculate R² for the full model
        if method == 'lasso':
            full_model = Lasso(alpha=0.1, max_iter=1000)
        elif method == 'rf':
            full_model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                             random_state=self.random_state, n_jobs=-1)
        elif method == 'xgboost':
            full_model = xgb.XGBRegressor(n_estimators=100, max_depth=5,
                                        random_state=self.random_state, n_jobs=-1)
        else:
            full_model = LinearRegression()
        full_model.fit(X, Y)
        Y_pred = full_model.predict(X)
        r2 = r2_score(Y, Y_pred)
        
        return {
            'theta': theta,
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(Y, Y_pred)),
            'method': method,
            'target': Y_name
        }
    
    def run_all_analyses(self):
        """Run analyses for all targets"""
        results_file = self.checkpoint_dir / "dml_results.pkl"
        
        targets = ['sc11', 'ai_average', 'ladder_standard_improved', 'human_macarthur_ladder_improved']
        methods = ['linear', 'lasso', 'rf', 'xgboost']
        
        all_results = []
        
        for target in targets:
            for method in methods:
                print(f"\nAnalyzing {target} with {method}...")
                result = self.run_simple_dml(target, method)
                all_results.append(result)
                print(f"  R² = {result['r2']:.3f}")
        
        # Save results
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        return all_results
    
    def create_summary_plot(self, results):
        """Create comparison plot"""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Group by target and method
        pivot = df.pivot(index='target', columns='method', values='r2')
        
        # Create grouped bar plot
        ax = pivot.plot(kind='bar', width=0.8)
        plt.title('R² Comparison: AI vs Actual Social Class', fontsize=14)
        plt.xlabel('Target Variable', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Method', loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        
        plt.tight_layout()
        plt.savefig('dml_r2_comparison_optimized.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSummary saved to dml_r2_comparison_optimized.png")

def main():
    """Run the optimized analysis"""
    print("Starting Optimized DML Analysis")
    print("=" * 50)
    
    # Initialize
    dml = OptimizedDMLAnalysis(
        n_folds=5,
        n_components=100,  # Reduced for speed
        batch_size=500,
        use_gpu=True
    )
    
    # Load data
    dml.load_data()
    
    # Generate embeddings (with batching and checkpointing)
    dml.generate_embeddings_batch()
    
    # Apply PCA
    dml.apply_pca()
    
    # Run analyses
    results = dml.run_all_analyses()
    
    # Create summary plot
    dml.create_summary_plot(results)
    
    # Print final summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    df_results = pd.DataFrame(results)
    print("\nR² Scores by Target:")
    for target in df_results['target'].unique():
        target_results = df_results[df_results['target'] == target]
        best = target_results.loc[target_results['r2'].idxmax()]
        print(f"{target:25} R² = {best['r2']:.3f} ({best['method']})")
    
    print("\nKey Finding:")
    ai_best = df_results[df_results['target'].str.contains('ai|ladder|human')]['r2'].max()
    sc_best = df_results[df_results['target'] == 'sc11']['r2'].max()
    ratio = ai_best / sc_best
    print(f"AI ratings predict essays {ratio:.1f}x better than actual social class")

if __name__ == "__main__":
    main()