#!/usr/bin/env python3
"""
Double Machine Learning (DML) analysis for social class prediction from essays
Combines embeddings, PCA, and multiple ML methods with LIME interpretability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For embeddings
from sentence_transformers import SentenceTransformer
import torch

# For interpretability
import lime
import lime.lime_tabular

# For progress tracking
from tqdm import tqdm

class DMLSocialClassAnalysis:
    """
    Double Machine Learning for social class analysis
    Following the framework from your DML lens model repositories
    """
    
    def __init__(self, n_folds=5, n_components=200, random_state=42):
        self.n_folds = n_folds
        self.n_components = n_components
        self.random_state = random_state
        self.results = {}
        
    def load_data(self):
        """Load essays and social class data"""
        print("Loading data...")
        
        # Load essays
        self.essays = pd.read_csv('/home/raymondli/social-class-dml-lme/data/asc_9513_essays.csv')
        
        # Load actual social class
        self.sc11_data = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
        
        # Load AI ratings
        self.ai_standard = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_ladder_standard_improved_20250524_165833.csv')
        self.ai_human = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_human_macarthur_ladder_improved_20250524_174149.csv')
        
        # Merge all data
        self.data = self.essays.merge(self.sc11_data, on='TID')
        self.data = self.data.merge(self.ai_standard[['essay_id', 'rating']], 
                                   left_on='TID', right_on='essay_id')
        self.data = self.data.rename(columns={'rating': 'ai_standard'})
        self.data = self.data.merge(self.ai_human[['essay_id', 'rating']], 
                                   left_on='TID', right_on='essay_id')
        self.data = self.data.rename(columns={'rating': 'ai_human'})
        self.data['ai_average'] = (self.data['ai_standard'] + self.data['ai_human']) / 2
        
        print(f"Loaded {len(self.data)} essays with social class labels")
        
    def generate_embeddings(self):
        """Generate sentence embeddings for essays"""
        print("\nGenerating embeddings...")
        
        # Use sentence-transformers for embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient model
        
        # Generate embeddings in batches
        batch_size = 64
        embeddings = []
        
        for i in tqdm(range(0, len(self.data), batch_size)):
            batch = self.data['original'].iloc[i:i+batch_size].tolist()
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings)
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        
    def apply_pca(self):
        """Apply PCA to reduce dimensionality to n_components"""
        print(f"\nApplying PCA to reduce to {self.n_components} dimensions...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.X = self.pca.fit_transform(embeddings_scaled)
        
        # Calculate explained variance
        explained_var = np.cumsum(self.pca.explained_variance_ratio_)
        print(f"PCA explains {explained_var[-1]:.1%} of variance with {self.n_components} components")
        
        # Save PCA results
        self.results['pca_explained_variance'] = explained_var[-1]
        
    def dml_estimation(self, Y_name, method='linear'):
        """
        Implement Double Machine Learning estimation
        Y_name: target variable ('sc11', 'ai_standard', 'ai_human', 'ai_average')
        method: ML method to use ('linear', 'lasso', 'ridge', 'rf', 'xgboost')
        """
        print(f"\nRunning DML with {method} for {Y_name}...")
        
        # Get target variable
        Y = self.data[Y_name].values
        X = self.X
        
        # Initialize storage for results
        theta_estimates = []
        residuals_X = np.zeros_like(Y, dtype=float)
        residuals_Y = np.zeros_like(Y, dtype=float)
        
        # K-fold cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Select ML method
            if method == 'linear':
                model_Y = LinearRegression()
                model_X = LinearRegression()
            elif method == 'lasso':
                model_Y = Lasso(alpha=0.1, random_state=self.random_state)
                model_X = Lasso(alpha=0.1, random_state=self.random_state)
            elif method == 'ridge':
                model_Y = Ridge(alpha=1.0, random_state=self.random_state)
                model_X = Ridge(alpha=1.0, random_state=self.random_state)
            elif method == 'rf':
                model_Y = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                               random_state=self.random_state)
                model_X = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                               random_state=self.random_state)
            elif method == 'xgboost':
                model_Y = xgb.XGBRegressor(n_estimators=100, max_depth=5, 
                                          random_state=self.random_state)
                model_X = xgb.XGBRegressor(n_estimators=100, max_depth=5, 
                                          random_state=self.random_state)
            
            # For DML, we need a treatment variable
            # Let's use the first principal component as "treatment"
            D_train, D_test = X_train[:, 0], X_test[:, 0]
            X_conf_train = X_train[:, 1:]  # Other components as confounders
            X_conf_test = X_test[:, 1:]
            
            # First stage: predict Y from confounders
            model_Y.fit(X_conf_train, Y_train)
            Y_pred = model_Y.predict(X_conf_test)
            residuals_Y[test_idx] = Y_test - Y_pred
            
            # First stage: predict D from confounders
            model_X.fit(X_conf_train, D_train)
            D_pred = model_X.predict(X_conf_test)
            residuals_X[test_idx] = D_test - D_pred
        
        # Second stage: regress Y residuals on D residuals
        theta = np.sum(residuals_Y * residuals_X) / np.sum(residuals_X ** 2)
        
        # Calculate standard error
        n = len(Y)
        sigma2 = np.mean((residuals_Y - theta * residuals_X) ** 2)
        se = np.sqrt(sigma2 / np.sum(residuals_X ** 2))
        
        # Calculate R-squared for the full model
        Y_pred_full = cross_val_predict(
            model_Y if method != 'xgboost' else xgb.XGBRegressor(n_estimators=100, max_depth=5),
            X, Y, cv=self.n_folds
        )
        r2 = r2_score(Y, Y_pred_full)
        
        # Store results
        result = {
            'method': method,
            'target': Y_name,
            'theta': theta,
            'se': se,
            't_stat': theta / se,
            'p_value': 2 * (1 - stats.t.cdf(abs(theta / se), df=n-2)),
            'r2': r2,
            'rmse': np.sqrt(mean_squared_error(Y, Y_pred_full))
        }
        
        return result
    
    def run_all_methods(self):
        """Run DML with all methods for all targets"""
        methods = ['linear', 'lasso', 'ridge', 'rf', 'xgboost']
        targets = ['sc11', 'ai_standard', 'ai_human', 'ai_average']
        
        all_results = []
        
        for target in targets:
            for method in methods:
                try:
                    result = self.dml_estimation(target, method)
                    all_results.append(result)
                    print(f"  {method}: R² = {result['r2']:.3f}, θ = {result['theta']:.3f} (p = {result['p_value']:.3f})")
                except Exception as e:
                    print(f"  Error with {method}: {e}")
        
        self.results['dml_results'] = pd.DataFrame(all_results)
        
    def lime_interpretation(self, target='sc11', n_samples=100):
        """Apply LIME for interpretability"""
        print(f"\nApplying LIME interpretation for {target}...")
        
        # Select best performing model based on R²
        best_result = self.results['dml_results'][
            self.results['dml_results']['target'] == target
        ].nlargest(1, 'r2').iloc[0]
        
        best_method = best_result['method']
        print(f"Using {best_method} model (R² = {best_result['r2']:.3f})")
        
        # Train the model on full data
        Y = self.data[target].values
        
        if best_method == 'linear':
            model = LinearRegression()
        elif best_method == 'lasso':
            model = Lasso(alpha=0.1)
        elif best_method == 'ridge':
            model = Ridge(alpha=1.0)
        elif best_method == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=10)
        elif best_method == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
        
        model.fit(self.X, Y)
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X,
            feature_names=[f'PC{i+1}' for i in range(self.n_components)],
            mode='regression',
            random_state=self.random_state
        )
        
        # Get explanations for random samples
        np.random.seed(self.random_state)
        sample_indices = np.random.choice(len(self.X), n_samples, replace=False)
        
        feature_importance = np.zeros(self.n_components)
        
        for idx in tqdm(sample_indices, desc="Computing LIME explanations"):
            exp = explainer.explain_instance(
                self.X[idx], 
                model.predict, 
                num_features=20  # Top 20 features
            )
            
            # Aggregate feature importance
            for feat_idx, importance in exp.as_list():
                # Extract feature index from name (e.g., 'PC1' -> 0)
                feat_num = int(feat_idx.split('PC')[1].split()[0]) - 1
                feature_importance[feat_num] += abs(importance)
        
        # Average importance
        feature_importance /= n_samples
        
        # Store top features
        top_features = np.argsort(feature_importance)[::-1][:20]
        self.results['lime_importance'] = {
            'target': target,
            'method': best_method,
            'top_features': top_features,
            'importance_scores': feature_importance[top_features]
        }
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. R² comparison across methods and targets
        ax1 = axes[0, 0]
        results_pivot = self.results['dml_results'].pivot(
            index='method', columns='target', values='r2'
        )
        sns.heatmap(results_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('R² Scores: DML Methods vs Targets')
        ax1.set_xlabel('Target Variable')
        ax1.set_ylabel('ML Method')
        
        # 2. Coefficient estimates with confidence intervals
        ax2 = axes[0, 1]
        for i, target in enumerate(['sc11', 'ai_standard', 'ai_human', 'ai_average']):
            subset = self.results['dml_results'][self.results['dml_results']['target'] == target]
            y_pos = np.arange(len(subset)) + i * 0.2
            ax2.errorbar(subset['theta'], y_pos, xerr=1.96*subset['se'], 
                        fmt='o', label=target, capsize=5)
        ax2.set_yticks(np.arange(len(subset)))
        ax2.set_yticklabels(subset['method'])
        ax2.set_xlabel('Coefficient Estimate (θ)')
        ax2.set_title('DML Coefficient Estimates with 95% CI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA explained variance
        ax3 = axes[1, 0]
        explained_var = np.cumsum(self.pca.explained_variance_ratio_)
        ax3.plot(range(1, len(explained_var)+1), explained_var)
        ax3.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.set_title('PCA Explained Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. LIME feature importance
        ax4 = axes[1, 1]
        if 'lime_importance' in self.results:
            lime_data = self.results['lime_importance']
            top_n = 15
            indices = lime_data['top_features'][:top_n]
            scores = lime_data['importance_scores'][:top_n]
            
            ax4.barh(range(top_n), scores)
            ax4.set_yticks(range(top_n))
            ax4.set_yticklabels([f'PC{i+1}' for i in indices])
            ax4.set_xlabel('Average Importance Score')
            ax4.set_title(f'Top {top_n} Features (LIME) for {lime_data["target"]}')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('DML Analysis: Social Class Prediction from Essays', fontsize=16)
        plt.tight_layout()
        plt.savefig('/home/raymondli/social-class-dml-lme/dml_analysis_results.png', 
                   dpi=300, bbox_inches='tight')
        
    def save_results(self):
        """Save all results to files"""
        print("\nSaving results...")
        
        # Save DML results
        self.results['dml_results'].to_csv(
            '/home/raymondli/social-class-dml-lme/dml_results_summary.csv', 
            index=False
        )
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'/home/raymondli/social-class-dml-lme/dml_analysis_report_{timestamp}.txt', 'w') as f:
            f.write("DOUBLE MACHINE LEARNING ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Dataset: {len(self.data)} ASC essays\n")
            f.write(f"Embedding dimensions: {self.embeddings.shape[1]}\n")
            f.write(f"PCA components: {self.n_components}\n")
            f.write(f"PCA explained variance: {self.results['pca_explained_variance']:.1%}\n")
            f.write(f"Cross-validation folds: {self.n_folds}\n\n")
            
            f.write("DML RESULTS SUMMARY\n")
            f.write("-"*50 + "\n")
            
            for target in ['sc11', 'ai_standard', 'ai_human', 'ai_average']:
                f.write(f"\nTarget: {target}\n")
                subset = self.results['dml_results'][self.results['dml_results']['target'] == target]
                for _, row in subset.iterrows():
                    f.write(f"  {row['method']:8s}: R²={row['r2']:.3f}, θ={row['theta']:6.3f} ")
                    f.write(f"(SE={row['se']:.3f}, p={row['p_value']:.3f})\n")
            
            if 'lime_importance' in self.results:
                f.write("\n\nLIME INTERPRETATION\n")
                f.write("-"*50 + "\n")
                lime_data = self.results['lime_importance']
                f.write(f"Target: {lime_data['target']}\n")
                f.write(f"Method: {lime_data['method']}\n")
                f.write("Top 10 important components:\n")
                for i in range(10):
                    f.write(f"  PC{lime_data['top_features'][i]+1}: {lime_data['importance_scores'][i]:.3f}\n")
        
        print(f"Results saved!")


def main():
    """Run the complete DML analysis"""
    print("Starting DML Social Class Analysis...")
    print("="*50)
    
    # Initialize analyzer
    analyzer = DMLSocialClassAnalysis(n_folds=5, n_components=200)
    
    # Run analysis pipeline
    analyzer.load_data()
    analyzer.generate_embeddings()
    analyzer.apply_pca()
    analyzer.run_all_methods()
    analyzer.lime_interpretation(target='sc11')
    analyzer.create_visualizations()
    analyzer.save_results()
    
    print("\n" + "="*50)
    print("Analysis complete!")


if __name__ == "__main__":
    main()