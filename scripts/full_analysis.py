#!/usr/bin/env python3
"""
Full analysis pipeline for social class detection using vLLM and DML-LME
Processes all essays with 100 prompts, then performs double machine learning analysis
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add local packages to path (update these paths as needed)
sys.path.append('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor')
# sys.path.append('/path/to/dml-lme')  # Update with actual dml-lme path

from vllm_batch_processor import BatchProcessor, ProcessorConfig, PromptConfig
from vllm_batch_processor.parallel_runner import ParallelModelRunner

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "full"
VIZ_DIR = BASE_DIR / "visualizations"
REPORTS_DIR = BASE_DIR / "reports"

class SocialClassAnalyzer:
    def __init__(self):
        self.essays_df = None
        self.prompts_df = None
        self.results_df = None
        
    def load_data(self):
        """Load essay dataset and prompts"""
        print("Loading data...")
        self.essays_df = pd.read_csv(DATA_DIR / "essay_dataset.csv")
        self.prompts_df = pd.read_csv(DATA_DIR / "prompts_100_complete.csv")
        
        print(f"Loaded {len(self.essays_df)} essays")
        print(f"Loaded {len(self.prompts_df)} prompt configurations")
        
        # Prepare essays for vllm processing
        self.essays_df = self.essays_df.rename(columns={
            'TID': 'id',
            'original': 'text'
        })[['id', 'text', 'criterion', 'judgement']]
        
        return self.essays_df, self.prompts_df
    
    def run_vllm_batch_processing(self, model_preset="qwen-72b", use_parallel=True):
        """Run vLLM batch processing on all essays with all prompts"""
        print(f"\n=== Running vLLM Batch Processing ===")
        print(f"Model preset: {model_preset}")
        print(f"Parallel processing: {use_parallel}")
        
        # Save essays in correct format
        input_file = OUTPUT_DIR / "all_essays.csv"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        self.essays_df[['id', 'text']].to_csv(input_file, index=False)
        
        prompts_file = DATA_DIR / "prompts_100_complete.csv"
        
        if use_parallel:
            # Use parallel runner for multiple models
            runner = ParallelModelRunner()
            
            # Configure models (adjust based on your GPU setup)
            model_configs = [
                {"preset": "qwen-32b", "gpu_ids": [0, 1]},
                {"preset": "qwen-14b", "gpu_ids": [2]},
                {"preset": "qwen-7b", "gpu_ids": [3]}
            ]
            
            results = runner.run_parallel(
                str(input_file),
                model_configs,
                prompt_config_file=str(prompts_file),
                output_dir=str(OUTPUT_DIR)
            )
            
        else:
            # Single model processing
            config = ProcessorConfig(
                preset=model_preset,
                batch_size=10,
                max_new_tokens=100,
                temperature=0.1,
                output_dir=str(OUTPUT_DIR),
                save_every_n_batches=50
            )
            
            processor = BatchProcessor(config)
            results = processor.process_file(
                str(input_file),
                prompt_config_file=str(prompts_file),
                run_name=f"full_analysis_{model_preset}"
            )
        
        print("Batch processing complete!")
        return results
    
    def parse_vllm_results(self, output_files):
        """Parse vLLM output files and create analysis dataframe"""
        print("\n=== Parsing vLLM Results ===")
        
        all_results = []
        
        for output_file in output_files:
            if not os.path.exists(output_file):
                continue
                
            df = pd.read_csv(output_file)
            prompt_name = Path(output_file).stem.split('_')[-1]
            
            for _, row in df.iterrows():
                try:
                    # Parse JSON response
                    response = json.loads(row['response'])
                    
                    # Extract binary value (handle different response formats)
                    if isinstance(response, dict):
                        value = response.get('answer', response.get('result', 0))
                    else:
                        value = int(response)
                    
                    all_results.append({
                        'essay_id': row['id'],
                        'prompt_name': prompt_name,
                        'value': value,
                        'timestamp': row['timestamp']
                    })
                except:
                    # Handle parsing errors
                    all_results.append({
                        'essay_id': row['id'],
                        'prompt_name': prompt_name,
                        'value': np.nan,
                        'timestamp': row['timestamp']
                    })
        
        # Create wide format dataframe
        results_df = pd.DataFrame(all_results)
        results_wide = results_df.pivot(
            index='essay_id',
            columns='prompt_name',
            values='value'
        )
        
        # Merge with original essay data
        self.results_df = self.essays_df.merge(
            results_wide,
            left_on='id',
            right_index=True,
            how='inner'
        )
        
        print(f"Parsed results for {len(self.results_df)} essays")
        print(f"Features extracted: {len(results_wide.columns)}")
        
        return self.results_df
    
    def run_dml_analysis(self):
        """Run Double Machine Learning analysis"""
        print("\n=== Running DML-LME Analysis ===")
        
        try:
            from dml_lme import DMLAnalyzer  # Import from dml-lme package
            
            # Prepare data for DML
            # X: text features from prompts (binary indicators)
            feature_cols = [col for col in self.results_df.columns 
                          if col not in ['id', 'text', 'criterion', 'judgement']]
            
            X = self.results_df[feature_cols].fillna(0)  # Fill NaN with 0
            
            # Y: outcome variable (judgement - social class rating)
            y = self.results_df['judgement']
            
            # T: treatment variable (criterion - education level)
            # Convert to binary: 1 for college+, 0 otherwise
            t = (self.results_df['criterion'].isin(['College', 'Graduate'])).astype(int)
            
            # Run DML
            dml = DMLAnalyzer()
            results = dml.fit(X, y, t)
            
            print(f"DML Treatment Effect: {results['ate']:.3f}")
            print(f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
            
            return results
            
        except ImportError:
            print("Warning: dml-lme package not found. Skipping DML analysis.")
            print("Using simple regression analysis instead...")
            
            # Fallback: Simple OLS analysis
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            feature_cols = [col for col in self.results_df.columns 
                          if col not in ['id', 'text', 'criterion', 'judgement']]
            
            X = self.results_df[feature_cols].fillna(0)
            y = self.results_df['judgement']
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'coefficient': model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            
            return {
                'model': model,
                'feature_importance': feature_importance,
                'r2_score': model.score(X_scaled, y)
            }
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        print("\n=== Creating Visualizations ===")
        
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Distribution of social class ratings by education level
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.results_df, x='criterion', y='judgement')
        plt.title('Social Class Ratings by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Social Class Rating (1-10)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'social_class_by_education.png', dpi=300)
        plt.close()
        
        # 2. Feature prevalence heatmap
        feature_cols = [col for col in self.results_df.columns 
                       if col not in ['id', 'text', 'criterion', 'judgement']]
        
        feature_means = self.results_df.groupby('criterion')[feature_cols].mean()
        
        plt.figure(figsize=(20, 8))
        sns.heatmap(feature_means.T, cmap='RdBu_r', center=0.5, 
                   cbar_kws={'label': 'Proportion'})
        plt.title('Feature Prevalence by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Features (Prompts)')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'feature_heatmap.png', dpi=300)
        plt.close()
        
        # 3. Correlation matrix of top features
        feature_data = self.results_df[feature_cols].fillna(0)
        
        # Select top 20 features by variance
        top_features = feature_data.var().nlargest(20).index
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = feature_data[top_features].corr()
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Top 20 Features')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'feature_correlations.png', dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {VIZ_DIR}")
    
    def generate_report(self, dml_results=None):
        """Generate analysis report"""
        print("\n=== Generating Report ===")
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        report = f"""
# Social Class Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total essays analyzed: {len(self.results_df)}
- Number of prompts used: {len(self.prompts_df)}
- Education levels: {', '.join(self.results_df['criterion'].unique())}

## Social Class Ratings by Education
{self.results_df.groupby('criterion')['judgement'].describe().round(2).to_string()}

## Feature Extraction Results
- Total features extracted: {len([col for col in self.results_df.columns if col not in ['id', 'text', 'criterion', 'judgement']])}
- Missing values: {self.results_df.isnull().sum().sum()}

"""
        
        if dml_results:
            if 'ate' in dml_results:
                report += f"""
## DML Analysis Results
- Average Treatment Effect: {dml_results['ate']:.3f}
- 95% Confidence Interval: [{dml_results['ci_lower']:.3f}, {dml_results['ci_upper']:.3f}]
- P-value: {dml_results.get('p_value', 'N/A')}
"""
            elif 'feature_importance' in dml_results:
                report += f"""
## Regression Analysis Results
- R² Score: {dml_results['r2_score']:.3f}

### Top 10 Most Important Features:
{dml_results['feature_importance'].head(10).to_string(index=False)}
"""
        
        # Save report
        report_file = REPORTS_DIR / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_file}")
        
        return report
    
    def run_full_pipeline(self, model_preset="qwen-72b", use_parallel=False):
        """Run complete analysis pipeline"""
        print("=== Starting Full Social Class Analysis Pipeline ===\n")
        
        # Load data
        self.load_data()
        
        # Run vLLM processing
        output_files = self.run_vllm_batch_processing(
            model_preset=model_preset,
            use_parallel=use_parallel
        )
        
        # Parse results
        self.parse_vllm_results(output_files)
        
        # Save intermediate results
        results_file = OUTPUT_DIR / "analysis_results.csv"
        self.results_df.to_csv(results_file, index=False)
        print(f"\nSaved analysis results to {results_file}")
        
        # Run DML analysis
        dml_results = self.run_dml_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        report = self.generate_report(dml_results)
        
        print("\n✅ Analysis pipeline completed successfully!")
        
        return self.results_df, dml_results

def main():
    """Main execution"""
    analyzer = SocialClassAnalyzer()
    
    # Run full pipeline
    # Set use_parallel=True if you have multiple GPUs available
    results_df, dml_results = analyzer.run_full_pipeline(
        model_preset="qwen-72b",
        use_parallel=False
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Visualizations saved in: {VIZ_DIR}")
    print(f"Reports saved in: {REPORTS_DIR}")

if __name__ == "__main__":
    main()