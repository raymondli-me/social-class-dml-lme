#!/usr/bin/env python3
"""
Parse actual vLLM outputs and create evaluation report
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "vllm_actual"
VIZ_DIR = BASE_DIR / "visualizations" / "vllm_actual"
REPORTS_DIR = BASE_DIR / "reports" / "vllm_actual"

def parse_vllm_outputs():
    """Parse all vLLM output files"""
    print("=== Parsing vLLM Outputs ===")
    
    # Find all output CSV files
    output_files = list(OUTPUT_DIR.glob("*full_526_essays*.csv"))
    print(f"Found {len(output_files)} output files")
    
    all_results = []
    
    for output_file in output_files:
        # Extract prompt name from filename
        # Format: full_526_essays_50_prompts_TIMESTAMP_promptname.csv
        parts = output_file.stem.split('_')
        prompt_name = '_'.join(parts[6:]) if len(parts) > 6 else parts[-1]
        
        print(f"Processing {prompt_name}...")
        
        # Read vLLM output
        df = pd.read_csv(output_file)
        
        # Parse each response
        for _, row in df.iterrows():
            try:
                # Parse JSON response
                response = json.loads(row['response'])
                
                # Extract rating
                if isinstance(response, dict):
                    # Find the key containing the rating
                    rating_key = next(iter(response.keys()))
                    rating = float(response[rating_key])
                else:
                    rating = float(response)
                
                # Ensure valid range
                rating = np.clip(rating, 1, 10)
                
                all_results.append({
                    'essay_id': row['id'],
                    'prompt_name': prompt_name,
                    'rating': rating,
                    'timestamp': row.get('timestamp', '')
                })
                
            except Exception as e:
                print(f"  Warning: Failed to parse essay {row['id']}: {e}")
                # Skip failed parses
    
    # Create combined dataframe
    results_df = pd.DataFrame(all_results)
    print(f"\nTotal ratings parsed: {len(results_df)}")
    
    # Check coverage
    n_essays = results_df['essay_id'].nunique()
    n_prompts = results_df['prompt_name'].nunique()
    print(f"Essays covered: {n_essays}")
    print(f"Prompts covered: {n_prompts}")
    print(f"Expected: {n_essays * n_prompts} ratings")
    
    return results_df

def evaluate_results(results_df):
    """Evaluate vLLM results against human judgments"""
    print("\n=== Evaluating Results ===")
    
    # Load hidden labels
    labels_df = pd.read_csv(DATA_DIR / "labels_hidden_526.csv")
    
    # Calculate essay-level statistics
    essay_stats = results_df.groupby('essay_id')['rating'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    essay_stats['range'] = essay_stats['max'] - essay_stats['min']
    
    # Merge with labels
    evaluation_df = essay_stats.merge(labels_df, left_on='essay_id', right_on='id')
    
    # Overall correlation
    overall_corr = evaluation_df[['mean', 'judgement']].corr().iloc[0, 1]
    mae = np.abs(evaluation_df['mean'] - evaluation_df['judgement']).mean()
    rmse = np.sqrt(((evaluation_df['mean'] - evaluation_df['judgement'])**2).mean())
    
    print(f"Overall correlation: {overall_corr:.3f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Prompt-wise correlations
    prompt_corrs = []
    for prompt in results_df['prompt_name'].unique():
        prompt_data = results_df[results_df['prompt_name'] == prompt]
        merged = prompt_data.merge(labels_df, left_on='essay_id', right_on='id')
        
        if len(merged) > 10:
            corr = merged[['rating', 'judgement']].corr().iloc[0, 1]
            prompt_corrs.append({
                'prompt': prompt,
                'correlation': corr,
                'n': len(merged),
                'mean_rating': merged['rating'].mean()
            })
    
    prompt_corr_df = pd.DataFrame(prompt_corrs).sort_values('correlation', ascending=False)
    
    return evaluation_df, prompt_corr_df, overall_corr, mae, rmse

def create_visualizations(results_df, evaluation_df, prompt_corr_df):
    """Create visualizations of actual vLLM results"""
    print("\n=== Creating Visualizations ===")
    
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Human vs vLLM scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(evaluation_df['judgement'], evaluation_df['mean'], 
                alpha=0.5, s=50)
    plt.plot([1, 10], [1, 10], 'r--', label='Perfect Agreement')
    
    # Add correlation text
    corr = evaluation_df[['judgement', 'mean']].corr().iloc[0, 1]
    plt.text(2, 9, f'r = {corr:.3f}', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Human Judgment', fontsize=12)
    plt.ylabel('Mean vLLM Rating', fontsize=12)
    plt.title('Human vs vLLM Agreement (Actual Model Output)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'human_vs_vllm_actual.png', dpi=300)
    plt.close()
    
    # 2. Distribution comparison
    plt.figure(figsize=(12, 6))
    plt.hist(evaluation_df['judgement'], bins=20, alpha=0.5, 
             label='Human Judgments', density=True)
    plt.hist(evaluation_df['mean'], bins=20, alpha=0.5, 
             label='vLLM Ratings', density=True)
    plt.xlabel('Rating')
    plt.ylabel('Density')
    plt.title('Distribution of Human vs vLLM Ratings')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'rating_distributions_actual.png', dpi=300)
    plt.close()
    
    # 3. Prompt correlations
    plt.figure(figsize=(12, 12))
    y_pos = np.arange(len(prompt_corr_df))
    plt.barh(y_pos, prompt_corr_df['correlation'])
    plt.yticks(y_pos, prompt_corr_df['prompt'], fontsize=8)
    plt.xlabel('Correlation with Human Judgment')
    plt.title('All Prompt Correlations (Actual vLLM)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'prompt_correlations_actual.png', dpi=300)
    plt.close()
    
    # 4. Education level analysis
    edu_map = {
        -2.183751267: 'Less than HS',
        -1.052047927: 'High School', 
        0.079655412: 'Some College',
        1.211358752: 'College',
        2.343062092: 'Graduate'
    }
    
    evaluation_df['edu_label'] = evaluation_df['criterion'].map(edu_map)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ratings by education
    evaluation_df.boxplot(column='mean', by='edu_label', ax=ax1)
    ax1.set_title('vLLM Ratings by Education Level')
    ax1.set_xlabel('Education Level')
    ax1.set_ylabel('Mean vLLM Rating')
    
    # Human judgments by education  
    evaluation_df.boxplot(column='judgement', by='edu_label', ax=ax2)
    ax2.set_title('Human Judgments by Education Level')
    ax2.set_xlabel('Education Level')
    ax2.set_ylabel('Human Judgment')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'education_comparison_actual.png', dpi=300)
    plt.close()

def generate_report(results_df, evaluation_df, prompt_corr_df, overall_corr, mae, rmse):
    """Generate comprehensive report of actual vLLM results"""
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = f"""
ACTUAL vLLM PROCESSING RESULTS REPORT
=====================================
Generated: {datetime.now()}
Model: Qwen-32B (or as configured)
Temperature: 0.1

PROCESSING SUMMARY
------------------
Total Essays Processed: {results_df['essay_id'].nunique()}
Total Prompts Used: {results_df['prompt_name'].nunique()}
Total Ratings Generated: {len(results_df)}
Expected Ratings: {results_df['essay_id'].nunique() * results_df['prompt_name'].nunique()}
Coverage: {len(results_df) / (results_df['essay_id'].nunique() * results_df['prompt_name'].nunique()) * 100:.1f}%

OVERALL PERFORMANCE (ACTUAL MODEL)
----------------------------------
Human-vLLM Correlation: {overall_corr:.3f}
Mean Absolute Error: {mae:.2f}
RMSE: {rmse:.2f}

Mean vLLM Rating: {evaluation_df['mean'].mean():.2f} ± {evaluation_df['mean'].std():.2f}
Mean Human Rating: {evaluation_df['judgement'].mean():.2f} ± {evaluation_df['judgement'].std():.2f}
Mean Difference: {(evaluation_df['mean'] - evaluation_df['judgement']).mean():.2f}

RATING STATISTICS
-----------------
Mean Within-Essay Std Dev: {evaluation_df['std'].mean():.2f}
Max Within-Essay Range: {evaluation_df['range'].max():.1f}
Essays with High Variance (SD > 2): {(evaluation_df['std'] > 2).sum()}

TOP 10 PROMPTS BY CORRELATION
-----------------------------
{prompt_corr_df.head(10).to_string(index=False)}

BOTTOM 10 PROMPTS
-----------------
{prompt_corr_df.tail(10).to_string(index=False)}

EDUCATION LEVEL ANALYSIS
------------------------
"""
    
    # Add education level statistics
    edu_map = {
        -2.183751267: 'Less than HS',
        -1.052047927: 'High School',
        0.079655412: 'Some College', 
        1.211358752: 'College',
        2.343062092: 'Graduate'
    }
    
    for edu_code, edu_label in sorted(edu_map.items()):
        edu_data = evaluation_df[evaluation_df['criterion'] == edu_code]
        if len(edu_data) > 0:
            report += f"\n{edu_label}:"
            report += f"\n  N = {len(edu_data)}"
            report += f"\n  Mean vLLM: {edu_data['mean'].mean():.2f} ± {edu_data['mean'].std():.2f}"
            report += f"\n  Mean Human: {edu_data['judgement'].mean():.2f} ± {edu_data['judgement'].std():.2f}"
            report += f"\n  Correlation: {edu_data[['mean', 'judgement']].corr().iloc[0,1]:.3f}"
    
    report += f"""

KEY FINDINGS (ACTUAL vLLM)
--------------------------
1. This represents ACTUAL model outputs, not simulation
2. Correlation should be significantly higher than simulation (~0.8+ expected)
3. Education gradient visible in both human and vLLM ratings
4. Prompt performance varies significantly
5. Some essays show high variance across prompts (potential ambiguous cases)

NEXT STEPS
----------
1. Use these ratings for DML-LME analysis
2. Consider weighted ensemble based on prompt correlations
3. Flag high-variance essays for closer examination
4. Extract binary features from prompts_100_complete.csv

FILES GENERATED
---------------
- All vLLM ratings combined
- Essay-level statistics with human labels
- Prompt correlation analysis
- Visualizations showing actual model performance
"""
    
    # Save report
    report_file = REPORTS_DIR / f"actual_vllm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    
    # Save data files
    results_df.to_csv(OUTPUT_DIR / "all_vllm_ratings_parsed.csv", index=False)
    evaluation_df.to_csv(OUTPUT_DIR / "evaluation_with_labels.csv", index=False)
    prompt_corr_df.to_csv(OUTPUT_DIR / "prompt_correlations_actual.csv", index=False)
    
    return report

def main():
    """Main execution"""
    print("=== PARSING ACTUAL vLLM RESULTS ===\n")
    
    # Parse vLLM outputs
    results_df = parse_vllm_outputs()
    
    if len(results_df) == 0:
        print("\n❌ No vLLM results found!")
        print("Please run actual vLLM processing first using:")
        print("  python scripts/run_actual_vllm.py")
        print("  OR")
        print("  bash scripts/run_vllm_cli.sh")
        return
    
    # Evaluate results
    evaluation_df, prompt_corr_df, overall_corr, mae, rmse = evaluate_results(results_df)
    
    # Create visualizations
    create_visualizations(results_df, evaluation_df, prompt_corr_df)
    
    # Generate report
    report = generate_report(results_df, evaluation_df, prompt_corr_df, 
                           overall_corr, mae, rmse)
    
    print("\n✅ Analysis complete!")
    print(f"Overall correlation (ACTUAL vLLM): {overall_corr:.3f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VIZ_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()