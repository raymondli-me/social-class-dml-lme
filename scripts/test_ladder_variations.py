#!/usr/bin/env python3
"""
Test ladder variations: 50 different social class measurement prompts
Analyzes consistency and correlation with human judgments
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add vllm-batch-processor to path
sys.path.append('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor')

from vllm_batch_processor import BatchProcessor, ProcessorConfig

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "ladder_test"
VIZ_DIR = BASE_DIR / "visualizations" / "ladder_test"

def prepare_test_data():
    """Prepare test dataset with 10 samples"""
    print("Loading essay dataset...")
    essays_df = pd.read_csv(DATA_DIR / "essay_dataset.csv")
    
    # Take first 10 essays
    test_essays = essays_df.head(10).copy()
    
    # Rename columns for vllm-batch-processor
    test_essays = test_essays.rename(columns={
        'TID': 'id',
        'original': 'text'
    })
    
    # Keep all columns for analysis
    test_data = test_essays[['id', 'text', 'criterion', 'judgement']]
    
    # Save test dataset
    test_file = OUTPUT_DIR / "test_essays_ladder.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_data[['id', 'text']].to_csv(test_file, index=False)
    
    print(f"Created test dataset with {len(test_data)} essays")
    print(f"Education levels: {test_data['criterion'].value_counts().to_dict()}")
    print(f"Human ratings range: {test_data['judgement'].min()}-{test_data['judgement'].max()}")
    
    return test_file, test_data

def run_ladder_processing(input_file):
    """Run vLLM processing with ladder variation prompts"""
    print("\nInitializing batch processor with Qwen-32B...")
    
    # Configure processor
    config = ProcessorConfig(
        preset="qwen-32b",
        batch_size=5,
        max_new_tokens=50,  # Just need a number 1-10
        temperature=0.1,
        output_dir=str(OUTPUT_DIR),
        save_every_n_batches=10
    )
    
    # Initialize processor
    processor = BatchProcessor(config)
    
    # Use ladder variations prompts
    prompts_file = DATA_DIR / "ladder_variations_50_complete.csv"
    
    # Process with all ladder prompts
    print(f"Processing with 50 ladder variation prompts...")
    output_files = processor.process_file(
        str(input_file),
        prompt_config_file=str(prompts_file),
        run_name="ladder_variations_test"
    )
    
    print(f"\nProcessing complete! Generated {len(output_files)} output files")
    return output_files

def parse_results(output_files, test_data):
    """Parse results and create analysis dataframe"""
    print("\n=== Parsing Results ===")
    
    results = []
    
    for output_file in output_files:
        if not os.path.exists(output_file):
            continue
            
        df = pd.read_csv(output_file)
        prompt_name = Path(output_file).stem.split('_')[-1]
        
        for _, row in df.iterrows():
            try:
                # Parse JSON response
                response = json.loads(row['response'])
                
                # Extract rating (handle different response formats)
                if isinstance(response, dict):
                    # Get the first numeric value from the dict
                    rating = next(iter(response.values()))
                else:
                    rating = float(response)
                
                # Ensure rating is in 1-10 range
                rating = max(1, min(10, rating))
                
                results.append({
                    'essay_id': row['id'],
                    'prompt_name': prompt_name,
                    'rating': rating
                })
            except Exception as e:
                print(f"Error parsing response for essay {row['id']}, prompt {prompt_name}: {e}")
                results.append({
                    'essay_id': row['id'],
                    'prompt_name': prompt_name,
                    'rating': np.nan
                })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Pivot to wide format (essays as rows, prompts as columns)
    ratings_matrix = results_df.pivot(
        index='essay_id',
        columns='prompt_name',
        values='rating'
    )
    
    # Merge with test data
    final_df = test_data.merge(
        ratings_matrix,
        left_on='id',
        right_index=True,
        how='inner'
    )
    
    print(f"Parsed {len(results_df)} total ratings")
    print(f"Rating matrix shape: {ratings_matrix.shape}")
    
    return final_df, ratings_matrix

def analyze_results(final_df, ratings_matrix):
    """Comprehensive analysis of ladder variation results"""
    print("\n=== Analyzing Results ===")
    
    # Create visualizations directory
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Calculate statistics for each essay
    prompt_cols = [col for col in final_df.columns if col not in ['id', 'text', 'criterion', 'judgement']]
    
    stats_df = pd.DataFrame({
        'essay_id': final_df['id'],
        'human_rating': final_df['judgement'],
        'education': final_df['criterion'],
        'mean_llm_rating': final_df[prompt_cols].mean(axis=1),
        'std_llm_rating': final_df[prompt_cols].std(axis=1),
        'min_llm_rating': final_df[prompt_cols].min(axis=1),
        'max_llm_rating': final_df[prompt_cols].max(axis=1),
        'range_llm_rating': final_df[prompt_cols].max(axis=1) - final_df[prompt_cols].min(axis=1)
    })
    
    # 2. Calculate correlation of each prompt with human judgment
    correlations = []
    for prompt in prompt_cols:
        corr = final_df[[prompt, 'judgement']].corr().iloc[0, 1]
        correlations.append({
            'prompt': prompt,
            'correlation': corr
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    # 3. Create visualizations
    
    # Distribution of ratings across all prompts
    plt.figure(figsize=(12, 6))
    all_ratings = final_df[prompt_cols].values.flatten()
    plt.hist(all_ratings[~np.isnan(all_ratings)], bins=10, edgecolor='black', alpha=0.7)
    plt.title('Distribution of All LLM Social Class Ratings')
    plt.xlabel('Rating (1-10)')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 11))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'rating_distribution.png', dpi=300)
    plt.close()
    
    # Heatmap of ratings matrix
    plt.figure(figsize=(20, 8))
    sns.heatmap(ratings_matrix.T, cmap='RdYlBu_r', center=5.5, 
                cbar_kws={'label': 'Social Class Rating'},
                xticklabels=[f"Essay {i}" for i in range(1, 11)])
    plt.title('Social Class Ratings: 50 Prompt Variations × 10 Essays')
    plt.xlabel('Essay')
    plt.ylabel('Prompt Variation')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'ratings_heatmap.png', dpi=300)
    plt.close()
    
    # Correlation with human judgment
    plt.figure(figsize=(10, 12))
    plt.barh(corr_df['prompt'][:20], corr_df['correlation'][:20])
    plt.xlabel('Correlation with Human Judgment')
    plt.title('Top 20 Prompts by Correlation with Human Ratings')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'prompt_correlations.png', dpi=300)
    plt.close()
    
    # Variance by education level
    plt.figure(figsize=(10, 6))
    stats_df.boxplot(column='std_llm_rating', by='education', rot=45)
    plt.suptitle('Rating Variance by Education Level')
    plt.title('')
    plt.ylabel('Standard Deviation of Ratings')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'variance_by_education.png', dpi=300)
    plt.close()
    
    # Human vs LLM mean comparison
    plt.figure(figsize=(8, 8))
    plt.scatter(stats_df['human_rating'], stats_df['mean_llm_rating'], 
                s=100, alpha=0.6, edgecolors='black')
    plt.plot([1, 10], [1, 10], 'r--', label='Perfect Agreement')
    plt.xlabel('Human Rating')
    plt.ylabel('Mean LLM Rating (across 50 prompts)')
    plt.title('Human vs LLM Social Class Ratings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'human_vs_llm_scatter.png', dpi=300)
    plt.close()
    
    return stats_df, corr_df

def generate_report(stats_df, corr_df, ratings_matrix):
    """Generate analysis report"""
    report = f"""
LADDER VARIATIONS TEST RESULTS
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
------------------
Total Essays Analyzed: {len(stats_df)}
Total Prompts Used: {ratings_matrix.shape[1]}
Total Ratings Generated: {ratings_matrix.size}

OVERALL FINDINGS
----------------
Mean LLM Rating (all essays, all prompts): {ratings_matrix.mean().mean():.2f}
Std Dev of All Ratings: {ratings_matrix.values.flatten().std():.2f}
Range of All Ratings: {ratings_matrix.min().min():.0f} - {ratings_matrix.max().max():.0f}

HUMAN VS LLM AGREEMENT
----------------------
Correlation (mean LLM vs human): {stats_df[['human_rating', 'mean_llm_rating']].corr().iloc[0,1]:.3f}
Mean Absolute Difference: {abs(stats_df['human_rating'] - stats_df['mean_llm_rating']).mean():.2f}

TOP 5 PROMPTS BY CORRELATION WITH HUMAN JUDGMENT
------------------------------------------------
{corr_df.head().to_string(index=False)}

BOTTOM 5 PROMPTS BY CORRELATION
--------------------------------
{corr_df.tail().to_string(index=False)}

RATING CONSISTENCY
------------------
Mean Within-Essay Std Dev: {stats_df['std_llm_rating'].mean():.2f}
Max Within-Essay Range: {stats_df['range_llm_rating'].max():.0f}
Most Consistent Essay: ID {stats_df.loc[stats_df['std_llm_rating'].idxmin(), 'essay_id']} (SD={stats_df['std_llm_rating'].min():.2f})
Most Variable Essay: ID {stats_df.loc[stats_df['std_llm_rating'].idxmax(), 'essay_id']} (SD={stats_df['std_llm_rating'].max():.2f})

VARIANCE BY EDUCATION LEVEL
---------------------------
{stats_df.groupby('education')['std_llm_rating'].agg(['mean', 'std']).round(2).to_string()}

"""
    
    # Save report
    report_file = OUTPUT_DIR / f"ladder_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    
    # Also save data files
    stats_df.to_csv(OUTPUT_DIR / "essay_statistics.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "prompt_correlations.csv", index=False)
    ratings_matrix.to_csv(OUTPUT_DIR / "ratings_matrix.csv")
    
    return report

def main():
    """Main execution"""
    print("=== Ladder Variations Social Class Test ===\n")
    
    try:
        # Prepare test data
        test_file, test_data = prepare_test_data()
        
        # Run ladder processing
        output_files = run_ladder_processing(test_file)
        
        # Parse results
        final_df, ratings_matrix = parse_results(output_files, test_data)
        
        # Analyze results
        stats_df, corr_df = analyze_results(final_df, ratings_matrix)
        
        # Generate report
        report = generate_report(stats_df, corr_df, ratings_matrix)
        
        print("\n✅ Ladder variations test completed successfully!")
        print(f"Results saved in: {OUTPUT_DIR}")
        print(f"Visualizations saved in: {VIZ_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()