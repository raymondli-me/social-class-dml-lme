#!/usr/bin/env python3
"""
Run ACTUAL vLLM processing on 526 essays with 50 prompts
This uses the real vllm-batch-processor, not simulation
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add vllm-batch-processor to path
sys.path.append('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor')

from vllm_batch_processor import BatchProcessor, ProcessorConfig
from vllm_batch_processor.parallel_runner import ParallelModelRunner

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "vllm_actual"

def run_actual_vllm_processing():
    """Run ACTUAL vLLM processing on all 526 essays"""
    print("=== ACTUAL vLLM PROCESSING (NOT SIMULATION) ===")
    print(f"Started: {datetime.now()}")
    
    # Input files
    essays_file = DATA_DIR / "essays_blinded_526.csv"
    prompts_file = DATA_DIR / "ladder_variations_50_complete.csv"
    
    print(f"\nInput essays: {essays_file}")
    print(f"Prompts: {prompts_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Configure vLLM processor
    # Using Qwen-32B for good balance of quality and speed
    from vllm_batch_processor.models import get_model_preset
    preset = get_model_preset("qwen-32b")
    
    config = ProcessorConfig(
        model_name=preset.model_path,
        quantization=preset.quantization,
        tensor_parallel_size=preset.tensor_parallel_size,
        batch_size=10,      # Process 10 essays at a time
        max_tokens=50,      # Just need a number 1-10
        temperature=0.1,    # Low temperature for consistency
        output_dir=str(OUTPUT_DIR),
        max_model_len=preset.max_model_len,
        gpu_memory_utilization=preset.gpu_memory_utilization,
        prompt_config_path=str(prompts_file)  # Add the prompt configuration file
    )
    
    # Initialize processor
    print("\nInitializing vLLM with Qwen-32B...")
    processor = BatchProcessor(config)
    
    # Process essays with all prompts
    print(f"\nProcessing {526} essays × {50} prompts = {26,300} ratings")
    print("This will take approximately 30-60 minutes depending on GPU...")
    
    try:
        result = processor.process_file(
            input_file=str(essays_file)
        )
        
        # Get output files from the result
        output_dir = Path(result['output_file'])
        output_files = list(output_dir.glob("*/results_*.csv"))
        
        print(f"\n✅ vLLM processing complete!")
        print(f"Generated {len(output_files)} output files")
        
        # List output files
        print("\nOutput files:")
        for f in output_files:
            print(f"  - {Path(f).name}")
            
        return output_files
        
    except Exception as e:
        print(f"\n❌ Error during vLLM processing: {e}")
        raise

def parse_vllm_outputs(output_files):
    """Parse vLLM output files into analysis-ready format"""
    print("\n=== Parsing vLLM Outputs ===")
    
    all_results = []
    
    for output_file in output_files:
        if not Path(output_file).exists():
            continue
            
        # Read vLLM output
        df = pd.read_csv(output_file)
        prompt_name = Path(output_file).stem.split('_')[-1]
        
        print(f"Processing {prompt_name}: {len(df)} responses")
        
        # Parse each response
        for _, row in df.iterrows():
            try:
                import json
                response = json.loads(row['response'])
                
                # Extract rating from JSON response
                if isinstance(response, dict):
                    # Get the rating value from the JSON
                    rating_key = next(iter(response.keys()))
                    rating = float(response[rating_key])
                else:
                    rating = float(response)
                
                # Ensure rating is in valid range
                rating = max(1, min(10, rating))
                
                all_results.append({
                    'essay_id': row['id'],
                    'prompt_name': prompt_name,
                    'rating': rating,
                    'raw_response': row['response'],
                    'timestamp': row.get('timestamp', '')
                })
                
            except Exception as e:
                print(f"  Warning: Failed to parse response for essay {row['id']}: {e}")
                all_results.append({
                    'essay_id': row['id'],
                    'prompt_name': prompt_name,
                    'rating': None,
                    'raw_response': row['response'],
                    'timestamp': row.get('timestamp', '')
                })
    
    # Create combined dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save combined results
    combined_file = OUTPUT_DIR / "all_vllm_results_526x50.csv"
    results_df.to_csv(combined_file, index=False)
    
    print(f"\nParsed {len(results_df)} total ratings")
    print(f"Missing/failed: {results_df['rating'].isna().sum()}")
    print(f"Combined results saved to: {combined_file}")
    
    return results_df

def create_evaluation_report(results_df):
    """Create evaluation report comparing to human judgments"""
    print("\n=== Creating Evaluation Report ===")
    
    # Load hidden labels for evaluation
    labels_df = pd.read_csv(DATA_DIR / "labels_hidden_526.csv")
    
    # Calculate essay-level statistics
    essay_stats = results_df.groupby('essay_id')['rating'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Merge with labels
    evaluation_df = essay_stats.merge(labels_df, left_on='essay_id', right_on='id')
    
    # Calculate overall correlation
    overall_corr = evaluation_df[['mean', 'judgement']].corr().iloc[0, 1]
    
    # Prompt-wise correlations
    prompt_corrs = []
    for prompt in results_df['prompt_name'].unique():
        prompt_data = results_df[results_df['prompt_name'] == prompt]
        merged = prompt_data.merge(labels_df, left_on='essay_id', right_on='id')
        if len(merged) > 10:  # Need enough data for correlation
            corr = merged[['rating', 'judgement']].corr().iloc[0, 1]
            prompt_corrs.append({
                'prompt': prompt,
                'correlation': corr,
                'n': len(merged)
            })
    
    prompt_corr_df = pd.DataFrame(prompt_corrs).sort_values('correlation', ascending=False)
    
    # Generate report
    report = f"""
ACTUAL vLLM PROCESSING RESULTS
==============================
Generated: {datetime.now()}

Processing Summary
------------------
Total Essays: {len(essay_stats)}
Total Prompts: {results_df['prompt_name'].nunique()}
Total Ratings: {len(results_df)}
Failed Ratings: {results_df['rating'].isna().sum()}

Overall Performance
-------------------
Human-vLLM Correlation: {overall_corr:.3f}
Mean vLLM Rating: {results_df['rating'].mean():.2f}
Mean Human Rating: {labels_df['judgement'].mean():.2f}

Top 5 Prompts by Correlation
----------------------------
{prompt_corr_df.head().to_string(index=False)}

Bottom 5 Prompts
----------------
{prompt_corr_df.tail().to_string(index=False)}

This represents ACTUAL vLLM outputs, not simulation!
"""
    
    # Save report
    report_file = OUTPUT_DIR / f"vllm_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    
    # Save evaluation data
    evaluation_df.to_csv(OUTPUT_DIR / "evaluation_with_vllm.csv", index=False)
    prompt_corr_df.to_csv(OUTPUT_DIR / "prompt_correlations_vllm.csv", index=False)

def main():
    """Main execution"""
    print("=== RUNNING ACTUAL vLLM (NOT SIMULATION) ===\n")
    
    # Check if we have vLLM outputs already
    output_dir = OUTPUT_DIR
    existing_outputs = list(output_dir.glob("*.csv")) if output_dir.exists() else []
    
    if existing_outputs and any("full_526_essays" in f.name for f in existing_outputs):
        print("Found existing vLLM outputs. Parsing those...")
        output_files = [str(f) for f in existing_outputs if "full_526_essays" in f.name]
    else:
        print("No existing outputs found. Running vLLM processing...")
        print("\n⚠️  This will take 30-60 minutes with actual model inference!")
        print("Starting automatically...")
            
        # Run actual vLLM
        output_files = run_actual_vllm_processing()
    
    # Parse outputs
    results_df = parse_vllm_outputs(output_files)
    
    # Create evaluation report
    create_evaluation_report(results_df)
    
    print(f"\n✅ Complete! Check {OUTPUT_DIR} for results")

if __name__ == "__main__":
    main()