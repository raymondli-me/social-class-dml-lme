#!/usr/bin/env python3
"""
Run full vLLM processing with error suppression
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Suppress compilation errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Disable compilation-heavy features
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import pandas as pd
from vllm import LLM, SamplingParams
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import shutil

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Create timestamped output directory for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / "outputs" / "vllm_actual_suppressed" / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory for this run: {OUTPUT_DIR.name}/")

def run_vllm_processing():
    """Run vLLM processing with error suppression"""
    print("=== ACTUAL vLLM PROCESSING WITH ERROR SUPPRESSION ===")
    print(f"Started: {datetime.now()}")
    
    # Load data
    essays_file = DATA_DIR / "essays_blinded_526.csv"
    prompts_file = DATA_DIR / "ladder_variations_51_human_first_fixed.csv"
    
    essays_df = pd.read_csv(essays_file)
    prompts_df = pd.read_csv(prompts_file)
    
    print(f"\nLoaded {len(essays_df)} essays and {len(prompts_df)} prompts")
    print(f"Total ratings to generate: {len(essays_df) * len(prompts_df):,}")
    
    # Initialize vLLM with error suppression
    print("\nInitializing vLLM with Qwen-32B-AWQ...")
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        quantization="awq",
        enforce_eager=True,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        dtype="float16"
    )
    print("✅ Model loaded successfully!")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=200  # Increased to allow complete responses
        # Removed stop tokens - let model generate complete response
    )
    
    # Process each prompt
    all_results = []
    
    for prompt_idx, prompt_row in prompts_df.iterrows():
        prompt_name = prompt_row['name']
        print(f"\n[{prompt_idx+1}/{len(prompts_df)}] Processing prompt: {prompt_name}")
        
        # Prepare prompts for all essays
        prompts = []
        for _, essay_row in essays_df.iterrows():
            full_prompt = f"{prompt_row['system_prompt']}\n{prompt_row['pre_text_prompt']}\n{essay_row['text']}\n{prompt_row['post_text_prompt']}"
            prompts.append(full_prompt)
        
        # Generate in batches
        batch_size = 50
        prompt_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_ids = essays_df['id'].iloc[i:i+batch_size].tolist()
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            try:
                # Generate responses (disable tqdm for nohup compatibility)
                outputs = llm.generate(batch, sampling_params, use_tqdm=False)
                
                # Parse responses
                for j, output in enumerate(outputs):
                    response_text = output.outputs[0].text.strip()
                    
                    # Don't parse - just save raw response
                    result = {
                        'essay_id': batch_ids[j],
                        'prompt_name': prompt_name,
                        'rating': None,  # We'll parse this later
                        'raw_response': response_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    prompt_results.append(result)
                    all_results.append(result)
                    
            except Exception as e:
                print(f"  Error in batch: {e}")
                # Add empty results for failed batch
                for j in range(len(batch)):
                    result = {
                        'essay_id': batch_ids[j],
                        'prompt_name': prompt_name,
                        'rating': None,
                        'raw_response': f"ERROR: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    }
                    prompt_results.append(result)
                    all_results.append(result)
                continue
        
        # Save prompt-specific results
        prompt_df = pd.DataFrame(prompt_results)
        prompt_file = OUTPUT_DIR / f"results_{prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        prompt_df.to_csv(prompt_file, index=False)
        print(f"Saved {len(prompt_results)} results to {prompt_file}")
        
        # Save all results periodically
        if (prompt_idx + 1) % 10 == 0:
            all_df = pd.DataFrame(all_results)
            checkpoint_file = OUTPUT_DIR / f"checkpoint_{prompt_idx+1}_prompts.csv"
            all_df.to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved: {checkpoint_file}")
    
    # Save final results
    final_df = pd.DataFrame(all_results)
    final_file = OUTPUT_DIR / f"all_results_526x50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_df.to_csv(final_file, index=False)
    
    print(f"\n✅ Processing complete!")
    print(f"Total results: {len(final_df)}")
    print(f"Failed ratings: {final_df['rating'].isna().sum()}")
    print(f"Final results saved to: {final_file}")
    
    return final_df

def create_evaluation_report(results_df):
    """Create evaluation report"""
    print("\n=== Creating Evaluation Report ===")
    
    # Load hidden labels
    labels_df = pd.read_csv(DATA_DIR / "labels_hidden_526.csv")
    
    # Calculate essay-level statistics
    essay_stats = results_df.groupby('essay_id')['rating'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Merge with labels
    evaluation_df = essay_stats.merge(labels_df, left_on='essay_id', right_on='id')
    
    # Calculate overall correlation
    mask = evaluation_df['mean'].notna()
    if mask.sum() > 10:
        overall_corr = evaluation_df[mask][['mean', 'judgement']].corr().iloc[0, 1]
    else:
        overall_corr = 0.0
    
    print(f"\nOverall Human-vLLM Correlation: {overall_corr:.3f}")
    print(f"Mean vLLM Rating: {results_df['rating'].mean():.2f}")
    print(f"Mean Human Rating: {labels_df['judgement'].mean():.2f}")
    
    # Save evaluation
    eval_file = OUTPUT_DIR / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(eval_file, 'w') as f:
        f.write(f"vLLM Evaluation Report\n")
        f.write(f"=====================\n\n")
        f.write(f"Overall Correlation: {overall_corr:.3f}\n")
        f.write(f"Total Essays: {len(evaluation_df)}\n")
        f.write(f"Valid Ratings: {mask.sum()}\n")
        f.write(f"Mean vLLM: {results_df['rating'].mean():.2f}\n")
        f.write(f"Mean Human: {labels_df['judgement'].mean():.2f}\n")
    
    return overall_corr

if __name__ == "__main__":
    # Run processing
    results = run_vllm_processing()
    
    # Create evaluation
    if len(results) > 0:
        correlation = create_evaluation_report(results)
        print(f"\n✅ Final correlation: {correlation:.3f}")
    
    print("\n✅ All done!")