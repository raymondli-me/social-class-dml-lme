#!/usr/bin/env python3
"""
Run vLLM processing on ASC essays (9513 rows) with only the human MacArthur ladder prompt
This processes only the human rating prompt since ladder_standard was already done
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
from datetime import datetime
import json
from tqdm import tqdm
import re

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "asc_analysis_2prompts" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def prepare_asc_essays_for_vllm():
    """Prepare ASC essays in the format expected by vLLM processor"""
    print("\n=== PREPARING ASC ESSAYS ===")
    
    # Read the ASC essays (using the smaller file)
    essays_df = pd.read_csv(DATA_DIR / "essays_9513.csv")
    print(f"Loaded {len(essays_df)} essays")
    
    # Create improved prompts with clearer JSON instructions
    prompts_data = [
        {
            'system_prompt': 'You are evaluating essays written by 11-year-old children. Respond ONLY with valid JSON containing a single integer rating.',
            'pre_text_prompt': 'Read the following essay written by an 11-year-old child:\n\n',
            'post_text_prompt': '\n\nBased on this essay, imagine a ladder from 1 (lowest) to 10 (highest) representing society. Where would this child\'s family be placed on this ladder? Consider factors like education, job status, and financial situation.\n\nRespond with ONLY this JSON format:\n{"class_rating": X}\n\nWhere X is a number from 1 to 10.',
            'name': 'ladder_standard_improved'
        },
        {
            'system_prompt': 'You are evaluating essays written by 11-year-old children. Respond ONLY with valid JSON containing a single integer rating.',
            'pre_text_prompt': 'Read the following essay written by an 11-year-old child:\n\n',
            'post_text_prompt': '\n\nImagine a ladder that pictures how society is set up. At the top (10) are people who are the best off — they have the most money, highest education, and most respected jobs. At the bottom (1) are people who are worst off — they have the least money, little education, and no job or jobs that no one respects.\n\nWhere do you think the family of the child who wrote this essay would be on this ladder?\n\nRespond with ONLY this JSON format:\n{"class_rating": X}\n\nWhere X is a number from 1 to 10.',
            'name': 'human_macarthur_ladder_improved'
        }
    ]
    
    prompts_df = pd.DataFrame(prompts_data)
    print(f"\nUsing {len(prompts_df)} improved prompts:")
    for idx, row in prompts_df.iterrows():
        print(f"  {idx+1}. {row['name']}")
    
    return essays_df, prompts_df

def run_asc_vllm_processing():
    """Run vLLM processing on ASC essays with improved prompts"""
    print("=== ASC ESSAYS vLLM PROCESSING (IMPROVED JSON PROMPTS) ===")
    print(f"Started: {datetime.now()}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    essays_df, prompts_df = prepare_asc_essays_for_vllm()
    
    # Save processed inputs
    essays_df.to_csv(OUTPUT_DIR / "input_essays.csv", index=False)
    prompts_df.to_csv(OUTPUT_DIR / "input_prompts.csv", index=False)
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Processing {len(essays_df)} essays with {len(prompts_df)} prompts")
    print(f"Total ratings to generate: {len(essays_df) * len(prompts_df):,}")
    
    # Initialize vLLM with Qwen-32B
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
    )
    
    # Process each prompt
    all_results = []
    
    for prompt_idx, prompt_row in prompts_df.iterrows():
        prompt_name = prompt_row['name']
        print(f"\n[{prompt_idx+1}/{len(prompts_df)}] Processing prompt: {prompt_name}")
        
        # Prepare prompts for all essays
        prompts = []
        for _, essay_row in essays_df.iterrows():
            system_prompt = prompt_row['system_prompt'] if pd.notna(prompt_row['system_prompt']) else ""
            pre_text = prompt_row['pre_text_prompt']
            post_text = prompt_row['post_text_prompt'] if pd.notna(prompt_row['post_text_prompt']) else ""
            
            # Construct full prompt
            full_prompt = f"{system_prompt}\n{pre_text}\n{essay_row['original']}\n{post_text}".strip()
            prompts.append(full_prompt)
        
        # Generate in batches
        batch_size = 100  # Larger batch for 2 prompts test
        prompt_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_ids = essays_df['TID'].iloc[i:i+batch_size].tolist()
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            try:
                # Generate responses
                outputs = llm.generate(batch, sampling_params, use_tqdm=False)
                
                # Parse responses
                for j, output in enumerate(outputs):
                    response_text = output.outputs[0].text.strip()
                    
                    # Try to extract numeric rating
                    rating = None
                    
                    # First try to parse as JSON
                    try:
                        response_json = json.loads(response_text)
                        if 'class_rating' in response_json:
                            rating = int(response_json['class_rating'])
                    except:
                        # If JSON parsing fails, look for a number between 1-10
                        numbers = re.findall(r'\b([1-9]|10)\b', response_text)
                        if numbers:
                            rating = int(numbers[0])
                    
                    result = {
                        'essay_id': batch_ids[j],
                        'prompt_name': prompt_name,
                        'rating': rating,
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
    
    # Save final results
    final_df = pd.DataFrame(all_results)
    final_file = OUTPUT_DIR / f"all_results_{len(essays_df)}x{len(prompts_df)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_df.to_csv(final_file, index=False)
    
    print(f"\n✅ Processing complete!")
    print(f"Total results: {len(final_df)}")
    print(f"Successful ratings: {final_df['rating'].notna().sum()}")
    print(f"Failed ratings: {final_df['rating'].isna().sum()}")

if __name__ == "__main__":
    # Check if vLLM is available
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("ERROR: vLLM not found. Please install vLLM first.")
        import sys
        sys.exit(1)
    
    run_asc_vllm_processing()