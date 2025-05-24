#!/usr/bin/env python3
"""
Test script for social class analysis using vLLM batch processor
Tests on 10 sample essays with all 100 prompts
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add vllm-batch-processor to path (update this path as needed)
sys.path.append('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor')

from vllm_batch_processor import BatchProcessor, ProcessorConfig, PromptConfig

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "test"

def prepare_test_data():
    """Prepare test dataset with 10 samples"""
    print("Loading essay dataset...")
    essays_df = pd.read_csv(DATA_DIR / "essay_dataset.csv")
    
    # Take first 10 essays
    test_essays = essays_df.head(10).copy()
    
    # Rename columns to match vllm-batch-processor format
    test_essays = test_essays.rename(columns={
        'TID': 'id',
        'original': 'text'
    })[['id', 'text']]
    
    # Save test dataset
    test_file = OUTPUT_DIR / "test_essays_10.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_essays.to_csv(test_file, index=False)
    print(f"Created test dataset with {len(test_essays)} essays")
    
    return test_file

def run_batch_processing(input_file):
    """Run vLLM batch processing with all prompts"""
    print("\nInitializing batch processor...")
    
    # Configure processor
    config = ProcessorConfig(
        preset="qwen-32b",  # Using 32B model for better quality test results
        batch_size=5,
        max_new_tokens=100,
        temperature=0.1,
        output_dir=str(OUTPUT_DIR),
        save_every_n_batches=10
    )
    
    # Initialize processor
    processor = BatchProcessor(config)
    
    # Load prompts
    prompts_file = DATA_DIR / "prompts_100_complete.csv"
    
    # Process with all prompts
    print(f"Processing with prompts from: {prompts_file}")
    output_files = processor.process_file(
        str(input_file),
        prompt_config_file=str(prompts_file),
        run_name="test_10_samples"
    )
    
    print(f"\nProcessing complete! Output files: {output_files}")
    return output_files

def analyze_results(output_files):
    """Quick analysis of test results"""
    print("\n=== Test Results Analysis ===")
    
    for output_file in output_files:
        if not os.path.exists(output_file):
            continue
            
        df = pd.read_csv(output_file)
        prompt_name = Path(output_file).stem.split('_')[-1]
        
        print(f"\nPrompt: {prompt_name}")
        print(f"Total responses: {len(df)}")
        
        # Show sample responses
        if len(df) > 0:
            print("Sample responses:")
            for idx in range(min(3, len(df))):
                print(f"  Essay {df.iloc[idx]['id']}: {df.iloc[idx]['response'][:100]}...")

def main():
    """Main test execution"""
    print("=== Social Class Analysis Test (10 Samples) ===\n")
    
    # Prepare test data
    test_file = prepare_test_data()
    
    # Run batch processing
    try:
        output_files = run_batch_processing(test_file)
        
        # Analyze results
        analyze_results(output_files)
        
        print("\n✅ Test completed successfully!")
        print(f"Results saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()