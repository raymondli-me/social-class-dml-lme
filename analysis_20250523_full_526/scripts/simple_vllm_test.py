#!/usr/bin/env python3
"""
Simplified vLLM script for testing without batch processor
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use GPUs 0 and 1

import pandas as pd
from vllm import LLM, SamplingParams
from pathlib import Path
import json
from datetime import datetime

# Disable eager mode to avoid compilation
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'

def test_vllm():
    """Test basic vLLM functionality"""
    print("Testing basic vLLM with Qwen-32B...")
    
    # Initialize with simpler settings
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        quantization="awq",
        enforce_eager=False,  # Disable eager mode
        trust_remote_code=True
    )
    
    # Test with a simple prompt
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50
    )
    
    test_prompt = """Please rate the perceived social class of the person who wrote the following text on a scale from 1 to 10, where 1 represents lower class and 10 represents upper class. Consider factors like education, occupation, lifestyle, and economic circumstances.

Text: "I grew up in a small town where my parents worked at the local factory. We didn't have much, but we were happy."

Please respond with ONLY a number from 1 to 10."""
    
    print("\nGenerating response...")
    outputs = llm.generate([test_prompt], sampling_params)
    
    for output in outputs:
        print(f"\nGenerated text: {output.outputs[0].text}")
    
    print("\n✅ Basic vLLM test successful!")
    return True

if __name__ == "__main__":
    test_vllm()