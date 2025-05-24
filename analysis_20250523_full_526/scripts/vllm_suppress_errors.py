#!/usr/bin/env python3
"""
Run vLLM with error suppression for compilation issues
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

def run_with_suppression():
    """Run vLLM with error suppression"""
    print("Running vLLM with error suppression...")
    
    # Initialize with error suppression
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        quantization="awq",
        enforce_eager=True,  # Keep eager mode but suppress errors
        trust_remote_code=True,
        disable_custom_all_reduce=True,  # Disable custom operations
        dtype="float16"
    )
    
    print("✅ Model loaded successfully!")
    
    # Test generation
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=10
    )
    
    test_prompt = "Rate this person's social class from 1-10: 'I work at a factory.' Answer with just a number:"
    
    outputs = llm.generate([test_prompt], sampling_params)
    print(f"Test output: {outputs[0].outputs[0].text}")
    
    return llm

if __name__ == "__main__":
    llm = run_with_suppression()