#!/usr/bin/env python3
"""
Fixed NV-Embed-v2 script with proper memory management
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import torch

# Clear GPU memory first
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"

print("="*60)
print("NV-EMBED-V2 EMBEDDING GENERATION (FIXED)")
print("="*60)

# Load essays
print("\nLoading essays...")
essays = pd.read_csv(ESSAYS_FILE)
print(f"Loaded {len(essays)} essays")
assert len(essays) == 9513, f"Expected 9513 essays, got {len(essays)}"

texts = essays['original'].tolist()
essay_ids = essays['TID'].values

try:
    print("\nLoading NV-Embed-v2 model...")
    from transformers import AutoModel, AutoTokenizer
    
    # Load model with memory optimization
    model = AutoModel.from_pretrained(
        'nvidia/NV-Embed-v2', 
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use half precision
        low_cpu_mem_usage=True,
        device_map="auto"  # Automatically distribute across GPUs
    )
    
    tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    
    print("✓ Model loaded successfully with automatic GPU distribution!")
    
    # Check GPU distribution
    if hasattr(model, 'hf_device_map'):
        print(f"Device map: {model.hf_device_map}")
    
    model.eval()
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    start_time = time.time()
    
    # Use smaller batch size to avoid memory issues
    batch_size = 4  # Conservative batch size
    instruction = "Instruct: Given a personal essay, analyze the social class indicators.\nQuery: "
    
    print("Using manual encoding with small batches...")
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(texts)} texts... ({i/len(texts)*100:.1f}%)")
        
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            [instruction + text for text in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Reduced from 4096 to save memory
        )
        
        # Move inputs to first GPU (model handles distribution automatically)
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get embeddings
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_embeddings = outputs.pooler_output
            else:
                # Use mean pooling
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
            
            # Normalize and move to CPU
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Clear GPU cache periodically
        if i % 1000 == 0:
            torch.cuda.empty_cache()
    
    embeddings = np.vstack(all_embeddings)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ Time elapsed: {elapsed/60:.1f} minutes")
    
    # Save embeddings
    output_file = "nvembed_embeddings.npy"
    np.save(output_file, embeddings)
    print(f"✓ Saved embeddings to {output_file}")
    
    # Also save essay IDs
    np.save("nvembed_essay_ids.npy", essay_ids)
    print(f"✓ Saved essay IDs for verification")
    
    print("\n✅ SUCCESS! Embeddings generated and saved.")
    print(f"File size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Copy to main directory
    print("\nCopying to main checkpoint directory...")
    import shutil
    target_dir = Path("../nvembed_checkpoints")
    target_dir.mkdir(exist_ok=True)
    
    shutil.copy(output_file, target_dir / output_file)
    shutil.copy("nvembed_essay_ids.npy", target_dir / "nvembed_essay_ids.npy")
    print(f"✓ Copied to {target_dir}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    # Clear GPU memory on error
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU memory cleared")