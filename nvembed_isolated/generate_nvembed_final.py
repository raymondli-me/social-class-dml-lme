#!/usr/bin/env python3
"""
Final working version for NV-Embed-v2 with correct output handling
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
print("NV-EMBED-V2 EMBEDDING GENERATION (FINAL)")
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
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    
    print("✓ Model loaded successfully with automatic GPU distribution!")
    
    if hasattr(model, 'hf_device_map'):
        print(f"Device map: {model.hf_device_map}")
    
    model.eval()
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    start_time = time.time()
    
    batch_size = 4
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
            max_length=2048
        )
        
        # Move inputs to first GPU
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Debug: check output structure
            if i == 0:
                print(f"Output type: {type(outputs)}")
                if isinstance(outputs, dict):
                    print(f"Output keys: {outputs.keys()}")
                else:
                    print(f"Output attributes: {dir(outputs)}")
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # It's a dictionary - try common keys
                if 'pooler_output' in outputs and outputs['pooler_output'] is not None:
                    batch_embeddings = outputs['pooler_output']
                elif 'last_hidden_state' in outputs:
                    # Manual pooling
                    hidden_states = outputs['last_hidden_state']
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                elif 'hidden_states' in outputs:
                    hidden_states = outputs['hidden_states']
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                elif 'embedding' in outputs:
                    batch_embeddings = outputs['embedding']
                else:
                    # Try to find the actual embeddings in the dict
                    possible_keys = [k for k in outputs.keys() if 'embed' in k.lower() or 'hidden' in k.lower()]
                    if possible_keys:
                        print(f"Using key: {possible_keys[0]}")
                        batch_embeddings = outputs[possible_keys[0]]
                        if len(batch_embeddings.shape) == 3:  # [batch, seq, hidden]
                            # Do mean pooling
                            attention_mask = inputs['attention_mask']
                            mask_expanded = attention_mask.unsqueeze(-1).expand(batch_embeddings.size()).float()
                            sum_embeddings = torch.sum(batch_embeddings * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            batch_embeddings = sum_embeddings / sum_mask
                    else:
                        raise ValueError(f"Can't find embeddings in output keys: {outputs.keys()}")
            else:
                # Handle object-style outputs
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                else:
                    raise ValueError(f"Unknown output format: {type(outputs)}")
            
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