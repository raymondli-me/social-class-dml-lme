#!/usr/bin/env python3
"""
CPU-first approach for NV-Embed-v2 to avoid GPU memory issues
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import torch

print("="*60)
print("NV-EMBED-V2 EMBEDDING GENERATION (CPU-FIRST)")
print("="*60)

# Clear all GPU memory first
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    print("🧹 All GPU memory cleared")

# Load essays
print("\nLoading essays...")
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"

essays = pd.read_csv(ESSAYS_FILE)
print(f"Loaded {len(essays)} essays")

texts = essays['original'].tolist()
essay_ids = essays['TID'].values

try:
    print("\nLoading NV-Embed-v2 model on CPU first...")
    from transformers import AutoModel, AutoTokenizer
    
    # Load everything on CPU first
    model = AutoModel.from_pretrained(
        'nvidia/NV-Embed-v2', 
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Keep as float32 on CPU
    )
    
    tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    
    print("✓ Model loaded on CPU successfully!")
    
    # Now move to GPU 1 (avoid GPU 0 which might have display)
    device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    print(f"Moving model to {device}...")
    
    model = model.to(device, dtype=torch.float16)  # Convert to half precision on GPU
    model.eval()
    
    print(f"✓ Model moved to {device} with half precision!")
    
    # Generate embeddings with very small batches
    print("\nGenerating embeddings...")
    start_time = time.time()
    
    batch_size = 1  # Start with batch size 1
    instruction = "Instruct: Given a personal essay, analyze the social class indicators.\nQuery: "
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(texts)} texts... ({i/len(texts)*100:.1f}%)")
        
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            [instruction + text for text in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Even shorter to save memory
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                
                # Get embeddings
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.cpu().numpy())
                
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch {i}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                continue
        
        # Clear cache more frequently
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    if not all_embeddings:
        print("❌ No embeddings generated due to memory issues")
        sys.exit(1)
    
    embeddings = np.vstack(all_embeddings)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ Time elapsed: {elapsed/60:.1f} minutes")
    
    # Save embeddings
    output_file = "nvembed_embeddings.npy"
    np.save(output_file, embeddings)
    print(f"✓ Saved embeddings to {output_file}")
    
    np.save("nvembed_essay_ids.npy", essay_ids)
    print(f"✓ Saved essay IDs")
    
    print(f"\n✅ SUCCESS! Generated {len(all_embeddings)} embeddings")
    print(f"File size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Always clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU memory cleared")