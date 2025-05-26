#!/usr/bin/env python3
"""
Minimal script to generate NV-Embed-v2 embeddings only.
Requires: transformers==4.42.4
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
ESSAYS_FILE = DATA_DIR / "asc_9513_essays.csv"

print("="*60)
print("NV-EMBED-V2 EMBEDDING GENERATION")
print("="*60)

# Check transformers version
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    if transformers.__version__ != "4.42.4":
        print("WARNING: This script requires transformers==4.42.4")
        print("Install with: pip install transformers==4.42.4")
except ImportError:
    print("ERROR: transformers not installed")
    sys.exit(1)

# Load essays
print("\nLoading essays...")
essays = pd.read_csv(ESSAYS_FILE)
print(f"Loaded {len(essays)} essays")
assert len(essays) == 9513, f"Expected 9513 essays, got {len(essays)}"

texts = essays['original'].tolist()
essay_ids = essays['TID'].values

# Try to load NV-Embed-v2
try:
    print("\nLoading NV-Embed-v2 model...")
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using device: {device}")
    print(f"Available GPUs: {n_gpus}")
    
    if torch.cuda.is_available():
        # Try to use DataParallel for multi-GPU
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    start_time = time.time()
    
    # Use the model's encode method if available
    instruction = "Instruct: Given a personal essay, analyze the social class indicators.\nQuery: "
    
    if hasattr(model, 'encode'):
        print("Using model.encode() method...")
        embeddings = model.encode(texts, instruction=instruction, max_length=4096)
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()
    else:
        print("Using manual encoding...")
        # Manual batch processing
        # Increase batch size based on GPU count
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        batch_size = 8 * n_gpus  # 8 per GPU, so 32 total with 4 GPUs
        print(f"Using batch size: {batch_size}")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(texts)} texts...")
            
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                [instruction + text for text in batch_texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            
            if device == 'cuda':
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ Time elapsed: {elapsed/60:.1f} minutes")
    
    # Save embeddings
    output_file = "nvembed_embeddings.npy"
    np.save(output_file, embeddings)
    print(f"✓ Saved embeddings to {output_file}")
    
    # Also save essay IDs for alignment
    np.save("nvembed_essay_ids.npy", essay_ids)
    print(f"✓ Saved essay IDs for verification")
    
    print("\n✅ SUCCESS! Embeddings generated and saved.")
    print(f"Next step: Copy {output_file} to ../nvembed_checkpoints/")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure transformers==4.42.4 is installed")
    print("2. Check CUDA availability if using GPU")
    print("3. Try with smaller batch size if memory error")
    import traceback
    traceback.print_exc()