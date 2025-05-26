#!/usr/bin/env python3
"""
Extract embeddings from Qwen model using transformers
Simple approach - just load the model and extract hidden states
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import time

print("="*80)
print("QWEN-3B EMBEDDING EXTRACTION")
print("="*80)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Smallest Qwen model
BATCH_SIZE = 8  # Adjust based on GPU memory
MAX_LENGTH = 1024  # Truncate long essays

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "qwen_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load essays
print("\nLoading essays...")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
print(f"Loaded {len(essays)} essays")

# For testing, start with a subset
TEST_MODE = True
if TEST_MODE:
    essays = essays.head(100)  # Just 100 for testing
    print(f"TEST MODE: Using only {len(essays)} essays")

# Load model
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Ensure padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Extract embeddings
texts = essays['original'].tolist()
essay_ids = essays['TID'].values
all_embeddings = []

print(f"\nExtracting embeddings...")
start_time = time.time()

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Move to GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        
        # Mean pooling over sequence length
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        
        # Compute mean, ignoring padding
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        mean_pooled = sum_hidden / sum_mask.clamp(min=1e-9)
        
        all_embeddings.append(mean_pooled.cpu().float().numpy())

# Combine all embeddings
embeddings = np.vstack(all_embeddings)
elapsed = time.time() - start_time

print(f"\nDone!")
print(f"Time: {elapsed:.1f} seconds ({len(texts)/elapsed:.1f} essays/sec)")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Dimensions: {embeddings.shape[1]}")

# Save
if TEST_MODE:
    np.save(OUTPUT_DIR / "qwen3b_embeddings_test.npy", embeddings)
    np.save(OUTPUT_DIR / "qwen3b_essay_ids_test.npy", essay_ids)
    print(f"\nSaved test embeddings to {OUTPUT_DIR}")
else:
    np.save(OUTPUT_DIR / "qwen3b_embeddings.npy", embeddings)
    np.save(OUTPUT_DIR / "qwen3b_essay_ids.npy", essay_ids)
    print(f"\nSaved full embeddings to {OUTPUT_DIR}")

# Quick analysis
print("\nQuick analysis:")
print(f"Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.2f}")
print(f"Std norm: {np.linalg.norm(embeddings, axis=1).std():.2f}")

# Test similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_sample = cosine_similarity(embeddings[:10])
print(f"\nSample similarities (first 10):")
print(f"Mean: {sim_sample.mean():.3f}")
print(f"Std: {sim_sample.std():.3f}")

print("\nNext steps:")
print("1. Run on full dataset (set TEST_MODE = False)")
print("2. Apply PCA to reduce dimensions") 
print("3. Run same DML analysis as other embeddings")
print("4. Compare with OpenAI, NV-Embed, and MPNet results")