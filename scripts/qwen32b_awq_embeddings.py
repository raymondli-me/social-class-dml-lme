#!/usr/bin/env python3
"""
Extract embeddings from Qwen2.5-32B-AWQ model
Using the same model that was used for VLLM inference earlier
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_version
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import gc

print("="*80)
print("QWEN2.5-32B-AWQ EMBEDDING EXTRACTION")
print("="*80)

print(f"Using PyTorch version: {torch.__version__}")
print(f"Using Transformers version: {transformers_version}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"  # The exact model you used
BATCH_SIZE = 2  # Small batch size for 32B model
MAX_LENGTH = 2048  # Same as your VLLM config
DEVICE_MAP = "auto"  # Let it use 2 GPUs as in your VLLM setup

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "qwen32b_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load essays
print("\n=== Loading Essays ===")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
print(f"✓ Loaded {len(essays)} essays")

# For testing, start with a small subset
TEST_MODE = False  # Changed to run full dataset
TEST_SIZE = 100

if TEST_MODE:
    essays = essays.head(TEST_SIZE)
    print(f"⚠️  TEST MODE: Using only {len(essays)} essays")
else:
    print("Running on FULL dataset")

# Load model and tokenizer
print(f"\n=== Loading {MODEL_NAME} ===")
print("This will use 2 GPUs as configured in your VLLM setup...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load with AWQ quantization support
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=DEVICE_MAP,  # Auto-distribute across GPUs
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        low_cpu_mem_usage=True
    )
    print("✓ Model and tokenizer loaded successfully")
    
    # Print device mapping
    if hasattr(model, 'hf_device_map'):
        print("\nDevice mapping:")
        for i, (name, device) in enumerate(list(model.hf_device_map.items())[:5]):
            print(f"  {name}: {device}")
        print("  ...")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Access embedding layer
embedding_layer = model.get_input_embeddings()
if embedding_layer:
    print(f"\n=== Embedding Layer Info ===")
    print(f"Type: {type(embedding_layer)}")
    print(f"Vocab Size: {embedding_layer.num_embeddings}")
    print(f"Embedding Dimension: {embedding_layer.embedding_dim}")
else:
    print("❌ Could not access embedding layer")
    exit()

# Set model to evaluation mode
model.eval()

# Ensure padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

# Function to extract embeddings with mean pooling
def get_essay_embeddings(texts, batch_size=BATCH_SIZE):
    """Extract embeddings for a list of texts using mean pooling"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Move to first GPU (model will handle distribution)
        input_ids = inputs["input_ids"].to('cuda:0')
        attention_mask = inputs["attention_mask"].to('cuda:0')
        
        with torch.no_grad():
            # Get token embeddings from embedding layer
            token_embeddings = embedding_layer(input_ids)
            
            # Mean pooling: average embeddings, ignoring padding tokens
            # Expand attention mask to match embedding dimensions
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings for non-padding tokens
            sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
            
            # Count non-padding tokens
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            
            # Compute mean
            mean_embeddings = sum_embeddings / sum_mask
            
            # Move to CPU and convert to numpy
            all_embeddings.append(mean_embeddings.cpu().float().numpy())
        
        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Stack all embeddings
    return np.vstack(all_embeddings)

# Extract embeddings
print(f"\n=== Extracting Embeddings ===")
print(f"Processing {len(essays)} essays in batches of {BATCH_SIZE}...")

texts = essays['original'].tolist()
essay_ids = essays['TID'].values

# Extract embeddings from raw essays (no instruction prefix)
# This ensures fair comparison with other embedding models
print("Note: Embedding raw essays without instruction prefix for fair comparison")

start_time = time.time()

try:
    # Extract embeddings from raw essay text
    embeddings = get_essay_embeddings(texts)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Extraction complete!")
    print(f"Time: {elapsed:.1f} seconds ({len(texts)/elapsed:.1f} essays/sec)")
    print(f"Embeddings shape: {embeddings.shape}")
    
except Exception as e:
    print(f"\n❌ Error during extraction: {e}")
    raise

# Save embeddings
print("\n=== Saving Results ===")
if TEST_MODE:
    embeddings_file = OUTPUT_DIR / "qwen32b_awq_embeddings_test.npy"
    ids_file = OUTPUT_DIR / "qwen32b_awq_essay_ids_test.npy"
else:
    embeddings_file = OUTPUT_DIR / "qwen32b_awq_embeddings.npy"
    ids_file = OUTPUT_DIR / "qwen32b_awq_essay_ids.npy"

np.save(embeddings_file, embeddings)
np.save(ids_file, essay_ids)
print(f"✓ Saved embeddings to {embeddings_file}")
print(f"✓ Saved essay IDs to {ids_file}")

# Analysis
print("\n=== Embedding Analysis ===")
print(f"Shape: {embeddings.shape}")
print(f"Dimensions: {embeddings.shape[1]}")
print(f"Data type: {embeddings.dtype}")
print(f"Memory usage: {embeddings.nbytes / 1024 / 1024:.1f} MB")

# Compute statistics
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nNorm statistics:")
print(f"  Mean: {norms.mean():.2f}")
print(f"  Std: {norms.std():.2f}")
print(f"  Min: {norms.min():.2f}")
print(f"  Max: {norms.max():.2f}")

# Sample similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
if len(embeddings) >= 10:
    sim_matrix = cosine_similarity(embeddings[:10])
    print(f"\nSample similarity matrix (first 10 essays):")
    print(f"  Mean similarity: {sim_matrix.mean():.3f}")
    print(f"  Std similarity: {sim_matrix.std():.3f}")
    print(f"  Min similarity: {sim_matrix.min():.3f}")
    print(f"  Max similarity: {sim_matrix.max():.3f}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. If test successful, run on full dataset:
   - Edit script: TEST_MODE = False
   - Run again (will take ~1-2 hours for 9,513 essays)

2. Process embeddings for DML:
   - Apply PCA reduction (e.g., to 200-500 components)
   - Run same analysis pipeline as other embeddings

3. Compare with other models:
   - OpenAI: R²(AI)=0.923, R²(SC)=0.537
   - NV-Embed: R²(AI)=0.597, R²(SC)=0.073
   - MPNet: R²(AI)=0.451, R²(SC)=0.050

4. Expected characteristics:
   - Qwen-32B embeddings dimension: ~5120
   - Should capture rich semantic information
   - May perform differently than dedicated embedding models
   - Interesting to see if scale (32B) beats specialization
""")

# Clean up
del model
torch.cuda.empty_cache()
gc.collect()
print("\n✓ Cleaned up GPU memory")