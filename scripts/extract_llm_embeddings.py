#!/usr/bin/env python3
"""
Extract embeddings from LLMs using VLLM
We'll use the hidden states from the last layer as embeddings
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import argparse

print("="*80)
print("LLM EMBEDDING EXTRACTION")
print("="*80)

# Available models (ordered by size)
MODELS = {
    'qwen-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'llama-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'llama-8b': 'meta-llama/Llama-3.2-8B-Instruct',
    'qwen-14b': 'Qwen/Qwen2.5-14B-Instruct-AWQ',
}

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=list(MODELS.keys()), default='qwen-3b',
                    help='Which model to use for embeddings')
parser.add_argument('--max-essays', type=int, default=100,
                    help='Maximum number of essays to process (for testing)')
args = parser.parse_args()

model_name = MODELS[args.model]
print(f"\nUsing model: {model_name}")

# Paths
BASE_DIR = Path("/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "llm_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load essays
print("\nLoading essays...")
essays = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")
print(f"Loaded {len(essays)} essays")

# Limit for testing
if args.max_essays < len(essays):
    essays = essays.head(args.max_essays)
    print(f"Limited to {len(essays)} essays for testing")

# Method 1: Using VLLM with hidden states (if supported)
def extract_embeddings_vllm(texts, model_name):
    """Extract embeddings using VLLM - requires custom modification"""
    print("\nInitializing VLLM...")
    
    # Standard VLLM doesn't expose hidden states directly
    # We'll need to use a workaround or modification
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
    )
    
    # This is a placeholder - VLLM doesn't directly support embedding extraction
    # We would need to modify VLLM or use a different approach
    print("Note: Standard VLLM doesn't expose hidden states directly.")
    print("See Method 2 for alternative approach.")
    
    return None

# Method 2: Using Transformers directly (more flexible but slower)
def extract_embeddings_transformers(texts, model_name, batch_size=4):
    """Extract embeddings using transformers library"""
    print("\nUsing transformers library for embedding extraction...")
    
    from transformers import AutoModel, AutoTokenizer
    
    # Load model and tokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    embeddings = []
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get embeddings from last hidden state
            # Options:
            # 1. Mean pooling over sequence
            # 2. Use [CLS] or first token
            # 3. Use last token before padding
            
            # Option 1: Mean pooling (most common for sentence embeddings)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
            
            # Mask out padding tokens
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = attention_mask.sum(dim=1)  # [batch, 1]
            
            # Avoid division by zero
            mean_pooled = sum_hidden / sum_mask.clamp(min=1e-9)
            
            embeddings.append(mean_pooled.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings

# Method 3: Using sentence-transformers style (if model supports it)
def extract_embeddings_sentence_style(texts, model_name):
    """Extract embeddings using sentence-transformers style pooling"""
    print("\nExtracting embeddings with sentence-style pooling...")
    
    # Prepare instruction for models that support it
    if 'qwen' in model_name.lower():
        instruction = "Instruct: Analyze the social class indicators in this text.\nQuery: "
        texts = [instruction + text for text in texts]
    
    return extract_embeddings_transformers(texts, model_name)

# Main execution
if __name__ == "__main__":
    # Get essay texts
    texts = essays['original'].tolist()
    essay_ids = essays['TID'].values
    
    # Extract embeddings
    print(f"\nExtracting embeddings using {args.model}...")
    start_time = time.time()
    
    try:
        # Try transformers method (most reliable)
        embeddings = extract_embeddings_transformers(texts, model_name)
        
        if embeddings is not None:
            # Save embeddings
            output_file = OUTPUT_DIR / f"{args.model}_embeddings.npy"
            np.save(output_file, embeddings)
            print(f"\nSaved embeddings to {output_file}")
            
            # Save essay IDs
            ids_file = OUTPUT_DIR / f"{args.model}_essay_ids.npy"
            np.save(ids_file, essay_ids)
            
            # Print statistics
            elapsed = time.time() - start_time
            print(f"\nStatistics:")
            print(f"- Time elapsed: {elapsed:.1f} seconds")
            print(f"- Embeddings shape: {embeddings.shape}")
            print(f"- Embedding dimensions: {embeddings.shape[1]}")
            print(f"- Essays per second: {len(texts)/elapsed:.1f}")
            
            # Quick quality check - compute similarity matrix for first 10
            if len(embeddings) >= 10:
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(embeddings[:10])
                print(f"\nSample similarity matrix (first 10 essays):")
                print(f"- Mean similarity: {sim_matrix.mean():.3f}")
                print(f"- Std similarity: {sim_matrix.std():.3f}")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nAlternative: Use dedicated embedding models instead:")
        print("- NV-Embed-v2 (state-of-the-art)")
        print("- E5-large-v2 (good balance)")
        print("- BGE-large (bilingual)")
        print("- Instructor-XL (task-specific)")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("""
1. For best results with LLM embeddings:
   - Use mean pooling over the last hidden layer
   - Consider the last 4 layers' mean for richer representations
   - Normalize embeddings before use

2. Model selection:
   - Qwen-3B: Good balance of speed and quality
   - Llama-3B: Alternative perspective
   - Qwen-7B: Better quality but slower
   
3. Comparison approach:
   - Run same DML analysis as with other embeddings
   - Compare R² values and causal effects
   - Check if LLM embeddings capture different aspects

4. Expected dimensions:
   - Qwen-3B: ~2048 dimensions
   - Llama-3B: ~3072 dimensions
   - Qwen-7B: ~3584 dimensions
""")