#!/usr/bin/env python3
"""
Simplified NV-Embed-v2 test to check if we can load and use the model
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

model_name = "nvidia/NV-Embed-v2"

print("Testing NV-Embed-v2...")

try:
    # Try loading tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    # Try loading model with simpler config
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    print("✓ Model loaded")
    
    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Test encoding
    print("\nTesting encoding...")
    text = "This is a test essay about social class."
    
    # Add instruction prefix as recommended
    instruction = "Instruct: Retrieve semantically similar text.\nQuery: "
    text_with_instruction = instruction + text
    
    # Tokenize
    inputs = tokenizer(text_with_instruction, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get embeddings (usually last hidden state with mean pooling)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            print(f"✓ Embeddings shape: {embeddings.shape}")
            print(f"✓ Embedding dimension: {embeddings.shape[-1]}")
        else:
            print("Output type:", type(outputs))
            if hasattr(outputs, 'keys'):
                print("Output keys:", outputs.keys())
    
    print("\n✅ NV-Embed-v2 is working! Ready for full analysis.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nThis might be due to:")
    print("1. Incompatible transformers version")
    print("2. Model requires specific dependencies")
    print("3. Model architecture has changed")
    
    import transformers
    print(f"\nCurrent transformers version: {transformers.__version__}")
    print("You might need to update: pip install --upgrade transformers")