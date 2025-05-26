#!/usr/bin/env python3
"""
Simple test to load Gemma-Embeddings-v1.0
"""

import torch
from transformers import AutoModel, AutoTokenizer

model_name = "google/Gemma-Embeddings-v1.0"

print("Loading Gemma-Embeddings-v1.0...")

# Load with trust_remote_code since it's a custom model
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Hidden size: {model.config.hidden_size}")
    
    # Test encoding
    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get embeddings - try different approaches
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state.mean(dim=1)
            print(f"✓ Embeddings shape: {embeddings.shape}")
        else:
            print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to understand the error...")
    import traceback
    traceback.print_exc()