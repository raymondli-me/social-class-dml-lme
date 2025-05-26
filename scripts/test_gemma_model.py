#!/usr/bin/env python3
"""
Quick test to verify Gemma model can be loaded
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, GemmaTokenizer, GemmaForCausalLM
    import torch
    
    print("Testing Gemma-Embeddings-v1.0 model loading...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load the Gemma embeddings model
    model_name = "google/Gemma-Embeddings-v1.0"
    
    print(f"\nLoading tokenizer from {model_name}...")
    # Try GemmaTokenizer first, fallback to AutoTokenizer
    try:
        tokenizer = GemmaTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded successfully!")
    
    print(f"\nLoading model from {model_name}...")
    print("(This may take a few minutes on first run to download the model)")
    
    # The Gemma-Embeddings model should be loaded with specific config
    try:
        # Try loading as Gemma model first
        model = GemmaForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    except:
        # Fallback to AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    print("✓ Model loaded successfully!")
    
    # Test encoding
    print("\nTesting encoding...")
    test_text = "This is a test sentence for the Gemma embeddings model."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[-1]}")
    
    print("\n✅ All tests passed! Ready to run full analysis.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPossible solutions:")
    print("1. Make sure you have transformers installed: pip install transformers")
    print("2. Make sure you have torch installed: pip install torch")
    print("3. Check your internet connection (model needs to be downloaded)")
    print("4. Try updating transformers: pip install --upgrade transformers")