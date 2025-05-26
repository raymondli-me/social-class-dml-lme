#!/bin/bash
# Script to set up environment and run NV-Embed-v2 generation

echo "Setting up NV-Embed-v2 environment..."

# Save current package versions
echo "Backing up current environment..."
pip freeze | grep -E "transformers|torch|sentence-transformers" > current_versions.txt

# Install specific versions needed for NV-Embed-v2
echo "Installing NV-Embed-v2 dependencies..."
pip install --user transformers==4.42.4 tokenizers==0.19.1

# Run the embedding generation
echo "Generating embeddings..."
python3 generate_nvembed_only.py

# Check if successful
if [ -f "nvembed_embeddings.npy" ]; then
    echo "Success! Copying embeddings to main checkpoint directory..."
    cp nvembed_embeddings.npy ../nvembed_checkpoints/
    cp nvembed_essay_ids.npy ../nvembed_checkpoints/
    echo "✓ Embeddings copied successfully"
else
    echo "❌ Embedding generation failed"
fi

echo "Done!"