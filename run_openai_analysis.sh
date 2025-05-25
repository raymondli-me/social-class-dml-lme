#!/bin/bash
# Run OpenAI embedding analysis with API key

# Set working directory
cd "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme"

# Source environment
source .env.openai

# Run analysis
echo "Starting OpenAI Text-Embedding-3-Large Analysis..."
echo "This will:"
echo "1. Generate embeddings for 9,513 essays (may take 15-20 minutes)"
echo "2. Compute 200 PCA components"
echo "3. Run DML analysis with 4 models"
echo "4. Create 3D UMAP visualization"
echo "5. Generate TreeSHAP explanations"
echo ""

python3 scripts/openai_embedding_analysis.py