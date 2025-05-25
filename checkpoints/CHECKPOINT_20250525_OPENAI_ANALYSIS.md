# OpenAI Text-Embedding-3-Large Analysis
**Date:** 2025-05-25  
**Status:** Ready to run  

## Overview
Separate analysis pipeline using OpenAI's state-of-the-art text-embedding-3-large model instead of sentence-transformers. This explores whether better embeddings can improve the already excellent R² = 0.870 achieved with MiniLM embeddings.

## Key Differences from Previous Analysis
1. **Embeddings:** OpenAI text-embedding-3-large (3072 dimensions) vs MiniLM (384 dimensions)
2. **PCA:** 200 components (instead of 100) to capture more variance from higher-dimensional embeddings
3. **Visualization:** TreeSHAP throughout (no LIME) for consistency and speed
4. **Background:** Black background for UMAP visualizations (better contrast)

## Implementation Details
- **Script:** `scripts/openai_embedding_analysis.py`
- **API Key:** Stored in `.env.openai` and embedded in script
- **Batch Processing:** 100 texts per API call with rate limiting
- **Cost Estimate:** ~$1-2 for 9,513 essays (at $0.00013 per 1K tokens)

## Expected Outputs
```
openai_checkpoints/
├── openai_embeddings.npy          # (9513, 3072) embeddings
├── pca_200_features.pkl           # PCA reduced to 200 dims
├── dml_results_openai.pkl         # Model performance metrics
└── umap_3d_openai.npy            # 3D UMAP coordinates

openai_visualizations/
├── umap_actual_sc_openai.html    # Interactive 3D colored by actual SC
├── umap_ai_rating_openai.html    # Interactive 3D colored by AI rating
└── shap_analysis/
    ├── shap_summary_openai.png   # Feature importance summary
    ├── shap_importance_openai.png # Bar plot of top features
    └── shap_waterfall_*.png      # Individual explanations
```

## Running the Analysis
```bash
# Method 1: Using run script
./run_openai_analysis.sh

# Method 2: Manual with environment
source .env.openai
python3 scripts/openai_embedding_analysis.py

# Method 3: Direct (API key embedded)
python3 scripts/openai_embedding_analysis.py
```

## Expected Runtime
- Embeddings generation: 15-20 minutes (API calls)
- PCA computation: 2-3 minutes
- DML analysis: 5-10 minutes
- UMAP computation: 3-5 minutes
- Visualization creation: 2-3 minutes
- **Total:** ~30-40 minutes

## Hypothesis
OpenAI's embeddings are trained on much larger and more diverse datasets than MiniLM. They may capture:
- Subtler linguistic patterns
- Better semantic understanding
- More nuanced social class indicators
- Cultural references and context

This could potentially improve the R² beyond 0.870, though the current performance is already excellent.

## Next Steps After Completion
1. Compare R² scores between embedding types
2. Analyze which PCA components are most important
3. Investigate if the UMAP clusters differently
4. Consider ensemble approach using both embedding types
5. Test other embeddings (Gemini, Cohere, etc.)

## Notes
- The script handles all aspects including checkpointing
- Black background chosen for better visibility of clusters
- TreeSHAP used throughout for consistency
- All paths are absolute for reliability