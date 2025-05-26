# Gemma Embeddings Analysis for Social Class DML

## Overview

This analysis replicates the OpenAI embeddings analysis using Google's open-source **Gemma-Embeddings-v1.0** model from Hugging Face.

Model page: https://huggingface.co/google/Gemma-Embeddings-v1.0

## Key Features

- **Free and open-source** - No API costs
- **Local inference** - Runs on your hardware
- **High-quality embeddings** - Based on Gemma architecture
- **Proper data loading** - Ensures 9,513 essays are loaded (not 526!)

## Requirements

```bash
# Core dependencies
pip install transformers torch
pip install pandas numpy scikit-learn
pip install xgboost econml
```

## Quick Start

1. **Test the model first** (recommended):
   ```bash
   python scripts/test_gemma_model.py
   ```

2. **Run the full analysis**:
   ```bash
   python scripts/run_gemma_analysis.py
   ```
   
   Or directly:
   ```bash
   python scripts/gemma_embedding_analysis.py
   ```

## What the Analysis Does

1. **Loads correct data** (9,513 essays from `asc_9513_essays.csv`)
2. **Generates embeddings** using Gemma-Embeddings-v1.0
3. **Reduces dimensions** from full embedding size to 200 PCs
4. **Evaluates predictions**:
   - Text → AI ratings (R²)
   - Text → Actual social class (R²)
5. **Runs DML analysis** to get causal effect of SC on AI ratings
6. **Compares with OpenAI results**

## Expected Timeline

- Model download: 5-10 minutes (first run only, ~2GB)
- Embedding generation: 30-60 minutes (depending on GPU)
- Analysis: 5-10 minutes
- Total: ~1 hour first run, ~45 minutes subsequent runs

## Output Files

All outputs saved to `gemma_checkpoints/`:
- `gemma_embeddings.npy` - Raw embeddings
- `gemma_pca_200_features.pkl` - PCA-reduced features
- `gemma_analysis_results.pkl` - All results

## Memory Requirements

- **Minimum**: 16GB RAM
- **Recommended**: 32GB RAM
- **GPU**: Optional but speeds up embedding generation significantly

## Comparison with OpenAI

| Metric | OpenAI text-embedding-3-large | Gemma-Embeddings-v1.0 |
|--------|-------------------------------|----------------------|
| Cost | ~$1-2 per run | Free |
| Speed | ~5 minutes | ~30-60 minutes |
| Text → AI R² | 0.923 | TBD |
| Text → SC R² | 0.537 | TBD |
| DML θ | 0.0527 | TBD |

## Troubleshooting

1. **Out of memory**: Reduce batch size in `encode()` method (default is 8)
2. **Model download fails**: Check internet connection, try again
3. **CUDA errors**: Set device='cpu' to run on CPU only
4. **Wrong number of essays**: Check you're loading `asc_9513_essays.csv`

## Notes

- First run downloads ~2GB model weights
- Embeddings are normalized using L2 norm
- Uses mean pooling over tokens
- Max sequence length is 512 tokens