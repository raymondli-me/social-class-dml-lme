# Embedding Model Comparison for Social Class DML Analysis

## Summary Results

We tested three embedding models on 9,513 essays to analyze the relationship between actual social class and AI-perceived social class.

### 1. OpenAI text-embedding-3-large (Baseline)
- **Dimensions**: 3,072 → 200 (PCA)
- **Text → AI Ratings**: R² = 0.923 ✨
- **Text → Actual SC**: R² = 0.537 ✨
- **Gap**: 38.6%
- **DML Effect (SC → AI)**: θ = 0.0527 (p < 0.001) ✅
- **Time**: ~5 minutes
- **Cost**: ~$1-2 per run

### 2. MPNet (all-mpnet-base-v2)
- **Dimensions**: 768 → 200 (PCA)
- **Text → AI Ratings**: R² = 0.451
- **Text → Actual SC**: R² = 0.050
- **Gap**: 40.2%
- **DML Effect (SC → AI)**: θ = 0.0018 (p = 0.107) ❌
- **Time**: 42 seconds
- **Cost**: Free

### 3. MiniLM-L12-v2 (Fallback for NV-Embed)
- **Dimensions**: 384 → 200 (PCA)
- **Text → AI Ratings**: R² = 0.385
- **Text → Actual SC**: R² = 0.036
- **Gap**: 34.9%
- **DML Effect (SC → AI)**: θ = 0.0027 (p = 0.197) ❌
- **Time**: 6 seconds
- **Cost**: Free

## Key Findings

1. **OpenAI dramatically outperforms open-source models**:
   - 2x better at predicting AI ratings
   - 10-15x better at predicting actual social class
   - Only model that detects significant causal effect

2. **The causal effect vanishes with weaker embeddings**:
   - OpenAI: θ = 0.0527 (highly significant)
   - MPNet: θ = 0.0018 (not significant)
   - MiniLM: θ = 0.0027 (not significant)

3. **All models show similar bias gap** (~35-40%):
   - AI ratings are much easier to predict than actual SC
   - This gap persists across embedding quality levels

4. **Speed-quality tradeoff**:
   - MiniLM: 6 seconds (worst quality)
   - MPNet: 42 seconds (moderate quality)
   - OpenAI: 5 minutes (best quality)

## Technical Issues Encountered

### NV-Embed-v2 Compatibility
- **Error**: `cannot import name 'MISTRAL_INPUTS_DOCSTRING'`
- **Cause**: Model requires specific transformers version (4.42.4)
- **Current version**: 4.52.3
- **Solution needed**: Downgrade transformers or wait for model update

## Recommendations

1. **For research requiring causal inference**: Use OpenAI embeddings
   - High-quality embeddings are crucial for detecting subtle effects
   - The $1-2 cost is worth it for accurate results

2. **For exploratory analysis or prototyping**: Use MPNet
   - Free and reasonably fast
   - Good enough to see general patterns

3. **For real-time applications**: Use MiniLM
   - Extremely fast (1,500+ texts/second)
   - Acceptable for coarse-grained analysis

## Code to Reproduce

```python
# OpenAI (requires API key)
python scripts/openai_embedding_analysis.py

# MPNet (free, local)
python scripts/mpnet_embedding_analysis.py

# NV-Embed-v2 (when fixed)
pip install transformers==4.42.4
python scripts/nvembed_analysis.py
```

## Conclusion

This comparison demonstrates that **embedding quality matters critically for causal inference**. While free models can capture broad patterns, detecting subtle causal relationships requires state-of-the-art embeddings like OpenAI's text-embedding-3-large.