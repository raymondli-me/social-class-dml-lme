# Complete Analysis Checkpoint: Social Class DML with LLM Embeddings

**Date:** May 25, 2025  
**Location:** `/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/`

## 1. Project Overview

This project analyzes the relationship between actual self-reported social class and AI-perceived social class using Double Machine Learning (DML) with text embeddings. We have 9,513 essays from the ASC corpus where participants wrote about their life at age 25.

### Key Research Questions:
1. How well can text embeddings predict AI social class ratings vs actual social class?
2. What is the causal effect of actual social class on AI ratings after controlling for text?
3. How much of the SC-AI relationship is mediated by textual content?

## 2. Data Sources

### Primary Data Files:
```
# Essays (9,513 samples)
data/asc_9513_essays.csv
- Columns: TID, criterion, judgement, original
- TID: unique identifier
- original: essay text

# Social class labels
/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv  
- Columns: TID, sc11
- sc11: self-reported social class (1-5 scale)

# AI ratings (2 prompts × 9,513 = 19,026 ratings)
asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv
- Columns: essay_id, prompt_name, rating, raw_response, timestamp
- rating: AI social class rating (1-10 scale)
- Averaged to get one rating per essay
```

### ⚠️ CRITICAL DATA LOADING BUG:
```python
# WRONG - only loads 526 essays!
df = pd.read_csv("data/essay_dataset.csv")  

# CORRECT - loads all 9,513 essays
df = pd.read_csv("data/asc_9513_essays.csv")
```

## 3. Embeddings Analysis

### OpenAI Embeddings (Current):
- Model: `text-embedding-3-large` (3,072 dimensions)
- Cost: ~$1-2 for 9,513 essays
- Reduced to 200 PCs for analysis
- Cached in: `openai_checkpoints/pca_200_features.pkl`

### Results with OpenAI + 200 PCs:
```
Text → AI Ratings:    R² = 0.923 (XGBoost)
Text → Actual SC:     R² = 0.537 (XGBoost)  
Gap:                  38.6 percentage points
```

## 4. DML Analysis Results

### 4.1 Methodology
- 5-fold cross-fitting to avoid overfitting bias
- Controls: Age, gender, education (randomly generated, seed=42)
- Models tested: Linear, Ridge, Lasso, Random Forest, XGBoost

### 4.2 First Stage Performance (R² values)

| Model | AI Ratings | Actual SC | Gap |
|-------|------------|-----------|-----|
| Linear/Ridge | 0.631 | 0.114 | 51.7% |
| Lasso | 0.631 | 0.121 | 51.0% |
| Random Forest | 0.495 | 0.079 | 41.6% |
| **XGBoost** | **0.923** | **0.537** | **38.6%** |

### 4.3 Causal Effects (θ)

**Model 1: Actual SC → AI Ratings**
| Model | θ | SE | p-value | Partial R² |
|-------|---|-----|---------|------------|
| Linear | 0.0332 | 0.0103 | 0.0013** | 0.11% |
| Ridge | 0.0332 | 0.0103 | 0.0013** | 0.11% |
| Lasso | 0.0340 | 0.0104 | 0.0010** | 0.11% |
| Random Forest | 0.1081 | 0.0119 | <0.0001*** | 0.86% |
| XGBoost | 0.0527 | 0.0108 | <0.0001*** | 0.24% |

**Model 2: AI Ratings → Actual SC**
| Model | θ | SE | p-value | Partial R² |
|-------|---|-----|---------|------------|
| Linear | 0.0321 | 0.0100 | 0.0013** | 0.11% |
| Ridge | 0.0321 | 0.0100 | 0.0013** | 0.11% |
| Lasso | 0.0328 | 0.0100 | 0.0011** | 0.11% |
| Random Forest | 0.0795 | 0.0088 | <0.0001*** | 0.86% |

## 5. Visualization Results

### 5.1 3D UMAP Visualization
- Created interactive Three.js visualization with 9,513 points
- Features: AI rating filters (1-9.5), social class filters (1-5), click to view essays
- Working version: `custom_visualizations/umap_ai_filters_final_20250525_093917.html`

### 5.2 Key Visual Findings:
- AI ratings cluster more tightly than actual social class
- Most essays rated 3-5 by AI (mean = 4.39)
- Color spectrum: Purple (1) → Blue → Green → Yellow → Red (10)

## 6. Key Findings Summary

1. **Text dominates AI ratings**: Embeddings explain 92.3% of AI rating variance
2. **Moderate prediction of actual SC**: Embeddings explain 53.7% of actual SC variance  
3. **Small causal effects**: After controlling for text, SC→AI effect is only θ ≈ 0.03-0.05
4. **Text mediates 87%** of the raw SC-AI correlation (0.251 → 0.033)
5. **AI bias quantified**: 38.6% gap represents what AI "thinks" indicates SC vs reality

## 7. Important Scripts

### Core Analysis:
```python
# Main analyses
scripts/openai_embedding_analysis.py          # OpenAI embeddings + DML (BUGGY - wrong data file)
scripts/rerun_social_class_xgboost.py        # FIXED XGBoost analysis  
scripts/dml_bidirectional_correct.py         # Bidirectional DML with correct data
scripts/dml_all_models_comparison.py         # Multi-model comparison

# Visualization
scripts/create_custom_visualization_fixed.py  # Three.js UMAP visualization
scripts/add_ai_filters_minimal.py            # Add dynamic filters to viz

# Utilities
scripts/compute_dml_coefficients_fast.py     # Get coefficients with SEs
scripts/check_data_alignment.py              # Verify data shapes
```

### Key Outputs:
```
# Results
openai_checkpoints/dml_results_openai.pkl    # BUGGY - has R²=0.000 for SC
outputs/dml_all_models_results.pkl           # Multi-model comparison results

# Documentation  
DML_BIDIRECTIONAL_ANALYSIS_DOCUMENTATION.md  # Detailed methods
DML_ALL_MODELS_DOCUMENTATION.md              # Multi-model results
CRITICAL_BUG_DOCUMENTATION.md                # Data loading bug warning
```

## 8. Critical Bugs to Avoid

1. **Data Loading**: Always use `asc_9513_essays.csv` (9,513 essays), not `essay_dataset.csv` (526)
2. **R² = 0.000**: If you see this for social class, it's the data bug - should be R² = 0.537
3. **Memory**: OpenAI embeddings file is 223MB, may cause memory issues
4. **Git**: Large files (>100MB) excluded from repo due to GitHub limits

## 9. Recommendations for Future Analysis

### 9.1 State-of-the-Art Open-Source Embeddings to Try

For replicating this analysis with SOTA open-source models, I recommend:

1. **Cohere Embed v3** (via API, but more affordable)
   - Model: `embed-english-v3.0` or `embed-multilingual-v3.0`
   - Dimensions: 1024
   - Performance: Comparable to OpenAI
   - Cost: ~$0.10 per million tokens

2. **Sentence-Transformers (Local, Free)**
   ```python
   # Best overall performance
   model_name = 'sentence-transformers/all-mpnet-base-v2'
   # 768 dimensions, excellent performance
   
   # For longer texts (up to 8192 tokens)
   model_name = 'sentence-transformers/gtr-t5-xl'
   # 768 dimensions, handles long essays better
   
   # Multilingual option
   model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
   ```

3. **Instructor Embeddings** (Task-specific, Local)
   ```python
   from InstructorEmbedding import INSTRUCTOR
   model = INSTRUCTOR('hkunlp/instructor-xl')
   
   # Can specify task-specific instructions
   instruction = "Represent the social class Essay for clustering:"
   embeddings = model.encode([[instruction, text] for text in essays])
   ```

4. **LLaMA-based Embeddings** (Newest, Local)
   ```python
   # Using GritLM-7B (Llama-based)
   model_name = 'GritLM/GritLM-7B'
   # State-of-the-art performance, 4096 dimensions
   ```

### 9.2 How to Restart the Analysis

1. **Load the correct data**:
   ```python
   import pandas as pd
   import numpy as np
   
   # Essays
   essays = pd.read_csv('data/asc_9513_essays.csv')
   
   # Social class labels  
   sc_labels = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
   essays = essays.merge(sc_labels, on='TID')
   
   # AI ratings
   ai_ratings = pd.read_csv('asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv')
   ai_avg = ai_ratings.groupby('essay_id')['rating'].mean().reset_index()
   ai_avg.columns = ['TID', 'ai_average']
   essays = essays.merge(ai_avg, on='TID')
   ```

2. **Generate new embeddings**:
   ```python
   from sentence_transformers import SentenceTransformer
   
   # Load model
   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   
   # Generate embeddings
   texts = essays['original'].tolist()
   embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
   
   # Save
   np.save('embeddings_mpnet.npy', embeddings)
   ```

3. **Run DML analysis**:
   ```python
   # Use scripts/dml_all_models_comparison.py as template
   # Just replace the embedding loading section
   ```

### 9.3 Computational Requirements

- **RAM**: 16GB minimum (32GB recommended for large embeddings)
- **GPU**: Helpful but not required for inference
- **Storage**: ~2GB for embeddings + intermediate files
- **Time**: ~30 min for embeddings, ~2 hours for full DML analysis

### 9.4 Expected Improvements

With modern open-source models, expect:
- Similar or better R² for AI ratings (>90%)
- Potentially better R² for actual SC (>60% possible)
- Faster inference (especially with local models)
- No API costs

## 10. Contact and Repository

- GitHub: https://github.com/raymondli-me/social-class-dml-lme
- Latest checkpoint: `/checkpoints/COMPLETE_ANALYSIS_CHECKPOINT_20250525.md`
- For questions: Check issue tracker on GitHub

---

This checkpoint contains everything needed to understand, replicate, or extend the analysis. The key insight remains: AI models infer social class primarily from textual cues, and this inference only moderately correlates with actual self-reported social class.