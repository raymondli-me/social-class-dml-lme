# CHECKPOINT: Full 4,096 Dimensions vs PCA 200 Comparison

**Date:** May 26, 2025  
**Time:** Current  
**Purpose:** Compare DML results using full NV-Embed dimensions vs PCA reduction

## Analysis Plan

### What We're Doing
Comparing two approaches with NV-Embed-v2 embeddings:
1. **PCA 200**: Using 200 principal components (72.1% variance retained)
2. **Full 4,096**: Using all original dimensions (100% variance)

### Research Question
Does PCA reduction lose important information for detecting the causal effect of social class on AI ratings?

## Data Files and Locations

### 1. Embeddings
- **File**: `/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_checkpoints/nvembed_embeddings.npy`
- **Shape**: (9513, 4096)
- **Description**: Full NV-Embed-v2 embeddings for all 9,513 essays

### 2. Essay IDs
- **File**: `/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_checkpoints/nvembed_essay_ids.npy`
- **Shape**: (9513,)
- **Description**: Essay TIDs for alignment

### 3. Essays
- **File**: `/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/data/asc_9513_essays.csv`
- **Columns**: TID, original
- **Rows**: 9,513 essays

### 4. Social Class Labels
- **File**: `/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv`
- **Columns**: TID, sc11
- **Description**: Self-reported social class (1-5 scale)

### 5. AI Ratings
- **File**: `/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv`
- **Columns**: essay_id, prompt_name, rating, raw_response, timestamp
- **Description**: AI ratings from 2 prompts, averaged per essay

## Methodology

### Data Processing
1. Load full embeddings (4,096 dims)
2. Standardize features
3. Merge with social class and AI ratings
4. NO demographics (as per checkpoint decision)

### Models to Test
Same 5 models as PCA analysis:
- Linear Regression
- Ridge Regression (α=1.0)
- Lasso Regression (α=0.1)
- Random Forest (100 trees, max_depth=10)
- XGBoost (50 trees, max_depth=6)

### Metrics to Compare
For each model:
- **AI R²**: How well text predicts AI ratings
- **SC R²**: How well text predicts actual social class
- **θ (theta)**: Causal effect of SC on AI after controlling for text
- **SE**: Standard error
- **p-value**: Statistical significance
- **95% CI**: Confidence interval

### Expected Outcomes

#### Hypothesis 1: Full dims perform better
- Higher R² for both AI and SC
- Smaller standard errors
- Possible detection of significant effects
- Justification: 27.9% lost variance might contain signal

#### Hypothesis 2: PCA performs similarly
- Similar R² values
- Similar (null) causal effects
- Justification: PCA captures main variation, rest is noise

## Key Comparisons

### 1. Prediction Performance
- Does using all 4,096 dimensions improve R²?
- Which approach better predicts AI ratings?
- Which approach better predicts actual social class?

### 2. Causal Estimates
- Do θ values change substantially?
- Do standard errors decrease with more dimensions?
- Do any models find significant effects with full dims?

### 3. Computational Trade-offs
- Time difference between approaches
- Memory usage
- Stability of estimates

## Implementation Notes

### Critical Details
- **NO demographics** (W=None in all DML calls)
- **Standardize** full embeddings before use
- **Same random seed** (42) for reproducibility
- **5-fold cross-validation** for all models
- **Two-tailed tests** for significance

### Potential Issues
- Memory: 4,096 dims × 9,513 samples = large matrices
- Computation: XGBoost/RF may be slow with high dims
- Overfitting: More parameters than PCA approach
- Multicollinearity: High correlation among embedding dims

## Success Criteria

The analysis succeeds if we can definitively answer:
1. Does PCA 200 lose important information?
2. Is the computational cost of full dims justified?
3. Do conclusions about SC→AI causality change?

---

**Next Step**: Run `nvembed_full_dims_analysis.py` to compare both approaches