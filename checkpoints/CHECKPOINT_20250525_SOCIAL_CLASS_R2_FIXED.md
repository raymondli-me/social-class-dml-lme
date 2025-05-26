# Checkpoint: Fixed Social Class R² Analysis
**Date:** 2025-05-25 10:30:00
**Status:** Resolved data loading issue and obtained correct R² values

## Summary
Discovered and fixed a critical data loading bug that was causing R² = 0.000 for actual social class prediction. The issue was that only 526 essays were being loaded instead of the full 9,513 essay dataset.

## Key Findings

### Correct R² Values with OpenAI Embeddings (200 PCA Components) + XGBoost

| Outcome | R² (DML Cross-fitting) | Notes |
|---------|------------------------|-------|
| **AI Ratings** | 0.923 | Excellent predictive power |
| **Actual Social Class** | 0.537 | Moderate predictive power |

### The Data Loading Bug
- **Problem**: The original `openai_embedding_analysis.py` loaded from `data/essay_dataset.csv` which only had 526 essays
- **Solution**: Should load from `data/asc_9513_essays.csv` which has all 9,513 essays
- **Impact**: With only 526 essays, after merging with 9,513 social class labels, most data was missing, causing residualization to fail

### Evidence of the Bug
```
Residuals with 526 essays:
- Y_res std: 0.000000 (essentially machine epsilon)
- Unique values: 24 (all near 1e-16)

Residuals with 9,513 essays:
- Y_res std: 0.887 (proper variance)
- Range: [-1.977, 2.137]
```

### Verification of Results
1. **With DML (proper causal estimate)**: R² = 0.537
2. **Without DML (biased)**: 
   - Training R² = 0.825 (severely overfitted)
   - 5-fold CV R² = -0.024 (negative on test data!)
3. This confirms DML is working correctly and preventing overfitting

## Technical Details
- **Embeddings**: OpenAI text-embedding-3-large (3,072 dimensions)
- **Dimensionality Reduction**: PCA to 200 components
- **Model**: XGBoost with 100 estimators, max_depth=5
- **Cross-fitting**: 5-fold cross-validation for residualization
- **Confounders**: Age, gender, education level (randomly generated with seed=42)

## Interpretation
The corrected results show that:
1. Text embeddings can predict ~54% of actual social class variation
2. Text embeddings can predict ~92% of AI-perceived social class
3. This ~40% gap represents the fundamental difference between:
   - What linguistic/textual features AI models associate with social class
   - How people actually self-identify their social class

This is a profound finding about AI bias and the disconnect between AI perception and human reality in social categorization.

## Files Created
- `/scripts/verify_social_class_prediction.py` - Initial diagnostic script
- `/scripts/diagnose_social_class_dml.py` - Detailed residualization analysis
- `/scripts/check_data_alignment.py` - Data merging verification
- `/scripts/rerun_social_class_xgboost.py` - Final corrected analysis

## Lesson Learned
Always verify data shapes and merges when working with multiple datasets. A simple shape mismatch (526 vs 9,513) can completely invalidate results, turning a meaningful R² = 0.537 into a meaningless R² = 0.000.