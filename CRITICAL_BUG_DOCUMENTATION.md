# CRITICAL BUG: R² = 0.000 for Social Class

## THE BUG
The stored DML results in `openai_checkpoints/dml_results_openai.pkl` show:
- Text → AI ratings: R² = 0.923 ✓ (correct)
- Text → Actual SC: R² = 0.000 ✗ (WRONG!)

## ROOT CAUSE
The `openai_embedding_analysis.py` script loads data from the WRONG location:
```python
# Line 57 - WRONG!
df = pd.read_csv(DATA_DIR / "essay_dataset.csv")  # Only 526 essays

# Should be:
df = pd.read_csv(DATA_DIR / "asc_9513_essays.csv")  # Full 9,513 essays
```

This causes:
1. Only 526 essays loaded instead of 9,513
2. After merging with social class labels, most data is missing
3. Residualization produces near-zero variance (machine epsilon ~1e-16)
4. R² = 0.000 (artifactually)

## CORRECT RESULTS
From `scripts/rerun_social_class_xgboost.py` with proper data loading:
- Text → AI ratings: R² = 0.923 ✓
- Text → Actual SC: R² = 0.537 ✓ (NOT 0.000!)

## LESSON LEARNED
**ALWAYS verify data shapes after loading!**
- Expected: 9,513 essays
- If you see 526: STOP - wrong file!

## DO NOT USE
- `openai_checkpoints/dml_results_openai.pkl` (contains bad SC results)
- Any cached results showing R² = 0.000 for social class

## USE INSTEAD
- Fresh calculations with proper data loading
- The corrected R² = 0.537 for text → actual SC