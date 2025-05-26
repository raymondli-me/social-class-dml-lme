# Mistakes and Lessons Learned: Complete Documentation for Future Models

**Date:** May 25, 2025  
**Purpose:** Document all mistakes, bugs, and lessons learned to help future models avoid repeating errors

## 1. CRITICAL MISTAKES TO AVOID

### 1.1 The 526 vs 9,513 Essay Bug (MOST CRITICAL)

**What happened:**
- The script `openai_embedding_analysis.py` loaded data from `data/essay_dataset.csv` which only has 526 essays
- Should have loaded from `data/asc_9513_essays.csv` which has all 9,513 essays
- This caused R² = 0.000 for social class (should be R² = 0.537)

**How to detect:**
```python
# ALWAYS CHECK DATA SHAPE AFTER LOADING
df = pd.read_csv(file_path)
print(f"Loaded {len(df)} essays")  # Should be 9,513, not 526!
```

**Files affected by this bug:**
- `openai_checkpoints/dml_results_openai.pkl` - Contains wrong R² = 0.000
- Any cached results showing R² = 0.000 for social class

### 1.2 Residualization Producing Zero Variance

**What happened:**
- After residualizing social class on demographics, variance was ~1e-16 (machine epsilon)
- This was a symptom of the data loading bug, not a real issue
- Led to ridiculous theta values like -7783250734345.271

**How to detect:**
```python
print(f"Y_res std: {np.std(Y_res)}")  # Should be ~0.89, not 0.000
```

### 1.3 Confusion About DML Direction

**Multiple instances of confusion:**
1. Initially computed effect of embeddings on outcomes (not what was asked)
2. Then tried to regress residuals on residuals (wrong)
3. Finally understood: effect of actual SC on AI ratings, controlling for embeddings

**Correct interpretation:**
- Y = AI ratings (outcome)
- D = Actual SC (treatment)
- X = Embeddings (controls)
- We want: θ in the equation Y = θD + g(X,W) + ε

## 2. FILE ORGANIZATION AND LOCATIONS

### 2.1 Complete File Structure
```
/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/
│
├── data/
│   ├── asc_9513_essays.csv (9,513 essays - CORRECT FILE)
│   ├── essay_dataset.csv (526 essays - WRONG FILE)
│   └── ladder_variations_*.csv (various prompt versions)
│
├── asc_analysis_2prompts/run_20250524_162055/
│   └── all_results_9513x2_20250524_174149.csv (AI ratings)
│
├── openai_checkpoints/
│   ├── dml_results_openai.pkl (BUGGY - has R²=0.000)
│   ├── pca_200_features.pkl (PCA reduced embeddings)
│   ├── openai_embeddings.npy (3072-dim embeddings - 223MB)
│   └── umap_3d_openai.npy (UMAP coordinates)
│
├── custom_visualizations/
│   ├── umap_ai_filters_final_20250525_093917.html (WORKING)
│   ├── umap_with_legend_20250525_090240.html (base version)
│   └── [many broken attempts...]
│
├── scripts/
│   ├── openai_embedding_analysis.py (BUGGY - wrong data file)
│   ├── rerun_social_class_xgboost.py (FIXED version)
│   ├── dml_bidirectional_correct.py (correct bidirectional)
│   ├── dml_all_models_comparison.py (multi-model analysis)
│   └── [many diagnostic and fix scripts...]
│
├── checkpoints/
│   ├── COMPLETE_ANALYSIS_CHECKPOINT_20250525.md
│   ├── CRITICAL_BUG_DOCUMENTATION.md
│   └── CHECKPOINT_20250525_SOCIAL_CLASS_R2_FIXED.md
│
└── outputs/
    └── dml_all_models_results.pkl (multi-model comparison)
```

### 2.2 External Data Location
```
/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv
```
This file contains the actual social class labels and MUST be loaded.

## 3. VISUALIZATION MISTAKES

### 3.1 Three.js Complexity Spiral

**What happened:**
- Started with working Plotly visualization
- Tried to add too many features at once to Three.js version
- Created "complete" version that completely broke
- Had to backtrack to incremental changes

**Lesson learned:**
- ALWAYS make incremental changes to working code
- Test each feature addition separately
- Keep a working backup before major changes

**Working visualization:**
`custom_visualizations/umap_ai_filters_final_20250525_093917.html`

### 3.2 AI Rating Scale Confusion

**What happened:**
- AI ratings are averaged from 2 prompts, giving values like 3.5, 4.5
- Initially thought scale was 1-5 (it's actually 1-10)
- Max average is 9.5 (when one prompt gives 9, another gives 10)

**Key facts:**
- Raw ratings: 1-10 (integers)
- Averaged ratings: 1.0-9.5 (0.5 increments)
- Most common: 3.0 and 5.0

## 4. ANALYSIS MISTAKES

### 4.1 Not Checking Cached Results

**What happened:**
- Kept referencing `dml_results_openai.pkl` which had the wrong R² = 0.000
- Should have re-run analysis after fixing data loading

**Lesson:**
- Always verify cached results match expected values
- Re-run analyses when in doubt

### 4.2 Confusing Model Types in Results

**Multiple times:**
- Mixed up Linear vs XGBoost R² values
- Forgot which model gave which results
- Didn't clearly label model types in outputs

**Solution:**
Always clearly label:
```python
results = {
    'model_type': 'xgboost',
    'r2_ai': 0.923,
    'r2_sc': 0.537,  # NOT 0.000!
}
```

### 4.3 Demographics Generation

**What happened:**
- Demographics (age, gender, education) are randomly generated
- Used seed=42 for reproducibility
- But this is a limitation - real demographics might change results

**Note for future:**
```python
np.random.seed(42)  # ALWAYS use same seed
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))
```

## 5. COMPUTATIONAL MISTAKES

### 5.1 XGBoost Timeout Issues

**What happened:**
- XGBoost with 100 estimators on 9,513 samples times out (>2 min)
- Had to fall back to linear models or cached results

**Solutions:**
- Reduce n_estimators to 50
- Use n_jobs=-1 for parallel processing
- Pre-compute and cache results

### 5.2 Memory Issues with Large Files

**Files that cause issues:**
- `openai_embeddings.npy` (223MB)
- Full visualization HTML files (>12MB each)

**GitHub wouldn't accept:**
- Files >100MB
- Had to add to .gitignore

## 6. CONCEPTUAL MISTAKES

### 6.1 Misunderstanding DML Output

**Initial confusion:**
- Thought we needed one theta for effect of embeddings on outcomes
- Actually needed TWO separate analyses (AI ratings and actual SC)
- Then realized user wanted effect of SC on AI, controlling for embeddings

**Correct understanding:**
DML gives us θ in: Y = θD + g(X,W) + ε
- Not the effect of X on Y
- But the effect of D on Y, controlling for X

### 6.2 Interpreting Partial R²

**Confusion about small values:**
- Partial R² = 0.11% seems tiny
- But this is AFTER controlling for text
- Text already explains 92.3%, so 0.11% of remaining 7.7% is meaningful

## 7. CODING PATTERNS TO AVOID

### 7.1 Don't Modify Original Data Arrays
```python
# BAD - modifies original
data.forEach((point, i) => {
    if (condition) data.splice(i, 1);  // Breaks iteration!
});

# GOOD - create new filtered array
const filtered = data.filter(point => condition);
```

### 7.2 Always Check File Exists
```python
# BAD
df = pd.read_csv(path)

# GOOD
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
else:
    print(f"ERROR: {path} not found!")
```

### 7.3 Label Everything Clearly
```python
# BAD
results = [0.923, 0.537]

# GOOD
results = {
    'model': 'xgboost',
    'data': 'full_9513_essays',
    'r2_ai_ratings': 0.923,
    'r2_actual_sc': 0.537,
    'note': 'NOT 0.000 - that was the bug!'
}
```

## 8. SUCCESSFUL PATTERNS TO REPEAT

### 8.1 Incremental Development
- Start with working code
- Make ONE change
- Test
- Commit if working
- Repeat

### 8.2 Diagnostic Scripts
Creating small diagnostic scripts helped solve issues:
- `check_data_alignment.py`
- `diagnose_social_class_dml.py`
- `verify_social_class_prediction.py`

### 8.3 Clear Documentation
- Document bugs immediately when found
- Create checkpoints after major milestones
- Include exact commands and outputs

## 9. CHECKLIST FOR FUTURE ANALYSES

Before starting ANY analysis:

- [ ] Verify data files exist and have correct shape
- [ ] Check essays dataset has 9,513 rows, not 526
- [ ] Confirm social class labels merge properly
- [ ] Set random seed for reproducibility
- [ ] Create output directory if needed
- [ ] Test with small subset first
- [ ] Save intermediate results frequently
- [ ] Document model type in all outputs
- [ ] Verify R² values are reasonable (not 0.000)
- [ ] Check residual standard deviations > 0

## 10. QUICK DIAGNOSTIC COMMANDS

```python
# Check data shape
print(f"Essays: {len(essays)}")  # Should be 9,513
print(f"SC labels: {sc_labels['sc11'].notna().sum()}")  # Should be 9,513
print(f"AI ratings: {len(ai_ratings)}")  # Should be 19,026 (2x)

# Check results sanity
print(f"R² values: AI={r2_ai:.3f}, SC={r2_sc:.3f}")
assert r2_sc > 0.1, "SC R² too low - likely data bug!"

# Check residuals
print(f"Residual stds: Y={Y_res.std():.3f}, D={D_res.std():.3f}")
assert Y_res.std() > 0.1, "No variance in residuals!"
```

## 11. FINAL WISDOM

1. **The 9,513 number is sacred** - if you don't see it, something's wrong
2. **R² = 0.000 is always a bug** - real social class has R² ≈ 0.54 with XGBoost
3. **Cached results can lie** - verify or regenerate when in doubt
4. **Document immediately** - future you (or future models) will thank you
5. **Test incrementally** - big bang changes lead to broken code

Remember: The goal is to understand how AI perceives social class from text, and how this differs from actual social class. The data tells a clear story when loaded correctly!