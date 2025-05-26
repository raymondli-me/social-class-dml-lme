# CHECKPOINT: Remove Demographics from All Analyses

**Date:** May 26, 2025  
**Status:** ⚠️ CRITICAL METHODOLOGY CHANGE

## Important Decision: No Demographics in DML

### What Changed
We are **removing all demographic variables** (age, gender, education) from the DML analyses because:

1. **They were randomly generated** - not real participant data
2. **They add noise** - random variables can't improve causal estimates
3. **They complicate interpretation** - we want the pure effect of SC on AI ratings controlling for text
4. **Consistency** - all models should use the same approach

### Previous Approach (INCORRECT)
```python
# DON'T DO THIS
np.random.seed(42)
df['age'] = np.random.normal(40, 15, len(df))
df['female'] = np.random.binomial(1, 0.5, len(df))
df['education_level_numeric'] = np.random.randint(1, 6, len(df))
W = df[['age', 'female', 'education_level_numeric']].values

# DML with fake demographics
dml.fit(Y, D, X=X, W=W)  # W contains random noise
```

### New Approach (CORRECT)
```python
# DO THIS INSTEAD
# No demographics generated
# DML with text embeddings only
dml.fit(Y, D, X=X, W=None)  # Only text controls
```

## DML Model Specification

### Variables
- **Y**: AI ratings (outcome)
- **D**: Actual social class (treatment)
- **X**: Text embeddings (PCA reduced)
- **W**: None (no demographics)

### Model
Y = θD + g(X) + ε

Where:
- θ = causal effect of actual SC on AI ratings
- g(X) = flexible function of text embeddings
- No demographic controls needed

## Action Items for All Future Analyses

1. **Remove demographic generation code**
2. **Set W=None in all DML calls**
3. **Update documentation to reflect this**
4. **Re-run any analyses that used demographics**

## Files Updated
- `nvembed_dml_multimodel_no_demographics.py` - New version without demographics
- Previous analyses may need re-running

## Rationale
Using randomly generated demographics:
- Doesn't improve estimates (random noise)
- May introduce spurious correlations
- Complicates interpretation
- Violates the assumption that controls should be meaningful

The text embeddings (X) already capture all relevant information from the essays. Adding fake demographics only muddies the water.

## Going Forward
**ALL DML ANALYSES SHOULD:**
1. Use only text embeddings as controls (X)
2. Set W=None 
3. Document that no demographics are used
4. Focus on the pure text-mediated relationship

---

**Remember**: The goal is to estimate how much actual social class affects AI ratings AFTER controlling for essay content. Fake demographics don't help with this goal.