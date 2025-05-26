# Double Machine Learning (DML) Bidirectional Analysis Documentation

**Date:** May 25, 2025  
**Author:** Analysis conducted with Claude  
**Dataset:** 9,513 essays from ASC corpus with self-reported social class labels

## 1. Research Questions

We conducted bidirectional DML analysis to answer:

1. **Model 1**: What is the causal effect of actual social class on AI ratings, after controlling for text content?
2. **Model 2**: What is the causal effect of AI ratings on actual social class, after controlling for text content?

## 2. Methodology

### 2.1 Double Machine Learning (DML) Framework

DML allows us to estimate causal effects in high-dimensional settings by:
- Using flexible ML methods for nuisance parameter estimation
- Avoiding regularization bias through cross-fitting
- Providing valid statistical inference

For a partially linear model: Y = θD + g(X,W) + ε

Where:
- Y = outcome
- D = treatment 
- X = high-dimensional controls (text embeddings)
- W = low-dimensional controls (demographics)
- θ = causal parameter of interest

### 2.2 DML Algorithm

1. **Split data** into K folds (we used K=5)
2. For each fold k:
   - **Train** nuisance models on all folds except k:
     - ĝ(-k)(X,W) to predict E[Y|X,W]
     - m̂(-k)(X,W) to predict E[D|X,W]
   - **Predict** on fold k:
     - Ŷ(k) = ĝ(-k)(X(k),W(k))
     - D̂(k) = m̂(-k)(X(k),W(k))
   - **Compute residuals**:
     - Y_res(k) = Y(k) - Ŷ(k)
     - D_res(k) = D(k) - D̂(k)
3. **Estimate θ** by regressing Y_res on D_res:
   - θ̂ = Σ(D_res × Y_res) / Σ(D_res²)
4. **Compute standard errors** using influence functions

### 2.3 Our Implementation

#### Variables:
- **Text embeddings (X)**: 200 principal components from OpenAI text-embedding-3-large (3072 dims → 200 PCs)
- **Demographics (W)**: Age, gender, education level (randomly generated with seed=42)
- **Actual social class**: Self-reported 1-5 scale
- **AI ratings**: Average of 2 LLM prompts, scale 1-10

#### Models tested:
- **Linear regression**: Fast, interpretable, may underfit
- **XGBoost**: Flexible, captures non-linearities, slower

## 3. Code Implementation

### 3.1 Core DML Function

```python
def compute_dml_effect(Y, D, X, W, n_splits=5):
    """
    Compute DML estimate: Y = θD + g(X,W) + ε
    
    Args:
        Y: Outcome (n,)
        D: Treatment (n,)
        X: High-dimensional controls (n, p) 
        W: Additional controls (n, q)
    """
    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Arrays for cross-fitted residuals
    Y_res = np.zeros_like(Y, dtype=float)
    D_res = np.zeros_like(D, dtype=float)
    
    # Cross-fitting
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        XW = np.hstack([X, W])  # Combine controls
        
        # Fit E[Y|X,W]
        model_Y = LinearRegression()  # or XGBoost
        model_Y.fit(XW[train_idx], Y[train_idx])
        Y_pred = model_Y.predict(XW[test_idx])
        Y_res[test_idx] = Y[test_idx] - Y_pred
        
        # Fit E[D|X,W]
        model_D = LinearRegression()  # or XGBoost
        model_D.fit(XW[train_idx], D[train_idx])
        D_pred = model_D.predict(XW[test_idx])
        D_res[test_idx] = D[test_idx] - D_pred
    
    # Final stage: regress Y_res on D_res
    theta = np.sum(D_res * Y_res) / np.sum(D_res**2)
    
    # Standard error via influence function
    psi = (Y_res - theta * D_res) * D_res
    var_theta = np.mean(psi**2) / (np.mean(D_res**2)**2 * n)
    se_theta = np.sqrt(var_theta)
    
    return theta, se_theta
```

### 3.2 Bidirectional Analysis

```python
# Model 1: SC → AI
results1 = compute_dml_effect(
    Y=ai_ratings,     # Outcome
    D=actual_sc,      # Treatment  
    X=pca_200,        # Text controls
    W=demographics    # Additional controls
)

# Model 2: AI → SC (reverse)
results2 = compute_dml_effect(
    Y=actual_sc,      # Outcome
    D=ai_ratings,     # Treatment
    X=pca_200,        # Text controls
    W=demographics    # Additional controls
)
```

## 4. Results

### 4.1 First Stage Model Performance

How well can we predict each variable from text + demographics?

| Outcome | Linear R² | XGBoost R² |
|---------|-----------|------------|
| AI ratings | 0.631 | **0.923** |
| Actual SC | 0.114 | **0.537** |

**Key finding**: Text embeddings explain 92.3% of AI rating variance but only 53.7% of actual social class variance.

### 4.2 Causal Effects (Second Stage)

After controlling for text content and demographics:

| Direction | θ (Causal Effect) | SE | t-stat | p-value | 95% CI | Partial R² |
|-----------|-------------------|-----|--------|---------|---------|------------|
| **SC → AI** | 0.0332 | 0.0103 | 3.22 | 0.001** | [0.013, 0.053] | 0.11% |
| **AI → SC** | 0.0321 | 0.0100 | 3.21 | 0.001** | [0.013, 0.052] | 0.11% |

### 4.3 Statistical Tests

- **Null hypothesis**: θ = 0 (no causal effect after controlling for text)
- **Result**: Reject null for both directions (p < 0.01)
- **Effect size**: Very small (partial R² = 0.11% for both)

## 5. Interpretation

### 5.1 Main Findings

1. **Text dominates**: After accounting for text content, actual social class explains only 0.11% of additional variance in AI ratings

2. **Symmetric effects**: The bidirectional effects are nearly identical (0.033 vs 0.032), suggesting the residual correlation is symmetric

3. **Mediation**: Text mediates 87% of the raw SC-AI correlation:
   - Raw correlation: r = 0.251  
   - Causal effect after controlling for text: θ = 0.033
   - Proportion mediated: (0.251 - 0.033)/0.251 = 87%

### 5.2 What This Means

1. **AI ratings are primarily text-driven**: The AI models base their social class assessments almost entirely on textual features, not on actual social class

2. **Text captures social class signals**: The 53.7% R² shows text does contain real social class information

3. **AI amplifies certain textual cues**: The gap between 92.3% (AI) and 53.7% (actual SC) suggests AI models may be overweighting certain textual features that correlate with but don't define social class

### 5.3 Causal Interpretation

The small but significant causal effects (θ ≈ 0.033) mean:
- A 1-unit increase in actual social class causes a 0.033 unit increase in AI rating (after controlling for text)
- This represents the "direct" effect not mediated through text
- Could reflect subtle patterns the 200 PCs don't fully capture

## 6. Technical Notes

### 6.1 Assumptions

1. **Conditional ignorability**: (Y,D) ⊥ ε | (X,W)
2. **Overlap**: 0 < P(D|X,W) < 1
3. **Linearity**: Partially linear model is correctly specified
4. **Cross-fitting**: Prevents overfitting bias

### 6.2 Limitations

1. **Demographics are simulated**: Real demographics might change results
2. **PCA compression**: 3072 → 200 dimensions loses some information
3. **Model dependence**: Linear models show lower R² than XGBoost

### 6.3 Robustness

Results are robust across:
- Linear vs XGBoost models (causal effects similar)
- Different numbers of PCs (tested with 100, 200)
- Cross-validation folds (5-fold vs 10-fold similar)

## 7. Conclusion

This analysis reveals that AI social class ratings are overwhelmingly determined by textual content rather than actual social class. The 38.6% gap between how well text predicts AI ratings (92.3%) versus actual social class (53.7%) quantifies the extent of potential AI bias in social class assessment. While both causal effects are statistically significant, their tiny magnitude (partial R² = 0.11%) confirms that once we account for text, there's almost no additional relationship between actual and AI-perceived social class.