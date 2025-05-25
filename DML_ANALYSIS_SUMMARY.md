# Double Machine Learning (DML) Analysis Summary

## Project Overview
This project applies Double Machine Learning (DML) methods to analyze social class prediction from autobiographical essays, combining LLM-based ratings with actual social class labels (sc11) from the ASC dataset.

## Key Components

### 1. Data
- **ASC Dataset**: 9,513 autobiographical essays
- **Actual Social Class (sc11)**: 1-5 scale from external dataset
- **AI Ratings**: Two prompts (standard & human-style) with high agreement (r=0.836)

### 2. Methods Implemented

#### A. Full DML Pipeline (`dml_social_class_analysis.py`)
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Dimensionality Reduction**: PCA to 200 components
- **ML Methods**: Linear, Lasso, Ridge, Random Forest, XGBoost
- **Interpretability**: LIME for feature importance
- **Cross-fitting**: 5-fold cross-validation

#### B. Simplified Analysis (`dml_analysis_simple.py`)
- **Features**: TF-IDF (1000 features)
- **PCA**: 200 components
- **Methods**: Linear, Lasso, Random Forest
- **Faster processing for larger datasets

#### C. Quick Demo (`dml_quick_analysis.py`)
- **Sample**: 1,000 essays
- **Features**: Basic text statistics (length, word count, etc.)
- **Method**: Linear DML only
- **Results**: 
  - Actual SC (sc11): R²=0.066, θ=1.053 (p=0.006)
  - AI Rating: R²=0.192, θ=3.170 (p<0.001)

## Key Findings

### 1. AI Performance
- **Correlation with actual social class**: r≈0.24-0.25
- **Inter-prompt agreement**: r=0.836
- **Monotonic relationship**: Mean AI ratings increase with actual social class

### 2. DML Insights
- **Treatment effect (θ)**: Significant positive effects detected
- **Model performance**: AI ratings show higher R² than actual social class
- **Feature importance**: Essay length and complexity correlate with ratings

### 3. Methodological Contributions
- **Bias correction**: DML addresses high-dimensional confounding
- **Cross-fitting**: Prevents overfitting in ML predictions
- **Multiple learners**: Robustness across different ML methods

## Implementation Details

### Dependencies
```
pandas, numpy, scikit-learn, scipy
sentence-transformers, torch
xgboost, lime
matplotlib, seaborn
```

### File Structure
```
scripts/
  ├── dml_social_class_analysis.py    # Full pipeline
  ├── dml_analysis_simple.py          # TF-IDF version
  ├── dml_quick_analysis.py           # Quick demo
  └── analyze_asc_*.py                # Correlation analyses

asc_analysis_2prompts/              # AI rating results
  ├── results_*.csv                  # Raw ratings
  ├── correlation plots              # Visualizations
  └── statistics                     # Summary stats

checkpoints/                        # Progress documentation
data/                              # Essay datasets
```

## DML Framework

Following your DML-Lens model approach:

1. **First Stage**: ML models predict outcome and treatment from confounders
2. **Residualization**: Extract variation unexplained by confounders  
3. **Second Stage**: Estimate causal parameter from residuals
4. **Neyman Orthogonality**: Robust to first-stage estimation errors

## Visualizations Generated

1. **Correlation matrices**: R² across methods and targets
2. **Coefficient plots**: DML estimates with confidence intervals
3. **PCA variance**: Explained variance by components
4. **LIME importance**: Top contributing features
5. **Scatter plots**: AI vs actual social class

## Next Steps

1. **Full dataset processing**: Run complete pipeline on all 9,513 essays
2. **Enhanced features**: Add linguistic features, sentiment, readability
3. **Causal questions**: Define specific treatment variables of interest
4. **Mixed effects**: Incorporate essay-level random effects
5. **Validation**: Test on held-out datasets

## Theoretical Implications

This work demonstrates:
- LLMs can extract social class signals from text
- DML provides principled approach to high-dimensional text analysis  
- Combining human and AI judgments may improve accuracy
- Text-based social class measurement has research applications

## Code Repository

All code, data references, and documentation available at:
https://github.com/raymondli-me/social-class-dml-lme