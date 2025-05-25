# CHECKPOINT: Interactive UMAP Visualization and SHAP Analysis
**Date:** 2025-05-25  
**Status:** Implementation Complete - Ready for Execution

## Overview

This checkpoint documents the implementation of two advanced analysis techniques:
1. **Interactive 3D UMAP visualizations** for exploring social class clusters in essay embeddings
2. **SHAP (TreeSHAP) analysis** for interpreting Random Forest and XGBoost model predictions

## 1. Interactive UMAP Visualization

### Implementation Details

**Script:** `scripts/create_umap_visualization.py`

#### Key Features:
- **3D UMAP embedding** of PCA-transformed essay features
- **Two visualization modes:**
  - Colored by actual social class (discrete, 1-5)
  - Colored by AI ratings (continuous, 1-10)
- **Interactive functionality:**
  - Rotate: Click and drag
  - Zoom: Scroll or pinch (touch-screen compatible)
  - Hover: Preview essay text, scores, and LIME explanations
  - Click: View full essay in expandable panel

#### Technical Specifications:
- **Embeddings:** Uses existing PCA features from DML analysis (100 components, 82.3% variance)
- **UMAP parameters:** n_neighbors=15, min_dist=0.1, n_components=3
- **LIME explanations:** 
  - Based on XGBoost model (R² = 0.870)
  - Shows top 3 principal components contributing to prediction
  - Computed for all 9,513 essays (no sampling)

#### Output Files:
- `umap_actual_social_class.html`: Visualization colored by actual SC (1-5)
- `umap_ai_ratings.html`: Visualization colored by AI ratings (1-10)

### Expected Insights:
1. **Cluster structure:** Whether essays naturally cluster by social class
2. **AI vs actual alignment:** How AI ratings map to actual social class regions
3. **Feature importance:** Which PCs drive predictions in different regions

## 2. SHAP Analysis

### Implementation Details

**Script:** `scripts/compute_shap_analysis.py`

#### Key Features:
- **TreeSHAP implementation** for both Random Forest and XGBoost
- **Comprehensive visualizations:**
  - Summary plots showing feature impact distributions
  - Feature importance comparison between models
  - Waterfall plots for individual predictions
  - Dependence plots for top features
- **Quantitative analysis:**
  - Feature importance rankings
  - Model consistency metrics
  - Detailed summary report

#### Technical Specifications:
- **Models:** 
  - Random Forest: 100 trees, max_depth=10 (R² = 0.727)
  - XGBoost: 100 estimators, max_depth=5 (R² = 0.870)
- **Target:** AI average rating (mean of two prompts)
- **Features:** 100 PCA components from essay embeddings

#### Output Files (in `shap_results/`):
- `rf_shap_summary.png`: RF feature impact visualization
- `xgb_shap_summary.png`: XGBoost feature impact visualization
- `feature_importance_comparison.png`: Side-by-side comparison
- `random_forest_waterfall.png`: Individual case explanations
- `xgboost_waterfall.png`: Individual case explanations
- `xgb_dependence_plots.png`: Feature interaction plots
- `feature_importance_summary.csv`: Detailed importance scores
- `shap_analysis_report.txt`: Comprehensive text report

### Expected Insights:
1. **Feature importance:** Which PCs are most predictive of social class
2. **Model differences:** How RF and XGBoost use features differently
3. **Non-linear patterns:** Feature interactions captured by XGBoost
4. **Individual explanations:** Why specific essays receive certain ratings

## Commands to Execute

### Run UMAP Visualization (includes LIME):
```bash
cd "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme"
python3 scripts/create_umap_visualization.py
```
**Estimated time:** 30-45 minutes for full LIME computation

### Run SHAP Analysis:
```bash
cd "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme"
python3 scripts/compute_shap_analysis.py
```
**Estimated time:** 15-20 minutes for full dataset

## Integration with Previous Analyses

This builds upon:
1. **vLLM inference:** Generated AI ratings for 9,513 essays
2. **DML analysis:** Established XGBoost as best model (R² = 0.870)
3. **Distribution analysis:** Revealed measurement issues in self-reported SC

The UMAP and SHAP analyses provide:
- **Visual exploration** of the high-dimensional space
- **Model interpretability** for the black-box predictions
- **Feature understanding** for downstream causal analysis

## Key Methodological Notes

1. **LIME vs SHAP:**
   - LIME: Local approximations, used in UMAP for hover tooltips
   - SHAP: Exact Shapley values via TreeSHAP, more accurate for tree models

2. **PCA components interpretation:**
   - Each PC is a linear combination of embedding dimensions
   - Top PCs capture major variations in essay writing style/content
   - SHAP/LIME scores indicate which variations predict social class

3. **Computational efficiency:**
   - UMAP embeddings are cached after first computation
   - LIME/SHAP values saved for reproducibility
   - Batch processing with progress bars

## Next Steps

After running both analyses:
1. Examine UMAP clusters for social class patterns
2. Compare LIME (local) vs SHAP (global) feature importance
3. Identify key PCs for targeted linguistic analysis
4. Use insights for refined DML causal estimates

---
**Status:** Ready for execution. All dependencies installed, data prepared.