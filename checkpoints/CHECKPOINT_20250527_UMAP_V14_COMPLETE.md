# UMAP Visualization v14 - Complete Implementation Documentation

## Date: 2025-05-27
## Status: Complete with minor UI fix needed (expand button)

## Overview
Created a comprehensive interactive UMAP visualization (v14) that displays the relationship between social class, AI ratings, and text features captured through principal components. This visualization integrates cross-fitted Double Machine Learning (DML) results with an intuitive, compact interface.

## Key Features Implemented

### 1. **Cross-fitted Double Machine Learning Integration**
- Successfully connected `compute_crossfitted_metrics.py` output to the visualization
- Shows both non-cross-fitted and cross-fitted (5-fold) DML results side by side
- Added naive OLS regression (SC→AI without text controls) to show baseline relationship
- Displays R² for naive model (~0.058), demonstrating that text features explain most of the predictive power
- Results show:
  - Naive θ: ~0.423 (simple SC→AI correlation)
  - DML θ with 200 PCs: ~0.054 (cross-fitted), showing text mediates the relationship
  - DML θ with top 5 PCs: ~0.106 (cross-fitted), showing some information loss

### 2. **Compact UI Design**
- **Gallery Mode**: Moved to right: 480px to avoid blocking bottom content
- **Essay Display**: 
  - Compact by default (max-height: 200px)
  - Top 5 PCs displayed inline in header with full SHAP values and variance
  - Format: "PC46: 87% | AI:+0.24 SC:-0.12 | 3.2%var"
  - Clickable PC badges for instant analysis
- **Merged Threshold Controls**: 
  - Single unified interface for percentiles and values
  - Synchronized sliders - change one, the other updates automatically
  - Default thresholds now at 10th/90th percentiles for better extreme case analysis

### 3. **Simplified PC Analysis**
- Removed "Update Analysis" button and PC threshold controls
- Clean high/low labels (10th percentile thresholds)
- Shows probability differences with color coding:
  - Green (bold): >20% difference (strong effect)
  - Yellow: 10-20% difference (moderate effect)  
  - Gray: <10% difference (weak effect)
- Click any PC to see:
  - P(High AI | High PC) vs P(High AI | Low PC) with difference
  - P(Low AI | High PC) vs P(Low AI | Low PC) with difference
  - Same for social class outcomes

### 4. **Enhanced Controls**
- **Essay Font Size**: 8-16px slider
- **Essay Background Opacity**: Now 0.1-1.0 range (was 0.5-1.0)
- **Expand Button**: Added to essay header (needs positioning fix)
- **DML Stats Toggle**: Checkbox to show/hide comprehensive DML table
- **PC Dropdown**: All 200 PCs available for individual analysis

### 5. **Data Flow**
```
Essays (9513) → NVEmbed → 4096 dims → PCA → 200 PCs → DML Analysis
                                                    ↓
                                              Top 5 PCs selected
                                                    ↓
                                              UMAP 3D visualization
```

### 6. **Statistical Methods**
- **DML (Double Machine Learning)**: Causal inference controlling for text confounders
- **Cross-fitting**: 5-fold to avoid overfitting bias
- **SHAP values**: Approximate TreeSHAP using feature ablation
- **Logistic models**: For probability predictions of extreme outcomes

## Technical Implementation Details

### Python Components
1. **create_minimal_umap_viz_v14.py**: Main script
   - Loads pre-computed UMAP coordinates and DML results
   - Computes naive OLS for comparison
   - Trains logistic models for PC effect analysis
   - Generates self-contained HTML with embedded data

2. **compute_crossfitted_metrics.py**: Generates true cross-fitted estimates
   - Implements proper k-fold cross-validation
   - Computes residuals on held-out data
   - Provides unbiased DML estimates

### JavaScript Features
- Three.js for 3D visualization with GPU-accelerated rendering
- OrbitControls for smooth navigation
- Custom raycasting for hover detection
- Synchronized threshold controls with bidirectional updates
- Gallery mode for systematic exploration of extreme cases

## Key Insights from the Visualization

1. **Text Mediation**: The naive R² of 0.058 vs much higher R² with PCs shows that most of the SC→AI relationship is mediated through writing style/content

2. **Cross-fitting Impact**: Cross-fitted R² values are much lower (0.505 vs 0.919 for 200 PCs), revealing substantial overfitting in non-cross-fitted models

3. **Top 5 PCs Capture Key Signal**: Despite using only 5 of 200 PCs, the DML effect remains significant, suggesting these components capture the essential mediating features

4. **Extreme Cases**: The 10/90 percentile defaults highlight where PC effects are strongest, making patterns more visible

## Issues to Fix

1. **Expand Button Positioning**: Currently overlaps with other elements, needs to be moved
   - Suggestion: Add to gallery controls or create separate essay controls section
   - Alternative: Make it a floating button or add to the main controls panel

2. **Performance**: With 9513 points, some operations can be slow
   - Consider implementing LOD (level of detail) for distant points
   - Add loading indicators for heavy computations

## File Outputs
- `/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_dml_pc_analysis/minimal_umap_viz_v14.html`
- Self-contained HTML file (~15MB) with all data embedded
- No external dependencies required

## Next Steps
1. Fix expand button positioning
2. Consider adding export functionality for selected essays
3. Add ability to save/load threshold configurations
4. Implement search functionality for specific essays

## Usage Notes
- Best viewed on screens ≥1920x1080
- Chrome/Firefox recommended for WebGL performance
- Use gallery mode to systematically explore extreme cases
- Click PCs in essay header for quick analysis
- Adjust thresholds to explore different populations