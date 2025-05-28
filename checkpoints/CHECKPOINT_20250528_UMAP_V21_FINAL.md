# CHECKPOINT: UMAP Visualization v21 - Final Dataset-Specific Version
**Date:** 2025-05-28
**Version:** v21 (Final version before generalization)
**Script:** `/scripts/create_minimal_umap_viz_v21.py`
**Output:** `/nvembed_dml_pc_analysis/minimal_umap_viz_v21.html`

## Overview
This is the final version of the UMAP visualization tool specifically designed for the AI Rating vs Social Class dataset. This checkpoint documents the complete functionality before transitioning to a generalized package that will work with arbitrary X/Y variables.

## Core Architecture

### 1. Data Pipeline
- **Input Files:**
  - `asc_9513_essays.csv` - Contains essays with AI ratings and social class (sc11)
  - `nvembed_embeddings.npy` - 4096-dimensional NVEmbed embeddings
  - `nvembed_pca_200_features.pkl` - PCA-reduced features (200 dimensions)
  - `umap_3d_nvembed_custom.npy` - 3D UMAP coordinates
  - `dml_pc_analysis_results_fixed_hover.pkl` - DML analysis results
  - `crossfitted_metrics_v13.pkl` - Cross-fitted DML metrics

### 2. Key Constants (Hardcoded for generalization)
- **PCA Dimensions:** 200 (full PCA features)
- **Top PCs for DML:** 5 (specifically PC0, PC2, PC5, PC13, PC46)
- **UMAP Dimensions:** 3D
- **Extreme Thresholds:** 10th and 90th percentiles

### 3. Major Components

#### A. Data Processing
```python
# Core data structures
- essays_df: DataFrame with essays, ai_rating, sc11
- X_pca: 200-dimensional PCA features
- X_umap_3d: 3D UMAP coordinates
- Y_ai: AI ratings (1-10 scale)
- Y_sc: Social class (1-5 scale)
```

#### B. Statistical Models
1. **XGBoost Models**
   - AI prediction model (200 PCs → AI rating)
   - SC prediction model (200 PCs → Social class)
   - Feature importance as proxy for contributions

2. **DML (Double Machine Learning)**
   - Naive model: SC → AI (no controls)
   - 200 PC model: SC → AI (controlling for all 200 PCs)
   - Top 5 PC model: SC → AI (controlling for PC0, PC2, PC5, PC13, PC46)
   - Both cross-fitted and non-cross-fitted versions

3. **HDBSCAN Clustering**
   - Performed on 3D UMAP coordinates
   - Generates topic clusters with c-TF-IDF keywords
   - Topic statistics for extreme groups

#### C. Visualization Features

##### 1. 3D Point Cloud
- Three.js WebGL rendering
- Points colored by category or mode
- Interactive camera controls
- Auto-rotation option

##### 2. Color Modes
- **AI/SC Mode:** 5 categories based on threshold combinations
  - Both High (green): High AI + High SC
  - AI High (magenta): High AI + Low SC  
  - SC High (cyan): Low AI + High SC
  - Both Low (yellow): Low AI + Low SC
  - Middle (gray): Everything else
- **Topics Mode:** Colors by HDBSCAN cluster
- **PC Gradient Mode:** Purple-to-yellow gradient based on PC percentile

##### 3. Interactive Panels
- **Info Panel** (Left, collapsible)
  - Threshold sliders
  - Category counts
  - Color legend
  
- **Controls Panel** (Right, collapsible)
  - Auto-rotation settings
  - Point/essay opacity
  - DML stats toggle
  - Topic stats toggle
  - Topic visibility controls
  
- **Color Mode Panel** (Center-left)
  - Three-button selector
  - Shifts with info panel collapse

- **Gallery Mode** (Right)
  - Navigate essays by category
  - Keyboard controls (← →)
  - Visual indicators

- **Essay Display** (Bottom)
  - Shows essay content on hover
  - Resizable (all edges + corners)
  - Layer toggle (front/back)
  - Minimize option
  - PC contributions inline

- **DML Stats Table** (Toggle)
  - Naive, 200 PC, and Top 5 PC results
  - Cross-fitted vs non-cross-fitted
  - Effect reduction percentages
  - R² values
  - Lists specific PCs used in Top 5

- **PC Analysis Popup** (Click PC name)
  - Importance rankings
  - SHAP value statistics  
  - Correlations with outcomes
  - Extreme group probabilities
  - Navigate with ← →

- **Topic Stats Panel** (Toggle, NEW in v21)
  - Shows % of essays in each topic falling into extremes
  - Ranked by maximum impact
  - Color-coded percentages
  - Scrollable table

##### 4. Topic Visualization
- Billboarded text labels at cluster centroids
- Dynamic visibility (N closest topics)
- Adjustable text size and opacity
- Option to layer on top

##### 5. Z-Index Management
- Click panels to bring to front
- Consistent stacking order
- Excludes interactive elements

## Key Algorithms

### 1. PC Contribution Calculation
```python
# Using XGBoost feature importance as proxy
contributions_ai = X_pca[i] * feature_importance_ai
contributions_sc = X_pca[i] * feature_importance_sc
```

### 2. Category Assignment
```python
# Based on user-adjustable thresholds
highAI = ai_rating > ai_high_threshold
lowAI = ai_rating < ai_low_threshold
highSC = sc11 >= sc_high_threshold
lowSC = sc11 <= sc_low_threshold
```

### 3. Topic Keywords (c-TF-IDF)
```python
# TF-IDF on concatenated cluster essays
# Top 5 keywords per cluster
# Noise cluster excluded
```

### 4. Proximity Sorting
```python
# For gallery navigation
# Sort by 3D distance from current position
# Enables "nearest essay" navigation
```

## Preset Configurations
1. **P10/P90** - 10th/90th percentiles
2. **P20/P80** - 20th/80th percentiles  
3. **P25/P75** - 25th/75th percentiles
4. **Bottom/Top 20%** - Fixed percentage thresholds

## Performance Optimizations
- Instanced rendering considered (not implemented)
- Dynamic topic visibility limits
- Efficient color buffer updates
- Debounced threshold updates

## Data Flow for Generalization

### Current (Specific)
```
essays.csv → AI Rating, Social Class → Visualize relationship
```

### Future (Generic)
```
data.csv → Variable X, Variable Y → Visualize relationship
```

### Fixed Parameters for Simplicity
- Always use 200 PCA dimensions
- Always use top 5 PCs for reduced model
- Always use 3D UMAP
- Always use 10th/90th percentiles for extremes

## Migration Notes for Generic Package

### 1. Variable Name Mappings
- `ai_rating` → `var_x` or `outcome_1`
- `sc11` → `var_y` or `outcome_2`
- `High AI + High SC` → `High X + High Y`
- Essay-specific language → Generic "item" or "sample"

### 2. Configurable Elements
- Variable names and labels
- Color schemes
- Threshold defaults
- Display text

### 3. Core Logic to Preserve
- All statistical calculations
- 3D visualization engine
- Interactive panel system
- Topic discovery pipeline
- DML analysis framework

### 4. Hardcoded Elements to Keep
- 200 PCA dimensions
- Top 5 PCs selection
- 3D UMAP projection
- Percentile calculations
- Panel layouts

## File Structure
```
/nvembed_dml_pc_analysis/
  ├── minimal_umap_viz_v21.html (output)
  ├── dml_pc_analysis_results_fixed_hover.pkl
  ├── crossfitted_metrics_v13.pkl
  └── umap_3d_coordinates.npy

/scripts/
  └── create_minimal_umap_viz_v21.py (main script)

/data/
  ├── asc_9513_essays.csv
  └── (other data files)

/nvembed_checkpoints/
  ├── nvembed_embeddings.npy
  ├── nvembed_pca_200_features.pkl
  └── umap_3d_nvembed_custom.npy
```

## Version History Highlights
- v14: Base GPU picking implementation
- v15: Fixed hover alignment 
- v16: Gold standard with proper alignment
- v17: Added HDBSCAN topic discovery
- v18: Billboarded topic labels
- v19: PC gradient coloring, UI improvements
- v20: Added PC list to DML table
- v21: Topic statistics panel (FINAL)

## Usage
```bash
cd /path/to/project
python3 scripts/create_minimal_umap_viz_v21.py
# Open nvembed_dml_pc_analysis/minimal_umap_viz_v21.html in browser
```

## Final Notes
This version represents the culmination of iterative development for the AI/SC dataset. All features are stable and tested. The architecture is ready for generalization while maintaining the sophisticated analysis capabilities developed throughout the project.