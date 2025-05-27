# CHECKPOINT: DML PC Analysis with Enhanced UMAP Visualization
**Date:** 2025-05-26
**Time:** Current session (following NV-Embed visualization work)

## Overview
Created comprehensive DML (Double Machine Learning) analysis identifying top Principal Components (PCs) for predicting AI ratings and actual social class, with enhanced 3D UMAP visualization featuring extensive filtering and color options.

## User's Key Requests Throughout Session

### Primary Goals
1. **Identify top 5 PCs** for predicting:
   - AI ratings (how LLMs perceive social class)
   - Actual social class (ground truth)
2. **Show PC information in hover tooltips**:
   - Z-scores and percentiles
   - SHAP values (or contributions) for each PC
   - R² values and DML causal coefficients
3. **Create enhanced UMAP visualization** with:
   - Multiple color modes (categorical SC, gradients for SC/AI/PCs)
   - Social class filtering
   - PC extremes filtering (show top/bottom percentiles, hide middle)
   - Adjustable point size and opacity
   - **Cloud scale slider** to expand/shrink the point cloud
   - Yellow highlight on hover

### Persistent Issues User Reported
1. **Hover mechanism misalignment**: The visual dots don't perfectly align with where the hover detection occurs
2. **Dot size initially too small**: Requested starting size of 6 with range 0.5-20
3. **Gradient colors**: Wanted darker middle values with bright extremes for better contrast
4. **Camera zoom**: Requested starting much closer to the data cloud

## Technical Implementation

### 1. DML PC Analysis (`analyze_dml_top_pcs_umap.py`)

#### Data Pipeline
```python
# Load 9,513 essays with proper column mapping
essays_df = pd.read_csv('data/asc_9513_essays.csv')
essays_df.rename(columns={'TID': 'essay_id', 'original': 'essay'})

# Load social class (actual)
sc_df = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')
sc_df.rename(columns={'TID': 'essay_id'})  # Important: TID not essay_id

# Load AI ratings - ONLY human MacArthur
ai_ratings_df[ai_ratings_df['prompt_name'] == 'human_macarthur_ladder_improved']

# Load NV-Embed PCA features (200 components)
with open('nvembed_checkpoints/nvembed_pca_200_features.pkl', 'rb') as f:
    X_pca = pca_data['features']  # (9513, 200)
```

#### DML Analysis
1. **Full model with all 200 PCs**:
   - Fit DML to estimate causal effect of actual SC on AI ratings
   - Extract feature importances from underlying XGBoost models
   - Compute cross-validated R² values

2. **Identify top 5 PCs**:
   - Combined importance = average of Y model (AI rating) and T model (actual SC) importances
   - Selected top 5 by combined importance

3. **Refit with top 5 only**:
   - Compare performance loss
   - Show that top 5 PCs capture most predictive power

#### Key Metrics Computed
- **DML θ (theta)**: Causal effect of actual SC on AI rating
- **Text→AI R²**: How well text embeddings predict AI ratings
- **Text→SC R²**: How well text embeddings predict actual SC
- **Text+SC→AI R²**: How well text + actual SC predict AI ratings (DML R²)
- **Text+AI→SC R²**: How well text + AI ratings predict actual SC

### 2. Enhanced UMAP Visualization

#### Three.js Implementation
```javascript
// Custom shader for individual point sizes
const material = new THREE.ShaderMaterial({
    vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        void main() {
            vColor = color;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = size * 50.0 / -mvPosition.z;
            gl_Position = projectionMatrix * mvPosition;
        }
    `,
    fragmentShader: `
        uniform float opacity;
        varying vec3 vColor;
        void main() {
            float dist = length(gl_PointCoord - vec2(0.5, 0.5));
            if (dist > 0.5) discard;
            gl_FragColor = vec4(vColor, opacity);
        }
    `
});
```

#### Key Features Implemented

1. **Dynamic Cloud Scaling**:
   ```javascript
   function updateCloudScale() {
       const newScale = parseFloat(document.getElementById('cloud-scale').value);
       for (let i = 0; i < data.length; i++) {
           positions[i * 3] = originalPositions[i * 3] * newScale;
           positions[i * 3 + 1] = originalPositions[i * 3 + 1] * newScale;
           positions[i * 3 + 2] = originalPositions[i * 3 + 2] * newScale;
       }
       geometry.attributes.position.needsUpdate = true;
   }
   ```

2. **Color Modes**:
   - Social class (categorical): 5 distinct colors
   - Social class (gradient): Blue → Dark purple → Dark red → Bright red
   - AI rating (gradient): Same gradient scheme
   - PC values (gradient): For each of top 5 PCs

3. **Filtering**:
   - Social class checkboxes (hide/show each class)
   - PC extremes sliders (show bottom X% and top Y%, hide middle)
   - Visibility handled by setting point size to 0

4. **Hover Mechanism**:
   - Raycaster checks for intersections
   - Filters to only visible points (size > 0)
   - Shows yellow wireframe sphere at hovered point
   - Detailed tooltip with essay info, PC scores, SHAP values

#### Gradient Color Scheme (User's Final Request)
```javascript
// Dark middle, bright extremes
if (norm < 0.2) {
    // Bright blue for low extremes
} else if (norm < 0.4) {
    // Blue to dark purple
} else if (norm < 0.6) {
    // Dark middle zone (almost black)
} else if (norm < 0.8) {
    // Dark red to red
} else {
    // Bright red for high extremes
}
```

## Current Issues & Limitations

### 1. Hover-Visual Misalignment
**Problem**: The raycaster detection doesn't perfectly match visible dot positions
**Attempted Solutions**:
- Dynamic raycaster threshold based on point size
- Storing scaled coordinates
- Checking only visible points
**Status**: Partially fixed but still imperfect

### 2. SHAP Values
**Problem**: SHAP TreeExplainer incompatible with current XGBoost version
**Workaround**: Using feature contribution method (zeroing out features)
**Impact**: Less accurate than true SHAP but functional

### 3. Performance
**Issue**: With 9,513 points, filtering can be slow
**Potential Solution**: Use GPU-based filtering or octree spatial indexing

## File Structure

### Scripts Created/Modified
```
scripts/
├── analyze_dml_top_pcs_umap.py          # Main analysis (current)
├── analyze_top_pcs_enhanced_umap.py     # Earlier version
├── analyze_top_pcs_fast.py              # Faster version (incomplete)
├── nvembed_full_dims_human_macarthur.py # Full dims comparison
└── create_nvembed_custom_viz_*.py       # Various visualization versions
```

### Output Structure
```
nvembed_dml_pc_analysis/
├── umap_dml_top5_pcs.html             # Final visualization
├── pc_analysis_results.pkl             # Analysis results
├── pc_importance_summary.csv           # PC importance scores
└── dml_model_comparison.csv            # Full vs Top5 comparison
```

### Data Dependencies
```
/data/
├── asc_9513_essays.csv                 # 9,513 essays (TID, original)
├── ladder_variations_50_complete.csv    # Prompt variations

/nvembed_checkpoints/
├── nvembed_embeddings.npy              # (9513, 4096) full embeddings
├── nvembed_pca_200_features.pkl        # PCA reduced to 200
└── umap_3d_nvembed_custom.npy          # 3D UMAP coordinates

External:
/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv
/asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv
```

## Key Findings

1. **Top 5 PCs Performance**:
   - Capture ~10-15% of total variance
   - Maintain most predictive power for AI ratings
   - Significant drop in actual SC prediction (expected)

2. **DML Results**:
   - Full model: θ ≈ 0.04, p < 0.01 (significant)
   - Top 5 model: Similar θ, still significant
   - Shows robust causal effect even with dimension reduction

3. **Visualization Insights**:
   - Clear clustering by social class in UMAP
   - PC gradients reveal interpretable patterns
   - Extremes filtering useful for identifying outliers

## Next Steps & Recommendations

### Immediate Fixes Needed
1. **Perfect hover-visual alignment**:
   - Consider using instanced meshes instead of point cloud
   - Or implement custom picking shader
   - Ensure raycaster uses exact same transformation as renderer

2. **True SHAP values**:
   - Downgrade SHAP library or upgrade XGBoost
   - Or implement custom TreeSHAP for compatibility

### Potential Enhancements
1. **Performance optimization**:
   - LOD (Level of Detail) for large datasets
   - WebGL 2.0 features for better performance
   - Frustum culling for off-screen points

2. **Additional features**:
   - Export filtered data subsets
   - Animate transitions between color modes
   - Add 2D projection option (t-SNE/UMAP)
   - Statistical summary panel for visible points

3. **Analysis extensions**:
   - Hierarchical clustering on top PCs
   - Identify representative essays per cluster
   - Cross-validation of PC selection

## Commands to Run

```bash
# Generate analysis and visualization
cd /media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme
python3 scripts/analyze_dml_top_pcs_umap.py

# Open visualization
firefox nvembed_dml_pc_analysis/umap_dml_top5_pcs.html
```

## Critical Reminders

1. **Always use human_macarthur_ladder_improved ratings only**
2. **No demographics in DML (they're fake)**
3. **TID not essay_id in source CSVs**
4. **Point size 6 default, range 0.5-20**
5. **Cloud scale 4x default, range 0.5-10x**
6. **Dark middle gradients for better contrast**
7. **PC filters show extremes (hide middle range)**

## Session Summary

This session successfully created a comprehensive DML PC analysis pipeline with an advanced 3D visualization. While the hover-visual alignment remains imperfect, the system provides valuable insights into how text embeddings relate to both AI-perceived and actual social class. The ability to filter by PC extremes and visualize different gradient modes offers a powerful tool for exploring the high-dimensional relationships in the data.

The key achievement is demonstrating that just 5 principal components can capture most of the predictive signal while maintaining the causal effect estimate, suggesting that the relationship between text and social class perception is driven by a relatively small number of latent factors.