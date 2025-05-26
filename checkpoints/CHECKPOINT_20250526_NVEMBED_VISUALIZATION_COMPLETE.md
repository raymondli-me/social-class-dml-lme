# CHECKPOINT: NV-Embed Visualization Complete
**Date:** 2025-05-26
**Time:** Current session

## Project Status
Successfully created interactive 3D UMAP visualizations for NV-Embed embeddings with filtering capabilities for both social class and AI ratings.

## Key Accomplishments
1. **NV-Embed Full Dimensions Analysis**
   - Compared full 4,096 dimensions vs PCA 200
   - Found minimal performance difference (<2% R²)
   - PCA retains 73.6% variance and is much faster
   - Recommendation: Use PCA 200 for efficiency

2. **3D UMAP Visualizations Created**
   - Social class visualization with checkbox filtering
   - AI rating visualization with percentile bins
   - Both use Three.js/WebGL for smooth performance
   - Interactive hover tooltips with essay previews

## Critical File Locations

### Data Files
- **Essays**: `/data/asc_9513_essays.csv` (9,513 essays)
- **Social Class Labels**: `/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv`
- **AI Ratings**: `/asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv`
  - Contains both 'human_macarthur_ladder_improved' and 'ladder_standard_improved'
  - We use ONLY human_macarthur ratings

### NV-Embed Files
- **Embeddings**: `/nvembed_checkpoints/nvembed_embeddings.npy` (9513, 4096)
- **Essay IDs**: `/nvembed_checkpoints/nvembed_essay_ids.npy`
- **UMAP Coordinates**: `/nvembed_checkpoints/umap_3d_nvembed_custom.npy`

### Visualization Scripts
- **Working Social Class**: `scripts/create_nvembed_custom_viz_working_filter.py`
- **Working AI Bins**: `scripts/create_nvembed_custom_viz_ai_bins.py`
- **Output Directory**: `/nvembed_visualizations/`

## Technical Details

### UMAP Parameters
```python
umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='cosine',
    random_state=42
)
```

### Visualization Architecture
- Uses separate Three.js particle groups for each category
- Filtering via visibility toggling (not size manipulation)
- Camera starts at (150, 150, 150) for proper zoom
- Rotation centered on data centroid

## Mistakes to Avoid

1. **Data Loading**
   - DON'T use `data/essay_dataset.csv` (only 526 essays)
   - DON'T average AI ratings - use only human_macarthur
   - DON'T include demographics (they're fake/random)

2. **AI Ratings**
   - AI ratings are discrete (many 3s, 4s, 5s)
   - Must use `duplicates='drop'` when binning
   - Percentile bins work better than fixed ranges

3. **Visualization**
   - Single particle system doesn't filter well
   - Use separate particle groups per category
   - Don't use custom shaders - visibility toggling is simpler
   - Ensure camera starts zoomed out enough

## Current Methodology

### For DML Analysis
- **Y**: AI ratings (human MacArthur only)
- **D**: Self-reported social class (sc11)
- **X**: NV-Embed PCA 200 features
- **No demographics** (removed due to being randomly generated)

### For Visualizations
1. Load NV-Embed embeddings (4,096 dims)
2. Apply PCA → 200 dimensions
3. Apply UMAP → 3 dimensions
4. Create interactive Three.js visualization
5. Group by category (SC or AI percentile)
6. Enable filtering via checkboxes

## Key Findings
- NV-Embed-v2 performs well (R² = 0.597 for AI ratings)
- PCA to 200 dims loses minimal information
- AI models predict AI ratings well but fail at actual social class
- Visualizations show clear clustering by both SC and AI ratings

## Next Steps Possibilities
1. Add more filtering options (essay length, specific words)
2. Create 2D t-SNE version for comparison
3. Add trajectory animations between filtered states
4. Export filtered data subsets
5. Add statistical summaries per cluster

## Commands to Run Visualizations
```bash
# Social class with filtering
python3 scripts/create_nvembed_custom_viz_working_filter.py

# AI ratings with percentile bins  
python3 scripts/create_nvembed_custom_viz_ai_bins.py
```

## Important Notes
- All visualizations use human MacArthur ratings only
- PCA variance explained: 73.6%
- Total essays: 9,513
- Social classes: 1-5 (Lower to Upper)
- AI rating range: 1-10 (discrete values)

## Session Context
This session focused on creating custom Three.js visualizations after discovering that:
1. OpenAI's reported R² was incorrect (training set evaluation)
2. NV-Embed is actually the best performer
3. Demographics should be removed (randomly generated)
4. Human MacArthur ratings should be used exclusively
5. Full dimensions vs PCA shows minimal difference