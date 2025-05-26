# CHECKPOINT: NV-Embed UMAP Visualization Plan
**Date:** 2025-05-26
**Time:** Current session

## Status Update
- Completed NV-Embed full dimensions (4,096) vs PCA (200) comparison scripts
- User wants to focus on UMAP visualization using NV-Embed

## Next Task: NV-Embed UMAP Visualization
Create 3D UMAP visualization using:
1. **Input**: NV-Embed embeddings → PCA 200 → UMAP 3D
2. **Data source**: Same 9,513 essays with NV-Embed embeddings
3. **Coloring**: 
   - Actual social class (sc11)
   - AI ratings (human MacArthur only)
4. **Visualization style**: Use techniques from earlier OpenAI visualization

## Key Files and Locations
### NV-Embed Data
- Embeddings: `/nvembed_checkpoints/nvembed_embeddings.npy` (9513, 4096)
- Essay IDs: `/nvembed_checkpoints/nvembed_essay_ids.npy`
- PCA model: Already computed in analysis scripts

### Previous UMAP Visualization
- OpenAI version: `/scripts/create_3d_umap_visualization.py`
- Output location: `/custom_visualizations/openai_umap_3d_9513.html`
- Used Plotly for interactive 3D scatter plot

### Data Files
- Essays: `/data/asc_9513_essays.csv`
- Social class: `/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv`
- AI ratings: `/asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv`
  - Filter for 'human_macarthur_ladder_improved' only

## Visualization Features to Include
1. Interactive 3D scatter plot with Plotly
2. Color by actual social class or AI ratings
3. Hover info: essay excerpt, SC, AI rating
4. Size variation based on confidence/variance
5. Save as HTML for easy sharing

## Implementation Plan
1. Load NV-Embed embeddings
2. Apply PCA to reduce to 200 dimensions
3. Apply UMAP to get 3D coordinates
4. Create interactive Plotly visualization
5. Save as HTML file