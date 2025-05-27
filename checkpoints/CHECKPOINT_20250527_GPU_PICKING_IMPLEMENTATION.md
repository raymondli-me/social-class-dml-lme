# CHECKPOINT: GPU Picking Implementation for UMAP Visualization

**Date:** May 27, 2025  
**Status:** ⚠️ In Progress - Cloud not rendering

## Summary

Attempted to implement GPU picking (color picking) to fix the hover-visual misalignment issue in the UMAP visualization.

## What We Did

1. **Analyzed the problem**: Three.js raycasting with Points + custom shaders is fundamentally unreliable
2. **Received solution from another developer**: Use GPU picking technique
3. **Attempted implementation**:
   - Added `pointIds` attribute to geometry
   - Created `pickingMaterial` shader that encodes point IDs as colors
   - Modified `onMouseMove` to use GPU picking instead of raycasting
   - Renders 1x1 pixel at mouse position to determine which point is hovered

## Current Issue

The data cloud is not showing - just a black screen with GUI visible. This suggests:
- The points are not being rendered, OR
- The shader is broken, OR
- The data is not being loaded properly

## Files Created/Modified

1. **Original working file**: 
   - `/scripts/analyze_dml_top_pcs_umap.py` - Generates the visualization

2. **Output HTML files**:
   - `/nvembed_dml_pc_analysis/umap_dml_top5_pcs.html` - Original (working but bad hover)
   - `/nvembed_dml_pc_analysis/umap_dml_top5_pcs_gpu_picking.html` - GPU picking version (not rendering)

3. **Failed attempts**:
   - `analyze_dml_top_pcs_umap_updated.py` - From other developer (template errors)
   - `analyze_dml_top_pcs_umap_fixed.py` - Fixed template but still broken
   - `analyze_dml_top_pcs_umap_gpu_picking.py` - String literal errors
   - `analyze_dml_top_pcs_umap_gpu_final.py` - String literal errors

4. **Patch scripts**:
   - `add_gpu_picking_to_html.py` - Modifies HTML after generation

## Technical Details

### GPU Picking Concept
```javascript
// Instead of raycasting, render each point with unique color
// Read pixel under mouse to determine which point is hovered

// Picking shader encodes ID as color:
float r = mod(id, 256.0) / 255.0;
float g = mod(floor(id / 256.0), 256.0) / 255.0;
float b = floor(id / (256.0 * 256.0)) / 255.0;
```

### Current Problem
The modification likely broke something in:
1. The shader code
2. The material switching
3. The render pipeline
4. The data binding

## Next Steps

1. **Debug the rendering issue**:
   - Check browser console for WebGL errors
   - Verify data is loaded
   - Check if points material is correctly set
   - Verify shader compilation

2. **Fallback options**:
   - Revert to original and try simpler fixes
   - Use screen-space 2D hover detection instead
   - Try instanced mesh approach

## Key Learning

Modifying complex Three.js visualizations requires careful attention to:
- Material management
- Shader compilation
- Render pipeline
- Data flow

The GPU picking approach is sound in theory but implementation needs debugging.